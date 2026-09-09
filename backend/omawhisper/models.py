"""Model catalog and local-only inference; imports happen inside worker jobs."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import re
import selectors
import shutil
import stat
import subprocess
import threading
import time

from .config import UserError, atomic_json
from .acceleration import onnx_gpu_compatible
from .engine_worker import EngineWorker, WorkerCancelled, engine_python
from .memory import GIB, MIB, memory_error, memory_status


ENGINE_NAMES = {"faster-whisper": "whisper", "onnx-asr": "parakeet", "command": "command"}
QUANTIZED_SUFFIXES = {
    "int8", "uint8", "int4", "uint4", "fp16", "float16", "fp32", "float32",
    "bf16", "bfloat16", "q4", "q8", "q4f16", "bnb4", "quant", "quantized", "qdq",
}


def onnx_quantization(model: dict) -> str | None:
    return model.get("quantization", "int8") or None


def onnx_download_files(filenames: list[str], quantization: str | None) -> list[str]:
    """Choose one ONNX weight variant plus its external data and shared models."""
    available = set(filenames)
    suffixes = QUANTIZED_SUFFIXES | ({quantization} if quantization else set())
    selected = set()
    for filename in available:
        path = Path(filename)
        if path.suffix not in {".onnx", ".ort"}:
            continue
        variant = path.stem.rsplit(".", 1)[-1]
        if variant in suffixes:
            if variant == quantization:
                selected.add(filename)
        elif not quantization or str(path.with_name(f"{path.stem}.{quantization}{path.suffix}")) not in available:
            selected.add(filename)
    result = set(selected)
    for filename in available:
        if Path(filename).suffix in {".json", ".yaml", ".yml", ".txt", ".model"}:
            result.add(filename)
        elif any(filename.startswith(model + ".") or filename.startswith(model + "_") for model in selected):
            result.add(filename)
    return sorted(result)


def catalog() -> list[dict]:
    entries = []
    for size in ("tiny", "base", "small", "medium", "large-v3", "turbo",
                 "tiny.en", "base.en", "small.en", "medium.en"):
        source = f"Systran/faster-whisper-{size}"
        if size == "turbo":
            source = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
        entries.append({
            "id": f"whisper-{size}", "name": f"Whisper {size}", "engine": "faster-whisper",
            "source": source, "description": "Local CPU / NVIDIA GPU speech recognition; " +
            ("English-only transcription; no translation." if size.endswith(".en") else
             "multilingual transcription; turbo is not trained for translation." if size == "turbo" else
             "multilingual transcription and English translation."),
        })
    entries.append({
        "id": "parakeet-v3", "name": "Parakeet TDT 0.6B v3 INT8", "engine": "onnx-asr",
        "source": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "architecture": "nemo-parakeet-tdt-0.6b-v3",
        "quantization": "int8",
        "estimated_ram_bytes": 1536 * MIB,
        "description": "Compact CPU INT8 weights (~670 MB); automatic language, no translation or prompts.",
    })
    entries.append({
        "id": "parakeet-v3-fp32", "name": "Parakeet TDT 0.6B v3 FP32", "engine": "onnx-asr",
        "source": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "architecture": "nemo-parakeet-tdt-0.6b-v3", "quantization": "",
        "estimated_ram_bytes": 5 * GIB,
        "description": "GPU-compatible FP32 weights (~2.5 GB); NVIDIA CUDA or CPU, automatic language.",
    })
    return entries


def validate_model(model: dict) -> dict:
    if type(model) is not dict:
        raise UserError("model must be an object.")
    allowed = {"id", "name", "engine", "source", "description", "architecture", "argv", "quantization"}
    if model.keys() - allowed:
        raise UserError("Unknown custom model field.")
    result = dict(model)
    for key in ("id", "name", "engine"):
        if type(result.get(key)) is not str or not result[key] or "\0" in result[key]:
            raise UserError(f"Model {key} must be a nonempty string.")
    result["engine"] = {"whisper": "faster-whisper", "parakeet": "onnx-asr"}.get(result["engine"], result["engine"])
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", result["id"]):
        raise UserError("Model ID must contain only letters, numbers, dots, underscores and hyphens.")
    if result["engine"] not in {"faster-whisper", "onnx-asr", "command"}:
        raise UserError("Model engine must be whisper, parakeet or command.")
    source = result.get("source", "")
    if type(source) is not str or "\0" in source or (not source and result["engine"] != "command"):
        raise UserError("Model source must be a nonempty string, except for command adapters.")
    for key in ("description", "architecture", "quantization"):
        if key in result and (type(result[key]) is not str or "\0" in result[key] or len(result[key]) > 2000):
            raise UserError(f"Model {key} must be a string up to 2000 characters.")
    if result["engine"] == "onnx-asr":
        result.setdefault("quantization", "int8")
    result.setdefault("description", "User-added compatible model; capabilities depend on its architecture." +
                      (" English-only and turbo models cannot translate." if result["engine"] == "faster-whisper" else ""))
    result["source"] = source
    if result["engine"] == "command":
        argv = result.get("argv")
        if type(argv) is not list or not argv or len(argv) > 128 or any(
            type(a) is not str or "\0" in a or len(a) > 8000 for a in argv
        ):
            raise UserError("Command models require an argv array of strings.")
        if not argv[0] or not any("{audio}" in a for a in argv[1:]):
            raise UserError("Command argv requires an executable and an {audio} placeholder.")
        result["source"] = source or argv[0]
    elif source.startswith(("/", "./", "../", "~")):
        path = Path(source).expanduser().resolve()
        if not path.is_dir():
            raise UserError("Local model source must be an existing directory.")
        result["source"] = str(path)
    elif not re.fullmatch(r"[\w.-]+/[\w.-]+", source):
        raise UserError("Source must be a Hugging Face owner/repository or an existing local directory.")
    if len(result["name"]) > 200 or len(result["source"]) > 2000:
        raise UserError("Model name or source is too long.")
    return result


class Models:
    def __init__(self, paths):
        self.paths = paths
        self.entries = {item["id"]: item for item in catalog()}
        self.custom = []
        registry = paths.config / "models.json"
        if registry.exists():
            records = json.loads(registry.read_text())
            if type(records) is not list:
                raise UserError("Custom model registry must be an array.")
            for raw in records:
                model = validate_model(raw)
                if model["id"] in self.entries:
                    raise UserError(f"Duplicate model ID in registry: {model['id']}")
                self.custom.append(model)
                self.entries[model["id"]] = model
        self.loaded_id = None
        self.instance = None
        self.load_key = None
        self.runtime = {}
        self.progress = {}

    def get(self, identifier: str) -> dict:
        if type(identifier) is not str or identifier not in self.entries:
            raise UserError(f"Unknown model: {identifier}")
        return self.entries[identifier]

    def directory(self, model: dict) -> Path:
        source = Path(model["source"])
        return source if source.is_absolute() else self.paths.models / model["id"]

    def installed(self, model: dict) -> bool:
        if model["engine"] == "command":
            return shutil.which(model["argv"][0]) is not None
        path = self.directory(model)
        if not path.is_dir():
            return False
        if not Path(model["source"]).is_absolute() and not (path / ".omawhisper-complete").is_file():
            return False
        if model["engine"] == "faster-whisper":
            return (path / "model.bin").is_file() and (path / "config.json").is_file()
        return any(path.rglob("*.onnx")) or any(path.rglob("*.ort"))

    def snapshot(self) -> list[dict]:
        instance = self.instance
        alive = not isinstance(instance, EngineWorker) or instance.alive
        memory = memory_status()
        entries = tuple(self.entries.values())
        estimates = {model["id"]: self.memory_requirement(model)
                     for model in entries if model["engine"] == "onnx-asr"}
        return [{
            **{key: model[key] for key in ("id", "name", "engine", "source", "description")},
            "engine": ENGINE_NAMES[model["engine"]],
            **({
                "quantization": onnx_quantization(model) or "",
                "gpu_compatible": onnx_gpu_compatible(onnx_quantization(model)),
                "estimated_ram_bytes": estimates[model["id"]],
                "memory_error": memory_error(model["name"], estimates[model["id"]], memory["total"]),
            } if model["engine"] == "onnx-asr" else {}),
            "installed": self.installed(model), "loaded": alive and self.loaded_id == model["id"],
            "progress": self.progress.get(model["id"], 100 if self.installed(model) else 0),
        } for model in entries]

    def runtime_snapshot(self) -> dict:
        instance = self.instance
        if isinstance(instance, EngineWorker) and not instance.alive:
            return {}
        return dict(self.runtime)

    def memory_requirement(self, model: dict) -> int:
        if model["engine"] != "onnx-asr":
            return 0
        directory = self.directory(model)
        files = {}
        for path in directory.rglob("*"):
            relative = path.relative_to(directory)
            if ".cache" in relative.parts:
                continue
            try:
                info = path.stat()
            except FileNotFoundError:
                # Status snapshots can overlap removal of downloaded model files.
                continue
            if stat.S_ISREG(info.st_mode):
                files[str(relative)] = info.st_size
        selected = onnx_download_files(list(files), onnx_quantization(model))
        weights = sum(files[name] for name in selected
                      if Path(name).suffix not in {".json", ".yaml", ".yml", ".txt", ".model"})
        # Initializers and optimized weights can coexist while ORT builds its sessions.
        estimate = weights * 2 + 256 * MIB if weights else 0
        return max(model.get("estimated_ram_bytes", 0), estimate)

    def check_memory(self, identifier: str, *, available: bool = False) -> None:
        model = self.get(identifier)
        required = self.memory_requirement(model)
        if not required:
            return
        memory = memory_status()
        error = memory_error(model["name"], required,
                             memory["available" if available else "total"], available=available)
        if error:
            raise UserError(error)

    def add(self, raw: dict) -> None:
        if len(self.custom) >= 100:
            raise UserError("At most 100 custom models may be registered.")
        model = validate_model(raw)
        if model["id"] in self.entries:
            raise UserError("A model with that ID already exists.")
        atomic_json(self.paths.config / "models.json", self.custom + [model])
        self.custom.append(model)
        self.entries[model["id"]] = model

    def check_settings(self, identifier: str, config: dict) -> None:
        model = self.get(identifier)
        if (model["engine"] == "onnx-asr" and config["acceleration"] == "gpu"
                and not onnx_gpu_compatible(onnx_quantization(model))):
            raise UserError("These quantized ONNX weights are CPU-oriented. Choose Parakeet v3 FP32 in Models for GPU inference.")
        if model["engine"] != "faster-whisper":
            unsupported = [key for key in ("translate", "initial_prompt", "show_timestamps") if config[key]]
            if config["language"] != "auto":
                unsupported.append("language")
            if unsupported:
                raise UserError(f"{model['name']} does not support: {', '.join(unsupported)}. Use automatic language, no translation, no prompt and no timestamps.")
        if model["engine"] == "faster-whisper":
            names = (identifier.lower(), Path(model["source"]).name.lower())
            if any(name.endswith(".en") for name in names):
                if config["translate"]:
                    raise UserError("English-only Whisper models do not support translation.")
                if config["language"] not in {"auto", "en"}:
                    raise UserError("This English-only Whisper model requires automatic language or en.")
            if config["translate"] and any(re.search(r"(?:^|[-_])turbo(?:$|[-_.])", name) for name in names):
                raise UserError("Whisper turbo is not trained for translation; choose another Whisper model.")

    def download(self, identifier: str, cancelled: threading.Event) -> None:
        model = self.get(identifier)
        if model["engine"] == "command" or Path(model["source"]).is_absolute():
            if not self.installed(model):
                raise UserError("Local model files or command executable are missing.")
            return
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise UserError("Install the appropriate Omawhisper whisper or parakeet extra before downloading.") from exc
        self.paths.models.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = self.directory(model)
        destination.mkdir(mode=0o700, exist_ok=True)
        self.progress[identifier] = -1
        try:
            if model["engine"] == "faster-whisper":
                patterns = ["*.json", "*.txt", "*.model", "model.bin"]
            else:
                from huggingface_hub import HfApi

                filenames = HfApi().list_repo_files(repo_id=model["source"])
                patterns = onnx_download_files(filenames, onnx_quantization(model))
                if not any(name.endswith((".onnx", ".ort")) for name in patterns):
                    raise UserError("Repository does not contain ONNX weights for the selected quantization.")
            if cancelled.is_set():
                raise UserError("Model download cancelled; partial files can be resumed.")
            snapshot_download(repo_id=model["source"], local_dir=str(destination),
                              allow_patterns=patterns, max_workers=2)
            if cancelled.is_set():
                raise UserError("Model download cancelled; partial files can be resumed.")
            if model["engine"] == "faster-whisper":
                if not (destination / "model.bin").is_file() or not (destination / "config.json").is_file():
                    raise UserError("Repository is not a compatible CTranslate2 faster-whisper model.")
            elif not any(destination.rglob("*.onnx")) and not any(destination.rglob("*.ort")):
                raise UserError("Repository does not contain ONNX model files.")
            atomic_json(destination / ".omawhisper-complete", {"source": model["source"]})
            self.progress[identifier] = 100
        except BaseException:
            self.progress[identifier] = 0
            raise

    def unload(self) -> None:
        if isinstance(self.instance, EngineWorker):
            self.instance.close()
        self.instance = None
        self.loaded_id = None
        self.load_key = None
        self.runtime = {}
        gc.collect()

    def load(self, identifier: str, config: dict, cancelled: threading.Event | None = None) -> None:
        self.check_settings(identifier, config)
        model = self.get(identifier)
        key = identifier, config["acceleration"], config["compute_type"], config["threads"]
        if key == self.load_key and (not isinstance(self.instance, EngineWorker) or self.instance.alive):
            return
        if not self.installed(model):
            raise UserError(f"{model['name']} is not installed. Download it in Models first.")
        self.check_memory(identifier)
        self.unload()
        self.check_memory(identifier, available=True)
        path = str(self.directory(model))
        if model["engine"] in {"faster-whisper", "onnx-asr"}:
            python = engine_python(self.paths, model["engine"], config["acceleration"],
                                   onnx_gpu_compatible(onnx_quantization(model)))
            instance = EngineWorker(python, model["engine"])
            try:
                runtime = instance.request("load", model=model, path=str(Path(path).resolve()),
                                           config=config, cancelled=cancelled)
            except WorkerCancelled:
                instance.close()
                return
            except BaseException:
                instance.close()
                raise
        else:
            instance = model["argv"]
            runtime = {"device": "external", "provider": "", "compute_type": "",
                       "detail": "The custom command manages its own hardware and precision."}
        self.instance, self.loaded_id, self.load_key = instance, identifier, key
        self.runtime = runtime

    def transcribe(self, audio: Path, identifier: str, config: dict, cancelled: threading.Event) -> str:
        self.load(identifier, config, cancelled)
        if cancelled.is_set():
            return ""
        model = self.get(identifier)
        if model["engine"] in {"faster-whisper", "onnx-asr"}:
            try:
                text = self.instance.request("transcribe", audio=str(audio.resolve()), config=config, cancelled=cancelled)
            except WorkerCancelled:
                return ""
            finally:
                if not self.instance.alive:
                    self.unload()
        else:
            argv = [arg.replace("{audio}", str(audio)) for arg in self.instance]
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                stdout, stderr = bytearray(), bytearray()
                deadline = time.monotonic() + 600
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ, stdout)
                    selector.register(process.stderr, selectors.EVENT_READ, stderr)
                    while selector.get_map():
                        if cancelled.is_set():
                            return ""
                        if time.monotonic() >= deadline:
                            raise UserError("Command transcription timed out after 10 minutes.")
                        for key, _ in selector.select(timeout=0.1):
                            chunk = key.fileobj.read1(16384)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            key.data.extend(chunk)
                            if len(key.data) > 1024 * 1024:
                                raise UserError("Command output exceeded the 1 MiB safety limit.")
                while process.poll() is None:
                    if cancelled.is_set():
                        return ""
                    if time.monotonic() >= deadline:
                        raise UserError("Command transcription timed out after 10 minutes.")
                    cancelled.wait(0.05)
                if process.returncode:
                    raise UserError(f"Command model failed ({process.returncode}): {stderr.decode(errors='replace')[:1000]}")
                text = stdout.decode("utf-8").strip()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
                process.stderr.close()
        if not isinstance(text, str):
            raise UserError("Speech engine did not return a text transcript.")
        if len(text) > 100000:
            raise UserError("Transcript exceeds the 100,000 character safety limit.")
        return text.strip()

    def remove(self, identifier: str) -> None:
        model = self.get(identifier)
        remaining = [row for row in self.custom if row["id"] != identifier]
        custom = len(remaining) != len(self.custom)
        external = model["engine"] == "command" or Path(model["source"]).is_absolute()
        if not external:
            directory = self.directory(model)
            if directory.is_symlink():
                raise UserError("Refusing to remove a symlinked model directory.")
            if self.loaded_id == identifier:
                self.unload()
            if directory.exists():
                shutil.rmtree(directory)
        if custom:
            atomic_json(self.paths.config / "models.json", remaining)
            if self.loaded_id == identifier:
                self.unload()
            self.custom = remaining
            self.entries.pop(identifier)
        self.progress.pop(identifier, None)
