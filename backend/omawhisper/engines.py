"""Engine implementations imported only inside their own runtime processes."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .acceleration import cuda_available, onnx_gpu_compatible, whisper_runtime
from .config import UserError

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


class WhisperEngine:
    def load(self, model: dict, path: str, config: dict) -> dict:
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise UserError("Whisper runtime is missing. Load the model from Models to install it.") from exc
        runtime = whisper_runtime(ctranslate2, config)
        try:
            self.model = WhisperModel(path, device=runtime["device"], compute_type=runtime["compute_type"],
                                      cpu_threads=config["threads"], local_files_only=True)
        except RuntimeError as exc:
            if runtime["device"] != "cuda":
                raise
            raise UserError(f"CUDA model loading failed: {exc}. Choose GPU in Models and select a model to install CUDA libraries, "
                            "check available GPU memory, or choose CPU.") from exc
        runtime["device"] = self.model.model.device
        runtime["compute_type"] = self.model.model.compute_type
        return runtime

    def transcribe(self, audio: str, config: dict) -> str:
        segments, _ = self.model.transcribe(
            audio, language=None if config["language"] == "auto" else config["language"],
            task="translate" if config["translate"] else "transcribe",
            beam_size=config["beam_size"] if config["use_beam_search"] else 1, best_of=5,
            initial_prompt=config["initial_prompt"] or None, vad_filter=config["vad"],
            suppress_blank=config["suppress_blank"], without_timestamps=not config["show_timestamps"],
            temperature=config["temperature"], no_speech_threshold=config["no_speech_threshold"],
        )
        result = [
            f"[{timestamp(segment.start)}->{timestamp(segment.end)}] {segment.text.strip()}"
            if config["show_timestamps"] else segment.text
            for segment in segments
        ]
        return ("\n" if config["show_timestamps"] else "").join(result).strip()


def onnx_device(runtime, config: dict, quantization: str | None) -> tuple[str, str]:
    if config["acceleration"] == "cpu":
        return CPU, "CPU selected."
    if not onnx_gpu_compatible(quantization):
        if config["acceleration"] == "gpu":
            raise UserError("These quantized ONNX weights are CPU-oriented. Choose Parakeet v3 FP32 in Models for GPU inference.")
        return CPU, "Auto selected CPU for quantized weights; choose Parakeet v3 FP32 for GPU inference."
    if CUDA not in runtime.get_available_providers():
        available, detail = False, "The ONNX CUDA runtime is not installed. Choose GPU in Models and select a model to install it."
    else:
        available, detail = cuda_available()
    if not available:
        if config["acceleration"] == "gpu":
            raise UserError(detail)
        return CPU, "Auto selected CPU: " + detail
    return CUDA, "CUDA inference; audio preprocessing and auxiliary operations may use CPU."


def verify_onnx_sessions(model, runtime, provider: str) -> list:
    # onnx-asr 0.12 exposes the ASR object; its direct session members exclude preprocessing.
    sessions = [value for value in vars(model.asr).values() if isinstance(value, runtime.InferenceSession)]
    if not sessions:
        raise UserError("Cannot verify the model's ONNX sessions. Load the model from Models to update the supported engine runtime.")
    for session in sessions:
        if provider not in session.get_providers():
            raise UserError(f"ONNX did not initialize {provider}; refusing to claim GPU inference after CPU fallback. "
                            "Check the NVIDIA driver and CUDA libraries, or choose CPU.")
        session.disable_fallback()
    return sessions


def verify_cuda_execution(sessions: list, model) -> None:
    import numpy as np

    ended = set()
    try:
        # Profile synthetic silence only, never a user's recording.
        model.recognize(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        for index, session in enumerate(sessions):
            profile = Path(session.end_profiling())
            ended.add(index)
            events = json.loads(profile.read_text())
            if not any(event.get("cat") == "Node" and event.get("args", {}).get("provider") == CUDA for event in events):
                raise UserError("The ONNX model did not execute on CUDA. Choose GPU-compatible FP32 weights or CPU.")
    finally:
        for index, session in enumerate(sessions):
            if index not in ended:
                session.end_profiling()


class OnnxEngine:
    def load(self, model: dict, path: str, config: dict) -> dict:
        try:
            import onnx_asr
            import onnxruntime as runtime
        except ImportError as exc:
            raise UserError("ONNX runtime is missing. Load the model from Models to install Parakeet support.") from exc
        quantization = model.get("quantization", "int8") or None
        provider, detail = onnx_device(runtime, config, quantization)
        architecture = model.get("architecture") or model["source"]
        if Path(architecture).is_absolute():
            metadata_path = Path(path) / "config.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            architecture = metadata.get("model_type")
            if type(architecture) is not str or not architecture or "/" in architecture:
                raise UserError("Local ONNX models require config.json model_type or an architecture field, e.g. nemo-conformer-tdt.")
        options = runtime.SessionOptions()
        options.intra_op_num_threads = config["threads"]
        options.inter_op_num_threads = 1
        asr_options = runtime.SessionOptions()
        asr_options.intra_op_num_threads = config["threads"]
        asr_options.inter_op_num_threads = 1
        with tempfile.TemporaryDirectory(prefix="omawhisper-onnx-") as directory:
            if provider == CUDA:
                asr_options.enable_profiling = True
                asr_options.profile_file_prefix = str(Path(directory) / "warmup")
            self.model = onnx_asr.load_model(
                architecture, path=path, quantization=quantization,
                sess_options=options, providers=[CPU],
                asr_config={"sess_options": asr_options, "providers": [CUDA, CPU] if provider == CUDA else [CPU]},
            )
            sessions = verify_onnx_sessions(self.model, runtime, provider)
            if provider == CUDA:
                verify_cuda_execution(sessions, self.model)
        return {"device": "cuda" if provider == CUDA else "cpu", "provider": provider,
                "compute_type": quantization or "float32", "detail": detail}

    def transcribe(self, audio: str, config: dict) -> str:
        return self.model.recognize(audio)
