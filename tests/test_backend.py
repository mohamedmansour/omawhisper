import asyncio
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import wave

from omawhisper.audio import Capture, rms
from omawhisper.acceleration import whisper_runtime
from omawhisper.config import DEFAULTS, Paths, UserError, atomic_json, validate
from omawhisper.daemon import Daemon
from omawhisper.models import Models, onnx_download_files, validate_model
from omawhisper.engines import OnnxEngine, WhisperEngine
from omawhisper.output import MIME, Output
from omawhisper.shortcuts import Shortcuts, lua_string


class Files:
    def setUp(self):
        self.root = Path("tests") / (".backend-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.paths = types.SimpleNamespace(
            config=self.root / "config", state=self.root / "state", data=self.root / "data",
            runtime=self.root / "runtime", models=self.root / "data/models",
            socket=self.root / "runtime/control.sock",
            load_config=lambda: dict(DEFAULTS),
        )
        self.paths.save_config = lambda config: atomic_json(self.paths.config / "config.json", config)

    def tearDown(self):
        shutil.rmtree(self.root)


class ConfigTests(Files, unittest.TestCase):
    def test_defaults_and_normalization(self):
        self.assertEqual(validate({}), DEFAULTS)
        self.assertEqual(DEFAULTS["shortcut"], "SUPER+ALT+V")
        self.assertEqual(validate({"shortcut": "super + ctrl + v"})["shortcut"], "SUPER+CTRL+V")
        self.assertEqual(validate({"paste_shortcut": "ctrl+shift+v"})["paste_shortcut"], "ctrl+shift+v")
        self.assertEqual(validate({"paste_shortcut": "CTRL+V"})["paste_shortcut"], "ctrl+v")

    def test_symbol_shortcut_key_names(self):
        for key in ("COMMA", "PERIOD", "SLASH", "SEMICOLON", "APOSTROPHE",
                    "BRACKETLEFT", "BRACKETRIGHT", "BACKSLASH", "GRAVE", "MINUS", "EQUAL", "PLUS"):
            with self.subTest(key=key):
                self.assertEqual(validate({"shortcut": f"super + {key.lower()}"})["shortcut"], f"SUPER+{key}")

    def test_push_to_talk_defaults_on_and_preserves_explicit_toggle_preference(self):
        self.assertEqual(validate({})["activation"], "hold")
        saved = validate({"activation": "toggle"})
        self.assertEqual(saved["activation"], "toggle")
        self.assertEqual(validate({"threads": 8}, saved)["activation"], "toggle")

    def test_acceleration_defaults_and_preserves_existing_precision(self):
        self.assertEqual(validate({})["acceleration"], "auto")
        self.assertEqual(validate({})["compute_type"], "auto")
        saved = validate({"compute_type": "int8", "device": "alsa_input.usb"})
        result = validate({"acceleration": "gpu"}, saved)
        self.assertEqual(result["compute_type"], "int8")
        self.assertEqual(result["device"], "alsa_input.usb")
        for precision in ("float16", "int8_float16", "int8_float32", "float32"):
            self.assertEqual(validate({"compute_type": precision})["compute_type"], precision)

    def test_strict_validation(self):
        for values in ({"threads": True}, {"translate": "false"}, {"max_duration": 0},
                       {"beam_size": 21}, {"threads": 65}, {"unknown": 1}, [],
                       {"language": "English"}, {"shortcut": "CTRL+CTRL+V"},
                       {"output": "shell"}, {"model": "../bad"}, {"initial_prompt": "\0"},
                       {"max_duration": 1.5}, {"bar_section": "middle"}, {"temperature": True},
                       {"temperature": "0.5"}, {"temperature": -0.1}, {"temperature": 1.1},
                       {"temperature": float("nan")}, {"no_speech_threshold": float("inf")},
                       {"no_speech_threshold": -0.1}, {"suppress_blank": 1},
                       {"show_timestamps": "false"}, {"use_beam_search": 1},
                       {"acceleration": "cuda"}, {"acceleration": True}, {"compute_type": "fp16"}):
            with self.subTest(values=values), self.assertRaises(UserError):
                validate(values)

    def test_decoding_defaults_and_json_numbers(self):
        config = validate({"temperature": 0, "no_speech_threshold": 1})
        self.assertEqual(config["temperature"], 0.0)
        self.assertIs(type(config["temperature"]), float)
        self.assertEqual(config["no_speech_threshold"], 1.0)
        self.assertTrue(config["suppress_blank"])
        self.assertFalse(config["show_timestamps"])
        self.assertFalse(config["use_beam_search"])

    def test_private_atomic_storage(self):
        target = self.root / "settings.json"
        atomic_json(target, {"message": "héllo"})
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(target.read_text()), {"message": "héllo"})
        atomic_json(target, {})
        self.assertEqual(list(self.root.iterdir()), [target])

    def test_runtime_required(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(UserError):
            Paths()


class AccelerationTests(unittest.TestCase):
    def setUp(self):
        self.ct2 = MagicMock()
        self.ct2.get_cuda_device_count.return_value = 0
        self.ct2.get_supported_compute_types.side_effect = lambda device: (
            {"float16", "float32", "int8_float16", "int8"} if device == "cuda"
            else {"int8", "int8_float32", "float32"}
        )

    def test_auto_without_gpu_uses_cpu_and_explains_why(self):
        runtime = whisper_runtime(self.ct2, validate({}))
        self.assertEqual(runtime["device"], "cpu")
        self.assertEqual(runtime["compute_type"], "int8")
        self.assertIn("CUDA unavailable to CTranslate2", runtime["detail"])

    def test_auto_prefers_gpu_and_gpu_precision(self):
        self.ct2.get_cuda_device_count.return_value = 1
        for mode in ("auto", "gpu"):
            with self.subTest(mode=mode):
                runtime = whisper_runtime(self.ct2, validate({"acceleration": mode}))
                self.assertEqual(runtime["device"], "cuda")
                self.assertEqual(runtime["compute_type"], "float16")

    def test_explicit_cpu_never_probes_cuda(self):
        self.ct2.get_cuda_device_count.side_effect = RuntimeError("broken CUDA")
        runtime = whisper_runtime(self.ct2, validate({"acceleration": "cpu"}))
        self.assertEqual(runtime["device"], "cpu")
        self.ct2.get_cuda_device_count.assert_not_called()

    def test_gpu_request_without_gpu_is_an_error(self):
        with self.assertRaisesRegex(UserError, "No supported CUDA GPU"):
            whisper_runtime(self.ct2, validate({"acceleration": "gpu"}))
        self.ct2.get_supported_compute_types.assert_not_called()

    def test_auto_precision_uses_only_supported_types(self):
        self.ct2.get_supported_compute_types.side_effect = None
        self.ct2.get_supported_compute_types.return_value = {"float32"}
        for count, device in ((0, "cpu"), (1, "cuda")):
            self.ct2.get_cuda_device_count.return_value = count
            runtime = whisper_runtime(self.ct2, validate({}))
            self.assertEqual(runtime["device"], device)
            self.assertEqual(runtime["compute_type"], "float32")

    def test_unsupported_explicit_precision_is_not_silently_changed(self):
        with self.assertRaisesRegex(UserError, "float16 is not supported on CPU"):
            whisper_runtime(self.ct2, validate({"compute_type": "float16"}))

    def test_device_detection_failure_is_actionable(self):
        self.ct2.get_cuda_device_count.side_effect = RuntimeError("driver failure")
        with self.assertRaisesRegex(UserError, "Cannot detect CUDA.*driver failure"):
            whisper_runtime(self.ct2, validate({}))

    def test_initialization_failure_is_not_masked_as_cpu_success(self):
        self.ct2.get_cuda_device_count.return_value = 1
        self.ct2.get_supported_compute_types.side_effect = RuntimeError("missing CUDA library")
        with self.assertRaisesRegex(UserError, "Cannot initialize CUDA.*missing CUDA library"):
            whisper_runtime(self.ct2, validate({}))


class ModelTests(Files, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.models = Models(self.paths)
        self.ct2 = MagicMock()
        self.ct2.get_cuda_device_count.return_value = 0
        self.ct2.get_supported_compute_types.return_value = {"int8", "float32", "int8_float32", "float16"}
        mock_ct2 = patch.dict(sys.modules, {"ctranslate2": self.ct2})
        mock_ct2.start()
        self.addCleanup(mock_ct2.stop)

    def test_catalog_and_snapshot_contract(self):
        rows = self.models.snapshot()
        self.assertEqual(len(rows), 12)
        self.assertEqual(set(rows[0]), {"id", "name", "engine", "source", "description", "installed", "loaded", "progress"})
        self.assertFalse(any(row["installed"] for row in rows))
        self.assertEqual(self.models.get("parakeet-v3")["engine"], "onnx-asr")
        self.assertEqual({row["engine"] for row in rows}, {"whisper", "parakeet"})

    def test_ui_model_engine_aliases(self):
        self.models.add({"id": "custom-whisper", "name": "Whisper", "engine": "whisper", "source": "owner/repo"})
        self.models.add({"id": "custom-parakeet", "name": "Parakeet", "engine": "parakeet", "source": "owner/repo"})
        self.assertEqual(self.models.get("custom-whisper")["engine"], "faster-whisper")
        self.assertEqual(self.models.get("custom-parakeet")["engine"], "onnx-asr")

    def test_snapshot_exposes_custom_onnx_quantization_and_gpu_compatibility(self):
        variants = (
            ("unquantized", {"quantization": ""}, "", True),
            ("fp16", {"quantization": "fp16"}, "fp16", True),
            ("fp32", {"quantization": "fp32"}, "fp32", True),
            ("int8", {"quantization": "int8"}, "int8", False),
            ("default", {}, "int8", False),
        )
        for identifier, extra, _, _ in variants:
            self.models.add({"id": identifier, "name": identifier, "engine": "parakeet",
                             "source": "owner/model", **extra})
        rows = {row["id"]: row for row in self.models.snapshot()}
        for identifier, _, quantization, compatible in variants:
            with self.subTest(identifier=identifier):
                self.assertEqual(rows[identifier]["quantization"], quantization)
                self.assertIs(rows[identifier]["gpu_compatible"], compatible)
        self.assertEqual(rows["parakeet-v3"]["quantization"], "int8")
        self.assertIs(rows["parakeet-v3"]["gpu_compatible"], False)
        self.assertNotIn("quantization", rows["whisper-base"])

    def test_snapshot_tolerates_model_removal_during_filesystem_checks(self):
        self.models.add({"id": "removed", "name": "Removed", "engine": "whisper", "source": "owner/repo"})
        installed = self.models.installed

        def check_installed(model):
            if "removed" in self.models.entries:
                self.models.remove("removed")
            return installed(model)

        self.models.installed = check_installed
        snapshot = self.models.snapshot()
        self.assertEqual(len(snapshot), 13)
        self.assertNotIn("removed", self.models.entries)
        self.assertEqual(len(self.models.snapshot()), 12)

    def test_add_and_persist_command(self):
        model = {"id": "custom", "name": "Custom", "source": "local-command",
                 "engine": "command", "argv": [sys.executable, "-c", "print('test')", "{audio}"]}
        self.models.add(model)
        self.assertTrue(self.models.installed(self.models.get("custom")))
        self.assertIn("custom", Models(self.paths).entries)
        with self.assertRaises(UserError):
            self.models.add(model)

    def test_command_source_may_be_empty_or_omitted(self):
        for identifier, extra in (("empty", {"source": ""}), ("omitted", {})):
            with self.subTest(identifier=identifier):
                self.models.add({"id": identifier, "name": "Adapter", "engine": "command",
                                 "argv": [sys.executable, "{audio}"], **extra})
                self.assertEqual(self.models.get(identifier)["source"], sys.executable)
                self.assertEqual(Models(self.paths).get(identifier)["source"], sys.executable)
        with self.assertRaises(UserError):
            validate_model({"id": "bad", "name": "Bad", "engine": "whisper", "source": ""})

    def test_english_only_and_custom_turbo_cannot_translate(self):
        self.models.add({"id": "english-custom", "name": "English", "engine": "whisper",
                         "source": "Systran/faster-whisper-base.en"})
        self.models.add({"id": "fast-custom", "name": "Fast", "engine": "whisper",
                         "source": "owner/faster-whisper-large-v3-turbo"})
        for identifier in ("whisper-tiny.en", "whisper-base.en", "whisper-small.en",
                           "whisper-medium.en", "english-custom", "fast-custom"):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(UserError, "translation"):
                self.models.check_settings(identifier, validate({"translate": True}))
        self.models.check_settings("whisper-base", validate({"translate": True}))
        self.assertIn("not trained for translation", self.models.get("whisper-turbo")["description"])
        self.assertIn("capabilities depend", self.models.get("fast-custom")["description"])

    def test_validation_and_unsupported_settings(self):
        for engine in ("onnx-asr", "command"):
            with self.subTest(engine=engine), self.assertRaises(UserError):
                validate_model({"id": "x", "name": "X", "engine": engine, "source": "bad"})
        for config in ({"language": "en"}, {"translate": True}, {"initial_prompt": "names"}, {"show_timestamps": True}):
            with self.subTest(config=config), self.assertRaises(UserError):
                self.models.check_settings("parakeet-v3", validate(config))
        with self.assertRaises(UserError):
            self.models.check_settings("whisper-turbo", validate({"translate": True}))

    def test_parakeet_gpu_requires_floating_weights(self):
        with self.assertRaisesRegex(UserError, "CPU-oriented"):
            self.models.load("parakeet-v3", validate({"acceleration": "gpu"}))
        self.assertIsNone(self.models.instance)
        self.assertEqual(self.models.runtime, {})
        self.models.check_settings("parakeet-v3", validate({"acceleration": "auto"}))
        self.models.check_settings("parakeet-v3", validate({"acceleration": "cpu"}))
        self.models.check_settings("parakeet-v3-fp32", validate({"acceleration": "gpu"}))

    def test_partial_download_not_installed(self):
        model = self.models.get("whisper-base")
        directory = self.models.directory(model)
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "model.bin").write_bytes(b"test")
        self.assertFalse(self.models.installed(model))
        atomic_json(directory / ".omawhisper-complete", {})
        self.assertTrue(self.models.installed(model))
        self.models.remove("whisper-base")
        self.assertFalse(directory.exists())
        self.assertIn("whisper-base", self.models.entries)

    def test_remove_custom_local_unregisters_without_deleting_external_files(self):
        directory = self.root / "external-model"
        directory.mkdir()
        weights = directory / "model.bin"
        weights.write_bytes(b"external weights")
        self.models.add({"id": "local", "name": "Local", "engine": "whisper", "source": str(directory.resolve())})
        self.models.loaded_id, self.models.instance = "local", object()
        self.models.remove("local")
        self.assertEqual(weights.read_bytes(), b"external weights")
        self.assertNotIn("local", self.models.entries)
        self.assertNotIn("local", Models(self.paths).entries)
        self.assertIsNone(self.models.loaded_id)

    def test_remove_custom_remote_deletes_only_owned_weights_and_unregisters(self):
        self.models.add({"id": "remote", "name": "Remote", "engine": "whisper", "source": "owner/repo"})
        owned = self.paths.models / "remote"
        owned.mkdir(parents=True)
        (owned / "model.bin").write_bytes(b"download")
        neighbor = self.paths.models / "unrelated"
        neighbor.mkdir()
        (neighbor / "keep").write_text("keep")
        self.models.remove("remote")
        self.assertFalse(owned.exists())
        self.assertTrue((neighbor / "keep").exists())
        self.assertNotIn("remote", Models(self.paths).entries)

    def test_remove_local_registry_failure_keeps_registration_and_external_files(self):
        directory = self.root / "external-model"
        directory.mkdir()
        self.models.add({"id": "local", "name": "Local", "engine": "whisper", "source": str(directory.resolve())})
        before = (self.paths.config / "models.json").read_bytes()
        with patch("omawhisper.models.atomic_json", side_effect=OSError("registry is read-only")):
            with self.assertRaises(OSError):
                self.models.remove("local")
        self.assertTrue(directory.is_dir())
        self.assertIn("local", self.models.entries)
        self.assertEqual((self.paths.config / "models.json").read_bytes(), before)

    def test_remove_remote_symlink_never_deletes_external_directory(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.models.add({"id": "remote", "name": "Remote", "engine": "whisper", "source": "owner/repo"})
        self.paths.models.mkdir(parents=True)
        (self.paths.models / "remote").symlink_to(outside.resolve(), target_is_directory=True)
        with self.assertRaises(UserError):
            self.models.remove("remote")
        self.assertTrue(outside.is_dir())
        self.assertIn("remote", Models(self.paths).entries)

    def test_download_validation_and_resume(self):
        module = types.ModuleType("huggingface_hub")
        module.snapshot_download = MagicMock()
        with patch.dict(sys.modules, {"huggingface_hub": module}), self.assertRaisesRegex(UserError, "compatible"):
            self.models.download("whisper-base", threading.Event())
        self.assertEqual(self.models.progress["whisper-base"], 0)
        self.assertFalse(self.models.installed(self.models.get("whisper-base")))

    def test_command_inference_literal_argv_and_cancel(self):
        model = {"id": "command", "name": "Command", "source": "local",
                 "engine": "command", "argv": [sys.executable, "-c", "import sys;print(sys.argv[1])", "{audio}"]}
        self.models.add(model)
        path = self.root / "literal;not-a-shell.wav"
        config = validate({"model": "command"})
        self.assertEqual(self.models.transcribe(path, "command", config, threading.Event()), str(path))
        event = threading.Event()
        event.set()
        self.assertEqual(self.models.transcribe(path, "command", config, event), "")
        self.models.remove("command")
        self.assertNotIn("command", Models(self.paths).entries)
        self.assertTrue(Path(sys.executable).is_file())

    def test_faster_whisper_uses_only_local_files(self):
        directory = self.root / "local"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "model.bin").write_bytes(b"test")
        self.models.add({"id": "local", "name": "Local", "engine": "faster-whisper", "source": str(directory.resolve())})
        module = types.ModuleType("faster_whisper")
        module.WhisperModel = MagicMock()
        module.WhisperModel.return_value.model = types.SimpleNamespace(device="cpu", compute_type="int8_float32")
        module.WhisperModel.return_value.transcribe.return_value = ([types.SimpleNamespace(text=" Hello")], {})
        engine = WhisperEngine()
        with patch.dict(sys.modules, {"faster_whisper": module}):
            runtime = engine.load(self.models.get("local"), str(directory.resolve()), validate({}))
            text = engine.transcribe("input.wav", validate({}))
        self.assertEqual(text, "Hello")
        self.assertTrue(module.WhisperModel.call_args.kwargs["local_files_only"])
        self.assertEqual(module.WhisperModel.call_args.kwargs["device"], "cpu")
        self.assertEqual(module.WhisperModel.call_args.kwargs["compute_type"], "int8")
        self.assertEqual(runtime["device"], "cpu")
        self.assertEqual(runtime["compute_type"], "int8_float32")
        options = module.WhisperModel.return_value.transcribe.call_args.kwargs
        self.assertEqual(options["beam_size"], 1)
        self.assertEqual(options["best_of"], 5)
        self.assertTrue(options["suppress_blank"])
        self.assertTrue(options["without_timestamps"])
        self.assertEqual(options["temperature"], 0.0)
        self.assertEqual(options["no_speech_threshold"], 0.6)
        self.models.unload()
        self.assertIsNone(self.models.instance)
        self.assertEqual(self.models.runtime, {})

    def test_whisper_gpu_loading(self):
        module = types.ModuleType("faster_whisper")
        module.WhisperModel = MagicMock(side_effect=lambda path, **options: types.SimpleNamespace(
            model=types.SimpleNamespace(device=options["device"], compute_type=options["compute_type"])))
        self.ct2.get_cuda_device_count.return_value = 1
        with patch.dict(sys.modules, {"faster_whisper": module}):
            runtime = WhisperEngine().load({}, "model", validate({}))
            self.assertEqual(module.WhisperModel.call_args.kwargs["device"], "cuda")
            self.assertEqual(module.WhisperModel.call_args.kwargs["compute_type"], "float16")
            self.assertEqual(runtime["device"], "cuda")

    def test_cuda_model_load_error_does_not_leave_a_loaded_gpu_label(self):
        module = types.ModuleType("faster_whisper")
        module.WhisperModel = MagicMock(side_effect=RuntimeError("out of memory"))
        self.ct2.get_cuda_device_count.return_value = 1
        with patch.dict(sys.modules, {"faster_whisper": module}):
            with self.assertRaisesRegex(UserError, "CUDA model loading failed.*out of memory"):
                WhisperEngine().load({}, "model", validate({}))
        self.assertIsNone(self.models.instance)
        self.assertIsNone(self.models.loaded_id)
        self.assertEqual(self.models.runtime, {})
        module.WhisperModel.assert_called_once()

    def test_beam_decoding_and_segment_timestamps(self):
        instance = MagicMock()
        instance.transcribe.return_value = ([
            types.SimpleNamespace(text=" Hello", start=0.0, end=1.25),
            types.SimpleNamespace(text=" world", start=1.25, end=62.1),
        ], {})
        engine = WhisperEngine()
        engine.model = instance
        config = validate({"use_beam_search": True, "beam_size": 7, "show_timestamps": True,
                           "suppress_blank": False, "temperature": 0.4, "no_speech_threshold": 0.8})
        text = engine.transcribe("input.wav", config)
        self.assertEqual(text, "[00:00.000->00:01.250] Hello\n[00:01.250->01:02.100] world")
        options = instance.transcribe.call_args.kwargs
        self.assertEqual(options["beam_size"], 7)
        self.assertFalse(options["without_timestamps"])
        self.assertFalse(options["suppress_blank"])
        self.assertEqual(options["temperature"], 0.4)
        self.assertEqual(options["no_speech_threshold"], 0.8)

    def test_onnx_explicit_empty_quantization_is_unquantized(self):
        directory = self.root / "onnx-model"
        directory.mkdir()
        (directory / "encoder.onnx").write_bytes(b"test")
        (directory / "config.json").write_text('{"model_type":"nemo-conformer-tdt"}')
        asr = types.ModuleType("onnx_asr")
        asr.load_model = MagicMock()
        runtime = types.ModuleType("onnxruntime")
        runtime.SessionOptions = MagicMock()
        runtime.get_available_providers = lambda: ["CPUExecutionProvider"]

        class Session:
            def get_providers(self):
                return ["CPUExecutionProvider"]

            def disable_fallback(self):
                pass

        runtime.InferenceSession = Session
        asr.load_model.return_value.asr = types.SimpleNamespace(encoder=Session())
        cases = (
            ("empty", {"quantization": ""}, "int8", None),
            ("int8", {"quantization": "int8"}, "float32", "int8"),
            ("default-int8", {}, "int8", "int8"),
            ("default-float", {}, "float32", "int8"),
        )
        with patch.dict(sys.modules, {"onnx_asr": asr, "onnxruntime": runtime}):
            for identifier, extra, compute, expected in cases:
                with self.subTest(identifier=identifier):
                    self.models.add({"id": identifier, "name": "ONNX", "engine": "parakeet",
                                     "source": str(directory.resolve()), **extra})
                    result = OnnxEngine().load(self.models.get(identifier), str(directory.resolve()), validate({"compute_type": compute}))
                    self.assertEqual(asr.load_model.call_args.kwargs["quantization"], expected)
                    self.assertEqual(asr.load_model.call_args.kwargs["providers"], ["CPUExecutionProvider"])
                    self.assertEqual(result["device"], "cpu")
                    self.assertEqual(asr.load_model.call_args.kwargs["asr_config"]["providers"], ["CPUExecutionProvider"])

    def test_onnx_download_selects_quantized_weights_and_shared_preprocessors(self):
        published = [
            ".gitattributes", "README.md", "config.json", "vocab.txt", "nemo128.onnx",
            "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx",
            "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx",
        ]
        int8 = onnx_download_files(published, "int8")
        self.assertEqual(set(int8), {
            "config.json", "vocab.txt", "nemo128.onnx",
            "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx",
        })
        unquantized = onnx_download_files(published, None)
        self.assertEqual(set(unquantized), {
            "config.json", "vocab.txt", "nemo128.onnx",
            "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx",
        })

    def test_onnx_download_keeps_only_selected_variant_external_data(self):
        files = [
            "models/encoder.onnx", "models/encoder.onnx.data",
            "models/encoder.int8.onnx", "models/encoder.int8.onnx_data",
            "models/encoder.fp16.onnx", "models/encoder.fp16.onnx.data",
            "shared.onnx", "shared.onnx.data", "tokens.txt",
        ]
        self.assertEqual(set(onnx_download_files(files, "int8")), {
            "models/encoder.int8.onnx", "models/encoder.int8.onnx_data",
            "shared.onnx", "shared.onnx.data", "tokens.txt",
        })
        self.assertEqual(set(onnx_download_files(files, None)), {
            "models/encoder.onnx", "models/encoder.onnx.data",
            "shared.onnx", "shared.onnx.data", "tokens.txt",
        })

    def test_onnx_downloader_uses_repo_listing_and_selected_files(self):
        module = types.ModuleType("huggingface_hub")
        module.HfApi = MagicMock()
        module.HfApi.return_value.list_repo_files.return_value = [
            "config.json", "vocab.txt", "nemo128.onnx",
            "encoder-model.int8.onnx", "encoder-model.onnx", "encoder-model.onnx.data",
            "decoder_joint-model.int8.onnx", "decoder_joint-model.onnx",
        ]

        def download(**kwargs):
            for filename in kwargs["allow_patterns"]:
                (Path(kwargs["local_dir"]) / filename).write_bytes(b"test")

        module.snapshot_download = MagicMock(side_effect=download)
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            self.models.download("parakeet-v3", threading.Event())
        module.HfApi.return_value.list_repo_files.assert_called_once_with(
            repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx")
        patterns = module.snapshot_download.call_args.kwargs["allow_patterns"]
        self.assertIn("encoder-model.int8.onnx", patterns)
        self.assertIn("decoder_joint-model.int8.onnx", patterns)
        self.assertIn("nemo128.onnx", patterns)
        self.assertNotIn("encoder-model.onnx", patterns)
        self.assertNotIn("encoder-model.onnx.data", patterns)
        self.assertNotIn("decoder_joint-model.onnx", patterns)
        self.assertTrue(self.models.installed(self.models.get("parakeet-v3")))


class LauncherTests(Files, unittest.TestCase):
    def test_optional_cuda_libraries_are_added_before_python_starts(self):
        venv = self.paths.data / "omawhisper/venv"
        python = venv / "bin/python"
        python.parent.mkdir(parents=True)
        python.write_text(f"#!{sys.executable}\nimport os\nprint(os.environ.get('LD_LIBRARY_PATH', ''))\n")
        python.chmod(0o700)
        environment = {**os.environ, "XDG_DATA_HOME": str(self.paths.data.resolve()),
                       "LD_LIBRARY_PATH": "/existing/cuda/lib"}
        command = ["bash", "scripts/omawhisper", "request", '{"action":"status"}']
        result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "/existing/cuda/lib")
        libraries = venv / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cublas/lib"
        libraries.mkdir(parents=True)
        result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), f"{libraries.resolve()}:/existing/cuda/lib")

class ShortcutTests(unittest.IsolatedAsyncioTestCase):
    async def test_symbol_shortcuts_register_and_reject_existing_bindings(self):
        for key in ("COMMA", "PERIOD", "SLASH", "SEMICOLON", "APOSTROPHE",
                    "BRACKETLEFT", "BRACKETRIGHT", "BACKSLASH", "GRAVE", "MINUS", "EQUAL", "PLUS"):
            with self.subTest(key=key):
                runner = AsyncMock(side_effect=[b"[]", b"ok"])
                shortcuts = Shortcuts(runner)
                await shortcuts.apply(f"SUPER+{key}", "hold")
                self.assertIn(f"hl.bind({lua_string(f'SUPER+{key}')}", runner.call_args.args[-1])
                runner.side_effect = None
                runner.return_value = json.dumps([{"key": key.lower(), "modmask": 64, "description": "Other"}]).encode()
                with self.assertRaisesRegex(UserError, "conflicts"):
                    await shortcuts.apply(f"SUPER+{key}", "hold", force=True)

    async def test_default_does_not_steal_clipboard_manager(self):
        runner = AsyncMock(side_effect=[
            b'[{"key":"V","modmask":68,"description":"Clipboard manager"}]', b"ok",
        ])
        shortcuts = Shortcuts(runner)
        await shortcuts.apply(DEFAULTS["shortcut"], DEFAULTS["activation"])
        self.assertEqual(shortcuts.shortcut, "SUPER+ALT+V")

    async def test_collision_preserves_old_binding(self):
        runner = AsyncMock(return_value=b"[]")
        shortcuts = Shortcuts(runner)
        shortcuts.shortcut, shortcuts.activation = "SUPER+CTRL+V", "toggle"
        runner.return_value = json.dumps([{"key": "B", "modmask": 68, "description": "Other"}]).encode()
        with self.assertRaisesRegex(UserError, "conflicts"):
            await shortcuts.apply("SUPER+CTRL+B", "toggle")
        self.assertEqual(shortcuts.shortcut, "SUPER+CTRL+V")
        self.assertEqual(runner.await_count, 1)

    async def test_lua_hold_bind_and_scoped_release(self):
        runner = AsyncMock(side_effect=[b"[]", b"ok"])
        shortcuts = Shortcuts(runner)
        await shortcuts.apply("SUPER+CTRL+V", "hold")
        source = runner.call_args.args[-1]
        self.assertIn("release=true,ignore_mods=true,non_consuming=true", source)
        self.assertIn("repeating=false", source)
        self.assertIn("pcall", source)
        self.assertEqual(runner.call_args.args[:2], ("hyprctl", "eval"))
        self.assertNotIn("keyword", source)

    async def test_failed_eval_does_not_commit_settings(self):
        runner = AsyncMock(side_effect=[b"[]", b"Lua error"])
        shortcuts = Shortcuts(runner)
        with self.assertRaises(UserError):
            await shortcuts.apply("SUPER+CTRL+V", "toggle")
        self.assertIsNone(shortcuts.shortcut)

    async def test_hold_rejects_other_modifier_binding_for_release_key(self):
        runner = AsyncMock(return_value=b'[{"key":"V","modmask":4,"release":true,"description":"Release action"}]')
        with self.assertRaises(UserError):
            await Shortcuts(runner).apply("SUPER+CTRL+V", "hold")

    async def test_scoped_hold_release_does_not_steal_other_press_bindings(self):
        runner = AsyncMock(side_effect=[
            b'[{"key":"V","modmask":68,"release":false,"description":"Clipboard manager"},'
            b'{"key":"V","modmask":64,"release":false,"description":"Paste"}]', b"ok",
        ])
        shortcuts = Shortcuts(runner)
        await shortcuts.apply(DEFAULTS["shortcut"], "hold")
        self.assertEqual(shortcuts.shortcut, "SUPER+ALT+V")

    def test_lua_quoting(self):
        self.assertEqual(lua_string('"\n'), '"\\034\\010"')


class OutputTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runner = AsyncMock(return_value=b"ok")
        self.output = Output(self.runner)
        self.output.focused = AsyncMock(return_value={"address": "0x123", "class": "kitty"})
        self.output.layer_active = AsyncMock(return_value=False)
        self.owner = types.SimpleNamespace(returncode=None)

        async def set_clip(data, mime=MIME):
            self.output.owner = self.owner
        self.output._set = AsyncMock(side_effect=set_clip)
        self.output._previous = AsyncMock(return_value=(MIME, b"previous"))
        self.config = validate({})
        self.target = {"address": "0x123", "class": "kitty"}

    async def test_changed_focus_does_not_touch_clipboard_or_paste(self):
        self.output.focused.return_value = {"address": "0x456", "class": "browser"}
        result = await self.output.deliver("hello", self.target, self.config)
        self.assertIn("focus changed", result)
        self.output._set.assert_not_awaited()
        self.runner.assert_not_awaited()

    async def test_interactive_overlay_prevents_paste_even_with_same_window(self):
        self.output.layer_active.return_value = True
        result = await self.output.deliver("hello", self.target, self.config)
        self.assertIn("overlay", result)
        self.output._set.assert_not_awaited()
        self.runner.assert_not_awaited()

    async def test_overlay_opening_after_copy_prevents_dispatch(self):
        self.output.layer_active.side_effect = [False, True]
        self.runner.return_value = b"hello "
        result = await self.output.deliver("hello", self.target, self.config)
        self.assertIn("before insertion", result)
        self.assertFalse(any("hl.dsp.send_shortcut" in str(call.args) for call in self.runner.call_args_list))

    async def test_targeted_terminal_paste_and_restore(self):
        self.runner.side_effect = [b"ok", b"hello "]
        with patch("omawhisper.output.asyncio.sleep", new=AsyncMock()):
            result = await self.output.deliver("hello", self.target, self.config)
        self.assertIn("original window", result)
        args = self.runner.call_args_list[0].args
        self.assertEqual(args[:2], ("hyprctl", "repl"))
        self.assertIn("hl.dispatch(hl.dsp.send_shortcut", args[2])
        self.assertIn(f"mods={lua_string('CTRL SHIFT')}", args[2])
        self.assertIn(f"key={lua_string('v')}", args[2])
        self.assertIn(f"window={lua_string('address:0x123')}", args[2])
        self.assertIn("blocked-focus", args[2])
        self.assertIn("blocked-layer", args[2])
        self.assertEqual(self.output._set.call_args.args, (b"previous", MIME))

    async def test_new_clipboard_not_overwritten(self):
        self.runner.side_effect = [b"ok", b"copied elsewhere"]
        with patch("omawhisper.output.asyncio.sleep", new=AsyncMock()):
            await self.output.deliver("hello", self.target, self.config)
        self.assertEqual(self.output._set.await_count, 1)

    async def test_lost_clipboard_owner_not_restored(self):
        async def dispatch(*args):
            self.owner.returncode = 0
            return b"ok"
        self.runner.side_effect = dispatch
        with patch("omawhisper.output.asyncio.sleep", new=AsyncMock()):
            await self.output.deliver("hello", self.target, self.config)
        self.assertEqual(self.output._set.await_count, 1)
        self.assertEqual(self.runner.await_count, 1)

    async def test_focus_changes_after_copy(self):
        self.output.focused.side_effect = [self.target, {"address": "0x456"}]
        self.runner.return_value = b"hello "
        result = await self.output.deliver("hello", self.target, self.config)
        self.assertIn("before insertion", result)
        self.assertFalse(any("hl.dsp.send_shortcut" in str(call.args) for call in self.runner.call_args_list))

    async def test_atomic_lua_guard_blocks_last_instant_focus_change(self):
        self.runner.side_effect = [b"blocked-focus", b"hello "]
        result = await self.output.deliver("hello", self.target, self.config)
        self.assertIn("Not pasted: focus changed", result)
        self.assertEqual(self.output._set.call_args.args, (b"previous", MIME))

    async def test_lua_dispatch_errors_are_not_success_shaped(self):
        self.runner.return_value = b"error:window not found"
        with self.assertRaisesRegex(UserError, "window not found"):
            await self.output.dispatch(self.target, "CTRL", "v")

    async def test_clipboard_only_never_checks_focus(self):
        await self.output.deliver("hello", {}, validate({"output": "clipboard"}))
        self.output.focused.assert_not_awaited()
        self.assertEqual(self.output._set.call_args.args, (b"hello",))


class AudioTests(Files, unittest.IsolatedAsyncioTestCase):
    def test_rms(self):
        self.assertEqual(rms(b"\0" * 1600), 0)
        self.assertAlmostEqual(rms(struct.pack("<hh", 16384, -16384)), 0.5)
        self.assertEqual(rms(b""), 0)

    async def test_capture_raw_and_private_wave(self):
        proc = MagicMock()
        proc.stdout = asyncio.StreamReader()
        proc.stderr = asyncio.StreamReader()
        proc.stdout.feed_data(struct.pack("<h", 1000) * 800)
        proc.stdout.feed_eof()
        proc.stderr.feed_eof()
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        meter = MagicMock()
        capture = Capture(self.paths.runtime, "mic;literal", 120, meter)
        with patch("omawhisper.audio.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as spawn:
            await capture.start()
            path = await capture.wait()
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with wave.open(str(path)) as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getnframes()), (16000, 1, 800))
        self.assertIn("mic;literal", spawn.call_args.args)
        self.assertIn("--raw", spawn.call_args.args)
        self.assertGreater(meter.call_args.args[0], 0)
        capture.cleanup()
        self.assertFalse(path.exists())

    async def test_capture_failure_cleans_wave(self):
        capture = Capture(self.paths.runtime, "", 120, MagicMock())
        with patch("omawhisper.audio.asyncio.create_subprocess_exec", new=AsyncMock(side_effect=FileNotFoundError)):
            with self.assertRaises(UserError):
                await capture.start()
        self.assertFalse(list(self.paths.runtime.glob("*.wav")))

    async def test_pipewire_signal_exit_one_is_expected(self):
        proc = MagicMock()
        proc.returncode = None
        proc.stdout, proc.stderr = asyncio.StreamReader(), asyncio.StreamReader()
        proc.stdout.feed_data(struct.pack("<h", 1000) * 800)
        proc.stderr.feed_eof()

        async def wait():
            proc.returncode = 1
            proc.stdout.feed_eof()
            return 1

        proc.wait = AsyncMock(side_effect=wait)
        capture = Capture(self.paths.runtime, "", 120, MagicMock())
        with patch("omawhisper.audio.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            await capture.start()
            await asyncio.sleep(0)
            await capture.stop()
            await capture.wait()
        capture.cleanup()
        self.assertTrue(capture._signalled)


class FakeCapture:
    def __init__(self, directory, device, maximum, meter):
        self.cancelled = False
        self.stopped = asyncio.Event()
        self.cleaned = False
        self.path = directory / "test.wav"
        self.meter = meter

    async def start(self):
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self.path.write_bytes(b"audio")

    async def wait(self):
        await self.stopped.wait()
        return self.path

    async def stop(self, cancel=False):
        self.cancelled |= cancel
        self.stopped.set()

    def cleanup(self):
        self.path.unlink(missing_ok=True)
        self.cleaned = True


class DaemonTests(Files, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.output = types.SimpleNamespace(
            focused=AsyncMock(return_value={"address": "0x123", "class": "editor"}),
            deliver=AsyncMock(return_value="Pasted"), copy=AsyncMock(return_value="Copied"), close=AsyncMock(),
        )
        self.shortcuts = types.SimpleNamespace(apply=AsyncMock(), close=AsyncMock())
        self.daemon = Daemon(self.paths, output=self.output, shortcuts=self.shortcuts, capture_factory=FakeCapture)
        installed = self.paths.models / "whisper-base"
        installed.mkdir(parents=True)
        (installed / "model.bin").write_bytes(b"test weights")
        (installed / "config.json").write_text("{}")
        atomic_json(installed / ".omawhisper-complete", {})

    async def asyncTearDown(self):
        await self.daemon.close()

    async def test_exact_state_and_invalid_requests(self):
        self.assertEqual(set(self.daemon.snapshot()), {
            "phase", "level", "elapsed", "error", "message", "transcript", "model_name",
            "config", "runtime", "models", "devices", "history",
        })
        for command in ([], {}, {"action": "bad"}, {"action": "configure", "values": {"threads": False}},
                        {"action": "load", "id": "unknown"}, {"action": "start", "unknown": True}):
            with self.subTest(command=command):
                response = await self.daemon.request(command)
                self.assertEqual(set(response), {"error"})
                self.assertEqual(self.daemon.phase, "error")
                self.assertTrue(self.daemon.error)

    async def test_capture_starts_before_model_load_and_cancel_cleans(self):
        self.daemon.models.transcribe = MagicMock(return_value="hello")
        response = await self.daemon.request({"action": "start"})
        self.assertEqual(response["phase"], "recording")
        capture = self.daemon.capture
        self.daemon.models.transcribe.assert_not_called()
        await self.daemon.request({"action": "cancel"})
        await self.daemon.job
        self.assertTrue(capture.cleaned)
        self.assertFalse(capture.path.exists())
        self.daemon.models.transcribe.assert_not_called()
        self.assertEqual(self.daemon.phase, "idle")

    async def test_missing_model_rejected_before_opening_microphone(self):
        self.daemon.models.remove("whisper-base")
        self.daemon.capture_factory = MagicMock()
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            response = await self.daemon.request({"action": "start"})
        self.assertIn("not installed", response["error"])
        self.assertEqual(self.daemon.phase, "error")
        self.daemon.capture_factory.assert_not_called()
        self.output.focused.assert_not_awaited()
        self.assertIsNone(self.daemon.capture)
        self.assertIsNone(self.daemon.job)

    async def test_stop_transcribes_and_history_is_opt_in(self):
        self.daemon.models.transcribe = MagicMock(return_value="hello")
        await self.daemon.request({"action": "start"})
        capture = self.daemon.capture
        await self.daemon.request({"action": "stop"})
        await self.daemon.job
        self.assertEqual(self.daemon.transcript, "hello")
        self.assertTrue(capture.cleaned)
        self.assertEqual(self.daemon.history, [])
        self.assertFalse((self.paths.state / "history.json").exists())
        self.output.deliver.assert_awaited_once()

    async def test_impossible_model_rejected_before_microphone_opens(self):
        self.daemon.config["model"] = "parakeet-v3-fp32"
        self.daemon.capture_factory = MagicMock()
        with (patch.object(self.daemon.models, "installed", return_value=True),
              patch("omawhisper.models.memory_status", return_value={"total": 4 * 1024 ** 3, "available": 2 * 1024 ** 3}),
              self.assertLogs("omawhisper.daemon", level="ERROR")):
            response = await self.daemon.request({"action": "start"})
        self.assertIn("estimated 5.00 GiB", response["error"])
        self.daemon.capture_factory.assert_not_called()
        self.output.focused.assert_not_awaited()
        self.assertIsNone(self.daemon.job)

    async def test_push_to_talk_records_until_release_then_pastes(self):
        self.assertEqual(self.daemon.config["activation"], "hold")
        self.shortcuts.held = AsyncMock(return_value=True)
        self.daemon.models.transcribe = MagicMock(return_value="spoken words")
        response = await self.daemon.request({"action": "start", "hotkey": True})
        await asyncio.sleep(0)
        self.assertEqual(response["phase"], "recording")
        self.daemon.models.transcribe.assert_not_called()
        self.output.deliver.assert_not_awaited()
        self.shortcuts.held.return_value = False
        await self.daemon.request({"action": "stop"})
        await self.daemon.job
        self.output.deliver.assert_awaited_once_with(
            "spoken words", {"address": "0x123", "class": "editor"}, self.daemon.config,
        )
        self.assertEqual(self.daemon.phase, "idle")

    async def test_history_persists_and_clears(self):
        self.daemon.models.transcribe = MagicMock(return_value="hello")
        await self.daemon.request({"action": "configure", "values": {"history": True}})
        await self.daemon.request({"action": "start"})
        await self.daemon.request({"action": "stop"})
        await self.daemon.job
        self.assertEqual(len(self.daemon.history), 1)
        self.assertEqual((self.paths.state / "history.json").stat().st_mode & 0o777, 0o600)
        await self.daemon.request({"action": "clear_history"})
        self.assertEqual(self.daemon.history, [])
        self.assertFalse((self.paths.state / "history.json").exists())

    async def test_disabling_history_preserves_saved_entries_and_reenabling_reloads(self):
        self.daemon.models.transcribe = MagicMock(return_value="saved transcript")
        await self.daemon.request({"action": "configure", "values": {"history": True}})
        await self.daemon.request({"action": "start"})
        await self.daemon.request({"action": "stop"})
        await self.daemon.job
        history_path = self.paths.state / "history.json"
        saved = history_path.read_bytes()
        original = list(self.daemon.history)
        await self.daemon.request({"action": "configure", "values": {"history": False}})
        self.assertEqual(self.daemon.history, [])
        self.assertEqual(history_path.read_bytes(), saved)
        self.daemon.models.transcribe.return_value = "not saved"
        await self.daemon.request({"action": "start"})
        await self.daemon.request({"action": "stop"})
        await self.daemon.job
        self.assertEqual(self.daemon.history, [])
        self.assertEqual(history_path.read_bytes(), saved)
        await self.daemon.request({"action": "configure", "values": {"history": True}})
        self.assertEqual(self.daemon.history, original)
        self.assertEqual(history_path.read_bytes(), saved)
        await self.daemon.request({"action": "clear_history"})
        self.assertFalse(history_path.exists())

    async def test_failed_shortcut_update_does_not_save(self):
        self.shortcuts.apply.side_effect = UserError("collision")
        response = await self.daemon.request({"action": "configure", "values": {"shortcut": "SUPER+CTRL+B"}})
        self.assertEqual(response, {"error": "collision"})
        self.assertEqual(self.daemon.config["shortcut"], DEFAULTS["shortcut"])
        self.assertFalse((self.paths.config / "config.json").exists())

    async def test_corrected_setting_clears_previous_validation_error(self):
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            result = await self.daemon.request({"action": "configure", "values": {"threads": False}})
        self.assertIn("error", result)
        result = await self.daemon.request({"action": "configure", "values": {"threads": 4}})
        self.assertEqual(result["phase"], "idle")
        self.assertEqual(result["error"], "")

    async def test_rejected_request_keeps_recording_indicator_active(self):
        self.daemon.models.transcribe = MagicMock(return_value="hello")
        await self.daemon.request({"action": "start"})
        self.daemon._meter(0.3, 1.0)
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            result = await self.daemon.request({"action": "load", "id": "whisper-base"})
        self.assertIn("error", result)
        self.assertEqual(self.daemon.phase, "recording")
        self.assertEqual(self.daemon.level, 0.3)
        await self.daemon.request({"action": "stop"})
        await self.daemon.job
        self.assertEqual(self.daemon.phase, "idle")

    async def test_model_change_unloads(self):
        self.daemon.models.loaded_id = "whisper-base"
        await self.daemon.request({"action": "configure", "values": {"model": "whisper-tiny"}})
        self.assertIsNone(self.daemon.models.loaded_id)

    async def test_acceleration_change_unloads_and_persists_without_changing_microphone(self):
        self.daemon.models.loaded_id = "whisper-base"
        self.daemon.models.runtime = {"device": "cpu"}
        self.daemon.config["device"] = "alsa_input.usb"
        response = await self.daemon.request({"action": "configure", "values": {"acceleration": "gpu"}})
        self.assertIsNone(self.daemon.models.loaded_id)
        self.assertEqual(response["runtime"], {})
        self.assertEqual(response["config"]["device"], "alsa_input.usb")
        saved = json.loads((self.paths.config / "config.json").read_text())
        self.assertEqual(saved["acceleration"], "gpu")

    async def test_acceleration_change_is_rejected_while_recording(self):
        await self.daemon.request({"action": "start"})
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            response = await self.daemon.request({"action": "configure", "values": {"acceleration": "gpu"}})
        self.assertIn("busy", response["error"])
        self.assertEqual(self.daemon.phase, "recording")
        self.assertEqual(self.daemon.config["acceleration"], "auto")
        await self.daemon.request({"action": "cancel"})
        await self.daemon.job

    def add_command_model(self):
        self.daemon.models.add({
            "id": "adapter", "name": "Adapter", "engine": "command", "source": "",
            "argv": [sys.executable, "-c", "print('test')", "{audio}"],
        })

    async def test_successful_load_selects_persists_and_next_recording_uses_model(self):
        self.add_command_model()
        await self.daemon.request({"action": "load", "id": "adapter"})
        await self.daemon.job
        self.assertEqual(self.daemon.config["model"], "adapter")
        self.assertEqual(self.daemon.models.loaded_id, "adapter")
        self.assertEqual(json.loads((self.paths.config / "config.json").read_text())["model"], "adapter")
        self.daemon.models.transcribe = MagicMock(return_value="using selected model")
        state = await self.daemon.request({"action": "start"})
        self.assertEqual(state["config"]["model"], "adapter")
        await self.daemon.request({"action": "stop"})
        await self.daemon.job
        args = self.daemon.models.transcribe.call_args.args
        self.assertEqual(args[1], "adapter")
        self.assertEqual(args[2]["model"], "adapter")

    async def test_load_failure_does_not_change_selection(self):
        self.paths.save_config(self.daemon.config)
        self.daemon.models.load = MagicMock(side_effect=UserError("cannot load"))
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            await self.daemon.request({"action": "load", "id": "parakeet-v3"})
            await self.daemon.job
        self.assertEqual(self.daemon.config["model"], "whisper-base")
        self.assertEqual(json.loads((self.paths.config / "config.json").read_text())["model"], "whisper-base")
        self.assertEqual(self.daemon.phase, "error")

    async def test_load_selection_save_failure_unloads_new_model_and_keeps_old_config(self):
        self.add_command_model()
        self.paths.save_config(self.daemon.config)
        before = (self.paths.config / "config.json").read_bytes()
        self.paths.save_config = MagicMock(side_effect=OSError("configuration is read-only"))
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            await self.daemon.request({"action": "load", "id": "adapter"})
            await self.daemon.job
        self.assertEqual(self.daemon.config["model"], "whisper-base")
        self.assertEqual((self.paths.config / "config.json").read_bytes(), before)
        self.assertIsNone(self.daemon.models.loaded_id)
        self.assertIsNone(self.daemon.models.instance)
        self.assertEqual(self.daemon.phase, "error")

    async def test_cancel_after_loading_does_not_persist_selection(self):
        self.add_command_model()
        loaded, release = threading.Event(), threading.Event()
        load = self.daemon.models.load

        def delayed_load(*args):
            load(*args)
            loaded.set()
            release.wait(3)

        self.daemon.models.load = delayed_load
        await self.daemon.request({"action": "load", "id": "adapter"})
        while not loaded.is_set():
            await asyncio.sleep(0.001)
        try:
            self.assertEqual(self.daemon.models.loaded_id, "adapter")
            await self.daemon.request({"action": "cancel"})
        finally:
            release.set()
        await self.daemon.job
        self.assertEqual(self.daemon.config["model"], "whisper-base")
        self.assertFalse((self.paths.config / "config.json").exists())
        self.assertIsNone(self.daemon.models.loaded_id)
        self.assertEqual(self.daemon.phase, "idle")

    async def test_load_selection_preserves_concurrent_settings_changes(self):
        self.add_command_model()
        loading, release = threading.Event(), threading.Event()
        load = self.daemon.models.load

        def delayed_load(*args):
            loading.set()
            release.wait(3)
            load(*args)

        self.daemon.models.load = delayed_load
        await self.daemon.request({"action": "load", "id": "adapter"})
        while not loading.is_set():
            await asyncio.sleep(0.001)
        try:
            await self.daemon.request({"action": "configure", "values": {"append_space": False}})
        finally:
            release.set()
        await self.daemon.job
        self.assertEqual(self.daemon.config["model"], "adapter")
        self.assertFalse(self.daemon.config["append_space"])
        saved = json.loads((self.paths.config / "config.json").read_text())
        self.assertEqual(saved["model"], "adapter")
        self.assertFalse(saved["append_space"])

    async def test_selected_custom_removal_saves_fallback_before_unregistering(self):
        self.add_command_model()
        await self.daemon.request({"action": "load", "id": "adapter"})
        await self.daemon.job
        remove = self.daemon.models.remove

        def checked_remove(identifier):
            self.assertEqual(self.daemon.config["model"], "whisper-base")
            self.assertEqual(json.loads((self.paths.config / "config.json").read_text())["model"], "whisper-base")
            self.assertIn(identifier, self.daemon.models.entries)
            remove(identifier)

        self.daemon.models.remove = checked_remove
        await self.daemon.request({"action": "remove_model", "id": "adapter"})
        await self.daemon.job
        self.assertNotIn("adapter", self.daemon.models.entries)
        self.assertNotIn("adapter", Models(self.paths).entries)
        self.assertEqual(self.daemon.snapshot()["config"]["model"], "whisper-base")
        self.assertTrue(Path(sys.executable).is_file())

    async def test_removal_selection_save_failure_does_not_mutate_registry_or_model(self):
        self.add_command_model()
        await self.daemon.request({"action": "load", "id": "adapter"})
        await self.daemon.job
        before = (self.paths.config / "models.json").read_bytes()
        self.paths.save_config = MagicMock(side_effect=OSError("configuration is read-only"))
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            await self.daemon.request({"action": "remove_model", "id": "adapter"})
            await self.daemon.job
        self.assertEqual(self.daemon.config["model"], "adapter")
        self.assertEqual(self.daemon.models.loaded_id, "adapter")
        self.assertEqual((self.paths.config / "models.json").read_bytes(), before)
        self.assertIn("adapter", Models(self.paths).entries)

    async def test_removing_unselected_custom_model_keeps_current_selection(self):
        self.add_command_model()
        self.paths.save_config = MagicMock()
        await self.daemon.request({"action": "remove_model", "id": "adapter"})
        await self.daemon.job
        self.assertEqual(self.daemon.config["model"], "whisper-base")
        self.paths.save_config.assert_not_called()
        self.assertNotIn("adapter", self.daemon.models.entries)

    async def test_selected_builtin_removal_resets_selection_but_keeps_catalog_entry(self):
        await self.daemon.request({"action": "configure", "values": {"model": "whisper-tiny"}})
        directory = self.paths.models / "whisper-tiny"
        directory.mkdir(parents=True)
        (directory / "model.bin").write_bytes(b"downloaded weights")
        await self.daemon.request({"action": "remove_model", "id": "whisper-tiny"})
        await self.daemon.job
        self.assertEqual(self.daemon.config["model"], "whisper-base")
        self.assertEqual(json.loads((self.paths.config / "config.json").read_text())["model"], "whisper-base")
        self.assertIn("whisper-tiny", self.daemon.models.entries)
        self.assertFalse(directory.exists())

    async def test_quick_hold_release_cannot_start_late_recording(self):
        self.shortcuts.held = AsyncMock(return_value=False)
        state = await self.daemon.request({"action": "start", "hotkey": True})
        self.assertEqual(state["phase"], "idle")
        self.assertIsNone(self.daemon.capture)

    async def test_cli_request_failure_exits_nonzero(self):
        self.paths.runtime = self.root / "omawhisper"
        self.paths.socket = self.paths.runtime / "control.sock"
        await self.daemon.start(shortcuts=False)
        environment = dict(os.environ, XDG_RUNTIME_DIR=str(self.root.resolve()),
                           PYTHONPATH=str(Path("backend").resolve()))
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "omawhisper.cli", "request", '{"action":"bad"}',
            env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        self.assertEqual(process.returncode, 1)
        self.assertIn("Unknown action", json.loads(stdout)["error"])

    async def test_worker_does_not_block_status_and_cancel_suppresses_output(self):
        started, release = threading.Event(), threading.Event()

        def transcribe(*args):
            started.set()
            release.wait(3)
            return "must not be pasted"

        self.daemon.models.transcribe = transcribe
        await self.daemon.request({"action": "start"})
        await self.daemon.request({"action": "stop"})
        while not started.is_set():
            await asyncio.sleep(0.001)
        try:
            state = await asyncio.wait_for(self.daemon.request({"action": "status"}), 0.2)
            self.assertEqual(state["phase"], "transcribing")
            await self.daemon.request({"action": "cancel"})
        finally:
            release.set()
        await self.daemon.job
        self.output.deliver.assert_not_awaited()
        self.assertEqual(self.daemon.transcript, "")

    async def test_socket_errors_single_instance_and_cleanup(self):
        await self.daemon.start(shortcuts=False)
        self.assertEqual(self.paths.socket.stat().st_mode & 0o777, 0o600)
        reader, writer = await asyncio.open_unix_connection(self.paths.socket)
        writer.write(b'{"action":"bad"}\n')
        await writer.drain()
        response = json.loads(await reader.readline())
        self.assertEqual(set(response), {"error"})
        writer.close()
        await writer.wait_closed()
        other = Daemon(self.paths, output=self.output, shortcuts=self.shortcuts)
        with self.assertRaises(UserError):
            await other.start(shortcuts=False)
        await other.close()
        self.assertTrue(self.paths.socket.exists())

    async def test_watch_coalesces_and_disconnects(self):
        await self.daemon.start(shortcuts=False)
        reader, writer = await asyncio.open_unix_connection(self.paths.socket)
        writer.write(b'{"action":"watch"}\n')
        await writer.drain()
        state = json.loads(await reader.readline())
        self.assertEqual(state["phase"], "idle")
        for n in range(100):
            self.daemon._meter(0.5, n)
        self.assertTrue(all(queue.qsize() <= 1 for queue in self.daemon.subscribers))
        update = json.loads(await reader.readline())
        self.assertEqual(update["elapsed"], 99)
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.03)
        self.assertEqual(len(self.daemon.subscribers), 0)


if __name__ == "__main__":
    unittest.main()
