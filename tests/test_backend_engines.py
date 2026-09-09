import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

from omawhisper.acceleration import cuda_available, onnx_gpu_compatible
from omawhisper.config import UserError, validate
from omawhisper.engine_worker import EngineWorker, LIMIT, WorkerCancelled, engine_python, worker_environment
from omawhisper.engines import CPU, CUDA, OnnxEngine, onnx_device, verify_cuda_execution, verify_onnx_sessions
from omawhisper.models import Models
from test_backend import Files


class OnnxDeviceTests(unittest.TestCase):
    def setUp(self):
        self.runtime = types.SimpleNamespace(get_available_providers=lambda: [CUDA, CPU])

    def test_gpu_supports_floating_models_and_not_cpu_integer_quantization(self):
        for quantization in (None, "", "fp16", "float16", "float32", "bf16"):
            self.assertTrue(onnx_gpu_compatible(quantization))
        for quantization in ("int8", "int4", "q4"):
            self.assertFalse(onnx_gpu_compatible(quantization))

    def test_auto_and_explicit_gpu_use_cuda_when_usable(self):
        with patch("omawhisper.engines.cuda_available", return_value=(True, "ready")):
            for acceleration in ("auto", "gpu"):
                self.assertEqual(onnx_device(self.runtime, validate({"acceleration": acceleration}), None)[0], CUDA)

    def test_explicit_cpu_and_auto_int8_never_probe_gpu(self):
        with patch("omawhisper.engines.cuda_available") as probe:
            self.assertEqual(onnx_device(self.runtime, validate({"acceleration": "cpu"}), None)[0], CPU)
            self.assertEqual(onnx_device(self.runtime, validate({}), "int8")[0], CPU)
            probe.assert_not_called()

    def test_explicit_gpu_rejects_cpu_weights(self):
        with self.assertRaisesRegex(UserError, "FP32"):
            onnx_device(self.runtime, validate({"acceleration": "gpu"}), "int8")

    def test_missing_provider_and_driver_do_not_claim_gpu(self):
        for providers, driver in (([CPU], (True, "ready")), ([CUDA, CPU], (False, "driver missing"))):
            self.runtime.get_available_providers = lambda: providers
            with patch("omawhisper.engines.cuda_available", return_value=driver):
                self.assertEqual(onnx_device(self.runtime, validate({}), None)[0], CPU)
                with self.assertRaises(UserError):
                    onnx_device(self.runtime, validate({"acceleration": "gpu"}), None)

    def test_cuda_driver_errors_are_reported_and_hidden_devices_respected(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}), patch("ctypes.CDLL") as library:
            self.assertFalse(cuda_available()[0])
            library.assert_not_called()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with patch("ctypes.CDLL", side_effect=OSError("missing")):
                self.assertIn("driver", cuda_available()[1])
            driver = MagicMock()
            driver.cuInit.return_value = 35
            with patch("ctypes.CDLL", return_value=driver):
                self.assertIn("35", cuda_available()[1])
                driver.cuInit.return_value = 0
                driver.cuDriverGetVersion.return_value = 0
                self.assertIn("CUDA 12", cuda_available()[1])

    def test_silently_dropped_cuda_provider_is_rejected(self):
        class Session:
            def get_providers(self):
                return [CPU]

            def disable_fallback(self):
                raise AssertionError("Not reached for an invalid provider")

        runtime = types.SimpleNamespace(InferenceSession=Session)
        adapter = types.SimpleNamespace(asr=types.SimpleNamespace(encoder=Session()))
        with self.assertRaisesRegex(UserError, "refusing to claim GPU"):
            verify_onnx_sessions(adapter, runtime, CUDA)

    def test_active_cuda_sessions_disable_inference_fallback(self):
        class Session:
            disable_fallback = MagicMock()

            def get_providers(self):
                return [CUDA, CPU]

        runtime = types.SimpleNamespace(InferenceSession=Session)
        encoder, decoder = Session(), Session()
        adapter = types.SimpleNamespace(asr=types.SimpleNamespace(encoder=encoder, decoder=decoder, config={}))
        self.assertEqual(verify_onnx_sessions(adapter, runtime, CUDA), [encoder, decoder])
        self.assertEqual(Session.disable_fallback.call_count, 2)

    def test_cuda_warmup_profiles_execution_not_just_provider_registration(self):
        numpy = types.SimpleNamespace(zeros=lambda *args, **kwargs: "synthetic silence", float32="float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"numpy": numpy}):
            profile = Path(directory) / "profile.json"
            session = MagicMock()
            session.end_profiling.return_value = str(profile)
            model = MagicMock()
            for provider, succeeds in ((CPU, False), (CUDA, True)):
                profile.write_text(json.dumps([{"cat": "Node", "args": {"provider": provider}}]))
                if succeeds:
                    verify_cuda_execution([session], model)
                else:
                    with self.assertRaisesRegex(UserError, "did not execute on CUDA"):
                        verify_cuda_execution([session], model)
            model.recognize.assert_called_with("synthetic silence", sample_rate=16000)

    def test_onnx_gpu_load_routes_only_asr_to_cuda_and_verifies_warmup(self):
        runtime = types.ModuleType("onnxruntime")
        runtime.get_available_providers = lambda: [CUDA, CPU]
        runtime.SessionOptions = MagicMock()
        asr = types.ModuleType("onnx_asr")
        asr.load_model = MagicMock()
        with (patch.dict(sys.modules, {"onnxruntime": runtime, "onnx_asr": asr}),
              patch("omawhisper.engines.cuda_available", return_value=(True, "ready")),
              patch("omawhisper.engines.verify_onnx_sessions", return_value=["encoder", "decoder"]) as sessions,
              patch("omawhisper.engines.verify_cuda_execution") as warmup):
            result = OnnxEngine().load(
                {"source": "owner/model", "architecture": "nemo-conformer-tdt", "quantization": ""},
                "/local/model", validate({"acceleration": "gpu"}))
        options = asr.load_model.call_args.kwargs
        self.assertEqual(options["providers"], [CPU])
        self.assertEqual(options["asr_config"]["providers"], [CUDA, CPU])
        self.assertIsNone(options["quantization"])
        self.assertEqual(result["device"], "cuda")
        sessions.assert_called_once_with(asr.load_model.return_value, runtime, CUDA)
        warmup.assert_called_once_with(["encoder", "decoder"], asr.load_model.return_value)


FAKE_WORKER = """
import json, os, signal, socket, sys, time
channel = socket.socket(fileno=int(sys.argv[1]))
mode = sys.argv[2]
with channel.makefile('rb') as reader:
    for line in reader:
        request = json.loads(line)
        if mode == 'hang':
            time.sleep(60)
        if mode == 'exit':
            sys.exit(3)
        if mode == 'kill':
            os.kill(os.getpid(), signal.SIGKILL)
        if mode == 'oversized':
            channel.sendall(b'x' * (2 * 1024 * 1024 + 1))
            continue
        response = {'id': request['id'], 'result': request['action']}
        if mode == 'error':
            response = {'id': request['id'], 'error': 'CUDA failed to initialize'}
        if mode == 'wrong-id':
            response['id'] += 1
        channel.sendall(json.dumps(response).encode() + b'\\n')
"""


class WorkerTests(Files, unittest.TestCase):
    def worker(self, mode="normal"):
        popen = subprocess.Popen

        def spawn(arguments, **kwargs):
            descriptor = arguments[arguments.index("--fd") + 1]
            return popen([sys.executable, "-c", FAKE_WORKER, descriptor, mode], **kwargs)

        with patch("omawhisper.engine_worker.subprocess.Popen", side_effect=spawn):
            worker = EngineWorker(Path(sys.executable), "onnx-asr")
        self.addCleanup(worker.close)
        return worker

    def test_persistent_private_channel_and_close_reap_process(self):
        worker = self.worker()
        process = worker.process
        self.assertEqual(worker.request("load"), "load")
        self.assertEqual(worker.request("transcribe"), "transcribe")
        self.assertTrue(worker.alive)
        self.assertIs(worker.process, process)
        worker.close()
        self.assertIsNotNone(process.poll())
        self.assertFalse(worker.alive)
        worker.close()

    def test_cancellation_terminates_native_worker(self):
        worker = self.worker("hang")
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(WorkerCancelled):
            worker.request("load", cancelled=cancelled)
        self.assertFalse(worker.alive)

    def test_sigkill_reports_possible_memory_exhaustion_and_reaps_worker(self):
        worker = self.worker("kill")
        process = worker.process
        with self.assertRaisesRegex(UserError, "SIGKILL, possibly because it ran out of RAM"):
            worker.request("load")
        self.assertIsNotNone(process.poll())
        self.assertFalse(worker.alive)
        self.assertIsNone(worker.channel)

    def test_native_worker_is_first_oom_candidate_without_changing_parent_priority(self):
        parent_score = Path("/proc/self/oom_score_adj").read_text()
        worker = EngineWorker(Path(sys.executable), "onnx-asr")
        self.addCleanup(worker.close)
        score = Path(f"/proc/{worker.process.pid}/oom_score_adj")
        for _ in range(200):
            if score.read_text().strip() == "1000":
                break
            time.sleep(0.01)
        self.assertEqual(score.read_text().strip(), "1000")
        self.assertEqual(Path("/proc/self/oom_score_adj").read_text(), parent_score)

    def test_timeout_and_invalid_responses_stop_the_worker(self):
        for mode, error in (("hang", "timed out"), ("exit", "disconnected"),
                            ("error", "CUDA failed"), ("wrong-id", "Invalid response"),
                            ("oversized", "safety limit")):
            with self.subTest(mode=mode):
                worker = self.worker(mode)
                with self.assertRaisesRegex(UserError, error):
                    worker.request("load", timeout=0.05 if mode == "hang" else 3)
                self.assertFalse(worker.alive)

    def test_worker_uses_own_python_packages_and_cuda_libraries(self):
        python = self.paths.data / "venv-onnx-cuda/bin/python"
        libraries = python.parent.parent / "lib/python3.14/site-packages/nvidia/cudnn/lib"
        libraries.mkdir(parents=True)
        with patch.dict(os.environ, {"PYTHONPATH": "/wrong/runtime", "PYTHONHOME": "/wrong", "LD_LIBRARY_PATH": "/system/cuda"}):
            environment = worker_environment(python)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("/wrong", environment["PYTHONPATH"])
        self.assertEqual(environment["LD_LIBRARY_PATH"], f"{libraries}:/system/cuda")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")

    def test_runtime_choice_preserves_environment_isolation(self):
        cuda = self.paths.data / "venv-onnx-cuda/bin/python"
        cpu = self.paths.data / "venv-onnx-cpu/bin/python"
        for python in (cuda, cpu):
            python.parent.mkdir(parents=True)
            python.touch()
        self.assertEqual(engine_python(self.paths, "onnx-asr", "gpu"), cuda)
        self.assertEqual(engine_python(self.paths, "onnx-asr", "auto"), cuda)
        self.assertEqual(engine_python(self.paths, "onnx-asr", "cpu"), cpu)
        self.assertEqual(engine_python(self.paths, "onnx-asr", "auto", False), cpu)
        cuda.unlink()
        with self.assertRaisesRegex(UserError, "Models > Processing device"):
            engine_python(self.paths, "onnx-asr", "gpu")

    def test_model_cache_and_unload_manage_worker_lifetime(self):
        workers = []

        class FakeWorker:
            def __init__(self, *args):
                self.alive = True
                self.request = MagicMock(return_value={"device": "cpu", "compute_type": "int8"})
                workers.append(self)

            def close(self):
                self.alive = False

        with patch("omawhisper.models.EngineWorker", FakeWorker):
            models = Models(self.paths)
            with patch.object(models, "installed", return_value=True):
                models.load("whisper-base", validate({}))
                models.load("whisper-base", validate({}))
                self.assertEqual(len(workers), 1)
                models.load("whisper-base", validate({"acceleration": "cpu"}))
                self.assertEqual(len(workers), 2)
                self.assertFalse(workers[0].alive)
                workers[1].alive = False
                self.assertEqual(models.runtime_snapshot(), {})
                self.assertFalse(any(row["loaded"] for row in models.snapshot()))
                models.load("whisper-base", validate({"acceleration": "cpu"}))
                self.assertEqual(len(workers), 3)
                models.unload()
                self.assertFalse(workers[2].alive)
                self.assertEqual(models.runtime, {})
