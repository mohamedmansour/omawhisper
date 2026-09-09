import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from omawhisper.config import UserError, validate
from omawhisper.memory import GIB, MIB, memory_status
from omawhisper.models import Models
from test_backend import Files


class MemoryTests(Files, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.models = Models(self.paths)
        self.memory = {"total": 4 * GIB, "available": 2 * GIB}
        mock_memory = patch("omawhisper.models.memory_status", side_effect=lambda: dict(self.memory))
        mock_memory.start()
        self.addCleanup(mock_memory.stop)

    def test_memory_reader_uses_available_ram_not_swap_or_free_pages(self):
        contents = "MemTotal: 4194304 kB\nMemFree: 100 kB\nMemAvailable: 2097152 kB\nSwapFree: 8388608 kB\n"
        with patch("omawhisper.memory.Path.read_text", return_value=contents):
            self.assertEqual(memory_status(), self.memory)
        for contents in ("MemTotal: bad kB", "MemTotal: 4194304 kB"):
            with patch("omawhisper.memory.Path.read_text", return_value=contents):
                with self.assertRaisesRegex(UserError, "Cannot check system RAM"):
                    memory_status()

    def test_catalog_reports_impossible_fp32_without_disabling_int8(self):
        rows = {row["id"]: row for row in self.models.snapshot()}
        self.assertEqual(rows["parakeet-v3-fp32"]["estimated_ram_bytes"], 5 * GIB)
        self.assertIn("5.00 GiB", rows["parakeet-v3-fp32"]["memory_error"])
        self.assertIn("4.00 GiB", rows["parakeet-v3-fp32"]["memory_error"])
        self.assertEqual(rows["parakeet-v3"]["memory_error"], "")
        self.models.check_memory("parakeet-v3", available=True)
        self.memory["total"] = 8 * GIB
        self.models.check_memory("parakeet-v3-fp32")

    def test_impossible_direct_load_keeps_previous_worker_and_never_spawns_native_engine(self):
        with (patch.object(self.models, "installed", return_value=True),
              patch.object(self.models, "unload") as unload,
              patch("omawhisper.models.EngineWorker") as worker):
            with self.assertRaisesRegex(UserError, "estimated 5.00 GiB"):
                self.models.load("parakeet-v3-fp32", validate({}))
        unload.assert_not_called()
        worker.assert_not_called()

    def test_available_memory_is_checked_after_releasing_previous_model(self):
        self.memory["total"] = 8 * GIB

        def unload():
            self.memory["available"] = 6 * GIB

        class Worker:
            alive = True
            request = MagicMock(return_value={"device": "cpu"})

            def __init__(self, *args):
                pass

        with (patch.object(self.models, "installed", return_value=True),
              patch.object(self.models, "unload", side_effect=unload),
              patch("omawhisper.models.EngineWorker", Worker)):
            self.models.load("parakeet-v3-fp32", validate({}))
        self.assertEqual(self.models.loaded_id, "parakeet-v3-fp32")

    def test_insufficient_available_ram_stops_before_worker_allocation(self):
        self.memory["total"] = 8 * GIB
        with (patch.object(self.models, "installed", return_value=True),
              patch.object(self.models, "unload") as unload,
              patch("omawhisper.models.EngineWorker") as worker):
            with self.assertRaisesRegex(UserError, "2.00 GiB is currently available"):
                self.models.load("parakeet-v3-fp32", validate({}))
        unload.assert_called_once()
        worker.assert_not_called()

    def test_custom_onnx_estimates_only_selected_weights_and_external_data(self):
        directory = self.root / "weights"
        directory.mkdir()
        for name, size in (
            ("encoder.int8.onnx", 16), ("encoder.int8.onnx.data", 32), ("shared.onnx", 8),
            ("encoder.onnx", 64), ("encoder.onnx.data", 128), ("config.json", 256),
        ):
            (directory / name).write_bytes(b" " * size)
        self.models.add({"id": "custom", "name": "Custom", "engine": "onnx-asr", "source": str(directory.resolve())})
        required = 2 * (16 + 32 + 8) + 256 * MIB
        self.assertEqual(self.models.memory_requirement(self.models.get("custom")), required)
        self.assertEqual(self.models.snapshot()[-1]["estimated_ram_bytes"], required)
        original_stat = Path.stat

        def removing_file(path, *args, **kwargs):
            if path.name == "shared.onnx":
                raise FileNotFoundError(path)
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", removing_file):
            self.assertEqual(self.models.snapshot()[-1]["estimated_ram_bytes"], required - 16)
        self.memory["total"] = required - 1
        self.assertTrue(self.models.snapshot()[-1]["memory_error"])

    def test_installed_service_keeps_daemon_alive_when_a_worker_is_oom_killed(self):
        spec = importlib.util.spec_from_file_location("desktop_installer", "scripts/install.py")
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        with (patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.paths.config)}),
              patch.object(installer, "run"), patch("builtins.print")):
            installer.install("right")
        unit = (self.paths.config / "systemd/user/omawhisper.service").read_text()
        self.assertIn("\nOOMPolicy=continue\n", unit)
