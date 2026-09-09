import asyncio
import json
import sys
import threading
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from omawhisper.config import UserError
from omawhisper.daemon import Daemon
from omawhisper.engine_worker import WorkerCancelled
from omawhisper.memory import GIB
from omawhisper.setup import run_installer
from test_backend import Files


class SetupTests(Files, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.shortcuts = types.SimpleNamespace(apply=AsyncMock(), close=AsyncMock(), events=AsyncMock())
        self.output = types.SimpleNamespace(close=AsyncMock())
        self.daemon = Daemon(self.paths, shortcuts=self.shortcuts, output=self.output)
        self.values = {"model": "whisper-base", "acceleration": "cpu", "activation": "hold", "shortcut": "SUPER+ALT+V"}
        self.installer = AsyncMock()
        self.patch_installer = patch("omawhisper.daemon.install_runtimes", self.installer)
        self.patch_installer.start()
        self.addCleanup(self.patch_installer.stop)
        self.memory = {"total": 16 * GIB, "available": 8 * GIB}
        mock_memory = patch("omawhisper.models.memory_status", side_effect=lambda: dict(self.memory))
        mock_memory.start()
        self.addCleanup(mock_memory.stop)
        self.daemon.models.installed = MagicMock(return_value=True)

        def load(identifier, config, cancelled=None):
            self.daemon.models.loaded_id = identifier
            self.daemon.models.runtime = {"device": "cpu"}

        self.daemon.models.load = MagicMock(side_effect=load)
        self.daemon.models.download = MagicMock()

    async def asyncTearDown(self):
        await self.daemon.close()

    async def setup_request(self):
        response = await self.daemon.request({"action": "setup", "values": self.values})
        self.assertNotIn("error", {k: v for k, v in response.items() if v and k == "error"})
        await self.daemon.job

    async def test_success_installs_downloads_loads_and_commits_settings(self):
        self.values = {"model": "whisper-tiny", "acceleration": "cpu"}
        self.daemon.models.installed.return_value = False
        calls = MagicMock()
        calls.attach_mock(self.installer, "install")
        calls.attach_mock(self.daemon.models.download, "download")
        calls.attach_mock(self.daemon.models.load, "load")
        await self.setup_request()
        self.installer.assert_awaited_once()
        self.assertFalse(self.installer.call_args.args[0])
        self.daemon.models.download.assert_called_once()
        self.daemon.models.load.assert_called_once()
        saved = json.loads((self.paths.config / "config.json").read_text())
        self.assertTrue(saved["setup_complete"])
        self.assertEqual(saved["acceleration"], "cpu")
        self.assertEqual(saved["activation"], "hold")
        self.assertEqual(saved["model"], "whisper-tiny")
        self.assertEqual([call[0] for call in calls.mock_calls], ["install", "download", "load"])
        self.assertEqual(self.daemon.phase, "idle")

    async def test_failed_first_download_does_not_activate_the_new_model(self):
        self.values = {"model": "whisper-tiny", "acceleration": "cpu"}
        self.daemon.models.installed.return_value = False
        self.daemon.models.download.side_effect = UserError("Download failed")
        before = dict(self.daemon.config)
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            await self.setup_request()
        self.assertEqual(self.daemon.config, before)
        self.daemon.models.load.assert_not_called()
        self.assertFalse((self.paths.config / "config.json").exists())
        self.assertIn("Download failed", self.daemon.error)

    async def test_reconfiguration_preserves_non_setup_preferences_and_adapts_precision(self):
        self.daemon.config.update(setup_complete=True, append_space=False, history=True, compute_type="float16")
        self.values["activation"] = "toggle"
        await self.setup_request()
        self.assertFalse(self.daemon.config["append_space"])
        self.assertTrue(self.daemon.config["history"])
        self.assertEqual(self.daemon.config["compute_type"], "auto")
        self.assertEqual(self.daemon.config["activation"], "toggle")

    async def test_general_activation_changes_apply_independently_and_model_setup_preserves_them(self):
        self.values = {"model": "whisper-base", "acceleration": "cpu"}
        self.daemon.models.loaded_id = "whisper-base"
        with patch.object(self.daemon.models, "unload") as unload:
            response = await self.daemon.request({
                "action": "configure", "values": {"shortcut": "CTRL+SHIFT+F8", "activation": "toggle"},
            })
        self.assertEqual(response["error"], "")
        self.shortcuts.apply.assert_awaited_once_with("CTRL+SHIFT+F8", "toggle")
        unload.assert_not_called()
        self.installer.assert_not_awaited()
        self.daemon.models.load.assert_not_called()
        self.assertEqual(self.daemon.models.loaded_id, "whisper-base")
        await self.setup_request()
        saved = json.loads((self.paths.config / "config.json").read_text())
        self.assertEqual(saved["shortcut"], "CTRL+SHIFT+F8")
        self.assertEqual(saved["activation"], "toggle")
        self.assertTrue(saved["setup_complete"])
        self.assertEqual(self.daemon.phase, "idle")

    async def test_parakeet_floating_model_accepts_gpu_and_installs_cuda_runtime(self):
        self.values.update(model="parakeet-v3-fp32", acceleration="gpu")
        with patch("omawhisper.daemon.cuda_available", return_value=(True, "ready")):
            await self.setup_request()
        self.assertTrue(self.installer.call_args.args[0])
        self.assertEqual(self.daemon.config["model"], "parakeet-v3-fp32")

    async def test_gpu_without_driver_fails_before_downloading_and_keeps_old_settings(self):
        before = dict(self.daemon.config)
        self.values.update(model="parakeet-v3-fp32", acceleration="gpu")
        with (patch("omawhisper.daemon.cuda_available", return_value=(False, "NVIDIA driver missing")),
              self.assertLogs("omawhisper.daemon", level="ERROR")):
            await self.setup_request()
        self.installer.assert_not_awaited()
        self.daemon.models.download.assert_not_called()
        self.assertEqual(self.daemon.config, before)
        self.assertIn("NVIDIA driver missing", self.daemon.error)

    async def test_failure_keeps_configuration_and_can_be_retried(self):
        before = dict(self.daemon.config)
        self.installer.side_effect = UserError("package download failed")
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            await self.setup_request()
        self.assertEqual(self.daemon.config, before)
        self.assertFalse(self.daemon.setup_running)
        self.assertIsNone(self.daemon.models.loaded_id)
        self.installer.side_effect = None
        await self.setup_request()
        self.assertTrue(self.daemon.config["setup_complete"])

    async def test_small_machine_rejects_fp32_before_any_setup_side_effect(self):
        before = dict(self.daemon.config)
        self.memory.update(total=4 * GIB, available=2 * GIB)
        self.values["model"] = "parakeet-v3-fp32"
        with (patch.object(self.daemon.models, "unload") as unload,
              self.assertLogs("omawhisper.daemon", level="ERROR")):
            await self.setup_request()
        unload.assert_not_called()
        self.installer.assert_not_awaited()
        self.daemon.models.download.assert_not_called()
        self.daemon.models.load.assert_not_called()
        self.shortcuts.apply.assert_not_awaited()
        self.assertEqual(self.daemon.config, before)
        self.assertFalse((self.paths.config / "config.json").exists())
        self.assertIn("Parakeet INT8", self.daemon.error)
        self.assertEqual((await self.daemon.request({"action": "status"}))["phase"], "error")
        self.values["model"] = "parakeet-v3"
        await self.setup_request()
        self.assertEqual(self.daemon.config["model"], "parakeet-v3")
        self.assertEqual(self.daemon.phase, "idle")

    async def test_memory_pressure_rejects_setup_before_provisioning(self):
        before = dict(self.daemon.config)
        self.memory["available"] = GIB
        self.values["model"] = "parakeet-v3"
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            await self.setup_request()
        self.installer.assert_not_awaited()
        self.daemon.models.load.assert_not_called()
        self.assertEqual(self.daemon.config, before)
        self.assertIn("currently available", self.daemon.error)

    async def test_cancel_stops_setup_and_rejects_concurrent_configuration(self):
        started = asyncio.Event()

        async def installing(cuda, cancelled, report):
            started.set()
            while not cancelled.is_set():
                await asyncio.sleep(0.001)
            raise WorkerCancelled()

        self.installer.side_effect = installing
        before = dict(self.daemon.config)
        await self.daemon.request({"action": "setup", "values": self.values})
        await started.wait()
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            response = await self.daemon.request({"action": "configure", "values": {"append_space": False}})
        self.assertIn("busy", response["error"])
        await self.daemon.request({"action": "cancel"})
        await self.daemon.job
        self.assertEqual(self.daemon.config, before)
        self.assertEqual(self.daemon.phase, "idle")
        self.daemon.models.load.assert_not_called()

    async def test_failed_save_restores_previous_shortcut(self):
        self.values["shortcut"] = "SUPER+ALT+B"
        self.paths.save_config = MagicMock(side_effect=OSError("read only"))
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            await self.setup_request()
        self.assertFalse(self.daemon.config["setup_complete"])
        self.assertEqual(self.shortcuts.apply.call_args.args, ("SUPER+ALT+V", "hold"))
        self.assertIsNone(self.daemon.models.loaded_id)

    async def test_cancellation_during_shortcut_update_rolls_back_before_commit(self):
        applying, release = asyncio.Event(), asyncio.Event()
        self.values["shortcut"] = "SUPER+ALT+B"

        async def apply(shortcut, activation):
            if shortcut == "SUPER+ALT+B":
                applying.set()
                await release.wait()

        self.shortcuts.apply.side_effect = apply
        before = dict(self.daemon.config)
        await self.daemon.request({"action": "setup", "values": self.values})
        await applying.wait()
        cancellation = asyncio.create_task(self.daemon.request({"action": "cancel"}))
        await asyncio.sleep(0)
        self.assertTrue(self.daemon.cancelled.is_set())
        release.set()
        await cancellation
        await self.daemon.job
        self.assertEqual(self.daemon.config, before)
        self.assertFalse((self.paths.config / "config.json").exists())
        self.assertEqual(self.shortcuts.apply.call_args.args, ("SUPER+ALT+V", "hold"))

    async def test_setup_requires_complete_valid_choices(self):
        with self.assertLogs("omawhisper.daemon", level="ERROR"):
            result = await self.daemon.request({"action": "setup", "values": {"model": "whisper-base"}})
        self.assertIn("requires", result["error"])
        self.installer.assert_not_awaited()

    async def test_shutdown_closes_connected_watchers_before_waiting_for_server(self):
        await self.daemon.start()
        reader, writer = await asyncio.open_unix_connection(self.paths.socket)
        writer.write(b'{"action":"watch"}\n')
        await writer.drain()
        await reader.readline()
        await asyncio.wait_for(self.daemon.close(), 2)
        self.assertEqual(await reader.read(), b"")
        writer.close()
        await writer.wait_closed()


class InstallerProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_dependency_installer_output_and_exit_status_are_propagated(self):
        report = MagicMock()
        code, output = await run_installer(
            [sys.executable, "-c", "print('install failure');raise SystemExit(2)"], threading.Event(), report)
        self.assertEqual(code, 2)
        self.assertIn("install failure", output)
        report.assert_called()

    async def test_cancel_and_timeout_terminate_installer(self):
        for cancelled, timeout, error in ((True, 10, WorkerCancelled), (False, 0.01, UserError)):
            event = threading.Event()
            if cancelled:
                event.set()
            with self.assertRaises(error):
                await run_installer([sys.executable, "-c", "import time;time.sleep(60)"], event, MagicMock(), timeout)
