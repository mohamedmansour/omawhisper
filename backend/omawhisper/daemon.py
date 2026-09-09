"""Single-instance, bounded Unix-socket service with background inference."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import threading
import uuid

from . import audio
from .config import DEFAULTS, Paths, UserError, atomic_json, validate
from .models import Models, onnx_quantization
from .output import Output
from .shortcuts import Shortcuts
from .acceleration import cuda_available, onnx_gpu_compatible
from .engine_worker import WorkerCancelled
from .setup import install_runtimes


LOG = logging.getLogger(__name__)
LIMIT = 2 * 1024 * 1024
BUSY = {"recording", "transcribing", "loading", "downloading", "installing"}


class Daemon:
    def __init__(self, paths=None, models=None, output=None, shortcuts=None, capture_factory=audio.Capture):
        self.paths = paths or Paths()
        self.initial_error = ""
        try:
            self.config = self.paths.load_config()
        except (UserError, OSError, ValueError) as exc:
            self.config = dict(DEFAULTS)
            self.initial_error = str(exc)
        self.models = models or Models(self.paths)
        self.output = output or Output()
        self.shortcuts = shortcuts or Shortcuts()
        self.capture_factory = capture_factory
        self.phase = "error" if self.initial_error else "idle"
        self.error = self.initial_error
        self.message = ""
        self.transcript = ""
        self.level = 0.0
        self.elapsed = 0.0
        self.devices = [{"id": "", "name": "Default microphone"}]
        self.history = []
        self.capture = None
        self.job = None
        self.cancelled = threading.Event()
        self.lock = asyncio.Lock()
        self.subscribers = set()
        self.clients = set()
        self.connections = set()
        self.closed = asyncio.Event()
        self.server = None
        self.lock_fd = None
        self.socket_identity = None
        self.event_task = None
        self.heartbeat_task = None
        self.setup_running = False
        self._load_history()

    def _load_history(self) -> None:
        path = self.paths.state / "history.json"
        if self.config["history"] and path.exists():
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
                if type(records) is not list or any(
                    type(row) is not dict or any(type(row.get(k)) is not str for k in ("id", "text", "timestamp", "model"))
                    for row in records
                ):
                    raise UserError("Invalid history format.")
                self.history = [{key: row[key] for key in ("id", "text", "timestamp", "model")} for row in records[:50]]
            except (OSError, ValueError, UserError) as exc:
                self.fail(UserError(f"Cannot read dictation history: {exc}"))

    def snapshot(self) -> dict:
        selected = self.models.get(self.config["model"])
        return {
            "phase": self.phase, "level": float(self.level), "elapsed": self.elapsed,
            "error": self.error, "message": self.message, "transcript": self.transcript,
            "model_name": selected["name"], "config": dict(self.config),
            "runtime": self.models.runtime_snapshot(),
            "models": self.models.snapshot(), "devices": list(self.devices), "history": list(self.history),
        }

    def publish(self) -> None:
        for queue in tuple(self.subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(True)

    def fail(self, exc: BaseException, *, preserve_activity: bool = True) -> None:
        LOG.error("%s", exc, exc_info=not isinstance(exc, UserError))
        self.error = str(exc) or type(exc).__name__
        if not (preserve_activity and self._busy()):
            self.phase = "error"
            self.level = 0.0
        self.publish()

    def idle(self, message: str = "") -> None:
        self.phase, self.error, self.message, self.level = "idle", "", message, 0.0
        self.publish()

    def _busy(self) -> bool:
        return self.job is not None and not self.job.done()

    def require_idle(self) -> None:
        if self._busy():
            raise UserError("Omawhisper is busy. Stop or cancel the current operation first.")

    def _launch(self, operation) -> None:
        self.cancelled = threading.Event()
        self.job = asyncio.create_task(self._boundary(operation))

    async def _boundary(self, operation) -> None:
        try:
            await operation
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(exc, preserve_activity=False)
        finally:
            self.publish()

    async def request(self, command: dict) -> dict:
        # Cancellation must reach workers even while a final shortcut update owns the lock.
        if type(command) is dict and command == {"action": "cancel"} and self._busy():
            self.cancelled.set()
        async with self.lock:
            try:
                await self._request(command)
                return self.snapshot()
            except Exception as exc:
                self.fail(exc)
                return {"error": self.error}

    async def _request(self, command: dict) -> None:
        if type(command) is not dict or type(command.get("action")) is not str:
            raise UserError("Request must be an object containing a string action.")
        action = command["action"]
        extra = {
            "configure": {"values"}, "setup": {"values"}, "download": {"id"}, "load": {"id"},
            "remove_model": {"id"}, "add_model": {"model"}, "copy": {"text"},
            "start": {"hotkey"},
        }.get(action, set())
        if command.keys() - {"action"} - extra:
            raise UserError("Request contains unknown fields.")
        if "hotkey" in command and type(command["hotkey"]) is not bool:
            raise UserError("hotkey must be a boolean.")
        if action == "status":
            return
        if action == "devices":
            self.devices = await audio.devices()
            self.publish()
        elif action in {"toggle", "start", "stop"}:
            if action == "toggle":
                action = "stop" if self.capture else "start"
            if action == "stop":
                if self.capture:
                    await self.capture.stop()
                    self.message = "Finishing recording…"
                    self.publish()
                return
            if self.capture:
                return
            if "hotkey" in command:
                # A very quick release can launch its IPC client before the press client.
                if command["hotkey"] and not await self.shortcuts.held():
                    return
            self.require_idle()
            self.models.check_settings(self.config["model"], self.config)
            selected = self.models.get(self.config["model"])
            if not self.models.installed(selected):
                raise UserError(f"{selected['name']} is not installed. Download it in Models or provide its local files/executable before recording.")
            self.models.check_memory(self.config["model"])
            target = await self.output.focused() if self.config["output"] == "paste" else {}
            settings = dict(self.config)
            capture = self.capture_factory(self.paths.runtime, settings["device"], settings["max_duration"], self._meter)
            await capture.start()
            self.capture = capture
            self.phase, self.error, self.message = "recording", "", "Listening…"
            self.elapsed, self.level = 0.0, 0.0
            self._launch(self._record(capture, target, settings))
            self.publish()
        elif action == "cancel":
            if self._busy():
                self.cancelled.set()
                if self.capture:
                    await self.capture.stop(cancel=True)
                self.message = "Cancelled. Waiting for the local worker to release resources."
                self.publish()
        elif action == "setup":
            self.require_idle()
            values = command.get("values")
            # Accept the original full request from older clients, while new setup only owns model/device.
            if type(values) is not dict or set(values) not in (
                {"model", "acceleration"}, {"model", "acceleration", "activation", "shortcut"},
            ):
                raise UserError("Setup requires model and acceleration choices.")
            config = validate(values, self.config)
            if config["acceleration"] != self.config["acceleration"]:
                config["compute_type"] = "auto"
            if self.models.get(config["model"])["engine"] != "faster-whisper":
                config.update(language="auto", translate=False, initial_prompt="", show_timestamps=False)
            self.models.check_settings(config["model"], config)
            self.setup_running = True
            self.phase, self.error, self.message = "installing", "", "Preparing your local dictation setup..."
            self._launch(self._setup_job(config))
            self.publish()
        elif action == "configure":
            if self.setup_running:
                self.require_idle()
            values = command.get("values")
            config = validate(values, self.config)
            self.models.check_settings(config["model"], config)
            changes = {key for key in config if config[key] != self.config[key]}
            reload_model = bool(changes & {"model", "acceleration", "compute_type", "threads"})
            if reload_model:
                self.require_idle()
            old = self.config
            if changes & {"shortcut", "activation"}:
                await self.shortcuts.apply(config["shortcut"], config["activation"])
            try:
                self.paths.save_config(config)
            except OSError:
                if changes & {"shortcut", "activation"}:
                    await self.shortcuts.apply(old["shortcut"], old["activation"])
                raise
            self.config = config
            self.error = ""
            if changes & {"shortcut", "activation"} and self.capture:
                await self.capture.stop()
            if reload_model:
                await asyncio.to_thread(self.models.unload)
            if "history" in changes:
                if config["history"]:
                    self._load_history()
                else:
                    self.history.clear()
            if not self._busy() and not self.error:
                self.idle("Settings saved.")
            else:
                self.publish()
        elif action in {"download", "load", "remove_model", "unload"}:
            self.require_idle()
            identifier = command.get("id") if action != "unload" else self.config["model"]
            self.models.get(identifier)
            if action == "load":
                self.models.check_settings(identifier, self.config)
            self.phase = "downloading" if action == "download" else "loading"
            self.error = ""
            self.message = f"{action.replace('_', ' ').capitalize()} in progress…"
            self._launch(self._model_job(action, identifier, dict(self.config)))
            self.publish()
        elif action == "add_model":
            self.require_idle()
            self.models.add(command.get("model"))
            self.idle("Custom model added. Only engine-compatible model formats can be loaded.")
        elif action == "clear_history":
            (self.paths.state / "history.json").unlink(missing_ok=True)
            self.history.clear()
            self.message = "History cleared."
            self.publish()
        elif action == "copy":
            text = command.get("text")
            if type(text) is not str or not text or len(text) > 500000 or "\0" in text:
                raise UserError("Copy requires a nonempty text string of at most 500,000 characters.")
            self.message = await self.output.copy(text)
            self.publish()
        elif action == "dismiss_error":
            if self._busy():
                raise UserError("Wait for the current operation before dismissing its error.")
            self.idle()
        else:
            raise UserError(f"Unknown action: {action}")

    def _meter(self, level: float, elapsed: float) -> None:
        self.level, self.elapsed = float(level), elapsed
        self.publish()

    async def _setup_job(self, config: dict) -> None:
        committed = False
        replaced = False
        try:
            model = self.models.get(config["model"])
            self.models.check_memory(model["id"])
            cuda = False
            if config["acceleration"] != "cpu" and model["engine"] != "command":
                compatible = model["engine"] == "faster-whisper" or onnx_gpu_compatible(onnx_quantization(model))
                if compatible:
                    cuda, detail = await asyncio.to_thread(cuda_available)
                    if config["acceleration"] == "gpu" and not cuda:
                        raise UserError(detail)
            await asyncio.to_thread(self.models.unload)
            replaced = True
            self.models.check_memory(model["id"], available=True)
            if model["engine"] != "command":
                def progress(message):
                    self.message = message
                    self.publish()

                await install_runtimes(cuda, self.cancelled, progress)
            if not self.models.installed(model):
                self.phase, self.message = "downloading", "Downloading " + model["name"] + "..."
                self.publish()
                await asyncio.to_thread(self.models.download, model["id"], self.cancelled)
            if self.cancelled.is_set():
                raise WorkerCancelled()
            self.phase, self.message = "loading", "Loading and preparing " + model["name"] + "..."
            self.publish()
            await asyncio.to_thread(self.models.load, model["id"], config, self.cancelled)
            async with self.lock:
                if self.cancelled.is_set():
                    raise WorkerCancelled()
                old = self.config
                await self.shortcuts.apply(config["shortcut"], config["activation"])
                if self.cancelled.is_set():
                    await self.shortcuts.apply(old["shortcut"], old["activation"])
                    raise WorkerCancelled()
                selected = dict(config, setup_complete=True)
                try:
                    self.paths.save_config(selected)
                except OSError:
                    await self.shortcuts.apply(old["shortcut"], old["activation"])
                    raise
                self.config = selected
                committed = True
            self.idle("Setup complete. Focus a text input and use your activation shortcut.")
        except WorkerCancelled:
            self.idle("Setup cancelled. Previous settings were kept.")
        except UserError:
            if self.cancelled.is_set():
                self.idle("Setup cancelled. Previous settings were kept.")
            else:
                raise
        finally:
            if not committed and replaced:
                await asyncio.to_thread(self.models.unload)
            self.setup_running = False

    async def _record(self, capture, target: dict, config: dict) -> None:
        try:
            path = await capture.wait()
            if self.cancelled.is_set() or capture.cancelled:
                self.idle("Recording cancelled; audio deleted.")
                return
            self.capture = None
            self.phase, self.level, self.message = "transcribing", 0.0, "Transcribing locally…"
            self.publish()
            text = await asyncio.to_thread(self.models.transcribe, path, config["model"], config, self.cancelled)
            if self.cancelled.is_set():
                self.idle("Transcription cancelled; audio deleted.")
                return
            self.transcript = text
            if not text:
                self.idle("No speech recognized.")
                return
            if self.config["history"]:
                self.history.insert(0, {
                    "id": uuid.uuid4().hex, "text": text,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": self.models.get(config["model"])["name"],
                })
                self.history = self.history[:50]
                while sum(len(row["text"]) for row in self.history) > 200000:
                    self.history.pop()
                atomic_json(self.paths.state / "history.json", self.history)
            self.publish()
            message = await self.output.deliver(text, target, config)
            if self.models.get(config["model"])["engine"] != "faster-whisper":
                message += " Whisper-only decoding, VAD and prompt controls are not applied."
            self.idle(message)
        finally:
            self.capture = None
            await capture.stop(cancel=self.cancelled.is_set())
            capture.cleanup()

    async def _model_job(self, action: str, identifier: str, config: dict) -> None:
        if action == "download":
            await asyncio.to_thread(self.models.download, identifier, self.cancelled)
        elif action == "load":
            await asyncio.to_thread(self.models.load, identifier, config, self.cancelled)
            try:
                async with self.lock:
                    if not self.cancelled.is_set():
                        selected = dict(self.config, model=identifier)
                        self.models.check_settings(identifier, selected)
                        self.paths.save_config(selected)
                        self.config = selected
            except Exception:
                await asyncio.to_thread(self.models.unload)
                raise
        elif action == "remove_model":
            async with self.lock:
                if self.cancelled.is_set():
                    self.idle("Operation cancelled.")
                    return
                old_identifier = self.config["model"]
                if old_identifier == identifier:
                    selected = dict(self.config, model="whisper-base")
                    self.paths.save_config(selected)
                    self.config = selected
            await asyncio.to_thread(self.models.remove, identifier)
        elif action == "unload":
            await asyncio.to_thread(self.models.unload)
        if self.cancelled.is_set():
            if action == "load":
                await asyncio.to_thread(self.models.unload)
            self.idle("Operation cancelled.")
        else:
            self.idle(f"{action.replace('_', ' ').capitalize()} complete.")

    async def _reapply(self) -> None:
        async with self.lock:
            if self.capture and self.config["activation"] == "hold":
                await self.capture.stop()
            await self.shortcuts.apply(self.config["shortcut"], self.config["activation"], force=True)

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(1)
            if self._busy():
                self.publish()

    async def start(self, shortcuts: bool = True) -> None:
        self.models.get(self.config["model"])
        runtime = self.paths.runtime
        runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = runtime.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise UserError("Unsafe runtime directory: must be a directory owned by the current user.")
        runtime.chmod(0o700)
        self.lock_fd = os.open(runtime / "daemon.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise UserError("An Omawhisper daemon is already running.") from exc
        if self.paths.socket.exists() or self.paths.socket.is_symlink():
            info = self.paths.socket.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise UserError("Refusing to replace an unsafe control socket path.")
            self.paths.socket.unlink()
        for path in runtime.glob("recording-*.wav"):
            if path.is_file() and not path.is_symlink():
                path.unlink()
        self.server = await asyncio.start_unix_server(self._client, path=self.paths.socket, limit=LIMIT)
        self.paths.socket.chmod(0o600)
        self.socket_identity = self.paths.socket.stat().st_ino
        self.heartbeat_task = asyncio.create_task(self._heartbeat())
        if shortcuts:
            try:
                await self._reapply()
            except Exception as exc:
                self.fail(exc)
            self.event_task = asyncio.create_task(self._events())

    async def _events(self) -> None:
        try:
            await self.shortcuts.events(self._reapply, self.fail)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(exc)

    async def _send(self, writer, payload: dict) -> None:
        writer.write(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        await asyncio.wait_for(writer.drain(), 5)

    async def _client(self, reader, writer) -> None:
        task = asyncio.current_task()
        self.clients.add(task)
        self.connections.add(writer)
        queue = None
        try:
            if len(self.clients) > 64:
                raise UserError("Too many connected clients.")
            peer = writer.get_extra_info("socket")
            _, uid, _ = struct.unpack("3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != os.getuid():
                raise UserError("IPC peer UID does not match.")
            raw = await asyncio.wait_for(reader.readline(), 10)
            if not raw:
                return
            command = json.loads(raw)
            if command == {"action": "watch"}:
                queue = asyncio.Queue(maxsize=1)
                self.subscribers.add(queue)
                await self._send(writer, self.snapshot())
                disconnected = asyncio.create_task(reader.read(1))
                changed = None
                try:
                    while True:
                        changed = asyncio.create_task(queue.get())
                        done, _ = await asyncio.wait({changed, disconnected}, return_when=asyncio.FIRST_COMPLETED)
                        if disconnected in done:
                            changed.cancel()
                            await asyncio.gather(changed, return_exceptions=True)
                            break
                        await self._send(writer, self.snapshot())
                finally:
                    if changed:
                        changed.cancel()
                        await asyncio.gather(changed, return_exceptions=True)
                    disconnected.cancel()
                    await asyncio.gather(disconnected, return_exceptions=True)
            else:
                await self._send(writer, await self.request(command))
        except asyncio.CancelledError:
            raise
        except (ConnectionError, BrokenPipeError):
            LOG.debug("IPC client disconnected.")
        except Exception as exc:
            self.fail(exc)
            with suppress(ConnectionError, TimeoutError):
                await self._send(writer, {"error": self.error})
        finally:
            if queue:
                self.subscribers.discard(queue)
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.clients.discard(task)
            self.connections.discard(writer)

    async def close(self) -> None:
        self.closed.set()
        if self.server:
            self.server.close()
        for task in (self.event_task, self.heartbeat_task):
            if task:
                task.cancel()
        await asyncio.gather(*(task for task in (self.event_task, self.heartbeat_task) if task), return_exceptions=True)
        self.cancelled.set()
        if self.capture:
            await self.capture.stop(cancel=True)
        if self.job:
            await self.job
        for task in tuple(self.clients):
            task.cancel()
        await asyncio.gather(*self.clients, return_exceptions=True)
        if self.server:
            await self.server.wait_closed()
        try:
            await self.shortcuts.close()
        except Exception as exc:
            LOG.error("Shortcut cleanup failed: %s", exc)
        await self.output.close()
        await asyncio.to_thread(self.models.unload)
        if self.socket_identity is not None:
            with suppress(FileNotFoundError):
                if self.paths.socket.lstat().st_ino == self.socket_identity:
                    self.paths.socket.unlink()
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None


async def serve() -> None:
    daemon = Daemon()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, daemon.closed.set)
    try:
        await daemon.start()
        await daemon.closed.wait()
    finally:
        await daemon.close()
