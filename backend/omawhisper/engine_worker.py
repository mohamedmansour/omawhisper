"""Persistent native-engine subprocesses, isolated from the desktop daemon."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from .config import UserError

LIMIT = 2 * 1024 * 1024


class WorkerCancelled(Exception):
    """An explicitly cancelled engine operation."""


def worker_environment(python: Path) -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    libraries = sorted(python.parent.parent.glob("lib/python*/site-packages/nvidia/*/lib"))
    if libraries:
        environment["LD_LIBRARY_PATH"] = ":".join(
            [*(str(path) for path in libraries), *filter(None, [environment.get("LD_LIBRARY_PATH")])]
        )
    return environment


class EngineWorker:
    def __init__(self, python: Path, engine: str):
        self.process = None
        self.channel = None
        self.buffer = bytearray()
        self.sequence = 0
        parent, child = socket.socketpair()
        try:
            self.process = subprocess.Popen(
                [str(python), "-m", "omawhisper.worker", "--engine", engine,
                 "--fd", str(child.fileno()), "--parent", str(os.getpid())],
                pass_fds=(child.fileno(),), env=worker_environment(python),
                stdin=subprocess.DEVNULL, stdout=2,
            )
            parent.settimeout(0.1)
            self.channel = parent
        except OSError as exc:
            parent.close()
            raise UserError(f"Cannot start the speech engine: {exc}. Load the model from Models to install its runtime.") from exc
        finally:
            child.close()

    @property
    def alive(self) -> bool:
        process = self.process
        return process is not None and process.poll() is None

    def disconnected_error(self) -> UserError:
        code = self.process.poll() if self.process is not None else None
        if code is None and self.process is not None:
            try:
                code = self.process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        if code == -signal.SIGKILL:
            return UserError("The speech worker was killed by SIGKILL, possibly because it ran out of RAM. "
                             "Choose Parakeet INT8 or a smaller Whisper model.")
        return UserError(f"The speech engine disconnected (exit {code}). "
                         "Check the runtime/driver and journalctl --user -u omawhisper.")

    def request(self, action: str, *, cancelled: threading.Event | None = None,
                timeout: float = 600, **values):
        if not self.alive:
            error = self.disconnected_error()
            self.close()
            raise error
        self.sequence += 1
        payload = json.dumps({"id": self.sequence, "action": action, **values}, allow_nan=False).encode() + b"\n"
        if len(payload) > LIMIT:
            raise UserError("Speech-engine request exceeds the safety limit.")
        deadline = time.monotonic() + timeout
        try:
            self.channel.sendall(payload)
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise WorkerCancelled()
                if time.monotonic() >= deadline:
                    raise UserError("The speech engine timed out. Try a smaller model or another processing device.")
                if b"\n" in self.buffer:
                    line, _, remainder = self.buffer.partition(b"\n")
                    self.buffer = bytearray(remainder)
                    result = json.loads(line)
                    if type(result) is not dict or result.get("id") != self.sequence:
                        raise UserError("Invalid response from the speech engine.")
                    if "error" in result:
                        raise UserError(str(result["error"]))
                    if set(result) != {"id", "result"}:
                        raise UserError("Incomplete response from the speech engine.")
                    return result["result"]
                try:
                    data = self.channel.recv(65536)
                except socket.timeout:
                    continue
                if not data:
                    raise self.disconnected_error()
                self.buffer.extend(data)
                if len(self.buffer) > LIMIT:
                    raise UserError("Speech-engine response exceeds the safety limit.")
        except (WorkerCancelled, UserError):
            self.close(interrupt=True)
            raise
        except (OSError, ValueError) as exc:
            error = self.disconnected_error() if not self.alive else UserError(f"Speech-engine communication failed: {exc}")
            self.close(interrupt=True)
            raise error from exc

    def close(self, *, interrupt: bool = False) -> None:
        if self.channel is not None:
            self.channel.close()
            self.channel = None
        if self.process is not None:
            if self.process.poll() is None:
                if interrupt:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            self.process = None
        self.buffer.clear()


def engine_python(paths, engine: str, acceleration: str, gpu_compatible: bool = True) -> Path:
    if engine == "faster-whisper":
        python = paths.data / "venv/bin/python"
    else:
        cuda = paths.data / "venv-onnx-cuda/bin/python"
        if acceleration != "cpu" and gpu_compatible and cuda.is_file():
            return cuda
        if acceleration == "gpu":
            raise UserError("The ONNX CUDA runtime is not installed. Choose GPU under Models > Processing device, then select a model.")
        python = paths.data / "venv-onnx-cpu/bin/python"
    # Existing installations remain usable until Models provisions the isolated environments.
    return python if python.is_file() else Path(sys.executable)
