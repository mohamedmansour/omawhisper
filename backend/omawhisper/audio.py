"""PipeWire capture with 20 Hz real PCM RMS metering."""

from __future__ import annotations

import array
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid
import wave

from .config import UserError
from .process import run


def rms(data: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(data[:len(data) // 2 * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    return min(1.0, math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768)


async def devices() -> list[dict]:
    result = [{"id": "", "name": "Default microphone"}]
    nodes = json.loads(await run("pw-dump"))
    for node in nodes:
        props = node.get("info", {}).get("props", {})
        if props.get("media.class") == "Audio/Source":
            identifier = str(props.get("node.name", node["id"]))
            result.append({"id": identifier, "name": str(props.get("node.description", identifier))})
    return result


class Capture:
    def __init__(self, directory: Path, device: str, maximum: int, meter):
        self.directory, self.device, self.maximum, self.meter = directory, device, maximum, meter
        self.process = None
        self.path: Path | None = None
        self.cancelled = False
        self.stopped = False
        self.frames = 0
        self._task = None
        self._signalled = False

    async def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / f"recording-{uuid.uuid4().hex}.wav"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        argv = ["pw-record", "--raw", "--format", "s16", "--rate", "16000", "--channels", "1"]
        if self.device:
            argv += ["--target", self.device]
        try:
            self.process = await asyncio.create_subprocess_exec(
                *argv, "-", stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            self.cleanup()
            raise UserError(f"Cannot open PipeWire capture: {exc}") from exc
        self._task = asyncio.create_task(self._read())

    async def _read(self) -> Path:
        assert self.process and self.process.stdout and self.process.stderr and self.path
        stderr_task = asyncio.create_task(self.process.stderr.read(65536))
        started = time.monotonic()
        try:
            with wave.open(str(self.path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                while not self.stopped:
                    remaining = self.maximum - (time.monotonic() - started)
                    if remaining <= 0:
                        break
                    try:
                        block = await asyncio.wait_for(self.process.stdout.readexactly(1600), remaining)
                    except asyncio.IncompleteReadError as exc:
                        block = exc.partial
                        self.stopped = True
                    except TimeoutError:
                        break
                    block = block[:len(block) // 2 * 2]
                    if block:
                        output.writeframesraw(block)
                        self.frames += len(block) // 2
                        self.meter(rms(block), min(self.maximum, time.monotonic() - started))
                await self._terminate()
            detail = (await stderr_task).decode(errors="replace").strip()
            # pw-record exits 1 (not -SIGINT) after handling an intentional SIGINT.
            expected = (0, -signal.SIGINT, -signal.SIGTERM, 1) if self._signalled else (0,)
            if self.process.returncode not in expected and not self.cancelled:
                raise UserError(f"Microphone capture failed: {detail[:1000] or self.process.returncode}")
            if not self.frames and not self.cancelled:
                raise UserError("No audio received. Check the microphone and PipeWire permissions.")
            return self.path
        except BaseException:
            await self._terminate()
            self.cleanup()
            raise
        finally:
            if not stderr_task.done():
                stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)

    async def _terminate(self) -> None:
        if self.process and self.process.returncode is None:
            try:
                self.process.send_signal(signal.SIGINT)
                self._signalled = True
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(self.process.wait(), 2)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def stop(self, cancel: bool = False) -> None:
        self.cancelled = self.cancelled or cancel
        self.stopped = True
        await self._terminate()

    async def wait(self) -> Path:
        if self._task is None:
            raise RuntimeError("Capture has not started.")
        return await self._task

    def cleanup(self) -> None:
        if self.path:
            self.path.unlink(missing_ok=True)
