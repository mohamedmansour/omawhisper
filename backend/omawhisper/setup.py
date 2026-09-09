"""Dependency provisioning for model and processing-device changes."""

import asyncio
from contextlib import suppress
import os
from pathlib import Path
import signal
import sys
import time

from .config import UserError
from .engine_worker import WorkerCancelled


async def run_installer(arguments: list[str], cancelled, report, timeout: float = 1200) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *arguments, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PIP_NO_INPUT": "1", "PIP_PROGRESS_BAR": "off", "PYTHONUNBUFFERED": "1",
             "OMAWHISPER_PYTHON": os.environ.get("OMAWHISPER_PYTHON") or sys.executable},
    )
    tail = ""
    deadline = time.monotonic() + timeout
    read = None
    try:
        while True:
            if cancelled.is_set():
                raise WorkerCancelled()
            if time.monotonic() >= deadline:
                raise UserError("Runtime installation timed out. Check your connection and retry loading in Models.")
            if read is None:
                read = asyncio.create_task(process.stdout.read(4096))
            done, _ = await asyncio.wait({read}, timeout=0.1)
            if not done:
                continue
            data = read.result()
            read = None
            if not data:
                break
            tail = (tail + data.decode(errors="replace"))[-8192:]
            lines = tail.strip().splitlines()
            if lines:
                report(lines[-1][-400:])
        return await process.wait(), tail
    finally:
        if read:
            read.cancel()
            await asyncio.gather(read, return_exceptions=True)
        # Pip/build children belong to this explicitly created process group.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


async def install_runtimes(cuda: bool, cancelled, report) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/install-engines.sh"
    arguments = ["bash", str(script), *(["--cuda"] if cuda else [])]
    code, output = await run_installer([*arguments, "--check"], cancelled, lambda line: None)
    if code == 0:
        return
    if code != 1:
        raise UserError("Cannot inspect speech runtimes: " + output.strip())
    report("Installing local speech runtimes. CUDA downloads can be several GB.")
    code, output = await run_installer(arguments, cancelled, report)
    if code:
        raise UserError("Runtime installation failed. Existing settings were kept. Retry loading in Models after resolving:\n" + output.strip())
