"""Bounded subprocess calls: argv only, never a shell."""

import asyncio

from .config import UserError


async def run(*argv: str, data: bytes | None = None, timeout: float = 5) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise UserError(f"Required executable is missing: {argv[0]}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(data), timeout)
    except (TimeoutError, asyncio.CancelledError):
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        detail = stderr.decode(errors="replace").strip()[:1000]
        raise UserError(f"{argv[0]} failed ({process.returncode}): {detail}")
    return stdout
