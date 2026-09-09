"""Private socket-pair protocol for a single persistent speech engine."""

import argparse
import ctypes
import json
import logging
import os
from pathlib import Path
import signal
import socket

from .config import UserError, validate
from .engine_worker import LIMIT
from .engines import OnnxEngine, WhisperEngine

LOG = logging.getLogger(__name__)


def serve(channel: socket.socket, engine) -> None:
    loaded = False
    with channel.makefile("rb") as reader:
        while line := reader.readline(LIMIT + 1):
            if len(line) > LIMIT or not line.endswith(b"\n"):
                raise UserError("Invalid speech-engine request framing.")
            request = json.loads(line)
            identifier = request["id"]
            try:
                config = validate(request["config"])
                if request["action"] == "load":
                    result = engine.load(request["model"], request["path"], config)
                    loaded = True
                elif request["action"] == "transcribe" and loaded:
                    result = engine.transcribe(request["audio"], config)
                    if type(result) is not str or len(result) > 100000:
                        raise UserError("Speech engine did not return a valid bounded text transcript.")
                else:
                    raise UserError("Invalid speech-engine operation or no model loaded.")
                response = {"id": identifier, "result": result}
            except Exception as exc:
                LOG.exception("Speech engine failed")
                channel.sendall(json.dumps({"id": identifier, "error": str(exc) or type(exc).__name__}).encode() + b"\n")
                return
            encoded = json.dumps(response, allow_nan=False).encode() + b"\n"
            if len(encoded) > LIMIT:
                raise UserError("Speech-engine response exceeds the safety limit.")
            channel.sendall(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("faster-whisper", "onnx-asr"), required=True)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Sacrifice this replaceable native worker, not the daemon or desktop, under memory pressure.
    Path("/proc/self/oom_score_adj").write_text("1000")
    # Register in the child, never preexec_fn in the daemon's multithreaded process.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot attach speech worker lifetime to its parent")
    if os.getppid() != args.parent:
        LOG.info("Parent exited before worker initialization")
        return
    with socket.socket(fileno=args.fd) as channel:
        serve(channel, WhisperEngine() if args.engine == "faster-whisper" else OnnxEngine())


if __name__ == "__main__":
    main()
