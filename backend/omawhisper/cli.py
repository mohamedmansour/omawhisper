"""Newline-delimited JSON control client and daemon entry point."""

import argparse
import asyncio
import json
import logging
import sys

from .config import Paths, UserError
from .daemon import LIMIT, serve


async def client(command: dict, watch: bool = False) -> bool:
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(Paths().socket, limit=LIMIT), 5)
    except (OSError, TimeoutError) as exc:
        raise UserError(f"Cannot connect to Omawhisper. Start the daemon/service first: {exc}") from exc
    try:
        writer.write(json.dumps(command, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        await writer.drain()
        while True:
            line = await reader.readline() if watch else await asyncio.wait_for(reader.readline(), 30)
            if not line:
                if watch:
                    raise UserError("Omawhisper daemon disconnected.")
                raise UserError("Omawhisper daemon closed without a response.")
            response = json.loads(line)
            print(json.dumps(response, ensure_ascii=False), flush=True)
            error = type(response) is dict and set(response) == {"error"}
            if error or not watch:
                return not error
    finally:
        writer.close()
        await writer.wait_closed()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Omawhisper local dictation service")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("daemon", help="Run the foreground dictation daemon")
    sub.add_parser("watch", help="Print live newline-delimited JSON state")
    request = sub.add_parser("request", help="Send a JSON command")
    request.add_argument("json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")
    try:
        if args.command == "daemon":
            asyncio.run(serve())
            return 0
        command = {"action": "watch"} if args.command == "watch" else json.loads(args.json)
        if type(command) is not dict:
            raise UserError("Request JSON must be an object.")
        return 0 if asyncio.run(client(command, args.command == "watch")) else 1
    except KeyboardInterrupt:
        return 130
    except (UserError, OSError, ValueError, TimeoutError) as exc:
        print(json.dumps({"error": str(exc) or type(exc).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
