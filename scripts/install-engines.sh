#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${OMAWHISPER_PYTHON:-python3}"
exec "$python" - "$root" "$python" "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


GUARD = """
from importlib.metadata import PackageNotFoundError, distribution
import sys
for name in sys.argv[1:]:
    try:
        distribution(name)
    except PackageNotFoundError:
        continue
    raise SystemExit(f"Refusing incompatible installed distribution: {name}")
"""

SMOKE = """
import importlib
import sys
for name in sys.argv[1:]:
    importlib.import_module(name)
"""


def main():
    root, python = Path(sys.argv[1]), sys.argv[2]
    parser = argparse.ArgumentParser(description="Provision isolated speech runtimes only.")
    parser.add_argument("--cuda", action="store_true", help="Include Whisper and Parakeet CUDA dependencies.")
    parser.add_argument("--check", action="store_true", help="Offline readiness check: 0 ready, 1 needs provisioning, 2 error.")
    args = parser.parse_args(sys.argv[3:])
    fingerprint = hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest()
    data = Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")) / "omawhisper"
    if not os.environ.get("OMAWHISPER_PYTHON") and os.access(data / "venv/bin/python", os.X_OK):
        python = str(data / "venv/bin/python")
    environments = [
        (data / "venv", ["whisper", "cuda"] if args.cuda else ["whisper"],
         ["onnxruntime-gpu"], ["omawhisper", "faster_whisper", "ctranslate2", "onnxruntime"]),
        (data / "venv-onnx-cpu", ["parakeet"], ["onnxruntime-gpu"],
         ["omawhisper", "onnx_asr", "onnxruntime", "huggingface_hub"]),
    ]
    if args.cuda:
        environments.append(
            (data / "venv-onnx-cuda", ["parakeet-cuda"], ["onnxruntime", "faster-whisper"],
             ["omawhisper", "onnx_asr", "onnxruntime", "huggingface_hub"])
        )

    def ready(environment, extras):
        if not os.access(environment / "bin/python", os.X_OK):
            return False
        try:
            marker = json.loads((environment / ".omawhisper-ready").read_text())
        except (FileNotFoundError, ValueError, UnicodeError):
            return False
        return (isinstance(marker, dict) and marker.get("manifest_sha256") == fingerprint
                and isinstance(marker.get("extras"), list)
                and all(isinstance(extra, str) for extra in marker["extras"])
                and set(extras).issubset(marker["extras"]))

    def guard(environment, forbidden):
        result = subprocess.run(
            [str(environment / "bin/python"), "-c", GUARD, *forbidden],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"Environment {environment}: {result.stderr.strip() or 'runtime inspection failed'}. "
                "Recreate only this virtual environment, leaving settings and models intact, "
                "then retry setup (or the installer with --cuda for GPU)."
            )

    # Check every existing selected environment before changing any packages.
    pending = []
    for environment, extras, forbidden, modules in environments:
        if os.access(environment / "bin/python", os.X_OK):
            guard(environment, forbidden)
        elif (environment / "bin/python").exists():
            raise RuntimeError(f"Environment Python is not executable: {environment / 'bin/python'}")
        if not ready(environment, extras):
            pending.append((environment, extras, forbidden, modules))
    if args.check:
        return 1 if pending else 0

    for environment, extras, forbidden, modules in pending:
        executable = environment / "bin/python"
        marker = environment / ".omawhisper-ready"
        marker.unlink(missing_ok=True)
        if not os.access(executable, os.X_OK):
            subprocess.run([python, "-m", "venv", str(environment)], check=True)
        guard(environment, forbidden)
        print(f"Installing {','.join(extras)} in {environment}", flush=True)
        subprocess.run(
            [str(executable), "-m", "pip", "install", "--disable-pip-version-check",
             "--editable", f"{root}[{','.join(extras)}]"], check=True,
        )
        guard(environment, forbidden)
        subprocess.run([str(executable), "-c", SMOKE, *modules], check=True)
        staged = environment / f".omawhisper-ready.pending-{os.getpid()}"
        try:
            staged.write_text(json.dumps({"manifest_sha256": fingerprint, "extras": extras}) + "\n")
            staged.replace(marker)
        finally:
            staged.unlink(missing_ok=True)
    return 0


try:
    raise SystemExit(main())
except (OSError, RuntimeError, subprocess.SubprocessError) as error:
    print(f"Runtime setup failed: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
