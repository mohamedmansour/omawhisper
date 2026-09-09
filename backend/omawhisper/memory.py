"""Conservative host-RAM checks for ONNX model initialization."""

from pathlib import Path

from .config import UserError

MIB = 1024 * 1024
GIB = 1024 * MIB


def memory_status() -> dict:
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in {"MemTotal", "MemAvailable"}:
                fields[name] = int(value.split()[0]) * 1024
        return {"total": fields["MemTotal"], "available": fields["MemAvailable"]}
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise UserError(f"Cannot check system RAM before loading the model: {exc}") from exc


def memory_error(name: str, required: int, budget: int, *, available: bool = False) -> str:
    if required <= budget:
        return ""
    kind = "currently available" if available else "installed"
    return (f"{name} needs an estimated {required / GIB:.2f} GiB of host RAM to load safely; "
            f"only {budget / GIB:.2f} GiB is {kind}. Choose Parakeet INT8 or a smaller Whisper model, "
            "or provide more RAM. Swap is not counted as model-loading capacity.")
