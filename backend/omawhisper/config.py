"""Validated configuration and private, atomic state storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import uuid


DEFAULTS = {
    "shortcut": "SUPER+ALT+V", "activation": "hold", "model": "whisper-base",
    "language": "auto", "translate": False, "device": "", "output": "paste",
    "paste_shortcut": "auto", "restore_clipboard": True, "append_space": True,
    "history": False, "max_duration": 120, "beam_size": 5, "initial_prompt": "",
    "vad": True, "compute_type": "int8", "threads": 4, "bar_section": "right",
    "suppress_blank": True, "show_timestamps": False, "temperature": 0.0,
    "no_speech_threshold": 0.6, "use_beam_search": False,
}
MODIFIERS = {"SUPER": 64, "CTRL": 4, "ALT": 8, "SHIFT": 1}


class UserError(Exception):
    """An actionable error safe to display over local IPC."""


def shortcut_parts(value: str) -> tuple[int, str]:
    parts = value.upper().replace(" ", "").split("+")
    if len(parts) < 2 or not re.fullmatch(r"[A-Z0-9_]+", parts[-1]):
        raise UserError("Shortcut must contain modifiers and a key, e.g. SUPER+ALT+V.")
    if any(p not in MODIFIERS for p in parts[:-1]) or len(set(parts[:-1])) != len(parts[:-1]):
        raise UserError("Shortcut modifiers must be unique SUPER, CTRL, ALT or SHIFT.")
    return sum(MODIFIERS[p] for p in parts[:-1]), parts[-1]


def validate(values: dict, current: dict | None = None) -> dict:
    if type(values) is not dict:
        raise UserError("Configuration values must be an object.")
    unknown = values.keys() - DEFAULTS.keys()
    if unknown:
        raise UserError(f"Unknown configuration setting: {', '.join(sorted(unknown))}")
    result = dict(DEFAULTS if current is None else current)
    for key, value in values.items():
        if type(DEFAULTS[key]) is float:
            if type(value) not in (int, float):
                raise UserError(f"{key} must be a number.")
            result[key] = float(value)
            continue
        if type(value) is not type(DEFAULTS[key]):
            raise UserError(f"{key} must be {type(DEFAULTS[key]).__name__}.")
        result[key] = value
    result["paste_shortcut"] = result["paste_shortcut"].lower()
    enums = {
        "activation": {"toggle", "hold"}, "output": {"paste", "clipboard"},
        "paste_shortcut": {"auto", "ctrl+v", "ctrl+shift+v", "shift+insert"},
        "compute_type": {"int8", "float32", "int8_float32"},
        "bar_section": {"left", "center", "right"},
    }
    for key, choices in enums.items():
        if result[key] not in choices:
            raise UserError(f"{key} must be one of: {', '.join(sorted(choices))}.")
    for key, low, high in (("max_duration", 1, 600), ("beam_size", 1, 20), ("threads", 1, 64)):
        if not low <= result[key] <= high:
            raise UserError(f"{key} must be between {low} and {high}.")
    for key in ("temperature", "no_speech_threshold"):
        if not 0.0 <= result[key] <= 1.0:
            raise UserError(f"{key} must be between 0 and 1.")
    for key, maximum in (("model", 128), ("device", 512), ("language", 16),
                         ("initial_prompt", 8000), ("shortcut", 100)):
        if len(result[key]) > maximum or "\0" in result[key]:
            raise UserError(f"{key} is too long or contains a NUL character.")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", result["model"]):
        raise UserError("Invalid model ID.")
    if result["language"] != "auto" and not re.fullmatch(r"[a-z]{2,3}", result["language"]):
        raise UserError("Language must be auto or a lowercase language code.")
    shortcut_parts(result["shortcut"])
    result["shortcut"] = result["shortcut"].upper().replace(" ", "")
    return result


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


class Paths:
    def __init__(self) -> None:
        home = Path.home()
        self.config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "omawhisper"
        self.data = Path(os.environ.get("XDG_DATA_HOME", home / ".local/share")) / "omawhisper"
        self.state = Path(os.environ.get("XDG_STATE_HOME", home / ".local/state")) / "omawhisper"
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        if not runtime:
            raise UserError("XDG_RUNTIME_DIR is unset; run Omawhisper in your desktop user session.")
        self.runtime = Path(runtime) / "omawhisper"
        self.socket = self.runtime / "control.sock"
        self.models = self.data / "models"

    def load_config(self) -> dict:
        path = self.config / "config.json"
        if not path.exists():
            return dict(DEFAULTS)
        try:
            return validate(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError) as exc:
            raise UserError(f"Cannot read configuration {path}: {exc}") from exc

    def save_config(self, config: dict) -> None:
        atomic_json(self.config / "config.json", config)
