#!/usr/bin/env python3
"""Install the user-owned plugin and its graphical-session service."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ID = "mohamedmansour.whisper"


def run(*args):
    subprocess.run(args, check=True)


def install(section):
    config = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    plugin = config / "omarchy" / "plugins" / PLUGIN_ID
    unit = config / "systemd" / "user" / "omawhisper.service"
    if plugin.exists() and plugin.resolve() != ROOT:
        raise RuntimeError(f"{plugin} already exists and belongs to another installation.")
    if plugin.is_symlink() and plugin.resolve() != ROOT:
        raise RuntimeError(f"{plugin} is an unrelated symlink; not replacing it.")
    plugin.parent.mkdir(parents=True, exist_ok=True)
    if not plugin.exists():
        plugin.symlink_to(ROOT, target_is_directory=True)

    executable = str(ROOT / "scripts" / "omawhisper")
    if any(character in executable for character in "\n\r"):
        raise RuntimeError("The installation path cannot contain newlines.")
    escaped = executable.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    unit_text = f"""# Managed by Omawhisper scripts/install.py
[Unit]
Description=Omawhisper local voice dictation
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
ExecStart="{escaped}" daemon
Restart=on-failure
RestartSec=3
OOMPolicy=continue
TimeoutStopSec=15
UMask=0077

[Install]
WantedBy=graphical-session.target
"""
    unit.parent.mkdir(parents=True, exist_ok=True)
    if unit.exists() and not unit.read_text().startswith("# Managed by Omawhisper"):
        raise RuntimeError(f"Refusing to overwrite unrelated service: {unit}")
    if unit.exists() and unit.read_text() != unit_text:
        shutil.copy2(unit, unit.with_suffix(f".service.bak.{time.time_ns()}"))
    unit.write_text(unit_text)
    # The compositor environment must be available to the graphical-session service.
    variables = [name for name in ("WAYLAND_DISPLAY", "HYPRLAND_INSTANCE_SIGNATURE", "XDG_CURRENT_DESKTOP") if name in os.environ]
    if variables:
        run("systemctl", "--user", "import-environment", *variables)
    run("systemctl", "--user", "daemon-reload")
    run("systemctl", "--user", "enable", "--now", "omawhisper.service")
    run("systemctl", "--user", "restart", "omawhisper.service")
    run("omarchy-shell", "shell", "rescanPlugins")
    run("omarchy", "bar", "put", PLUGIN_ID, "--section", section)
    print(f"Omawhisper installed on the {section}. Click the microphone to download a model.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("left", "center", "right"), default="right")
    args = parser.parse_args()
    try:
        install(args.section)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Installation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
