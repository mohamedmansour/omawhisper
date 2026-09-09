#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for tool in omarchy hyprctl pw-record pw-dump wl-copy wl-paste systemctl; do
  if ! command -v "$tool" >/dev/null; then
    printf 'Missing dependency: %s. See README.md for packages.\n' "$tool" >&2
    exit 1
  fi
done

python="${OMAWHISPER_PYTHON:-python3}"
venv="${XDG_DATA_HOME:-$HOME/.local/share}/omawhisper/venv"
if [[ ! -x "$venv/bin/python" ]]; then
  "$python" -m venv "$venv"
fi
"$venv/bin/python" -m pip install --disable-pip-version-check --editable "$root[engines]"
omarchy plugin validate "$root"
"$venv/bin/python" "$root/scripts/install.py" "$@"
