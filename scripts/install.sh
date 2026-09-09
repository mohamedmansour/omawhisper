#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
engine_args=()
install_args=()
for argument in "$@"; do
  if [[ "$argument" == "--cuda" ]]; then
    engine_args=(--cuda)
  else
    install_args+=("$argument")
  fi
done
for tool in omarchy hyprctl pw-record pw-dump wl-copy wl-paste systemctl; do
  if ! command -v "$tool" >/dev/null; then
    printf 'Missing dependency: %s. See README.md for packages.\n' "$tool" >&2
    exit 1
  fi
done

"$root/scripts/install-engines.sh" "${engine_args[@]}"
venv="${XDG_DATA_HOME:-$HOME/.local/share}/omawhisper/venv"
omarchy plugin validate "$root"
"$venv/bin/python" "$root/scripts/install.py" "${install_args[@]}"
