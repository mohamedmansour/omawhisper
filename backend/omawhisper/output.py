"""Explicitly targeted paste and conservative clipboard restoration."""

from __future__ import annotations

import asyncio
import json
import re

from .config import UserError
from .process import run
from .shortcuts import lua_string


TERMINALS = {
    "alacritty", "kitty", "foot", "footclient", "com.mitchellh.ghostty", "ghostty",
    "org.wezfurlong.wezterm", "wezterm", "org.gnome.terminal", "gnome-terminal",
    "konsole", "org.kde.konsole", "xfce4-terminal", "tilix",
}
MIME = "text/plain;charset=utf-8"


class Output:
    def __init__(self, runner=run):
        self.run = runner
        self.owner = None
        self._lock = asyncio.Lock()

    async def focused(self) -> dict:
        window = json.loads(await self.run("hyprctl", "-j", "activewindow"))
        address = window.get("address", "")
        if address and not re.fullmatch(r"0x[0-9a-fA-F]+", address):
            raise UserError("Hyprland returned an invalid window address.")
        return {"address": address, "class": window.get("class", "")}

    async def layer_active(self) -> bool:
        result = (await self.run(
            "hyprctl", "repl",
            "for _,layer in ipairs(hl.get_layers()) do "
            "if layer.mapped and layer.interactivity > 0 then return true end end; return false",
        )).strip()
        if result not in (b"true", b"false"):
            raise UserError("Cannot verify compositor layer focus; transcript retained without pasting.")
        return result == b"true"

    async def dispatch(self, target: dict, modifiers: str, key: str) -> str:
        address = lua_string(target["address"])
        source = (
            f"local w=hl.get_active_window(); if not w or w.address~={address} then return 'blocked-focus' end;"
            "for _,layer in ipairs(hl.get_layers()) do "
            "if layer.mapped and layer.interactivity>0 then return 'blocked-layer' end end;"
            "local result=hl.dispatch(hl.dsp.send_shortcut({"
            f"mods={lua_string(modifiers)},key={lua_string(key)},"
            f"window={lua_string('address:' + target['address'])}"
            "})); if not result.ok then return 'error:'..tostring(result.error) end; return 'ok'"
        )
        # Focus/layer checks and the address-targeted dispatch execute atomically in Lua.
        result = (await self.run("hyprctl", "repl", source)).decode(errors="replace").strip()
        if result not in {"ok", "blocked-focus", "blocked-layer"}:
            raise UserError(f"Paste dispatch rejected: {result[:1000]}")
        return result

    async def _set(self, data: bytes, mime: str = MIME) -> None:
        process = await asyncio.create_subprocess_exec(
            "wl-copy", "--foreground", "--type", mime,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin
        try:
            process.stdin.write(data)
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()
            await asyncio.sleep(0.04)
            if process.returncode is not None:
                error = await process.stderr.read()
                raise UserError(f"Cannot own clipboard: {error.decode(errors='replace')[:500]}")
        except BaseException:
            if process.returncode is None:
                process.terminate()
            await process.wait()
            raise
        old_owner, self.owner = self.owner, process
        if old_owner and old_owner.returncode is None:
            old_owner.terminate()
            await old_owner.wait()

    async def _previous(self) -> tuple[str, bytes] | None:
        try:
            types = (await self.run("wl-paste", "--list-types")).decode().splitlines()
        except UserError as exc:
            if "Nothing is copied" in str(exc) or "No selection" in str(exc):
                return None
            raise
        selected = next((t for t in (MIME, "text/plain", "UTF8_STRING") if t in types), None)
        if selected is None:
            raise UserError("Clipboard is not text; cannot safely preserve it. Use clipboard output or disable restoration.")
        return selected, await self.run("wl-paste", "--no-newline", "--type", selected)

    async def copy(self, text: str) -> str:
        async with self._lock:
            await self._set(text.encode())
        return "Copied to clipboard."

    async def deliver(self, text: str, target: dict, config: dict) -> str:
        if config["output"] == "clipboard":
            return await self.copy(text)
        if not target.get("address"):
            return "Not pasted: no focused window at recording start. Transcript retained; use Copy."
        async with self._lock:
            if await self.layer_active():
                return "Not pasted: an interactive desktop overlay is open. Transcript retained; close it and use Copy."
            if (await self.focused())["address"] != target["address"]:
                return "Not pasted: focus changed since recording started. Transcript retained; use Copy."
            previous = await self._previous() if config["restore_clipboard"] else None
            data = (text + (" " if config["append_space"] else "")).encode()
            await self._set(data)
            owner = self.owner
            try:
                if await self.layer_active():
                    return "Not pasted: an interactive desktop overlay opened before insertion. Transcript retained; use Copy."
                # The final focus check is immediately before the address-targeted dispatch.
                if (await self.focused())["address"] != target["address"]:
                    return "Not pasted: focus changed before insertion. Transcript retained; use Copy."
                shortcut = config["paste_shortcut"]
                if shortcut == "auto":
                    shortcut = "CTRL+SHIFT+V" if target.get("class", "").lower() in TERMINALS else "CTRL+V"
                else:
                    shortcut = shortcut.upper()
                modifiers, key = shortcut.rsplit("+", 1)
                result = await self.dispatch(target, modifiers.replace("+", " "), "Insert" if key == "INSERT" else key.lower())
                if result != "ok":
                    reason = "focus changed" if result == "blocked-focus" else "an interactive overlay opened"
                    return f"Not pasted: {reason} before insertion. Transcript retained; use Copy."
                await asyncio.sleep(0.3)
                return "Paste shortcut sent to the original window."
            finally:
                if config["restore_clipboard"] and owner is self.owner and owner.returncode is None:
                    current = await self.run("wl-paste", "--no-newline", "--type", MIME)
                    if current == data and owner.returncode is None:
                        if previous is not None:
                            await self._set(previous[1], previous[0])
                        else:
                            await self.run("wl-copy", "--clear")

    async def close(self) -> None:
        if self.owner and self.owner.returncode is None:
            self.owner.terminate()
            await self.owner.wait()
