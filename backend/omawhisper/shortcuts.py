"""Live Hyprland 0.56 Lua bindings without editing compositor configuration."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shlex

from .config import UserError, shortcut_parts
from .process import run


def lua_string(value: str) -> str:
    # Byte escapes are valid Lua and cannot introduce executable source.
    return '"' + "".join(f"\\{byte:03d}" for byte in value.encode()) + '"'


class Shortcuts:
    def __init__(self, runner=run):
        self.run = runner
        self.shortcut = None
        self.activation = None
        self.script = Path(__file__).resolve().parents[2] / "scripts/omawhisper"
        self.description = f"Omawhisper ({os.getpid()})"
        self.slot = f"__omawhisper_{os.getpid()}"

    async def _eval(self, source: str) -> None:
        result = await self.run("hyprctl", "eval", source)
        if result.strip() != b"ok":
            raise UserError(f"Hyprland shortcut update failed: {result.decode(errors='replace')[:1000]}")

    def _collision(self, bindings: list, shortcut: str, activation: str) -> None:
        mask, key = shortcut_parts(shortcut)
        for binding in bindings:
            if binding.get("description", "").startswith(self.description):
                continue
            same_key = str(binding.get("key", "")).upper() == key
            # A keycode binding may alias this keysym: reject conservatively.
            keycode = bool(binding.get("keycode"))
            if same_key or keycode:
                if (binding.get("modmask") == mask or binding.get("ignore_mods")
                        or (activation == "hold" and binding.get("release"))):
                    raise UserError(f"Shortcut conflicts with an existing Hyprland binding: {binding.get('description') or binding.get('key') or 'keycode'}.")

    async def apply(self, shortcut: str, activation: str, force: bool = False) -> None:
        if not force and (shortcut, activation) == (self.shortcut, self.activation):
            return
        bindings = json.loads(await self.run("hyprctl", "-j", "binds"))
        self._collision(bindings, shortcut, activation)
        # Never remove a combo that has acquired somebody else's binding.
        if self.shortcut:
            self._collision(bindings, self.shortcut, self.activation)
        actions = [("start" if activation == "hold" else "toggle", False)]
        if activation == "hold":
            actions.append(("stop", True))
        definitions = []
        for action, release in actions:
            request = {"action": action}
            if activation == "hold" and not release:
                request["hotkey"] = True
            command = shlex.join([str(self.script), "request", json.dumps(request)])
            options = f"description={lua_string(self.description)},repeating=false"
            dispatcher = f"hl.dsp.exec_cmd({lua_string(command)})"
            if activation == "hold":
                held = f"_G.{self.slot}_held"
                if release:
                    dispatcher = f"function() if {held} then {held}=false;hl.exec_cmd({lua_string(command)}) end end"
                else:
                    dispatcher = f"function() if not {held} then {held}=true;hl.exec_cmd({lua_string(command)}) end end"
            if release:
                options += ",release=true,ignore_mods=true,non_consuming=true"
            definitions.append(
                f"hl.bind({lua_string(shortcut)},{dispatcher},{{{options}}})"
            )
        # One Lua IPC evaluation prevents compositor input between removal and registration.
        # On failure, restore the old definitions rather than leaving an unbound shortcut.
        restore = getattr(self, "_definitions", [])
        remove_old = f"hl.unbind({lua_string(self.shortcut)});" if self.shortcut else ""
        source = (
            f"local old=_G.{self.slot} or {{}};{remove_old}"
            "local new={};local ok,err=pcall(function() "
            + ";".join(f"new[#new+1]=assert({d},'Failed to register shortcut')" for d in definitions)
            + f" end);if not ok then hl.unbind({lua_string(shortcut)});"
            + ";".join(restore)
            + f";error(err) end;_G.{self.slot}=new;_G.{self.slot}_held=false"
        )
        await self._eval(source)
        self.shortcut, self.activation, self._definitions = shortcut, activation, definitions

    async def held(self) -> bool:
        value = (await self.run("hyprctl", "repl", f"return _G.{self.slot}_held == true")).strip()
        if value not in (b"true", b"false"):
            raise UserError("Cannot verify whether the dictation shortcut is still held.")
        return value == b"true"

    async def close(self) -> None:
        if self.shortcut:
            bindings = json.loads(await self.run("hyprctl", "-j", "binds"))
            self._collision(bindings, self.shortcut, self.activation)
            await self._eval(f"hl.unbind({lua_string(self.shortcut)});_G.{self.slot}=nil;_G.{self.slot}_held=nil")
            self.shortcut = None

    async def events(self, reapply, report) -> None:
        signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        if not signature or not runtime:
            raise UserError("Hyprland session environment is unavailable.")
        path = Path(runtime) / "hypr" / signature / ".socket2.sock"
        while True:
            writer = None
            try:
                reader, writer = await asyncio.open_unix_connection(path)
                while line := await reader.readline():
                    if line.startswith(b"configreloaded>>"):
                        await reapply()
                raise UserError("Hyprland event socket disconnected; retrying.")
            except asyncio.CancelledError:
                raise
            except (OSError, UserError, ValueError) as exc:
                report(exc)
                await asyncio.sleep(2)
            finally:
                if writer:
                    writer.close()
                    await writer.wait_closed()
