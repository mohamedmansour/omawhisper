# Omawhisper

Local voice dictation for the **Omarchy shell**. Press a shortcut, speak, and
press it again: your words are inserted into the application you were using.
The bar microphone becomes a live, audio-driven level meter while recording.
Click it for a keyboard-first settings popover.

Inspired by [OpenSuperWhisper](https://github.com/Starmel/OpenSuperWhisper),
with the modular bar/panel/view structure and arrow-key navigation of
[Omafinance](https://github.com/mohamedmansour/omafinance). This is an independent
Linux implementation, not a port of the macOS UI or its Apple-only engines.

## Install

Requires Omarchy's Quickshell-based shell, Hyprland with Lua configuration,
PipeWire, `wl-clipboard`, Python and systemd user services.

```bash
cd ~/Projects/omawhisper
./scripts/install.sh
```

The installer creates an isolated `~/.local/share/omawhisper/venv`, installs the local speech engines,
links the plugin into `~/.config/omarchy/plugins/mohamedmansour.whisper`,
and enables a graphical-session user service. It does not modify packaged
Omarchy files or overwrite existing keyboard bindings.

Choose initial placement with `./scripts/install.sh --section left` (also
`center` or `right`). After installation, placement is adjustable in General
settings or with:

```bash
omarchy bar move mohamedmansour.whisper --section center
```

If your system Python is too new for a speech engine's binary wheels, select
an installed supported interpreter before creating the virtual environment:

```bash
OMAWHISPER_PYTHON=python3.12 ./scripts/install.sh
```

Click the microphone, open **Models**, download a model, then choose **Load**.
Downloads are explicit; simply enabling the plugin does not download weights.
The microphone is never opened until you activate dictation.

## Dictation

**Push to talk is on by default.** Hold **Super + Alt + V** while speaking,
then release to transcribe and insert your words into the focused application.
Turn off **General > Push to talk** to press once to start and again to stop.
The default deliberately avoids Omarchy's clipboard-manager shortcut.

Focus a text input before starting. The settings panel is not opened by
recording, and closes if recording starts while it is visible. Recognition
is local; the model stays in memory between dictations. A small pulsing dot
indicates loading, downloading or transcription; the five recording bars follow
actual microphone volume rather than a decorative animation.

Pasting targets the original window and is refused if focus changes. The
transcript remains available in the popover for copying. Automatic paste uses
Ctrl+V, or Ctrl+Shift+V for known terminal applications. Applications with
unusual paste shortcuts can use the explicit paste setting or clipboard-only
output. Like other desktop dictation tools, this depends on the target
application accepting clipboard paste; it cannot identify every application's
individual input field or safely dictate into password fields.

Middle-click the microphone to cancel. The CLI also supports cancellation:

```bash
./scripts/omawhisper request '{"action":"cancel"}'
```

## Settings and keyboard navigation

| Tab | Settings |
| --- | --- |
| General | Activation shortcut, push-to-talk switch (on by default), paste/clipboard output, left/middle/right bar placement, clipboard preservation, trailing space |
| Models | Download, load, unload and remove models; add compatible Hugging Face repositories, local models or custom engines |
| Audio | Microphone, language, translation to English, silence filtering, recording time limit |
| Advanced | Vocabulary prompt, greedy/beam search, temperature, no-speech threshold, blank suppression, timestamps, CPU precision/threads, paste shortcut, opt-in history |
| History | Copy the latest transcript, copy saved transcripts and clear history |

Use **Up/Down** to move through rows. On the tab strip, **Left/Right** changes
tabs; on a setting, it adjusts its value. **Enter** activates a row or edits a
text/number field. **Tab/Shift+Tab** changes tabs, except in an editor where it
saves and moves to the next/previous row. **Escape** cancels editing or closes
the popover. In Models, **X** asks to remove the selected model; **Enter**
confirms. Mouse and keyboard use the same controls.

Shortcuts are entered as chords such as `SUPER+ALT+V`. Conflicting bindings
are rejected, not silently replaced. Shortcut changes apply immediately and
are re-registered when Hyprland reloads its configuration.

The portable recognition settings follow OpenSuperWhisper's
[`AppPreferences.swift`](https://github.com/Starmel/OpenSuperWhisper/blob/master/OpenSuperWhisper/Utils/AppPreferences.swift):
greedy decoding by default, beam size 5 when enabled, temperature 0, no-speech
threshold 0.6, blank suppression on and timestamps off. Linux-specific
differences are intentional: language detection is automatic by default,
recording retention is replaced by opt-in text history, and
Apple's modifier-only/Fn triggers and Asian-language autocorrection library
are not emulated. Translation is an additional Whisper capability.

## Models and extensibility

- **Whisper** runs through `faster-whisper` on CPU. Select a catalog model,
  enter any compatible Hugging Face CTranslate2 model repository, or use
  an existing local CTranslate2 model directory.
- **Parakeet** runs locally through `onnx-asr` and ONNX Runtime. Use a
  supported ONNX export, not unconverted NeMo or MLX weights. For local ONNX
  directories, also specify the architecture (for example
  `nemo-parakeet-tdt-0.6b-v3`).
- **Custom command** connects other installed local engines. Supply an
  argument array containing `{audio}`; the executable receives a temporary
  16 kHz mono WAV and prints the recognized text to stdout.

For example, a custom adapter may have arguments:

```json
["/absolute/path/to/my-transcriber", "--audio", "{audio}"]
```

Arguments are not evaluated by a shell. An adapter is a program you explicitly
trust and install; its network and data-handling behavior is its own.
Whisper.cpp `.bin` files, MLX checkpoints and arbitrary neural-network weights
are **not interchangeable** with the bundled engines. To use another format,
install its matching runtime and connect it through a custom adapter.

Translation, prompting, beam search and Whisper VAD are engine-specific;
unsupported controls are disabled for other engines. Model licenses and
download sizes are set by their publishers.

## Privacy and storage

Audio and text are not sent to a transcription API. Model downloads contact
Hugging Face; custom adapters can have their own behavior. Temporary recordings
are deleted after processing or cancellation. **History is off by default**;
the latest transcript is held in memory for recovery. Enable history only if
you want transcripts written to disk. Disabling history stops new saves but
does not delete older history; use **Clear saved history** to delete it explicitly.

| Location | Purpose |
| --- | --- |
| `~/.config/omawhisper/config.json` | Dictation settings |
| `~/.config/omawhisper/models.json` | Custom model definitions |
| `~/.local/share/omawhisper/models/` | Downloaded weights |
| `~/.local/share/omawhisper/venv/` | Isolated Python runtime and speech engines |
| `~/.local/state/omawhisper/history.json` | Opt-in transcript history |
| `$XDG_RUNTIME_DIR/omawhisper/control.sock` | Private local service IPC |

XDG directory overrides are honored. Clipboard-only output intentionally
replaces the clipboard. Clipboard preservation applies to paste mode, and
does not overwrite a clipboard changed by another application. Your desktop's
clipboard manager may independently retain pasted dictation.

## Architecture

```text
manifest.json                  Omarchy bar-widget entry point
src/BarWidget.qml              Microphone / audio meter and popup lifecycle
src/Panel.qml                  Keyboard cursor, tabs and composition
src/WhisperService.qml         Streaming JSON IPC and serialized requests
src/PreferencesView.qml        Data-driven settings
src/PreferenceRow.qml          Shared keyboard/pointer/editor behavior
src/ModelsView.qml             Catalog and custom model form
src/HistoryView.qml            Transcript recovery and opt-in history
src/Settings.js                Pure setting descriptions and navigation
backend/omawhisper/            Capture, engines, shortcuts, output and daemon
scripts/                      Launcher and user-service installer
tests/                        Standard-library Python and Node tests
```

The UI never performs inference. A long-running local daemon owns capture,
model lifecycle and focus-safe output; its Unix socket streams state updates
to every bar instance. Download and recognition work stays off the IPC loop.
This keeps the bar responsive even while a large model loads.

## Service and development

```bash
systemctl --user status omawhisper
journalctl --user -u omawhisper -n 50 --no-pager
./scripts/omawhisper request '{"action":"status"}'
./scripts/omawhisper watch
omarchy-shell mohamedmansour.whisper toggle

PYTHONPATH=backend python -m unittest discover -s tests -p 'test_backend*.py'
node --test tests/ui.test.js
omarchy plugin validate .
```

After backend edits, restart `systemctl --user restart omawhisper`.
Plugin edits use Omarchy's normal hot reload; if Qt retains a cached component,
run `omarchy restart shell`.

## Remove

```bash
systemctl --user disable --now omawhisper.service
omarchy plugin disable mohamedmansour.whisper
```

To remove this local installation's links and service definition:

```bash
unlink ~/.config/omarchy/plugins/mohamedmansour.whisper
rm ~/.config/systemd/user/omawhisper.service
systemctl --user daemon-reload
```

Downloaded models, settings and history are left in place. The project
directory is not deleted.

## License

MIT. Speech models and third-party runtimes retain their respective licenses.
