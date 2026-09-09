# Omawhisper

Local voice dictation for the **Omarchy shell**. Hold a shortcut, speak, and
release: your words are inserted into the application you were using.
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

The installer creates isolated Python environments under `~/.local/share/omawhisper/`,
installs Whisper in `venv` and the Parakeet CPU worker in `venv-onnx-cpu`,
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

Click the microphone and open **Models**. The **Processing device** radio buttons at
the top offer **Auto**, **GPU** and **CPU**. Choose a device, then click a model.
That single click installs any missing runtime, downloads the model if needed,
and activates it. There is no Apply button or separate download/load step.

Progress appears on the model's row; click it again to cancel. Settings are saved
only after the model loads successfully. The activation shortcut and push-to-talk
controls remain at the top of **General**, and apply independently.
**GPU runtime installation is built into Models; no terminal command is needed
after the initial plugin install.**
Downloads are explicit; simply enabling the plugin does not download weights.
Selecting a model never opens the microphone, installs a GPU driver, or restarts the daemon.

## Dictation

**Push to talk is on by default.** Hold **Super + Alt + V** while speaking,
then release to transcribe and insert your words into the focused application.
Turn off **Push to talk** in **General** to press once
to start and again to stop.
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
| General | Activation shortcut, push-to-talk, paste/clipboard output, bar placement, clipboard preservation, trailing space |
| Models | Auto/GPU/CPU selector and one-click model download/activation; inline progress, cancellation and removal |
| Audio | Microphone, language, translation to English, silence filtering, recording time limit |
| Advanced | Add custom models, unload model memory; vocabulary prompt, decoding settings, Whisper precision, CPU threads, paste shortcut, opt-in history |
| History | Copy the latest transcript, copy saved transcripts and clear history |

Use **Up/Down** to move through rows. On the tab strip, **Left/Right** changes
tabs; on a setting, it adjusts its value. **Enter** activates a row or edits a
text/number field. **Tab/Shift+Tab** changes tabs, except in an editor where it
saves and moves to the next/previous row. **Escape** cancels editing or closes
the popover. In Models, **X** asks to remove the selected model; **Enter**
confirms. Mouse and keyboard use the same controls.

Shortcuts are captured by pressing the combination, including punctuation such as
**Super + .**, **Super + /**, or **Ctrl + ;**; **Enter** saves it.
Symbol keys are displayed as typed and stored using Hyprland key names
(for example, `SUPER+PERIOD`). Conflicting bindings
are rejected, not silently replaced. Shortcut changes apply when saved in General and
are re-registered when Hyprland reloads its configuration.

The portable recognition settings follow OpenSuperWhisper's
[`AppPreferences.swift`](https://github.com/Starmel/OpenSuperWhisper/blob/master/OpenSuperWhisper/Utils/AppPreferences.swift):
greedy decoding by default, beam size 5 when enabled, temperature 0, no-speech
threshold 0.6, blank suppression on and timestamps off. Linux-specific
differences are intentional: language detection is automatic by default,
recording retention is replaced by opt-in text history, and
Apple's modifier-only/Fn triggers and Asian-language autocorrection library
are not emulated. Translation is an additional Whisper capability.

## Acceleration

**Models > Processing device** is keyboard-accessible like the other settings:

- **Auto** (default) prefers supported NVIDIA CUDA acceleration, otherwise CPU.
- **GPU** requires a supported GPU and runtime; it never silently switches to CPU.
- **CPU** keeps recognition on the processor even when a GPU is available.

Choose a device with **Left/Right**, move down to a model, and press **Enter**.
Merely cycling the device never starts an installation or switches models.
Missing runtimes and weights are downloaded automatically when selecting a model.
Changing Advanced precision
unloads it; load it again in Models or let the next dictation load it on demand. The settings header reports
the loaded device and effective Whisper precision, including why Auto chose CPU.
The status IPC response exposes the same information in `runtime`. The microphone
selection is independent of the processing device.

**Advanced > Whisper precision > Auto** chooses INT8 on supported CPUs and FP16
on supported GPUs, otherwise FP32. Explicit precision settings are validated
against the device rather than silently converted to a different precision family.
Existing saved precision preferences are preserved. `int8_float16` is an explicit
lower-memory GPU option; FP16 modes are not supported on ordinary CPUs.

CUDA initialization, missing libraries, unsupported model operators and
out-of-memory errors are surfaced rather than disguised as successful GPU loads.
If CTranslate2 cannot enumerate a usable CUDA device, Auto reports that CUDA is
unavailable and uses CPU; that can mean absent hardware or a driver/runtime
problem, not necessarily that the machine has no GPU. Once a CUDA device has
been selected, initialization errors are surfaced; select CPU while resolving them.
AMD, Intel and virtual display GPUs are not CUDA devices; they use CPU with the
bundled engines. Custom commands manage their own devices; this selection does
not override an adapter's hardware settings.

Recording behavior is unchanged: recognition starts after you release the
shortcut and processes the complete recording for context.

ONNX loading is checked against a conservative host-RAM estimate, including
temporary weights and session initialization. The catalog budgets about **1.5 GiB
for Parakeet INT8** and **5 GiB for Parakeet FP32**; GPU inference still needs host
RAM to initialize. Models marks entries that exceed installed RAM and checks
currently available RAM again before loading. Swap is not treated as available
model capacity. If a worker is nevertheless killed under memory pressure, the
daemon stays running and reports the failure instead of losing the settings panel.

### NVIDIA setup for Whisper

On the physical Linux machine, first install the NVIDIA driver appropriate for
its GPU and kernel. Then use **Models > Processing device** to select GPU or
Auto and a Whisper model. It installs the optional CUDA libraries as needed.
For unattended initial provisioning, the equivalent optional command is:

```bash
./scripts/install.sh --cuda
```

This installs CUDA 12 cuBLAS into Omawhisper's isolated environment and requires
CTranslate2 4.6.3 or newer, where cuDNN is optional. The inference worker makes
these libraries visible before Python starts, including under systemd. Newly
installed libraries are picked up by a new worker without restarting the daemon.
It does not install or replace the system NVIDIA driver. Compatible CTranslate2
GPU wheels must be available for the machine's architecture and Python version.
If compatible libraries are already installed system-wide, the regular installer
can also use them.

Choose **GPU** in Models, then click a Whisper model.
Its row should show **Active (GPU)**. For larger models, use
**Advanced > Whisper precision > int8_float16** to reduce GPU memory use.

### NVIDIA setup for Parakeet

Parakeet runs in its own persistent ONNX worker, separate from Whisper. On a
physical machine with a supported NVIDIA GPU and a working driver compatible
with CUDA 12, choose **GPU** (or **Auto**) under **Models > Processing device**,
click **Parakeet TDT v3 FP32** once to download and load it. Loading installs
the CUDA runtime as needed before applying your model and device choices.
There is no terminal prerequisite after installing the plugin.

For unattended initial provisioning, optionally install both engines' GPU
dependencies first:

```bash
./scripts/install.sh --cuda
```

Check that its row reports **Active (GPU)**. FP32 weights require a larger download and more
memory than the INT8 CPU model; allow several gigabytes for weights, CUDA/cuDNN
packages and GPU working memory. Actual VRAM needs depend on recording length.
Whisper precision settings do not convert Parakeet weights.

The existing INT8 Parakeet catalog models are CPU-oriented. **Auto** uses CPU
with a displayed reason for these weights, or when a usable CUDA runtime is
unavailable. Explicit **GPU** requires compatible floating-point weights and a
working CUDA worker; otherwise it fails with an actionable error. A GPU load
must not silently run the encoder on CPU. CUDA initialization, operator and
out-of-memory failures are reported rather than hidden.

Applying GPU (or `--cuda`) adds `venv-onnx-cuda` with `onnx-asr` 0.12 and
`onnxruntime-gpu[cuda,cudnn]` 1.24.2–1.24.x, including CUDA 12 and cuDNN 9
libraries. The worker exposes its own NVIDIA library directories before Python
starts, also under systemd. These packages do not install the NVIDIA driver.
The version range is deliberate: onnx-asr 0.12 excludes ORT 1.25.x and 1.26.0,
and upstream ORT 1.27+ GPU wheels default to CUDA 13. Do not upgrade the GPU
runtime independently of these constraints.

Compatible binary wheels must exist for **all** dependencies. The selected
Linux GPU ORT wheels target x86-64 and glibc 2.27/2.28 or newer; Python 3.12 is
a useful supported choice via `OMAWHISPER_PYTHON=python3.12`. ARM/Jetson or
unsupported Python versions need a separately supported runtime rather than an
assumption that these wheels work. Changing `OMAWHISPER_PYTHON` only affects
new environments, not existing ones.

The separation avoids an upstream packaging conflict: `faster-whisper` requires
CPU `onnxruntime`, while GPU inference requires `onnxruntime-gpu`. Their files
overlap, so they must not share an environment. The installer checks for the
wrong or coinstalled distributions before installing, and never uninstalls one
to overwrite it with the other. Do not combine the `whisper`/`engines` extras
with `parakeet-cuda` in a manual installation. If a guard rejects an environment,
recreate only the named virtual environment, preserving settings and downloaded
models, then rerun the appropriate installer command.

Every install provisions the CPU worker. Running the normal installer after
`--cuda` leaves the GPU environment available; it does not remove GPU support
or reset saved device/model preferences. Older CPU ONNX packages in the main
environment can remain safely. Whisper's small silence detector still runs on
CPU even when its speech-recognition model runs on GPU.

The persistent worker reuses its loaded model across dictations. Recognition
still starts after recording finishes; this does not add streaming transcription.
Automated tests exercise the isolated-worker contracts and mocked installation;
real Whisper and Parakeet CPU inference has also been verified over workers.
Real NVIDIA GPU execution has not been verified in the current VM.

References: [CTranslate2 release notes](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md),
[faster-whisper dependencies](https://github.com/SYSTRAN/faster-whisper/blob/master/requirements.txt),
and [ONNX Runtime CUDA requirements and Python extras](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Models and extensibility

- **Whisper** runs through `faster-whisper` on CPU or NVIDIA CUDA. Select a catalog model,
  enter any compatible Hugging Face CTranslate2 model repository, or use
  an existing local CTranslate2 model directory.
- **Parakeet** runs locally through `onnx-asr` and ONNX Runtime on CPU or NVIDIA
  CUDA, using isolated workers. Choose FP32 catalog weights for GPU. Use a
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
| `~/.local/share/omawhisper/venv/` | Daemon and Whisper worker runtime, including CPU ONNX for Whisper VAD |
| `~/.local/share/omawhisper/venv-onnx-cpu/` | Dedicated Parakeet CPU worker runtime |
| `~/.local/share/omawhisper/venv-onnx-cuda/` | Optional Parakeet CUDA worker runtime and NVIDIA libraries |
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
src/ProcessingView.qml         Inline processing control and runtime progress
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
Inference lives in persistent subprocesses: Whisper uses the main environment's
Python, and Parakeet uses the selected CPU or CUDA environment's Python.
The daemon's worker communication uses only the standard library; CPU/GPU ONNX
distributions never need to coexist in one interpreter. Workers retain their
loaded model between requests and report the actual device/provider rather than
assuming that a GPU package implies CUDA. Applying/loading runs dependency provisioning as
a managed background child, then downloads/loads the chosen model and applies
the requested processing settings without changing the activation shortcut or mode.

## Service and development

`scripts/install-engines.sh [--cuda]` is the dependency-only entry point used by
the backend's setup flow. It provisions the selected Python environments only:
it never restarts the daemon, touches desktop configuration, or installs an
NVIDIA driver. `scripts/install.sh` wraps it, then validates the plugin and
installs/enables the desktop plugin and user service.

For an offline, read-only readiness probe, use
`scripts/install-engines.sh --check [--cuda]`. Exit **0** means the requested
environments match the current manifest; **1** means provisioning is needed;
**2 or higher** means invalid arguments or an inspection/runtime error. Check
mode never invokes pip, creates environments, downloads packages, or writes
markers. It checks distribution collisions using Python's local metadata.

Each environment's `.omawhisper-ready` JSON stores `manifest_sha256` (SHA-256
of `pyproject.toml`) and its installed `extras`. It is atomically published only
after pip succeeds, collision checks pass and engine imports succeed; failed
updates invalidate the old marker. A current main-environment `whisper,cuda`
marker satisfies a CPU-only check too. `--cuda` additionally requires the CUDA
extras in the main environment and a separate GPU ONNX environment, alongside
the always-required CPU ONNX environment. Current environments are reused
without pip; CPU-only provisioning leaves the GPU environment intact.
These markers certify dependency provisioning, not usable GPU hardware:
the inference worker still validates CUDA when loading a model.

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
