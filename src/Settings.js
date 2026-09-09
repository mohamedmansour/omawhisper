function choices(values) {
    return values.map(function (v) {
        return typeof v === "string" ? { value: v, label: v } : v;
    });
}

function general(config) {
    return [
        { key: "shortcut", title: "Activation shortcut", help: "Enter a chord, for example SUPER+ALT+V. Existing bindings are never replaced.", kind: "text" },
        { key: "activation", title: "Recording mode", help: "Toggle twice, or hold the shortcut and release to transcribe.", kind: "choice", options: choices(["toggle", "hold"]) },
        { key: "output", title: "Insert text", help: "Paste into the original focused window, or only copy to the clipboard.", kind: "choice", options: choices([{ value: "paste", label: "Paste at cursor" }, { value: "clipboard", label: "Clipboard only" }]) },
        { key: "bar_section", title: "Bar position", help: "Place the microphone on the left, middle or right.", kind: "choice", options: choices([{ value: "left", label: "Left" }, { value: "center", label: "Middle" }, { value: "right", label: "Right" }]) },
        { key: "restore_clipboard", title: "Preserve clipboard", help: "Restore the previous clipboard after pasting, unless it has changed.", kind: "toggle" },
        { key: "append_space", title: "Trailing space", help: "Add a space after each dictation to keep typing naturally.", kind: "toggle" }
    ];
}

function audio(devices) {
    return [
        { key: "device", title: "Microphone", help: "Uses the system default input unless you choose a device.", kind: "choice", options: (devices || []).map(function (d) { return { value: d.id, label: d.name }; }) },
        { key: "language", title: "Language", help: "auto, or an ISO language code such as en, fr, de, ja or ar. Whisper only.", kind: "text", whisperOnly: true },
        { key: "translate", title: "Translate to English", help: "Whisper translation, not a separate cloud service.", kind: "toggle", whisperOnly: true },
        { key: "vad", title: "Filter silence", help: "Whisper voice activity detection reduces silence hallucinations.", kind: "toggle", whisperOnly: true },
        { key: "max_duration", title: "Recording limit", help: "Automatically stop and transcribe after this many seconds.", kind: "number", from: 5, to: 600, step: 5 }
    ];
}

function advanced() {
    return [
        { key: "initial_prompt", title: "Vocabulary / initial prompt", help: "Names, specialist terms or a short style hint for Whisper.", kind: "text", whisperOnly: true },
        { key: "use_beam_search", title: "Beam search", help: "Search several hypotheses instead of greedy decoding.", kind: "toggle", whisperOnly: true },
        { key: "beam_size", title: "Beam size", help: "Larger searches trade speed for accuracy. Whisper only.", kind: "number", from: 1, to: 10, step: 1, whisperOnly: true },
        { key: "temperature", title: "Temperature", help: "Lower is more deterministic; higher allows more variation.", kind: "number", from: 0, to: 1, step: 0.1, whisperOnly: true },
        { key: "no_speech_threshold", title: "No-speech threshold", help: "Skip segments classified as silence above this probability.", kind: "number", from: 0, to: 1, step: 0.1, whisperOnly: true },
        { key: "suppress_blank", title: "Suppress blank audio", help: "Suppress blank tokens at the beginning of decoding.", kind: "toggle", whisperOnly: true },
        { key: "show_timestamps", title: "Include timestamps", help: "Prefix transcript segments with their start and end times.", kind: "toggle", whisperOnly: true },
        { key: "compute_type", title: "Whisper CPU precision", help: "int8 uses less memory; float32 uses full precision.", kind: "choice", options: choices(["int8", "float32"]), whisperOnly: true },
        { key: "threads", title: "CPU threads", help: "Applied the next time the model is loaded.", kind: "number", from: 1, to: 32, step: 1 },
        { key: "paste_shortcut", title: "Paste shortcut", help: "Auto uses Ctrl+Shift+V for terminals and Ctrl+V elsewhere.", kind: "choice", options: choices([{ value: "auto", label: "Automatic" }, { value: "ctrl+v", label: "Ctrl+V" }, { value: "ctrl+shift+v", label: "Ctrl+Shift+V" }]) },
        { key: "history", title: "Save transcript history", help: "Off by default. Disabling stops saving; use History to clear older transcripts. Audio is never retained.", kind: "toggle" }
    ];
}

function moveCursor(cursor, delta, count) {
    return Math.max(-1, Math.min(count - 1, cursor + delta));
}

function cycle(options, value, direction) {
    if (!options.length)
        return value;
    var index = options.findIndex(function (option) { return option.value === value; });
    return options[(Math.max(0, index) + direction + options.length) % options.length].value;
}

if (typeof module !== "undefined")
    module.exports = { general: general, audio: audio, advanced: advanced, moveCursor: moveCursor, cycle: cycle };
