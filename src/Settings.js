function choices(values) {
    return values.map(function (v) {
        return typeof v === "string" ? { value: v, label: v } : v;
    });
}

function general() {
    return activationRows().concat([
        { key: "output", title: "Insert text", help: "Paste into the original focused window, or only copy to the clipboard.", kind: "choice", options: choices([{ value: "paste", label: "Paste at cursor" }, { value: "clipboard", label: "Clipboard only" }]) },
        { key: "bar_section", title: "Bar position", help: "Place the microphone on the left, middle or right.", kind: "choice", options: choices([{ value: "left", label: "Left" }, { value: "center", label: "Middle" }, { value: "right", label: "Right" }]) },
        { key: "restore_clipboard", title: "Preserve clipboard", help: "Restore the previous clipboard after pasting, unless it has changed.", kind: "toggle" },
        { key: "append_space", title: "Trailing space", help: "Add a space after each dictation to keep typing naturally.", kind: "toggle" }
    ]);
}

function processingRows() {
    return [
        { key: "acceleration", title: "Processing device", help: "Auto uses a compatible NVIDIA GPU when available, otherwise CPU. GPU requires an NVIDIA driver.", kind: "choice", options: choices([{ value: "auto", label: "Auto" }, { value: "gpu", label: "GPU" }, { value: "cpu", label: "CPU" }]) }
    ];
}

function activationRows() {
    return [
        { key: "shortcut", title: "Activation shortcut", help: "Press Enter, then your keyboard chord. Enter saves; Escape cancels. Existing system bindings are never replaced.", kind: "shortcut" },
        { key: "activation", title: "Push to talk", help: "On: hold the shortcut to speak, release to insert text. Off: press once to start and again to stop.", kind: "toggle", enabledValue: "hold", disabledValue: "toggle" }
    ];
}

function setupDraft(config) {
    var defaults = { acceleration: "auto", model: "whisper-base" };
    if (!config)
        return defaults;
    Object.keys(defaults).forEach(function (key) {
        if (config[key] !== undefined)
            defaults[key] = config[key];
    });
    return defaults;
}

function isBusy(phase) {
    return ["installing", "downloading", "loading", "transcribing"].indexOf(phase) !== -1;
}

function modelCompatibility(model) {
    if (!model)
        return { gpu: false, label: "Model unavailable", detail: "Choose a model from the catalog." };
    if (model.engine === "command")
        return { gpu: true, external: true, label: "Hardware managed by adapter", detail: "This custom command manages its own runtime and hardware. Only use trusted local adapters." };
    if (model.engine === "whisper" || model.engine === "faster-whisper")
        return { gpu: true, label: "CPU / NVIDIA GPU", detail: "Runs locally on CPU or CUDA. Auto chooses a usable GPU, otherwise CPU." };
    var quantization = model.quantization;
    if (quantization === undefined)
        quantization = model.id === "parakeet-v3-fp32" ? "" : "int8";
    var gpu = model.gpu_compatible !== undefined ? model.gpu_compatible
        : ["", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"].indexOf(String(quantization || "").toLowerCase()) !== -1;
    return {
        gpu: gpu,
        label: gpu ? "CPU / NVIDIA GPU · floating-point ONNX" : "CPU only · quantized ONNX",
        detail: gpu ? "Floating-point ONNX weights support CUDA. These weights can require several GB of disk and memory."
            : "Quantized ONNX weights use CPU, including in Auto. For GPU choose Parakeet FP32 or Whisper."
    };
}

function setupModels(models, draft) {
    return (models || []).map(function (model) {
        var compatibility = modelCompatibility(model);
        return Object.assign({}, model, {
            compatibility: compatibility,
            memoryDetail: model.memory_error || (model.estimated_ram_bytes
                ? "Estimated host RAM to load: " + (model.estimated_ram_bytes / 1073741824).toFixed(1) + " GiB" : ""),
            compatible: !model.memory_error && (draft.acceleration !== "gpu" || compatibility.gpu)
        });
    });
}

function modelStatus(model, config, runtime, device) {
    if (model.loaded)
        return runtime && runtime.device === "cuda" ? "Active (GPU)"
            : runtime && runtime.device === "cpu" ? "Active (CPU)" : "Active";
    if (model.progress === -1)
        return "Downloading...";
    if (model.progress > 0 && model.progress < 100)
        return Math.round(model.progress) + "%";
    if (model.memory_error)
        return "Needs more RAM";
    if (device === "gpu" && !modelCompatibility(model).gpu)
        return "CPU only";
    if (model.installed)
        return model.id === config.model ? "Selected" : "Ready";
    return model.engine === "command" || String(model.source || "").startsWith("/") ? "Unavailable" : "Download";
}

function modelProgress(phase) {
    return { installing: "Installing...", downloading: "Downloading...", loading: "Loading..." }[phase] || "Preparing...";
}

var shortcutSymbols = {
    "!": "EXCLAM", "\"": "QUOTEDBL", "#": "NUMBERSIGN", "$": "DOLLAR",
    "%": "PERCENT", "&": "AMPERSAND", "'": "APOSTROPHE", "(": "PARENLEFT",
    ")": "PARENRIGHT", "*": "ASTERISK", "+": "PLUS", ",": "COMMA",
    "-": "MINUS", ".": "PERIOD", "/": "SLASH", ":": "COLON",
    ";": "SEMICOLON", "<": "LESS", "=": "EQUAL", ">": "GREATER",
    "?": "QUESTION", "@": "AT", "[": "BRACKETLEFT", "\\": "BACKSLASH",
    "]": "BRACKETRIGHT", "^": "ASCIICIRCUM", "_": "UNDERSCORE", "`": "GRAVE",
    "{": "BRACELEFT", "|": "BAR", "}": "BRACERIGHT", "~": "ASCIITILDE"
};

function shortcutChord(modifiers, key) {
    var parts = [];
    if (modifiers.super)
        parts.push("SUPER");
    if (modifiers.ctrl)
        parts.push("CTRL");
    if (modifiers.alt)
        parts.push("ALT");
    if (modifiers.shift)
        parts.push("SHIFT");
    return parts.length && key ? parts.concat([shortcutSymbols[key] || key]).join("+") : "";
}

function shortcutLabel(shortcut) {
    var parts = String(shortcut || "").split("+");
    var key = parts.pop();
    var symbol = Object.keys(shortcutSymbols).find(function (symbol) { return shortcutSymbols[symbol] === key; });
    return parts.concat([symbol || key]).join(" + ");
}

function setupError(draft, models) {
    var model = (models || []).find(function (entry) { return entry.id === draft.model; });
    if (!model)
        return "Choose an available speech model.";
    if (model.memory_error)
        return model.memory_error;
    if (draft.acceleration === "gpu" && !modelCompatibility(model).gpu)
        return "These ONNX weights are CPU only. Choose Parakeet FP32 or Whisper for GPU, or switch processing to Auto / CPU.";
    return "";
}

function setupMatches(config, draft) {
    return !!config && config.setup_complete === true
        && ["acceleration", "model"].every(function (key) { return config[key] === draft[key]; });
}

function setupOutcome(config, draft, phase, accepted, observed, cancelled, error, message) {
    if (!accepted || !observed)
        return "pending";
    if (phase === "error")
        return "failed";
    if (phase !== "idle")
        return "pending";
    if (cancelled)
        return "cancelled";
    if (error)
        return "failed";
    return setupMatches(config, draft) && String(message || "").indexOf("Setup complete.") === 0 ? "complete" : "failed";
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
        { key: "add_model", title: "Add custom model", help: "Use a compatible repository, local model or custom command.", kind: "action", actionLabel: "Add" },
        { key: "unload_model", title: "Unload model", help: "Free its memory. The next dictation loads it again.", kind: "action", actionLabel: "Unload" },
        { key: "initial_prompt", title: "Vocabulary / initial prompt", help: "Names, specialist terms or a short style hint for Whisper.", kind: "text", whisperOnly: true },
        { key: "use_beam_search", title: "Beam search", help: "Search several hypotheses instead of greedy decoding.", kind: "toggle", whisperOnly: true },
        { key: "beam_size", title: "Beam size", help: "Larger searches trade speed for accuracy. Whisper only.", kind: "number", from: 1, to: 10, step: 1, whisperOnly: true },
        { key: "temperature", title: "Temperature", help: "Lower is more deterministic; higher allows more variation.", kind: "number", from: 0, to: 1, step: 0.1, whisperOnly: true },
        { key: "no_speech_threshold", title: "No-speech threshold", help: "Skip segments classified as silence above this probability.", kind: "number", from: 0, to: 1, step: 0.1, whisperOnly: true },
        { key: "suppress_blank", title: "Suppress blank audio", help: "Suppress blank tokens at the beginning of decoding.", kind: "toggle", whisperOnly: true },
        { key: "show_timestamps", title: "Include timestamps", help: "Prefix transcript segments with their start and end times.", kind: "toggle", whisperOnly: true },
        { key: "compute_type", title: "Whisper precision", help: "Auto chooses INT8 on CPU or FP16 on supported GPUs. FP16 modes require compatible hardware. ONNX weight quantization is separate.", kind: "choice", options: choices([{ value: "auto", label: "Auto" }, "int8", "float16", "int8_float16", "float32", "int8_float32"]), whisperOnly: true },
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

function toggleValue(spec, enabled) {
    var value = enabled ? spec.enabledValue : spec.disabledValue;
    return value === undefined ? enabled : value;
}

function runtimeLabel(runtime) {
    if (!runtime || !runtime.device)
        return "";
    if (runtime.device === "external")
        return "External adapter";
    var label = runtime.device === "cuda" ? "GPU (CUDA)" : "CPU";
    return label + (runtime.compute_type ? " / " + runtime.compute_type : "");
}

if (typeof module !== "undefined")
    module.exports = { general: general, processingRows: processingRows, activationRows: activationRows,
        setupDraft: setupDraft, isBusy: isBusy, modelCompatibility: modelCompatibility, setupModels: setupModels,
        modelStatus: modelStatus, modelProgress: modelProgress,
        shortcutChord: shortcutChord, shortcutLabel: shortcutLabel, setupError: setupError,
        setupMatches: setupMatches, setupOutcome: setupOutcome,
        audio: audio, advanced: advanced, moveCursor: moveCursor, cycle: cycle, toggleValue: toggleValue, runtimeLabel: runtimeLabel };
