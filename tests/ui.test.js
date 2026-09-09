const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const vm = require("node:vm");
const Settings = require("../src/Settings.js");
const root = path.resolve(__dirname, "..");

function qmlMethods(file, values) {
  const source = fs.readFileSync(path.join(root, "src", file), "utf8");
  const methods = source.match(/^    function \w+\([^\n]*\) \{[\s\S]*?^    \}/gm) || [];
  const context = vm.createContext({ Settings, Qt: { callLater: fn => fn() }, ...values });
  context.root = context;
  vm.runInContext(methods.join("\n"), context);
  return context;
}

test("arrow navigation includes tabs and clamps at the last setting", () => {
  assert.equal(Settings.moveCursor(-1, 1, 6), 0);
  assert.equal(Settings.moveCursor(0, -1, 6), -1);
  assert.equal(Settings.moveCursor(5, 1, 6), 5);
  assert.equal(Settings.moveCursor(-1, -1, 6), -1);
  assert.equal(Settings.moveCursor(0, 1, 0), -1);
});

test("choice navigation wraps and keeps option types", () => {
  const choices = [{ value: "left" }, { value: "center" }, { value: "right" }];
  assert.equal(Settings.cycle(choices, "left", -1), "right");
  assert.equal(Settings.cycle(choices, "right", 1), "left");
  assert.equal(Settings.cycle([], "default", 1), "default");
});

test("push-to-talk switch maps on/off to the existing activation modes", () => {
  const row = Settings.general({}).find(row => row.key === "activation");
  assert.equal(row.title, "Push to talk");
  assert.equal(row.kind, "toggle");
  assert.equal(Settings.toggleValue(row, true), "hold");
  assert.equal(Settings.toggleValue(row, false), "toggle");
});

test("ordinary switches still emit boolean values", () => {
  const rows = [...Settings.general({}), ...Settings.audio([]), ...Settings.advanced()];
  for (const row of rows.filter(row => row.kind === "toggle" && row.key !== "activation")) {
    assert.equal(Settings.toggleValue(row, true), true, row.key);
    assert.equal(Settings.toggleValue(row, false), false, row.key);
  }
});

test("processing device is keyboard selectable independently of the microphone", () => {
  const row = Settings.processingRows().find(row => row.key === "acceleration");
  assert.equal(row.kind, "choice");
  assert.deepEqual(row.options.map(option => option.value), ["auto", "gpu", "cpu"]);
  assert.equal(Settings.cycle(row.options, "auto", 1), "gpu");
  assert.equal(Settings.cycle(row.options, "gpu", 1), "cpu");
  assert.equal(Settings.cycle(row.options, "cpu", 1), "auto");
});

test("precision controls support automatic, CPU and GPU types", () => {
  const row = Settings.advanced().find(row => row.key === "compute_type");
  assert.equal(row.whisperOnly, true);
  assert.deepEqual(row.options.map(option => option.value),
    ["auto", "int8", "float16", "int8_float16", "float32", "int8_float32"]);
});

test("runtime labels describe the loaded device, not the requested mode", () => {
  assert.equal(Settings.runtimeLabel(undefined), "");
  assert.equal(Settings.runtimeLabel({}), "");
  assert.equal(Settings.runtimeLabel({ device: "cpu", compute_type: "int8_float32" }), "CPU / int8_float32");
  assert.equal(Settings.runtimeLabel({ device: "cuda", compute_type: "float16" }), "GPU (CUDA) / float16");
  assert.equal(Settings.runtimeLabel({ device: "external" }), "External adapter");
});

test("all requested settings are reachable via generated keyboard rows", () => {
  const rows = [...Settings.general({}), ...Settings.processingRows(), ...Settings.audio([]), ...Settings.advanced()];
  const keys = rows.map(row => row.key);
  assert.equal(new Set(keys).size, keys.length);
  for (const key of ["shortcut", "activation", "bar_section", "device", "language", "translate", "history", "initial_prompt", "restore_clipboard"])
    assert.ok(keys.includes(key), key);
});

const models = [
  { id: "whisper-base", name: "Whisper base", engine: "whisper", installed: false },
  { id: "parakeet-v3", name: "Parakeet INT8", engine: "parakeet", quantization: "int8" },
  { id: "parakeet-v3-fp32", name: "Parakeet FP32", engine: "parakeet", quantization: "" },
  { id: "custom-fp16", name: "Custom FP16", engine: "parakeet", quantization: "fp16", installed: true },
  { id: "custom-command", name: "Custom adapter", engine: "command" }
];

test("General keeps activation controls while processing lives only in Models", () => {
  const rows = Settings.general({ setup_complete: true });
  assert.deepEqual(rows.slice(0, 2).map(row => row.key), ["shortcut", "activation"]);
  for (const key of ["setup", "acceleration", "model"])
    assert.ok(!rows.some(row => row.key === key), key);
  assert.ok(!Settings.advanced().some(row => row.key === "acceleration"));
  assert.equal(Settings.processingRows()[0].key, "acceleration");
  assert.equal(rows[0].kind, "shortcut");
});

test("first run has safe defaults and preserves settings from before onboarding existed", () => {
  const expected = { acceleration: "auto", model: "whisper-base" };
  assert.deepEqual(Settings.setupDraft(), expected);
  assert.deepEqual(Settings.setupDraft({ setup_complete: false, model: "parakeet-v3", activation: "toggle" }),
    { ...expected, model: "parakeet-v3" });
});

test("reconfiguration owns only model and processing, never shortcut or activation", () => {
  const config = { setup_complete: true, acceleration: "cpu", model: "parakeet-v3",
    activation: "toggle", shortcut: "CTRL+SHIFT+F8", history: true };
  const draft = Settings.setupDraft(config);
  assert.deepEqual(draft, { acceleration: "cpu", model: "parakeet-v3" });
  const next = Settings.setupDraft({ ...config, acceleration: "auto" });
  assert.equal(next.acceleration, "auto");
  assert.equal(draft.acceleration, "cpu");
  assert.equal(config.activation, "toggle");
  assert.ok(!Object.hasOwn(draft, "setup_complete"));
});

test("GPU model compatibility supports built-ins, floating ONNX and external adapters", () => {
  const gpu = Settings.setupModels(models, { acceleration: "gpu" });
  assert.deepEqual(gpu.map(model => model.compatible), [true, false, true, true, true]);
  assert.equal(gpu[1].compatibility.label, "CPU only · quantized ONNX");
  assert.equal(gpu[4].compatibility.external, true);
  for (const acceleration of ["auto", "cpu"])
    assert.ok(Settings.setupModels(models, { acceleration }).every(model => model.compatible));
  assert.equal(Settings.modelCompatibility({ id: "parakeet-v3-fp32", engine: "parakeet" }).gpu, true);
  assert.equal(Settings.modelCompatibility({ id: "unknown", engine: "parakeet" }).gpu, false);
  assert.equal(Settings.modelCompatibility({ engine: "onnx-asr", quantization: "float32" }).gpu, true);
  assert.equal(Settings.modelCompatibility({ engine: "parakeet", quantization: "BF16" }).gpu, true);
  assert.equal(Settings.modelCompatibility({ engine: "parakeet", quantization: "bfloat16" }).gpu, true);
  assert.equal(Settings.modelCompatibility({ engine: "parakeet", quantization: "uint8" }).gpu, false);
  assert.equal(Settings.modelCompatibility({ engine: "parakeet", gpu_compatible: true }).gpu, true);
  assert.equal(Settings.modelCompatibility(undefined).gpu, false);
  assert.equal(gpu[0].installed, false);
  assert.equal(gpu[3].installed, true);
  assert.equal(models[0].compatible, undefined);
});

test("choosing a device never downloads or silently switches model variants", () => {
  const requests = [];
  const config = { ...Settings.setupDraft(), model: "parakeet-v3" };
  const controller = { config, service: { request: value => requests.push(value) } };
  const context = qmlMethods("ProcessingView.qml", { controller, canEdit: true, pendingDevice: "" });
  context.changeDevice("gpu");
  assert.equal(context.pendingDevice, "gpu");
  assert.equal(controller.config.model, "parakeet-v3");
  assert.equal(controller.config.acceleration, "auto");
  assert.deepEqual(requests, []);
});

test("the radio device selector is one keyboard row and never submits a model", () => {
  const context = qmlMethods("ProcessingView.qml", {
    controller: {}, canEdit: true, device: "auto", pendingDevice: "",
    options: Settings.processingRows()[0].options
  });
  context.activate(0);
  assert.equal(context.pendingDevice, "gpu");
  context.device = "gpu";
  context.adjust(0, 1);
  assert.equal(context.pendingDevice, "cpu");
  context.canEdit = false;
  context.adjust(0, 1);
  assert.equal(context.pendingDevice, "cpu");
  context.activate(1);
  assert.equal(context.pendingDevice, "cpu");
});

test("compact model labels retain hardware, installation and RAM state", () => {
  const config = { model: "whisper-base" };
  assert.equal(Settings.modelStatus({ ...models[0], loaded: true }, config, { device: "cpu" }, "gpu"), "Active (CPU)");
  assert.equal(Settings.modelStatus({ ...models[0], loaded: true }, config, { device: "cuda" }, "auto"), "Active (GPU)");
  assert.equal(Settings.modelStatus(models[0], config, {}, "auto"), "Download");
  assert.equal(Settings.modelStatus({ ...models[0], installed: true }, config, {}, "auto"), "Selected");
  assert.equal(Settings.modelStatus(models[3], config, {}, "auto"), "Ready");
  assert.equal(Settings.modelStatus(models[1], config, {}, "gpu"), "CPU only");
  assert.equal(Settings.modelStatus({ ...models[2], memory_error: "Not enough RAM" }, config, {}, "gpu"), "Needs more RAM");
  assert.equal(Settings.modelStatus({ ...models[0], progress: 45 }, config, {}, "auto"), "45%");
  assert.equal(Settings.modelStatus(models[4], config, {}, "auto"), "Unavailable");
  assert.equal(Settings.modelProgress("installing"), "Installing...");
  assert.equal(Settings.modelProgress("downloading"), "Downloading...");
  assert.equal(Settings.modelProgress("loading"), "Loading...");
});

test("setup exposes host RAM estimates and rejects impossible models separately from GPU format", () => {
  const error = "Parakeet FP32 needs an estimated 5.00 GiB of host RAM; only 4.00 GiB is installed.";
  const constrained = models.map(model => ({
    ...model,
    estimated_ram_bytes: model.id === "parakeet-v3" ? 1.5 * 1024 ** 3 : 5 * 1024 ** 3,
    memory_error: model.id === "parakeet-v3-fp32" ? error : ""
  }));
  for (const acceleration of ["auto", "cpu", "gpu"]) {
    const draft = { ...Settings.setupDraft(), model: "parakeet-v3-fp32", acceleration };
    const options = Settings.setupModels(constrained, draft);
    assert.equal(options[2].compatible, false);
    assert.equal(options[2].compatibility.gpu, true, "model format is GPU-compatible, but host RAM is not");
    assert.equal(Settings.setupError(draft, constrained), error);
    assert.match(options[1].memoryDetail, /1.5 GiB/);
    const controller = { config: draft, service: { state: { models: constrained } } };
    const context = qmlMethods("ProcessingView.qml", { controller, canEdit: true, device: acceleration, draft });
    context.submit(draft.model);
    assert.equal(controller.localError, error);
    assert.equal(context.draft, draft);
  }
});

test("review validates model availability and GPU compatibility without activation fields", () => {
  const draft = Settings.setupDraft();
  assert.equal(Settings.setupError(draft, models), "");
  assert.match(Settings.setupError({ ...draft, model: "missing" }, models), /available speech model/);
  assert.match(Settings.setupError({ ...draft, model: "parakeet-v3", acceleration: "gpu" }, models), /FP32/);
  assert.equal(Settings.setupError({ ...draft, model: "parakeet-v3", acceleration: "auto" }, models), "");
  assert.equal(Object.hasOwn(draft, "shortcut"), false);
  assert.equal(Object.hasOwn(draft, "activation"), false);
});

test("shortcut capture uses stable Hyprland modifier order and requires a chord", () => {
  assert.equal(Settings.shortcutChord({ alt: true, super: true }, "V"), "SUPER+ALT+V");
  assert.equal(Settings.shortcutChord({ shift: true, alt: true, ctrl: true, super: true }, "F8"), "SUPER+CTRL+ALT+SHIFT+F8");
  assert.equal(Settings.shortcutChord({}, "V"), "");
  assert.equal(Settings.shortcutChord({ super: true }, ""), "");
});

const shortcutSymbols = {
  "!": "EXCLAM", "\"": "QUOTEDBL", "#": "NUMBERSIGN", "$": "DOLLAR",
  "%": "PERCENT", "&": "AMPERSAND", "'": "APOSTROPHE", "(": "PARENLEFT",
  ")": "PARENRIGHT", "*": "ASTERISK", "+": "PLUS", ",": "COMMA",
  "-": "MINUS", ".": "PERIOD", "/": "SLASH", ":": "COLON",
  ";": "SEMICOLON", "<": "LESS", "=": "EQUAL", ">": "GREATER",
  "?": "QUESTION", "@": "AT", "[": "BRACKETLEFT", "\\": "BACKSLASH",
  "]": "BRACKETRIGHT", "^": "ASCIICIRCUM", "_": "UNDERSCORE", "`": "GRAVE",
  "{": "BRACELEFT", "|": "BAR", "}": "BRACERIGHT", "~": "ASCIITILDE"
};

test("punctuation uses Hyprland key names but displays the pressed symbol", () => {
  for (const [symbol, name] of Object.entries(shortcutSymbols)) {
    assert.equal(Settings.shortcutChord({ super: true }, symbol), `SUPER+${name}`);
    assert.equal(Settings.shortcutLabel(`SUPER+${name}`), `SUPER + ${symbol}`);
    assert.equal(Settings.shortcutChord({}, symbol), "");
  }
  assert.equal(Settings.shortcutLabel("SUPER+ALT+V"), "SUPER + ALT + V");
  assert.equal(Settings.shortcutLabel("CTRL+SHIFT+F8"), "CTRL + SHIFT + F8");
  assert.equal(Settings.shortcutLabel(""), "");
});

test("the shortcut editor captures and saves punctuation without saving display text", () => {
  const Qt = { Key_Exclam: 0x21, Key_AsciiTilde: 0x7e,
    MetaModifier: 0x10000000, ControlModifier: 0x04000000,
    AltModifier: 0x08000000, ShiftModifier: 0x02000000 };
  for (const [symbol, name] of Object.entries({ ...shortcutSymbols, C: "C", 0: "0" })) {
    const saved = [];
    const context = qmlMethods("PreferenceRow.qml", {
      Qt, controller: { localError: "", editor: {}, focusPanel() {} },
      spec: { kind: "shortcut" }, editing: true,
      editor: { text: "SUPER + ALT + V" }, shortcutDraft: "SUPER+ALT+V",
      modified: value => saved.push(value)
    });
    context.captureChord({ key: symbol.charCodeAt(0), modifiers: Qt.MetaModifier });
    assert.equal(context.editor.text, `SUPER + ${symbol}`);
    assert.equal(context.shortcutDraft, `SUPER+${name}`);
    context.finish(true);
    assert.deepEqual(saved, [`SUPER+${name}`]);
    assert.equal(context.editing, false);
    assert.equal(context.controller.editor, null);
  }
});

test("punctuation without a modifier and repeated keys do not change the shortcut", () => {
  const Qt = { Key_Exclam: 0x21, Key_AsciiTilde: 0x7e, MetaModifier: 0x10000000 };
  const context = qmlMethods("PreferenceRow.qml", {
    Qt, controller: { localError: "" }, editor: { text: "SUPER + ALT + V" },
    shortcutDraft: "SUPER+ALT+V"
  });
  context.captureChord({ key: 0x2e, modifiers: 0 });
  assert.match(context.controller.localError, /Include Super, Ctrl, Alt or Shift/);
  assert.equal(context.shortcutDraft, "SUPER+ALT+V");
  context.captureChord({ key: 0x2e, modifiers: Qt.MetaModifier, isAutoRepeat: true });
  assert.equal(context.shortcutDraft, "SUPER+ALT+V");
  context.captureChord({ key: 0x2e, modifiers: Qt.MetaModifier });
  assert.equal(context.shortcutDraft, "SUPER+PERIOD");
  assert.equal(context.controller.localError, "");
});

test("reopening or cancelling a symbol shortcut preserves its serialized value", () => {
  const context = qmlMethods("PreferenceRow.qml", {
    enabled: true, controller: { cancelEditor() {}, revealCursor() {}, focusPanel() {} },
    spec: { kind: "shortcut" }, value: "SUPER+PLUS", editing: false, shortcutDraft: "",
    editor: { text: "", forceActiveFocus() {} }, engaged() {},
    modified: () => assert.fail("Cancelling must not save a shortcut")
  });
  context.activate();
  assert.equal(context.shortcutDraft, "SUPER+PLUS");
  assert.equal(context.editor.text, "SUPER + +");
  context.finish(false);
  assert.equal(context.editing, false);
  assert.equal(context.controller.editor, null);
});

test("installation phases participate in busy status", () => {
  for (const phase of ["installing", "downloading", "loading", "transcribing"])
    assert.equal(Settings.isBusy(phase), true, phase);
  for (const phase of ["idle", "error", "recording", undefined])
    assert.equal(Settings.isBusy(phase), false, phase);
});

test("setup cannot complete until the acknowledged observed job is idle and every field matches", () => {
  const draft = Settings.setupDraft();
  const config = { ...draft, setup_complete: true };
  assert.equal(Settings.setupMatches(config, draft), true);
  assert.equal(Settings.setupMatches({ ...config, setup_complete: false }, draft), false);
  assert.equal(Settings.setupMatches(null, draft), false);
  for (const key of Object.keys(draft))
    assert.equal(Settings.setupMatches({ ...config, [key]: "different" }, draft), false, key);
  assert.equal(Settings.setupOutcome(config, draft, "idle", false, true, false, ""), "pending");
  assert.equal(Settings.setupOutcome(config, draft, "idle", true, false, false, ""), "pending");
  for (const phase of ["installing", "downloading", "loading", "transcribing", "recording"])
    assert.equal(Settings.setupOutcome(config, draft, phase, true, true, false, ""), "pending", phase);
  assert.equal(Settings.setupOutcome(config, draft, "idle", true, true, false, "", "Setup complete. Ready."), "complete");
  assert.equal(Settings.setupOutcome(config, draft, "idle", true, true, false, "", ""), "failed");
  assert.equal(Settings.setupOutcome(config, draft, "idle", true, true, false, "", "Service restarted"), "failed");
  assert.equal(Settings.setupOutcome({ ...config, model: "other" }, draft, "idle", true, true, false, ""), "failed");
});

test("reconfiguring an already-complete identical config does not mask cancellation or errors", () => {
  const config = { ...Settings.setupDraft(), setup_complete: true };
  const draft = Settings.setupDraft(config);
  assert.equal(Settings.setupOutcome(config, draft, "idle", true, false, false, ""), "pending");
  assert.equal(Settings.setupOutcome(config, draft, "idle", true, true, true, ""), "cancelled");
  assert.equal(Settings.setupOutcome(config, draft, "idle", true, true, false, "Shortcut conflict"), "failed");
  assert.equal(Settings.setupOutcome(config, draft, "error", true, true, false, "Model failed"), "failed");
  assert.equal(Settings.setupOutcome(config, draft, "error", true, false, false, "Stale error"), "pending");
});

test("QML exposes inline processing without a wizard and retains installation and cancellation", () => {
  const panel = fs.readFileSync(path.join(root, "src/Panel.qml"), "utf8");
  const processing = fs.readFileSync(path.join(root, "src/ProcessingView.qml"), "utf8");
  const catalog = fs.readFileSync(path.join(root, "src/ModelsView.qml"), "utf8");
  const row = fs.readFileSync(path.join(root, "src/PreferenceRow.qml"), "utf8");
  assert.match(panel, /currentView: views\[tab\]/);
  assert.match(panel, /if \(opened && backend\.connected && !setupChecked\)/);
  assert.match(panel, /onOpenedChanged:/);
  assert.match(panel, /root\.tabNavigate\(direction\)/);
  assert.match(panel, /Settings\.isBusy\(backend\.state\.phase\)/);
  assert.doesNotMatch(panel, /setupView|startSetup|Reconfigure dictation/);
  assert.match(processing, /action: "setup", values: Object\.assign\(\{\}, draft\)/);
  assert.match(processing, /action: "cancel"/);
  assert.doesNotMatch(processing, /steps:|Settings\.activationRows|draft\.(shortcut|activation)/);
  assert.doesNotMatch(processing, /action: "(configure|load|download)"/);
  assert.doesNotMatch(processing, /applyButton|canApply|Apply to|PreferenceRow\s*\{/);
  assert.match(processing, /Accessible\.RadioButton/);
  assert.match(processing, /Controls\.RadioButton\s*\{/);
  assert.match(processing, /checked: root\.device === modelData\.value/);
  assert.match(processing, /background: Item \{\}/);
  assert.match(processing, /focusPolicy: Qt\.NoFocus/);
  assert.match(processing, /text: "Processing device"/);
  assert.match(processing, /Layout\.alignment: Qt\.AlignRight \| Qt\.AlignVCenter/);
  assert.match(processing, /PanelSeparator\s*\{/);
  assert.ok(processing.indexOf("id: deviceLabel") < processing.indexOf("id: deviceChoices"));
  assert.ok(processing.indexOf("id: deviceChoices") < processing.indexOf("id: deviceDivider"));
  assert.doesNotMatch(processing, /Process\s*\{|scripts\/install|pip install|controller\.show\(/);
  assert.match(catalog, /ProcessingView\s*\{/);
  assert.match(catalog, /deviceControl\.submit\(model\.id\)/);
  assert.match(catalog, /modelData\.memory_error/);
  assert.doesNotMatch(catalog, /id: addButton|id: unloadButton|action: "download"/);
  assert.match(catalog, /implicitHeight: Style\.space\(42\)/);
  assert.match(row, /readOnly: root\.spec\.kind === "shortcut"/);
  assert.match(row, /controller\.editor = root/);
  assert.match(row, /root\.captureChord\(event\)/);
});

test("the shared header stays visible when switching to Models", () => {
  const panel = fs.readFileSync(path.join(root, "src/Panel.qml"), "utf8");
  const header = panel.slice(panel.indexOf("id: chrome"), panel.indexOf("Flickable {"));
  assert.match(header, /Settings\.runtimeLabel/);
  assert.match(header, /root\.config\.shortcut/);
  assert.doesNotMatch(header, /visible:.*\btab\b/);
  assert.doesNotMatch(header, /root\.tab === 1 \? ""/);
});

test("Tab navigation stays on the standard settings tabs", () => {
  const changes = [];
  const context = qmlMethods("Panel.qml", { tab: 1 });
  context.changeTab = index => changes.push(index);
  context.tabNavigate(1);
  context.tabNavigate(-1);
  assert.deepEqual(changes, [2, 0]);
});

test("first opening takes new users to Models without a separate setup screen", () => {
  const context = qmlMethods("Panel.qml", {
    opened: true, backend: { connected: true }, setupChecked: false, config: { setup_complete: false }
  });
  const changes = [];
  context.changeTab = index => changes.push(index);
  context.checkFirstRun();
  context.checkFirstRun();
  assert.deepEqual(changes, [1]);
});

test("a model selection waits for a fresh completed job without moving the cursor", () => {
  const draft = Settings.setupDraft();
  const original = { ...draft, setup_complete: true, activation: "hold", shortcut: "SUPER+ALT+V" };
  const requests = [];
  const service = { connected: true, state: { phase: "idle", error: "", message: "", models, config: original }, request: value => requests.push(value) };
  const controller = {
    config: original, service, cursor: 2, localError: "", localNotice: "",
    cancelEditor() {}, resetScroll() {}, focusPanel() {}, revealCursor() {}, changeTab() {}
  };
  const context = qmlMethods("ProcessingView.qml", {
    controller, draft, device: "auto", pendingDevice: "", running: false, canEdit: true,
    phase: "idle", accepted: false, observed: false, cancelling: false, checking: false
  });
  context.submit("whisper-base");
  assert.equal(requests.length, 1);
  assert.equal(requests[0].action, "setup");
  assert.deepEqual({ ...requests[0].values }, draft);
  assert.notEqual(requests[0].values, context.draft);
  assert.equal(context.running, true);
  assert.equal(controller.cursor, 2);
  context.accepted = true;
  context.checkState();
  assert.equal(context.running, true, "already-true setup_complete must not finish an unobserved job");
  assert.equal(requests[1].action, "status", "fresh status confirms terminal watcher snapshots");
  context.canEdit = false;
  context.changeDevice("gpu");
  assert.equal(context.draft.acceleration, "auto");
  assert.equal(context.pendingDevice, "");
  context.phase = "loading";
  service.state.phase = "loading";
  context.checkState(service.state);
  assert.equal(context.observed, true);
  assert.equal(context.running, true);
  context.phase = "idle";
  service.state.phase = "idle";
  service.state.message = "Setup complete. Ready.";
  context.checkState(service.state);
  assert.equal(context.running, false);
  assert.equal(context.notice, "");
  assert.deepEqual(controller.config, original);
});

test("cancellation keeps the pending device and leaves saved preferences unchanged", () => {
  const original = { ...Settings.setupDraft(), setup_complete: true };
  const draft = { ...Settings.setupDraft(), acceleration: "gpu", model: "parakeet-v3-fp32" };
  const requests = [];
  const state = { phase: "installing", error: "", message: "", models, config: original };
  const controller = {
    config: original, service: { connected: true, state, request: request => requests.push(request) },
    cancelEditor() {}, resetScroll() {}, focusPanel() {}, revealCursor() {}, changeTab() {}
  };
  const context = qmlMethods("ProcessingView.qml", {
    controller, draft, device: "gpu", pendingDevice: "gpu", operationBusy: true, running: true, canEdit: false,
    phase: "installing", accepted: true, observed: true, cancelling: false
  });
  context.cancelJob();
  context.cancelJob();
  assert.equal(requests.length, 1, "cancel cannot queue duplicate requests");
  assert.equal(requests[0].action, "cancel");
  assert.equal(context.running, true, "wait until backend stops before allowing changes");
  context.phase = "idle";
  state.phase = "idle";
  state.message = "Setup cancelled. Previous settings were kept.";
  context.checkState(state);
  assert.equal(context.running, false);
  assert.equal(context.pendingDevice, "gpu");
  assert.equal(context.draft, draft);
  assert.equal(controller.config, original);
  assert.match(context.notice, /Previous settings were kept/);
  context.canEdit = true;
  context.submit(draft.model);
  assert.equal(requests[1].action, "setup");
  assert.deepEqual({ ...requests[1].values }, draft);
  assert.equal(context.running, true);
  assert.equal(context.accepted, false);
  assert.equal(context.observed, false);
});

test("asynchronous runtime errors unlock the inline control without discarding the device choice", () => {
  const draft = Settings.setupDraft();
  const controller = {
    config: { ...draft, setup_complete: false },
    service: { state: { phase: "error", error: "Shortcut already bound", message: "", models }, error: "Shortcut already bound" },
    revealCursor() {}
  };
  const context = qmlMethods("ProcessingView.qml", {
    controller, draft, pendingDevice: "gpu", running: true, phase: "error",
    accepted: true, observed: true, cancelling: false
  });
  context.checkState(controller.service.state);
  assert.equal(context.running, false);
  assert.equal(context.pendingDevice, "gpu");
  assert.equal(context.draft, draft);
  assert.match(context.notice, /Shortcut already bound/);
  assert.equal(controller.config.setup_complete, false);
});

test("a fresh successful result wins if cancellation arrived after the atomic commit", () => {
  const draft = Settings.setupDraft();
  const controller = {
    service: { state: {}, error: "" }, cancelEditor() {}, changeTab() {}
  };
  const context = qmlMethods("ProcessingView.qml", {
    controller, draft, pendingDevice: "cpu", running: true, accepted: true, observed: true,
    cancelling: true
  });
  context.checkState({
    phase: "idle", config: { ...draft, setup_complete: true, activation: "toggle", shortcut: "CTRL+SHIFT+F8" },
    runtime: { device: "cpu", compute_type: "int8" }, error: "", message: "Setup complete."
  });
  assert.equal(context.running, false);
  assert.equal(context.pendingDevice, "");
  assert.equal(context.notice, "");
});

test("closing the panel discards unapplied choices but does not interrupt an active operation", () => {
  const requests = [];
  const config = { ...Settings.setupDraft(), setup_complete: true };
  const controller = {
    config, service: { connected: true, state: { phase: "error" }, request: value => requests.push(value) },
    cancelEditor() {}, changeTab() {}
  };
  const context = qmlMethods("ProcessingView.qml", { controller, pendingDevice: "gpu", running: false });
  context.discard();
  assert.equal(context.pendingDevice, "");
  context.pendingDevice = "gpu";
  context.running = true;
  context.discard();
  assert.equal(context.pendingDevice, "gpu");
  assert.deepEqual(requests, []);
  assert.equal(controller.config, config);
});

test("one model activation downloads and loads both new and installed models, with row-local cancellation", () => {
  const requests = [];
  const actions = [];
  const entries = models.map(model => ({ ...model, installed: model.id === "parakeet-v3" }));
  const controller = { cursor: 1, busy: false, service: { connected: true, request: value => requests.push(value) } };
  const deviceControl = {
    count: 1, running: false, draft: {}, activate: index => actions.push(["activate", index]),
    adjust: (index, direction) => actions.push(["adjust", index, direction]),
    submit: id => actions.push(["select", id]), cancelJob: () => actions.push(["cancel"])
  };
  const context = qmlMethods("ModelsView.qml", {
    controller, deviceControl, models: entries, modelOffset: 1, adding: false, deleteId: ""
  });
  context.activate(0);
  context.adjust(0, 1);
  context.activate(1);
  context.activate(2);
  assert.deepEqual(actions, [["activate", 0], ["adjust", 0, 1], ["select", "whisper-base"], ["select", "parakeet-v3"]]);
  assert.deepEqual(requests, [], "there is no separate download command or second click");
  deviceControl.running = true;
  deviceControl.draft = { model: "whisper-base" };
  controller.busy = true;
  context.activate(1);
  assert.deepEqual(actions[4], ["cancel"]);
  context.activate(2);
  assert.equal(actions.length, 5, "other model rows remain disabled during the operation");
  deviceControl.running = false;
  controller.busy = false;
  context.removeSelected();
  context.activate(1);
  assert.equal(requests[0].action, "remove_model");
  assert.equal(requests[0].id, "whisper-base");
  assert.equal(context.rowAt(-1), null);
});

test("custom model and unload controls stay available in Advanced rather than the model list", () => {
  const rows = Settings.advanced();
  assert.equal(rows.find(row => row.key === "add_model").kind, "action");
  assert.equal(rows.find(row => row.key === "unload_model").kind, "action");
  const actions = [];
  const context = qmlMethods("Panel.qml", {
    backend: { connected: true, request: value => actions.push(value.action) },
    busy: false, recording: false, modelsView: { startAdding: () => actions.push("add") }
  });
  context.changeTab = tab => actions.push(tab);
  context.preferenceAction("add_model");
  context.preferenceAction("unload_model");
  assert.deepEqual(actions, [1, "add", "unload"]);
});

test("microphone choices use stable PipeWire IDs", () => {
  const rows = Settings.audio([{ id: "", name: "System default" }, { id: "alsa_input.usb", name: "USB mic" }]);
  assert.deepEqual(rows[0].options, [
    { value: "", label: "System default" },
    { value: "alsa_input.usb", label: "USB mic" }
  ]);
});

test("engine-specific settings cannot be edited for Parakeet or custom engines", () => {
  const rows = [...Settings.audio([]), ...Settings.advanced()];
  for (const key of ["language", "translate", "vad", "initial_prompt", "beam_size", "compute_type",
    "temperature", "no_speech_threshold", "suppress_blank", "show_timestamps", "use_beam_search"])
    assert.equal(rows.find(row => row.key === key).whisperOnly, true, key);
  for (const key of ["device", "max_duration", "history", "paste_shortcut"])
    assert.notEqual(rows.find(row => row.key === key).whisperOnly, true, key);
});

test("manifest entry points and README screenshots exist", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"));
  assert.equal(manifest.id, "mohamedmansour.whisper");
  assert.equal(manifest.barWidget.defaultSection, "right");
  for (const entry of Object.values(manifest.entryPoints))
    assert.ok(fs.existsSync(path.join(root, entry)));
  assert.equal(manifest.homepage, "https://github.com/mohamedmansour/omawhisper");
  assert.ok(manifest.screenshots.length > 0);
  const readme = fs.readFileSync(path.join(root, "README.md"), "utf8");
  for (const screenshot of manifest.screenshots) {
    assert.match(screenshot, /^assets\/screenshots\/[^/]+\.png$/);
    const image = fs.readFileSync(path.join(root, screenshot));
    assert.equal(image.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
    assert.ok(readme.includes(`](${screenshot})`));
  }
});

test("QML components parse with the installed Qt tooling", () => {
  for (const file of fs.readdirSync(path.join(root, "src")).filter(file => file.endsWith(".qml"))) {
    const result = spawnSync("/usr/lib/qt6/bin/qmlformat", [path.join(root, "src", file)], { encoding: "utf8" });
    assert.equal(result.status, 0, `${file}: ${result.stderr}`);
  }
});
