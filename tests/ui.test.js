const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const Settings = require("../src/Settings.js");
const root = path.resolve(__dirname, "..");

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

test("all requested settings are reachable via generated keyboard rows", () => {
  const rows = [...Settings.general({}), ...Settings.audio([]), ...Settings.advanced()];
  const keys = rows.map(row => row.key);
  assert.equal(new Set(keys).size, keys.length);
  for (const key of ["shortcut", "activation", "bar_section", "device", "language", "translate", "history", "initial_prompt", "restore_clipboard"])
    assert.ok(keys.includes(key), key);
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

test("manifest entry points exist", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"));
  assert.equal(manifest.id, "mohamedmansour.whisper");
  assert.equal(manifest.barWidget.defaultSection, "right");
  for (const entry of Object.values(manifest.entryPoints))
    assert.ok(fs.existsSync(path.join(root, entry)));
});

test("QML components parse with the installed Qt tooling", () => {
  for (const file of fs.readdirSync(path.join(root, "src")).filter(file => file.endsWith(".qml"))) {
    const result = spawnSync("/usr/lib/qt6/bin/qmlformat", [path.join(root, "src", file)], { encoding: "utf8" });
    assert.equal(result.status, 0, `${file}: ${result.stderr}`);
  }
});
