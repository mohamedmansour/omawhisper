import QtQuick
import qs.Commons
import qs.Ui
import "Settings.js" as Settings

Item {
    id: root
    required property var controller
    required property var spec
    property var value
    property bool hasCursor: false
    property bool editing: false
    property string shortcutDraft: ""
    readonly property bool checked: value === Settings.toggleValue(spec, true)
    signal engaged()
    signal modified(var value)
    signal activated()
    implicitHeight: editing ? Style.space(100) : Math.max(Style.space(66), labels.implicitHeight + Style.space(20))
    opacity: enabled ? 1 : 0.4
    Accessible.role: spec.kind === "toggle" ? Accessible.CheckBox : Accessible.Button
    Accessible.name: spec.title
    Accessible.description: (spec.help || "") + (value === undefined ? "" : " " + String(value))
    Accessible.focused: hasCursor
    Accessible.checkable: spec.kind === "toggle"
    Accessible.checked: checked
    Accessible.onPressAction: activate()

    function adjust(direction) {
        if (!enabled)
            return;
        if (spec.kind === "choice")
            modified(Settings.cycle(spec.options || [], value, direction));
        else if (spec.kind === "toggle")
            modified(Settings.toggleValue(spec, !checked));
        else if (spec.kind === "number")
            modified(Number(Math.max(spec.from, Math.min(spec.to, Number(value) + spec.step * direction)).toFixed(6)));
    }

    function activate() {
        if (!enabled)
            return;
        if (controller.editor && controller.editor !== root)
            controller.cancelEditor();
        engaged();
        if (spec.kind === "text" || spec.kind === "number" || spec.kind === "shortcut") {
            controller.cancelEditor();
            editing = true;
            editor.text = value == null ? "" : String(value);
            if (spec.kind === "shortcut") {
                shortcutDraft = editor.text;
                editor.text = Settings.shortcutLabel(shortcutDraft);
            }
            controller.editor = root;
            Qt.callLater(function () {
                editor.forceActiveFocus();
                if (root.spec.kind !== "shortcut")
                    editor.selectAll();
                controller.revealCursor();
            });
        } else if (spec.kind === "action") {
            activated();
        } else {
            adjust(1);
        }
    }

    function captureChord(event) {
        if (event.isAutoRepeat)
            return;
        var modifiers = {
            super: (event.modifiers & Qt.MetaModifier) !== 0,
            ctrl: (event.modifiers & Qt.ControlModifier) !== 0,
            alt: (event.modifiers & Qt.AltModifier) !== 0,
            shift: (event.modifiers & Qt.ShiftModifier) !== 0
        };
        var key = "";
        if (event.key >= Qt.Key_Exclam && event.key <= Qt.Key_AsciiTilde)
            key = String.fromCharCode(event.key);
        else if (event.key >= Qt.Key_F1 && event.key <= Qt.Key_F24)
            key = "F" + (event.key - Qt.Key_F1 + 1);
        else {
            var names = {};
            names[Qt.Key_Space] = "SPACE";
            names[Qt.Key_Tab] = "TAB";
            names[Qt.Key_Return] = "RETURN";
            names[Qt.Key_Backspace] = "BACKSPACE";
            names[Qt.Key_Delete] = "DELETE";
            names[Qt.Key_Insert] = "INSERT";
            names[Qt.Key_Home] = "HOME";
            names[Qt.Key_End] = "END";
            names[Qt.Key_PageUp] = "PRIOR";
            names[Qt.Key_PageDown] = "NEXT";
            names[Qt.Key_Left] = "LEFT";
            names[Qt.Key_Right] = "RIGHT";
            names[Qt.Key_Up] = "UP";
            names[Qt.Key_Down] = "DOWN";
            key = names[event.key] || "";
        }
        var chord = Settings.shortcutChord(modifiers, key);
        if (chord) {
            shortcutDraft = chord;
            editor.text = Settings.shortcutLabel(chord);
            controller.localError = "";
        } else if (key) {
            controller.localError = "Include Super, Ctrl, Alt or Shift in your shortcut.";
        }
    }

    function finish(save) {
        if (!editing)
            return;
        if (save) {
            var next = spec.kind === "shortcut" ? shortcutDraft : editor.text;
            if (spec.kind === "number") {
                next = Number(next);
                if (!editor.text.trim() || !isFinite(next) || (spec.step >= 1 && Math.floor(next) !== next) || next < spec.from || next > spec.to) {
                    controller.localError = "Enter " + (spec.step >= 1 ? "a whole number" : "a number") + " between " + spec.from + " and " + spec.to + ".";
                    return;
                }
            }
            modified(next);
        }
        editing = false;
        controller.editor = null;
        controller.focusPanel();
    }

    Button {
        anchors.fill: parent
        hasCursor: root.hasCursor
        foreground: root.controller.foreground
        onClicked: root.activate()

        Column {
            id: labels
            anchors.left: parent.left
            anchors.leftMargin: Style.space(10)
            anchors.verticalCenter: root.editing ? undefined : parent.verticalCenter
            y: root.editing ? Style.space(8) : 0
            width: parent.width - Style.space(24) - (root.editing ? 0 : (root.spec.kind === "toggle" ? toggle.width : valueLabel.width) + Style.space(12))
            spacing: Style.space(4)
            Text {
                width: parent.width
                text: root.spec.title
                textFormat: Text.PlainText
                color: root.controller.foreground
                font.family: root.controller.fontFamily
                font.pixelSize: Style.font.body
                font.bold: true
                elide: Text.ElideRight
            }
            Text {
                width: parent.width
                text: root.spec.help || ""
                textFormat: Text.PlainText
                color: root.controller.dim
                font.family: root.controller.fontFamily
                font.pixelSize: Style.font.caption
                wrapMode: Text.WordWrap
                visible: !root.editing
            }
        }
        Text {
            id: valueLabel
            visible: !root.editing && root.spec.kind !== "toggle"
            width: Math.min(implicitWidth, parent.width * 0.35)
            anchors.right: parent.right
            anchors.rightMargin: Style.space(10)
            anchors.verticalCenter: parent.verticalCenter
            text: {
                if (root.spec.kind === "toggle")
                    return root.checked ? "On" : "Off";
                if (root.spec.kind === "choice") {
                    var match = (root.spec.options || []).find(function (o) { return o.value === root.value; });
                    return (match ? match.label : String(root.value || "Default")) + "  >";
                }
                if (root.spec.kind === "action")
                    return root.spec.actionLabel || ">";
                if (root.spec.kind === "shortcut")
                    return Settings.shortcutLabel(root.value) || "Not set";
                return String(root.value == null || root.value === "" ? "Not set" : root.value);
            }
            textFormat: Text.PlainText
            color: root.controller.foreground
            font.family: root.controller.fontFamily
            font.pixelSize: Style.font.bodySmall
            elide: Text.ElideMiddle
        }
        ToggleSwitch {
            id: toggle
            visible: root.spec.kind === "toggle"
            anchors.right: parent.right
            anchors.rightMargin: Style.space(10)
            anchors.verticalCenter: parent.verticalCenter
            checked: root.checked
            foreground: root.controller.foreground
            interactive: false
        }
    }

    TextField {
        id: editor
        visible: root.editing
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: Style.space(8)
        foreground: root.controller.foreground
        placeholderText: root.spec.kind === "shortcut" ? "Press your shortcut…" : root.spec.placeholder || ""
        readOnly: root.spec.kind === "shortcut"
        Accessible.name: root.spec.title
        Accessible.description: root.spec.kind === "shortcut" ? "Press a modifier and key. Enter saves. Escape cancels." : root.spec.help || ""
        Keys.onPressed: function (event) {
            if (event.key === Qt.Key_Escape) {
                root.finish(false);
            } else if (root.spec.kind === "shortcut" && (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) && event.modifiers === Qt.NoModifier) {
                root.finish(true);
            } else if ((event.key === Qt.Key_Tab && event.modifiers === Qt.NoModifier)
                    || event.key === Qt.Key_Backtab || (event.key === Qt.Key_Tab && event.modifiers === Qt.ShiftModifier)) {
                root.finish(true);
                if (!root.editing)
                    root.controller.move(0, event.modifiers === Qt.ShiftModifier || event.key === Qt.Key_Backtab ? -1 : 1);
            } else if (root.spec.kind === "shortcut") {
                root.captureChord(event);
            } else {
                return;
            }
            event.accepted = true;
        }
        onAccepted: root.finish(true)
    }
}
