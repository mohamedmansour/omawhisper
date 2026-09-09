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
    readonly property bool checked: value === Settings.toggleValue(spec, true)
    signal engaged()
    signal modified(var value)
    signal activated()
    implicitHeight: editing ? Style.space(100) : Math.max(Style.space(66), labels.implicitHeight + Style.space(20))
    opacity: enabled ? 1 : 0.4

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
        if (spec.kind === "text" || spec.kind === "number") {
            controller.cancelEditor();
            editing = true;
            editor.text = value == null ? "" : String(value);
            controller.editor = root;
            Qt.callLater(function () {
                editor.forceActiveFocus();
                editor.selectAll();
                controller.revealCursor();
            });
        } else if (spec.kind === "action") {
            activated();
        } else {
            adjust(1);
        }
    }

    function finish(save) {
        if (!editing)
            return;
        if (save) {
            var next = editor.text;
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
        placeholderText: root.spec.placeholder || ""
        onAccepted: root.finish(true)
        Keys.onEscapePressed: root.finish(false)
        Keys.onTabPressed: {
            root.finish(true);
            if (!root.editing)
                root.controller.move(0, 1);
        }
        Keys.onBacktabPressed: {
            root.finish(true);
            if (!root.editing)
                root.controller.move(0, -1);
        }
    }
}
