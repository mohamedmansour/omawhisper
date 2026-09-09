import QtQuick
import QtQuick.Controls as Controls
import QtQuick.Layouts
import qs.Commons
import qs.Ui
import "Settings.js" as Settings

Column {
    id: root
    required property var controller
    property string pendingDevice: ""
    property var draft: Settings.setupDraft({})
    property bool running: false
    property bool accepted: false
    property bool observed: false
    property bool cancelling: false
    property bool checking: false
    property string notice: ""
    readonly property string device: pendingDevice || controller.config.acceleration || "auto"
    readonly property bool changed: device !== (controller.config.acceleration || "auto")
    readonly property string phase: controller.service.state.phase
    readonly property bool operationBusy: running || Settings.isBusy(phase) || controller.recording
    readonly property bool canEdit: controller.service.connected && !operationBusy
    readonly property int count: 1
    readonly property var options: Settings.processingRows()[0].options
    spacing: Style.space(6)

    function rowAt(index) {
        return index === 0 ? deviceRow : null;
    }

    function activate(index) {
        if (index === 0)
            adjust(0, 1);
    }

    function adjust(index, direction) {
        if (index === 0)
            changeDevice(Settings.cycle(options, device, direction));
    }

    function changeDevice(value) {
        if (!canEdit)
            return;
        pendingDevice = value;
        notice = "";
        controller.localError = "";
        controller.localNotice = "";
    }

    function discard() {
        if (running)
            return;
        pendingDevice = "";
        notice = "";
    }

    function submit(modelId) {
        if (!canEdit)
            return;
        var values = { model: modelId === undefined ? controller.config.model : modelId, acceleration: device };
        var error = Settings.setupError(values, controller.service.state.models);
        if (error) {
            controller.localError = error;
            return;
        }
        controller.cancelEditor();
        controller.localError = "";
        controller.localNotice = "";
        notice = "";
        draft = values;
        accepted = false;
        observed = false;
        cancelling = false;
        checking = false;
        running = true;
        controller.service.request({ action: "setup", values: Object.assign({}, draft) });
    }

    function cancelJob() {
        if (!operationBusy || cancelling || !controller.service.connected)
            return;
        cancelling = true;
        controller.service.request({ action: "cancel" });
    }

    function requestCheck() {
        if (running && accepted && !checking && controller.service.connected) {
            checking = true;
            controller.service.request({ action: "status" });
        }
    }

    function checkState(snapshot) {
        if (!running) {
            if (!Settings.isBusy(controller.service.state.phase) && !controller.recording)
                cancelling = false;
            return;
        }
        var state = snapshot || controller.service.state;
        if (Settings.isBusy(state.phase))
            observed = true;
        // Confirm terminal watcher updates with a fresh reply, not a stale success from an earlier job.
        if (!snapshot && !Settings.isBusy(state.phase)) {
            requestCheck();
            return;
        }
        var cancelled = (state.message || "").toLowerCase().indexOf("setup cancelled") !== -1
            || (cancelling && !Settings.setupMatches(state.config, draft));
        var outcome = Settings.setupOutcome(state.config, draft, state.phase, accepted, observed, cancelled, state.error, state.message);
        if (outcome === "pending")
            return;
        running = false;
        cancelling = false;
        checking = false;
        if (outcome === "complete") {
            pendingDevice = "";
            notice = "";
        } else {
            notice = outcome === "cancelled" ? "Cancelled. Previous settings were kept."
                : state.error || controller.service.error || "Could not apply the model and processing device. Previous settings were kept.";
        }
    }

    Connections {
        target: root.controller.service
        function onStateChanged() { root.checkState(); }
        function onConnectedChanged() { root.requestCheck(); }
        function onCommandSucceeded(request, result) {
            if (request.action === "setup" && root.running) {
                root.accepted = true;
                if (result)
                    root.checkState(result);
                root.requestCheck();
            } else if (request.action === "status" && root.running) {
                root.checking = false;
                if (result)
                    root.checkState(result);
            }
        }
        function onCommandFailed(request, message) {
            if (request.action === "setup" && root.running) {
                root.running = false;
                root.cancelling = false;
                root.notice = message + " Previous settings were kept.";
            } else if (request.action === "cancel") {
                root.cancelling = false;
                root.notice = "Could not cancel: " + message;
            } else if (request.action === "status" && root.running) {
                root.checking = false;
                root.notice = "Waiting to confirm the operation: " + message;
            }
        }
    }

    Timer {
        interval: 2000
        repeat: true
        running: root.running && root.accepted && root.controller.service.connected && !Settings.isBusy(root.phase)
        onTriggered: root.requestCheck()
    }

    RowLayout {
        id: deviceRow
        width: parent.width
        height: Math.max(Style.space(40), implicitHeight)
        spacing: Style.space(16)
        Accessible.role: Accessible.Grouping
        Accessible.name: deviceLabel.text
        Text {
            id: deviceLabel
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Layout.leftMargin: Style.space(10)
            text: "Processing device"
            textFormat: Text.PlainText
            color: root.controller.foreground
            font.family: root.controller.fontFamily
            font.pixelSize: Style.font.body
            font.bold: true
            wrapMode: Text.Wrap
        }
        Row {
            id: deviceChoices
            Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
            spacing: Style.space(12)
            Repeater {
                model: root.options
                Controls.RadioButton {
                    id: deviceRadio
                    required property var modelData
                    readonly property bool hasCursor: root.controller.cursor === 0 && checked
                    implicitHeight: Style.space(32)
                    padding: Style.space(10)
                    spacing: Style.space(8)
                    text: modelData.label
                    checked: root.device === modelData.value
                    enabled: root.canEdit
                    opacity: enabled ? 1 : 0.4
                    focusPolicy: Qt.NoFocus
                    Accessible.role: Accessible.RadioButton
                    Accessible.name: deviceLabel.text + ": " + modelData.label
                    Accessible.checkable: true
                    Accessible.checked: checked
                    Accessible.focused: hasCursor
                    background: Item {}
                    indicator: Rectangle {
                        x: deviceRadio.leftPadding
                        y: (deviceRadio.height - height) / 2
                        width: Style.space(14)
                        height: width
                        radius: width / 2
                        color: "transparent"
                        border.width: deviceRadio.hasCursor ? 2 : 1
                        border.color: deviceRadio.checked || deviceRadio.hovered ? root.controller.foreground : root.controller.dim
                        Rectangle {
                            anchors.centerIn: parent
                            width: Style.space(6)
                            height: width
                            radius: width / 2
                            color: root.controller.foreground
                            opacity: deviceRadio.checked ? 1 : 0
                            Behavior on opacity { NumberAnimation { duration: 120 } }
                        }
                    }
                    contentItem: Text {
                        text: deviceRadio.text
                        textFormat: Text.PlainText
                        leftPadding: deviceRadio.indicator.width + deviceRadio.spacing
                        verticalAlignment: Text.AlignVCenter
                        color: root.controller.foreground
                        font.family: root.controller.fontFamily
                        font.pixelSize: Style.font.body
                        font.underline: deviceRadio.hasCursor
                    }
                    onClicked: {
                        root.controller.cursor = 0;
                        root.changeDevice(modelData.value);
                    }
                }
            }
        }
    }
    Text {
        x: Style.space(10)
        width: parent.width - Style.space(20)
        visible: text !== ""
        text: root.controller.service.error || root.controller.localError ? ""
            : root.notice || (root.changed && !root.operationBusy ? "Select a model to use this device." : "")
        textFormat: Text.PlainText
        color: root.controller.dim
        font.family: root.controller.fontFamily
        font.pixelSize: Style.font.caption
        wrapMode: Text.WordWrap
        Accessible.role: Accessible.StaticText
        Accessible.name: text
    }
    PanelSeparator {
        id: deviceDivider
        x: Style.space(10)
        width: parent.width - Style.space(20)
        foreground: root.controller.foreground
    }
}
