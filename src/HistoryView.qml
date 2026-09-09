import QtQuick
import qs.Commons
import qs.Ui

Column {
    id: root
    required property var controller
    property bool confirmClear: false
    readonly property int count: 2 + controller.service.state.history.length
    spacing: Style.space(8)

    function rowAt(index) { return index === 0 ? lastButton : (index === 1 ? clearButton : entries.itemAt(index - 2)); }
    function adjust(index, direction) {}
    function activate(index) {
        if (index === 0) {
            if (controller.service.state.transcript)
                controller.service.request({ action: "copy", text: controller.service.state.transcript });
            else
                controller.localError = "There is no transcript to copy yet.";
        } else if (index === 1) {
            if (!confirmClear) {
                confirmClear = true;
                controller.localError = "Clear saved transcripts? Press Enter again to confirm.";
            } else {
                controller.service.request({ action: "clear_history" });
                confirmClear = false;
                controller.localError = "";
            }
        } else {
            var entry = controller.service.state.history[index - 2];
            if (entry)
                controller.service.request({ action: "copy", text: entry.text });
        }
    }

    Text {
        width: parent.width
        text: controller.config.history ? "History stays on this device. Select a transcript to copy it." : "History is off. Your latest transcript stays in memory until the service exits. Previously saved history remains on disk until you clear it."
        color: controller.dim
        font.family: controller.fontFamily
        font.pixelSize: Style.font.caption
        wrapMode: Text.WordWrap
    }
    Button {
        id: lastButton
        width: parent.width
        text: "Copy latest transcript"
        leftAlign: true
        bordered: true
        hasCursor: controller.cursor === 0
        foreground: controller.foreground
        onClicked: root.activate(0)
    }
    Text {
        width: parent.width
        text: controller.service.state.transcript || "Your next dictation will appear here."
        textFormat: Text.PlainText
        color: controller.foreground
        font.family: controller.fontFamily
        font.pixelSize: Style.font.body
        wrapMode: Text.WordWrap
        maximumLineCount: 8
        elide: Text.ElideRight
    }
    Button {
        id: clearButton
        width: parent.width
        text: root.confirmClear ? "Confirm: clear saved history" : "Clear saved history"
        leftAlign: true
        hasCursor: controller.cursor === 1
        foreground: controller.foreground
        onClicked: root.activate(1)
    }
    Repeater {
        id: entries
        model: root.controller.service.state.history
        Button {
            required property var modelData
            required property int index
            width: root.width
            implicitHeight: Style.space(76)
            foreground: controller.foreground
            hasCursor: controller.cursor === index + 2
            onClicked: root.activate(index + 2)
            Text {
                anchors.fill: parent
                anchors.margins: Style.space(10)
                text: modelData.text
                textFormat: Text.PlainText
                color: controller.foreground
                font.family: controller.fontFamily
                font.pixelSize: Style.font.bodySmall
                wrapMode: Text.WordWrap
                maximumLineCount: 3
                elide: Text.ElideRight
            }
        }
    }
}
