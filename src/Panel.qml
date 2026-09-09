import QtQuick
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Ui
import "Settings.js" as Settings

Panel {
    id: root
    moduleName: "mohamedmansour.whisper"
    ipcTarget: moduleName
    property var anchorItem: null
    property var hostWidget: null
    property alias service: backend
    property int tab: 0
    property int cursor: -1
    property var editor: null
    property string localError: ""
    readonly property var config: {
        var values = Object.assign({}, backend.state.config);
        if (bar && typeof bar.layoutEntries === "function") {
            var sections = ["left", "center", "right"];
            for (var i = 0; i < sections.length; i++) {
                if (bar.layoutEntries(sections[i]).some(function (entry) { return (typeof entry === "string" ? entry : entry.id) === root.moduleName; }))
                    values.bar_section = sections[i];
            }
        }
        return values;
    }
    readonly property color foreground: bar ? bar.foreground : Color.foreground
    readonly property color urgent: bar ? bar.urgent : Color.urgent
    readonly property color dim: Qt.darker(foreground, 1.45)
    readonly property string fontFamily: bar ? bar.fontFamily : Style.font.family
    readonly property var selectedModel: backend.state.models.find(function (m) { return m.id === root.config.model; })
    readonly property bool isWhisper: !selectedModel || selectedModel.engine === "whisper"
    readonly property var views: [generalView, modelsView, audioView, advancedView, historyView]
    readonly property var currentView: views[tab]
    readonly property var tabNames: ["General", "Models", "Audio", "Advanced", "History"]
    readonly property bool recording: backend.state.phase === "recording"
    readonly property bool busy: ["loading", "transcribing", "downloading"].indexOf(backend.state.phase) !== -1
    readonly property string statusText: !backend.connected ? "Service offline"
        : recording ? "Listening  " + Math.floor(backend.state.elapsed) + "s"
        : busy ? backend.state.phase.charAt(0).toUpperCase() + backend.state.phase.slice(1)
        : backend.error ? "Needs attention"
        : selectedModel && selectedModel.installed ? (selectedModel.loaded ? "Ready - " : "On demand - ") + selectedModel.name
        : "Download a model to begin"

    function open() {
        cursor = -1;
        controller.show();
        backend.request({ action: "devices" });
    }
    function close() {
        cancelEditor();
        controller.hide();
    }
    function focusPanel() { keyCatcher.forceActiveFocus(); }
    function cancelEditor() { if (editor) editor.finish(false); }
    function changeTab(index) {
        cancelEditor();
        tab = (index + views.length) % views.length;
        cursor = -1;
        modelsView.deleteId = "";
        historyView.confirmClear = false;
        localError = "";
        scroll.contentY = 0;
        focusPanel();
    }
    function move(dx, dy) {
        if (dy) {
            cursor = Settings.moveCursor(cursor, dy, currentView.count);
            revealCursor();
        } else if (cursor < 0) {
            changeTab(tab + dx);
        } else {
            currentView.adjust(cursor, dx);
        }
    }
    function revealCursor() {
        Qt.callLater(function () {
            var row = root.currentView.rowAt(root.cursor);
            if (!row)
                return;
            var point = row.mapToItem(body, 0, 0);
            if (point.y < scroll.contentY)
                scroll.contentY = point.y;
            else if (point.y + row.height > scroll.contentY + scroll.height)
                scroll.contentY = Math.max(0, point.y + row.height - scroll.height);
        });
    }
    function activate() {
        if (cursor < 0)
            move(0, 1);
        else
            currentView.activate(cursor);
    }
    function configure(key, value) {
        localError = "";
        var values = {};
        values[key] = value;
        backend.request({ action: "configure", values: values });
    }

    onRecordingChanged: if (recording) close()
    onCursorChanged: {
        if (modelsView.deleteId) {
            modelsView.deleteId = "";
            localError = "";
        }
        historyView.confirmClear = false;
    }

    WhisperService {
        id: backend
        onCommandSucceeded: function (request) {
            if (request.action === "configure" && request.values.bar_section !== undefined) {
                barMove.command = ["omarchy", "bar", "move", root.moduleName, "--section", request.values.bar_section];
                barMove.running = true;
            }
        }
    }
    Process {
        id: barMove
        stderr: StdioCollector { id: moveError }
        onExited: function (code) {
            if (code !== 0)
                root.localError = "Could not move microphone: " + moveError.text.trim();
        }
    }

    KeyboardPanel {
        id: popup
        anchorItem: root.anchorItem
        owner: root.hostWidget || root
        bar: root.bar
        open: root.opened
        focusTarget: keyCatcher
        contentWidth: fittedContentWidth(Style.space(580))
        contentHeight: fittedContentHeight(chrome.implicitHeight + Style.space(410), Style.space(640))

        PanelKeyCatcher {
            id: keyCatcher
            anchors.fill: parent
            blocked: root.editor !== null
            onMoveRequested: function (dx, dy) { root.move(dx, dy); }
            onActivateRequested: root.activate()
            onTabRequested: function (direction) { root.changeTab(root.tab + direction); }
            onCloseRequested: {
                if (modelsView.adding || modelsView.deleteId)
                    modelsView.cancel();
                else if (root.localError) {
                    root.localError = "";
                    historyView.confirmClear = false;
                } else
                    root.close();
            }
            onDeleteRequested: if (root.tab === 1) modelsView.removeSelected()

            Column {
                id: chrome
                width: parent.width
                spacing: Style.space(12)
                Row {
                    width: parent.width
                    spacing: Style.space(12)
                    Microphone {
                        width: Style.space(24)
                        height: Style.space(26)
                        anchors.verticalCenter: parent.verticalCenter
                        color: root.foreground
                    }
                    Column {
                        width: parent.width - Style.space(40)
                        spacing: Style.space(4)
                        Text {
                            text: "Omawhisper"
                            color: root.foreground
                            font.family: root.fontFamily
                            font.pixelSize: Style.font.heading
                            font.bold: true
                        }
                        Text {
                            width: parent.width
                            text: root.statusText
                            textFormat: Text.PlainText
                            color: root.dim
                            font.family: root.fontFamily
                            font.pixelSize: Style.font.caption
                            elide: Text.ElideRight
                        }
                    }
                }
                Rectangle {
                    width: parent.width
                    height: Style.space(38)
                    color: Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.05)
                    radius: Style.cornerRadius
                    Text {
                        anchors.centerIn: parent
                        text: (root.config.shortcut || "SUPER+ALT+V").split("+").join(" + ")
                            + (root.config.activation === "hold" ? "   Hold to speak" : "   Press to start / stop")
                        color: root.foreground
                        font.family: root.fontFamily
                        font.pixelSize: Style.font.bodySmall
                    }
                }
                Row {
                    width: parent.width
                    Repeater {
                        model: root.tabNames
                        Button {
                            required property string modelData
                            required property int index
                            width: parent.width / root.tabNames.length
                            text: modelData
                            selected: root.tab === index
                            hasCursor: root.cursor < 0 && root.tab === index
                            foreground: root.foreground
                            fontFamily: root.fontFamily
                            fontSize: Style.font.bodySmall
                            horizontalPadding: 0
                            onClicked: root.changeTab(index)
                        }
                    }
                }
                Text {
                    width: parent.width
                    visible: !backend.connected || text !== ""
                    text: !backend.connected ? "Run scripts/install.sh from the Omawhisper folder to start the local service."
                        : root.localError || backend.error || backend.state.message || ""
                    textFormat: Text.PlainText
                    color: root.localError || backend.error ? root.urgent : root.dim
                    font.family: root.fontFamily
                    font.pixelSize: Style.font.caption
                    wrapMode: Text.WrapAnywhere
                    maximumLineCount: 4
                    elide: Text.ElideRight
                }
            }

            Flickable {
                id: scroll
                anchors.top: chrome.bottom
                anchors.topMargin: Style.space(10)
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: footer.top
                anchors.bottomMargin: Style.space(10)
                contentWidth: width
                contentHeight: body.implicitHeight
                boundsBehavior: Flickable.StopAtBounds
                clip: true
                interactive: contentHeight > height
                Column {
                    id: body
                    width: scroll.width
                    PreferencesView {
                        id: generalView
                        width: parent.width
                        visible: root.tab === 0
                        controller: root
                        rows: Settings.general(root.config)
                    }
                    ModelsView {
                        id: modelsView
                        width: parent.width
                        visible: root.tab === 1
                        controller: root
                    }
                    PreferencesView {
                        id: audioView
                        width: parent.width
                        visible: root.tab === 2
                        controller: root
                        rows: Settings.audio(backend.state.devices)
                    }
                    PreferencesView {
                        id: advancedView
                        width: parent.width
                        visible: root.tab === 3
                        controller: root
                        rows: Settings.advanced()
                    }
                    HistoryView {
                        id: historyView
                        width: parent.width
                        visible: root.tab === 4
                        controller: root
                    }
                }
            }
            Text {
                id: footer
                anchors.bottom: parent.bottom
                width: parent.width
                text: root.editor ? "Enter save   Esc cancel" : "Arrows navigate / adjust   Enter select   Tab switch tab   Esc close" + (root.tab === 1 && !modelsView.adding ? "   X remove" : "")
                color: root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                wrapMode: Text.WordWrap
            }
        }
    }
}
