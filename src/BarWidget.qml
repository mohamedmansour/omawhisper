import QtQuick
import qs.Commons
import qs.Ui
import "." as Local

BarWidget {
    id: root
    moduleName: "mohamedmansour.whisper"
    readonly property bool opened: panel.opened
    readonly property bool popoutSwitchClosing: panel.popoutSwitchClosing
    readonly property real openPanelIndicatorWidth: Style.space(22)
    readonly property color microphoneColor: panel.recording ? panel.urgent : (root.bar ? root.bar.barForeground : Color.foreground)
    implicitWidth: button.implicitWidth
    implicitHeight: button.implicitHeight

    function open() { panel.open(); }
    function close() { panel.close(); }
    function closeForPopoutSwitch() { panel.closeForPopoutSwitch(); }

    Local.Panel {
        id: panel
        bar: root.bar
        settings: root.settings
        anchorItem: button
        hostWidget: root
    }

    WidgetButton {
        id: button
        anchors.fill: parent
        bar: root.bar
        labelVisible: false
        hasVisualContent: true
        fixedWidth: root.vertical ? -1 : Style.space(23) + scaledHorizontalMargin * 2
        horizontalMargin: 8.75
        verticalPadding: 8.75
        tooltipText: panel.statusText
        foreground: root.microphoneColor
        useActiveColor: false
        onPressed: function (mouseButton) {
            if (mouseButton === Qt.MiddleButton)
                panel.service.request({ action: "cancel" });
            else
                panel.toggle();
        }

        Microphone {
            anchors.centerIn: parent
            scale: 0.75
            color: root.microphoneColor
            opacity: panel.recording ? 0 : 1
            Behavior on opacity { NumberAnimation { duration: 120 } }
        }
        AudioMeter {
            anchors.centerIn: parent
            scale: 0.75
            color: root.microphoneColor
            active: panel.recording
            level: panel.service.state.level
            opacity: panel.recording ? 1 : 0
            Behavior on opacity { NumberAnimation { duration: 120 } }
        }
        Rectangle {
            anchors.right: parent.right
            anchors.rightMargin: Style.space(4)
            anchors.top: parent.top
            anchors.topMargin: Style.space(4)
            width: Style.space(4)
            height: width
            radius: width / 2
            visible: panel.busy || panel.service.error !== "" || !panel.service.connected
            color: panel.service.error || !panel.service.connected ? panel.urgent : root.microphoneColor
            SequentialAnimation on opacity {
                running: panel.busy
                loops: Animation.Infinite
                NumberAnimation { to: 0.3; duration: 600 }
                NumberAnimation { to: 1; duration: 600 }
            }
        }
    }
}
