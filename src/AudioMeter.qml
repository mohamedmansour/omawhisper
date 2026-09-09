import QtQuick

Item {
    id: root
    property real level: 0
    property color color: "white"
    property bool active: false
    implicitWidth: 23
    implicitHeight: 20

    Row {
        anchors.centerIn: parent
        spacing: 2
        Repeater {
            model: [0.5, 0.85, 1, 0.7, 0.4]
            Rectangle {
                required property real modelData
                width: 3
                height: 3 + (root.height - 3) * modelData * (root.active ? Math.min(1, Math.sqrt(Math.max(0, root.level)) * 2.2) : 0)
                anchors.verticalCenter: parent.verticalCenter
                radius: width / 2
                color: root.color
                Behavior on height { NumberAnimation { duration: 75; easing.type: Easing.OutCubic } }
            }
        }
    }
}
