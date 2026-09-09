import QtQuick

Item {
    id: root
    property color color: "white"
    implicitWidth: 20
    implicitHeight: 22
    Rectangle {
        width: 7
        height: 12
        radius: 4
        color: "transparent"
        border.width: 1.6
        border.color: root.color
        anchors.horizontalCenter: parent.horizontalCenter
        y: 1
    }
    Canvas {
        id: drawing
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();
            ctx.strokeStyle = root.color;
            ctx.lineWidth = 1.6;
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.arc(width / 2, 10, 6.5, 0, Math.PI, false);
            ctx.moveTo(width / 2, 16.5);
            ctx.lineTo(width / 2, 20);
            ctx.moveTo(width / 2 - 3, 20);
            ctx.lineTo(width / 2 + 3, 20);
            ctx.stroke();
        }
        Connections {
            target: root
            function onColorChanged() { drawing.requestPaint(); }
        }
    }
}
