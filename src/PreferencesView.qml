import QtQuick
import qs.Commons

Column {
    id: root
    required property var controller
    required property var rows
    readonly property int count: rows.length
    spacing: Style.space(3)

    function rowAt(index) { return entries.itemAt(index); }
    function activate(index) { var row = rowAt(index); if (row) row.activate(); }
    function adjust(index, direction) { var row = rowAt(index); if (row) row.adjust(direction); }

    Repeater {
        id: entries
        model: root.rows
        PreferenceRow {
            required property var modelData
            required property int index
            width: root.width
            controller: root.controller
            spec: modelData
            value: root.controller.config[modelData.key]
            hasCursor: root.controller.cursor === index
            enabled: root.controller.service.connected && !root.controller.busy
                && !(modelData.kind === "action" && root.controller.recording)
                && !(modelData.whisperOnly && !root.controller.isWhisper)
                && !(modelData.builtinOnly && root.controller.selectedModel && root.controller.selectedModel.engine === "command")
                && !(modelData.key === "beam_size" && !root.controller.config.use_beam_search)
            onEngaged: root.controller.cursor = index
            onModified: function (value) { root.controller.configure(modelData.key, value); }
            onActivated: root.controller.preferenceAction(modelData.key)
        }
    }
}
