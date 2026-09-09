import QtQuick
import qs.Commons
import qs.Ui

Column {
    id: root
    required property var controller
    property bool adding: false
    property string deleteId: ""
    property var draft: ({ id: "", name: "", engine: "whisper", source: "", architecture: "", quantization: "int8", argv: "[]" })
    readonly property var fields: [
        { key: "name", title: "Model name", help: "A friendly label for this model.", kind: "text" },
        { key: "id", title: "Model ID", help: "A unique short ID, using letters, numbers and hyphens.", kind: "text" },
        { key: "engine", title: "Engine", help: "Weights must match the selected engine.", kind: "choice", options: [
            { value: "whisper", label: "Whisper" }, { value: "parakeet", label: "Parakeet / ONNX" }, { value: "command", label: "Custom command" }
        ] },
        { key: "source", title: "Model repository or path", help: "Hugging Face owner/repo, or an absolute path to a local model directory.", kind: "text" },
        { key: "architecture", title: "ONNX architecture", help: "Required for local ONNX models, e.g. nemo-parakeet-tdt-0.6b-v3.", kind: "text" },
        { key: "quantization", title: "ONNX weights", help: "Choose the format actually present in your ONNX repository.", kind: "choice", options: [
            { value: "int8", label: "INT8" }, { value: "", label: "Unquantized" }
        ] },
        { key: "argv", title: "Custom command arguments", help: "JSON array, e.g. [\"my-asr\", \"{audio}\"]. Prints text to stdout. Custom engine only.", kind: "text" }
    ]
    readonly property int count: adding ? fields.length + 2 : controller.service.state.models.length + 2
    spacing: Style.space(6)

    function rowAt(index) {
        if (adding)
            return index < fields.length ? form.itemAt(index) : (index === fields.length ? saveButton : cancelButton);
        return index === 0 ? addButton : (index === 1 ? unloadButton : entries.itemAt(index - 2));
    }

    function activate(index) {
        if (adding) {
            if (index < fields.length) {
                form.itemAt(index).activate();
                return;
            }
            if (index === fields.length)
                save();
            else
                cancel();
        } else if (index === 0) {
            adding = true;
            controller.cursor = 0;
        } else if (index === 1) {
            controller.service.request({ action: "unload" });
        } else {
            var model = controller.service.state.models[index - 2];
            if (!model)
                return;
            if (deleteId === model.id) {
                controller.service.request({ action: "remove_model", id: model.id });
                deleteId = "";
            } else {
                controller.service.request({ action: model.installed ? "load" : "download", id: model.id });
            }
        }
    }

    function adjust(index, direction) {
        if (adding && index < fields.length)
            form.itemAt(index).adjust(direction);
    }

    function removeSelected() {
        if (adding || controller.cursor < 2)
            return;
        var model = controller.service.state.models[controller.cursor - 2];
        if (model) {
            deleteId = model.id;
            controller.localError = "Remove " + model.name + "? Press Enter to confirm, or Escape to cancel.";
        }
    }

    function cancel() {
        controller.cancelEditor();
        adding = false;
        deleteId = "";
        controller.localError = "";
        controller.cursor = 0;
    }

    function save() {
        var model = { id: draft.id.trim(), name: draft.name.trim(), engine: draft.engine, source: draft.source.trim() };
        if (model.engine === "parakeet" && draft.architecture.trim())
            model.architecture = draft.architecture.trim();
        if (model.engine === "parakeet")
            model.quantization = draft.quantization;
        if (!model.id || !model.name || (model.engine !== "command" && !model.source)) {
            controller.localError = "Enter a name, unique model ID and repository or local path.";
            return;
        }
        if (model.engine === "command") {
            try {
                model.argv = JSON.parse(draft.argv);
                if (!Array.isArray(model.argv) || !model.argv.length || model.argv.some(function (v) { return typeof v !== "string"; }))
                    throw new Error("Enter a nonempty JSON array of strings.");
            } catch (error) {
                controller.localError = "Custom command: " + error.message;
                return;
            }
        }
        controller.service.request({ action: "add_model", model: model });
    }

    Connections {
        target: root.controller.service
        function onCommandSucceeded(request) {
            if (request.action === "add_model") {
                root.cancel();
                root.draft = { id: "", name: "", engine: "whisper", source: "", architecture: "", quantization: "int8", argv: "[]" };
            }
            if (request.action === "remove_model")
                root.controller.localError = "";
        }
    }

    Text {
        width: parent.width
        text: root.adding ? "Bring your own model" : "Local models"
        color: controller.foreground
        font.family: controller.fontFamily
        font.pixelSize: Style.font.subtitle
        font.bold: true
    }
    Text {
        width: parent.width
        text: root.adding
            ? "Whisper uses CTranslate2 weights. Parakeet uses onnx-asr-compatible repositories. A custom command can connect other local engines."
            : "Download a model, then select it to load. No account or cloud transcription. Large models can use several GB of disk and memory."
        color: controller.dim
        font.family: controller.fontFamily
        font.pixelSize: Style.font.caption
        wrapMode: Text.WordWrap
    }
    Button {
        id: addButton
        width: parent.width
        visible: !root.adding
        text: "+ Add model"
        leftAlign: true
        bordered: true
        foreground: controller.foreground
        hasCursor: controller.cursor === 0
        onClicked: root.activate(0)
    }
    Button {
        id: unloadButton
        width: parent.width
        visible: !root.adding
        text: "Unload model from memory"
        leftAlign: true
        foreground: controller.foreground
        hasCursor: controller.cursor === 1
        onClicked: root.activate(1)
    }
    Repeater {
        id: entries
        model: root.adding ? [] : root.controller.service.state.models
        Button {
            id: modelButton
            required property var modelData
            required property int index
            width: root.width
            implicitHeight: Style.space(76)
            hasCursor: controller.cursor === index + 2
            selected: modelData.loaded === true
            foreground: controller.foreground
            onClicked: { controller.cursor = index + 2; root.activate(index + 2); }

            Column {
                anchors.left: parent.left
                anchors.right: status.left
                anchors.margins: Style.space(10)
                anchors.verticalCenter: parent.verticalCenter
                spacing: Style.space(4)
                Text {
                    width: parent.width
                    text: modelButton.modelData.name
                    textFormat: Text.PlainText
                    color: controller.foreground
                    font.family: controller.fontFamily
                    font.pixelSize: Style.font.body
                    font.bold: true
                    elide: Text.ElideRight
                }
                Text {
                    width: parent.width
                    text: modelButton.modelData.description || modelButton.modelData.source || modelButton.modelData.engine
                    textFormat: Text.PlainText
                    color: controller.dim
                    font.family: controller.fontFamily
                    font.pixelSize: Style.font.caption
                    elide: Text.ElideRight
                }
            }
            Text {
                id: status
                anchors.right: parent.right
                anchors.rightMargin: Style.space(10)
                anchors.verticalCenter: parent.verticalCenter
                text: root.deleteId === modelButton.modelData.id ? "Confirm removal"
                    : modelButton.modelData.loaded ? "Loaded"
                    : modelButton.modelData.progress === -1 ? "Downloading"
                    : modelButton.modelData.progress > 0 && modelButton.modelData.progress < 100 ? Math.round(modelButton.modelData.progress) + "%"
                    : modelButton.modelData.installed ? "Load  >" : "Download"
                color: controller.foreground
                font.family: controller.fontFamily
                font.pixelSize: Style.font.bodySmall
            }
        }
    }
    Repeater {
        id: form
        model: root.adding ? root.fields : []
        PreferenceRow {
            required property var modelData
            required property int index
            width: root.width
            controller: root.controller
            spec: modelData
            value: root.draft[modelData.key]
            hasCursor: root.controller.cursor === index
            enabled: (modelData.key !== "argv" || root.draft.engine === "command")
                && (modelData.key !== "architecture" || root.draft.engine === "parakeet")
                && (modelData.key !== "quantization" || root.draft.engine === "parakeet")
            onEngaged: root.controller.cursor = index
            onModified: function (value) {
                var next = Object.assign({}, root.draft);
                next[modelData.key] = value;
                root.draft = next;
            }
        }
    }
    Button {
        id: saveButton
        visible: root.adding
        width: parent.width
        text: "Add model"
        bordered: true
        foreground: controller.foreground
        hasCursor: controller.cursor === root.fields.length
        onClicked: root.save()
    }
    Button {
        id: cancelButton
        visible: root.adding
        width: parent.width
        text: "Cancel"
        foreground: controller.foreground
        hasCursor: controller.cursor === root.fields.length + 1
        onClicked: root.cancel()
    }
}
