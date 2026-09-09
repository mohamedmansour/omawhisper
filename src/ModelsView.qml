import QtQuick
import qs.Commons
import qs.Ui
import "Settings.js" as Settings

Column {
    id: root
    required property var controller
    property bool adding: false
    property string deleteId: ""
    property alias processing: deviceControl
    readonly property var models: Settings.setupModels(controller.service.state.models, { acceleration: deviceControl.device })
    readonly property int modelOffset: deviceControl.count
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
    readonly property int count: adding ? fields.length + 2 : models.length + modelOffset
    spacing: Style.space(6)

    function rowAt(index) {
        if (index < 0)
            return null;
        if (adding)
            return index < fields.length ? form.itemAt(index) : (index === fields.length ? saveButton : cancelButton);
        if (index < deviceControl.count)
            return deviceControl.rowAt(index);
        return entries.itemAt(index - modelOffset);
    }

    function loadingModel(identifier) {
        return deviceControl.running && deviceControl.draft.model === identifier;
    }

    function startAdding() {
        if (!controller.service.connected || controller.busy || controller.recording)
            return;
        controller.cancelEditor();
        adding = true;
        controller.cursor = 0;
        controller.resetScroll();
        controller.focusPanel();
    }

    function activate(index) {
        if (!adding && index >= 0 && index < deviceControl.count) {
            deviceControl.activate(index);
            return;
        }
        if (!controller.service.connected)
            return;
        if (!adding && models[index - modelOffset] && loadingModel(models[index - modelOffset].id)) {
            deviceControl.cancelJob();
            return;
        }
        if (controller.busy || controller.recording)
            return;
        if (adding) {
            if (index < fields.length) {
                form.itemAt(index).activate();
                return;
            }
            if (index === fields.length)
                save();
            else
                cancel();
        } else {
            var model = models[index - modelOffset];
            if (!model)
                return;
            if (deleteId === model.id) {
                controller.service.request({ action: "remove_model", id: model.id });
                deleteId = "";
            } else {
                deviceControl.submit(model.id);
            }
        }
    }

    function adjust(index, direction) {
        if (adding && index < fields.length)
            form.itemAt(index).adjust(direction);
        else if (!adding && index < deviceControl.count)
            deviceControl.adjust(index, direction);
    }

    function removeSelected() {
        if (adding || controller.cursor < modelOffset || controller.busy)
            return;
        var model = models[controller.cursor - modelOffset];
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
        if (!controller.service.connected || controller.busy)
            return;
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

    ProcessingView {
        id: deviceControl
        width: parent.width
        visible: !root.adding
        controller: root.controller
    }
    Text {
        width: parent.width
        visible: root.adding
        text: "Bring your own model"
        color: controller.foreground
        font.family: controller.fontFamily
        font.pixelSize: Style.font.subtitle
        font.bold: true
    }
    Text {
        width: parent.width
        visible: root.adding
        text: "Whisper uses CTranslate2 weights. Parakeet uses onnx-asr-compatible repositories. A custom command can connect other local engines."
        color: controller.dim
        font.family: controller.fontFamily
        font.pixelSize: Style.font.caption
        wrapMode: Text.WordWrap
    }
    Repeater {
        id: entries
        model: root.adding ? [] : root.models
        Button {
            id: modelButton
            required property var modelData
            required property int index
            width: root.width
            implicitHeight: Style.space(42)
            hasCursor: controller.cursor === index + root.modelOffset
            selected: modelData.loaded === true || root.loadingModel(modelData.id)
            foreground: controller.foreground
            enabled: controller.service.connected && !controller.recording
                && (!controller.busy || (root.loadingModel(modelData.id) && !deviceControl.cancelling))
            Accessible.name: modelData.name + ". " + status.text
            Accessible.description: modelData.memory_error || modelData.compatibility.detail
            onClicked: { controller.cursor = index + root.modelOffset; root.activate(index + root.modelOffset); }

            Text {
                anchors.left: parent.left
                anchors.right: status.left
                anchors.margins: Style.space(10)
                anchors.verticalCenter: parent.verticalCenter
                text: modelButton.modelData.name
                textFormat: Text.PlainText
                color: controller.foreground
                font.family: controller.fontFamily
                font.pixelSize: Style.font.body
                elide: Text.ElideMiddle
            }
            Text {
                id: status
                anchors.right: parent.right
                anchors.rightMargin: Style.space(10)
                anchors.verticalCenter: parent.verticalCenter
                width: Math.min(implicitWidth, parent.width * 0.5)
                text: root.deleteId === modelButton.modelData.id ? "Confirm removal"
                    : root.loadingModel(modelButton.modelData.id)
                        ? (deviceControl.cancelling ? "Cancelling..." : Settings.modelProgress(deviceControl.phase) + " / Cancel")
                    : Settings.modelStatus(modelButton.modelData, controller.config, controller.service.state.runtime, deviceControl.device)
                textFormat: Text.PlainText
                color: controller.foreground
                font.family: controller.fontFamily
                font.pixelSize: Style.font.caption
                elide: Text.ElideRight
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
            enabled: root.controller.service.connected && !root.controller.busy
                && (modelData.key !== "argv" || root.draft.engine === "command")
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
        enabled: controller.service.connected && !controller.busy
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
