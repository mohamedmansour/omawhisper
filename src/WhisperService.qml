import QtQuick
import Quickshell.Io

Item {
    id: root
    property bool connected: false
    property string transportError: ""
    property var state: ({
        phase: "idle", level: 0, elapsed: 0, error: "", message: "",
        transcript: "", model_name: "", config: {}, models: [], devices: [], history: []
    })
    property var pending: []
    property var current: null
    readonly property string executable: decodeURIComponent(Qt.resolvedUrl("../scripts/omawhisper").toString().replace(/^file:\/\//, ""))
    readonly property string error: transportError || state.error || ""
    signal commandSucceeded(var request)

    function request(payload) {
        pending = pending.concat([payload]);
        next();
    }

    function next() {
        if (command.running || current || !pending.length)
            return;
        current = pending[0];
        pending = pending.slice(1);
        command.command = [executable, "request", JSON.stringify(current)];
        command.running = true;
    }

    Process {
        id: watcher
        command: [root.executable, "watch"]
        running: true
        stdout: SplitParser {
            onRead: function (data) {
                try {
                    var update = JSON.parse(data);
                    if (!update.config || !Array.isArray(update.models))
                        throw new Error("Invalid service state");
                    root.state = update;
                    if (!root.connected)
                        root.transportError = "";
                    root.connected = true;
                    retry.interval = 1000;
                } catch (error) {
                    root.transportError = "Cannot read dictation service: " + error.message;
                }
            }
        }
        stderr: StdioCollector {}
        onExited: {
            root.connected = false;
            retry.restart();
        }
    }

    Timer {
        id: retry
        interval: 1000
        onTriggered: {
            interval = Math.min(interval * 2, 10000);
            watcher.running = true;
        }
    }

    Process {
        id: command
        stdout: StdioCollector { id: response }
        stderr: StdioCollector { id: errors }
        onExited: function (exitCode) {
            var completed = root.current;
            root.current = null;
            if (exitCode !== 0) {
                var detail = errors.text.trim() || response.text.trim();
                try {
                    detail = JSON.parse(response.text).error || detail;
                } catch (error) {
                    // CLI launch failures have plain-text diagnostics, not JSON.
                }
                root.transportError = detail || "Dictation service request failed.";
            } else {
                root.transportError = "";
                root.commandSucceeded(completed);
            }
            Qt.callLater(root.next);
        }
    }
}
