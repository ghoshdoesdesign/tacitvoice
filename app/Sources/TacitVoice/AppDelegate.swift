import AVFoundation
import AppKit
import ApplicationServices

final class AppDelegate: NSObject, NSApplicationDelegate {
    private enum Phase { case idle, recording, processing }

    private let statusItem = NSStatusBar.system.statusItem(
        withLength: NSStatusItem.variableLength)
    private let serverLine = NSMenuItem(title: "Server: checking…", action: nil, keyEquivalent: "")
    private let contextItem = NSMenuItem(
        title: "Send text before cursor as context", action: #selector(toggleContext),
        keyEquivalent: "")

    private let hotkey = HoldKey.rightOption
    /// Held instead of right Option, the words go to the OpenClaw agent as a
    /// task for it to do, and are not typed.
    private let commandKey = HoldKey.rightCommand
    private let capture = Capture()
    private let client = ServerClient()
    private let pill = Pill()
    private let caption = Caption()

    private var phase = Phase.idle
    private var commanding = false
    private var pressedAt = Date()
    private var targetApp = ""
    private var textBefore = ""
    private var limitTimer: Timer?

    private let contextKey = "sendFieldContext"
    /// Shorter holds are treated as an accidental tap of the key.
    private let minimumHold = 0.35
    private let maximumHold = 60.0

    func applicationDidFinishLaunching(_ notification: Notification) {
        UserDefaults.standard.register(defaults: [contextKey: true])
        buildMenu()
        setIcon(recording: false)
        requestPermissions()

        hotkey.onPress = { [weak self] in self?.keyDown(command: false) }
        hotkey.onRelease = { [weak self] in self?.keyUp(command: false) }
        hotkey.start()
        commandKey.onPress = { [weak self] in self?.keyDown(command: true) }
        commandKey.onRelease = { [weak self] in self?.keyUp(command: true) }
        commandKey.start()

        checkServer()
        Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            self?.checkServer()
        }
    }

    // MARK: Menu

    private func buildMenu() {
        let menu = NSMenu()
        let hint = NSMenuItem(
            title: "Hold right Option, whisper or mouth, release", action: nil, keyEquivalent: "")
        hint.isEnabled = false
        let commandHint = NSMenuItem(
            title: "Hold right Command to give OpenClaw a task", action: nil, keyEquivalent: "")
        commandHint.isEnabled = false
        serverLine.isEnabled = false
        contextItem.target = self
        contextItem.state = UserDefaults.standard.bool(forKey: contextKey) ? .on : .off
        let access = NSMenuItem(
            title: "Open Accessibility settings…", action: #selector(openAccessibility),
            keyEquivalent: "")
        access.target = self
        menu.addItem(hint)
        menu.addItem(commandHint)
        menu.addItem(serverLine)
        menu.addItem(.separator())
        let history = NSMenuItem(
            title: "Open history…", action: #selector(openHistory), keyEquivalent: "")
        history.target = self
        menu.addItem(history)
        menu.addItem(contextItem)
        menu.addItem(access)
        menu.addItem(.separator())
        menu.addItem(
            NSMenuItem(
                title: "Quit TacitVoice", action: #selector(NSApplication.terminate(_:)),
                keyEquivalent: "q"))
        statusItem.menu = menu
    }

    private func setIcon(recording: Bool) {
        let name = recording ? "mouth.fill" : "mouth"
        if let image = NSImage(systemSymbolName: name, accessibilityDescription: "TacitVoice") {
            statusItem.button?.image = image
            statusItem.button?.title = ""
        } else {
            statusItem.button?.title = recording ? "●" : "TV"
        }
    }

    @objc private func toggleContext() {
        let on = !UserDefaults.standard.bool(forKey: contextKey)
        UserDefaults.standard.set(on, forKey: contextKey)
        contextItem.state = on ? .on : .off
    }

    /// The history page is served by the local server. The token goes after
    /// the #, which the browser keeps to itself and the page's script reads.
    @objc private func openHistory() {
        if let url = URL(string: "http://127.0.0.1:8765/dashboard#\(client.token())") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func openAccessibility() {
        let link = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        if let url = URL(string: link) { NSWorkspace.shared.open(url) }
    }

    private func checkServer() {
        client.health { [weak self] health in
            guard let self = self else { return }
            guard let health = health else {
                self.serverLine.title = "Server: not running (start scripts/server.sh)"
                return
            }
            if let error = health.error {
                self.serverLine.title = "Server error: \(error)"
            } else if !health.ready {
                self.serverLine.title = "Server: loading models…"
            } else {
                var parts = ["whisper"]
                if health.lips { parts.append("lips") }
                if health.cleanup { parts.append("cleanup") }
                self.serverLine.title = "Server: ready (\(parts.joined(separator: ", ")))"
            }
        }
    }

    // MARK: Permissions

    private var accessibilityGranted: Bool { AXIsProcessTrusted() }

    private func requestPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { _ in }
        AVCaptureDevice.requestAccess(for: .audio) { _ in }
        // Shows the system prompt that links to the Accessibility list. The
        // hold key and the paste both depend on it.
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        _ = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
    }

    private func missingPermission() -> String? {
        if AVCaptureDevice.authorizationStatus(for: .video) != .authorized {
            return "Allow Camera for TacitVoice in System Settings"
        }
        if AVCaptureDevice.authorizationStatus(for: .audio) != .authorized {
            return "Allow Microphone for TacitVoice in System Settings"
        }
        if !accessibilityGranted {
            return "Allow Accessibility for TacitVoice in System Settings"
        }
        return nil
    }

    // MARK: Hold to talk

    private func keyDown(command: Bool) {
        guard phase == .idle else { return }
        if let problem = missingPermission() {
            pill.show(.note(problem), hideAfter: 4)
            return
        }
        // Read what the text will join now, while the user's app has focus.
        commanding = command
        caption.hide()
        // The server lists a task once in the history, not also as dictation.
        targetApp = command ? "TacitVoice command" : FocusContext.appName()
        let sendContext = UserDefaults.standard.bool(forKey: contextKey)
        // A command is not typed anywhere, so the field's text has no bearing on it.
        textBefore = sendContext && !command ? FocusContext.textBeforeCursor() : ""

        phase = .recording
        pressedAt = Date()
        setIcon(recording: true)
        pill.show(.observing)
        capture.start { [weak self] problem in
            guard let self = self else { return }
            self.phase = .idle
            self.setIcon(recording: false)
            self.pill.show(.note(problem), hideAfter: 4)
        }
        limitTimer = Timer.scheduledTimer(withTimeInterval: maximumHold, repeats: false) {
            [weak self] _ in
            guard let self = self else { return }
            self.keyUp(command: self.commanding)
        }
    }

    private func keyUp(command: Bool) {
        // Only the key that started the recording ends it.
        guard phase == .recording, command == commanding else { return }
        limitTimer?.invalidate()
        let held = Date().timeIntervalSince(pressedAt)
        phase = .processing
        setIcon(recording: false)
        pill.show(.polishing)

        capture.stop { [weak self] clip in
            guard let self = self else { return }
            if held < self.minimumHold || clip.frames.count < 8 {
                self.phase = .idle
                self.pill.hide()
                return
            }
            self.client.transcribe(clip, app: self.targetApp, textBeforeCursor: self.textBefore) {
                result in
                self.phase = .idle
                switch result {
                case .failure(let error):
                    self.pill.show(.note(error.text), hideAfter: 4)
                case .success(let transcript):
                    if self.commanding {
                        self.runCommand(transcript)
                    } else {
                        self.finish(transcript)
                    }
                }
            }
        }
    }

    /// The orb keeps spinning with the understood words above it while the
    /// agent works; then the orb closes and the answer takes their place.
    private func runCommand(_ transcript: Transcript) {
        guard !transcript.text.isEmpty else {
            pill.show(.note(transcript.warnings.first ?? "Nothing recognised"), hideAfter: 3)
            return
        }
        phase = .processing  // no new recording until the agent has answered
        caption.show(transcript.text)  // the orb is already spinning from the key release
        client.command(transcript.text) { [weak self] result in
            guard let self = self else { return }
            self.phase = .idle
            switch result {
            case .failure(let error):
                self.pill.hide()
                self.caption.show(error.text, hideAfter: 6)
            case .success(let reply):
                self.pill.show(.done, hideAfter: 1.4)
                self.caption.show(reply, hideAfter: Caption.readingTime(reply))
            }
        }
    }

    private func finish(_ transcript: Transcript) {
        guard !transcript.text.isEmpty else {
            let reason = transcript.warnings.first ?? "Nothing recognised"
            pill.show(.note(reason), hideAfter: 3)
            return
        }
        var text = transcript.text
        // Keep a space between this and what is already in the field.
        if let last = textBefore.last, !last.isWhitespace, !"([{\"'".contains(last) {
            text = " " + text
        }
        Inserter.paste(text)

        pill.show(.inserted, hideAfter: 1.4)
    }
}
