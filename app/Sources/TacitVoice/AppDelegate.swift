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

    private let hotkey = HoldKey()
    private let capture = Capture()
    private let client = ServerClient()
    private let pill = Pill()

    private var phase = Phase.idle
    private var pressedAt = Date()
    private var anchor: NSRect?
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

        hotkey.onPress = { [weak self] in self?.keyDown() }
        hotkey.onRelease = { [weak self] in self?.keyUp() }
        hotkey.start()
        capture.onFirstFrame = { [weak self] in
            guard let self = self, self.phase == .recording else { return }
            self.pill.show("● Listening", near: self.anchor)
        }

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
        serverLine.isEnabled = false
        contextItem.target = self
        contextItem.state = UserDefaults.standard.bool(forKey: contextKey) ? .on : .off
        let access = NSMenuItem(
            title: "Open Accessibility settings…", action: #selector(openAccessibility),
            keyEquivalent: "")
        access.target = self
        menu.addItem(hint)
        menu.addItem(serverLine)
        menu.addItem(.separator())
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

    private func keyDown() {
        guard phase == .idle else { return }
        // Read where the text will go now, while the user's app has focus.
        anchor = FocusContext.caretRect()
        if let problem = missingPermission() {
            pill.show(problem, near: anchor, hideAfter: 4)
            return
        }
        targetApp = FocusContext.appName()
        let sendContext = UserDefaults.standard.bool(forKey: contextKey)
        textBefore = sendContext ? FocusContext.textBeforeCursor() : ""

        phase = .recording
        pressedAt = Date()
        setIcon(recording: true)
        pill.show("Starting camera…", near: anchor)
        capture.start { [weak self] problem in
            guard let self = self else { return }
            self.phase = .idle
            self.setIcon(recording: false)
            self.pill.show(problem, near: self.anchor, hideAfter: 4)
        }
        limitTimer = Timer.scheduledTimer(withTimeInterval: maximumHold, repeats: false) {
            [weak self] _ in self?.keyUp()
        }
    }

    private func keyUp() {
        guard phase == .recording else { return }
        limitTimer?.invalidate()
        let held = Date().timeIntervalSince(pressedAt)
        phase = .processing
        setIcon(recording: false)

        capture.stop { [weak self] clip in
            guard let self = self else { return }
            if held < self.minimumHold || clip.frames.count < 8 {
                self.phase = .idle
                self.pill.hide()
                return
            }
            self.pill.show("Transcribing…", near: self.anchor)
            self.client.transcribe(clip, app: self.targetApp, textBeforeCursor: self.textBefore) {
                result in
                self.phase = .idle
                switch result {
                case .failure(let error):
                    self.pill.show(error.text, near: self.anchor, hideAfter: 4)
                case .success(let transcript):
                    self.finish(transcript)
                }
            }
        }
    }

    private func finish(_ transcript: Transcript) {
        guard !transcript.text.isEmpty else {
            let reason = transcript.warnings.first ?? "Nothing recognised"
            pill.show(reason, near: anchor, hideAfter: 3)
            return
        }
        var text = transcript.text
        // Keep a space between this and what is already in the field.
        if let last = textBefore.last, !last.isWhitespace, !"([{\"'".contains(last) {
            text = " " + text
        }
        Inserter.paste(text)

        var note = transcript.mode == "silent" ? "Silent (beta)" : "Whisper"
        if let warning = transcript.warnings.first { note += " · " + warning }
        pill.show(note, near: anchor, hideAfter: transcript.warnings.isEmpty ? 1.2 : 3)
    }
}
