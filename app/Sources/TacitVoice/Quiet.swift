import Foundation

/// Silences the Mac's own sound output while a key is held, so that music
/// playing through the speakers is not picked up with the whisper. It mutes
/// the output and does not pause the player: that works for every app,
/// including a browser tab, and needs no extra permission.
final class Quiet {
    /// True only while this class is the one that muted the output, so that
    /// a Mac the user had already muted stays muted afterwards.
    private var mutedHere = false

    func begin() {
        guard !mutedHere else { return }
        let alreadyMuted = run("output muted of (get volume settings)")?.booleanValue ?? true
        guard !alreadyMuted else { return }
        if run("set volume output muted true") != nil { mutedHere = true }
    }

    func end() {
        guard mutedHere else { return }
        mutedHere = false
        _ = run("set volume output muted false")
    }

    /// Runs one line of AppleScript addressed to the system itself, not to
    /// another app. Returns nil if it failed.
    private func run(_ source: String) -> NSAppleEventDescriptor? {
        var error: NSDictionary?
        let result = NSAppleScript(source: source)?.executeAndReturnError(&error)
        return error == nil ? result : nil
    }
}
