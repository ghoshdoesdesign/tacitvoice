import AppKit

/// Watches one modifier key system-wide and reports when it goes down and up.
/// Needs Accessibility permission to see key events in other apps.
final class HoldKey {
    var onPress: (() -> Void)?
    var onRelease: (() -> Void)?

    private let keyCode: UInt16 = 61  // right Option
    /// Device-specific bit for the right Option key. The generic `.option`
    /// flag stays set while the left one is held, so it cannot be used.
    private let deviceMask: UInt = 0x40
    private var monitors: [Any] = []
    private var isDown = false

    func start() {
        let handle: (NSEvent) -> Void = { [weak self] event in self?.handle(event) }
        if let m = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged, handler: handle) {
            monitors.append(m)
        }
        let local = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { event in
            handle(event)
            return event
        }
        if let m = local { monitors.append(m) }
    }

    private func handle(_ event: NSEvent) {
        guard event.keyCode == keyCode else { return }
        let down = event.modifierFlags.rawValue & deviceMask != 0
        guard down != isDown else { return }
        isDown = down
        if down { onPress?() } else { onRelease?() }
    }
}
