import AppKit

/// Watches one modifier key system-wide and reports when it goes down and up.
/// Needs Accessibility permission to see key events in other apps.
final class HoldKey {
    var onPress: (() -> Void)?
    var onRelease: (() -> Void)?

    private let keyCode: UInt16
    /// Device-specific bit for this one key. The generic flag (`.option`,
    /// `.command`) stays set while the key on the other side is held, so it
    /// cannot be used.
    private let deviceMask: UInt

    static let rightOption = HoldKey(keyCode: 61, deviceMask: 0x40)
    static let rightCommand = HoldKey(keyCode: 54, deviceMask: 0x10)

    init(keyCode: UInt16, deviceMask: UInt) {
        self.keyCode = keyCode
        self.deviceMask = deviceMask
    }

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
