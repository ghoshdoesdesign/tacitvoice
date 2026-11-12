import AppKit

/// The small floating label near the cursor: the app's whole interface.
final class Pill {
    private let panel: NSPanel
    private let label = NSTextField(labelWithString: "")
    private var hideWork: DispatchWorkItem?

    init() {
        panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 200, height: 30),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: true)
        panel.level = .statusBar
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        let background = NSVisualEffectView(frame: panel.contentLayoutRect)
        background.material = .hudWindow
        background.state = .active
        background.wantsLayer = true
        background.layer?.cornerRadius = 15
        background.layer?.masksToBounds = true
        background.autoresizingMask = [.width, .height]

        label.font = .systemFont(ofSize: 13, weight: .medium)
        label.textColor = .white
        label.lineBreakMode = .byTruncatingTail
        background.addSubview(label)
        panel.contentView = background
    }

    /// Show `text` just below `anchor` (the text cursor), or near the mouse
    /// when the cursor position is unknown. Hides itself after `seconds` if given.
    func show(_ text: String, near anchor: NSRect?, hideAfter seconds: Double? = nil) {
        hideWork?.cancel()
        label.stringValue = text
        label.sizeToFit()
        let width = min(max(label.frame.width + 28, 90), 520)
        let size = NSSize(width: width, height: 30)
        label.frame = NSRect(x: 14, y: (size.height - label.frame.height) / 2,
                             width: width - 28, height: label.frame.height)

        var origin: NSPoint
        if let anchor = anchor {
            origin = NSPoint(x: anchor.minX, y: anchor.minY - size.height - 6)
        } else {
            let mouse = NSEvent.mouseLocation
            origin = NSPoint(x: mouse.x + 12, y: mouse.y - size.height - 12)
        }
        let screen = NSScreen.screens.first { $0.frame.contains(origin) } ?? NSScreen.main
        if let visible = screen?.visibleFrame {
            origin.x = min(max(origin.x, visible.minX + 4), visible.maxX - size.width - 4)
            origin.y = min(max(origin.y, visible.minY + 4), visible.maxY - size.height - 4)
        }
        panel.setFrame(NSRect(origin: origin, size: size), display: true)
        panel.orderFrontRegardless()

        if let seconds = seconds {
            let work = DispatchWorkItem { [weak self] in self?.panel.orderOut(nil) }
            hideWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
        }
    }

    func hide() {
        hideWork?.cancel()
        panel.orderOut(nil)
    }
}
