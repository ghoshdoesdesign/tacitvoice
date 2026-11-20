import AppKit

/// A translucent black panel just above the orb, for words: what a task was
/// heard as, and what came back. It never takes focus or clicks, so the user
/// stays in the app they were using.
final class Caption {
    private let panel: NSPanel
    private let background = NSView()
    private let label = NSTextField(wrappingLabelWithString: "")
    private var hideWork: DispatchWorkItem?

    private let maxTextWidth: CGFloat = 480
    private let padding: CGFloat = 14
    /// Distance from the bottom of the usable screen to the panel's lower
    /// edge: the orb's margin (56) and size (120) in Pill, plus a gap.
    private let lift: CGFloat = 56 + 120 + 10
    /// Longer text is cut here; the full reply is in the history page.
    private let maxCharacters = 600

    init() {
        panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 240, height: 40),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: true)
        panel.level = .statusBar
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        let backing = CALayer()
        backing.backgroundColor = NSColor.black.withAlphaComponent(0.72).cgColor
        backing.cornerRadius = 12
        background.layer = backing
        background.wantsLayer = true

        label.font = .systemFont(ofSize: 13, weight: .medium)
        label.textColor = .white
        label.maximumNumberOfLines = 12
        label.lineBreakMode = .byWordWrapping
        label.isSelectable = false

        background.addSubview(label)
        panel.contentView = background
    }

    /// Show text above the orb. Stays until replaced or hidden unless `seconds` is given.
    func show(_ text: String, hideAfter seconds: Double? = nil) {
        hideWork?.cancel()
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        var shown = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if shown.count > maxCharacters { shown = String(shown.prefix(maxCharacters - 1)) + "…" }
        label.stringValue = shown

        let fit = label.cell?.cellSize(
            forBounds: NSRect(x: 0, y: 0, width: maxTextWidth, height: 2000))
            ?? NSSize(width: maxTextWidth, height: 20)
        let textSize = NSSize(width: min(ceil(fit.width), maxTextWidth), height: ceil(fit.height))
        let size = NSSize(
            width: textSize.width + padding * 2, height: textSize.height + padding * 2)
        label.frame = NSRect(origin: NSPoint(x: padding, y: padding), size: textSize)
        let origin = NSPoint(
            x: screen.frame.midX - size.width / 2,
            y: screen.visibleFrame.minY + lift)
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

    /// Time to read `text`: a short "Done" goes quickly, an email stays longer.
    static func readingTime(_ text: String) -> Double {
        min(30, max(4, Double(text.count) / 14))
    }
}
