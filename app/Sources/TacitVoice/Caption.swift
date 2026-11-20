import AppKit
import QuartzCore

/// The one place the app speaks in words: a dark card at the top centre of
/// the screen, just under the menu bar. It shows what a task was heard as,
/// what came back, questions from the agent, and every notice ("No words
/// detected", "Cancelled", errors). It never takes focus or clicks, so the
/// user stays in the app they were using.
final class Caption {
    private let panel: NSPanel
    private let background = NSView()
    private let label = NSTextField(wrappingLabelWithString: "")
    private let dots = CALayer()
    private var dotLayers: [CALayer] = []
    private var hideWork: DispatchWorkItem?

    /// Every card is the same width, whatever it says.
    private let width: CGFloat = 460
    private let sidePadding: CGFloat = 22
    private let verticalPadding: CGFloat = 18
    private let dotsSize = NSSize(width: 34, height: 20)
    private let dotsGap: CGFloat = 14
    private let minimumHeight: CGFloat = 64
    /// Gap between the menu bar and the card.
    private let topMargin: CGFloat = 8
    /// Longer text is cut here; the full reply is in the history page.
    private let maxCharacters = 600

    init() {
        panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 460, height: 64),
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

        // Near-black with a slight lift toward one corner and a thin cool edge.
        let backing = CAGradientLayer()
        backing.colors = [
            NSColor(white: 0.10, alpha: 0.94).cgColor,
            NSColor(white: 0.16, alpha: 0.94).cgColor,
        ]
        backing.startPoint = CGPoint(x: 0, y: 1)
        backing.endPoint = CGPoint(x: 1, y: 0)
        backing.cornerRadius = 22
        backing.borderWidth = 1
        backing.borderColor = NSColor(red: 0.36, green: 0.40, blue: 0.78, alpha: 0.55).cgColor
        backing.masksToBounds = true
        background.layer = backing
        background.wantsLayer = true

        // Three discs: one bright, two dimmer, as in the loading orb.
        let discs: [(x: CGFloat, y: CGFloat, size: CGFloat, alpha: CGFloat)] = [
            (0, 0, 11, 1.0), (11, 10, 9, 0.70), (25, 10, 9, 0.45),
        ]
        for disc in discs {
            let layer = CALayer()
            layer.frame = CGRect(x: disc.x, y: disc.y, width: disc.size, height: disc.size)
            layer.cornerRadius = disc.size / 2
            layer.backgroundColor = NSColor.white.cgColor
            layer.opacity = Float(disc.alpha)
            dots.addSublayer(layer)
            dotLayers.append(layer)
        }
        backing.addSublayer(dots)

        label.font = .systemFont(ofSize: 15, weight: .semibold)
        label.textColor = .white
        label.alignment = .center
        label.maximumNumberOfLines = 10
        label.lineBreakMode = .byWordWrapping
        label.isSelectable = false

        background.addSubview(label)
        panel.contentView = background
    }

    /// Show a card. With `busy` the discs pulse in turn, for work in progress.
    /// It stays until replaced or hidden unless `seconds` is given.
    func show(_ text: String, busy: Bool = false, hideAfter seconds: Double? = nil) {
        hideWork?.cancel()
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        var shown = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if shown.count > maxCharacters { shown = String(shown.prefix(maxCharacters - 1)) + "…" }
        label.stringValue = shown

        // The text is centred in the space to the right of the discs, with the
        // same space kept free on the other side so it sits in the card's middle.
        let reserved = sidePadding + dotsSize.width + dotsGap
        let textWidth = width - reserved * 2
        let fit = label.cell?.cellSize(
            forBounds: NSRect(x: 0, y: 0, width: textWidth, height: 2000))
            ?? NSSize(width: textWidth, height: 20)
        let textHeight = ceil(fit.height)
        let height = max(minimumHeight, textHeight + verticalPadding * 2)

        label.frame = NSRect(
            x: reserved, y: (height - textHeight) / 2, width: textWidth, height: textHeight)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        dots.frame = CGRect(
            x: sidePadding, y: (height - dotsSize.height) / 2,
            width: dotsSize.width, height: dotsSize.height)
        CATransaction.commit()
        pulse(busy)

        // visibleFrame ends below the menu bar (and so below a notch).
        let origin = NSPoint(
            x: screen.frame.midX - width / 2,
            y: screen.visibleFrame.maxY - topMargin - height)
        panel.setFrame(
            NSRect(origin: origin, size: NSSize(width: width, height: height)), display: true)
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

    /// Each disc brightens in turn while work is in progress; otherwise they rest.
    private func pulse(_ on: Bool) {
        for (index, layer) in dotLayers.enumerated() {
            layer.removeAnimation(forKey: "pulse")
            guard on else { continue }
            let wave = CAKeyframeAnimation(keyPath: "opacity")
            wave.values = [0.35, 1.0, 0.35, 0.35]
            wave.keyTimes = [0, 0.2, 0.5, 1]
            wave.duration = 1.2
            wave.repeatCount = .infinity
            wave.beginTime = CACurrentMediaTime() + Double(index) * 0.25
            layer.add(wave, forKey: "pulse")
        }
    }

    /// Time to read `text`: a short "Done" goes quickly, an email stays longer.
    static func readingTime(_ text: String) -> Double {
        min(30, max(4, Double(text.count) / 14))
    }
}
