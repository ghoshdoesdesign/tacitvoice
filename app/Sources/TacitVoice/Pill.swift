import AppKit
import QuartzCore

/// The small black label at the top centre of the screen: the app's whole
/// interface. It has three working states and a plain one for messages.
final class Pill {
    enum State {
        case observing        // key held: animated orb, "Observing lip movement…"
        case polishing        // key released: spinner, "Polishing your thoughts…"
        case inserted         // text pasted: green tick, "Inserted"
        case note(String)     // anything else worth saying, no icon
    }

    private let panel: NSPanel
    private let background = NSView()
    private let icon = NSView()
    private let label = NSTextField(labelWithString: "")
    private var hideWork: DispatchWorkItem?

    private let height: CGFloat = 36
    private let iconSize: CGFloat = 22
    private let padding: CGFloat = 12
    private let gap: CGFloat = 10

    init() {
        panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 220, height: 36),
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

        let backing = CALayer()
        backing.backgroundColor = NSColor.black.cgColor
        backing.cornerRadius = 10
        background.layer = backing
        background.wantsLayer = true
        background.autoresizingMask = [.width, .height]

        icon.layer = CALayer()
        icon.wantsLayer = true

        label.font = .systemFont(ofSize: 13, weight: .semibold)
        label.textColor = .white
        label.lineBreakMode = .byTruncatingTail

        background.addSubview(icon)
        background.addSubview(label)
        panel.contentView = background
    }

    /// Show a state at the top centre of the active screen, just under the
    /// menu bar. Hides itself after `seconds` if given.
    func show(_ state: State, hideAfter seconds: Double? = nil) {
        hideWork?.cancel()

        let text: String
        var layers: [CALayer] = []
        switch state {
        case .observing:
            text = "Observing lip movement…"
            layers = orbLayers()
        case .polishing:
            text = "Polishing your thoughts…"
            layers = spinnerLayers()
        case .inserted:
            text = "Inserted"
            layers = tickLayers()
        case .note(let message):
            text = message
        }

        icon.layer?.sublayers = nil
        for layer in layers { icon.layer?.addSublayer(layer) }
        let showsIcon = !layers.isEmpty
        icon.isHidden = !showsIcon

        label.stringValue = text
        label.sizeToFit()
        let textWidth = min(label.frame.width, 460)
        let textHeight = label.frame.height
        let iconSpace = showsIcon ? iconSize + gap : 0
        let width = padding + iconSpace + textWidth + padding + 4

        icon.frame = NSRect(
            x: padding, y: (height - iconSize) / 2, width: iconSize, height: iconSize)
        label.frame = NSRect(
            x: padding + iconSpace, y: (height - textHeight) / 2, width: textWidth,
            height: textHeight)

        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        // visibleFrame ends below the menu bar, which on notched Macs is as
        // tall as the notch, so the pill sits just beneath it.
        let origin = NSPoint(
            x: screen.frame.midX - width / 2,
            y: screen.visibleFrame.maxY - height - 8)
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

    // MARK: Icons

    private var iconBounds: CGRect { CGRect(x: 0, y: 0, width: iconSize, height: iconSize) }

    private func color(_ red: CGFloat, _ green: CGFloat, _ blue: CGFloat) -> CGColor {
        NSColor(srgbRed: red / 255, green: green / 255, blue: blue / 255, alpha: 1).cgColor
    }

    /// One ring filled with a colour wheel, turning forever.
    private func ring(
        inset: CGFloat, lineWidth: CGFloat, seconds: Double, clockwise: Bool, opacity: Float
    ) -> CAGradientLayer {
        let gradient = CAGradientLayer()
        gradient.type = .conic
        gradient.frame = iconBounds
        gradient.startPoint = CGPoint(x: 0.5, y: 0.5)
        gradient.endPoint = CGPoint(x: 0.5, y: 0)
        gradient.colors = [
            color(255, 138, 61),   // orange
            color(255, 79, 163),   // pink
            color(139, 92, 255),   // violet
            color(61, 139, 255),   // blue
            color(61, 224, 255),   // cyan
            color(255, 138, 61),   // back to orange, so the wheel has no seam
        ]
        gradient.opacity = opacity

        let mask = CAShapeLayer()
        mask.frame = iconBounds
        mask.path = CGPath(ellipseIn: iconBounds.insetBy(dx: inset, dy: inset), transform: nil)
        mask.fillColor = NSColor.clear.cgColor
        mask.strokeColor = NSColor.black.cgColor
        mask.lineWidth = lineWidth
        gradient.mask = mask

        let spin = CABasicAnimation(keyPath: "transform.rotation.z")
        spin.fromValue = 0.0
        spin.toValue = (clockwise ? -2.0 : 2.0) * Double.pi
        spin.duration = seconds
        spin.repeatCount = .infinity
        spin.isRemovedOnCompletion = false
        gradient.add(spin, forKey: "spin")
        return gradient
    }

    /// The listening orb: two colour rings turning against each other, the
    /// outer one breathing gently.
    private func orbLayers() -> [CALayer] {
        let outer = ring(inset: 2.5, lineWidth: 3.2, seconds: 2.2, clockwise: true, opacity: 1)
        let breathe = CABasicAnimation(keyPath: "transform.scale")
        breathe.fromValue = 0.9
        breathe.toValue = 1.0
        breathe.duration = 0.9
        breathe.autoreverses = true
        breathe.repeatCount = .infinity
        breathe.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        breathe.isRemovedOnCompletion = false
        outer.add(breathe, forKey: "breathe")

        let inner = ring(inset: 6.8, lineWidth: 1.6, seconds: 3.4, clockwise: false, opacity: 0.75)
        return [outer, inner]
    }

    /// A spoke spinner: ten short bars fading one after another.
    private func spinnerLayers() -> [CALayer] {
        let count = 10
        let replicator = CAReplicatorLayer()
        replicator.frame = iconBounds
        replicator.instanceCount = count
        replicator.instanceTransform = CATransform3DMakeRotation(
            -2 * CGFloat.pi / CGFloat(count), 0, 0, 1)
        replicator.instanceDelay = 1.0 / Double(count)

        let bar = CALayer()
        bar.frame = CGRect(x: iconSize / 2 - 1, y: iconSize - 7.5, width: 2, height: 6)
        bar.cornerRadius = 1
        bar.backgroundColor = NSColor.white.cgColor
        bar.opacity = 0.15

        let fade = CABasicAnimation(keyPath: "opacity")
        fade.fromValue = 1.0
        fade.toValue = 0.15
        fade.duration = 1.0
        fade.repeatCount = .infinity
        fade.isRemovedOnCompletion = false
        bar.add(fade, forKey: "fade")

        replicator.addSublayer(bar)
        return [replicator]
    }

    /// A green tick that draws itself.
    private func tickLayers() -> [CALayer] {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 4.5, y: 11.5))
        path.addLine(to: CGPoint(x: 9.3, y: 6.5))
        path.addLine(to: CGPoint(x: 18, y: 16))

        let tick = CAShapeLayer()
        tick.frame = iconBounds
        tick.path = path
        tick.fillColor = NSColor.clear.cgColor
        tick.strokeColor = color(52, 211, 83)
        tick.lineWidth = 3.2
        tick.lineCap = .round
        tick.lineJoin = .round

        let draw = CABasicAnimation(keyPath: "strokeEnd")
        draw.fromValue = 0.0
        draw.toValue = 1.0
        draw.duration = 0.22
        draw.timingFunction = CAMediaTimingFunction(name: .easeOut)
        tick.add(draw, forKey: "draw")
        return [tick]
    }
}
