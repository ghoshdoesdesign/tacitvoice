import AppKit
import ImageIO
import QuartzCore

/// One state's animation: transparent frames made by tools/make_orb.py or
/// tools/make_loading.py from videos the owner supplies.
struct OrbClip {
    var frames: [CGImage] = []
    var fps: Double = 30
    /// Frames before this index play once; the rest repeat.
    var loopStart = 0
    var isEmpty: Bool { frames.isEmpty }
}

/// The app's whole interface, at the bottom centre of the screen. With the
/// owner's animations installed every working state is a floating orb with no
/// text: it morphs while the key is held, shows circling dots while the text
/// is worked out, and closes when the text has been inserted. Without them it
/// falls back to a small black label. Messages always use the label.
final class Pill {
    enum State {
        case observing        // key held
        case polishing        // key released, text being worked out
        case inserted         // text pasted
        case done             // a task for the agent finished
        case note(String)     // anything else worth saying, as text
    }

    private let panel: NSPanel
    private let background = NSView()
    private let icon = NSView()
    private let label = NSTextField(labelWithString: "")
    private var hideWork: DispatchWorkItem?

    private let height: CGFloat = 36
    /// Side of the floating orb, in points.
    private let orbSize: CGFloat = 120
    /// Gap between the bottom of the usable screen and the orb.
    private let bottomMargin: CGFloat = 56

    /// Empty until loaded, or if the owner has not installed that animation.
    private var listening = OrbClip()   // ~/.tacitvoice/orb
    private var working = OrbClip()     // ~/.tacitvoice/orb-loading
    private var finished = OrbClip()    // ~/.tacitvoice/orb-done
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
        loadOrbFrames()
    }

    /// Read the animation folders off the main thread, so launch stays quick.
    private func loadOrbFrames() {
        let home = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".tacitvoice")
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let listening = Pill.readClip(home.appendingPathComponent("orb"))
            let working = Pill.readClip(home.appendingPathComponent("orb-loading"))
            let finished = Pill.readClip(home.appendingPathComponent("orb-done"))
            DispatchQueue.main.async {
                self?.listening = listening
                self?.working = working
                self?.finished = finished
            }
        }
    }

    private static func readClip(_ folder: URL) -> OrbClip {
        var clip = OrbClip()
        let files = (try? FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil)) ?? []
        let pngs = files.filter { $0.pathExtension == "png" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for url in pngs {
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else { continue }
            clip.frames.append(image)
        }
        func number(_ name: String) -> Double? {
            let url = folder.appendingPathComponent(name)
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
            return Double(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        if let rate = number("fps.txt"), rate > 0 { clip.fps = rate }
        if let start = number("loop.txt"), start >= 1, Int(start) < clip.frames.count {
            clip.loopStart = Int(start)
        }
        return clip
    }

    private func frameAnimation(_ frames: [CGImage], fps: Double) -> CAKeyframeAnimation {
        let play = CAKeyframeAnimation(keyPath: "contents")
        play.values = frames
        play.calculationMode = .discrete
        play.duration = Double(frames.count) / fps
        play.isRemovedOnCompletion = false
        play.fillMode = .forwards
        return play
    }

    /// Show only an animated orb, with no label or background. With
    /// `repeats` the clip's opening plays once and the rest loops; without
    /// it the whole clip plays once. Returns how long one play-through takes.
    @discardableResult
    private func showOrb(_ clip: OrbClip, repeats: Bool, on screen: NSScreen) -> Double {
        let image = CALayer()
        image.frame = CGRect(x: 0, y: 0, width: orbSize, height: orbSize)
        image.contentsGravity = .resizeAspect
        image.contents = repeats ? clip.frames[0] : clip.frames[clip.frames.count - 1]

        let opening = repeats ? Array(clip.frames[..<clip.loopStart]) : clip.frames
        let openingTime = Double(opening.count) / clip.fps
        if !opening.isEmpty {
            image.add(frameAnimation(opening, fps: clip.fps), forKey: "opening")
        }
        if repeats {
            // Added second, so it takes over from the opening once it begins.
            let loop = frameAnimation(Array(clip.frames[clip.loopStart...]), fps: clip.fps)
            loop.repeatCount = .infinity
            if openingTime > 0 { loop.beginTime = CACurrentMediaTime() + openingTime }
            image.add(loop, forKey: "loop")
        }

        icon.layer?.sublayers = nil
        icon.layer?.addSublayer(image)
        icon.isHidden = false
        label.isHidden = true
        background.layer?.backgroundColor = NSColor.clear.cgColor
        panel.hasShadow = false

        icon.frame = NSRect(x: 0, y: 0, width: orbSize, height: orbSize)
        let origin = NSPoint(
            x: screen.frame.midX - orbSize / 2,
            y: screen.visibleFrame.minY + bottomMargin)
        panel.setFrame(
            NSRect(origin: origin, size: NSSize(width: orbSize, height: orbSize)), display: true)
        panel.orderFrontRegardless()
        return openingTime
    }

    private func hide(after seconds: Double) {
        let work = DispatchWorkItem { [weak self] in self?.panel.orderOut(nil) }
        hideWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
    }

    /// Show a state at the bottom centre of the active screen. Every state is
    /// centred on the same point. Hides itself after `seconds` if given.
    func show(_ state: State, hideAfter seconds: Double? = nil) {
        hideWork?.cancel()
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }

        switch state {
        case .observing where !listening.isEmpty:
            showOrb(listening, repeats: true, on: screen)
            return
        case .polishing where !working.isEmpty:
            showOrb(working, repeats: true, on: screen)
            return
        case .inserted where !finished.isEmpty, .done where !finished.isEmpty:
            // The closing animation is the confirmation: play it once, then go.
            hide(after: showOrb(finished, repeats: false, on: screen))
            return
        default:
            break
        }
        label.isHidden = false
        background.layer?.backgroundColor = NSColor.black.cgColor
        panel.hasShadow = true

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
        case .done:
            text = "Done"
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

        // Centred where the orb's centre is, so the states replace each other
        // in place. visibleFrame starts above the Dock.
        let origin = NSPoint(
            x: screen.frame.midX - width / 2,
            y: screen.visibleFrame.minY + bottomMargin + (orbSize - height) / 2)
        panel.setFrame(
            NSRect(origin: origin, size: NSSize(width: width, height: height)), display: true)
        panel.orderFrontRegardless()

        if let seconds = seconds { hide(after: seconds) }
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
