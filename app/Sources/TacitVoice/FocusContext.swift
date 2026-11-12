import AppKit
import ApplicationServices

/// What the user is typing into, read through the Accessibility interface.
enum FocusContext {
    static func appName() -> String {
        NSWorkspace.shared.frontmostApplication?.localizedName ?? ""
    }

    private static func focusedElement() -> AXUIElement? {
        let system = AXUIElementCreateSystemWide()
        var ref: CFTypeRef?
        let attr = kAXFocusedUIElementAttribute as CFString
        guard AXUIElementCopyAttributeValue(system, attr, &ref) == .success, let value = ref,
            CFGetTypeID(value) == AXUIElementGetTypeID()
        else { return nil }
        return (value as! AXUIElement)
    }

    private static func string(_ element: AXUIElement, _ attribute: String) -> String? {
        var ref: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &ref) == .success else {
            return nil
        }
        return ref as? String
    }

    private static func selectedRange(_ element: AXUIElement) -> CFRange? {
        var ref: CFTypeRef?
        let attr = kAXSelectedTextRangeAttribute as CFString
        guard AXUIElementCopyAttributeValue(element, attr, &ref) == .success, let value = ref,
            CFGetTypeID(value) == AXValueGetTypeID()
        else { return nil }
        var range = CFRange()
        return AXValueGetValue(value as! AXValue, .cfRange, &range) ? range : nil
    }

    /// The text just before the cursor. Empty for password fields and for apps
    /// that do not expose their text.
    static func textBeforeCursor(limit: Int = 600) -> String {
        guard let element = focusedElement() else { return "" }
        if string(element, kAXSubroleAttribute as String) == (kAXSecureTextFieldSubrole as String) {
            return ""
        }
        guard let value = string(element, kAXValueAttribute as String) else { return "" }
        let text = value as NSString
        var end = text.length
        if let range = selectedRange(element) { end = min(max(range.location, 0), text.length) }
        let start = max(0, end - limit)
        return text.substring(with: NSRange(location: start, length: end - start))
    }

    /// Where the text cursor is on screen, in AppKit coordinates.
    static func caretRect() -> NSRect? {
        guard let element = focusedElement(), var range = selectedRange(element),
            let rangeValue = AXValueCreate(.cfRange, &range)
        else { return nil }
        var ref: CFTypeRef?
        let attr = kAXBoundsForRangeParameterizedAttribute as CFString
        guard
            AXUIElementCopyParameterizedAttributeValue(element, attr, rangeValue, &ref) == .success,
            let value = ref, CFGetTypeID(value) == AXValueGetTypeID()
        else { return nil }
        var rect = CGRect.zero
        guard AXValueGetValue(value as! AXValue, .cgRect, &rect), rect.height > 0 else { return nil }
        // Accessibility measures from the top-left of the main screen; AppKit
        // measures from its bottom-left.
        guard let mainHeight = NSScreen.screens.first?.frame.height else { return nil }
        return NSRect(x: rect.minX, y: mainHeight - rect.maxY, width: rect.width, height: rect.height)
    }
}
