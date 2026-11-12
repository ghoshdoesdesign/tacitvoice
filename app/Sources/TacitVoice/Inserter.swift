import AppKit

/// Puts text at the cursor of the frontmost app by pasting it, then restores
/// whatever was on the clipboard. Pasting works in every app; typing key by
/// key is slow and writing through Accessibility only works in some.
enum Inserter {
    static func paste(_ text: String) {
        let board = NSPasteboard.general
        let saved: [NSPasteboardItem] = (board.pasteboardItems ?? []).map { item in
            let copy = NSPasteboardItem()
            for type in item.types {
                if let data = item.data(forType: type) { copy.setData(data, forType: type) }
            }
            return copy
        }

        board.clearContents()
        board.setString(text, forType: .string)

        let source = CGEventSource(stateID: .combinedSessionState)
        let vKey: CGKeyCode = 9
        let down = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: true)
        let up = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: false)
        // Set the flags outright so a modifier still physically held cannot
        // turn this into a different shortcut.
        down?.flags = .maskCommand
        up?.flags = .maskCommand
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)

        // Give the target app time to read the clipboard before restoring it.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            board.clearContents()
            if !saved.isEmpty { board.writeObjects(saved) }
        }
    }
}
