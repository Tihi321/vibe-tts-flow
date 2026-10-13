import struct

from vibetts import selection
from vibetts.selection import VK_C, VK_INSERT, Snapshot, copy_key_for


def test_copy_key_for_terminals():
    assert copy_key_for("CASCADIA_HOSTING_WINDOW_CLASS") == VK_INSERT
    assert copy_key_for("ConsoleWindowClass") == VK_INSERT
    assert copy_key_for("mintty") == VK_INSERT
    assert copy_key_for("Chrome_WidgetWin_1") == VK_C
    assert copy_key_for("") == VK_C


def test_hdrop_bytes():
    data = selection._hdrop_bytes(("C:\\a.txt", "C:\\b.txt"))
    assert struct.unpack("<IiiII", data[:20]) == (20, 0, 0, 0, 1)
    assert data[20:].decode("utf-16-le") == "C:\\a.txt\0C:\\b.txt\0\0"


class Fake:
    def __init__(self, monkeypatch, clipboard_after, changes=True, prev="old"):
        self.seq = 1
        self.sent = []
        self.restored = []
        self.text = prev
        self.changes = changes
        self.after = clipboard_after
        monkeypatch.setattr(selection, "sequence_number", lambda: self.seq)
        monkeypatch.setattr(selection, "foreground_class", lambda: "Notepad")
        monkeypatch.setattr(selection, "send_copy", self.copy)
        monkeypatch.setattr(selection, "clipboard_text", lambda: self.text)
        monkeypatch.setattr(selection, "snapshot_clipboard",
                            lambda: Snapshot([(13, prev)] if prev else []))
        monkeypatch.setattr(selection, "restore_clipboard", self.restored.append)

    def copy(self, vk):
        self.sent.append(vk)
        if self.changes:
            self.seq += 1
            self.text = self.after


def test_grab_selected_text_restores(monkeypatch):
    f = Fake(monkeypatch, "selected words")
    g = selection.grab_selection(200)
    assert g.text == "selected words" and not g.from_clipboard
    assert f.sent == [VK_C]
    assert f.restored and f.restored[0].formats == [(13, "old")]


def test_nothing_selected_falls_back_to_clipboard(monkeypatch):
    f = Fake(monkeypatch, None, changes=False, prev="clipboard text")
    g = selection.grab_selection(50, fallback_to_clipboard=True)
    assert g.text == "clipboard text" and g.from_clipboard
    assert not f.restored


def test_nothing_selected_no_fallback(monkeypatch):
    Fake(monkeypatch, None, changes=False)
    assert selection.grab_selection(50, fallback_to_clipboard=False).text is None


def test_whitespace_selection_is_nothing(monkeypatch):
    Fake(monkeypatch, "   \n ")
    assert selection.grab_selection(200).text is None


def test_no_restore(monkeypatch):
    f = Fake(monkeypatch, "x y")
    selection.grab_selection(200, restore=False)
    assert not f.restored
