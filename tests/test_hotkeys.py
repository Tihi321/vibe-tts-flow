import pytest

from vibetts import hotkeys
from vibetts.hotkeys import ChordTracker, HotkeyListener, parse_chord


def test_parse_default_chord():
    c = parse_chord("alt+win")
    assert [set(g) for g in c.groups] == [{"alt_l", "alt_r"}, {"cmd_l", "cmd_r"}]
    assert c.has_win
    assert c.describe() == "Alt + Win"


@pytest.mark.parametrize("bad", ["", "ctrl+", "ctrl+bogus"])
def test_parse_errors(bad):
    with pytest.raises(ValueError):
        parse_chord(bad)


def test_tracker():
    t = ChordTracker(parse_chord("alt+win"))
    assert t.press("alt_l") is False
    assert t.press("cmd_l") is True
    assert t.press("cmd_l") is False  # repeat
    assert t.release("alt_l") is True
    assert t.press("alt_r") is True


class Key:
    def __init__(self, vk):
        self.vk = vk


def test_listener_multiple_bindings(monkeypatch):
    monkeypatch.setattr(hotkeys, "_send_noop_key", lambda: None)
    fired, escapes = [], []
    lst = HotkeyListener({"read": parse_chord("alt+win"), "other": parse_chord("f9")},
                         fired.append, lambda: escapes.append(1))
    lst._on_press(Key(0xA4))   # alt_l
    lst._on_press(Key(0x5B))   # cmd_l
    lst._on_press(Key(0x5B))   # repeat
    lst._on_release(Key(0x5B))
    lst._on_release(Key(0xA4))
    lst._on_press(Key(0x78))   # f9
    lst._on_press(Key(0x1B))   # esc
    lst._on_press(Key(0xA4), injected=True)
    assert fired == ["read", "other"]
    assert escapes == [1]
    assert lst.chord("read").spec == "alt+win"
    assert lst.chord("missing") is None


def test_listener_set_bindings_resets(monkeypatch):
    monkeypatch.setattr(hotkeys, "_send_noop_key", lambda: None)
    fired = []
    lst = HotkeyListener({"read": parse_chord("f8")}, fired.append, lambda: None)
    lst.set_bindings({"read": parse_chord("f9")})
    lst._on_press(Key(0x77))  # f8
    lst._on_press(Key(0x78))  # f9
    assert fired == ["read"]


def test_wait_for_release_includes_modifiers(monkeypatch):
    down = {"shift_l"}
    monkeypatch.setattr(hotkeys, "key_is_down", lambda k: k in down)
    assert hotkeys.wait_for_chord_release(parse_chord("f9"), timeout=0.05) is False
    down.clear()
    assert hotkeys.wait_for_chord_release(parse_chord("f9"), timeout=0.05) is True
