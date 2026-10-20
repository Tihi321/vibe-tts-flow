import sys
from pathlib import Path

import pytest

from vibetts import fileassoc
from vibetts.fileassoc import PROG_ID, open_command, register, registered_exts, unregister


class FakeRegistry:
    """In-memory HKCU: {key: {name: value}}."""

    def __init__(self):
        self.keys: dict[str, dict[str, str]] = {}

    def set_value(self, key, name, value):
        self.keys.setdefault(key, {})[name] = value

    def get_value(self, key, name):
        return self.keys.get(key, {}).get(name)

    def delete_value(self, key, name):
        self.keys.get(key, {}).pop(name, None)

    def delete_tree(self, key):
        for k in [k for k in self.keys if k == key or k.startswith(key + "\\")]:
            del self.keys[k]


CLASSES = fileassoc.CLASSES


@pytest.fixture
def source_run(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Apps\VibeTTS\VibeTTS.exe")


def test_open_command_from_source(source_run, monkeypatch):
    monkeypatch.setattr(sys, "executable", r"C:\py\python.exe")
    cmd = open_command()
    run = Path(fileassoc.__file__).resolve().parent.parent / "run.pyw"
    assert cmd == f'"{Path(sys.executable).with_name("pythonw.exe")}" "{run}" --open "%1"'
    assert "pythonw.exe" in cmd


def test_open_command_frozen(frozen):
    assert open_command() == r'"C:\Apps\VibeTTS\VibeTTS.exe" --open "%1"'


def test_register_frozen_writes_everything(frozen):
    reg = FakeRegistry()
    register((".epub", ".txt"), reg)
    prog = f"{CLASSES}\\{PROG_ID}"
    assert reg.get_value(prog, "") == "VibeTTS book"
    assert reg.get_value(prog + "\\DefaultIcon", "") == r"C:\Apps\VibeTTS\VibeTTS.exe,0"
    assert reg.get_value(prog + "\\shell\\open\\command", "") == open_command()
    for ext in (".epub", ".txt"):
        assert reg.get_value(f"{CLASSES}\\{ext}\\OpenWithProgids", PROG_ID) == ""
    app = f"{CLASSES}\\Applications\\VibeTTS.exe"
    assert reg.get_value(app, "FriendlyAppName") == "VibeTTS"
    assert reg.get_value(app + "\\SupportedTypes", ".epub") == ""
    assert registered_exts(reg) == [".epub", ".txt"]


def test_register_never_writes_extension_default(frozen):
    reg = FakeRegistry()
    reg.set_value(f"{CLASSES}\\.txt", "", "txtfile")
    register((".txt",), reg)
    assert reg.get_value(f"{CLASSES}\\.txt", "") == "txtfile"
    assert reg.get_value(f"{CLASSES}\\.epub", "") is None
    assert reg.get_value(f"{CLASSES}\\.txt\\OpenWithProgids", PROG_ID) == ""


def test_register_only_selected_extensions(source_run):
    reg = FakeRegistry()
    register((".epub",), reg)
    assert registered_exts(reg) == [".epub"]
    assert reg.get_value(f"{CLASSES}\\.txt\\OpenWithProgids", PROG_ID) is None
    register((".txt",), reg)  # adds, does not remove .epub
    assert registered_exts(reg) == [".epub", ".txt"]


def test_register_from_source_skips_icon_and_applications(source_run):
    reg = FakeRegistry()
    register((".epub",), reg)
    assert f"{CLASSES}\\{PROG_ID}\\DefaultIcon" not in reg.keys
    assert not any("Applications" in key for key in reg.keys)


def test_register_rejects_other_extensions(source_run):
    reg = FakeRegistry()
    with pytest.raises(ValueError):
        register((".pdf",), reg)
    assert reg.keys == {}


def test_unregister_removes_only_ours(frozen):
    reg = FakeRegistry()
    reg.set_value(f"{CLASSES}\\.txt", "", "txtfile")
    reg.set_value(f"{CLASSES}\\.epub\\OpenWithProgids", "Other.App", "")
    register((".epub", ".txt"), reg)
    unregister(reg)
    assert registered_exts(reg) == []
    assert reg.get_value(f"{CLASSES}\\.txt", "") == "txtfile"
    assert reg.get_value(f"{CLASSES}\\.epub\\OpenWithProgids", "Other.App") == ""
    assert not any(PROG_ID in key or "Applications" in key for key in reg.keys)
    assert all(PROG_ID not in vals for vals in reg.keys.values())
    unregister(reg)  # idempotent


def test_registered_exts_needs_the_prog_id(frozen):
    reg = FakeRegistry()
    reg.set_value(f"{CLASSES}\\.epub\\OpenWithProgids", PROG_ID, "")  # stale, no ProgID
    assert registered_exts(reg) == []


def test_open_default_apps(monkeypatch):
    opened = []
    monkeypatch.setattr(fileassoc.os, "startfile", opened.append, raising=False)
    fileassoc.open_default_apps()
    assert opened == ["ms-settings:defaultapps"]
