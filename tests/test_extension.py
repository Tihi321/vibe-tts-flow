"""Static checks for the Chrome extension in extension/ (no browser needed)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "extension"
CONTRACT_KEY = (
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtMJxUvQneCKRQGrumd1fCDi4dkktxzGc10HRiDg0evpBCJm+NuQ050F/ugqOeJ0J17do"
    "Dj9jB5v65BQR+faLiOpQK3uOyPrPDjn0uBINIKSlhfpj53Uxo9d2wCniAKqSNR2jFi9zGQXZu5SyAg3fFApRIMQLpZ/4nYwTll28bxFyriI6FwpKy4f4e17w"
    "s5TaheLifldYHABuBiboSQUoPa9c+msdT2IPKw4HdeZ1DpdepboUscxFbu1oPXxNxsqW653NljjWEpGiTPfsHpxWrqFs54lfILFy1/8IpLcc4NnHJai0Kg8I"
    "NmJjBAIMJcWnT2hvHshGrrew6KtM/enSUwIDAQAB"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_basics(manifest):
    assert manifest["manifest_version"] == 3
    assert manifest["name"] == "VibeTTS Reader"
    assert manifest["key"] == CONTRACT_KEY
    assert {"contextMenus", "activeTab", "scripting", "storage"} <= set(manifest["permissions"])
    assert "content_scripts" not in manifest
    assert "<all_urls>" not in manifest["host_permissions"]


def test_host_permissions_cover_loopback(manifest):
    hosts = manifest["host_permissions"]
    assert any(h.startswith("http://127.0.0.1/") for h in hosts)
    assert any(h.startswith("http://localhost/") for h in hosts)


def test_commands(manifest):
    cmds = manifest["commands"]
    assert cmds["read"]["suggested_key"]["default"] == "Alt+Shift+R"
    assert cmds["toggle-pause"]["suggested_key"]["default"] == "Alt+Shift+P"
    assert cmds["stop"]["suggested_key"]["default"] == "Alt+Shift+S"


def test_referenced_files_exist(manifest):
    refs = [manifest["background"]["service_worker"], manifest["action"]["default_popup"]]
    refs += manifest["icons"].values()
    refs += manifest["action"]["default_icon"].values()
    assert set(manifest["icons"]) == {"16", "32", "48", "128"}
    for ref in refs:
        assert (EXT / ref).is_file(), ref
    # files the background injects / the popup loads
    for name in ("segment.js", "content.js", "popup.js", "popup.css"):
        assert (EXT / name).is_file(), name


def test_segment_js_is_identical():
    assert (EXT / "segment.js").read_bytes() == (ROOT / "vibetts" / "web" / "segment.js").read_bytes()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.parametrize("script", sorted(p.name for p in EXT.glob("*.js")))
def test_js_syntax(script):
    node = shutil.which("node")
    args = [node, "--check", str(EXT / script)]
    if script == "background.js":  # ES module in the manifest
        args = [node, "--check", "--input-type=module", "-"]
        result = subprocess.run(args, input=(EXT / script).read_bytes(), capture_output=True)
    else:
        result = subprocess.run(args, capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


# ---- jump to a selection made while reading

def test_background_seek_to_checks_owner_tab():
    src = (EXT / "background.js").read_text(encoding="utf-8")
    assert 'case "seek-to"' in src
    assert "sender.tab.id !== ours.tabId" in src  # only the reading tab may move the reading
    assert "/api/seek" in src and "st.count - 1" in src  # absolute item, clamped


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_content_jump_on_selection():
    node = shutil.which("node")
    result = subprocess.run([node, str(ROOT / "tests" / "extension_content_harness.js")], capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    r = json.loads(result.stdout)
    assert r["count"] == 6
    assert r["beforeProgress"] == {"queued": None, "sent": 0}  # not reading yet: selection ignored
    assert r["afterSelect"]["queued"] == 5  # sentence where the selection starts is marked
    assert r["insideParagraph"] == {"queued": 5, "sent": 0}  # collapsed click keeps it, no jump yet
    assert r["atBoundary"] == {"queued": None, "sent": [{"cmd": "seek-to", "item": 5}]}
    assert r["afterJump"]["sent"] == 0  # the progress caused by the jump does not re-trigger
    assert r["replaced"]["queued"] == 4  # newer selection replaces the older one
    assert r["inCurrent"]["queued"] is None  # selecting inside the current sentence cancels
    assert r["escaped"] == {"queued": None, "sent": 0}  # Escape cancels
    assert r["cleared"]["queued"] is None and r["afterClear"]["queued"] is None
    assert r["skipped"] == {"queued": None, "sent": [{"cmd": "seek-to", "item": 4}]}  # coalesced prev+2 still jumps
    assert r["outside"]["queued"] is None  # selection beyond the collected text is ignored
