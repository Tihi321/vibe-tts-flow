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
