import threading
import types

import pytest
import requests

from vibetts import model_setup


class Resp:
    def __init__(self, data: bytes, status=200):
        self.data = data
        self.status = status
        self.headers = {"Content-Length": str(len(data))}

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def iter_content(self, chunk_size):
        for i in range(0, len(self.data), 4):
            yield self.data[i:i + 4]

    def close(self):
        pass


def test_download_ok(tmp_path, monkeypatch):
    urls = []
    monkeypatch.setattr(model_setup.requests, "get",
                        lambda url, **kw: urls.append(url) or Resp(b"0123456789"))
    progress = []
    path = model_setup.download("f.bin", tmp_path, lambda d, t: progress.append((d, t)),
                                expected_size=10)
    assert path.read_bytes() == b"0123456789"
    assert urls == [model_setup.BASE_URL + "f.bin"]
    assert progress[-1] == (10, 10)
    assert not (tmp_path / "f.bin.part").exists()
    model_setup.download("f.bin", tmp_path, expected_size=10)  # installed: no new request
    assert len(urls) == 1


def test_size_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"short"))
    with pytest.raises(model_setup.SetupError, match="expected 10"):
        model_setup.download("f.bin", tmp_path, expected_size=10)
    assert not list(tmp_path.iterdir())


def test_cancel_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"x" * 100))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(model_setup.DownloadCancelled):
        model_setup.download("f.bin", tmp_path, cancel=cancel)
    assert not list(tmp_path.iterdir())


def test_http_error(tmp_path, monkeypatch):
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"", status=404))
    with pytest.raises(model_setup.SetupError, match="404"):
        model_setup.download("f.bin", tmp_path)


def test_download_named_unknown(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        model_setup.download_named("nope", tmp_path)


def test_files_table():
    assert set(model_setup.files()) == {"int8", "fp32", "voices"}
    assert model_setup.DEFAULT_MODEL in model_setup.MODELS


def test_download_full_url(tmp_path, monkeypatch):
    urls = []
    monkeypatch.setattr(model_setup.requests, "get",
                        lambda url, **kw: urls.append(url) or Resp(b"0123456789"))
    url = "https://huggingface.co/some/repo/resolve/abc123/sub/dir/f.bin"
    path = model_setup.download("f.bin", tmp_path, expected_size=10, url=url)
    assert urls == [url]  # not the Kokoro release URL
    assert path.read_bytes() == b"0123456789"


def test_download_full_url_cancel_and_size_check(tmp_path, monkeypatch):
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"short"))
    with pytest.raises(model_setup.SetupError, match="expected 10"):
        model_setup.download("f.bin", tmp_path, expected_size=10, url="https://x/f.bin")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(model_setup.DownloadCancelled):
        model_setup.download("f.bin", tmp_path, cancel=cancel, url="https://x/f.bin")
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------- Supertonic

FAKE_FILES = (("onnx/a.onnx", 10), ("onnx/b.json", 4), ("voice_styles/F1.json", 6), ("LICENSE", 3))


@pytest.fixture
def fake_files(monkeypatch):
    monkeypatch.setattr(model_setup, "MODEL_FILES", FAKE_FILES)
    return FAKE_FILES


def fake_get(urls):
    def get(url, **kw):
        urls.append(url)
        size = dict(FAKE_FILES)[url.split("/resolve/", 1)[1].split("/", 1)[1]]
        return Resp(b"x" * size)
    return get


def test_supertonic_files_and_url():
    assert model_setup.supertonic_files() == model_setup.MODEL_FILES
    assert len(model_setup.supertonic_files()) == 17
    assert model_setup.SUPERTONIC_BASE_URL == (
        "https://huggingface.co/Supertone/supertonic-3/resolve/"
        "724fb5abbf5502583fb520898d45929e62f02c0b/")


def test_supertonic_missing(tmp_path, fake_files):
    assert model_setup.supertonic_missing(tmp_path) == [p for p, _ in FAKE_FILES]
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "a.onnx").write_bytes(b"x" * 10)
    (tmp_path / "onnx" / "b.json").write_bytes(b"xx")  # wrong size
    assert model_setup.supertonic_missing(tmp_path) == ["onnx/b.json", "voice_styles/F1.json",
                                                         "LICENSE"]


def test_download_supertonic_only_missing(tmp_path, monkeypatch, fake_files):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "a.onnx").write_bytes(b"x" * 10)
    urls = []
    monkeypatch.setattr(model_setup.requests, "get", fake_get(urls))
    progress = []
    model_setup.download_supertonic(tmp_path, lambda d, t: progress.append((d, t)))
    base = model_setup.SUPERTONIC_BASE_URL
    assert urls == [base + "onnx/b.json", base + "voice_styles/F1.json", base + "LICENSE"]
    assert model_setup.supertonic_missing(tmp_path) == []
    assert (tmp_path / "voice_styles" / "F1.json").read_bytes() == b"x" * 6
    assert not list(tmp_path.rglob("*.part"))
    assert progress[0] == (0, 13)
    assert progress[-1] == (13, 13)
    done = [d for d, _ in progress]
    assert done == sorted(done)
    assert {t for _, t in progress} == {13}
    urls.clear()
    model_setup.download_supertonic(tmp_path)  # nothing left to fetch
    assert urls == []


def test_download_supertonic_cancel_leaves_no_partial(tmp_path, monkeypatch, fake_files):
    monkeypatch.setattr(model_setup.requests, "get", fake_get([]))
    cancel = threading.Event()

    def progress(done, total):
        if done >= 18:  # after the first chunk of the third file
            cancel.set()

    with pytest.raises(model_setup.DownloadCancelled):
        model_setup.download_supertonic(tmp_path, progress, cancel)
    assert not list(tmp_path.rglob("*.part"))
    assert (tmp_path / "onnx" / "a.onnx").exists()  # finished files stay for the next resume
    assert (tmp_path / "onnx" / "b.json").exists()
    assert not (tmp_path / "voice_styles" / "F1.json").exists()
    assert not (tmp_path / "LICENSE").exists()


def test_download_supertonic_cancelled_up_front(tmp_path, monkeypatch, fake_files):
    monkeypatch.setattr(model_setup.requests, "get", fake_get([]))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(model_setup.DownloadCancelled):
        model_setup.download_supertonic(tmp_path, cancel=cancel)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_download_supertonic_http_error(tmp_path, monkeypatch, fake_files):
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"", status=404))
    with pytest.raises(model_setup.SetupError, match="404"):
        model_setup.download_supertonic(tmp_path)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_download_supertonic_size_mismatch(tmp_path, monkeypatch, fake_files):
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"short"))
    with pytest.raises(model_setup.SetupError, match="expected 10"):
        model_setup.download_supertonic(tmp_path)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_delete_supertonic(tmp_path, monkeypatch, fake_files):
    monkeypatch.setattr(model_setup.requests, "get", fake_get([]))
    model_setup.download_supertonic(tmp_path)
    (tmp_path / "keep.txt").write_text("mine")
    model_setup.delete_supertonic(tmp_path)
    assert model_setup.supertonic_missing(tmp_path) == [p for p, _ in FAKE_FILES]
    assert not (tmp_path / "onnx").exists() and not (tmp_path / "voice_styles").exists()
    assert (tmp_path / "keep.txt").exists()
    model_setup.delete_supertonic(tmp_path / "nope")  # nothing to delete: no error


def test_cli_supertonic(tmp_path, monkeypatch, fake_files, capsys):
    monkeypatch.setattr(model_setup.requests, "get", fake_get([]))
    assert model_setup.main(["--supertonic", "--dest", str(tmp_path)]) == 0
    assert model_setup.supertonic_missing(tmp_path) == []
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"", status=500))
    model_setup.delete_supertonic(tmp_path)
    assert model_setup.main(["--supertonic", "--dest", str(tmp_path)]) == 1


# ---------------------------------------------------------------- Higgs

HIGGS_FAKE = (("cpu_int4/a.onnx", 10), ("cpu_int4/a.onnx.data", 20), ("cpu_int4/tok.json", 4),
              ("LICENSE", 3))


@pytest.fixture
def higgs_fake(monkeypatch):
    monkeypatch.setattr(model_setup, "HIGGS_MODEL_FILES", HIGGS_FAKE)
    return HIGGS_FAKE


def higgs_get(urls):
    def get(url, **kw):
        urls.append(url)
        return Resp(b"x" * dict(HIGGS_FAKE)[url.split("/resolve/", 1)[1].split("/", 1)[1]])
    return get


def test_higgs_files_and_url():
    from vibetts import higgs
    assert model_setup.higgs_files() == higgs.MODEL_FILES
    assert model_setup.HIGGS_BASE_URL == (
        "https://huggingface.co/onnx-community/higgs-audio-v3-tts-4b/resolve/"
        "ecc25d6cc6e48909401bfa7ce5ac95e84f41f331/")


def test_higgs_missing(tmp_path, higgs_fake):
    assert model_setup.higgs_missing(tmp_path) == [p for p, _ in HIGGS_FAKE]
    (tmp_path / "cpu_int4").mkdir()
    (tmp_path / "cpu_int4" / "a.onnx").write_bytes(b"x" * 10)
    (tmp_path / "cpu_int4" / "a.onnx.data").write_bytes(b"x" * 5)   # wrong size
    assert model_setup.higgs_missing(tmp_path) == ["cpu_int4/a.onnx.data", "cpu_int4/tok.json",
                                                    "LICENSE"]


def test_download_higgs_only_missing_and_progress(tmp_path, monkeypatch, higgs_fake):
    (tmp_path / "cpu_int4").mkdir()
    (tmp_path / "cpu_int4" / "a.onnx").write_bytes(b"x" * 10)
    urls = []
    monkeypatch.setattr(model_setup.requests, "get", higgs_get(urls))
    progress = []
    model_setup.download_higgs(tmp_path, lambda d, t: progress.append((d, t)))
    base = model_setup.HIGGS_BASE_URL
    assert urls == [base + "cpu_int4/a.onnx.data", base + "cpu_int4/tok.json", base + "LICENSE"]
    assert model_setup.higgs_missing(tmp_path) == []
    assert not list(tmp_path.rglob("*.part"))
    assert progress[0] == (0, 27) and progress[-1] == (27, 27)
    assert [d for d, _ in progress] == sorted(d for d, _ in progress)
    urls.clear()
    model_setup.download_higgs(tmp_path)
    assert urls == []


def test_download_higgs_cancel_leaves_no_partial(tmp_path, monkeypatch, higgs_fake):
    monkeypatch.setattr(model_setup.requests, "get", higgs_get([]))
    cancel = threading.Event()

    def progress(done, total):
        if done >= 12:  # inside the second (20 byte) file
            cancel.set()

    with pytest.raises(model_setup.DownloadCancelled):
        model_setup.download_higgs(tmp_path, progress, cancel)
    assert not list(tmp_path.rglob("*.part"))
    assert (tmp_path / "cpu_int4" / "a.onnx").exists()
    assert not (tmp_path / "cpu_int4" / "a.onnx.data").exists()


def test_download_higgs_http_error_and_size_mismatch(tmp_path, monkeypatch, higgs_fake):
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"", status=404))
    with pytest.raises(model_setup.SetupError, match="404"):
        model_setup.download_higgs(tmp_path)
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"short"))
    with pytest.raises(model_setup.SetupError, match="expected 10"):
        model_setup.download_higgs(tmp_path)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_download_higgs_checks_free_space_first(tmp_path, monkeypatch, higgs_fake):
    urls = []
    monkeypatch.setattr(model_setup.requests, "get", higgs_get(urls))
    seen = []

    def usage(path):
        seen.append(path)
        return types.SimpleNamespace(total=10**12, used=10**12 - 1000, free=1000)

    monkeypatch.setattr(model_setup.shutil, "disk_usage", usage)
    dest = tmp_path / "not" / "yet" / "there"
    with pytest.raises(OSError, match="Not enough free disk space"):
        model_setup.download_higgs(dest)
    assert urls == [] and not dest.exists()
    assert seen and seen[0].exists()                       # probed the nearest existing folder
    monkeypatch.setattr(model_setup.shutil, "disk_usage",
                        lambda p: types.SimpleNamespace(free=model_setup.DISK_MARGIN + 37))
    model_setup.download_higgs(dest)
    assert model_setup.higgs_missing(dest) == []


def test_download_higgs_nothing_missing_skips_space_check(tmp_path, monkeypatch, higgs_fake):
    monkeypatch.setattr(model_setup.requests, "get", higgs_get([]))
    model_setup.download_higgs(tmp_path)
    monkeypatch.setattr(model_setup.shutil, "disk_usage",
                        lambda p: (_ for _ in ()).throw(AssertionError("no check needed")))
    model_setup.download_higgs(tmp_path)


def test_delete_higgs(tmp_path, monkeypatch, higgs_fake):
    monkeypatch.setattr(model_setup.requests, "get", higgs_get([]))
    model_setup.download_higgs(tmp_path)
    (tmp_path / "cpu_int4" / "a.onnx.part").write_bytes(b"x")
    (tmp_path / "keep.txt").write_text("mine")
    model_setup.delete_higgs(tmp_path)
    assert model_setup.higgs_missing(tmp_path) == [p for p, _ in HIGGS_FAKE]
    assert not (tmp_path / "cpu_int4").exists() and (tmp_path / "keep.txt").exists()
    model_setup.delete_higgs(tmp_path / "nope")


def test_cli_higgs(tmp_path, monkeypatch, higgs_fake, capsys):
    monkeypatch.setattr(model_setup.requests, "get", higgs_get([]))
    assert model_setup.main(["--higgs", "--dest", str(tmp_path)]) == 0
    assert model_setup.higgs_missing(tmp_path) == []
    model_setup.delete_higgs(tmp_path)
    monkeypatch.setattr(model_setup.requests, "get", lambda url, **kw: Resp(b"", status=500))
    assert model_setup.main(["--higgs", "--dest", str(tmp_path)]) == 1
    monkeypatch.setattr(model_setup.shutil, "disk_usage",
                        lambda p: types.SimpleNamespace(free=0))
    assert model_setup.main(["--higgs", "--dest", str(tmp_path)]) == 1
    assert "Not enough free disk space" in capsys.readouterr().err
