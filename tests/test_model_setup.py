import threading

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
