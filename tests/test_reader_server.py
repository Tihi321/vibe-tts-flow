"""Tests for the local web reader server (live HTTP against a fake API)."""
from __future__ import annotations

import http.client
import json
import socket
import time
from urllib.parse import quote

import pytest

from vibetts import docload
from vibetts.library import LibraryError
from vibetts.reader_server import ReaderServer
from vibetts.reading import ReadingError

BOOK = "0123456789abcdef"
EXT = "abcdefghijklmnopabcdefghijklmnop"


class FakeSession:
    def __init__(self):
        self.calls = []
        self.raise_exc = None
        self.version = 1

    def _do(self, name, *args):
        self.calls.append((name,) + args)
        if self.raise_exc:
            raise self.raise_exc
        return {"version": self.version, "state": name}

    def start(self, items, start=0, voice=None, lang=None, speed=None, source="reader", meta=None,
              max_chars=500_000):
        self.calls.append(("start", items, start, voice, lang, speed, source, meta, max_chars))
        if self.raise_exc:
            raise self.raise_exc
        return {"session": "abc", "count": len(items)}

    def seek(self, item):
        return self._do("seek", item)

    def pause(self):
        return self._do("pause")

    def resume(self):
        return self._do("resume")

    def stop(self):
        return self._do("stop")

    def wait(self, since, timeout):
        self.calls.append(("wait", since, timeout))
        if since is None or since != self.version:
            return {"version": self.version, "state": "playing"}
        time.sleep(min(timeout, 0.05))  # honours the timeout (shortened for the tests)
        return {"version": self.version, "state": "playing"}


class FakeLibrary:
    def __init__(self):
        self.books = {BOOK: {"id": BOOK, "title": "T", "chapters": 2}}
        self.calls = []
        self.raise_exc = None

    def _rec(self, *call):
        self.calls.append(call)
        if self.raise_exc:
            raise self.raise_exc

    def add(self, data, filename):
        self._rec("add", data, filename)
        return {"id": BOOK, "title": filename}

    def list(self):
        self._rec("list")
        return list(self.books.values())

    def get(self, book_id):
        self._rec("get", book_id)
        return {**self.books[book_id], "toc": []}

    def chapter(self, book_id, n):
        self._rec("chapter", book_id, n)
        if book_id not in self.books:
            raise KeyError(book_id)
        if n >= 2:
            raise IndexError(n)
        return {"index": n, "title": "c", "paragraphs": ["p"]}

    def set_position(self, book_id, chapter, item):
        self._rec("set_position", book_id, chapter, item)

    def delete(self, book_id):
        self._rec("delete", book_id)
        del self.books[book_id]


class FakeApi:
    version = "9.9"

    def __init__(self):
        self.session = FakeSession()
        self.library = FakeLibrary()
        self.opened = []
        self.open_exc = None

    def list_voices(self):
        return {"voices": [{"id": "af_heart"}]}

    def open_book(self, path):
        self.opened.append(path)
        if self.open_exc:
            raise self.open_exc
        return {"id": BOOK, "url": f"http://127.0.0.1/#/book/{BOOK}"}


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def web(tmp_path):
    root = tmp_path / "web"
    root.mkdir()
    (root / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    (root / "app.js").write_text("1", encoding="utf-8")
    (root / "style.css").write_text("a{}", encoding="utf-8")
    (root / "icon.svg").write_text("<svg/>", encoding="utf-8")
    (root / "icon.png").write_bytes(b"\x89PNG")
    (root / "favicon.ico").write_bytes(b"ico")
    (root / "data.json").write_text("{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")  # outside web_root
    return root


@pytest.fixture
def server(api, web):
    s = ReaderServer(api, port=0, extension_ids=[EXT], max_chars=1234, web_root=web)
    s.start()
    yield s
    s.stop()


def req(server, method, path, body=None, headers=None, origin="own", host=None, ctype="json"):
    """Send a request; origin 'own' = the reader's own origin, None = no Origin header."""
    h = {"Host": host or f"127.0.0.1:{server.port}"}
    if origin == "own":
        origin = f"http://127.0.0.1:{server.port}"
    if origin is not None:
        h["Origin"] = origin
    data = None
    if body is not None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode()
            if ctype == "json":
                h["Content-Type"] = "application/json"
        else:
            data = body
        if ctype == "bin":
            h["Content-Type"] = "application/octet-stream"
        elif ctype not in ("json", None):
            h["Content-Type"] = ctype
    h.update(headers or {})
    c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        raw = r.read()
        return r.status, dict(r.getheaders()), raw
    finally:
        c.close()


def jreq(server, method, path, body=None, **kw):
    status, headers, raw = req(server, method, path, body, **kw)
    return status, json.loads(raw) if raw else None


# ---- lifecycle ------------------------------------------------------------
def test_url_and_stop_frees_port(api, web):
    s = ReaderServer(api, port=0, web_root=web)
    s.start()
    port = s.port
    assert port > 0 and s.url == f"http://127.0.0.1:{port}/" and s.running
    s.stop()
    assert not s.running
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))  # would raise if the port were still held
    finally:
        probe.close()
    s.stop()  # idempotent


# ---- routes ---------------------------------------------------------------
def test_info_and_voices(server):
    assert jreq(server, "GET", "/api/info") == (200, {"app": "vibetts", "version": "9.9", "api": 1})
    status, body = jreq(server, "GET", "/api/voices")
    assert status == 200 and body["voices"][0]["id"] == "af_heart"


def test_json_headers(server):
    status, headers, _ = req(server, "GET", "/api/info")
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Cache-Control"] == "no-cache"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert not any(k.lower().startswith("access-control") for k in headers)


def test_speak(server, api):
    body = {"items": [{"text": "Hello"}, {"text": "World", "pause": "paragraph"}], "start": 1, "voice": "af_heart",
            "lang": "en", "speed": 1.25, "source": "extension", "meta": {"title": "x"}}
    assert jreq(server, "POST", "/api/speak", body) == (200, {"session": "abc", "count": 2})
    assert api.session.calls[-1] == ("start", body["items"], 1, "af_heart", "en", 1.25, "extension",
                                     {"title": "x"}, 1234)


def test_speak_defaults_and_bad_shapes(server, api):
    assert jreq(server, "POST", "/api/speak", {"items": [{"text": "a"}]})[0] == 200
    call = api.session.calls[-1]
    assert call[2:8] == (0, None, None, None, "reader", None)
    assert jreq(server, "POST", "/api/speak", {"items": "nope"})[0] == 400
    assert jreq(server, "POST", "/api/speak", {"items": [], "start": "x"})[0] == 400
    assert jreq(server, "POST", "/api/speak", {"items": [], "meta": []})[0] == 400
    assert jreq(server, "POST", "/api/speak", [1])[0] == 400
    status, _, _ = req(server, "POST", "/api/speak", b"{not json", headers={"Content-Type": "application/json"})
    assert status == 400


@pytest.mark.parametrize("name", ["pause", "resume", "stop"])
def test_pause_resume_stop(server, api, name):
    assert jreq(server, "POST", f"/api/{name}", {}) == (200, {"version": 1, "state": name})
    assert jreq(server, "POST", f"/api/{name}", b"", ctype="json")[0] == 200  # no body, Content-Length 0
    assert api.session.calls[-1] == (name,)


def test_seek(server, api):
    assert jreq(server, "POST", "/api/seek", {"item": 3}) == (200, {"version": 1, "state": "seek"})
    assert api.session.calls[-1] == ("seek", 3)
    assert jreq(server, "POST", "/api/seek", {"item": "3"})[0] == 400
    assert jreq(server, "POST", "/api/seek", {})[0] == 400


def test_status_long_poll(server, api):
    api.session.version = 7
    t = time.time()
    assert jreq(server, "GET", "/api/status?since=3&timeout=20")[0] == 200
    assert time.time() - t < 2  # since differs -> immediate
    assert api.session.calls[-1] == ("wait", 3, 20.0)
    jreq(server, "GET", "/api/status")
    assert api.session.calls[-1] == ("wait", None, 0.0)
    jreq(server, "GET", "/api/status?since=7&timeout=999")
    assert api.session.calls[-1] == ("wait", 7, 25.0)
    jreq(server, "GET", "/api/status?since=7&timeout=-5")
    assert api.session.calls[-1] == ("wait", 7, 0.0)
    assert jreq(server, "GET", "/api/status?since=abc")[0] == 400
    assert jreq(server, "GET", "/api/status?timeout=abc")[0] == 400
    assert jreq(server, "GET", "/api/status?timeout=nan")[0] == 400


def test_status_get_needs_no_origin(server):
    assert jreq(server, "GET", "/api/status", origin=None)[0] == 200


def test_books_list_and_get(server, api):
    status, body = jreq(server, "GET", "/api/books")
    assert status == 200 and body["books"][0]["id"] == BOOK
    status, body = jreq(server, "GET", f"/api/books/{BOOK}")
    assert status == 200 and body["toc"] == [] and body["title"] == "T"
    assert jreq(server, "GET", f"/api/books/{BOOK}/chapters/1") == (
        200, {"index": 1, "title": "c", "paragraphs": ["p"]})
    assert api.library.calls[-1] == ("chapter", BOOK, 1)


def test_book_position_and_delete(server, api):
    assert jreq(server, "PUT", f"/api/books/{BOOK}/position", {"chapter": 1, "item": 4}) == (200, {"ok": True})
    assert api.library.calls[-1] == ("set_position", BOOK, 1, 4)
    assert jreq(server, "PUT", f"/api/books/{BOOK}/position", {"chapter": "1", "item": 4})[0] == 400
    assert jreq(server, "DELETE", f"/api/books/{BOOK}") == (200, {"ok": True})
    assert api.library.calls[-1] == ("delete", BOOK)
    assert jreq(server, "DELETE", f"/api/books/{BOOK}")[0] == 404  # KeyError from the library


def test_upload_filename_decoded(server, api):
    name = "Mój život (1).epub"
    status, body = jreq(server, "POST", "/api/books", b"PKdata", ctype="bin", headers={"X-Filename": quote(name)})
    assert status == 200 and body["id"] == BOOK
    assert api.library.calls[-1] == ("add", b"PKdata", name)


def test_upload_filename_basename_only_and_required(server, api):
    jreq(server, "POST", "/api/books", b"x", ctype="bin", headers={"X-Filename": quote("..\\..\\evil/a.txt")})
    assert api.library.calls[-1] == ("add", b"x", "a.txt")
    n = len(api.library.calls)
    assert jreq(server, "POST", "/api/books", b"x", ctype="bin")[0] == 400
    assert jreq(server, "POST", "/api/books", b"x", ctype="bin", headers={"X-Filename": "%2F"})[0] == 400
    assert len(api.library.calls) == n


def test_upload_content_type_and_size(server, api, monkeypatch):
    assert jreq(server, "POST", "/api/books", b"x", ctype="text/plain", headers={"X-Filename": "a.txt"})[0] == 415
    assert jreq(server, "POST", "/api/books", {"a": 1}, headers={"X-Filename": "a.txt"})[0] == 415
    monkeypatch.setattr(docload, "MAX_FILE_BYTES", 10)
    assert jreq(server, "POST", "/api/books", b"x" * 11, ctype="bin", headers={"X-Filename": "a.txt"})[0] == 413
    assert jreq(server, "POST", "/api/books", b"x" * 10, ctype="bin", headers={"X-Filename": "a.txt"})[0] == 200


def test_open(server, api):
    status, body = jreq(server, "POST", "/api/open", {"path": "C:\\b.epub"})
    assert status == 200 and body["url"].endswith(f"#/book/{BOOK}")
    assert api.opened == ["C:\\b.epub"]
    assert jreq(server, "POST", "/api/open", {"path": ""})[0] == 400
    assert jreq(server, "POST", "/api/open", {})[0] == 400


def test_open_accepts_no_origin_and_extension(server, api):
    assert jreq(server, "POST", "/api/open", {"path": "a.epub"}, origin=None)[0] == 200
    assert jreq(server, "POST", "/api/open", {"path": "a.epub"}, origin=f"chrome-extension://{EXT}")[0] == 200
    assert api.opened == ["a.epub", "a.epub"]


def test_open_website_origin_rejected(server, api):
    assert jreq(server, "POST", "/api/open", {"path": "a.epub"}, origin="https://evil.example")[0] == 403
    assert jreq(server, "POST", "/api/open", {"path": "a.epub"}, origin="null")[0] == 403
    assert api.opened == []


def test_open_errors(server, api):
    api.open_exc = LibraryError("Unsupported file type")
    assert jreq(server, "POST", "/api/open", {"path": "a.zip"}) == (400, {"error": "Unsupported file type"})
    api.open_exc = RuntimeError("no browser")
    assert jreq(server, "POST", "/api/open", {"path": "a.epub"})[0] == 503


# ---- Host / Origin ----------------------------------------------------------
def test_host_check(server):
    assert jreq(server, "GET", "/api/info", host="evil.example")[0] == 403
    assert jreq(server, "GET", "/api/info", host="127.0.0.1:1")[0] == 403
    assert jreq(server, "GET", "/api/info", host="127.0.0.1")[0] == 403
    assert jreq(server, "GET", "/api/info", host=f"localhost:{server.port}")[0] == 200
    assert req(server, "GET", "/", host="evil.example")[0] == 403  # static too


def test_origin_check(server):
    p = server.port
    for ok in (f"http://127.0.0.1:{p}", f"http://localhost:{p}", f"chrome-extension://{EXT}"):
        assert jreq(server, "GET", "/api/info", origin=ok)[0] == 200
        assert jreq(server, "POST", "/api/stop", {}, origin=ok)[0] == 200
    for bad in ("https://example.com", f"http://127.0.0.1:{p + 1}", "chrome-extension://zzzzzzzz",
                f"chrome-extension://{EXT}x", "null", f"https://127.0.0.1:{p}"):
        assert jreq(server, "GET", "/api/info", origin=bad)[0] == 403
        assert req(server, "GET", "/", origin=bad)[0] == 403
        assert jreq(server, "POST", "/api/stop", {}, origin=bad)[0] == 403


def test_no_extension_ids_rejects_extensions(api, web):
    s = ReaderServer(api, port=0, web_root=web)
    s.start()
    try:
        assert jreq(s, "GET", "/api/info", origin=f"chrome-extension://{EXT}")[0] == 403
    finally:
        s.stop()


def test_state_changing_requires_origin(server, api):
    assert jreq(server, "POST", "/api/speak", {"items": []}, origin=None)[0] == 403
    assert jreq(server, "POST", "/api/stop", {}, origin=None)[0] == 403
    assert jreq(server, "POST", "/api/books", b"x", ctype="bin", origin=None, headers={"X-Filename": "a"})[0] == 403
    assert jreq(server, "PUT", f"/api/books/{BOOK}/position", {"chapter": 0, "item": 0}, origin=None)[0] == 403
    assert jreq(server, "DELETE", f"/api/books/{BOOK}", origin=None)[0] == 403
    assert api.session.calls == [] and api.library.calls == []


# ---- body validation -------------------------------------------------------
def test_json_content_type_411_413(server, api):
    assert jreq(server, "POST", "/api/speak", b'{"items": []}', ctype="text/plain")[0] == 415
    assert jreq(server, "POST", "/api/seek", b'{"item": 1}', ctype=None)[0] == 415
    assert jreq(server, "POST", "/api/seek", b'{"item": 1}', ctype="application/json; charset=utf-8")[0] == 200
    # oversized declared length: rejected from the header alone, the body is never sent or read
    assert jreq(server, "POST", "/api/speak", b"", ctype="application/json",
                headers={"Content-Length": str(4 * 1024 * 1024 + 1)})[0] == 413
    assert api.session.calls == [("seek", 1)]


def test_missing_content_length_411(server):
    c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        c.putrequest("POST", "/api/seek", skip_host=True)
        c.putheader("Host", f"127.0.0.1:{server.port}")
        c.putheader("Origin", f"http://127.0.0.1:{server.port}")
        c.putheader("Content-Type", "application/json")
        c.endheaders()
        r = c.getresponse()
        assert r.status == 411
        assert "error" in json.loads(r.read())
    finally:
        c.close()


# ---- error mapping ----------------------------------------------------------
def test_error_mapping_session(server, api):
    body = {"items": [{"text": "a"}]}
    api.session.raise_exc = ReadingError("No items")
    assert jreq(server, "POST", "/api/speak", body) == (400, {"error": "No items"})
    api.session.raise_exc = ValueError("bad")
    assert jreq(server, "POST", "/api/seek", {"item": 0})[0] == 400
    api.session.raise_exc = TypeError("shape")
    assert jreq(server, "POST", "/api/speak", body)[0] == 400
    api.session.raise_exc = RuntimeError("Voice model missing")
    assert jreq(server, "POST", "/api/speak", body) == (503, {"error": "Voice model missing"})
    assert jreq(server, "POST", "/api/seek", {"item": 0})[0] == 503
    api.session.raise_exc = KeyError("x")
    assert jreq(server, "POST", "/api/seek", {"item": 0})[0] == 404
    api.session.raise_exc = IndexError("x")
    assert jreq(server, "POST", "/api/seek", {"item": 0})[0] == 404


def test_error_mapping_library(server, api):
    api.library.raise_exc = LibraryError("Unreadable")
    assert jreq(server, "POST", "/api/books", b"x", ctype="bin", headers={"X-Filename": "a.epub"}) == (
        400, {"error": "Unreadable"})
    api.library.raise_exc = KeyError("gone")
    assert jreq(server, "GET", f"/api/books/{BOOK}")[0] == 404
    api.library.raise_exc = ValueError("bad position")
    assert jreq(server, "PUT", f"/api/books/{BOOK}/position", {"chapter": 9, "item": 0}) == (
        400, {"error": "bad position"})
    # RuntimeError is only a 503 for speak/seek/open; elsewhere it is unexpected
    api.library.raise_exc = RuntimeError("boom")
    assert jreq(server, "GET", "/api/books")[0] == 500


def test_unexpected_exception_is_generic_500(server, api):
    api.session.raise_exc = OSError("secret C:\\path detail")
    status, raw = req(server, "POST", "/api/pause", {})[::2]
    assert status == 500
    assert json.loads(raw) == {"error": "Internal error"}
    assert b"secret" not in raw
    # the server survives
    assert jreq(server, "GET", "/api/info")[0] == 200


def test_invalid_book_id_404_without_library_call(server, api):
    for bad in ("short", "0123456789ABCDEF", "0123456789abcdeg", f"{BOOK}0", "..%2f..%2fx"):
        assert jreq(server, "GET", f"/api/books/{bad}")[0] == 404
        assert jreq(server, "DELETE", f"/api/books/{bad}")[0] == 404
        assert jreq(server, "PUT", f"/api/books/{bad}/position", {"chapter": 0, "item": 0})[0] == 404
        assert jreq(server, "GET", f"/api/books/{bad}/chapters/0")[0] == 404
    assert jreq(server, "GET", f"/api/books/{BOOK}/chapters/x")[0] == 404
    assert jreq(server, "GET", f"/api/books/{BOOK}/chapters/-1")[0] == 404
    assert api.library.calls == []


def test_unknown_api_and_wrong_method(server):
    assert jreq(server, "GET", "/api/nope")[0] == 404
    assert jreq(server, "GET", "/api/speak")[0] == 405
    assert jreq(server, "POST", "/api/info", {})[0] == 405
    assert jreq(server, "POST", "/index.html", {})[0] == 404


def test_options_405(server):
    status, headers, _ = req(server, "OPTIONS", "/api/speak", origin=f"chrome-extension://{EXT}")
    assert status == 405
    assert not any(k.lower().startswith("access-control") for k in headers)
    assert req(server, "OPTIONS", "/")[0] == 405


# ---- static files ------------------------------------------------------------
def test_static_index_and_types(server):
    status, headers, raw = req(server, "GET", "/", origin=None)
    assert status == 200 and raw == b"<h1>hi</h1>"
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Cache-Control"] == "no-cache" and headers["X-Frame-Options"] == "DENY"
    assert req(server, "GET", "/index.html")[2] == b"<h1>hi</h1>"
    expected = {
        "/app.js": "text/javascript; charset=utf-8", "/style.css": "text/css; charset=utf-8",
        "/icon.svg": "image/svg+xml", "/icon.png": "image/png", "/favicon.ico": "image/x-icon",
        "/data.json": "application/json",
    }
    for path, ctype in expected.items():
        status, headers, _ = req(server, "GET", path)
        assert status == 200 and headers["Content-Type"] == ctype, path
    assert req(server, "GET", "/index.html?x=1")[0] == 200  # query ignored
    assert req(server, "GET", "/missing.js")[0] == 404
    assert req(server, "GET", "/sub")[0] == 404


@pytest.mark.parametrize("path", [
    "/../secret.txt", "/%2e%2e/secret.txt", "/..%5csecret.txt", "/C:/Windows/win.ini", "/%2e%2e%2fsecret.txt",
    "/..\\secret.txt", "//etc/passwd", "/a/../../secret.txt", "/%00", "/index.html%00.png",
])
def test_static_traversal_rejected(server, path):
    status, _, raw = req(server, "GET", path)
    assert status == 404
    assert b"secret" not in raw


def test_static_symlink_escape_rejected(server, web, tmp_path):
    link = web / "link.txt"
    try:
        link.symlink_to(tmp_path / "secret.txt")
    except OSError:
        pytest.skip("symlinks not permitted")
    assert req(server, "GET", "/link.txt")[0] == 404
