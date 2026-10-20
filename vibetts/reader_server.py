"""Local web reader server: static reader page plus a JSON API for reading sessions and the book library.

Stdlib only, binds 127.0.0.1. Never prints (the windowed exe has no stdout); never logs request bodies or
text, only lengths and counts. Security: Host and Origin checks on every request, state-changing methods
need an allowed Origin (POST /api/open also accepts none), no CORS headers.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import parse_qs, unquote, urlsplit

from . import docload
from .library import BOOK_ID, LibraryError
from .reading import ReadingError

log = logging.getLogger(__name__)

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_WAIT = 25.0  # longest status long-poll, seconds

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json",
}

SECURITY_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

_BOOK_ID = re.compile(BOOK_ID + r"\Z")


class ReaderApi(Protocol):
    version: str
    session: Any   # ReadingSession-like: start, seek, pause, resume, stop, status, wait
    library: Any   # Library-like: add, list, get, chapter, set_position, delete

    def list_voices(self) -> dict: ...

    def open_book(self, path: str) -> dict: ...


class _Server(ThreadingHTTPServer):
    daemon_threads = True  # the status long-poll holds a thread for up to 25 s
    block_on_close = False
    allow_reuse_address = False  # on Windows, reuse would permit double-binding a port


class _HttpError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ReaderServer:
    def __init__(self, api: ReaderApi, port: int, extension_ids: Sequence[str] = (),
                 max_chars: int = 500_000, web_root: Path | None = None) -> None:
        self._api = api
        self.port = port
        self._extension_ids = tuple(extension_ids)
        self._max_chars = max_chars
        self._web_root = Path(web_root) if web_root is not None else Path(__file__).parent / "web"
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._routes: list[tuple[str, re.Pattern, Callable[..., Any]]] = self._build_routes()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._httpd is not None:
                return
            httpd = _Server(("127.0.0.1", self.port), self._make_handler())
            self.port = httpd.server_address[1]
            self._httpd = httpd
            self._thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.05), name="vibetts-reader", daemon=True)
            self._thread.start()
            log.info("Reader server listening on %s", self.url)

    def stop(self) -> None:
        with self._lock:
            httpd, thread = self._httpd, self._thread
            self._httpd = None
            self._thread = None
        if httpd is None:
            return
        try:
            httpd.shutdown()
            httpd.server_close()
        finally:
            if thread is not None:
                thread.join(timeout=5)

    # ---- security --------------------------------------------------------
    def _host_allowed(self, host: str | None) -> bool:
        p = self.port
        return (host or "").lower() in {f"127.0.0.1:{p}", f"localhost:{p}"}

    def _origin_allowed(self, origin: str) -> bool:
        p = self.port
        if origin.lower() in {f"http://127.0.0.1:{p}", f"http://localhost:{p}"}:
            return True
        prefix = "chrome-extension://"
        return origin.startswith(prefix) and origin[len(prefix):] in self._extension_ids

    # ---- routes ----------------------------------------------------------
    def _build_routes(self) -> list[tuple[str, re.Pattern, Callable[..., Any]]]:
        bid = BOOK_ID
        table: list[tuple[str, str, Callable[..., Any]]] = [
            ("GET", r"/api/info", self._info),
            ("GET", r"/api/voices", self._voices),
            ("POST", r"/api/speak", self._speak),
            ("POST", r"/api/pause", self._simple("pause")),
            ("POST", r"/api/resume", self._simple("resume")),
            ("POST", r"/api/stop", self._simple("stop")),
            ("POST", r"/api/seek", self._seek),
            ("GET", r"/api/status", self._status),
            ("GET", r"/api/books", self._books_list),
            ("POST", r"/api/books", self._books_add),
            ("GET", rf"/api/books/(?P<id>{bid})", self._book_get),
            ("GET", rf"/api/books/(?P<id>{bid})/chapters/(?P<n>\d{{1,9}})", self._chapter),
            ("PUT", rf"/api/books/(?P<id>{bid})/position", self._position),
            ("DELETE", rf"/api/books/(?P<id>{bid})", self._book_delete),
            ("POST", r"/api/open", self._open),
        ]
        return [(m, re.compile(p + r"\Z"), h) for m, p, h in table]

    def _info(self, req: "_Request") -> dict:
        return {"app": "vibetts", "version": self._api.version, "api": 1}

    def _voices(self, req: "_Request") -> dict:
        return self._api.list_voices()

    def _simple(self, name: str) -> Callable[["_Request"], dict]:
        def handler(req: "_Request") -> dict:
            req.json_body(optional=True)
            return getattr(self._api.session, name)()
        return handler

    def _speak(self, req: "_Request") -> dict:
        body = req.json_body()
        items = body.get("items")
        if not isinstance(items, list):
            raise _HttpError(400, "'items' must be an array")
        source = body.get("source", "reader")
        if not isinstance(source, str):
            raise _HttpError(400, "'source' must be a string")
        meta = body.get("meta")
        if meta is not None and not isinstance(meta, dict):
            raise _HttpError(400, "'meta' must be an object")
        start = body.get("start", 0)
        if isinstance(start, bool) or not isinstance(start, int):
            raise _HttpError(400, "'start' must be an integer")
        chars = sum(len(i["text"]) for i in items if isinstance(i, dict) and isinstance(i.get("text"), str))
        log.debug("speak: %d items, %d chars", len(items), chars)
        try:
            return self._api.session.start(
                items, start=start, voice=body.get("voice"), lang=body.get("lang"), speed=body.get("speed"),
                source=source, meta=meta, max_chars=self._max_chars)
        except RuntimeError as e:
            raise _HttpError(503, str(e)) from e

    def _seek(self, req: "_Request") -> dict:
        item = req.json_body().get("item")
        if isinstance(item, bool) or not isinstance(item, int):
            raise _HttpError(400, "'item' must be an integer")
        try:
            return self._api.session.seek(item)
        except RuntimeError as e:
            raise _HttpError(503, str(e)) from e

    def _status(self, req: "_Request") -> dict:
        since: int | None = None
        raw = req.query.get("since", [None])[0]
        if raw is not None and raw != "":
            try:
                since = int(raw)
            except ValueError:
                raise _HttpError(400, "'since' must be an integer") from None
        timeout = 0.0
        raw = req.query.get("timeout", [None])[0]
        if raw is not None and raw != "":
            try:
                timeout = float(raw)
            except ValueError:
                raise _HttpError(400, "'timeout' must be a number") from None
            if not math.isfinite(timeout):
                raise _HttpError(400, "'timeout' must be a number")
        timeout = min(max(timeout, 0.0), MAX_WAIT)
        return self._api.session.wait(since, timeout)

    def _books_list(self, req: "_Request") -> dict:
        return {"books": self._api.library.list()}

    def _books_add(self, req: "_Request") -> dict:
        data = req.binary_body(docload.MAX_FILE_BYTES)
        raw_name = req.headers.get("X-Filename")
        name = Path(unquote(raw_name).replace("\\", "/")).name if raw_name else ""
        if not name:
            raise _HttpError(400, "X-Filename header is required")
        log.debug("upload: %d bytes", len(data))
        return self._api.library.add(data, name)

    def _book_get(self, req: "_Request", id: str) -> dict:
        return self._api.library.get(id)

    def _chapter(self, req: "_Request", id: str, n: str) -> dict:
        return self._api.library.chapter(id, int(n))

    def _position(self, req: "_Request", id: str) -> dict:
        body = req.json_body()
        chapter, item = body.get("chapter"), body.get("item")
        for v in (chapter, item):
            if isinstance(v, bool) or not isinstance(v, int):
                raise _HttpError(400, "'chapter' and 'item' must be integers")
        self._api.library.set_position(id, chapter, item)
        return {"ok": True}

    def _book_delete(self, req: "_Request", id: str) -> dict:
        self._api.library.delete(id)
        return {"ok": True}

    def _open(self, req: "_Request") -> dict:
        path = req.json_body().get("path")
        if not isinstance(path, str) or not path:
            raise _HttpError(400, "'path' must be a non-empty string")
        try:
            return self._api.open_book(path)
        except RuntimeError as e:
            raise _HttpError(503, str(e)) from e

    # ---- static files ----------------------------------------------------
    def _static(self, raw_path: str) -> tuple[bytes, str] | None:
        path = unquote(raw_path)
        if path == "/":
            path = "/index.html"
        if (not path.startswith("/") or "\\" in path or ":" in path or "\x00" in path
                or ".." in path.split("/")):
            return None
        try:
            root = self._web_root.resolve()
            target = (root / path.lstrip("/")).resolve()
            target.relative_to(root)
            if not target.is_file():
                return None
            return target.read_bytes(), CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        except (ValueError, OSError):
            return None

    # ---- HTTP ------------------------------------------------------------
    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                log.debug("http: " + format, *args)

            def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
                log.debug("http error: " + format, *args)

            def _send(self, status: int, body: bytes = b"", ctype: str = "application/json; charset=utf-8") -> None:
                log.debug("%s %s -> %d", self.command, self.path.split("?", 1)[0], status)
                self.send_response(status)
                for k, v in SECURITY_HEADERS.items():
                    self.send_header(k, v)
                if body:
                    self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _send_json(self, status: int, obj: Any) -> None:
                self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

            def _send_error(self, status: int, message: str) -> None:
                self._send_json(status, {"error": message})

            def _not_allowed(self) -> None:
                self._send_error(405, "Method not allowed")

            do_OPTIONS = do_PATCH = do_HEAD = _not_allowed

            def _dispatch(self) -> None:
                method = self.command
                try:
                    if not server._host_allowed(self.headers.get("Host")):
                        raise _HttpError(403, "Forbidden host")
                    origin = self.headers.get("Origin")
                    if origin is not None and not server._origin_allowed(origin):
                        raise _HttpError(403, "Forbidden origin")
                    parts = urlsplit(self.path)
                    path = parts.path
                    if method == "GET" and not path.startswith("/api/"):
                        found = server._static(path)
                        if found is None:
                            raise _HttpError(404, "Not found")
                        return self._send(200, found[0], found[1])
                    handler, kwargs, matched = None, {}, False
                    for m, rx, h in server._routes:
                        mt = rx.match(path)
                        if mt is None:
                            continue
                        matched = True
                        if m == method:
                            handler, kwargs = h, mt.groupdict()
                            break
                    if handler is None:
                        raise _HttpError(405 if matched else 404, "Method not allowed" if matched else "Not found")
                    if method != "GET" and origin is None and not (method == "POST" and path == "/api/open"):
                        raise _HttpError(403, "Origin required")
                    req = _Request(self, parse_qs(parts.query))
                    self._send_json(200, handler(req, **kwargs))
                except _HttpError as e:
                    self._send_error(e.status, e.message or "Error")
                except (ReadingError, LibraryError, ValueError) as e:
                    self._send_error(400, str(e))
                except TypeError:
                    self._send_error(400, "Invalid request")
                except (KeyError, IndexError):
                    self._send_error(404, "Not found")
                except Exception:
                    log.exception("reader request failed: %s %s", method, self.path.split("?", 1)[0])
                    self._send_error(500, "Internal error")

            do_GET = do_POST = do_PUT = do_DELETE = _dispatch

        return Handler


class _Request:
    """Query string and body access for one request; raises _HttpError for malformed input."""

    def __init__(self, handler: BaseHTTPRequestHandler, query: dict[str, list[str]]) -> None:
        self._h = handler
        self.headers = handler.headers
        self.query = query

    def _length(self, limit: int) -> int:
        cl = self.headers.get("Content-Length")
        if cl is None:
            raise _HttpError(411, "Content-Length required")
        try:
            length = int(cl)
            if length < 0:
                raise ValueError
        except ValueError:
            raise _HttpError(400, "Bad Content-Length") from None
        if length > limit:
            raise _HttpError(413, "Request body too large")
        return length

    def _ctype(self) -> str:
        return (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()

    def json_body(self, optional: bool = False) -> dict:
        if self._ctype() != "application/json":
            # an empty body needs no type on routes where the body is optional
            if not (optional and self.headers.get("Content-Length") in (None, "0")):
                raise _HttpError(415, "Content-Type must be application/json")
        length = self._length(MAX_JSON_BYTES) if self.headers.get("Content-Length") is not None or not optional else 0
        raw = self._h.rfile.read(length) if length else b""
        if not raw.strip():
            if optional:
                return {}
            raise _HttpError(400, "A JSON body is required")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise _HttpError(400, "Invalid JSON") from None
        if not isinstance(body, dict):
            raise _HttpError(400, "JSON body must be an object")
        return body

    def binary_body(self, limit: int) -> bytes:
        if self._ctype() != "application/octet-stream":
            raise _HttpError(415, "Content-Type must be application/octet-stream")
        return self._h.rfile.read(self._length(limit))
