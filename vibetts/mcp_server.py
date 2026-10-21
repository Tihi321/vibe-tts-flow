"""Minimal stdlib MCP server (Streamable HTTP, JSON responses only) exposing VibeTTS tools to agents.

No third-party packages. Never prints (the windowed exe has no stdout); never logs tool argument text.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Protocol

from .config import SUPERTONIC_LANGUAGES

log = logging.getLogger(__name__)

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")  # newest first

INSTRUCTIONS = (
    "VibeTTS is a local text-to-speech app: Kokoro (English, Spanish, French, Hindi, Italian, Portuguese) and Supertonic 3 (31 languages, "
    "when installed). Use synthesize_to_file to turn text or a document (.txt/.md/.html/.docx/.epub/.pdf) into a "
    "WAV or MP3 file (44.1 kHz mono). Use speak to read text aloud on the user's speakers; use it sparingly "
    "because it is audible to the user. Call list_voices first to get valid voice ids and language codes; "
    "pass 'lang' to read in a specific language (omitted: the language is detected among the languages the user "
    "enabled and read with that language's configured voice). synthesize_batch writes many short clips in one call. Use stop to "
    "interrupt playback or cancel a running export."
)


class ToolError(Exception):
    """Raised by a tool handler; becomes a tool result with isError: true."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], dict]
    annotations: dict = field(default_factory=dict)


class AgentApi(Protocol):
    def list_voices(self) -> dict: ...

    def speak(self, text: str, voice: str | None, speed: float | None, wait: bool,
              timeout: float, lang: str | None = None) -> dict: ...

    def stop(self, cancel_export: bool) -> dict: ...

    def synthesize_to_file(self, args: dict) -> dict: ...

    def synthesize_batch(self, args: dict) -> dict: ...


def _error(id_: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False  # on Windows, reuse would permit double-binding a port


class McpServer:
    def __init__(self, tools: list[Tool], port: int, host: str = "127.0.0.1", path: str = "/mcp",
                 name: str = "vibetts", version: str = "", instructions: str = "",
                 max_body: int = 4 * 1024 * 1024) -> None:
        self._tools = {t.name: t for t in tools}
        self._order = list(tools)
        self._host = host
        self.port = port
        self._path = path
        self._name = name
        self._version = version
        self._instructions = instructions
        self._max_body = max_body
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self._path}"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._httpd is not None:
                return
            httpd = _Server((self._host, self.port), self._make_handler())
            self.port = httpd.server_address[1]
            self._httpd = httpd
            self._thread = threading.Thread(target=httpd.serve_forever, name="vibetts-mcp", daemon=True)
            self._thread.start()
            log.info("MCP server listening on %s", self.url)

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

    # ---- JSON-RPC --------------------------------------------------------
    def handle_message(self, msg: Any) -> dict | None:
        if not isinstance(msg, dict):
            return _error(None, -32600, "Invalid Request")
        id_ = msg.get("id")
        method = msg.get("method")
        if method is None:
            if "result" in msg or "error" in msg:
                return None  # a response from the client
            return _error(id_, -32600, "Invalid Request: missing method")
        if not isinstance(method, str):
            return _error(id_, -32600, "Invalid Request: method must be a string")
        params = msg.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error(id_, -32600, "Invalid Request: params must be an object")
        if method.startswith("notifications/") or "id" not in msg:
            return None
        try:
            if method == "initialize":
                return {"jsonrpc": "2.0", "id": id_, "result": self._initialize(params)}
            if method == "ping":
                return {"jsonrpc": "2.0", "id": id_, "result": {}}
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": id_, "result": {"tools": [self._describe(t) for t in self._order]}}
            if method == "tools/call":
                return self._call(id_, params)
        except Exception as e:  # defensive: never let dispatch kill the request
            log.exception("MCP dispatch failed")
            return _error(id_, -32603, f"Internal error: {type(e).__name__}")
        return _error(id_, -32601, f"Method not found: {method}")

    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        result: dict = {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self._name, "version": self._version},
        }
        if self._instructions:
            result["instructions"] = self._instructions
        return result

    @staticmethod
    def _describe(t: Tool) -> dict:
        d = {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
        if t.annotations:
            d["annotations"] = t.annotations
        return d

    def _call(self, id_: Any, params: dict) -> dict:
        name = params.get("name")
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            return _error(id_, -32602, f"Unknown tool: {name}")
        args = params.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return _error(id_, -32602, "arguments must be an object")
        text = args.get("text")
        log.info("tool call: %s%s", name, f" (text length {len(text)})" if isinstance(text, str) else "")
        try:
            out = tool.handler(args)
            result = {
                "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
                "structuredContent": out,
                "isError": False,
            }
        except ToolError as e:
            result = {"content": [{"type": "text", "text": str(e)}], "isError": True}
        except Exception as e:
            log.exception("tool %s failed", name)
            result = {"content": [{"type": "text", "text": f"Internal error: {type(e).__name__}: {e}"}],
                      "isError": True}
        return {"jsonrpc": "2.0", "id": id_, "result": result}

    # ---- HTTP ------------------------------------------------------------
    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                log.debug("http: " + format, *args)

            def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
                log.debug("http error: " + format, *args)

            def _send(self, status: int, body: bytes = b"", headers: dict | None = None) -> None:
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                if body:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _send_json(self, status: int, obj: Any) -> None:
                self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

            def _method_not_allowed(self) -> None:
                if self.path.split("?", 1)[0] != server._path:
                    self._send(404)
                else:
                    self._send(405, headers={"Allow": "POST"})

            do_GET = do_DELETE = do_PUT = do_PATCH = _method_not_allowed

            def do_POST(self) -> None:
                if self.path.split("?", 1)[0] != server._path:
                    return self._send(404)
                p = server.port
                allowed_hosts = {f"127.0.0.1:{p}", f"localhost:{p}"}
                if (self.headers.get("Host") or "").lower() not in allowed_hosts:
                    return self._send(403)
                origin = self.headers.get("Origin")
                if origin is not None and origin.lower() not in {f"http://{h}" for h in allowed_hosts}:
                    return self._send(403)
                pv = self.headers.get("MCP-Protocol-Version")
                if pv is not None and pv not in PROTOCOL_VERSIONS:
                    return self._send(400)
                ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                if ctype and ctype != "application/json":
                    return self._send(415)
                cl = self.headers.get("Content-Length")
                if cl is None:
                    return self._send(411)
                try:
                    length = int(cl)
                    if length < 0:
                        raise ValueError
                except ValueError:
                    return self._send(400)
                if length > server._max_body:
                    return self._send(413)
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    return self._send_json(400, _error(None, -32700, "Parse error"))
                if isinstance(payload, list):
                    if not payload:
                        return self._send_json(400, _error(None, -32600, "Invalid Request: empty batch"))
                    replies = [r for r in (server.handle_message(m) for m in payload) if r is not None]
                    if not replies:
                        return self._send(202)
                    return self._send_json(200, replies)
                reply = server.handle_message(payload)
                if reply is None:
                    return self._send(202)
                self._send_json(200, reply)

        return Handler


# ---- tools ---------------------------------------------------------------
_PASSTHROUGH_ERRORS = (FileExistsError, FileNotFoundError, ValueError, RuntimeError)


def _check_unknown(args: dict, allowed: set[str]) -> None:
    extra = sorted(set(args) - allowed)
    if extra:
        raise ToolError(f"Unknown argument(s): {', '.join(extra)}")


def _opt_str(args: dict, key: str) -> str | None:
    v = args.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise ToolError(f"'{key}' must be a string")
    return v


def _check_text(text: Any, max_chars: int) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ToolError("'text' must be a non-empty string")
    if max_chars > 0 and len(text) > max_chars:
        raise ToolError(f"Text is too long ({len(text)} characters); the limit is {max_chars} characters")
    return text


def _check_speed(args: dict) -> float | None:
    v = args.get("speed")
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.5 <= v <= 2:
        raise ToolError("'speed' must be a number between 0.5 and 2")
    return float(v)


def _check_enum(args: dict, key: str, values: tuple[str, ...]) -> str | None:
    v = args.get(key)
    if v is None:
        return None
    if v not in values:
        raise ToolError(f"'{key}' must be one of: {', '.join(values)}")
    return v


def _check_bool(args: dict, key: str, default: bool) -> bool:
    v = args.get(key, default)
    if not isinstance(v, bool):
        raise ToolError(f"'{key}' must be a boolean")
    return v


MAX_BATCH_ITEMS = 500


def build_tools(api: AgentApi, max_chars: int) -> list[Tool]:
    def synthesize(args: dict) -> dict:
        _check_unknown(args, {"text", "path", "output", "format", "voice", "lang", "speed",
                              "overwrite"})
        has_text, has_path = args.get("text") is not None, args.get("path") is not None
        if has_text == has_path:
            raise ToolError("Provide exactly one of 'text' or 'path'")
        if has_text:
            _check_text(args["text"], max_chars)
        else:
            _opt_str(args, "path")
        _opt_str(args, "output")
        _opt_str(args, "voice")
        _check_enum(args, "lang", SUPERTONIC_LANGUAGES)
        _check_speed(args)
        _check_enum(args, "format", ("wav", "mp3"))
        _check_bool(args, "overwrite", False)
        try:
            return api.synthesize_to_file(dict(args))
        except ToolError:
            raise
        except Exception as e:
            if type(e).__name__ == "ExportCancelled":
                raise ToolError("The export was cancelled; no file was written") from e
            if type(e).__name__ == "ExportBusy" or isinstance(e, _PASSTHROUGH_ERRORS):
                raise ToolError(str(e)) from e
            raise

    def speak(args: dict) -> dict:
        _check_unknown(args, {"text", "voice", "lang", "speed", "wait", "timeout_seconds"})
        text = _check_text(args.get("text"), max_chars)
        voice = _opt_str(args, "voice")
        lang = _check_enum(args, "lang", SUPERTONIC_LANGUAGES)
        speed = _check_speed(args)
        wait = _check_bool(args, "wait", False)
        timeout = args.get("timeout_seconds", 120)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
            raise ToolError("'timeout_seconds' must be a number greater than 0 and at most 600")
        try:
            return api.speak(text, voice, speed, wait, float(timeout), lang=lang)
        except ToolError:
            raise
        except Exception as e:
            if type(e).__name__ == "ExportBusy" or isinstance(e, _PASSTHROUGH_ERRORS):
                raise ToolError(str(e)) from e
            raise

    def list_voices(args: dict) -> dict:
        return api.list_voices()

    def synthesize_batch(args: dict) -> dict:
        _check_unknown(args, {"items", "overwrite"})
        items = args.get("items")
        if not isinstance(items, list) or not items:
            raise ToolError("'items' must be a non-empty array")
        if len(items) > MAX_BATCH_ITEMS:
            raise ToolError(f"Too many items ({len(items)}); the limit is {MAX_BATCH_ITEMS} per call")
        allowed = {"text", "output", "voice", "lang", "speed"}
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                raise ToolError(f"items[{i}] must be an object")
            extra = sorted(set(it) - allowed)
            if extra:
                raise ToolError(f"items[{i}]: unknown field(s): {', '.join(extra)}")
            if it.get("lang") is not None and it["lang"] not in SUPERTONIC_LANGUAGES:
                raise ToolError(f"items[{i}]: 'lang' must be one of: {', '.join(SUPERTONIC_LANGUAGES)}")
        _check_bool(args, "overwrite", False)
        try:
            return api.synthesize_batch(dict(args))
        except ToolError:
            raise
        except Exception as e:
            if type(e).__name__ == "ExportBusy" or isinstance(e, _PASSTHROUGH_ERRORS):
                raise ToolError(str(e)) from e
            raise

    def stop(args: dict) -> dict:
        _check_unknown(args, {"cancel_export"})
        return api.stop(_check_bool(args, "cancel_export", False))

    def schema(props: dict, required: list[str] | None = None) -> dict:
        return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}

    voice_p = {"type": "string", "description": "Voice id: a Kokoro voice (af_heart; Kokoro reads en, es, fr, hi, it, pt) or a Supertonic voice (supertonic:F1; reads all 31 languages); call list_voices for valid values. Defaults to the configured voice of the text's language."}
    lang_p = {"type": "string", "enum": list(SUPERTONIC_LANGUAGES), "description": "Language of the text (ISO code, e.g. en, hr, de). Default: detected among the languages the user enabled, read with that language's configured voice. A Kokoro voice only reads its own language (en, es, fr, hi, it, pt); Supertonic voices read all."}
    speed_p = {"type": "number", "minimum": 0.5, "maximum": 2, "description": "Speech speed multiplier, 0.5 to 2 (default 1)."}
    limit = f" At most {max_chars} characters." if max_chars > 0 else ""

    return [
        Tool(
            "synthesize_to_file",
            "Convert text or a document to a speech audio file (44.1 kHz mono WAV or MP3) using local TTS. "
            "Provide exactly one of 'text' or 'path'. Runs to completion and returns the output path, "
            "duration and size. Only one export can run at a time; long documents can take a while.",
            schema({
                "text": {"type": "string", "description": "Text to convert." + limit},
                "path": {"type": "string", "description": "Absolute path of a .txt, .md, .html, .docx, .epub or .pdf document to convert."},
                "output": {"type": "string", "description": "Absolute output path, or just a file name to save in the user's VibeTTS export folder; .wav or .mp3. Defaults to an auto-generated name in the export folder."},
                "format": {"type": "string", "enum": ["wav", "mp3"], "description": "Audio format (default from the output extension, else wav)."},
                "voice": voice_p,
                "lang": lang_p,
                "speed": speed_p,
                "overwrite": {"type": "boolean", "default": False, "description": "Replace the output file if it already exists (default false)."},
            }),
            synthesize,
            {"title": "Synthesize speech to an audio file", "readOnlyHint": False, "destructiveHint": False,
             "idempotentHint": False, "openWorldHint": False},
        ),
        Tool(
            "speak",
            "Read text aloud on the user's speakers. This is audible to the user, so use it sparingly. "
            "By default returns as soon as speech starts; set wait=true to block until it finishes.",
            schema({
                "text": {"type": "string", "description": "Text to speak." + limit},
                "voice": voice_p,
                "lang": lang_p,
                "speed": speed_p,
                "wait": {"type": "boolean", "default": False, "description": "Return when speech has finished instead of when it starts (default false)."},
                "timeout_seconds": {"type": "number", "default": 120, "maximum": 600, "description": "Maximum seconds to wait when wait=true (default 120, max 600)."},
            }, ["text"]),
            speak,
            {"title": "Speak text aloud", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
        ),
        Tool(
            "list_voices",
            "List the available voices (id, label, engine, languages), the language codes, the voice configured for "
            "each language (language_voices, empty = off), the read_language setting (auto or a code) and whether Supertonic is installed. Call this to get valid voice ids.",
            schema({}),
            list_voices,
            {"title": "List voices", "readOnlyHint": True, "openWorldHint": False},
        ),
        Tool(
            "synthesize_batch",
            "Convert many short texts to audio files in one call (WAV or MP3 by file extension). The "
            "voice model is loaded once, so this is much faster than repeated synthesize_to_file calls. "
            f"Up to {MAX_BATCH_ITEMS} items; each item is limited by the text size limit. Runs in the "
            "same one-at-a-time export queue and returns one result per item "
            "({output, duration_s, ok, error}); a failing item does not stop the others.",
            schema({
                "items": {
                    "type": "array", "minItems": 1, "maxItems": MAX_BATCH_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to convert." + limit},
                            "output": {"type": "string", "description": "Absolute output path ending in .wav or .mp3; the folder must exist."},
                            "voice": voice_p,
                            "lang": lang_p,
                            "speed": speed_p,
                        },
                        "required": ["text", "output"],
                        "additionalProperties": False,
                    },
                },
                "overwrite": {"type": "boolean", "default": False, "description": "Replace output files that already exist (default false; existing files then fail that item)."},
            }, ["items"]),
            synthesize_batch,
            {"title": "Synthesize many clips to files", "readOnlyHint": False, "destructiveHint": False,
             "idempotentHint": False, "openWorldHint": False},
        ),
        Tool(
            "stop",
            "Stop speech that is currently playing. Optionally also cancel a running file export.",
            schema({
                "cancel_export": {"type": "boolean", "default": False, "description": "Also cancel a running synthesize_to_file export (default false)."},
            }),
            stop,
            {"title": "Stop speech", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
        ),
    ]
