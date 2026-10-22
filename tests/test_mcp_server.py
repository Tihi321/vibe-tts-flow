"""Tests for the stdlib MCP server and its tool layer."""
from __future__ import annotations

import http.client
import json
import socket

import pytest

from vibetts.mcp_server import PROTOCOL_VERSIONS, McpServer, Tool, ToolError, build_tools


class FakeApi:
    def __init__(self):
        self.calls = []
        self.raise_exc = None

    def _maybe_raise(self):
        if self.raise_exc:
            raise self.raise_exc

    def list_voices(self):
        self.calls.append(("list_voices",))
        return {"voices": ["af_heart"]}

    def speak(self, text, voice, speed, wait, timeout, lang=None):
        self.calls.append(("speak", text, voice, speed, wait, timeout, lang))
        self._maybe_raise()
        return {"ok": True}

    def stop(self, cancel_export):
        self.calls.append(("stop", cancel_export))
        return {"stopped": True}

    def synthesize_batch(self, args):
        self.calls.append(("batch", args))
        self._maybe_raise()
        return {"results": []}

    def synthesize_to_file(self, args):
        self.calls.append(("synth", args))
        self._maybe_raise()
        return {"path": "C:\\x.wav"}


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def server(api):
    return McpServer(build_tools(api, 100), port=0, version="1.0", instructions="hi")


def rpc(server, method, params=None, id_=1):
    msg = {"jsonrpc": "2.0", "method": method}
    if id_ is not None:
        msg["id"] = id_
    if params is not None:
        msg["params"] = params
    return server.handle_message(msg)


def call(server, name, args):
    return rpc(server, "tools/call", {"name": name, "arguments": args})["result"]


def test_initialize_negotiation(server):
    r = rpc(server, "initialize", {"protocolVersion": "2025-06-18"})["result"]
    assert r["protocolVersion"] == "2025-06-18"
    assert r["serverInfo"] == {"name": "vibetts", "version": "1.0"}
    assert r["instructions"] == "hi"
    assert r["capabilities"] == {"tools": {"listChanged": False}}
    assert rpc(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]


def test_ping_and_list(server):
    assert rpc(server, "ping")["result"] == {}
    tools = rpc(server, "tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["synthesize_to_file", "speak", "list_voices", "synthesize_batch",
                                           "stop"]
    assert all(t["inputSchema"]["additionalProperties"] is False for t in tools)


def test_call_success(server, api):
    r = call(server, "list_voices", {})
    assert r["isError"] is False
    assert r["structuredContent"] == {"voices": ["af_heart"]}
    assert json.loads(r["content"][0]["text"]) == {"voices": ["af_heart"]}


def test_tool_error_and_generic_exception():
    def bad(args):
        raise ToolError("nope")

    def boom(args):
        raise KeyError("k")

    s = McpServer([Tool("bad", "", {}, bad), Tool("boom", "", {}, boom)], port=0)
    r = call(s, "bad", {})
    assert r["isError"] is True and r["content"][0]["text"] == "nope"
    r = call(s, "boom", {})
    assert r["isError"] is True and r["content"][0]["text"].startswith("Internal error: KeyError")


def test_errors_and_notifications(server):
    assert rpc(server, "tools/call", {"name": "zzz"})["error"]["code"] == -32602
    assert rpc(server, "nope")["error"]["code"] == -32601
    assert rpc(server, "notifications/initialized", id_=None) is None
    assert server.handle_message({"jsonrpc": "2.0", "id": 1, "result": {}}) is None
    assert server.handle_message({"jsonrpc": "2.0", "id": 1})["error"]["code"] == -32600
    assert server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": 3})["error"]["code"] == -32600


def test_synthesize_validation(server, api):
    for args in [
        {"text": "a", "path": "b"},
        {},
        {"text": "x" * 101},
        {"text": "  "},
        {"text": "a", "speed": 3},
        {"text": "a", "format": "ogg"},
        {"text": "a", "lang": "xx"},
        {"text": "a", "lang": 3},
        {"text": "a", "bogus": 1},
    ]:
        assert call(server, "synthesize_to_file", args)["isError"] is True, args
    assert "100" in call(server, "synthesize_to_file", {"text": "x" * 101})["content"][0]["text"]
    assert api.calls == []
    r = call(server, "synthesize_to_file", {"text": "hello", "format": "mp3"})
    assert r["isError"] is False and api.calls == [("synth", {"text": "hello", "format": "mp3"})]


def test_api_exceptions_mapped(server, api):
    class ExportBusy(RuntimeError):
        pass

    for exc in (ExportBusy("busy now"), FileExistsError("exists"), ValueError("bad doc")):
        api.raise_exc = exc
        r = call(server, "synthesize_to_file", {"text": "hi"})
        assert r["isError"] is True and r["content"][0]["text"] == str(exc)


def test_cancelled_export_is_a_tool_error(server, api):
    class ExportCancelled(Exception):
        pass

    api.raise_exc = ExportCancelled()
    r = call(server, "synthesize_to_file", {"text": "hi"})
    assert r["isError"] is True and "cancelled" in r["content"][0]["text"]


def test_speak_and_stop(server, api):
    assert call(server, "speak", {"text": "hi", "wait": True, "timeout_seconds": 5})["isError"] is False
    assert api.calls[-1] == ("speak", "hi", None, None, True, 5.0, None)
    assert call(server, "speak", {"text": "x" * 101})["isError"] is True
    assert call(server, "speak", {"text": "hi", "timeout_seconds": 601})["isError"] is True
    assert call(server, "speak", {"text": "hi", "speed": 0.1})["isError"] is True
    assert call(server, "speak", {"text": "hi", "lang": "xx"})["isError"] is True
    assert call(server, "speak", {"text": "hi", "lang": "hr", "voice": "supertonic:F1"})["isError"] is False
    assert api.calls[-1] == ("speak", "hi", "supertonic:F1", None, False, 120.0, "hr")
    call(server, "stop", {"cancel_export": True})
    assert api.calls[-1] == ("stop", True)


# ---- HTTP ----
@pytest.fixture
def live(server):
    server.start()
    yield server
    server.stop()


def post(srv, body, headers=None, path="/mcp", host=None, method="POST"):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    c.putrequest(method, path, skip_host=True)
    c.putheader("Host", host or f"127.0.0.1:{srv.port}")
    if isinstance(body, (dict, list)):
        body = json.dumps(body)
    raw = body.encode() if isinstance(body, str) else (body or b"")
    for k, v in h.items():
        c.putheader(k, v)
    c.putheader("Content-Length", str(len(raw)))
    c.endheaders(raw)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}


def test_http_initialize(live):
    r, data = post(live, INIT)
    assert r.status == 200 and r.getheader("Content-Type") == "application/json"
    assert json.loads(data)["result"]["protocolVersion"] == "2025-11-25"
    assert live.running and live.url == f"http://127.0.0.1:{live.port}/mcp"


def test_http_notification_and_batch(live):
    r, data = post(live, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r.status == 202 and data == b""
    r, data = post(live, [INIT, {"jsonrpc": "2.0", "method": "notifications/x"}, {"jsonrpc": "2.0", "id": 2, "method": "ping"}])
    out = json.loads(data)
    assert r.status == 200 and [m["id"] for m in out] == [1, 2]


def test_http_methods_and_path(live):
    r, _ = post(live, "", method="GET")
    assert r.status == 405 and r.getheader("Allow") == "POST"
    r, _ = post(live, "", method="DELETE")
    assert r.status == 405
    r, _ = post(live, INIT, path="/other")
    assert r.status == 404


def test_http_host_origin(live):
    assert post(live, INIT, host="evil.com")[0].status == 403
    assert post(live, INIT, host=f"LOCALHOST:{live.port}")[0].status == 200
    assert post(live, INIT, headers={"Origin": "http://evil.com"})[0].status == 403
    assert post(live, INIT, headers={"Origin": f"http://localhost:{live.port}"})[0].status == 200


def test_http_bad_requests(live):
    r, data = post(live, "{not json")
    assert r.status == 400 and json.loads(data)["error"]["code"] == -32700
    assert post(live, INIT, headers={"MCP-Protocol-Version": "1.0"})[0].status == 400
    assert post(live, INIT, headers={"MCP-Protocol-Version": "2025-06-18"})[0].status == 200


def test_http_oversize(api):
    s = McpServer(build_tools(api, 100), port=0, max_body=50)
    s.start()
    try:
        assert post(s, "x" * 200)[0].status == 413
    finally:
        s.stop()


def test_stop_rebinds_and_idempotent(server):
    server.stop()  # never started
    server.start()
    port = server.port
    other = McpServer([], port=port)
    with pytest.raises(OSError):
        other.start()
    server.stop()
    server.stop()
    assert server.running is False
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def test_synthesize_batch_validation(server, api):
    good = {"text": "Hi.", "output": "C:/x/a.mp3", "voice": "bf_emma", "speed": 1}
    r = call(server, "synthesize_batch", {"items": [good], "overwrite": True})
    assert r["isError"] is False and api.calls[-1][0] == "batch"
    for args in [
        {}, {"items": []}, {"items": "x"}, {"items": [3]}, {"items": [{"text": "a", "output": "b", "zz": 1}]},
        {"items": [{"text": "a", "output": "b", "lang": "xx"}]},
        {"items": [good], "overwrite": "yes"}, {"items": [good], "other": 1},
        {"items": [good] * 501},
    ]:
        n = len(api.calls)
        assert call(server, "synthesize_batch", args)["isError"] is True, args
        assert len(api.calls) == n


def test_lang_param_is_an_enum_of_all_codes(server):
    from vibetts.config import ALL_LANGUAGES

    tools = {t["name"]: t for t in rpc(server, "tools/list")["result"]["tools"]}
    assert len(ALL_LANGUAGES) == 101
    for name in ("synthesize_to_file", "speak"):
        lang = tools[name]["inputSchema"]["properties"]["lang"]
        assert lang["type"] == "string" and lang["enum"] == list(ALL_LANGUAGES)
    item = tools["synthesize_batch"]["inputSchema"]["properties"]["items"]["items"]
    assert item["properties"]["lang"]["enum"] == list(ALL_LANGUAGES)
    assert {"hr", "en", "ja", "bs", "sr", "mk", "ca"} <= set(ALL_LANGUAGES)
    assert "lang" not in tools["list_voices"]["inputSchema"]["properties"]


def test_lang_is_passed_through(server, api):
    assert call(server, "synthesize_to_file", {"text": "x", "lang": "hr"})["isError"] is False
    assert api.calls[-1] == ("synth", {"text": "x", "lang": "hr"})
    item = {"text": "Hi.", "output": "C:/x/a.mp3", "lang": "de", "voice": "supertonic:M1"}
    assert call(server, "synthesize_batch", {"items": [item]})["isError"] is False
    assert api.calls[-1][1]["items"][0]["lang"] == "de"


def test_descriptions_say_44k(server):
    from vibetts.mcp_server import INSTRUCTIONS

    tools = rpc(server, "tools/list")["result"]["tools"]
    text = INSTRUCTIONS + " ".join(t["description"] for t in tools)
    assert "44.1 kHz" in text and "24 kHz" not in text
    assert "Supertonic" in INSTRUCTIONS


def test_voice_and_lang_text_describes_per_language_voices(server):
    from vibetts.mcp_server import INSTRUCTIONS

    tools = {t["name"]: t for t in rpc(server, "tools/list")["result"]["tools"]}
    assert "Spanish" in INSTRUCTIONS and "other-language" not in INSTRUCTIONS
    speak = tools["speak"]["inputSchema"]["properties"]
    assert "en, es, fr, hi, it, pt" in speak["voice"]["description"]
    assert "English only" not in speak["lang"]["description"]
    assert "language_voices" in tools["list_voices"]["description"]
    assert "other-language" not in tools["list_voices"]["description"]


def test_descriptions_mention_higgs(server):
    from vibetts import mcp_server

    tools = {t["name"]: t for t in rpc(server, "tools/list")["result"]["tools"]}
    assert "higgs:" in mcp_server.INSTRUCTIONS and "slower than real time" in mcp_server.INSTRUCTIONS
    voice = tools["speak"]["inputSchema"]["properties"]["voice"]["description"]
    assert "higgs:F1" in voice and "slower than real time" in voice
    assert "higgs_installed" in tools["list_voices"]["description"]


def test_higgs_only_language_is_accepted(server, api):
    assert call(server, "synthesize_to_file", {"text": "x", "lang": "bs", "voice": "higgs:F1"})["isError"] is False
    assert api.calls[-1] == ("synth", {"text": "x", "lang": "bs", "voice": "higgs:F1"})
