import json
import struct
import threading
from pathlib import Path

import numpy as np
import pytest

from vibetts import model_setup, omnivoice, omnivoice_setup as setup


# ---------------------------------------------------------------- helpers

def bf16_bytes(a: np.ndarray) -> bytes:
    """float32 -> bfloat16 by truncating the low 16 bits (exact for values made that way)."""
    return (np.asarray(a, "<f4").view("<u4") >> 16).astype("<u2").tobytes()


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, blob = {}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    h = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + blob)


def rng_weight(n: int, k: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, k)).astype(np.float32)


# ---------------------------------------------------------------- safetensors reader

def test_reader_f32_f16_bf16_and_ints(tmp_path):
    f32 = np.arange(6, dtype="<f4").reshape(2, 3) * 0.5
    f16 = np.array([[1.5, -2.25], [0.0, 65504.0]], "<f2")
    bf_src = np.array([1.0, -2.0, 0.5, 3.0, 1e10, -1e-10], "<f4")
    bf_src = (bf_src.view("<u4") & 0xFFFF0000).view("<f4")  # exactly representable in bf16
    i64 = np.array([0, 64, 128], "<i8")
    p = tmp_path / "t.safetensors"
    write_safetensors(p, {"a": ("F32", [2, 3], f32.tobytes()), "b": ("F16", [2, 2], f16.tobytes()),
                          "c": ("BF16", [2, 3], bf16_bytes(bf_src)),
                          "d": ("I64", [3], i64.tobytes())})
    st = setup.SafeTensors(p)
    try:
        assert st.keys() == ["a", "b", "c", "d"]
        assert st.shape("a") == [2, 3]
        got = st.get("a")
        assert got.dtype == np.float32 and np.array_equal(got, f32)
        got = st.get("b")
        assert got.dtype == np.float32 and np.array_equal(got, f16.astype(np.float32))
        got = st.get("c")
        assert got.dtype == np.float32 and np.array_equal(got, bf_src.reshape(2, 3))
        assert np.array_equal(st.get("d"), i64)
        with pytest.raises(setup.SetupError, match="no tensor"):
            st.get("missing")
    finally:
        st.close()


def test_reader_rejects_truncated_and_garbage(tmp_path):
    p = tmp_path / "t.safetensors"
    write_safetensors(p, {"a": ("F32", [4], np.zeros(4, "<f4").tobytes())})
    p.write_bytes(p.read_bytes()[:-4])
    st = setup.SafeTensors(p)
    with pytest.raises(setup.SetupError, match="truncated"):
        st.get("a")
    st.close()
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\xff" * 40)
    with pytest.raises(setup.SetupError):
        setup.SafeTensors(bad)


# ---------------------------------------------------------------- int4 quantizer

def test_quantizer_hand_computed_vectors():
    # Block 1: amax is -8 (first value of largest magnitude), scale 1: q = v + 8
    blk1 = np.tile(np.arange(-8, 8, dtype=np.float32), 8)
    # Block 2: amax +4 -> scale -0.5, reciprocal -2; 0.75 -> 6.5 rounds away from zero to 7
    blk2 = np.zeros(128, np.float32)
    blk2[:4] = [4.0, 0.75, -4.0, 1.0]
    # Block 3: all zeros -> scale 0, every value 8
    blk3 = np.zeros(128, np.float32)
    w = np.stack([np.concatenate([blk1, blk2, blk3]), np.concatenate([blk2, blk3, blk1])])
    packed, scales = setup.quantize_q4(w)
    assert packed.shape == (2, 3, 64) and packed.dtype == np.uint8
    assert scales.shape == (2, 3) and scales.dtype == np.float32
    assert scales.tolist() == [[1.0, -0.5, 0.0], [-0.5, 0.0, 1.0]]
    assert packed[0, 0].tolist() == [(2 * i) | ((2 * i + 1) << 4) for i in range(8)] * 8
    expected2 = [0x70, 0x6F] + [0x88] * 62
    assert packed[0, 1].tolist() == expected2 and packed[1, 0].tolist() == expected2
    assert packed[0, 2].tolist() == [0x88] * 64


def test_quantizer_rejects_ragged_k():
    with pytest.raises(setup.SetupError):
        setup.quantize_q4(np.zeros((2, 100), np.float32))


def test_quantizer_matches_onnxruntime():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    n, k = 16, 384
    w = rng_weight(n, k, 3)
    w[1, :128] = 0.0          # an all-zero block
    w[2, 5] = -w[2, 5] * 0 + 9.0   # a large positive outlier: positive amax
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"], name="/m/layers.0/mlp/up_proj/MatMul")],
        "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, n])],
        [numpy_helper.from_array(np.ascontiguousarray(w.T), "W")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    quant = MatMulNBitsQuantizer(model, block_size=128, is_symmetric=True, bits=4)
    quant.process()
    inits = {t.name: numpy_helper.to_array(t) for t in quant.model.model.graph.initializer}
    ref_b = next(v for key, v in inits.items() if key.endswith("_Q4"))
    ref_s = next(v for key, v in inits.items() if key.endswith("_scales"))
    packed, scales = setup.quantize_q4(w)
    assert np.array_equal(packed, ref_b.reshape(packed.shape))
    assert np.array_equal(scales.reshape(-1), ref_s.reshape(-1))


# ---------------------------------------------------------------- converter on a tiny model

def tiny_model(tmp_path: Path):
    """A fake safetensors (F32 + BF16 sources) with a hand-made weight map for it."""
    q1 = rng_weight(4, 256, 1)
    q2 = rng_weight(2, 128, 2)
    q2_bf = (q2.view("<u4") & 0xFFFF0000).view("<f4")
    norm = rng_weight(1, 8, 3)[0]
    st_path = tmp_path / "model.safetensors"
    write_safetensors(st_path, {
        "audio_embeddings.weight": ("F32", [1, 1], np.zeros(1, "<f4").tobytes()),
        "llm.layers.0.q_proj.weight": ("F32", [4, 256], q1.tobytes()),
        "llm.layers.0.k_proj.weight": ("BF16", [2, 128], bf16_bytes(q2_bf)),
        "llm.norm.weight": ("F32", [8], norm.tobytes())})
    # layout: gap of 16 bytes at the start, then norm, q1 (B, scales), q2 (B, scales)
    off = 16
    spec_norm = {"name": "m.norm.weight", "offset": off, "length": 32, "dtype": "float32",
                 "shape": [8]}
    off += 32
    b1 = {"name": "q1_Q4", "offset": off, "length": 4 * 2 * 64, "dtype": "uint8",
          "shape": [4, 2, 64]}
    off += b1["length"]
    s1 = {"name": "q1_scales", "offset": off, "length": 4 * 2 * 4, "dtype": "float32",
          "shape": [4, 2]}
    off += s1["length"]
    b2 = {"name": "q2_Q4", "offset": off, "length": 2 * 1 * 64, "dtype": "uint8",
          "shape": [2, 1, 64]}
    off += b2["length"]
    s2 = {"name": "q2_scales", "offset": off, "length": 2 * 1 * 4, "dtype": "float32",
          "shape": [2, 1]}
    off += s2["length"]
    wm = {"format": 1, "data_file": "x.data", "data_size": off + 8,
          "quant": {"bits": 4, "block_size": 128, "symmetric": True},
          "tensors": [
              {"kind": "q4", "tensor": "llm.layers.0.k_proj.weight", "shape": [2, 128],
               "B": b2, "scales": s2},  # deliberately listed out of offset order
              {"kind": "copy", "tensor": "llm.norm.weight", **spec_norm},
              {"kind": "q4", "tensor": "llm.layers.0.q_proj.weight", "shape": [4, 256],
               "B": b1, "scales": s1}]}
    expected = bytearray(wm["data_size"])
    expected[16:48] = norm.astype("<f4").tobytes()
    for w, b, s in ((q1, b1, s1), (q2_bf, b2, s2)):
        packed, scales = setup.quantize_q4(w)
        expected[b["offset"]:b["offset"] + b["length"]] = packed.tobytes()
        expected[s["offset"]:s["offset"] + s["length"]] = scales.astype("<f4").tobytes()
    return st_path, wm, bytes(expected)


def test_convert_llm_end_to_end(tmp_path):
    st_path, wm, expected = tiny_model(tmp_path)
    out = tmp_path / "out" / "x.data"
    seen = []
    setup.convert_llm(st_path, wm, out, lambda text, d, t: seen.append((text, d, t)))
    assert out.read_bytes() == expected
    assert seen[0] == ("Converting 0 / 3", 0, 3) and seen[-1] == ("Converting 3 / 3", 3, 3)


def test_convert_llm_cancel_and_bad_input(tmp_path):
    st_path, wm, _ = tiny_model(tmp_path)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(model_setup.DownloadCancelled):
        setup.convert_llm(st_path, wm, tmp_path / "a.data", cancel=cancel)
    wm["tensors"][1]["shape"] = [9]
    with pytest.raises(setup.SetupError, match="shape"):
        setup.convert_llm(st_path, wm, tmp_path / "b.data")
    wm["tensors"][1]["shape"] = [8]
    wm["tensors"][1]["length"] = 99
    with pytest.raises(setup.SetupError, match="expected 99"):
        setup.convert_llm(st_path, wm, tmp_path / "c.data")


def test_shipped_weight_map_is_consistent():
    wm = setup.load_weight_map()
    assert wm["quant"] == {"bits": 4, "block_size": 128, "symmetric": True}
    spans = sorted((s["offset"], s["length"]) for e in wm["tensors"] for s in setup._out_specs(e))
    assert spans[0][0] == 0
    for (o1, l1), (o2, _) in zip(spans, spans[1:]):
        assert o1 + l1 <= o2
    assert spans[-1][0] + spans[-1][1] <= wm["data_size"]
    graph = setup.data_dir() / setup.LLM_NAME
    assert graph.stat().st_size < 3_000_000  # graph only, no weights
    assert wm["data_file"] == setup.LLM_DATA
    q4 = [e for e in wm["tensors"] if e["kind"] == "q4"]
    assert len(q4) == 196 and all(e["tensor"].startswith("llm.layers.") for e in q4)


# ---------------------------------------------------------------- install plan

def test_plan_urls_sizes_and_totals():
    items = setup.plan(True)
    urls = {i.name: i.url for i in items}
    base_c = "https://huggingface.co/Emerald7664/OmniVoice-Onnx/resolve/" \
             "f312525229ea4c4d9101c2b3e317ac77d995ec59/"
    base_u = "https://huggingface.co/k2-fsa/OmniVoice/resolve/" \
             "c5fdb5ccb189668d56333f77ba2629f4cd7535f4/"
    assert urls["higgs_decoder.onnx.data"] == base_c + "audio_tokenizer/fp16/higgs_decoder.onnx.data"
    assert urls["tokenizer.json"] == base_c + "tokenizer.json"
    assert urls["model.safetensors"] == base_u + "model.safetensors"
    assert urls["config.json"] == base_u + "config.json"
    assert urls["semantic_encoder.onnx"] == base_c + "audio_tokenizer/semantic_encoder.onnx"
    assert {i.name for i in setup.plan(False)} == {i.name for i in items} - {
        "acoustic_encoder.onnx", "semantic_encoder.onnx", "quantizer_encoder.onnx"}
    assert next(i for i in items if i.name == "model.safetensors").size == 2_450_344_112
    assert setup.DOWNLOAD_BYTES(False) == sum(i.size for i in setup.plan(False)) == 2_849_402_794
    assert setup.DOWNLOAD_BYTES(True) - setup.DOWNLOAD_BYTES(False) == 654_414_629


# ---------------------------------------------------------------- install / uninstall / status

class FakeWorld:
    """Tiny plan, fake downloads and a fake graph + map, so install() runs offline."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.tmp = tmp_path
        st_path, self.wm, self.expected = tiny_model(tmp_path / "src")  # tiny_model needs dir
        self.st_bytes = st_path.read_bytes()
        self.files = {"a.onnx": b"A" * 10, "tokenizer.json": b"{}", "e1.onnx": b"E" * 7,
                      "model.safetensors": self.st_bytes, "config.json": b"{}"}
        mp = monkeypatch
        mp.setattr(setup, "COMMUNITY_FILES", [("a.onnx", 10), ("sub/tokenizer.json", 2)])
        mp.setattr(setup, "ENCODER_FILES", [("audio_tokenizer/e1.onnx", 7)])
        mp.setattr(setup, "UPSTREAM_FILES", [("config.json", 2),
                                             ("model.safetensors", len(self.st_bytes))])
        d = tmp_path / "data"
        d.mkdir()
        (d / setup.LLM_NAME).write_bytes(b"GRAPH")
        self.wm["data_file"] = setup.LLM_DATA
        mp.setattr(setup, "data_dir", lambda: d)
        mp.setattr(setup, "load_weight_map", lambda path=None: self.wm)
        self.downloads: list[str] = []
        mp.setattr(model_setup, "download", self.fake_download)
        self.voice_calls: list = []
        mp.setattr(setup, "_default_voice", self.fake_voice)

    def fake_download(self, filename, dest_dir, progress=None, cancel=None, expected_size=None,
                      url=None):
        self.downloads.append(url)
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        data = self.files[filename]
        assert len(data) == expected_size
        if progress:
            progress(0, len(data))
        if cancel is not None and cancel.is_set():
            raise model_setup.DownloadCancelled()
        if progress:
            progress(len(data), len(data))
        (Path(dest_dir) / filename).write_bytes(data)
        return Path(dest_dir) / filename

    def fake_voice(self, model_dir, seed, emit, cancel):
        if (Path(model_dir) / "voices" / "default.npy").is_file():  # like the real one
            return
        self.voice_calls.append(seed)
        emit("Creating the default voice", 0, None)
        v = Path(model_dir) / "voices"
        v.mkdir(exist_ok=True)
        np.save(v / "default.npy", np.zeros((8, 3), np.int16))
        (v / "default.txt").write_text("x\n")


@pytest.fixture
def world(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    return FakeWorld(tmp_path, monkeypatch)


def test_install_full_flow(world):
    dest = world.tmp / "model"
    events = []
    setup.install(dest, True, lambda t, d, n: events.append((t, d, n)))
    assert (dest / "a.onnx").read_bytes() == b"A" * 10
    assert (dest / "tokenizer.json").read_bytes() == b"{}"  # flat, like today
    assert (dest / "encoders" / "e1.onnx").is_file()
    assert (dest / setup.LLM_DATA).read_bytes() == world.expected
    assert (dest / setup.LLM_NAME).read_bytes() == b"GRAPH"
    assert not (dest / setup.CACHE_DIR).exists()  # safetensors removed
    assert world.voice_calls == [0]
    manifest = json.loads((dest / "manifest.json").read_text())
    assert set(manifest) == {"files", "sources", "license", "built", "encoders"}
    assert manifest["files"]["tokenizer.json"] == 2
    assert manifest["sources"]["llm"]["commit"] == setup.UPSTREAM_REV
    assert manifest["encoders"]["dir"] == "encoders"
    texts = [t for t, _, _ in events]
    assert any(t.startswith("Downloading model.safetensors") for t in texts)
    assert "Converting 0 / 3" in texts and "Creating the default voice" in texts
    assert texts.index("Converting 0 / 3") < texts.index("Creating the default voice")
    dl = [(d, n) for t, d, n in events if t.startswith("Downloading")]
    assert dl[-1][0] == dl[-1][1]  # overall bytes reach the total
    assert all(url.startswith("https://huggingface.co/") for url in world.downloads)
    st = setup.status(dest)
    assert st["installed"] and st["bytes_on_disk"] > len(world.expected)
    assert omnivoice.models_present(dest)


def test_install_without_encoders_and_keep_cache(world):
    dest = world.tmp / "model"
    setup.install(dest, False, keep_cache=True)
    assert not (dest / "encoders").exists()
    assert "encoders" not in json.loads((dest / "manifest.json").read_text())
    assert (dest / setup.CACHE_DIR / "model.safetensors").read_bytes() == world.st_bytes
    assert setup.status(dest)["encoders"] is False
    assert setup.status(dest)["installed"] is True


def test_install_skips_what_is_present(world):
    dest = world.tmp / "model"
    setup.install(dest, True)
    first = len(world.downloads)
    assert first == 5  # 2 community, config, safetensors, 1 encoder
    world.voice_calls.clear()
    setup.install(dest, True)
    assert len(world.downloads) == first  # nothing fetched again, not even the safetensors
    assert (dest / setup.LLM_DATA).read_bytes() == world.expected


def test_install_resumes_after_cancel(world, monkeypatch):
    dest = world.tmp / "model"
    cancel = threading.Event()
    real = world.fake_download

    def cancelling(filename, *a, **kw):
        if filename == "model.safetensors":
            cancel.set()
        return real(filename, *a, **kw)

    monkeypatch.setattr(model_setup, "download", cancelling)
    with pytest.raises(model_setup.DownloadCancelled):
        setup.install(dest, True, cancel=cancel)
    assert not (dest / "manifest.json").exists()
    assert not omnivoice.models_present(dest)
    assert (dest / "a.onnx").is_file()  # finished downloads are kept
    monkeypatch.setattr(model_setup, "download", real)
    cancel.clear()
    done = len(world.downloads)
    setup.install(dest, True, cancel=cancel)
    assert omnivoice.models_present(dest)
    assert len(world.downloads) - done == 2  # safetensors and the encoder (a.onnx etc. are present)


def test_install_cancel_during_convert_leaves_no_partial_data(world):
    dest = world.tmp / "model"
    cancel = threading.Event()
    events = []

    def progress(text, done, total):
        events.append(text)
        if text.startswith("Converting 1 /"):
            cancel.set()

    with pytest.raises(model_setup.DownloadCancelled):
        setup.install(dest, False, progress, cancel)
    assert not (dest / setup.LLM_DATA).exists()
    assert not (dest / (setup.LLM_DATA + ".part")).exists()
    assert not (dest / "manifest.json").exists()


def test_install_uses_local_dirs_without_downloading(world):
    local = world.tmp / "have"
    (local / "cpu").mkdir(parents=True)
    (local / "hf_upstream").mkdir()
    (local / "cpu" / "a.onnx").write_bytes(b"A" * 10)
    (local / "cpu" / "sub").mkdir()
    (local / "cpu" / "sub" / "tokenizer.json").write_bytes(b"{}")
    (local / "cpu" / "audio_tokenizer").mkdir()
    (local / "cpu" / "audio_tokenizer" / "e1.onnx").write_bytes(b"E" * 7)
    (local / "hf_upstream" / "config.json").write_bytes(b"{}")
    (local / "hf_upstream" / "model.safetensors").write_bytes(world.st_bytes)
    dest = world.tmp / "model"
    setup.install(dest, True, local_dirs=[local], keep_cache=False)
    assert world.downloads == []
    assert (dest / setup.LLM_DATA).read_bytes() == world.expected
    assert (local / "hf_upstream" / "model.safetensors").is_file()  # never deleted


def test_install_wrong_size_local_file_is_ignored(world):
    local = world.tmp / "have"
    (local / "hf_upstream").mkdir(parents=True)
    (local / "hf_upstream" / "model.safetensors").write_bytes(b"short")
    setup.install(world.tmp / "model", False, local_dirs=[local])
    assert any(u.endswith("model.safetensors") for u in world.downloads)


def test_uninstall_keeps_voices_by_default(world):
    dest = world.tmp / "model"
    setup.install(dest, True, keep_cache=True)
    setup.uninstall(dest)
    assert sorted(p.name for p in dest.iterdir()) == ["voices"]
    st = setup.status(dest)
    assert st["installed"] is False and st["encoders"] is False
    setup.uninstall(dest, keep_voices=False)
    assert not dest.exists()
    setup.uninstall(dest)  # nothing there: no error


def test_uninstall_leaves_unknown_files(world):
    dest = world.tmp / "model"
    setup.install(dest, False)
    (dest / "notes.txt").write_text("mine")
    setup.uninstall(dest, keep_voices=False)
    assert [p.name for p in dest.iterdir()] == ["notes.txt"]


def test_status_of_empty_dir(tmp_path):
    assert setup.status(tmp_path / "nothing") == {"installed": False, "encoders": False,
                                                  "bytes_on_disk": 0}


def test_missing_bundled_graph_is_a_setup_error(world, monkeypatch):
    (world.tmp / "data" / setup.LLM_NAME).unlink()
    with pytest.raises(setup.SetupError, match="graph"):
        setup.install(world.tmp / "model", False)


def test_default_voice_uses_engine_and_seed(tmp_path, monkeypatch):
    calls = []

    class Eng:
        def __init__(self, model_dir, steps=6):
            calls.append(("engine", Path(model_dir), steps))

    monkeypatch.setattr(omnivoice, "OmniVoiceEngine", Eng)
    monkeypatch.setattr(omnivoice, "make_voice",
                        lambda engine, name, seed, *a, **k: calls.append((name, seed)))
    events = []
    setup._default_voice(tmp_path, 7, lambda *a: events.append(a), None)
    assert calls == [("engine", tmp_path, 16), ("default", 7)]
    assert events == [("Creating the default voice", 0, None)]
    v = tmp_path / "voices"
    v.mkdir()
    (v / "default.npy").write_bytes(b"x")
    (v / "default.txt").write_text("x")
    setup._default_voice(tmp_path, 7, lambda *a: events.append(a), None)  # present: untouched
    assert len(calls) == 2


def test_add_encoders_to_existing_install_is_idempotent(world, monkeypatch):
    dest = world.tmp / "model"
    setup.install(dest, False)
    voices = dest / "voices"
    (voices / "mine.npy").write_bytes(b"mine")
    default_before = (voices / "default.npy").read_bytes()
    data_before = (dest / setup.LLM_DATA).stat().st_mtime_ns
    world.downloads.clear()
    world.voice_calls.clear()
    converted = []
    real_convert = setup.convert_llm
    monkeypatch.setattr(setup, "convert_llm", lambda *a, **k: converted.append(1) or real_convert(*a, **k))
    setup.install(dest, True)
    assert [u.rsplit("/", 1)[-1] for u in world.downloads] == ["e1.onnx"]  # only the encoder
    assert converted == [] and world.voice_calls == []
    assert (dest / setup.LLM_DATA).stat().st_mtime_ns == data_before
    assert (voices / "mine.npy").read_bytes() == b"mine"
    assert (voices / "default.npy").read_bytes() == default_before
    assert "encoders" in json.loads((dest / "manifest.json").read_text())
    assert not (dest / setup.CACHE_DIR).exists()
