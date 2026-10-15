"""Install the local OmniVoice model files into models/omnivoice/.

Normal use (no torch needed) simply runs vibetts.omnivoice_setup.install, the same code as the
Install button in Settings: it downloads the community ONNX parts and the upstream weights
(pinned commits), converts the language model to int4 with numpy and creates the default voice.

  python scripts/build_omnivoice.py [--dest models/omnivoice] [--cache DIR] [--seed N]
                                    [--no-encoders] [--keep-cache] [--from DIR]

Developer path, only needed when the model graph changes (needs torch, onnx, transformers:
scripts/omnivoice-build-requirements.txt): exports the bidirectional LLM with torch, quantizes it
with onnxruntime's MatMulNBitsQuantizer and regenerates the files that ship with the app in
vibetts/data/omnivoice/ (llm_bidir_int4.onnx without weights + llm_weight_map.json), after
checking that the numpy converter reproduces the quantized weights byte for byte:

  python scripts/build_omnivoice.py --export-graph [--cache DIR]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibetts import omnivoice_setup as setup  # noqa: E402
from vibetts.omnivoice_setup import (  # noqa: E402,F401  (re-exported for old callers)
    COMMUNITY_FILES, COMMUNITY_REPO, COMMUNITY_REV, ENCODER_FILES, ENCODERS_DIR, LLM_NAME,
    UPSTREAM_FILES, UPSTREAM_REPO, UPSTREAM_REV)

DATA_DIR = ROOT / "vibetts" / "data" / "omnivoice"


def say(msg: str) -> None:
    print(f"[build_omnivoice] {msg}", flush=True)


def has(path: Path, size: int | None = None) -> bool:
    return path.is_file() and (size is None or path.stat().st_size == size)


def export_llm(upstream: Path, out_dir: Path) -> Path:
    """Bidirectional Qwen3 (inputs_embeds -> hidden_states) from the upstream weights, int4."""
    final = out_dir / LLM_NAME
    if has(final) and has(Path(f"{final}.data")):
        say(f"have {final.name}")
        return final
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from safetensors import safe_open
    from transformers.models.qwen3 import Qwen3Config, Qwen3Model

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((upstream / "config.json").read_text(encoding="utf-8"))
    llm_cfg = Qwen3Config(**{k: v for k, v in cfg["llm_config"].items() if k != "architectures"})
    llm_cfg._attn_implementation = "sdpa"
    llm = Qwen3Model(llm_cfg).eval()
    state = {}
    with safe_open(str(upstream / "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("llm."):
                state[k[4:]] = f.get_tensor(k)
    res = llm.load_state_dict(state, strict=False)
    if res.missing_keys:
        raise SystemExit(f"LLM weights missing: {res.missing_keys[:5]}")

    class Bidirectional(torch.nn.Module):
        """All-True 4D mask, like upstream: every position attends to every other."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inputs_embeds):
            b, s, _ = inputs_embeds.shape
            mask = torch.ones(b, 1, s, s, dtype=torch.bool)
            return self.m(inputs_embeds=inputs_embeds, attention_mask=mask,
                          use_cache=False).last_hidden_state

    wrap = Bidirectional(llm).eval()
    x = torch.randn(1, 50, 1024)
    fp32 = out_dir / "llm_bidir_fp32.onnx"
    say("exporting the bidirectional LLM (fp32)")
    with torch.no_grad():
        ref = wrap(x).numpy()
        torch.onnx.export(wrap, (x,), str(fp32), input_names=["inputs_embeds"],
                          output_names=["hidden_states"],
                          dynamic_axes={"inputs_embeds": {0: "B", 1: "S"},
                                        "hidden_states": {0: "B", 1: "S"}},
                          opset_version=20, dynamo=False)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(fp32), sess_options=opts, providers=["CPUExecutionProvider"])
    diff = float(np.abs(sess.run(None, {"inputs_embeds": x.numpy()})[0] - ref).max())
    say(f"onnx vs torch max abs diff {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("exported LLM does not match torch")
    del sess

    say("quantizing to int4 (block 128, symmetric)")
    logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)  # one line per node
    quant = MatMulNBitsQuantizer(onnx.load(str(fp32)), block_size=128, is_symmetric=True, bits=4)
    quant.process()
    quant.model.save_model_to_file(str(final), True)
    fp32.unlink()
    Path(f"{fp32}.data").unlink(missing_ok=True)
    say(f"wrote {final.name} ({final.stat().st_size + Path(f'{final}.data').stat().st_size:,} B)")
    return final


def ensure_upstream(upstream: Path, local: list[Path]) -> None:
    """config.json + model.safetensors of the pinned upstream commit in `upstream`."""
    from vibetts import model_setup

    for it in [i for i in setup.plan(False) if i.group == "upstream"]:
        target = upstream / it.name
        if has(target, it.size):
            continue
        src = setup._find_local(it, local)
        if src is not None:
            say(f"using {src}")
            upstream.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            continue
        say(f"downloading {it.url} ({it.size / 1e6:.0f} MB)")
        model_setup.download(it.name, upstream, None, None, it.size, url=it.url)


def derive_weight_map(onnx_path: Path, safetensors: Path, out_path: Path) -> dict:
    """Map every initializer of the quantized graph to its safetensors tensor and prove, by
    comparing bytes with the quantized export, that the numpy converter reproduces it."""
    import numpy as np
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    data_file = onnx_path.with_name(onnx_path.name + ".data")
    data = np.memmap(data_file, dtype=np.uint8, mode="r")
    st = setup.SafeTensors(safetensors)

    def ext(t):
        d = {e.key: e.value for e in t.external_data}
        if d.get("location") != data_file.name:
            raise SystemExit(f"{t.name}: unexpected external data location {d}")
        return int(d["offset"]), int(d["length"])

    def dtype_of(t) -> str:
        return {1: "float32", 2: "uint8"}[t.data_type]

    entries, used, embedded = [], set(), []
    for node in graph.node:
        if node.op_type != "MatMulNBits":
            continue
        a = {x.name: x.i for x in node.attribute}
        if a.get("bits") != 4 or a.get("block_size") != setup.QUANT_BLOCK or len(node.input) != 3:
            raise SystemExit(f"{node.name}: unexpected MatMulNBits attributes")
        m = re.fullmatch(r"/m/(layers\.\d+)/(\w+)/(\w+)/MatMul_Q4", node.name)
        if not m:
            raise SystemExit(f"cannot derive the weight of node {node.name}")
        tensor = f"llm.{m[1]}.{m[2]}.{m[3]}.weight"
        b, sc = inits[node.input[1]], inits[node.input[2]]
        w = st.get(tensor)
        packed, scales = setup.quantize_q4(w)
        (bo, bl), (so, sl) = ext(b), ext(sc)
        if (bytes(data[bo:bo + bl]) != packed.tobytes()
                or bytes(data[so:so + sl]) != scales.astype("<f4").tobytes()):
            raise SystemExit(f"{node.name}: numpy quantizer output differs from the export")
        entries.append({"kind": "q4", "tensor": tensor, "shape": list(w.shape),
                        "B": {"name": b.name, "offset": bo, "length": bl, "dtype": "uint8",
                              "shape": list(b.dims)},
                        "scales": {"name": sc.name, "offset": so, "length": sl,
                                   "dtype": "float32", "shape": list(sc.dims)}})
        used |= {b.name, sc.name}
    for t in graph.initializer:
        if t.name in used:
            continue
        m = re.fullmatch(r"m\.((?:layers\.\d+\.)?[\w.]+\.weight)", t.name)
        if not m:
            raise SystemExit(f"cannot derive initializer {t.name}")
        tensor = f"llm.{m[1]}"
        w = st.get(tensor)
        want = w.astype(np.dtype(dtype_of(t)).newbyteorder("<")).tobytes()
        if t.external_data:
            off, ln = ext(t)
            if bytes(data[off:off + ln]) != want or list(w.shape) != list(t.dims):
                raise SystemExit(f"{t.name}: differs from {tensor}")
            entries.append({"kind": "copy", "tensor": tensor, "name": t.name, "offset": off,
                            "length": ln, "dtype": dtype_of(t), "shape": list(t.dims)})
        else:  # small tensor embedded in the graph file
            if t.raw_data != want:
                raise SystemExit(f"{t.name}: embedded value differs from {tensor}")
            embedded.append({"name": t.name, "tensor": tensor})
    covered = used | {e["name"] for e in entries if "name" in e} | {e["name"] for e in embedded}
    leftovers = [t.name for t in graph.initializer if t.name not in covered]
    if leftovers:
        raise SystemExit(f"initializers without a source: {leftovers[:5]}")
    st.close()
    spans = sorted((s["offset"], s["length"]) for e in entries for s in setup._out_specs(e))
    for (o1, l1), (o2, _l2) in zip(spans, spans[1:]):
        if o1 + l1 > o2:
            raise SystemExit("overlapping external-data ranges")
    wm = {
        "format": 1,
        "about": ("Source of every external initializer of llm_bidir_int4.onnx. kind 'copy': "
                  "the safetensors tensor (cast to dtype) is written at offset. kind 'q4': the "
                  "safetensors weight [N, K] (torch out x in, no transpose needed) is quantized "
                  "to int4, blocks of 128 along K, symmetric, onnxruntime MatMulNBits layout: "
                  "B uint8 [N, K/128, 64] (low nibble first, zero point 8) at B.offset and "
                  "scales float32 [N, K/128] at scales.offset. 'embedded' initializers are "
                  "small tensors stored inside the graph file; the rotary tables and all other "
                  "constants are graph nodes, computed from the config at export time."),
        "data_file": data_file.name,
        "data_size": data_file.stat().st_size,
        "source": {"repo": setup.UPSTREAM_REPO, "commit": setup.UPSTREAM_REV,
                   "file": setup.SAFETENSORS},
        "quant": {"bits": 4, "block_size": setup.QUANT_BLOCK, "symmetric": True},
        "tensors": entries,
        "embedded": embedded,
    }
    out_path.write_text(json.dumps(wm, separators=(",", ":")) + "\n", encoding="utf-8")
    say(f"wrote {out_path.name}: {len(entries)} tensors "
        f"({sum(e['kind'] == 'q4' for e in entries)} quantized), {len(embedded)} embedded, "
        f"{out_path.stat().st_size / 1e3:.0f} kB")
    return wm


def export_graph(cache: Path, local: list[Path]) -> None:
    upstream = cache / "hf_upstream"
    ensure_upstream(upstream, local)
    llm = export_llm(upstream, cache / "bidir")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(llm, DATA_DIR / LLM_NAME)  # graph only: weights stay in the .data file
    say(f"graph-only copy: {DATA_DIR / LLM_NAME} ({(DATA_DIR / LLM_NAME).stat().st_size:,} B)")
    derive_weight_map(llm, upstream / setup.SAFETENSORS, DATA_DIR / setup.MAP_NAME)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=ROOT / "models" / "omnivoice")
    ap.add_argument("--cache", type=Path, default=None,
                    help="install: where the safetensors is downloaded (default <dest>/_cache); "
                         "--export-graph: work dir (hf_upstream/, bidir/), default "
                         "models/omnivoice-build")
    ap.add_argument("--from", dest="local", type=Path, action="append", default=[],
                    help="folder with already-downloaded files to reuse (repeatable)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the default voice preset")
    ap.add_argument("--no-encoders", action="store_true",
                    help="skip the fp32 encoders (654 MB, only needed to clone voices from WAVs)")
    ap.add_argument("--keep-cache", action="store_true", help="keep the downloaded safetensors")
    ap.add_argument("--no-voice", action="store_true", help="skip creating the default preset")
    ap.add_argument("--export-graph", action="store_true",
                    help="developer path (torch): regenerate vibetts/data/omnivoice/*")
    a = ap.parse_args()

    if a.export_graph:
        export_graph(a.cache or ROOT / "models" / "omnivoice-build", a.local)
        return 0
    argv = ["install", "--dest", str(a.dest), "--seed", str(a.seed)]
    argv += ["--no-encoders"] if a.no_encoders else []
    argv += ["--keep-cache"] if a.keep_cache else []
    argv += ["--no-voice"] if a.no_voice else []
    argv += ["--cache", str(a.cache)] if a.cache else []
    for d in a.local:
        argv += ["--from", str(d)]
    return setup.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
