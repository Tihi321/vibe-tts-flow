"""Build the local OmniVoice model files into models/omnivoice/ (dev-only, needs torch).

Steps, each skipped when its result already exists, so the script can be re-run:
  1. download the community ONNX parts (audio embeddings, audio heads, Higgs decoder,
     tokenizer) from Emerald7664/OmniVoice-Onnx and the upstream weights from k2-fsa/OmniVoice,
     both pinned to a commit;
  2. export the bidirectional LLM from the upstream weights and quantize it to int4 (the
     community LLM is causal, but OmniVoice is bidirectional, so it produces noise);
  3. copy the runtime files into the destination and write manifest.json;
  4. download the fp32 Higgs encoders (needed only to clone voices from recordings; skip with
     --no-encoders) into <dest>/encoders and list them in the manifest's "encoders" section;
  5. create the default voice preset.

Run in a venv with scripts/omnivoice-build-requirements.txt installed:
  python scripts/build_omnivoice.py [--dest models/omnivoice] [--cache DIR] [--seed N]
                                    [--no-encoders]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COMMUNITY_REPO = "Emerald7664/OmniVoice-Onnx"
COMMUNITY_REV = "f312525229ea4c4d9101c2b3e317ac77d995ec59"
UPSTREAM_REPO = "k2-fsa/OmniVoice"
UPSTREAM_REV = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"

# (path in the community repo, size in bytes); lands flat in the destination
COMMUNITY_FILES = [
    ("audio_embeddings_encoder.onnx", 2172),
    ("audio_embeddings_encoder.onnx.data", 327426048),
    ("audio_heads_decoder.onnx", 16795584),
    ("tokenizer.json", 11423986),
    ("audio_tokenizer/fp16/higgs_decoder.onnx", 308494),
    ("audio_tokenizer/fp16/higgs_decoder.onnx.data", 43100160),
]
# fp32 encoders for voice cloning (the fp16 semantic encoder does not load on onnxruntime CPU);
# (path in the community repo, size in bytes); they land in <dest>/ENCODERS_DIR
ENCODER_FILES = [
    ("audio_tokenizer/acoustic_encoder.onnx", 205546480),
    ("audio_tokenizer/semantic_encoder.onnx", 436736856),
    ("audio_tokenizer/quantizer_encoder.onnx", 12131293),
]
ENCODERS_DIR = "encoders"
UPSTREAM_FILES = [("config.json", 2238), ("model.safetensors", 2450344112)]

LLM_NAME = "llm_bidir_int4.onnx"
LICENSE_NOTE = ("OmniVoice weights are CC-BY-NC (non-commercial use only); the OmniVoice code "
                "and the community ONNX export code are Apache-2.0.")


def say(msg: str) -> None:
    print(f"[build_omnivoice] {msg}", flush=True)


def has(path: Path, size: int | None = None) -> bool:
    return path.is_file() and (size is None or path.stat().st_size == size)


def download(cache: Path, groups=None) -> None:
    from huggingface_hub import hf_hub_download

    for repo, rev, folder, files in groups or (
            (COMMUNITY_REPO, COMMUNITY_REV, "cpu", COMMUNITY_FILES),
            (UPSTREAM_REPO, UPSTREAM_REV, "hf_upstream", UPSTREAM_FILES)):
        for name, size in files:
            target = cache / folder / name
            if has(target, size):
                say(f"have {folder}/{name}")
                continue
            say(f"downloading {repo}/{name} ({size / 1e6:.0f} MB)")
            hf_hub_download(repo, name, revision=rev, local_dir=str(cache / folder))
            if not has(target, size):
                raise SystemExit(f"{target} has an unexpected size; delete it and retry")


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


def install(cache: Path, llm: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    sources = [(cache / "cpu" / n, Path(n).name) for n, _ in COMMUNITY_FILES]
    sources += [(llm, llm.name), (Path(f"{llm}.data"), f"{llm.name}.data")]
    for src, name in sources:
        target = dest / name
        if has(target, src.stat().st_size):
            continue
        say(f"copying {name}")
        shutil.copy2(src, target)
    files = {name: (dest / name).stat().st_size for _, name in sources}
    manifest = {
        "files": files,
        "sources": {
            "community_onnx": {"repo": COMMUNITY_REPO, "commit": COMMUNITY_REV,
                               "parts": [n for n, _ in COMMUNITY_FILES]},
            "llm": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_REV,
                    "note": "bidirectional export, int4 MatMulNBits block 128 symmetric"},
        },
        "license": LICENSE_NOTE,
        "built": time.strftime("%Y-%m-%d"),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    say(f"wrote manifest.json ({sum(files.values()) / 1e6:.0f} MB in {len(files)} files)")


def install_encoders(cache: Path, dest: Path) -> None:
    """Download the fp32 encoders (pinned, size-checked), copy them and update the manifest."""
    download(cache, [(COMMUNITY_REPO, COMMUNITY_REV, "cpu", ENCODER_FILES)])
    enc_dir = dest / ENCODERS_DIR
    enc_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, size in ENCODER_FILES:
        target = enc_dir / Path(name).name
        if not has(target, size):
            say(f"copying {ENCODERS_DIR}/{target.name}")
            shutil.copy2(cache / "cpu" / name, target)
        sizes[target.name] = size
    mpath = dest / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    manifest["encoders"] = {"dir": ENCODERS_DIR, "files": sizes,
                            "source": {"repo": COMMUNITY_REPO, "commit": COMMUNITY_REV,
                                       "parts": [n for n, _ in ENCODER_FILES]}}
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    say(f"encoders installed ({sum(sizes.values()) / 1e6:.0f} MB)")


def default_voice(dest: Path, seed: int) -> None:
    if has(dest / "voices" / "default.npy") and has(dest / "voices" / "default.txt"):
        say("have voice preset 'default'")
        return
    from vibetts.omnivoice import OmniVoiceEngine, make_voice

    say(f"creating voice preset 'default' (seed {seed})")
    engine = OmniVoiceEngine(dest, steps=16)
    wav = make_voice(engine, "default", seed)
    say(f"saved {wav}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=ROOT / "models" / "omnivoice")
    ap.add_argument("--cache", type=Path, default=ROOT / "models" / "omnivoice-build",
                    help="download/export work dir (cpu/, hf_upstream/, bidir/); reused on re-runs")
    ap.add_argument("--seed", type=int, default=0, help="seed of the default voice preset")
    ap.add_argument("--no-encoders", action="store_true",
                    help="skip the fp32 encoders (650 MB, only needed to clone voices from WAVs)")
    ap.add_argument("--no-voice", action="store_true", help="skip creating the default preset")
    a = ap.parse_args()

    manifest_ok = (a.dest / "manifest.json").is_file()
    llm = a.cache / "bidir" / LLM_NAME
    if manifest_ok and all(has(a.dest / n) for n in json.loads(
            (a.dest / "manifest.json").read_text(encoding="utf-8"))["files"]):
        say("model files already installed")
    else:
        download(a.cache)
        llm = export_llm(a.cache / "hf_upstream", a.cache / "bidir")
        install(a.cache, llm, a.dest)
    if not a.no_encoders:
        install_encoders(a.cache, a.dest)
    if not a.no_voice:
        default_voice(a.dest, a.seed)
    say(f"done: {a.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
