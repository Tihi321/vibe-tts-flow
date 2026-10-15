"""Install OmniVoice from the official sources into models/omnivoice (no torch, no hosting).

What happens (each step is skipped when its result is already in place, so an interrupted install
resumes where it stopped):
  1. download the community ONNX parts (audio embeddings, audio heads, Higgs decoder, tokenizer)
     from Emerald7664/OmniVoice-Onnx and, for the language model, the upstream
     config.json + model.safetensors from k2-fsa/OmniVoice, both pinned to a commit;
  2. convert the bidirectional language model: the int4 ONNX graph ships with the app
     (vibetts/data/omnivoice/llm_bidir_int4.onnx, no weights) together with llm_weight_map.json,
     which says where every weight lives in the safetensors file and in the external-data file.
     The converter below reads the safetensors file with numpy, quantizes the projection weights
     to int4 (block 128, symmetric: the exact MatMulNBits layout and rounding of onnxruntime's
     MatMulNBitsQuantizer) and streams the result into llm_bidir_int4.onnx.data, one tensor at a
     time, so memory use stays small;
  3. optionally download the fp32 Higgs encoders (only needed to clone voices from recordings);
  4. write manifest.json, delete the 2.45 GB safetensors (unless keep_cache) and create the
     default voice preset.

Also a CLI: python -m vibetts.omnivoice_setup install|status|uninstall ...
The model weights are CC-BY-NC (non-commercial use only).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import model_setup
from .model_setup import DownloadCancelled, SetupError

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, "int | None"], None]

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
# they land in <dest>/ENCODERS_DIR
ENCODER_FILES = [
    ("audio_tokenizer/acoustic_encoder.onnx", 205546480),
    ("audio_tokenizer/semantic_encoder.onnx", 436736856),
    ("audio_tokenizer/quantizer_encoder.onnx", 12131293),
]
ENCODERS_DIR = "encoders"
UPSTREAM_FILES = [("config.json", 2238), ("model.safetensors", 2450344112)]
SAFETENSORS = "model.safetensors"

LLM_NAME = "llm_bidir_int4.onnx"
LLM_DATA = f"{LLM_NAME}.data"
MAP_NAME = "llm_weight_map.json"
CACHE_DIR = "_cache"
MANIFEST = "manifest.json"
LICENSE_NOTE = ("OmniVoice weights are CC-BY-NC (non-commercial use only); the OmniVoice code "
                "and the community ONNX export code are Apache-2.0.")
QUANT_BLOCK = 128

# Folder names used by scripts/build_omnivoice.py's old work dir (and the --from lookup)
_LOCAL_FOLDERS = {COMMUNITY_REPO: "cpu", UPSTREAM_REPO: "hf_upstream"}


def data_dir() -> Path:
    """Folder with the shipped graph and weight map (also inside the PyInstaller bundle)."""
    return Path(__file__).resolve().parent / "data" / "omnivoice"


def hf_url(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


class Item:
    """One file to fetch: where from, size, and where it ends up."""

    def __init__(self, repo: str, rev: str, path: str, size: int, target: str, group: str):
        self.repo, self.rev, self.path, self.size = repo, rev, path, size
        self.target = target  # path relative to the model dir (or to the cache dir: "cache:...")
        self.group = group  # "community" | "encoders" | "upstream"

    @property
    def url(self) -> str:
        return hf_url(self.repo, self.rev, self.path)

    @property
    def name(self) -> str:
        return Path(self.path).name


def plan(include_encoders: bool = True) -> list[Item]:
    """Every download of an install, in order."""
    items = [Item(COMMUNITY_REPO, COMMUNITY_REV, p, s, Path(p).name, "community")
             for p, s in COMMUNITY_FILES]
    items += [Item(UPSTREAM_REPO, UPSTREAM_REV, p, s, f"cache:{p}", "upstream")
              for p, s in UPSTREAM_FILES]
    if include_encoders:
        items += [Item(COMMUNITY_REPO, COMMUNITY_REV, p, s, f"{ENCODERS_DIR}/{Path(p).name}",
                       "encoders") for p, s in ENCODER_FILES]
    return items


def DOWNLOAD_BYTES(include_encoders: bool = True) -> int:  # noqa: N802 (name fixed by contract)
    return sum(i.size for i in plan(include_encoders))


# ---------------------------------------------------------------- safetensors reader (numpy)

_DTYPES = {"F32": (np.dtype("<f4"), 4), "F16": (np.dtype("<f2"), 2), "BF16": (None, 2),
           "F64": (np.dtype("<f8"), 8), "I64": (np.dtype("<i8"), 8), "I32": (np.dtype("<i4"), 4)}


class SafeTensors:
    """Read-only safetensors file: JSON header, then a numpy memory map (no torch)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            with open(self.path, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                if n > 100_000_000:
                    raise SetupError(f"{self.path.name}: not a safetensors file")
                header = json.loads(fh.read(n).decode("utf-8"))
        except (OSError, struct.error, ValueError) as e:
            raise SetupError(f"Could not read {self.path.name}: {e}") from e
        header.pop("__metadata__", None)
        self.header: dict[str, dict] = header
        self._base = 8 + n
        size = self.path.stat().st_size - self._base
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self._base,
                             shape=(size,)) if size > 0 else np.zeros(0, np.uint8)

    def keys(self) -> list[str]:
        return list(self.header)

    def shape(self, name: str) -> list[int]:
        return list(self.header[name]["shape"])

    def get(self, name: str) -> np.ndarray:
        """The tensor as a float32 (or its native integer) array; a view for F32."""
        try:
            info = self.header[name]
        except KeyError:
            raise SetupError(f"{self.path.name} has no tensor {name!r}") from None
        dt, width = _DTYPES.get(info["dtype"], (None, 0))
        if not width:
            raise SetupError(f"{name}: unsupported dtype {info['dtype']}")
        lo, hi = info["data_offsets"]
        count = int(np.prod(info["shape"], dtype=np.int64))
        if hi - lo != count * width or hi > len(self._mm):
            raise SetupError(f"{name}: tensor data is truncated or inconsistent")
        raw = self._mm[lo:hi]
        if info["dtype"] == "BF16":  # the top 16 bits of a float32
            arr = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            arr = raw.view(dt)
            if info["dtype"] in ("F16", "F64"):
                arr = arr.astype(np.float32)
        return arr.reshape(info["shape"])

    def close(self) -> None:
        mm = getattr(self._mm, "_mmap", None)
        self._mm = np.zeros(0, np.uint8)
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError, OSError):
                pass


# ---------------------------------------------------------------- int4 quantizer

def quantize_q4(w: np.ndarray, block: int = QUANT_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """float32 weight [N, K] (torch `out x in` layout) -> (B uint8 [N, K/block, block/2],
    scales float32 [N, K/block]) exactly like onnxruntime's MatMulNBitsQuantizer, 4 bits,
    symmetric: per block of `block` values along K the scale is amax / -8 where amax is the
    signed value of largest magnitude (first one on ties); q = clip(round(v / scale + 8), 0, 15)
    with halves rounded away from zero; two values per byte, the first in the low nibble."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % block:
        raise SetupError(f"cannot quantize a tensor of shape {w.shape} in blocks of {block}")
    n, k = w.shape
    blk = w.reshape(n, k // block, block)
    amax = np.take_along_axis(blk, np.abs(blk).argmax(-1)[..., None], -1)[..., 0]
    scale = (amax / np.float32(-8)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        recip = np.where(scale != 0, np.float32(1) / scale, np.float32(0)).astype(np.float32)
    q = np.floor(blk * recip[..., None] + np.float32(8.5))  # == roundf(v * recip + 8), v*recip+8 >= 0
    q = np.clip(q, 0, 15).astype(np.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    return packed, scale


# ---------------------------------------------------------------- weight map and converter

def load_weight_map(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else data_dir() / MAP_NAME
    try:
        wm = json.loads(p.read_text(encoding="utf-8"))
        if wm["format"] != 1:
            raise SetupError(f"{p.name}: unsupported format {wm['format']}")
        wm["tensors"]
        wm["data_size"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SetupError(f"Could not read the weight map {p}: {e}") from e
    return wm


def _out_specs(entry: dict) -> list[dict]:
    return [entry["B"], entry["scales"]] if entry["kind"] == "q4" else [entry]


def convert_llm(safetensors: Path, weight_map: dict[str, Any], out_path: Path,
                progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> None:
    """Write the external-data file of the int4 LLM graph at out_path, tensor by tensor."""
    st = SafeTensors(safetensors)
    entries = sorted(weight_map["tensors"], key=lambda e: _out_specs(e)[0]["offset"])
    total = len(entries)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "wb") as fh:
            fh.truncate(weight_map["data_size"])
            for i, e in enumerate(entries):
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled()
                if progress:
                    progress(f"Converting {i} / {total}", i, total)
                arr = st.get(e["tensor"])
                if list(arr.shape) != e["shape"]:
                    raise SetupError(f"{e['tensor']}: shape {list(arr.shape)} differs from the "
                                     f"expected {e['shape']}")
                if e["kind"] == "q4":
                    packed, scales = quantize_q4(arr, weight_map["quant"]["block_size"])
                    parts = [(e["B"], packed), (e["scales"], scales)]
                elif e["kind"] == "copy":
                    parts = [(e, np.ascontiguousarray(arr.astype(e["dtype"], copy=False)))]
                else:
                    raise SetupError(f"unknown weight map entry kind {e['kind']!r}")
                for spec, data in parts:
                    buf = data.astype(data.dtype.newbyteorder("<"), copy=False).tobytes()
                    if len(buf) != spec["length"]:
                        raise SetupError(f"{spec['name']}: produced {len(buf)} bytes, "
                                         f"expected {spec['length']}")
                    fh.seek(spec["offset"])
                    fh.write(buf)
                del arr, parts
            if progress:
                progress(f"Converting {total} / {total}", total, total)
    finally:
        st.close()


# ---------------------------------------------------------------- install

def _fmt_mb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1_000_000_000 else f"{n / 1e6:.0f} MB"


def _has(path: Path, size: int | None = None) -> bool:
    try:
        return path.is_file() and (size is None or path.stat().st_size == size)
    except OSError:
        return False


def _find_local(item: Item, dirs: list[Path]) -> Path | None:
    """A file of the plan that is already somewhere on disk (--from), size-checked."""
    folder = _LOCAL_FOLDERS[item.repo]
    for d in dirs:
        for rel in (Path(folder) / item.path, Path(item.path), Path(item.name)):
            p = d / rel
            if _has(p, item.size):
                return p
    return None


def _target_path(item: Item, model_dir: Path, cache_dir: Path) -> Path:
    if item.target.startswith("cache:"):
        return cache_dir / item.target[len("cache:"):]
    return model_dir / item.target


def _link_or_copy(src: Path, dst: Path) -> None:
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(src, part)
        os.replace(part, dst)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise


def _manifest(model_dir: Path, encoders: bool) -> dict[str, Any]:
    names = [Path(p).name for p, _ in COMMUNITY_FILES] + [LLM_NAME, LLM_DATA]
    files = {n: (model_dir / n).stat().st_size for n in names}
    manifest: dict[str, Any] = {
        "files": files,
        "sources": {
            "community_onnx": {"repo": COMMUNITY_REPO, "commit": COMMUNITY_REV,
                               "parts": [p for p, _ in COMMUNITY_FILES]},
            "llm": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_REV,
                    "note": "bidirectional export, int4 MatMulNBits block 128 symmetric"},
        },
        "license": LICENSE_NOTE,
        "built": time.strftime("%Y-%m-%d"),
    }
    if encoders:
        manifest["encoders"] = {
            "dir": ENCODERS_DIR,
            "files": {Path(p).name: s for p, s in ENCODER_FILES},
            "source": {"repo": COMMUNITY_REPO, "commit": COMMUNITY_REV,
                       "parts": [p for p, _ in ENCODER_FILES]},
        }
    return manifest


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def install(model_dir: Path, include_encoders: bool = True,
            progress: ProgressFn | None = None, cancel: threading.Event | None = None,
            keep_cache: bool = False, *, cache_dir: Path | None = None,
            local_dirs: list[Path] | tuple[Path, ...] = (), seed: int = 0,
            make_default_voice: bool = True) -> None:
    """Download, convert and set up OmniVoice in model_dir (see the module docstring).

    progress(phase_text, done, total): "Downloading <file> a / b MB" with overall byte counts,
    "Converting n / m", "Creating the default voice". Raises model_setup.DownloadCancelled when
    `cancel` is set, model_setup.SetupError on any other failure. `local_dirs` lists folders
    searched for already-downloaded files (nothing is fetched for those); `cache_dir` holds the
    upstream safetensors during the install (default <model_dir>/_cache, removed afterwards
    unless keep_cache)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else model_dir / CACHE_DIR
    local = [Path(d) for d in local_dirs]
    emit = progress or (lambda *_: None)

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise DownloadCancelled()

    wm = load_weight_map()
    llm_data = model_dir / LLM_DATA
    llm_done = _has(llm_data, wm["data_size"])
    items = plan(include_encoders)
    # what still has to be fetched or copied; the safetensors is not needed once the LLM is converted
    todo = []
    for it in items:
        if it.group == "upstream" and llm_done:
            continue
        if not _has(_target_path(it, model_dir, cache_dir), it.size):
            todo.append(it)
    grand = sum(i.size for i in todo)
    finished = 0
    resolved: dict[str, Path] = {}  # upstream files, where they can be read from
    for it in todo:
        check_cancel()
        target = _target_path(it, model_dir, cache_dir)
        src = _find_local(it, local)
        if src is not None and it.group == "upstream":
            resolved[it.name] = src  # read in place, never copied or deleted
            finished += it.size
            continue
        if src is not None:
            emit(f"Copying {it.name}", finished, grand)
            _link_or_copy(src, target)
            finished += it.size
            continue

        def on_bytes(done: int, total: int | None, it=it, base=finished) -> None:
            emit(f"Downloading {it.name} {_fmt_mb(done)} / {_fmt_mb(it.size)}", base + done, grand)

        target.parent.mkdir(parents=True, exist_ok=True)
        model_setup.download(target.name, target.parent, on_bytes, cancel, it.size, url=it.url)
        finished += it.size
    check_cancel()

    if not llm_done:
        safet = resolved.get(SAFETENSORS) or cache_dir / SAFETENSORS
        part = llm_data.with_name(llm_data.name + ".part")
        try:
            convert_llm(safet, wm, part, emit, cancel)
            os.replace(part, llm_data)
        except BaseException:
            try:
                part.unlink()
            except OSError:
                pass
            raise
    graph = data_dir() / LLM_NAME
    if not _has(graph):
        raise SetupError(f"The bundled LLM graph is missing: {graph}")
    if not _has(model_dir / LLM_NAME, graph.stat().st_size):
        shutil.copyfile(graph, model_dir / LLM_NAME)
    check_cancel()

    encoders_ok = all(_has(model_dir / ENCODERS_DIR / Path(p).name, s) for p, s in ENCODER_FILES)
    _write_json(model_dir / MANIFEST, _manifest(model_dir, encoders_ok))
    if not keep_cache:
        gc.collect()  # release any memory map of the safetensors before deleting it
        shutil.rmtree(cache_dir, ignore_errors=True)
    if make_default_voice:
        _default_voice(model_dir, seed, emit, cancel)


def _default_voice(model_dir: Path, seed: int, emit: ProgressFn,
                   cancel: threading.Event | None) -> None:
    voices = model_dir / "voices"
    if _has(voices / "default.npy") and _has(voices / "default.txt"):
        return
    if cancel is not None and cancel.is_set():
        raise DownloadCancelled()
    from .omnivoice import OmniVoiceEngine, make_voice

    emit("Creating the default voice", 0, None)
    try:
        engine = OmniVoiceEngine(model_dir, steps=16)
        make_voice(engine, "default", seed)
    except Exception as e:  # onnxruntime / tokenizer problems: report as a setup failure
        raise SetupError(f"Could not create the default voice: {e}") from e
    finally:
        engine = None
        gc.collect()


# ---------------------------------------------------------------- status / uninstall

def status(model_dir: Path) -> dict[str, Any]:
    from .omnivoice import encoders_present, models_present

    model_dir = Path(model_dir)
    total = 0
    if model_dir.is_dir():
        for root, _dirs, files in os.walk(model_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return {"installed": models_present(model_dir), "encoders": encoders_present(model_dir),
            "bytes_on_disk": total}


def uninstall(model_dir: Path, keep_voices: bool = True) -> None:
    """Remove the model files (and voices unless keep_voices). Only known files are deleted."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return
    names = {Path(p).name for p, _ in COMMUNITY_FILES} | {LLM_NAME, LLM_DATA, MANIFEST}
    try:
        names |= set(json.loads((model_dir / MANIFEST).read_text(encoding="utf-8"))["files"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        for n in names:
            for suffix in ("", ".part"):
                (model_dir / f"{n}{suffix}").unlink(missing_ok=True)
        shutil.rmtree(model_dir / ENCODERS_DIR, ignore_errors=True)
        shutil.rmtree(model_dir / CACHE_DIR, ignore_errors=True)
        if not keep_voices:
            shutil.rmtree(model_dir / "voices", ignore_errors=True)
        try:
            model_dir.rmdir()  # only succeeds when nothing (e.g. voices) is left
        except OSError:
            pass
    except OSError as e:
        raise SetupError(f"Could not remove OmniVoice files: {e}") from e


# ---------------------------------------------------------------- CLI

def _cli_progress() -> ProgressFn:
    state = {"text": "", "t": 0.0}

    def fn(text: str, done: int, total: int | None) -> None:
        head = text.split(" ")[0]
        now = time.monotonic()
        if head != state["text"] or now - state["t"] > 5 or (total and done >= total):
            state["text"], state["t"] = head, now
            pct = f" ({done * 100 // total}%)" if total else ""
            print(f"  {text}{pct}", flush=True)
    return fn


def main(argv: list[str] | None = None) -> int:
    default_dir = model_setup.ROOT / "models" / "omnivoice"
    ap = argparse.ArgumentParser(prog="python -m vibetts.omnivoice_setup",
                                 description="Install OmniVoice (download + convert, no torch).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("install", help="download, convert and set up OmniVoice")
    ins.add_argument("--dest", type=Path, default=default_dir)
    ins.add_argument("--no-encoders", action="store_true",
                     help="skip the fp32 encoders (654 MB, only needed to clone voices from WAVs)")
    ins.add_argument("--keep-cache", action="store_true", help="keep the downloaded safetensors")
    ins.add_argument("--cache", type=Path, default=None,
                     help="where the safetensors is downloaded (default <dest>/_cache)")
    ins.add_argument("--from", dest="local", type=Path, action="append", default=[],
                     help="folder with already-downloaded files to reuse (repeatable)")
    ins.add_argument("--seed", type=int, default=0, help="seed of the default voice preset")
    ins.add_argument("--no-voice", action="store_true", help="skip creating the default preset")
    st = sub.add_parser("status", help="print what is installed")
    st.add_argument("--dest", type=Path, default=default_dir)
    un = sub.add_parser("uninstall", help="remove the model files")
    un.add_argument("--dest", type=Path, default=default_dir)
    un.add_argument("--delete-voices", action="store_true")
    a = ap.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        if a.cmd == "install":
            t0 = time.perf_counter()
            enc = not a.no_encoders
            print(f"Installing OmniVoice into {a.dest} (download {DOWNLOAD_BYTES(enc) / 1e9:.2f} GB "
                  f"without local files)", flush=True)
            install(a.dest, enc, _cli_progress(), None, a.keep_cache, cache_dir=a.cache,
                    local_dirs=a.local, seed=a.seed, make_default_voice=not a.no_voice)
            print(f"done in {time.perf_counter() - t0:.0f}s: {status(a.dest)}")
        elif a.cmd == "status":
            print(json.dumps(status(a.dest)))
        else:
            uninstall(a.dest, keep_voices=not a.delete_voices)
            print("removed")
    except (SetupError, DownloadCancelled) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
