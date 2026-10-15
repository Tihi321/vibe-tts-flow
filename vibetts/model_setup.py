"""Download the Kokoro model and voice files (also a CLI: python -m vibetts.model_setup)."""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Callable

import requests

from vibetts.config import ROOT

BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

# name -> (file name, size in bytes)
MODELS: dict[str, tuple[str, int]] = {
    "int8": ("kokoro-v1.0.int8.onnx", 92_361_271),   # same speed and nearly the same sound
    "fp32": ("kokoro-v1.0.onnx", 325_532_387),
}
VOICES: tuple[str, int] = ("voices-v1.0.bin", 28_214_398)
DEFAULT_MODEL = "int8"
USER_AGENT = "vibetts-setup"
_CHUNK = 256 * 1024

ProgressFn = Callable[[int, "int | None"], None]


class DownloadCancelled(Exception):
    pass


class SetupError(Exception):
    pass


def files() -> dict[str, tuple[str, int]]:
    """Every downloadable file: model names plus "voices"."""
    return {**MODELS, "voices": VOICES}


def file_url(filename: str) -> str:
    return BASE_URL + filename


def is_installed(path: Path, expected_size: int | None = None) -> bool:
    try:
        size = Path(path).stat().st_size
    except OSError:
        return False
    return size == expected_size if expected_size else size > 1024 * 1024


def _stream_to(url: str, target: Path, progress: ProgressFn | None,
               cancel: threading.Event | None) -> None:
    try:
        resp = requests.get(url, stream=True, timeout=30, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as e:
        raise SetupError(f"Could not connect to {url}: {e}") from e
    try:
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            raise SetupError(f"Download failed ({url}): {e}") from e
        length = resp.headers.get("Content-Length")
        total = int(length) if length and str(length).isdigit() else None
        done = 0
        if progress:
            progress(0, total)
        try:
            with open(target, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if cancel is not None and cancel.is_set():
                        raise DownloadCancelled()
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        except requests.RequestException as e:
            raise SetupError(f"Download interrupted ({url}): {e}") from e
        except OSError as e:
            raise SetupError(f"Could not write {target}: {e}") from e
    finally:
        close = getattr(resp, "close", None)
        if close:
            close()


def download(filename: str, dest_dir: Path, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None, expected_size: int | None = None,
             url: str | None = None) -> Path:
    """Fetch `url` (default: the Kokoro release URL of `filename`) to dest_dir/filename."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / filename
    if is_installed(final, expected_size):
        return final
    part = final.with_name(final.name + ".part")
    try:
        _stream_to(url or file_url(filename), part, progress, cancel)
        if expected_size and part.stat().st_size != expected_size:
            raise SetupError(f"{filename}: got {part.stat().st_size} bytes, "
                             f"expected {expected_size}")
        os.replace(part, final)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return final


def download_named(name: str, dest_dir: Path, progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None) -> Path:
    table = files()
    if name not in table:
        raise ValueError(f"Unknown file '{name}'. Valid: {', '.join(table)}")
    filename, size = table[name]
    return download(filename, dest_dir, progress, cancel, size)


# ---------------------------------------------------------------- CLI

def _cli_progress(label: str) -> ProgressFn:
    state = {"last": -1}

    def fn(done: int, total: int | None) -> None:
        if total:
            pct = int(done * 100 / total)
            if pct != state["last"] and pct % 10 == 0:
                state["last"] = pct
                print(f"  {label}: {pct}% ({done // 1_000_000} / {total // 1_000_000} MB)",
                      flush=True)
    return fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vibetts.model_setup",
                                 description="Download the Kokoro model and voices.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--dest", default=None, help="target folder (default: <app>/models)")
    args = ap.parse_args(argv)
    dest = Path(args.dest) if args.dest else ROOT / "models"
    try:
        for name in (args.model, "voices"):
            print(f"{name}...", flush=True)
            print(f"  {download_named(name, dest, _cli_progress(name))}", flush=True)
    except (SetupError, DownloadCancelled) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
