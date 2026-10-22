"""Download the Kokoro, Supertonic and Higgs model files (also a CLI: python -m vibetts.model_setup)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Callable

import requests

from vibetts.config import ROOT
from vibetts import higgs as _higgs
from vibetts.supertonic import MODEL_FILES, REPO, REVISION

BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

# name -> (file name, size in bytes)
MODELS: dict[str, tuple[str, int]] = {
    "int8": ("kokoro-v1.0.int8.onnx", 92_361_271),   # same speed and nearly the same sound
    "fp32": ("kokoro-v1.0.onnx", 325_532_387),
}
VOICES: tuple[str, int] = ("voices-v1.0.bin", 28_214_398)
DEFAULT_MODEL = "int8"
SUPERTONIC_BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
HIGGS_MODEL_FILES = _higgs.MODEL_FILES
HIGGS_BASE_URL = f"https://huggingface.co/{_higgs.REPO}/resolve/{_higgs.REVISION}/"
DISK_MARGIN = 64 * 1024 * 1024  # room for .part files, metadata and the voices
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


# ---------------------------------------------------------------- Supertonic

def _missing(dest_dir: Path, table: tuple[tuple[str, int], ...]) -> list[str]:
    return [rel for rel, size in table if not is_installed(Path(dest_dir) / rel, size)]


def _download_set(dest_dir: Path, table: tuple[tuple[str, int], ...], base_url: str,
                  progress: ProgressFn | None, cancel: threading.Event | None,
                  check_space: bool = False) -> None:
    """Fetch the missing files of `table`; progress covers all of them together."""
    dest_dir = Path(dest_dir)
    missing = set(_missing(dest_dir, table))
    todo = [(rel, size) for rel, size in table if rel in missing]
    total = sum(size for _, size in todo)
    if check_space and todo:
        _check_free_space(dest_dir, total)
    base = 0
    if progress:
        progress(0, total)
    for rel, size in todo:
        if cancel is not None and cancel.is_set():
            raise DownloadCancelled()
        offset = base

        def part_progress(done: int, _total: int | None, offset: int = offset) -> None:
            if progress:
                progress(offset + done, total)

        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        download(target.name, target.parent, part_progress, cancel, size,
                 url=base_url + rel)
        base += size


def _delete_set(dest_dir: Path, table: tuple[tuple[str, int], ...]) -> None:
    """Remove the files of `table` and any subfolders left empty."""
    dest_dir = Path(dest_dir)
    for rel, _ in table:
        for path in (dest_dir / rel, (dest_dir / rel).with_name(Path(rel).name + ".part")):
            try:
                path.unlink()
            except OSError:
                pass
    for sub in sorted({(dest_dir / rel).parent for rel, _ in table}, reverse=True):
        if sub != dest_dir:
            try:
                sub.rmdir()
            except OSError:
                pass


def _check_free_space(dest_dir: Path, needed: int) -> None:
    """OSError with a readable message when the drive of `dest_dir` cannot hold `needed` bytes."""
    probe = Path(dest_dir).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed + DISK_MARGIN:
        raise OSError(f"Not enough free disk space on {probe.anchor or probe}: the download "
                      f"needs {(needed + DISK_MARGIN) / 1e9:.1f} GB, {free / 1e9:.1f} GB is free.")


# ---------------------------------------------------------------- Supertonic

def supertonic_files() -> tuple[tuple[str, int], ...]:
    return MODEL_FILES


def supertonic_missing(dest_dir: Path) -> list[str]:
    """Relative paths of the Supertonic files that are not installed (or have the wrong size)."""
    return _missing(dest_dir, MODEL_FILES)


def download_supertonic(dest_dir: Path, progress: ProgressFn | None = None,
                        cancel: threading.Event | None = None) -> None:
    """Fetch the missing Supertonic files; progress covers all of them together."""
    _download_set(dest_dir, MODEL_FILES, SUPERTONIC_BASE_URL, progress, cancel)


def delete_supertonic(dest_dir: Path) -> None:
    """Remove the Supertonic files and any subfolders left empty."""
    _delete_set(dest_dir, MODEL_FILES)


# ---------------------------------------------------------------- Higgs

def higgs_files() -> tuple[tuple[str, int], ...]:
    return HIGGS_MODEL_FILES


def higgs_missing(dest_dir: Path) -> list[str]:
    """Relative paths of the Higgs files that are not installed (or have the wrong size)."""
    return _missing(dest_dir, HIGGS_MODEL_FILES)


def download_higgs(dest_dir: Path, progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None) -> None:
    """Fetch the missing Higgs files (about 3.85 GB). OSError when the disk is short of space."""
    _download_set(dest_dir, HIGGS_MODEL_FILES, HIGGS_BASE_URL, progress, cancel,
                  check_space=True)


def delete_higgs(dest_dir: Path) -> None:
    """Remove the Higgs model files and any subfolders left empty (custom voices live elsewhere)."""
    _delete_set(dest_dir, HIGGS_MODEL_FILES)


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
                                 description="Download the Kokoro, Supertonic or Higgs model files.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--supertonic", action="store_true",
                    help="download the Supertonic 3 files instead of Kokoro")
    ap.add_argument("--higgs", action="store_true",
                    help="download the Higgs Audio v3 files (3.85 GB) instead of Kokoro")
    ap.add_argument("--dest", default=None,
                    help="target folder (default: <app>/models, models/supertonic-3 or "
                         "models/higgs-audio-v3)")
    args = ap.parse_args(argv)
    if args.higgs:
        dest = Path(args.dest) if args.dest else ROOT / "models" / "higgs-audio-v3"
        try:
            print(f"higgs-audio-v3: {len(higgs_missing(dest))} file(s) to download...", flush=True)
            download_higgs(dest, _cli_progress("higgs-audio-v3"))
            print(f"  {dest}", flush=True)
        except (SetupError, DownloadCancelled, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0
    if args.supertonic:
        dest = Path(args.dest) if args.dest else ROOT / "models" / "supertonic-3"
        try:
            print(f"supertonic-3: {len(supertonic_missing(dest))} file(s) to download...", flush=True)
            download_supertonic(dest, _cli_progress("supertonic-3"))
            print(f"  {dest}", flush=True)
        except (SetupError, DownloadCancelled) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0
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
