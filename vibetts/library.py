"""Book library for the web reader: original files plus a parsed-chapter cache, in one folder.

Layout: <id><ext> (the original bytes), <id>.json (parsed book: title, author, chapters) and
index.json (one summary per book, including the reading position). The id is the first 16 hex
characters of the sha256 of the file, so the same file is only stored once.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import docload
from .docload import DocumentError

log = logging.getLogger(__name__)

BOOK_ID = r"[0-9a-f]{16}"
_BOOK_ID = re.compile(BOOK_ID + r"\Z")
_FORMATS = {".markdown": "md", ".htm": "html"}   # everything else: the extension without the dot
_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class LibraryError(ValueError):
    """The file cannot be added (unsupported type or unreadable document); shown to the user as is."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_json(path: Path, obj) -> None:
    _write_atomic(path, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


class Library:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._lock = threading.Lock()
        self._index: dict[str, dict] = {}
        self._load_index()

    # ---- public api ---------------------------------------------------------------------------

    def add(self, data: bytes, filename: str) -> dict:
        name = _UNSAFE_NAME.sub("_", Path(filename.replace("\\", "/")).name).strip() or "book"
        ext = Path(name).suffix.lower()
        if ext not in docload.SUPPORTED:
            raise LibraryError(f"Unsupported file type '{ext}'. "
                               f"Supported: {', '.join(docload.SUPPORTED)}")
        book_id = hashlib.sha256(data).hexdigest()[:16]
        with self._lock:
            if book_id in self._index:
                return self._summary(self._index[book_id])
            self.folder.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.folder, prefix=".tmp-") as tmp:
                src = Path(tmp) / name   # keeps the real file stem, which becomes the default title
                src.write_bytes(data)
                try:
                    book = docload.extract_book(src)
                except DocumentError as e:
                    raise LibraryError(str(e)) from None
            _write_atomic(self.folder / f"{book_id}{ext}", data)
            self._write_cache(book_id, book.title, book.author,
                              [{"title": c.title, "text": c.text} for c in book.chapters])
            entry = {"id": book_id, "title": book.title, "author": book.author,
                     "format": _FORMATS.get(ext, ext[1:]), "ext": ext,
                     "chapters": len(book.chapters), "position": {"chapter": 0, "item": 0},
                     "added": _now(), "opened": None}
            self._index[book_id] = entry
            self._save_index()
            return self._summary(entry)

    def add_path(self, path: str | Path) -> dict:
        p = Path(path)
        try:
            if not p.is_file():
                raise LibraryError(f"File not found: {p.name or p}")
            size = p.stat().st_size
            if size > docload.MAX_FILE_BYTES:
                raise LibraryError(f"File is too large ({size // (1024 * 1024)} MB, "
                                   f"limit {docload.MAX_FILE_BYTES // (1024 * 1024)} MB)")
            data = p.read_bytes()
        except OSError as e:
            raise LibraryError(f"Cannot read {p.name}: {e.strerror or e}") from None
        return self.add(data, p.name)

    def list(self) -> list[dict]:
        with self._lock:
            entries = sorted(self._index.values(), key=lambda e: e["opened"] or e["added"], reverse=True)
            return [self._summary(e) for e in entries]

    def get(self, book_id: str) -> dict:
        with self._lock:
            entry = self._entry(book_id)
            cache = self._cache(entry)
            entry["opened"] = _now()
            self._save_index()
            out = self._summary(entry)
        out["toc"] = [{"index": i, "title": c["title"], "chars": len(c["text"])}
                      for i, c in enumerate(cache["chapters"])]
        return out

    def chapter(self, book_id: str, n: int) -> dict:
        with self._lock:
            chapters = self._cache(self._entry(book_id))["chapters"]
        if not 0 <= n < len(chapters):
            raise IndexError(f"Chapter {n} does not exist")
        ch = chapters[n]
        paras = [p.strip() for p in re.split(r"\n[ \t]*\n", ch["text"])]
        return {"index": n, "title": ch["title"], "paragraphs": [p for p in paras if p]}

    def set_position(self, book_id: str, chapter: int, item: int) -> None:
        with self._lock:
            entry = self._entry(book_id)
            if not 0 <= chapter < entry["chapters"]:
                raise ValueError(f"Chapter {chapter} does not exist")
            if item < 0:
                raise ValueError("Item must not be negative")
            entry["position"] = {"chapter": chapter, "item": item}
            self._save_index()

    def delete(self, book_id: str) -> None:
        with self._lock:
            entry = self._entry(book_id)
            del self._index[book_id]
            self._save_index()
            for f in (self.folder / f"{book_id}{entry.get('ext', '')}", self.folder / f"{book_id}.json"):
                try:
                    f.unlink()
                except OSError:
                    pass

    # ---- internals (call with the lock held) --------------------------------------------------

    def _entry(self, book_id: str) -> dict:
        if not isinstance(book_id, str) or not _BOOK_ID.match(book_id):
            raise KeyError(book_id)
        try:
            return self._index[book_id]
        except KeyError:
            raise KeyError(book_id) from None

    @staticmethod
    def _summary(entry: dict) -> dict:
        out = {k: v for k, v in entry.items() if k != "ext"}
        out["position"] = dict(entry["position"])
        return out

    def _load_index(self) -> None:
        path = self.folder / "index.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            books = raw["books"]
        except FileNotFoundError:
            return
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("Library index %s is unreadable (%s); starting empty", path, e)
            return
        for entry in books if isinstance(books, list) else []:
            try:
                if _BOOK_ID.match(entry["id"]) and entry["ext"] in docload.SUPPORTED:
                    entry.setdefault("title", entry["id"])
                    entry.setdefault("author", "")
                    entry.setdefault("format", _FORMATS.get(entry["ext"], entry["ext"][1:]))
                    entry["chapters"] = int(entry["chapters"])
                    entry["position"] = {"chapter": int(entry["position"]["chapter"]),
                                         "item": int(entry["position"]["item"])}
                    entry.setdefault("added", _now())
                    entry.setdefault("opened", None)
                    self._index[entry["id"]] = entry
            except (KeyError, TypeError, ValueError):
                log.warning("Skipping a bad entry in the library index")

    def _save_index(self) -> None:
        _write_json(self.folder / "index.json", {"books": list(self._index.values())})

    def _write_cache(self, book_id: str, title: str, author: str, chapters: list[dict]) -> None:
        _write_json(self.folder / f"{book_id}.json",
                    {"title": title, "author": author, "chapters": chapters})

    def _cache(self, entry: dict) -> dict:
        """Parsed book; re-parsed from the stored original when the cache file is missing or bad."""
        book_id = entry["id"]
        try:
            cache = json.loads((self.folder / f"{book_id}.json").read_text(encoding="utf-8"))
            if isinstance(cache["chapters"], list):
                return cache
        except (OSError, ValueError, KeyError, TypeError):
            pass
        original = self.folder / f"{book_id}{entry['ext']}"
        try:
            book = docload.extract_book(original)
        except DocumentError as e:
            log.warning("Cannot rebuild the cache of book %s: %s", book_id, e)
            raise KeyError(book_id) from None
        chapters = [{"title": c.title, "text": c.text} for c in book.chapters]
        entry["chapters"] = len(chapters)
        self._write_cache(book_id, entry["title"], entry["author"], chapters)
        return {"title": entry["title"], "author": entry["author"], "chapters": chapters}
