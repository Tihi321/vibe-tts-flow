import json
import re

import pytest

from vibetts import docload
from vibetts.library import BOOK_ID, Library, LibraryError

NOVEL = ("Chapter 1\n\nFirst body.\n\n  Second   paragraph.  \n\n"
         "Chapter 2\n\nOther body.").encode("utf-8")


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path / "library")


def test_add_returns_summary_and_stores_files(lib, tmp_path):
    s = lib.add(NOVEL, "My Novel.txt")
    assert re.fullmatch(BOOK_ID, s["id"]) and s["format"] == "txt"
    assert (s["title"], s["author"], s["chapters"]) == ("My Novel", "", 2)
    assert s["position"] == {"chapter": 0, "item": 0}
    assert s["opened"] is None and s["added"].endswith("+00:00")
    folder = tmp_path / "library"
    assert (folder / f"{s['id']}.txt").read_bytes() == NOVEL
    cache = json.loads((folder / f"{s['id']}.json").read_text(encoding="utf-8"))
    assert [c["title"] for c in cache["chapters"]] == ["Chapter 1", "Chapter 2"]
    assert json.loads((folder / "index.json").read_text(encoding="utf-8"))
    assert not [f for f in folder.iterdir() if f.name.startswith(".tmp")]


def test_same_bytes_returns_existing_without_reparse(lib, monkeypatch):
    first = lib.add(NOVEL, "a.txt")
    monkeypatch.setattr(docload, "extract_book", lambda p: pytest.fail("re-parsed"))
    again = lib.add(NOVEL, "other name.txt")
    assert again == first
    assert len(lib.list()) == 1


def test_format_names(lib):
    assert lib.add(b"# T\n\ntext", "n.md")["format"] == "md"
    assert lib.add(b"<p>hi</p>", "n.HTM")["format"] == "html"


def test_unsupported_and_unreadable(lib):
    with pytest.raises(LibraryError):
        lib.add(b"x", "a.xyz")
    with pytest.raises(LibraryError):
        lib.add(b"x", "noext")
    with pytest.raises(LibraryError):
        lib.add(b"not a zip", "a.epub")
    with pytest.raises(LibraryError):
        lib.add(b"   ", "a.txt")
    assert lib.list() == []


def test_list_most_recently_opened_first(lib):
    a = lib.add(b"Alpha text", "a.txt")["id"]
    b = lib.add(b"Bravo text", "b.txt")["id"]
    c = lib.add(b"Charlie text", "c.txt")["id"]
    assert [x["id"] for x in lib.list()] == [c, b, a]   # never opened: by added
    lib.get(a)
    assert [x["id"] for x in lib.list()] == [a, c, b]
    lib.get(b)
    assert [x["id"] for x in lib.list()][:2] == [b, a]


def test_get_has_toc_and_marks_opened(lib):
    s = lib.add(NOVEL, "n.txt")
    got = lib.get(s["id"])
    assert got["opened"] is not None
    assert [(t["index"], t["title"]) for t in got["toc"]] == [(0, "Chapter 1"), (1, "Chapter 2")]
    assert got["toc"][1]["chars"] == len("Other body.")
    assert "ext" not in got


def test_chapter_paragraphs(lib):
    s = lib.add(NOVEL, "n.txt")
    ch = lib.chapter(s["id"], 0)
    assert ch == {"index": 0, "title": "Chapter 1",
                  "paragraphs": ["First body.", "Second   paragraph."]}
    with pytest.raises(IndexError):
        lib.chapter(s["id"], 2)
    with pytest.raises(IndexError):
        lib.chapter(s["id"], -1)


def test_position_persists_across_instances(lib, tmp_path):
    s = lib.add(NOVEL, "n.txt")
    lib.set_position(s["id"], 1, 7)
    again = Library(tmp_path / "library")
    assert again.list()[0]["position"] == {"chapter": 1, "item": 7}
    assert again.get(s["id"])["id"] == s["id"]


def test_set_position_validation(lib):
    s = lib.add(NOVEL, "n.txt")
    for chapter, item in ((2, 0), (-1, 0), (0, -1)):
        with pytest.raises(ValueError):
            lib.set_position(s["id"], chapter, item)
    with pytest.raises(KeyError):
        lib.set_position("0" * 16, 0, 0)
    assert lib.list()[0]["position"] == {"chapter": 0, "item": 0}


def test_delete_removes_files(lib, tmp_path):
    s = lib.add(NOVEL, "n.txt")
    lib.delete(s["id"])
    assert lib.list() == []
    assert sorted(f.name for f in (tmp_path / "library").iterdir()) == ["index.json"]
    with pytest.raises(KeyError):
        lib.delete(s["id"])
    assert Library(tmp_path / "library").list() == []


@pytest.mark.parametrize("bad", ["../index", "..", "", "ABCDEF0123456789", "0" * 15, "0" * 17,
                                 "0123456789abcde\n", "zzzzzzzzzzzzzzzz", "a/b"])
def test_invalid_ids_rejected(lib, bad):
    lib.add(NOVEL, "n.txt")
    for call in (lambda: lib.get(bad), lambda: lib.chapter(bad, 0), lambda: lib.delete(bad),
                 lambda: lib.set_position(bad, 0, 0)):
        with pytest.raises(KeyError):
            call()


def test_unknown_valid_id(lib):
    with pytest.raises(KeyError):
        lib.get("0123456789abcdef")


def test_corrupt_index_tolerated(tmp_path, caplog):
    folder = tmp_path / "library"
    folder.mkdir()
    (folder / "index.json").write_text("{not json", encoding="utf-8")
    lib = Library(folder)
    assert lib.list() == []
    assert "unreadable" in caplog.text
    s = lib.add(NOVEL, "n.txt")   # works and rewrites a valid index
    assert [b["id"] for b in Library(folder).list()] == [s["id"]]


def test_missing_cache_is_rebuilt_from_original(lib, tmp_path):
    s = lib.add(NOVEL, "n.txt")
    cache = tmp_path / "library" / f"{s['id']}.json"
    cache.unlink()
    assert lib.chapter(s["id"], 1)["paragraphs"] == ["Other body."]
    assert cache.is_file()
    cache.write_text("garbage", encoding="utf-8")
    assert lib.chapter(s["id"], 0)["title"] == "Chapter 1"


def test_add_path(lib, tmp_path, monkeypatch):
    f = tmp_path / "Some Book.txt"
    f.write_bytes(NOVEL)
    s = lib.add_path(f)
    assert s["title"] == "Some Book"
    assert lib.add_path(str(f)) == s
    with pytest.raises(LibraryError):
        lib.add_path(tmp_path / "missing.txt")
    monkeypatch.setattr(docload, "MAX_FILE_BYTES", 10)
    with pytest.raises(LibraryError, match="too large"):
        lib.add_path(f)
