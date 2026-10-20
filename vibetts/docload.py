"""Extract readable text from documents: .txt .md .html .docx .epub .pdf.

Pure stdlib (pypdf is imported lazily for PDFs). Every failure is a DocumentError whose message
can be shown to the user as is. Paragraphs in the result are separated by blank lines; hard-wrapped
sources (PDF, wrapped .txt) are reflowed so textprep does not treat each wrapped line as a sentence.
"""
from __future__ import annotations

import codecs
import posixpath
import re
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

SUPPORTED = (".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".epub", ".pdf")
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_MEMBER_BYTES = 200 * 1024 * 1024  # per zip member, guards against zip bombs


class DocumentError(ValueError):
    """A document could not be read; the message is shown to the user as is."""


@dataclass
class Document:
    text: str
    markdown: bool   # text may contain markdown (clean it before speaking)
    title: str       # file stem, used for the default output file name


@dataclass
class Chapter:
    title: str
    text: str        # paragraphs separated by blank lines


@dataclass
class Book:
    title: str
    author: str
    chapters: list[Chapter]


def decode_bytes(data: bytes) -> str:
    """utf-8 (with or without BOM), then Windows-1250 (Central European), then latin-1."""
    for enc in ("utf-8-sig", "cp1250"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1")


# ---- reflow -----------------------------------------------------------------------------------

_PAGE_NUM = re.compile(r"^(?:page\s+)?\d{1,5}(?:\s*(?:/|of)\s*\d{1,5})?$", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^(?:[-*•]\s|\d+[.)]\s)")
_SENTENCE_END = (".", "!", "?", ":", "…")


def reflow(text: str) -> str:
    """Join hard-wrapped lines into paragraphs, drop page numbers, mend hyphenated words."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [ln.strip() for ln in block.split("\n")]
        lines = [ln for ln in lines if ln and not _PAGE_NUM.match(ln)]
        if not lines:
            continue
        out = lines[0]
        for nxt in lines[1:]:
            if out.endswith(_SENTENCE_END) or _LIST_ITEM.match(nxt):
                out += "\n" + nxt
            elif len(out) > 1 and out.endswith("-") and out[-2].isalpha() and nxt[:1].islower():
                out = out[:-1] + nxt
            else:
                out += " " + nxt
        paragraphs.append(out)
    return "\n\n".join(paragraphs)


def _looks_hard_wrapped(text: str) -> bool:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) < 4:
        return False
    wrapped = sum(1 for ln in lines if 50 <= len(ln) <= 90)
    return wrapped / len(lines) >= 0.6


# ---- html -------------------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "head", "noscript", "svg", "template"}
_BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
               "article", "blockquote", "pre", "ul", "ol", "table", "hr"}
_BREAK = "\x00"
_CHARSET = re.compile(rb"(?:charset|encoding)\s*=\s*[\"']?([A-Za-z0-9_-]+)", re.IGNORECASE)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append(_BREAK)

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append(_BREAK)

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        paras = (" ".join(p.split()) for p in "".join(self._parts).split(_BREAK))
        return "\n\n".join(p for p in paras if p)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _decode_html(data: bytes) -> str:
    if not data.startswith(codecs.BOM_UTF8):
        m = _CHARSET.search(data[:2048])
        if m:
            try:
                return data.decode(m.group(1).decode("ascii"))
            except (LookupError, UnicodeDecodeError):
                pass
    return decode_bytes(data)


# ---- zip based formats ------------------------------------------------------------------------

def _read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    try:
        info = zf.getinfo(name)
    except KeyError:
        raise DocumentError(f"Document is damaged: {name} is missing") from None
    if info.file_size > MAX_MEMBER_BYTES:
        raise DocumentError("Document is too large to read (suspicious archive)")
    return zf.read(info)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo("word/document.xml") if "word/document.xml" in zf.namelist() else None
        if info is None:
            raise DocumentError("Not a valid .docx file (word/document.xml is missing)")
        if info.file_size > MAX_MEMBER_BYTES:
            raise DocumentError("Document is too large to read (suspicious archive)")
        parts: list[str] = []
        with zf.open(info) as fh:
            for _, el in ET.iterparse(fh, events=("end",)):
                if el.tag == _W + "t":
                    parts.append(el.text or "")
                elif el.tag == _W + "tab":
                    parts.append(" ")
                elif el.tag in (_W + "br", _W + "cr"):
                    parts.append("\n")
                elif el.tag == _W + "p":
                    parts.append("\n\n")
                    el.clear()
    text = "".join(parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _epub_member(base: str, href: str) -> str:
    """Zip member name for an href relative to the folder `base`, without the #fragment."""
    return posixpath.normpath(posixpath.join(base, unquote(href).split("#", 1)[0]))


def _epub_spine(zf: zipfile.ZipFile) -> tuple[list[str], ET.Element | None, str, dict[str, ET.Element]]:
    """Spine members in reading order, plus the parsed OPF, its folder and the manifest items by id.

    Without a usable spine every html member is returned in name order (and the OPF is None).
    """
    names = set(zf.namelist())
    order: list[str] = []
    opf: ET.Element | None = None
    base = ""
    items: dict[str, ET.Element] = {}
    try:
        container = ET.fromstring(_read_member(zf, "META-INF/container.xml"))
        opf_path = next(el.get("full-path") for el in container.iter()
                        if _local(el.tag) == "rootfile" and el.get("full-path"))
        opf = ET.fromstring(_read_member(zf, opf_path))
        base = posixpath.dirname(opf_path)
        items = {el.get("id"): el for el in opf.iter()
                 if _local(el.tag) == "item" and el.get("href")}
        for el in opf.iter():
            if _local(el.tag) == "itemref" and el.get("idref") in items:
                order.append(_epub_member(base, items[el.get("idref")].get("href")))
    except (StopIteration, ET.ParseError, DocumentError):
        order, opf, base, items = [], None, "", {}
    order = [n for n in order if n in names]
    if not order:  # no usable spine: every html member in name order
        order = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
    return order, opf, base, items


def _epub_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        order = _epub_spine(zf)[0]
        texts = []
        for name in order:
            t = _html_to_text(_decode_html(_read_member(zf, name)))
            if t:
                texts.append(t)
    return "\n\n".join(texts)


_OPS_TYPE = "{http://www.idpf.org/2007/ops}type"


def _epub_toc(zf: zipfile.ZipFile, opf: ET.Element, base: str,
              items: dict[str, ET.Element]) -> dict[str, str]:
    """Chapter titles by zip member name, from the EPUB3 nav document or the EPUB2 NCX.

    The first title seen for a member wins. A missing or unreadable TOC gives an empty dict.
    """
    names = set(zf.namelist())
    nav = next((it for it in items.values() if "nav" in (it.get("properties") or "").split()), None)
    ncx = None
    spine = next((el for el in opf.iter() if _local(el.tag) == "spine"), None)
    if spine is not None and spine.get("toc") in items:
        ncx = items[spine.get("toc")]
    if ncx is None:
        ncx = next((it for it in items.values()
                    if it.get("media-type") == "application/x-dtbncx+xml"), None)
    toc: dict[str, str] = {}

    def add(src: str, href: str | None, title: str) -> None:
        title = " ".join(title.split())
        if href and title:
            target = _epub_member(posixpath.dirname(src), href)
            toc.setdefault(target, title)

    for item, kind in ((nav, "nav"), (ncx, "ncx")):
        if item is None:
            continue
        src = _epub_member(base, item.get("href"))
        if src not in names:
            continue
        try:
            root = ET.fromstring(_read_member(zf, src))
        except (ET.ParseError, DocumentError):
            continue
        if kind == "nav":
            for nav_el in root.iter():
                if _local(nav_el.tag) == "nav" and "toc" in (nav_el.get(_OPS_TYPE) or "").split():
                    for a in nav_el.iter():
                        if _local(a.tag) == "a":
                            add(src, a.get("href"), "".join(a.itertext()))
                    break
        else:
            for point in root.iter():
                if _local(point.tag) != "navPoint":
                    continue
                label = next((t for lab in point if _local(lab.tag) == "navLabel"
                              for t in lab if _local(t.tag) == "text"), None)
                content = next((c for c in point if _local(c.tag) == "content"), None)
                if label is not None and content is not None:
                    add(src, content.get("src"), "".join(label.itertext()))
        if toc:
            break
    return toc


_HEADING = re.compile(r"<h[1-3]\b[^>]*>(.*?)</h[1-3]>", re.IGNORECASE | re.DOTALL)


def _first_heading(html: str) -> str:
    m = _HEADING.search(html)
    return " ".join(_html_to_text(m.group(1)).split()) if m else ""


def _epub_book(path: Path) -> Book:
    with zipfile.ZipFile(path) as zf:
        order, opf, base, items = _epub_spine(zf)
        title, author = path.stem, ""
        toc: dict[str, str] = {}
        if opf is not None:
            found_title = False
            for el in opf.iter():
                text = " ".join("".join(el.itertext()).split())
                if _local(el.tag) == "title" and text and not found_title:
                    title, found_title = text, True
                elif _local(el.tag) == "creator" and text and not author:
                    author = text
            toc = _epub_toc(zf, opf, base, items)
        pages = [(name, _decode_html(_read_member(zf, name))) for name in order]
    texts = [(name, html, _html_to_text(html)) for name, html in pages]
    chapters: list[Chapter] = []
    if any(name in toc for name, _, _ in texts):
        for name, _, text in texts:
            if name in toc:
                chapters.append(Chapter(toc[name], text))
            elif chapters:
                last = chapters[-1]
                last.text = "\n\n".join(t for t in (last.text, text) if t)
            else:
                chapters.append(Chapter("Front matter", text))
    else:
        for name, html, text in texts:
            if text:
                chapters.append(Chapter(_first_heading(html) or f"Section {len(chapters) + 1}", text))
    return Book(title=title, author=author, chapters=[c for c in chapters if c.text.strip()])


_TXT_HEADING = re.compile(r"^(?:(?:chapter|part|book|prologue|epilogue)\b|(?:[ivxlc]+|\d+)\.?$)",
                          re.IGNORECASE)
_PART_CHARS = 20_000
_OPENING_MIN_CHARS = 200  # text before the first heading is a chapter only if it is more than a title page


def _txt_chapters(text: str, title: str) -> list[Chapter]:
    paras = [p.strip() for p in re.split(r"\n[ \t]*\n", text) if p.strip()]
    heads = [i for i, p in enumerate(paras) if len(p) <= 80 and _TXT_HEADING.match(p)]
    if len(heads) < 2:
        parts: list[list[str]] = [[]]
        size = 0
        for p in paras:
            if parts[-1] and size + len(p) > _PART_CHARS:
                parts.append([])
                size = 0
            parts[-1].append(p)
            size += len(p)
        if len(parts) == 1:
            return [Chapter(title, "\n\n".join(parts[0]))]
        return [Chapter(f"Part {n}", "\n\n".join(ps)) for n, ps in enumerate(parts, 1)]
    chapters: list[Chapter] = []
    opening = "\n\n".join(paras[:heads[0]])
    if len(opening) >= _OPENING_MIN_CHARS:
        chapters.append(Chapter("Opening", opening))
    for k, i in enumerate(heads):
        end = heads[k + 1] if k + 1 < len(heads) else len(paras)
        body = "\n\n".join(paras[i + 1:end])
        if body:
            chapters.append(Chapter(paras[i], body))
    return chapters


def _pdf_text(path: Path) -> str:
    try:
        import pypdf
    except ImportError:
        raise DocumentError("PDF support needs the pypdf package, which is not installed") from None
    reader = pypdf.PdfReader(str(path))
    if reader.is_encrypted:
        try:
            ok = bool(reader.decrypt(""))
        except Exception:
            ok = False
        if not ok:
            raise DocumentError("This PDF is encrypted")
    pages = [(page.extract_text() or "") for page in reader.pages]
    return reflow("\n\n".join(pages))


# ---- entry point ------------------------------------------------------------------------------

def _check_file(p: Path) -> str:
    """Validate the path (exists, supported, size) and return the lower-case suffix."""
    if not p.is_file():
        raise DocumentError(f"File not found: {p.name or p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED:
        raise DocumentError(f"Unsupported file type '{p.suffix}'. Supported: {', '.join(SUPPORTED)}")
    try:
        size = p.stat().st_size
    except OSError as e:
        raise DocumentError(f"Cannot read {p.name}: {e.strerror or e}") from None
    if size > MAX_FILE_BYTES:
        raise DocumentError(f"File is too large ({size // (1024 * 1024)} MB, "
                            f"limit {MAX_FILE_BYTES // (1024 * 1024)} MB)")
    return suffix


@contextmanager
def _guarded(p: Path, suffix: str):
    """Turn whatever a format reader raises into a DocumentError."""
    try:
        yield
    except DocumentError:
        raise
    except zipfile.BadZipFile:
        raise DocumentError(f"{p.name} is not a valid {suffix[1:]} file") from None
    except OSError as e:
        raise DocumentError(f"Cannot read {p.name}: {e.strerror or e}") from None
    except Exception as e:  # parser internals (pypdf, xml, ...) vary widely
        raise DocumentError(f"Could not read {p.name}: {e}") from None


def extract_book(path: str | Path) -> Book:
    """Like extract_text but keeps chapters: EPUB (TOC or spine) and .txt (headings) are split."""
    p = Path(path)
    suffix = _check_file(p)
    if suffix not in (".epub", ".txt"):
        doc = extract_text(p)
        return Book(title=doc.title, author="", chapters=[Chapter(doc.title, doc.text)])
    with _guarded(p, suffix):
        if suffix == ".epub":
            book = _epub_book(p)
        else:
            text = decode_bytes(p.read_bytes()).replace("\r\n", "\n").replace("\r", "\n")
            if _looks_hard_wrapped(text):
                text = reflow(text)
            chapters = _txt_chapters(text, p.stem) if text.strip() else []
            book = Book(title=p.stem, author="", chapters=chapters)
    if not book.chapters:
        raise DocumentError(f"No text found in {p.name}.")
    return book


def extract_text(path: str | Path) -> Document:
    p = Path(path)
    suffix = _check_file(p)
    markdown = suffix in (".md", ".markdown")
    with _guarded(p, suffix):
        if suffix in (".txt", ".md", ".markdown"):
            text = decode_bytes(p.read_bytes())
            if suffix == ".txt" and _looks_hard_wrapped(text):
                text = reflow(text)
        elif suffix in (".html", ".htm"):
            text = _html_to_text(_decode_html(p.read_bytes()))
        elif suffix == ".docx":
            text = _docx_text(p)
        elif suffix == ".epub":
            text = _epub_text(p)
        else:
            text = _pdf_text(p)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        hint = " It may be scanned images without a text layer." if suffix == ".pdf" else ""
        raise DocumentError(f"No text found in {p.name}.{hint}")
    return Document(text=text, markdown=markdown, title=p.stem)
