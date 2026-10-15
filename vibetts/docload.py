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


def _epub_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        order: list[str] = []
        try:
            container = ET.fromstring(_read_member(zf, "META-INF/container.xml"))
            opf_path = next(el.get("full-path") for el in container.iter()
                            if _local(el.tag) == "rootfile" and el.get("full-path"))
            opf = ET.fromstring(_read_member(zf, opf_path))
            base = posixpath.dirname(opf_path)
            manifest = {el.get("id"): el.get("href") for el in opf.iter()
                        if _local(el.tag) == "item" and el.get("href")}
            for el in opf.iter():
                if _local(el.tag) == "itemref" and manifest.get(el.get("idref")):
                    href = unquote(manifest[el.get("idref")]).split("#", 1)[0]
                    order.append(posixpath.normpath(posixpath.join(base, href)))
        except (StopIteration, ET.ParseError, DocumentError):
            order = []
        order = [n for n in order if n in names]
        if not order:  # no usable spine: every html member in name order
            order = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
        texts = []
        for name in order:
            t = _html_to_text(_decode_html(_read_member(zf, name)))
            if t:
                texts.append(t)
    return "\n\n".join(texts)


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

def extract_text(path: str | Path) -> Document:
    p = Path(path)
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
    markdown = suffix in (".md", ".markdown")
    try:
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
    except DocumentError:
        raise
    except zipfile.BadZipFile:
        raise DocumentError(f"{p.name} is not a valid {suffix[1:]} file") from None
    except OSError as e:
        raise DocumentError(f"Cannot read {p.name}: {e.strerror or e}") from None
    except Exception as e:  # parser internals (pypdf, xml, ...) vary widely
        raise DocumentError(f"Could not read {p.name}: {e}") from None
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        hint = " It may be scanned images without a text layer." if suffix == ".pdf" else ""
        raise DocumentError(f"No text found in {p.name}.{hint}")
    return Document(text=text, markdown=markdown, title=p.stem)
