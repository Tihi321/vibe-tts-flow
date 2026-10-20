import sys
import types
import zipfile

import pytest

from vibetts import docload
from vibetts.docload import DocumentError, decode_bytes, extract_book, extract_text, reflow


# ---- text / markdown --------------------------------------------------------------------------

def test_txt_utf8_sig(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("﻿Hello čćž.".encode("utf-8"))
    doc = extract_text(p)
    assert doc.text == "Hello čćž."
    assert doc.markdown is False and doc.title == "a"


def test_txt_cp1250():
    assert decode_bytes("čćžšđ".encode("cp1250")) == "čćžšđ"


def test_decode_latin1_fallback():
    # 0x81 is undefined in cp1250 -> latin-1
    assert decode_bytes(b"a\x81b") == "a\x81b"


def test_txt_cp1250_file(tmp_path):
    p = tmp_path / "hr.txt"
    p.write_bytes("Žuta šuma, đak.".encode("cp1250"))
    assert extract_text(p).text == "Žuta šuma, đak."


def test_md_flag_no_reflow(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("# Title\n\n- one\n- two", encoding="utf-8")
    doc = extract_text(p)
    assert doc.markdown is True
    assert doc.text == "# Title\n\n- one\n- two"
    q = tmp_path / "n.markdown"
    q.write_text("x", encoding="utf-8")
    assert extract_text(q).markdown is True


def test_hard_wrapped_txt_is_reflowed(tmp_path):
    line = "The quick brown fox jumps over the lazy dog and keeps running far"
    text = "\n".join([line] * 5)
    p = tmp_path / "w.txt"
    p.write_text(text, encoding="utf-8")
    out = extract_text(p).text
    assert "\n" not in out
    assert out.count("running far The quick") == 4


def test_short_lines_txt_untouched(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("one\ntwo\nthree\nfour", encoding="utf-8")
    assert extract_text(p).text == "one\ntwo\nthree\nfour"


# ---- html -------------------------------------------------------------------------------------

def test_html(tmp_path):
    p = tmp_path / "p.html"
    p.write_text(
        "<html><head><title>T</title><style>p{}</style></head><body>"
        "<script>var x = 1;</script><h1>Head</h1><p>Tom &amp; Jerry\n  are   &eacute;ager</p>"
        "<noscript>nope</noscript><div>Second<br>line</div></body></html>", encoding="utf-8")
    assert extract_text(p).text == "Head\n\nTom & Jerry are éager\n\nSecond\n\nline"


def test_html_meta_charset(tmp_path):
    p = tmp_path / "c.htm"
    p.write_bytes(b'<meta charset="windows-1250"><p>' + "šđ".encode("cp1250") + b"</p>")
    assert extract_text(p).text == "šđ"


# ---- docx -------------------------------------------------------------------------------------

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(path, body, extra=None):
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", xml)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)


def test_docx(tmp_path):
    p = tmp_path / "d.docx"
    make_docx(p, "<w:p><w:r><w:t>First</w:t><w:tab/><w:t>para</w:t></w:r></w:p>"
                 "<w:p></w:p>"
                 "<w:p><w:r><w:t>Line</w:t><w:br/><w:t>break</w:t></w:r></w:p>")
    assert extract_text(p).text == "First para\n\nLine\nbreak"


def test_docx_bad_zip(tmp_path):
    p = tmp_path / "bad.docx"
    p.write_bytes(b"not a zip")
    with pytest.raises(DocumentError):
        extract_text(p)


def test_docx_zip_bomb(tmp_path, monkeypatch):
    p = tmp_path / "b.docx"
    make_docx(p, "<w:p><w:r><w:t>x</w:t></w:r></w:p>")
    monkeypatch.setattr(docload, "MAX_MEMBER_BYTES", 10)
    with pytest.raises(DocumentError, match="too large"):
        extract_text(p)


def test_docx_missing_document_xml(tmp_path):
    p = tmp_path / "e.docx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("other.xml", "<a/>")
    with pytest.raises(DocumentError):
        extract_text(p)


# ---- epub -------------------------------------------------------------------------------------

def xhtml(body):
    return f"<?xml version='1.0'?><html><head><title>x</title></head><body>{body}</body></html>"


def test_epub_spine_order_and_hrefs(tmp_path):
    p = tmp_path / "book.epub"
    opf = ('<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf">'
           '<manifest>'
           '<item id="a" href="text/a.xhtml" media-type="application/xhtml+xml"/>'
           '<item id="b" href="text/b%20two.xhtml" media-type="application/xhtml+xml"/>'
           '<item id="c" href="c.xhtml" media-type="application/xhtml+xml"/>'
           '</manifest><spine><itemref idref="c"/><itemref idref="b"/><itemref idref="a"/></spine>'
           '</package>')
    container = ('<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                 '<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/text/a.xhtml", xhtml("<p>Alpha</p>"))
        zf.writestr("OEBPS/text/b two.xhtml", xhtml("<p>Bravo</p>"))
        zf.writestr("OEBPS/c.xhtml", xhtml("<p>Charlie</p>"))
    doc = extract_text(p)
    assert doc.text == "Charlie\n\nBravo\n\nAlpha"
    assert doc.markdown is False


def test_epub_fallback_without_spine(tmp_path):
    p = tmp_path / "f.epub"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("z.xhtml", xhtml("<p>Zed</p>"))
        zf.writestr("b.html", xhtml("<p>Bee</p>"))
    assert extract_text(p).text == "Bee\n\nZed"


CONTAINER = ('<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
             '<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
NAV = ('<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml" '
       'xmlns:epub="http://www.idpf.org/2007/ops"><body>'
       '<nav epub:type="landmarks"><ol><li><a href="c1.xhtml">Landmark</a></li></ol></nav>'
       '<nav epub:type="toc"><ol>'
       '<li><a href="c1.xhtml">One</a></li><li><a href="c2.xhtml#sec">Two</a></li>'
       '<li><a href="c2.xhtml#other">Two again</a></li></ol></nav></body></html>')
NCX = ('<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>'
       '<navPoint id="n1"><navLabel><text>One</text></navLabel><content src="c1.xhtml"/></navPoint>'
       '<navPoint id="n2"><navLabel><text>Two</text></navLabel><content src="c2.xhtml#x"/></navPoint>'
       '</navMap></ncx>')
XH = "application/xhtml+xml"
NCX_TYPE = "application/x-dtbncx+xml"


def make_epub(path, files, manifest, spine, spine_attrs="", meta=""):
    items = "".join(f'<item id="{i}" href="{h}" media-type="{m}" {a}/>' for i, h, m, a in manifest)
    refs = "".join(f'<itemref idref="{i}"/>' for i in spine)
    opf = ('<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf">'
           f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">{meta}</metadata>'
           f'<manifest>{items}</manifest><spine {spine_attrs}>{refs}</spine></package>')
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("META-INF/container.xml", CONTAINER)
        zf.writestr("OEBPS/content.opf", opf)
        for name, body in files.items():
            zf.writestr("OEBPS/" + name, body)
    return path


CHAPTER_FILES = {"c1.xhtml": xhtml("<h1>Heading one</h1><p>Alpha</p>"),
                 "c2.xhtml": xhtml("<h2>Heading two</h2><p>Bravo</p>"),
                 "c3.xhtml": xhtml("<p>Charlie</p>")}
CHAPTER_ITEMS = [("c1", "c1.xhtml", XH, ""), ("c2", "c2.xhtml", XH, ""), ("c3", "c3.xhtml", XH, "")]


def test_epub_book_nav_toc_title_author_and_merge(tmp_path):
    meta = "<dc:title>The Book</dc:title><dc:creator>Ann Author</dc:creator><dc:creator>Other</dc:creator>"
    p = make_epub(tmp_path / "b.epub", {**CHAPTER_FILES, "nav.xhtml": NAV},
                  CHAPTER_ITEMS + [("nav", "nav.xhtml", XH, 'properties="nav"')],
                  ["c1", "c2", "c3"], meta=meta)
    book = extract_book(p)
    assert (book.title, book.author) == ("The Book", "Ann Author")
    # c3 has no TOC entry: merged into "Two"; the first title for c2 wins
    assert [c.title for c in book.chapters] == ["One", "Two"]
    assert book.chapters[0].text == "Heading one\n\nAlpha"
    assert book.chapters[1].text == "Heading two\n\nBravo\n\nCharlie"


def test_epub_book_ncx_toc(tmp_path):
    p = make_epub(tmp_path / "b.epub", {**CHAPTER_FILES, "toc.ncx": NCX},
                  CHAPTER_ITEMS + [("ncx", "toc.ncx", NCX_TYPE, "")],
                  ["c1", "c2", "c3"], spine_attrs='toc="ncx"')
    book = extract_book(p)
    assert (book.title, book.author) == ("b", "")
    assert [c.title for c in book.chapters] == ["One", "Two"]
    assert "Charlie" in book.chapters[1].text


def test_epub_book_ncx_found_by_media_type(tmp_path):
    p = make_epub(tmp_path / "b.epub", {**CHAPTER_FILES, "toc.ncx": NCX},
                  CHAPTER_ITEMS + [("ncx", "toc.ncx", NCX_TYPE, "")], ["c1", "c2", "c3"])
    assert [c.title for c in extract_book(p).chapters] == ["One", "Two"]


def test_epub_book_front_matter(tmp_path):
    files = {**CHAPTER_FILES, "cover.xhtml": xhtml("<p>Dedication</p>"), "toc.ncx": NCX}
    p = make_epub(tmp_path / "b.epub", files,
                  [("cv", "cover.xhtml", XH, "")] + CHAPTER_ITEMS + [("ncx", "toc.ncx", NCX_TYPE, "")],
                  ["cv", "c1", "c2", "c3"], spine_attrs='toc="ncx"')
    book = extract_book(p)
    assert [c.title for c in book.chapters] == ["Front matter", "One", "Two"]
    assert book.chapters[0].text == "Dedication"


def test_epub_book_without_toc_uses_headings(tmp_path):
    files = {**CHAPTER_FILES, "c4.xhtml": xhtml("<p>   </p>")}
    p = make_epub(tmp_path / "b.epub", files, CHAPTER_ITEMS + [("c4", "c4.xhtml", XH, "")],
                  ["c1", "c2", "c3", "c4"])
    book = extract_book(p)
    # the empty spine item is dropped; the untitled one is "Section N" by its chapter number
    assert [c.title for c in book.chapters] == ["Heading one", "Heading two", "Section 3"]


def test_epub_book_drops_empty_toc_chapter(tmp_path):
    files = {"c1.xhtml": xhtml("<p></p>"), "c2.xhtml": xhtml("<p>Bravo</p>"), "toc.ncx": NCX}
    p = make_epub(tmp_path / "b.epub", files,
                  CHAPTER_ITEMS[:2] + [("ncx", "toc.ncx", NCX_TYPE, "")], ["c1", "c2"])
    assert [c.title for c in extract_book(p).chapters] == ["Two"]


def test_epub_book_fallback_without_opf(tmp_path):
    p = tmp_path / "f.epub"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("z.xhtml", xhtml("<p>Zed</p>"))
        zf.writestr("b.html", xhtml("<h1>Bee</h1><p>Buzz</p>"))
    book = extract_book(p)
    assert book.title == "f"
    assert [(c.title, c.text) for c in book.chapters] == [("Bee", "Bee\n\nBuzz"), ("Section 2", "Zed")]


def test_epub_book_no_text(tmp_path):
    p = make_epub(tmp_path / "b.epub", {"c1.xhtml": xhtml("<p></p>")}, CHAPTER_ITEMS[:1], ["c1"])
    with pytest.raises(DocumentError):
        extract_book(p)


def test_extract_text_still_joins_epub(tmp_path):
    p = make_epub(tmp_path / "b.epub", CHAPTER_FILES, CHAPTER_ITEMS, ["c1", "c2", "c3"])
    assert extract_text(p).text == "Heading one\n\nAlpha\n\nHeading two\n\nBravo\n\nCharlie"


# ---- txt books --------------------------------------------------------------------------------

def test_txt_book_headings(tmp_path):
    intro = "A long enough introduction. " * 10
    p = tmp_path / "novel.txt"
    p.write_text(f"{intro}\n\nChapter 1\n\nFirst body.\n\nMore first.\n\n"
                 "II.\n\nSecond body.\n\nEPILOGUE\n\nThe end.", encoding="utf-8")
    book = extract_book(p)
    assert (book.title, book.author) == ("novel", "")
    assert [c.title for c in book.chapters] == ["Opening", "Chapter 1", "II.", "EPILOGUE"]
    assert book.chapters[1].text == "First body.\n\nMore first."
    assert book.chapters[3].text == "The end."


def test_txt_book_short_preamble_is_not_a_chapter(tmp_path):
    p = tmp_path / "n.txt"
    p.write_text("My Title\n\nChapter 1\n\nOne.\n\nChapter 2\n\nTwo.", encoding="utf-8")
    assert [c.title for c in extract_book(p).chapters] == ["Chapter 1", "Chapter 2"]


def test_txt_book_single_chapter_when_short(tmp_path):
    p = tmp_path / "short.txt"
    p.write_text("Chapter 1\n\nJust one heading.\n\nText.", encoding="utf-8")
    book = extract_book(p)
    assert [c.title for c in book.chapters] == ["short"]
    assert book.chapters[0].text == "Chapter 1\n\nJust one heading.\n\nText."


def test_txt_book_part_fallback(tmp_path):
    p = tmp_path / "long.txt"
    para = "Words go here and there. " * 40   # 1000 chars
    p.write_text("\n\n".join([para.strip()] * 45), encoding="utf-8")
    book = extract_book(p)
    assert [c.title for c in book.chapters] == ["Part 1", "Part 2", "Part 3"]
    assert all(len(c.text) < 21000 for c in book.chapters)
    assert sum(c.text.count("\n\n") + 1 for c in book.chapters) == 45


def test_other_types_are_one_chapter(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<p>One</p><p>Two</p>", encoding="utf-8")
    book = extract_book(p)
    assert (book.title, book.author) == ("page", "")
    assert [(c.title, c.text) for c in book.chapters] == [("page", "One\n\nTwo")]


def test_extract_book_errors(tmp_path):
    with pytest.raises(DocumentError):
        extract_book(tmp_path / "missing.txt")
    bad = tmp_path / "x.xyz"
    bad.write_text("a")
    with pytest.raises(DocumentError):
        extract_book(bad)
    empty = tmp_path / "e.txt"
    empty.write_text("   ")
    with pytest.raises(DocumentError):
        extract_book(empty)
    notzip = tmp_path / "n.epub"
    notzip.write_bytes(b"nope")
    with pytest.raises(DocumentError):
        extract_book(notzip)


# ---- pdf --------------------------------------------------------------------------------------

def install_fake_pypdf(monkeypatch, pages, encrypted=False, decrypt_result=0):
    class Page:
        def __init__(self, t):
            self.t = t

        def extract_text(self):
            return self.t

    class PdfReader:
        def __init__(self, source):
            self.is_encrypted = encrypted
            self.pages = [Page(t) for t in pages]

        def decrypt(self, pw):
            return decrypt_result

    mod = types.ModuleType("pypdf")
    mod.PdfReader = PdfReader
    monkeypatch.setitem(sys.modules, "pypdf", mod)


def test_pdf_pages(tmp_path, monkeypatch):
    install_fake_pypdf(monkeypatch, ["Hello wor-\nld of text\nhere.", "12", "Next page."])
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    doc = extract_text(p)
    assert doc.text == "Hello world of text here.\n\nNext page."


def test_pdf_encrypted(tmp_path, monkeypatch):
    install_fake_pypdf(monkeypatch, ["secret"], encrypted=True, decrypt_result=0)
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    with pytest.raises(DocumentError, match="encrypted"):
        extract_text(p)


def test_pdf_encrypted_empty_password_ok(tmp_path, monkeypatch):
    install_fake_pypdf(monkeypatch, ["open"], encrypted=True, decrypt_result=1)
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    assert extract_text(p).text == "open"


def test_pdf_empty(tmp_path, monkeypatch):
    install_fake_pypdf(monkeypatch, ["", "  "])
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    with pytest.raises(DocumentError, match="scanned"):
        extract_text(p)


# ---- reflow -----------------------------------------------------------------------------------

def test_reflow_joins_lines_and_hyphens():
    assert reflow("a wrapped\nline of text") == "a wrapped line of text"
    assert reflow("exam-\nple word") == "example word"
    assert reflow("Well-\nKnown") == "Well- Known"  # uppercase after hyphen: keep the hyphen


def test_reflow_keeps_breaks_after_sentence_end_and_lists():
    assert reflow("One.\nTwo") == "One.\nTwo"
    assert reflow("Heading:\nbody") == "Heading:\nbody"
    assert reflow("Items\n- a\n- b\n1. c") == "Items\n- a\n- b\n1. c"


def test_reflow_drops_page_numbers_and_normalizes():
    assert reflow("Text one\r\n\r\n12\r\n\r\nPage 3\r\n\r\nText two") == "Text one\n\nText two"
    assert reflow("line\n42\nmore") == "line more"


def test_reflow_paragraphs():
    assert reflow("a\nb\n\n\n\nc\nd") == "a b\n\nc d"
    assert reflow("") == ""


# ---- errors -----------------------------------------------------------------------------------

def test_unsupported_suffix(tmp_path):
    p = tmp_path / "a.xyz"
    p.write_text("hi")
    with pytest.raises(DocumentError, match=r"Unsupported.*\.pdf"):
        extract_text(p)


def test_missing_file(tmp_path):
    with pytest.raises(DocumentError, match="not found"):
        extract_text(tmp_path / "nope.txt")


def test_too_large(tmp_path, monkeypatch):
    p = tmp_path / "big.txt"
    p.write_text("x" * 100)
    monkeypatch.setattr(docload, "MAX_FILE_BYTES", 10)
    with pytest.raises(DocumentError, match="too large"):
        extract_text(p)


def test_empty_text(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("  \n ")
    with pytest.raises(DocumentError, match="No text"):
        extract_text(p)
