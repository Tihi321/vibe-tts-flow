"""Text preparation for speech: markdown cleanup and splitting into short chunks.

Chunks are spoken one after another while the next one is generated, so the first
chunk is kept short: that is what sets how soon reading starts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

FIRST_CHUNK_CHARS = 100   # split the first sentence at a comma when it is longer
MAX_CHUNK_CHARS = 400     # split any sentence longer than this
MERGE_BELOW_CHARS = 40    # glue very short sentences ("Hi.", "OK.") to the next one...
MERGE_MAX_CHARS = 100     # ...as long as the result stays short

_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>()\[\]]+", re.IGNORECASE)
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_FENCE = re.compile(r"^\s*(```|~~~).*$", re.MULTILINE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+•]\s+", re.MULTILINE)
_RULE = re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|\*|~~)(?=\S)(.+?)(?<=\S)\1")
_UNDERSCORE_EMPHASIS = re.compile(r"(?<!\w)(__|_)(?=\S)(.+?)(?<=\S)\1(?!\w)")  # not snake_case
_TABLE_SEP = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*\n?",
                        re.MULTILINE)  # also eats its newline: no paragraph break mid-table
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

# sentence end: . ! ? … (plus closing quotes/brackets) followed by whitespace
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:—–])\s+")
_SPEAKABLE = re.compile(r"\w")


def clean_text(text: str, markdown: bool = True) -> str:
    """Normalise whitespace; with `markdown`, drop markup that sounds bad when spoken."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = text.replace("\t", " ")
    if markdown:
        text = _FENCE.sub("", text)
        text = _MD_IMAGE.sub(r"\1", text)
        text = _MD_LINK.sub(r"\1", text)
        text = _URL.sub("link", text)
        text = _HTML_TAG.sub("", text)
        text = _TABLE_SEP.sub("", text)
        text = _RULE.sub("", text)
        text = _HEADING.sub("", text)
        text = _QUOTE.sub("", text)
        text = _BULLET.sub("", text)
        text = _EMPHASIS.sub(r"\2", text)
        text = _UNDERSCORE_EMPHASIS.sub(r"\2", text)
        text = text.replace("`", "")
        text = re.sub(r"[ \t]*\|[ \t]*", ", ", text)  # table cells
        text = re.sub(r"^(, )+|(, )+$", "", text, flags=re.MULTILINE)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    # keep paragraph breaks (blank lines) as "\n\n", single line breaks as "\n"
    out: list[str] = []
    blank = False
    for line in lines:
        if not line:
            blank = bool(out)
            continue
        if out:
            out.append("\n\n" if blank else "\n")
        out.append(line)
        blank = False
    return "".join(out).strip()


@dataclass
class Chunk:
    text: str
    pause_after: float = 1.0  # multiple of the sentence pause (paragraph ends use more)
    tag: Any = None  # reader sessions: index of the item the chunk came from


def _split_long(sentence: str, limit: int, words: bool = True) -> list[str]:
    """Split at clause marks, then (with `words`) at spaces, so no piece is longer
    than `limit`."""
    if len(sentence) <= limit:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for part in _CLAUSE_END.split(sentence):
        if current and len(current) + 1 + len(part) > limit:
            pieces.append(current)
            current = part
        else:
            current = f"{current} {part}" if current else part
    if current:
        pieces.append(current)
    if not words:
        return pieces
    result: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            result.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            result.append(piece)
    return result


def split_chunks(text: str, short_first: bool = True) -> list[Chunk]:
    """Split cleaned text into speakable chunks (sentences; line breaks also end one).
    `short_first`: cut the very first chunk short (it sets the start delay)."""
    chunks: list[Chunk] = []
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    for pi, para in enumerate(paragraphs):
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        sentences: list[str] = []
        for line in lines:
            # pieces with no letter or digit ("**", "---") have nothing to say
            sentences.extend(s.strip() for s in _SENTENCE_END.split(line) if _SPEAKABLE.search(s))
        merged: list[str] = []
        for s in sentences:
            if (merged and len(merged[-1]) < MERGE_BELOW_CHARS
                    and len(merged[-1]) + 1 + len(s) <= MERGE_MAX_CHARS):
                merged[-1] = f"{merged[-1]} {s}"
            else:
                merged.append(s)
        for si, s in enumerate(merged):
            parts = _split_long(s, MAX_CHUNK_CHARS)
            if short_first and not chunks:
                # the very first piece sets the start delay: cut it at a clause mark
                # (never mid-phrase) when one comes early enough; rejoin the rest
                head = _split_long(parts[0], FIRST_CHUNK_CHARS, words=False)
                if len(head) > 1 and len(head[0]) <= FIRST_CHUNK_CHARS:
                    rest = _split_long(" ".join([*head[1:], *parts[1:]]), MAX_CHUNK_CHARS)
                    parts = [head[0], *rest]
            for k, part in enumerate(parts):
                last_in_sentence = k == len(parts) - 1
                if not last_in_sentence:
                    pause = 0.4
                elif si == len(merged) - 1 and pi < len(paragraphs) - 1:
                    pause = 2.0
                else:
                    pause = 1.0
                chunks.append(Chunk(part, pause))
    if chunks:
        chunks[-1].pause_after = 0.0
    return chunks


def prepare(text: str, markdown: bool = True, max_chars: int = 0) -> list[Chunk]:
    """Clean and split `text`."""
    text = clean_text(text, markdown)
    if max_chars and len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        text = text[: cut if cut > 0 else max_chars]
    return split_chunks(text)


def prepare_items(items: list[tuple[str, str]], markdown: bool = False) -> list[Chunk]:
    """Chunks of reader items, each tagged with its index in `items`. `items` are
    (text, pause) pairs; the pause after an item is a sentence ("sentence") or a paragraph
    ("paragraph") long. Items with nothing to say give no chunks; the other indices stay."""
    chunks: list[Chunk] = []
    for index, (text, pause) in enumerate(items):
        part = split_chunks(clean_text(text, markdown), short_first=not chunks)
        if not part:
            continue
        for c in part:
            c.tag = index
        part[-1].pause_after = 2.0 if pause == "paragraph" else 1.0
        chunks.extend(part)
    if chunks:
        chunks[-1].pause_after = 0.0
    return chunks



# ---------------------------------------------------------------- language guess

# English function words that are rare in the other 30 Supertonic languages (no "a", "i", "to",
# "on", "do", "so", "he", "me", "no", "as", "an", "in", "was": those are common words elsewhere)
_EN_WORDS = frozenset("""
the and of is are were been be it its for with at by this that these those from or but not
have has had you your he she we they my his her our their would could should can what which who
there here if then than very just about into when also because while where how why more some
all any one only other over such will don't it's i'm that's can't isn't i've you're we're
they're didn't doesn't won't there's what's let's
""".split())
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
EN_SHORT_WORDS = 4  # fewer words than this are too little to judge: counts as English


def word_count(text: str) -> int:
    """Number of words in `text` (letters only, as looks_english counts them)."""
    return len(_WORD.findall(text))


def _letter_kind(ch: str) -> int:
    """0 plain Latin (a-z), 1 Latin with a diacritic, 2 any other script."""
    o = ord(ch)
    if o < 0x80:
        return 0
    if 0xC0 <= o < 0x250 or 0x1E00 <= o < 0x1F00:
        return 1
    return 2


def looks_english(text: str) -> bool:
    """Cheap guess whether `text` is English (anything else gets the other-language voice).

    Not English: a good share of letters outside the Latin script (Cyrillic, Greek, Arabic,
    Hindi, CJK ...), or Latin text with non-English diacritics (č ć đ š ž ä ö ü é ñ ...) and
    few common English function words, or (without diacritics, so Croatian typed without
    them, Spanish ...) a long enough text with almost none. Short or ambiguous text counts
    as English, as does English text with a foreign name in it."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    kinds = [_letter_kind(c) for c in letters]
    other_script, accented = kinds.count(2), kinds.count(1)
    if other_script >= 2 and other_script / len(letters) > 0.25:
        return False
    words = [w.lower().replace("’", "'") for w in _WORD.findall(text)]
    if len(words) < EN_SHORT_WORDS:
        return not (accented >= 2 and not any(w in _EN_WORDS for w in words))
    ratio = sum(1 for w in words if w in _EN_WORDS) / len(words)
    if ratio >= 0.25:
        return True
    if accented:
        return ratio >= 0.15
    return len(words) < 6 or ratio >= 0.06
