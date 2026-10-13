"""Text preparation for speech: markdown cleanup and splitting into short chunks.

Chunks are spoken one after another while the next one is generated, so the first
chunk is kept short: that is what sets how soon reading starts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

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


def split_chunks(text: str) -> list[Chunk]:
    """Split cleaned text into speakable chunks (sentences; line breaks also end one)."""
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
            if merged and len(merged[-1]) < MERGE_BELOW_CHARS and \
                    len(merged[-1]) + 1 + len(s) <= MERGE_MAX_CHARS:
                merged[-1] = f"{merged[-1]} {s}"
            else:
                merged.append(s)
        for si, s in enumerate(merged):
            parts = _split_long(s, MAX_CHUNK_CHARS)
            if not chunks:
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
    text = clean_text(text, markdown)
    if max_chars and len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        text = text[: cut if cut > 0 else max_chars]
    return split_chunks(text)
