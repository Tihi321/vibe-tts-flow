"""Language detection for one read: which of the enabled languages is `text` in (py3langid)."""
from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

from .textprep import EN_SHORT_WORDS, word_count

log = logging.getLogger(__name__)

SHORT_LETTERS = 15  # unspaced scripts (Japanese ...) have few 'words' but plenty of letters
_lock = threading.Lock()
_identifier: Any = None
_known: frozenset[str] = frozenset()  # codes py3langid knows (its model classes), set on load


def _get_identifier() -> Any:
    """The shared LanguageIdentifier, loaded on first use (the import and model load are slow)."""
    global _identifier, _known
    if _identifier is None:
        from py3langid.langid import MODEL_FILE, LanguageIdentifier

        ident = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
        _known = frozenset(ident._full_model[2])
        _identifier = ident
    return _identifier


def detectable_languages() -> frozenset[str]:
    """The language codes py3langid knows (its model classes); loads the model on first use.
    Other languages (ceb, kab, ...) are never detected: they need Read "Always X" or `lang`."""
    with _lock:
        _get_identifier()
        return _known


def detect_language(text: str, candidates: Sequence[str], fallback: str) -> str:
    """The language of `text` among `candidates` (language codes), decided once for the whole
    text. Candidates py3langid does not know are ignored: those languages are reached only by
    Read "Always X" or an explicit `lang`. `fallback` when there is no candidate, the text is too short to judge (fewer than
    EN_SHORT_WORDS words and SHORT_LETTERS letters; "en" if it is a candidate) or detection fails."""
    cands = list(dict.fromkeys(candidates))
    if not cands:
        return fallback
    if len(cands) == 1:
        return cands[0]
    if word_count(text) < EN_SHORT_WORDS and sum(c.isalpha() for c in text) < SHORT_LETTERS:
        return "en" if "en" in cands else fallback
    try:
        with _lock:
            ident = _get_identifier()
            use = [c for c in cands if c in _known]  # the rest (ceb, kab, ...) are never detected
            if not use:
                return fallback
            ident.set_languages(use)
            lang, prob = ident.classify(text)
    except Exception:
        log.warning("language detection failed; using %s", fallback, exc_info=True)
        return fallback
    log.debug("detected language %s (%.2f) among %s", lang, prob, ",".join(cands))
    return lang if lang in cands else fallback
