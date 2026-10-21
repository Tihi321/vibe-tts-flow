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


def _get_identifier() -> Any:
    """The shared LanguageIdentifier, loaded on first use (the import and model load are slow)."""
    global _identifier
    if _identifier is None:
        from py3langid.langid import MODEL_FILE, LanguageIdentifier

        _identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    return _identifier


def detect_language(text: str, candidates: Sequence[str], fallback: str) -> str:
    """The language of `text` among `candidates` (language codes), decided once for the whole
    text. `fallback` when there is no candidate, the text is too short to judge (fewer than
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
            known = set(ident._full_model[2])
            use = [c for c in cands if c in known]
            if not use:
                return fallback
            ident.set_languages(use)
            lang, prob = ident.classify(text)
    except Exception:
        log.warning("language detection failed; using %s", fallback, exc_info=True)
        return fallback
    log.debug("detected language %s (%.2f) among %s", lang, prob, ",".join(cands))
    return lang if lang in cands else fallback
