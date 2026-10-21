import pytest

pytest.importorskip("py3langid")

from vibetts.config import SUPERTONIC_LANGUAGES
from vibetts.langdetect import detect_language

from tests.test_textprep import ENGLISH, NOT_ENGLISH

SAMPLES = {
    "en": "The weather is lovely today, so we decided to walk through the park and have lunch.",
    "hr": "Danas je prekrasan dan pa smo odlučili prošetati parkom i zajedno ručati u gradu.",
    "de": "Das Wetter ist heute wunderbar, deshalb haben wir beschlossen, im Park spazieren zu gehen.",
    "fr": "Il fait très beau aujourd'hui, alors nous avons décidé de nous promener dans le parc.",
    "ru": "Сегодня прекрасная погода, поэтому мы решили погулять по парку и пообедать вместе.",
    "sl": "Danes je čudovit dan, zato smo se odločili, da se sprehodimo po parku in skupaj kosimo.",
    "cs": "Dnes je krásné počasí, a proto jsme se rozhodli projít se parkem a spolu poobědvat.",
    "es": "Hoy hace un día maravilloso, así que decidimos pasear por el parque y almorzar juntos.",
}


@pytest.mark.parametrize("lang", list(SAMPLES))
def test_detects_each_language(lang):
    assert detect_language(SAMPLES[lang], SUPERTONIC_LANGUAGES, "en") == lang


def test_shared_fixtures():
    assert detect_language(ENGLISH[0], SUPERTONIC_LANGUAGES, "hr") == "en"
    assert detect_language(NOT_ENGLISH["german"], SUPERTONIC_LANGUAGES, "en") == "de"
    assert detect_language(NOT_ENGLISH["russian"], SUPERTONIC_LANGUAGES, "en") == "ru"
    assert detect_language(NOT_ENGLISH["hindi"], SUPERTONIC_LANGUAGES, "en") == "hi"
    assert detect_language(NOT_ENGLISH["japanese"], SUPERTONIC_LANGUAGES, "en") == "ja"
    assert detect_language(NOT_ENGLISH["slovenian"], SUPERTONIC_LANGUAGES, "en") == "sl"


def test_restricted_candidates_are_honoured():
    # Croatian text with Croatian off: the closest enabled language wins, never an excluded one
    got = detect_language(SAMPLES["hr"], ["en", "de"], "en")
    assert got in ("en", "de")
    assert detect_language(SAMPLES["hr"], ["en", "hr"], "en") == "hr"
    assert detect_language(SAMPLES["de"], ["en", "fr"], "en") in ("en", "fr")
    assert detect_language(SAMPLES["fr"], ["de", "es", "fr"], "de") == "fr"


def test_single_or_no_candidate():
    assert detect_language(SAMPLES["de"], ["fr"], "en") == "fr"
    assert detect_language(SAMPLES["de"], [], "hr") == "hr"


def test_short_text_uses_fallback():
    assert detect_language("Guten Tag", ["en", "de"], "de") == "en"  # English is a candidate
    assert detect_language("Guten Tag", ["de", "fr"], "fr") == "fr"
    assert detect_language("", ["de", "fr"], "de") == "de"
    assert detect_language("12345 67 89", ["en", "de"], "en") == "en"


def test_failure_uses_fallback(monkeypatch):
    import vibetts.langdetect as ld

    def boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(ld, "_get_identifier", boom)
    assert detect_language(SAMPLES["de"], ["en", "de"], "en") == "en"
