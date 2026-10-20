import pytest

from vibetts.textprep import (FIRST_CHUNK_CHARS, Chunk, clean_text, looks_english, prepare,
                              prepare_items, split_chunks)


def test_markdown_cleanup():
    md = ("# Title\n\nSome **bold** and *italic* and `code` text.\n\n"
          "- item one\n* item two\n> quoted\n\n[a link](https://x.com) and ![img](p.png)\n"
          "see https://example.com/page?q=1 now")
    out = clean_text(md)
    for bad in ("#", "**", "`", "](", "https://", "> "):
        assert bad not in out
    assert "Title" in out and "bold" in out and "italic" in out and "code" in out
    assert "item one\nitem two\nquoted" in out
    assert "a link and img" in out
    assert "see link now" in out


def test_snake_case_and_math_survive():
    assert clean_text("call snake_case_name now") == "call snake_case_name now"
    assert clean_text("_emphasis_ here") == "emphasis here"


def test_no_markdown_mode_keeps_symbols():
    assert clean_text("**bold** # x", markdown=False) == "**bold** # x"


def test_whitespace_and_paragraphs():
    assert clean_text("a\r\n  b\t c\n\n\n\nd") == "a\nb c\n\nd"


def test_table_becomes_commas():
    out = clean_text("| a | b |\n|---|---|\n| 1 | 2 |")
    assert out == "a, b\n1, 2"


def test_first_chunk_short_at_clause():
    text = ("When you select a long first sentence that keeps going and going, with a comma "
            "somewhere in the middle of it, the first piece is cut there. Then the rest.")
    chunks = prepare(text)
    assert len(chunks[0].text) <= FIRST_CHUNK_CHARS
    assert chunks[0].text.endswith(",")
    assert chunks[0].pause_after < 1.0
    assert " ".join(c.text for c in chunks) == text


def test_first_chunk_never_cut_mid_phrase():
    text = "word " * 40 + "end."
    chunks = prepare(text.strip())
    assert len(chunks) == 1


def test_short_sentences_merge_but_stay_short():
    chunks = prepare("Hi! OK. This is a much longer sentence that should stand on its own "
                     "because merging it would make the first chunk too long to start fast.")
    assert chunks[0].text == "Hi! OK."
    assert len(chunks) == 2


def test_paragraph_pause_and_last_pause():
    chunks = split_chunks("One here.\n\nTwo there.\nThree.")
    assert [c.text for c in chunks] == ["One here.", "Two there. Three."]
    assert chunks[0].pause_after == 2.0
    assert chunks[-1].pause_after == 0.0


def test_lines_without_punctuation_are_separate():
    chunks = split_chunks("A heading without a period that is long enough\nThen body text "
                          "that is also long enough to not be merged at all.")
    assert len(chunks) == 2


def test_long_sentence_split_under_limit():
    long = ("alpha beta gamma delta " * 40).strip() + "."
    chunks = prepare("Short start. " + long)
    assert all(len(c.text) <= 400 for c in chunks)


def test_max_chars_and_empty():
    chunks = prepare("word " * 100, max_chars=50)
    assert sum(len(c.text) for c in chunks) <= 50
    assert prepare("   \n\n ") == []
    assert prepare("**") == []


def test_url_is_read_as_link():
    text = "See https://example.com/a for more."
    assert clean_text(text) == "See link for more."
    assert "link" in prepare(text)[0].text


# ---------------------------------------------------------------- looks_english

ENGLISH = [
    "This is a simple test of the text to speech system, and it should be read in English.",
    "I want to go to the store and buy some milk, but it is closed today.",
    "It is what it is.",
    "Please send the report to Željko Čović before the meeting tomorrow.",  # a Croatian name
    "We met with Dr. Šimić and Mrs. Kovačević at the conference, and they were very kind.",
    "The café near the station serves what you would expect from a naïve résumé.",
    "Run the build script and check the output logs for errors.",
    "Hello there",  # short: counts as English
    "OK",
    "",
    "12345 67 89",
    "Visit https://example.com/page for more information about the project.",
]

NOT_ENGLISH = {
    "croatian": "Ovo je jednostavan test sustava za pretvaranje teksta u govor, čitat će na hrvatskom jeziku.",
    "croatian no diacritics": "Ovo je jednostavan test sustava za pretvaranje teksta u govor koji cita hrvatski tekst.",
    "croatian short": "Što radiš danas?",
    "slovenian": "To je preprost preizkus sistema za pretvorbo besedila v govor, ki bere slovensko besedilo.",
    "slovenian diacritics": "Danes je lep dan in šli bomo na sprehod v gozd, ker je toplo.",
    "german": "Das ist ein einfacher Test des Systems, der den deutschen Text vorliest, weil es schön ist.",
    "spanish": "Este es un sencillo test del sistema que lee el texto en español para todos los usuarios.",
    "russian": "Это простой тест системы преобразования текста в речь на русском языке.",
    "ukrainian": "Це простий тест системи перетворення тексту в мовлення українською мовою.",
    "greek": "Αυτή είναι μια απλή δοκιμή του συστήματος που διαβάζει κείμενο.",
    "hindi": "यह पाठ को वाक में बदलने वाली प्रणाली का एक सरल परीक्षण है।",
    "japanese": "これはテキストを音声に変換するシステムの簡単なテストです。",
    "russian short": "Привет, мир",
}


@pytest.mark.parametrize("text", ENGLISH)
def test_looks_english(text):
    assert looks_english(text) is True, text


@pytest.mark.parametrize("name", list(NOT_ENGLISH))
def test_does_not_look_english(name):
    assert looks_english(NOT_ENGLISH[name]) is False, name


def test_looks_english_mostly_foreign_script_with_english_name():
    assert looks_english("Привет, Anna!") is False
    assert looks_english("Reading the word привет is easy for the team.") is True  # one foreign word


def test_looks_english_ignores_markup_and_case():
    assert looks_english("# THE TITLE\n\n- the first item and the second item") is True
    assert looks_english("**Ovo** je _tekst_ koji je pisan na hrvatskom jeziku, bez engleskog.") is False


# ---------------------------------------------------------------- reader items

def test_chunk_tag_defaults_to_none():
    assert Chunk("x").tag is None and Chunk("x", 1.0, 3).tag == 3


def test_prepare_items_tags_and_pauses():
    chunks = prepare_items([("First one.", "sentence"), ("Second one here.", "paragraph"),
                            ("Third one.", "sentence")])
    assert [(c.text, c.tag) for c in chunks] == [("First one.", 0), ("Second one here.", 1),
                                                 ("Third one.", 2)]
    assert [c.pause_after for c in chunks] == [1.0, 2.0, 0.0]  # the very last one has none


def test_prepare_items_several_chunks_in_one_item():
    chunks = prepare_items([("This sentence is long enough to stand alone here. "
                             "Another sentence follows it right after that.", "paragraph"),
                            ("Next.", "sentence")])
    assert [c.tag for c in chunks] == [0, 0, 1]
    assert [c.pause_after for c in chunks] == [1.0, 2.0, 0.0]  # only the item's last chunk


def test_prepare_items_skips_unspeakable_items_without_shifting_indices():
    chunks = prepare_items([("---", "sentence"), ("", "sentence"), ("Real text here.", "sentence")])
    assert [(c.text, c.tag) for c in chunks] == [("Real text here.", 2)]
    assert prepare_items([("...", "sentence")]) == [] and prepare_items([]) == []


def test_prepare_items_short_first_chunk_only_for_the_first_chunk():
    long = ("Alpha beta gamma delta, " * 6).strip() + "."  # longer than FIRST_CHUNK_CHARS
    assert len(long) > FIRST_CHUNK_CHARS
    chunks = prepare_items([(long, "sentence"), (long, "sentence")])
    first, second = [c for c in chunks if c.tag == 0], [c for c in chunks if c.tag == 1]
    assert len(first) > 1 and len(first[0].text) <= FIRST_CHUNK_CHARS
    assert len(second) == 1  # not cut at a clause mark
    skipped = prepare_items([("", "sentence"), (long, "sentence")])
    assert len(skipped) > 1 and skipped[0].tag == 1  # the first speakable chunk is the short one


def test_prepare_items_markdown_off_by_default():
    assert prepare_items([("A *star* here.", "sentence")])[0].text == "A *star* here."
    assert prepare_items([("A *star* here.", "sentence")], markdown=True)[0].text == "A star here."
