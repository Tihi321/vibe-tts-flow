from vibetts.textprep import FIRST_CHUNK_CHARS, clean_text, looks_croatian, prepare, split_chunks


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


def test_url_word_follows_language():
    text = "Vidi https://example.com/a za više."
    assert "link" in clean_text(text) and "poveznica" not in clean_text(text)
    assert "poveznica" in clean_text(text, True, "hr")
    assert "poveznica" in prepare(text, lang="hr")[0].text
    assert "link" in prepare(text)[0].text  # default stays English


def test_looks_croatian_with_diacritics():
    assert looks_croatian("Dobar dan, ovo je test čitanja na hrvatskom jeziku.")
    assert looks_croatian("Žaba šuti đače")


def test_looks_croatian_without_diacritics():
    assert looks_croatian("Ovo je jednostavna recenica koja se cita na hrvatskom jeziku.")
    assert looks_croatian("Nije sve tako crno kako izgleda, ali nema problema.")


def test_looks_croatian_english_is_not():
    assert not looks_croatian("This is a plain English sentence about nothing in particular.")
    assert not looks_croatian("I think I am going to see you in the U.S. soon, so I will call.")


def test_looks_croatian_mixed_short_english():
    assert not looks_croatian("Meet me at the Šimić cafe")  # one stray letter, English words
    assert not looks_croatian("OK, I do")
    assert not looks_croatian("Yes. I am.")


def test_looks_croatian_empty():
    assert not looks_croatian("")
    assert not looks_croatian("   \n ")
    assert not looks_croatian("123 456 !!!")


def test_long_chunks_merge_after_the_first():
    text = "Dobar dan. " + " ".join(f"Ovo je kratka recenica broj {i}." for i in range(8))
    en, hr = prepare(text), prepare(text, long_chunks=True)
    assert hr[0].text == en[0].text  # the first chunk stays short: audio starts quickly
    assert len(hr) < len(en)
    assert all(len(c.text) <= 260 for c in hr)
    assert " ".join(c.text for c in hr) == " ".join(c.text for c in en)


def test_long_chunks_do_not_depend_on_language():
    text = "Hello. " + " ".join(f"This is short sentence number {i}." for i in range(8))
    assert len(prepare(text, long_chunks=True)) < len(prepare(text))
    assert prepare(text, lang="hr") == prepare(text)  # language only picks the URL word
