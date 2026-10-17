import unicodedata

from src.normalization import (
    character_count,
    is_devanagari_word,
    is_mixed_script,
    normalize_text,
    strip_punctuation_for_tokenization,
    whitespace_words,
)


def test_nfc_normalization_idempotent():
    text = "नेपाली भाषा।"
    normalized = normalize_text(text)
    assert normalized == unicodedata.normalize("NFC", normalized)
    assert normalize_text(normalized) == normalized


def test_vowel_signs_preserved():
    # मा (consonant + dependent vowel sign आ) must survive normalization intact.
    word = "घरमा"
    out = normalize_text(word)
    assert out == word
    assert len(out) == len(word)


def test_conjunct_consonants_preserved():
    # विद्यार्थी contains conjuncts (द्य, र्थ) built with virama (्).
    word = "विद्यार्थी"
    out = normalize_text(word)
    assert out == word


def test_whitespace_collapsed():
    text = "घर   मा   साथी"
    assert normalize_text(text) == "घर मा साथी"


def test_leading_trailing_whitespace_stripped():
    assert normalize_text("   घर   ") == "घर"


def test_empty_and_none_input():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_punctuation_not_removed_by_normalize_text():
    text = "नमस्ते।"
    assert normalize_text(text) == "नमस्ते।"


def test_strip_punctuation_for_tokenization_removes_danda():
    text = "नमस्ते।"
    out = strip_punctuation_for_tokenization(text)
    assert "।" not in out
    assert "नमस्ते" in out


def test_digits_preserved():
    assert normalize_text("२०२५ मा") == "२०२५ मा"
    assert normalize_text("2025 मा") == "2025 मा"


def test_is_devanagari_word():
    assert is_devanagari_word("घर") is True
    assert is_devanagari_word("house") is False
    assert is_devanagari_word("घरhouse") is False
    assert is_devanagari_word("") is False
    assert is_devanagari_word("123") is False


def test_is_mixed_script():
    assert is_mixed_script("घरhouse") is True
    assert is_mixed_script("घर") is False
    assert is_mixed_script("house") is False


def test_whitespace_words_simple_split():
    assert whitespace_words("घर मा साथी") == ["घर", "मा", "साथी"]
    assert whitespace_words("") == []


def test_character_count_counts_everything():
    text = "घर मा"
    assert character_count(text) == len(text)
