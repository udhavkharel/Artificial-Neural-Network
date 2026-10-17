from pathlib import Path

import pytest

from src.morphology import load_lexicon
from src.optimized_tokenizer import (
    OptimizedMorphTokenizer,
    pretokenize,
    train_optimized_tokenizer,
)
from src.tokenizer import TokenizerConfig, tokenize_word_baseline, train_bpe

LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"

LINES = [
    "घरहरूलाई साथीसँग हेर्छु।",
    "विद्यार्थीहरू विद्यालयमा पढ्छन्।",
    "किसानहरूले खेतमा काम गर्छ।",
    "शिक्षकले विद्यार्थीलाई किताब दिन्छ।",
    "मान्छेहरू सहरमा बस्छन्।",
    "साथीहरू नदीसम्म खेल्छन्।",
] * 20


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon(LEXICON_DIR)


def test_min_token_encoding_prefers_fewest_tokens():
    tok = OptimizedMorphTokenizer({"घर", "हरू", "लाई", "घरहरू", "घ", "र"})
    assert tok.encode_word("घरहरूलाई") == ["घरहरू", "लाई"]


def test_reversible_even_with_unknown_characters():
    tok = OptimizedMorphTokenizer({"घर"})
    word = "घरXz"
    assert "".join(tok.encode_word(word)) == word


def test_morpheme_tiebreak():
    tok = OptimizedMorphTokenizer({"घर", "हरू", "घरह", "रू"}, morphemes={"घर", "हरू"})
    assert tok.encode_word("घरहरू") == ["घर", "हरू"]


def test_vocab_budget_is_respected(lexicon):
    cfg = TokenizerConfig(vocab_size=300, min_frequency=1)
    tok, info = train_optimized_tokenizer(LINES, cfg, lexicon)
    assert tok.vocab_size() <= 300


def test_never_worse_than_baseline_at_same_budget(lexicon):
    cfg = TokenizerConfig(vocab_size=300, min_frequency=1)
    base = train_bpe(LINES, cfg)
    tok, _ = train_optimized_tokenizer(LINES, cfg, lexicon)
    words = [w for l in LINES for w in l.split()]
    n_base = sum(len(tokenize_word_baseline(base, w)) for w in words)
    n_opt = sum(len(tok.tokenize(w)) for w in words)
    assert n_opt <= n_base


def test_save_load_roundtrip(tmp_path):
    tok = OptimizedMorphTokenizer({"घर", "हरू"}, {"घर"})
    tok.save(tmp_path / "t.json")
    loaded = OptimizedMorphTokenizer.load(tmp_path / "t.json")
    assert loaded.tokenize("घरहरू") == tok.tokenize("घरहरू")


def test_pretokenize_keeps_punctuation_attached_and_pieces_reversible():
    assert pretokenize("घर। मा") == ["घर।", "मा"]
    tok = OptimizedMorphTokenizer({"घर", "।"})
    assert tok.tokenize("घर।") == ["घर", "।"]
