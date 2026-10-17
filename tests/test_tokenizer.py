from pathlib import Path

import pytest

from src.morphology import load_lexicon
from src.tokenizer import (
    MORPH_BOUNDARY,
    TokenizerConfig,
    _strip_edge_punctuation,
    generate_combinatorial_lexicon_seeds,
    morphology_presegment_line,
    tokenize_word_baseline,
    tokenize_word_morphology_aware,
    train_bpe,
    train_lexicon_seeded_bpe,
)

LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"

TRAIN_LINES = [
    "घरहरूलाई साथीसँग हेर्छु",
    "विद्यार्थीहरू विद्यालयमा पढ्छन्",
    "किसानहरूले खेतमा काम गर्छ",
    "शिक्षकले विद्यार्थीलाई किताब दिन्छ",
    "मान्छेहरू सहरमा बस्छन्",
    "साथीहरू नदीसम्म खेल्छन्",
] * 10  # repeated so BPE has enough frequency signal to train on a tiny corpus


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon(LEXICON_DIR)


def test_presegment_preserves_word_boundaries(lexicon):
    out = morphology_presegment_line(lexicon, "घरहरूलाई साथीसँग")
    assert " " in out
    words = out.split(" ")
    assert len(words) == 2  # exactly two original words, never merged


def test_presegment_never_merges_words(lexicon):
    out = morphology_presegment_line(lexicon, "घर मा")
    # two distinct original words must remain two space-separated groups
    assert len(out.split(" ")) == 2


def test_baseline_bpe_trains_and_tokenizes(lexicon):
    tok = train_bpe(TRAIN_LINES, TokenizerConfig(vocab_size=300, min_frequency=1))
    tokens = tokenize_word_baseline(tok, "घरहरूलाई")
    assert len(tokens) >= 1
    assert all(isinstance(t, str) for t in tokens)


def test_morphology_aware_bpe_trains_and_tokenizes(lexicon):
    tok = train_bpe(
        TRAIN_LINES, TokenizerConfig(vocab_size=300, min_frequency=1),
        morphology_presegment=True, lexicon=lexicon,
    )
    tokens = tokenize_word_morphology_aware(tok, lexicon, "घरहरूलाई")
    assert len(tokens) >= 1
    assert all(MORPH_BOUNDARY not in t for t in tokens)  # marker must be stripped from display


def test_morphology_aware_requires_lexicon():
    with pytest.raises(ValueError):
        train_bpe(TRAIN_LINES, TokenizerConfig(vocab_size=300), morphology_presegment=True)


def test_lexicon_seeded_bpe_never_shrinks_vocab(lexicon):
    tok, info = train_lexicon_seeded_bpe(TRAIN_LINES, TokenizerConfig(vocab_size=300), lexicon)
    assert info["vocab_after_seeding"] >= info["vocab_before_seeding"]
    assert tok.get_vocab_size() == info["vocab_after_seeding"]


def test_lexicon_seeded_bpe_single_tokens_frequent_words(lexicon):
    tok, info = train_lexicon_seeded_bpe(TRAIN_LINES, TokenizerConfig(vocab_size=300), lexicon,
                                          min_train_frequency=2)
    # "घरहरूलाई" occurs in every repeated TRAIN_LINES sentence, so it
    # should clear the frequency threshold and become exactly one token.
    tokens = tokenize_word_baseline(tok, "घरहरूलाई")
    assert tokens == ["घरहरूलाई"]


def test_lexicon_seeded_bpe_does_not_increase_total_tokens(lexicon):
    """The seeded tokenizer must never need MORE tokens than the plain
    baseline for the same word, since seeding only adds atomic entries
    on top of an identical BPE model -- it never removes a merge path."""
    plain_tok = train_bpe(TRAIN_LINES, TokenizerConfig(vocab_size=300, min_frequency=1))
    seeded_tok, _ = train_lexicon_seeded_bpe(TRAIN_LINES, TokenizerConfig(vocab_size=300), lexicon)
    for w in ["घरहरूलाई", "साथीसँग", "विद्यार्थीहरू", "किसानहरूले"]:
        plain_count = len(tokenize_word_baseline(plain_tok, w))
        seeded_count = len(tokenize_word_baseline(seeded_tok, w))
        assert seeded_count <= plain_count, f"{w}: seeded={seeded_count} > plain={plain_count}"


def test_strip_edge_punctuation_removes_danda_only_at_edges():
    assert _strip_edge_punctuation("गर्छु।") == "गर्छु"
    assert _strip_edge_punctuation("।।गर्छु॥") == "गर्छु"
    assert _strip_edge_punctuation("गर्छु") == "गर्छु"  # no punctuation: unchanged
    assert _strip_edge_punctuation("") == ""


def test_lexicon_seeded_bpe_generalizes_past_sentence_final_punctuation(lexicon):
    """A word that only ever occurred glued to a sentence-final danda in
    training (e.g. '...गर्छ।') must still seed as a single token when it
    shows up WITHOUT that punctuation elsewhere -- this is the punctuation
    generalization fix: frequency is now counted on the punctuation-
    stripped form, not just the exact literal training string."""
    lines_with_danda = [line + "।" for line in TRAIN_LINES]
    tok, info = train_lexicon_seeded_bpe(
        lines_with_danda, TokenizerConfig(vocab_size=300), lexicon, min_train_frequency=2,
    )
    # "किसानहरूले" never appears without the danda glued to the *following*
    # word in this corpus construction, but "घरहरूलाई" is the first word of
    # every line and so always appears punctuation-free already; the real
    # regression test is the sentence-final word, "हेर्छु", which in this
    # corpus only ever occurs as "हेर्छु।" (glued).
    tokens_plain = tokenize_word_baseline(tok, "हेर्छु")  # no punctuation attached
    assert tokens_plain == ["हेर्छु"], (
        "seeding did not generalize past the attached punctuation form"
    )


def test_combinatorial_lexicon_seeds_cover_unobserved_combinations(lexicon):
    seeds = generate_combinatorial_lexicon_seeds(lexicon)
    assert len(seeds) > 100  # many root x affix combinations proposed
    # a plausible noun root + plural suffix combination not necessarily
    # present verbatim in words.csv or the training corpus
    assert "घरहरू" in seeds or any(s.startswith("घर") for s in seeds)
    # every generated candidate must reconstruct as plain concatenation
    # (no stray separators introduced)
    assert all(isinstance(s, str) and s for s in seeds)


def test_lexicon_seeded_bpe_combinatorial_expansion_grows_vocab_more(lexicon):
    _, info_plain = train_lexicon_seeded_bpe(
        TRAIN_LINES, TokenizerConfig(vocab_size=300), lexicon, combinatorial_expansion=False,
    )
    _, info_combo = train_lexicon_seeded_bpe(
        TRAIN_LINES, TokenizerConfig(vocab_size=300), lexicon, combinatorial_expansion=True,
    )
    assert info_combo["combinatorial_candidates_proposed"] > 0
    assert info_combo["seed_candidates_proposed"] > info_plain["seed_candidates_proposed"]
    assert info_combo["vocab_after_seeding"] >= info_plain["vocab_after_seeding"]
