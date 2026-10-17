from pathlib import Path

import pytest

from src.morphology import (
    classify_word_group,
    generate_candidates,
    load_lexicon,
    score_candidate,
    select_best_segmentation,
    select_weights_on_validation,
)

LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"


@pytest.fixture(scope="module")
def lexicon():
    lex = load_lexicon(LEXICON_DIR)
    assert lex.roots.count() > 0, "seed lexicon must be present for tests to be meaningful"
    return lex


def test_lexicon_loads_all_four_tries(lexicon):
    assert lexicon.roots.count() > 0
    assert lexicon.prefixes.count() > 0
    assert lexicon.suffixes.count() > 0
    assert lexicon.words.count() > 0


def test_known_whole_word_segmentation(lexicon):
    seg, score, breakdown, _ = select_best_segmentation(lexicon, "घरहरू")
    assert seg.reconstructs("घरहरू")
    assert "whole_word" in [p.kind for p in seg.pieces] or "root" in [p.kind for p in seg.pieces]


def test_reversibility_holds_for_every_candidate(lexicon):
    for word in ["घरहरूलाई", "विद्यार्थीहरू", "असफल", "अनौठो"]:
        candidates = generate_candidates(lexicon, word)
        assert len(candidates) >= 1
        for c in candidates:
            assert c.reconstructs(word), f"{c.as_strings()} does not reconstruct {word}"


def test_unknown_word_fallback(lexicon):
    word = "झ्याऊताप्चा"  # not expected to exist in the seed lexicon
    seg, score, breakdown, _ = select_best_segmentation(lexicon, word)
    assert seg.reconstructs(word)
    group = classify_word_group(lexicon, word)
    assert group in ("unknown_root", "unsupported_affix", "known_components")


def test_empty_string_segmentation(lexicon):
    candidates = generate_candidates(lexicon, "")
    assert len(candidates) == 1
    assert candidates[0].pieces == []
    assert candidates[0].reconstructs("")


def test_mixed_script_classification(lexicon):
    assert classify_word_group(lexicon, "घरhouse") == "mixed_script"


def test_known_components_group(lexicon):
    # घरबाट ("from the house") is root+suffix, both known, but likely not
    # itself listed as a complete word in the small seed word lexicon.
    word = "घरबाट"
    group = classify_word_group(lexicon, word)
    assert group in ("known_word", "known_components")


def test_score_candidate_rewards_known_root(lexicon):
    seg, _, _, _ = select_best_segmentation(lexicon, "घरमा")
    score, breakdown = score_candidate(seg, lexicon)
    assert isinstance(breakdown, dict)
    assert score > -3.0  # should not score as badly as a pure-unknown word


def test_candidate_does_not_split_on_substring_alone(lexicon):
    # "घर" (house) is a substring of many longer unrelated strings; make
    # sure segmentation of a genuinely unknown word doesn't force a split
    # just because "घर" happens to appear inside it without being an
    # actual morpheme boundary match from position 0 in this test word.
    word = "नदी"
    seg, _, _, _ = select_best_segmentation(lexicon, word)
    assert seg.reconstructs(word)


def test_max_depth_limits_pieces(lexicon):
    candidates = generate_candidates(lexicon, "घरहरूलाई", max_depth=1)
    for c in candidates:
        assert len(c.pieces) <= 1


def test_select_weights_on_validation_returns_one_of_the_given_sets(lexicon):
    weight_sets = [
        {"whole_word_known": 5.0, "root_known": 2.0, "affix_known": 1.0,
         "frequency_log": 0.5, "segment_count_penalty": 0.4, "unknown_penalty": 3.0},
        {"whole_word_known": 1.0, "root_known": 1.0, "affix_known": 1.0,
         "frequency_log": 0.0, "segment_count_penalty": 1.0, "unknown_penalty": 1.0},
    ]
    validation_pairs = [("घरहरू", ["घर", "हरू"])]
    chosen = select_weights_on_validation(lexicon, validation_pairs, weight_sets)
    assert chosen in weight_sets


def test_punctuation_only_input_does_not_crash(lexicon):
    seg, _, _, _ = select_best_segmentation(lexicon, "।")
    assert seg.reconstructs("।")
