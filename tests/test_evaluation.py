from src.evaluation import (
    bootstrap_ci_mean_difference,
    character_preservation_rate,
    compute_fertility,
    holm_correct,
    reduction_percentage,
    word_coverage,
)


def _char_tokenizer(word):
    """A trivial tokenizer (one token per character) used to make fertility
    arithmetic easy to check by hand."""
    return list(word)


def test_compute_fertility_matches_hand_calculation():
    words = ["घर", "मा"]  # 2 chars, 2 chars = 4 total tokens with char tokenizer; 2 words
    report = compute_fertility(words, _char_tokenizer)
    assert report.total_words == 2
    assert report.total_tokens == 4
    assert report.token_fertility == 2.0  # 4 tokens / 2 words
    assert report.chars_per_token == 1.0  # 4 chars / 4 tokens


def test_fertility_and_chars_per_token_are_distinct_metrics():
    words = ["घरमा"]  # 4 chars -> char tokenizer gives 4 tokens
    report = compute_fertility(words, _char_tokenizer)
    assert report.token_fertility == 4.0   # 4 tokens / 1 word
    assert report.chars_per_token == 1.0   # 4 chars / 4 tokens
    assert report.token_fertility != report.chars_per_token


def test_reduction_percentage_positive_means_fewer_tokens():
    assert reduction_percentage(100, 80) == 20.0
    assert reduction_percentage(100, 120) == -20.0
    assert reduction_percentage(0, 10) == 0.0


def test_character_preservation_rate():
    original = ["घर", "मा", "साथी"]
    reconstructed = ["घर", "मा", "XXX"]
    assert character_preservation_rate(original, reconstructed) == 2 / 3


def test_word_coverage():
    known = {"घर", "मा"}
    words = ["घर", "मा", "नदी"]
    rate = word_coverage(words, lambda w: w in known)
    assert rate == 2 / 3


def test_holm_correct_marks_strongest_signal_significant():
    p_values = [0.001, 0.2, 0.9]
    results = holm_correct(p_values, alpha=0.05)
    assert results[0]["rejected"] is True
    assert results[2]["rejected"] is False


def test_holm_correct_order_matches_input():
    p_values = [0.5, 0.001]
    results = holm_correct(p_values)
    assert results[1]["rank"] == 1  # the smaller p-value is ranked first
    assert results[0]["rank"] == 2


def test_bootstrap_ci_contains_observed_difference():
    a = [3, 3, 3, 3, 3, 3]
    b = [1, 1, 1, 1, 1, 1]
    result = bootstrap_ci_mean_difference(a, b, n_boot=200)
    assert result["observed_diff"] == 2.0
    assert result["ci_lower"] <= result["observed_diff"] <= result["ci_upper"]


def test_bootstrap_ci_requires_equal_length():
    import pytest
    with pytest.raises(AssertionError):
        bootstrap_ci_mean_difference([1, 2, 3], [1, 2])
