"""
Evaluation metrics: token fertility, characters-per-token, coverage,
unknown-token rate, ablation-study scoring, bootstrap confidence
intervals, and lookup-method runtime benchmarking (the latter delegates to
src/trie_index.benchmark_lookup_methods).

Two metrics are defined here and are never conflated anywhere in this
project, per the specification's explicit warning:

    token fertility        = total subword tokens / total whitespace-
                              separated words
    characters per token   = total normalized text characters / total
                              subword tokens

Token fertility answers "how many pieces does the tokenizer cut each word
into, on average" — this is the metric the project is trying to reduce.
Characters-per-token answers "how much text, in characters, does one
token typically cover" — a different question (a tokenizer could have low
fertility by producing very long multi-word tokens that still rate badly
on this second metric, or vice versa), reported alongside fertility for a
fuller picture but never used as if it were the same number.

Special tokens (<unk>, <pad>, <s>, </s>) are excluded from both metrics'
token counts by default, since they are not substantive content —
documented exceptions (reporting with them included) are explicitly
labeled when used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.normalization import character_count, whitespace_words

SPECIAL_TOKENS = {"<unk>", "<pad>", "<s>", "</s>"}


@dataclass
class FertilityReport:
    total_tokens: int
    total_words: int
    total_chars: int
    token_fertility: float
    chars_per_token: float
    mean_tokens_per_word: float
    median_tokens_per_word: float
    p95_tokens_per_word: float
    tokens_per_char: float
    unknown_token_rate: float
    per_word_token_counts: list[int] = field(default_factory=list)


def exclude_special(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in SPECIAL_TOKENS]


def compute_fertility(
    words: list[str], tokenize_fn, include_special: bool = False,
) -> FertilityReport:
    """
    Compute the full fertility report for a list of whitespace-separated
    `words`, each tokenized independently by `tokenize_fn(word) -> list[str]`.
    Tokenizing word-by-word (rather than whole lines) keeps the
    denominator ("total words") unambiguous and matches how the
    per-word distributions (mean/median/p95 tokens-per-word) are defined.
    """
    per_word_counts: list[int] = []
    total_tokens = 0
    total_chars = 0
    unknown_count = 0

    for w in words:
        tokens = tokenize_fn(w)
        if not include_special:
            tokens = exclude_special(tokens)
        per_word_counts.append(len(tokens))
        total_tokens += len(tokens)
        total_chars += character_count(w)
        unknown_count += sum(1 for t in tokens if t == "<unk>")

    total_words = len(words)
    token_fertility = total_tokens / total_words if total_words else 0.0
    chars_per_token = total_chars / total_tokens if total_tokens else 0.0
    tokens_per_char = total_tokens / total_chars if total_chars else 0.0
    arr = np.array(per_word_counts) if per_word_counts else np.array([0])

    return FertilityReport(
        total_tokens=total_tokens,
        total_words=total_words,
        total_chars=total_chars,
        token_fertility=token_fertility,
        chars_per_token=chars_per_token,
        mean_tokens_per_word=float(np.mean(arr)),
        median_tokens_per_word=float(np.median(arr)),
        p95_tokens_per_word=float(np.percentile(arr, 95)),
        tokens_per_char=tokens_per_char,
        unknown_token_rate=unknown_count / total_tokens if total_tokens else 0.0,
        per_word_token_counts=per_word_counts,
    )


def reduction_percentage(baseline_tokens: int, proposed_tokens: int) -> float:
    """100 * (baseline - proposed) / baseline. Positive = fewer tokens
    (an improvement); negative = more tokens (a regression)."""
    if baseline_tokens == 0:
        return 0.0
    return 100.0 * (baseline_tokens - proposed_tokens) / baseline_tokens


def character_preservation_rate(original_words: list[str], reconstructed_words: list[str]) -> float:
    """Fraction of words whose concatenated tokens reconstruct the
    original word exactly (reversibility at the whole-corpus level)."""
    if not original_words:
        return 1.0
    matches = sum(1 for o, r in zip(original_words, reconstructed_words) if o == r)
    return matches / len(original_words)


def word_coverage(words: list[str], known_fn) -> float:
    """Fraction of `words` for which `known_fn(word)` is True (e.g. the
    word or all its morphological pieces are present in the lexicon)."""
    if not words:
        return 1.0
    return sum(1 for w in words if known_fn(w)) / len(words)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci_mean_difference(
    sample_a: list[int], sample_b: list[int], n_boot: int = 2000, ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Paired bootstrap confidence interval for the mean difference
    (mean(sample_a) - mean(sample_b)), e.g. per-word token counts for
    baseline vs. morphology-aware tokenizers on the same held-out words.
    Resamples paired indices with replacement `n_boot` times. Returns the
    observed difference and the (lower, upper) percentile CI bounds.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    n = len(a)
    assert n == len(b), "paired bootstrap requires equal-length samples"

    observed_diff = float(np.mean(a) - np.mean(b))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(a[idx]) - np.mean(b[idx])

    alpha = 1 - ci
    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {"observed_diff": observed_diff, "ci_lower": lower, "ci_upper": upper,
            "n_boot": n_boot, "ci_level": ci}


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction (manual implementation; no statsmodels dependency)
# ---------------------------------------------------------------------------

def holm_correct(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.
    Returns a list (in the original input order) of dicts with the
    original p-value, its rank, the Holm-adjusted alpha threshold, and
    whether it's rejected (significant) under the correction.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    results = [None] * m
    still_significant = True
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        p = p_values[idx]
        if not still_significant or p > threshold:
            still_significant = False
            rejected = False
        else:
            rejected = True
        results[idx] = {
            "p_value": p, "rank": rank + 1, "threshold": threshold,
            "rejected": rejected,
        }
    return results


# ---------------------------------------------------------------------------
# Runtime benchmarking per document / per 1000 characters
# ---------------------------------------------------------------------------

def benchmark_tokenizer_runtime(tokenize_fn, documents: list[str]) -> dict[str, float]:
    """
    Measure wall-clock time to tokenize each document in `documents` with
    `tokenize_fn(text) -> list[str]`, and report total time, time per
    document, and time per 1000 characters of input.
    """
    total_chars = sum(len(d) for d in documents)
    t0 = time.perf_counter()
    for d in documents:
        _ = tokenize_fn(d)
    elapsed = time.perf_counter() - t0
    n_docs = max(1, len(documents))
    return {
        "total_seconds": elapsed,
        "seconds_per_document": elapsed / n_docs,
        "seconds_per_1000_chars": (elapsed / total_chars * 1000) if total_chars else 0.0,
        "n_documents": len(documents),
        "total_chars": total_chars,
    }
