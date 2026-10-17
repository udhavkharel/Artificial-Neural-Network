"""
A Trie (prefix tree) for indexed lookup of lexicon entries (roots,
prefixes, suffixes, or whole words).

Why a Trie here: the segmentation engine in morphology.py repeatedly asks
two kinds of question while scanning a word left to right — "is this exact
substring a known unit?" and "is this substring at least a *prefix* of
some known unit, so it's worth extending further?" A plain dict/hash-map
answers the first question in O(1) but cannot answer the second without
scanning every key; a Trie answers both by walking one node per character,
and `prefix_matches` (listing every known entry along the path from the
root) comes essentially free once the tree is built. The actual
performance difference measured against a linear scan and a dict is
reported in outputs/tables/trie_benchmark.csv and outputs/figures/ — see
`benchmark_lookup_methods` below — rather than assumed.

Each inserted key can carry an arbitrary metadata dict (e.g. the lexicon
row for that surface form: root, category, frequency, source, etc.),
retrievable with `get_metadata`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    is_end: bool = False
    metadata: dict[str, Any] | None = None


class Trie:
    """A character-level Trie supporting insert, exact lookup, prefix
    lookup, metadata retrieval, and a count of stored keys."""

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._count = 0

    def insert(self, key: str, metadata: dict[str, Any] | None = None) -> None:
        """Insert `key` into the trie. If `key` is already present, its
        metadata is updated (overwritten) rather than duplicated, and the
        key count does not increase."""
        if key == "":
            return
        node = self._root
        for ch in key:
            node = node.children.setdefault(ch, _TrieNode())
        if not node.is_end:
            self._count += 1
        node.is_end = True
        node.metadata = metadata

    def exists(self, key: str) -> bool:
        """True if `key` was inserted as a complete entry (not merely a
        prefix of a longer entry)."""
        node = self._walk(key)
        return node is not None and node.is_end

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Return the metadata dict stored for `key`, or None if `key` is
        not a complete entry."""
        node = self._walk(key)
        if node is not None and node.is_end:
            return node.metadata
        return None

    def starts_with(self, prefix: str) -> bool:
        """True if any stored key starts with `prefix` (including the key
        being exactly equal to `prefix`)."""
        return self._walk(prefix) is not None

    def prefix_matches(self, text: str) -> list[str]:
        """
        Return every complete key that is a prefix of `text`, in
        increasing length order, by walking `text` once. For example, if
        the trie holds {"घर", "घरमा"} and text is "घरमाथि", this returns
        ["घर", "घरमा"].
        """
        matches: list[str] = []
        node = self._root
        for i, ch in enumerate(text):
            if ch not in node.children:
                break
            node = node.children[ch]
            if node.is_end:
                matches.append(text[: i + 1])
        return matches

    def longest_prefix_match(self, text: str) -> str | None:
        """Return the single longest complete key that is a prefix of
        `text`, or None if there is no match. Used by the greedy
        longest-match baseline in the ablation study."""
        matches = self.prefix_matches(text)
        return matches[-1] if matches else None

    def count(self) -> int:
        """Number of distinct complete keys stored."""
        return self._count

    def all_keys(self) -> list[str]:
        """Return every complete key stored, for debugging/tests only.
        Not used on a hot path."""
        out: list[str] = []

        def _walk_all(node: _TrieNode, prefix: str) -> None:
            if node.is_end:
                out.append(prefix)
            for ch, child in node.children.items():
                _walk_all(child, prefix + ch)

        _walk_all(self._root, "")
        return out

    def _walk(self, key: str) -> _TrieNode | None:
        node = self._root
        for ch in key:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node


# ---------------------------------------------------------------------------
# Benchmark: Trie vs. linear scan vs. dict/hash-map lookup.
#
# This measures actual wall-clock time for repeated "does this exact key
# exist" queries over the same vocabulary, using the three lookup methods.
# A Trie is not categorically faster than a dict for exact-match lookup
# (a dict's O(1) hashing is typically faster in practice for exact lookup
# in Python); the Trie's advantage is prefix-matching, which a dict cannot
# do without a full scan. This benchmark deliberately measures both query
# types so the comparison is honest rather than one-sided.
# ---------------------------------------------------------------------------

def benchmark_lookup_methods(
    vocabulary: list[str], queries: list[str], repeats: int = 3
) -> dict[str, dict[str, float]]:
    """
    Build a Trie, a dict, and keep the vocabulary as a list (linear scan),
    then time `repeats` passes of:
      (a) exact-match lookup of every string in `queries`
      (b) prefix-match lookup of every string in `queries` (Trie uses
          `prefix_matches`; dict and linear scan must check every
          vocabulary entry against `str.startswith`, since neither
          supports prefix search natively)

    Returns a dict: {"exact": {"trie": secs, "dict": secs, "linear": secs},
                      "prefix": {...}}
    Seconds are the best-of-`repeats` wall-clock time (time.perf_counter),
    matching the standard `timeit`-style convention of reporting the
    minimum to reduce noise from scheduling jitter.
    """
    trie = Trie()
    for w in vocabulary:
        trie.insert(w)
    vocab_set = {w: True for w in vocabulary}  # dict lookup
    vocab_list = list(vocabulary)  # linear scan

    results: dict[str, dict[str, float]] = {"exact": {}, "prefix": {}}

    # Exact match
    best = {"trie": None, "dict": None, "linear": None}
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in queries:
            _ = trie.exists(q)
        t = time.perf_counter() - t0
        best["trie"] = t if best["trie"] is None else min(best["trie"], t)

        t0 = time.perf_counter()
        for q in queries:
            _ = q in vocab_set
        t = time.perf_counter() - t0
        best["dict"] = t if best["dict"] is None else min(best["dict"], t)

        t0 = time.perf_counter()
        for q in queries:
            _ = q in vocab_list
        t = time.perf_counter() - t0
        best["linear"] = t if best["linear"] is None else min(best["linear"], t)
    results["exact"] = best  # type: ignore[assignment]

    # Prefix match
    best2 = {"trie": None, "dict": None, "linear": None}
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in queries:
            _ = trie.prefix_matches(q)
        t = time.perf_counter() - t0
        best2["trie"] = t if best2["trie"] is None else min(best2["trie"], t)

        t0 = time.perf_counter()
        for q in queries:
            _ = [w for w in vocab_set if q.startswith(w)]
        t = time.perf_counter() - t0
        best2["dict"] = t if best2["dict"] is None else min(best2["dict"], t)

        t0 = time.perf_counter()
        for q in queries:
            _ = [w for w in vocab_list if q.startswith(w)]
        t = time.perf_counter() - t0
        best2["linear"] = t if best2["linear"] is None else min(best2["linear"], t)
    results["prefix"] = best2  # type: ignore[assignment]

    return results
