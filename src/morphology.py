"""
Morphological lexicon loading and indexed candidate segmentation.

This module turns the seed lexicon CSVs (data/lexicons/{roots,prefixes,
suffixes,words}.csv) into Trie indexes, then uses those indexes to
segment a normalized word into a root plus zero or more prefixes/suffixes.

Segmentation is treated as a search problem: for a given word, there can
be more than one way to carve it into known pieces (or pieces plus an
unknown remainder). Rather than brute-force enumerating every possible
split point combination (which is exponential in word length), a dynamic
program finds every valid split using a left-to-right table-filling
approach — each position in the word is reached by extending a valid
split ending at an earlier position with one more known unit, so a split
that is already known to be invalid at position i is never re-explored
from a later position. This keeps the work linear in (word length x
average number of Trie branches considered per position), not exponential
in the number of candidate boundaries.

Each full candidate segmentation is then scored by `score_candidate`
using a small, explicit set of weighted terms (see its docstring), and the
highest-scoring candidate is selected. The scoring weights are treated as
tunable parameters, selected by checking which weight settings produce the
best agreement with a small manually labeled validation set (see
`select_weights_on_validation`), never by looking at the test set.

If no known unit matches at all, the word falls back to being treated as
a single unknown root (the "unknown-word fallback" required by the
project's evaluation groups).

Reversibility: every returned segmentation's pieces, concatenated in
order, reproduce the exact original (normalized) word. This is checked by
`Segmentation.reconstructs(word)` and is asserted in tests/test_morphology.py.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from src.trie_index import Trie


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------

LEXICON_FIELDS = [
    "surface_form", "root", "prefix_sequence", "suffix_sequence",
    "grammatical_category", "frequency", "source", "confidence",
]


@dataclass
class Lexicon:
    """Holds the four Trie indexes built from the seed lexicon CSVs, plus
    the raw rows (for metadata/debugging) and a frequency lookup."""
    roots: Trie
    prefixes: Trie
    suffixes: Trie
    words: Trie
    raw_rows: dict[str, list[dict]] = field(default_factory=dict)

    def root_frequency(self, root: str) -> int:
        meta = self.roots.get_metadata(root)
        return int(meta["frequency"]) if meta else 0

    def affix_frequency(self, affix: str, kind: str) -> int:
        trie = self.prefixes if kind == "prefix" else self.suffixes
        meta = trie.get_metadata(affix)
        return int(meta["frequency"]) if meta else 0


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_lexicon(lexicon_dir: Path) -> Lexicon:
    """
    Build Trie indexes from data/lexicons/{roots,prefixes,suffixes,words}.csv.
    Missing files produce empty (but valid) Tries rather than raising, so
    that a partially populated lexicon directory degrades gracefully with
    everything falling back to the unknown-word path — this is logged by
    the caller, not silently hidden.
    """
    lexicon_dir = Path(lexicon_dir)
    root_rows = _load_csv(lexicon_dir / "roots.csv")
    prefix_rows = _load_csv(lexicon_dir / "prefixes.csv")
    suffix_rows = _load_csv(lexicon_dir / "suffixes.csv")
    word_rows = _load_csv(lexicon_dir / "words.csv")

    roots, prefixes, suffixes, words = Trie(), Trie(), Trie(), Trie()
    for row in root_rows:
        roots.insert(row["surface_form"], row)
    for row in prefix_rows:
        prefixes.insert(row["surface_form"], row)
    for row in suffix_rows:
        suffixes.insert(row["surface_form"], row)
    for row in word_rows:
        words.insert(row["surface_form"], row)

    return Lexicon(
        roots=roots, prefixes=prefixes, suffixes=suffixes, words=words,
        raw_rows={"roots": root_rows, "prefixes": prefix_rows,
                  "suffixes": suffix_rows, "words": word_rows},
    )


def update_frequencies_from_corpus(lexicon: Lexicon, word_counts: dict[str, int]) -> None:
    """
    Overwrite the seed (placeholder) frequency fields with real counts
    from a corpus, where available. Any lexicon entry not seen in
    `word_counts` keeps its seed frequency (so rare-but-valid seed entries
    are not zeroed out just because they didn't appear in a particular
    sample). This is applied to the `words` trie's metadata in place.
    """
    lexicon.__dict__.pop("_seg_cache", None)  # frequencies change scoring
    for surface, meta in list(_iter_trie_metadata(lexicon.words)):
        if surface in word_counts:
            meta["frequency"] = word_counts[surface]


def _iter_trie_metadata(trie: Trie):
    for key in trie.all_keys():
        meta = trie.get_metadata(key)
        if meta is not None:
            yield key, meta


# ---------------------------------------------------------------------------
# Candidate segmentation (DP)
# ---------------------------------------------------------------------------

@dataclass
class Piece:
    text: str
    kind: str  # "prefix" | "root" | "suffix" | "whole_word" | "unknown"


@dataclass
class Segmentation:
    pieces: list[Piece]
    word: str

    def reconstructs(self, original: str) -> bool:
        return "".join(p.text for p in self.pieces) == original

    def as_strings(self) -> list[str]:
        return [p.text for p in self.pieces]


def _candidate_pieces_at(lexicon: Lexicon, remainder: str, is_start: bool) -> list[Piece]:
    """All known units that are a prefix of `remainder`, as Piece objects.
    `is_start` controls whether a derivational prefix is eligible here
    (prefixes only attach at the very start of a word)."""
    out: list[Piece] = []
    if is_start:
        for m in lexicon.prefixes.prefix_matches(remainder):
            out.append(Piece(m, "prefix"))
    for m in lexicon.roots.prefix_matches(remainder):
        out.append(Piece(m, "root"))
    for m in lexicon.suffixes.prefix_matches(remainder):
        out.append(Piece(m, "suffix"))
    return out


def generate_candidates(
    lexicon: Lexicon, word: str, max_candidates: int = 25, max_depth: int = 6,
) -> list[Segmentation]:
    """
    Generate up to `max_candidates` full segmentations of `word` using a
    dynamic-programming walk: `paths[i]` holds every partial piece-sequence
    (capped at `max_candidates`) that exactly covers word[0:i]. Each step
    extends every entry in `paths[i]` by one more known unit that matches
    word[i:], land at `paths[i+len(unit)]`, and the process is linear in
    len(word) times the (small, constant) number of Trie branches
    considered at each position. If `word` itself is a complete entry in
    the whole-word lexicon, that single-piece segmentation is included as
    a candidate on its own.

    `max_depth` caps the number of pieces in any one candidate, so a
    pathological word cannot produce an unbounded number of tiny-piece
    segmentations.

    If the word cannot be split into known pieces at all (not even a
    length-1 prefix match anywhere), a single-item list containing the
    "unknown root" fallback segmentation is returned — this guarantees
    `generate_candidates` never returns an empty list for a non-empty word.
    """
    n = len(word)
    if n == 0:
        return [Segmentation([], word)]

    # paths[i] = list of (pieces_so_far,) ending exactly at position i
    paths: list[list[list[Piece]]] = [[] for _ in range(n + 1)]
    paths[0] = [[]]

    if lexicon.words.exists(word):
        meta = lexicon.words.get_metadata(word) or {}
        paths[n].append([Piece(word, "whole_word")])

    for i in range(n):
        if not paths[i]:
            continue
        remainder = word[i:]
        candidates_here = _candidate_pieces_at(lexicon, remainder, is_start=(i == 0))
        for piece in candidates_here:
            j = i + len(piece.text)
            for seq in paths[i]:
                if len(seq) >= max_depth:
                    continue
                new_seq = seq + [piece]
                if len(paths[j]) < max_candidates:
                    paths[j].append(new_seq)

    results = [Segmentation(seq, word) for seq in paths[n][:max_candidates]]

    if not results:
        results = [Segmentation([Piece(word, "unknown")], word)]

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "whole_word_known": 5.0,   # big bonus: word matched as-is in the word lexicon
    "root_known": 2.0,         # bonus per root piece that is a known root
    "affix_known": 1.0,        # bonus per prefix/suffix piece that is known
    "frequency_log": 0.5,      # bonus scaled by log(1+frequency) of each piece
    "segment_count_penalty": 0.4,  # penalty per extra segment (prefers fewer splits)
    "unknown_penalty": 3.0,    # penalty for any "unknown" piece
}


def score_candidate(
    candidate: Segmentation, lexicon: Lexicon, weights: dict[str, float] = None,
) -> tuple[float, dict[str, float]]:
    """
    Score one candidate segmentation. Returns (total_score, breakdown) so
    the breakdown can be shown transparently (required by the project
    spec: "must show example output ... with scores ... and explanation").

    The terms, each independently interpretable:
      - whole_word_known: +weight if the entire word matched a lexicon
        entry directly (no splitting needed) — the strongest evidence.
      - root_known: +weight per piece tagged "root" (a split is only
        rewarded for containing a real root, not merely for existing).
      - affix_known: +weight per piece tagged "prefix" or "suffix".
      - frequency_log: +weight * log(1+frequency) summed over pieces with
        lexicon frequency data — more common units score higher, using a
        log so one very frequent piece cannot dominate the whole score.
      - segment_count_penalty: -weight * (num_pieces - 1) — biases toward
        fewer, larger pieces so segmentation does not fragment words
        merely because many short known substrings exist (this is the
        explicit anti-overfragmentation requirement in the spec).
      - unknown_penalty: -weight per "unknown" piece.
    """
    import math

    w = weights or DEFAULT_WEIGHTS
    breakdown: dict[str, float] = {k: 0.0 for k in w}

    if len(candidate.pieces) == 1 and candidate.pieces[0].kind == "whole_word":
        breakdown["whole_word_known"] = w["whole_word_known"]

    for piece in candidate.pieces:
        if piece.kind == "root":
            breakdown["root_known"] += w["root_known"]
            freq = lexicon.root_frequency(piece.text)
            if freq:
                breakdown["frequency_log"] += w["frequency_log"] * math.log(1 + freq)
        elif piece.kind in ("prefix", "suffix"):
            breakdown["affix_known"] += w["affix_known"]
            freq = lexicon.affix_frequency(piece.text, piece.kind)
            if freq:
                breakdown["frequency_log"] += w["frequency_log"] * math.log(1 + freq)
        elif piece.kind == "unknown":
            breakdown["unknown_penalty"] -= w["unknown_penalty"]

    num_pieces = len(candidate.pieces)
    breakdown["segment_count_penalty"] = -w["segment_count_penalty"] * max(0, num_pieces - 1)

    total = sum(breakdown.values())
    return total, breakdown


def select_best_segmentation(
    lexicon: Lexicon, word: str, weights: dict[str, float] = None,
    max_candidates: int = 25, max_depth: int = 6,
) -> tuple[Segmentation, float, dict[str, float], list[tuple[Segmentation, float]]]:
    """
    Generate candidates, score each one, and return the best one along
    with its score breakdown and the full ranked candidate list (so the
    demo and the report can show "all candidates, scores, selected
    segmentation" as the spec requires).
    """
    candidates = generate_candidates(lexicon, word, max_candidates, max_depth)
    scored = [(c, score_candidate(c, lexicon, weights)[0]) for c in candidates]
    scored.sort(key=lambda t: t[1], reverse=True)
    best_candidate, best_score = scored[0]
    _, breakdown = score_candidate(best_candidate, lexicon, weights)
    for c, _ in scored:
        assert c.reconstructs(word), (
            f"Reversibility violated: {c.as_strings()} does not reconstruct {word!r}"
        )
    return best_candidate, best_score, breakdown, scored


def select_weights_on_validation(
    lexicon: Lexicon, validation_pairs: list[tuple[str, list[str]]],
    candidate_weight_sets: list[dict[str, float]],
) -> dict[str, float]:
    """
    Pick the weight dict (from `candidate_weight_sets`) whose selected
    segmentation agrees with the most entries in `validation_pairs`
    (word, expected_pieces). This is the only place scoring weights are
    "tuned" — by construction it must only ever be called with the
    validation split, never the test split; the notebook enforces this by
    only ever passing validation data into this function.
    """
    best_weights = candidate_weight_sets[0]
    best_agreement = -1
    for weights in candidate_weight_sets:
        agreement = 0
        for word, expected in validation_pairs:
            best_seg, _, _, _ = select_best_segmentation(lexicon, word, weights)
            if best_seg.as_strings() == expected:
                agreement += 1
        if agreement > best_agreement:
            best_agreement = agreement
            best_weights = weights
    return best_weights


# ---------------------------------------------------------------------------
# Evaluation-group classification (Section 11 of the spec)
# ---------------------------------------------------------------------------

def classify_word_group(lexicon: Lexicon, word: str) -> str:
    """
    Classify a word into one of the unknown-word evaluation groups:
      - "known_word": the whole word is in the word lexicon.
      - "known_components": not a whole-word match, but every piece in the
        best segmentation is a known root/prefix/suffix (no "unknown"
        piece).
      - "unknown_root": the best segmentation has exactly one "unknown"
        piece and it is the only piece (the whole word is unanalyzable).
      - "unsupported_affix": the best segmentation mixes a known root with
        at least one "unknown" trailing/leading piece (root recognized,
        affix not).
      - "mixed_script": contains both Devanagari and Latin characters.
      - "ambiguous": more than one top-scoring candidate exists with an
        equal best score (tie at the top).
    """
    from src.normalization import is_mixed_script

    if is_mixed_script(word):
        return "mixed_script"
    if lexicon.words.exists(word):
        return "known_word"

    best_seg, best_score, _, scored = select_best_segmentation(lexicon, word)
    ties = [s for c, s in scored if abs(s - best_score) < 1e-9]
    if len(ties) > 1:
        return "ambiguous"

    kinds = [p.kind for p in best_seg.pieces]
    if "unknown" not in kinds:
        return "known_components"
    if kinds == ["unknown"]:
        return "unknown_root"
    return "unsupported_affix"
