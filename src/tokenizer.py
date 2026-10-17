"""
Baseline BPE and morphology-aware BPE training and wrapping.

Both tokenizers in this comparison are trained with the exact same
algorithm (byte-level-free, whitespace-pretokenized BPE, as implemented by
HuggingFace `tokenizers`), the same vocabulary size, the same training
corpus, and the same training configuration (minimum merge frequency,
special tokens). The two tokenizers differ in exactly one respect: before
the BPE trainer ever sees a line of text, the morphology-aware pipeline
first replaces each word with its best morphological segmentation (root
and affix pieces, joined with a boundary marker) using the Trie/DP engine
in morphology.py, while the baseline pipeline leaves each word untouched.
This isolates the pre-segmentation step as the only variable, which is
what makes the comparison a fair ablation of "does morphology-aware
pre-segmentation reduce fertility" rather than a comparison confounded by
different algorithms or different data.

The pre-segmentation step never merges separate whitespace-delimited
words and never removes the word-boundary information: pieces belonging
to the same original word are still written adjacent to each other
(joined by the internal marker below) so the trained BPE model learns
merges within a word's morphological pieces, not across word boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from src.morphology import Lexicon, select_best_segmentation
from src.normalization import normalize_text

MORPH_BOUNDARY = "▁"  # a visible, corpus-absent marker between morphological pieces

SPECIAL_TOKENS = ["<unk>", "<pad>", "<s>", "</s>"]


@dataclass
class TokenizerConfig:
    vocab_size: int = 8000
    min_frequency: int = 2


def morphology_presegment_line(lexicon: Lexicon, line: str) -> str:
    """
    Replace each whitespace-separated word in `line` with its best
    morphological segmentation, pieces joined by MORPH_BOUNDARY, words
    still separated by a single space. Example:
        "घरहरूलाई साथीसँग" -> "घर▁हरू▁लाई साथी▁सँग"
    Word boundaries (the single space) are always preserved; the marker
    only appears *inside* what was one word, never introduced between
    two originally separate words.
    """
    words = normalize_text(line).split(" ")
    out_words = []
    for w in words:
        if w == "":
            continue
        cache = lexicon.__dict__.setdefault("_seg_cache", {})
        if w not in cache:
            seg, _, _, _ = select_best_segmentation(lexicon, w)
            cache[w] = MORPH_BOUNDARY.join(seg.as_strings())
        out_words.append(cache[w])
    return " ".join(out_words)


def train_bpe(
    lines: list[str], config: TokenizerConfig, morphology_presegment: bool = False,
    lexicon: Lexicon | None = None,
) -> Tokenizer:
    """
    Train a BPE tokenizer on `lines`. If `morphology_presegment` is True,
    every line is first passed through `morphology_presegment_line` (which
    requires `lexicon`); otherwise lines are used as-is (after
    `normalize_text`). Both branches use the identical BPE/trainer
    configuration — only the text fed to the trainer differs.
    """
    if morphology_presegment:
        if lexicon is None:
            raise ValueError("lexicon is required when morphology_presegment=True")
        prepared = [morphology_presegment_line(lexicon, line) for line in lines]
    else:
        prepared = [normalize_text(line) for line in lines]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(prepared, trainer=trainer)
    return tokenizer


import re as _re

# ASCII punctuation + the Devanagari danda marks, matched only at the very
# start/end of a token — used to strip sentence-final punctuation that is
# glued directly onto the last word of a line (e.g. "गर्छु।"), so that
# seeding a word generalizes to it regardless of what punctuation follows,
# instead of only matching that exact literal punctuation-attached string.
_EDGE_PUNCT_RE = _re.compile(r"^(?:[^\wऀ-ॿ]|[।॥])+|(?:[^\wऀ-ॿ]|[।॥])+$")


def _strip_edge_punctuation(word: str) -> str:
    """Remove leading/trailing punctuation (ASCII or Devanagari danda) from
    `word`, leaving internal characters untouched. '' in -> '' out."""
    return _EDGE_PUNCT_RE.sub("", word)


def generate_combinatorial_lexicon_seeds(lexicon: Lexicon) -> set[str]:
    """
    Generate additional single-token-worthy candidate strings by
    *combining* lexicon entries, instead of only using whole words that
    were literally observed (in the lexicon's words.csv or in the
    training corpus). This reuses the same root/prefix/suffix inventory
    the Trie/DP segmentation engine (morphology.py) already indexes, but
    here it is used generatively, as a seed-*proposal* step, not as
    inference-time forced splitting — the distinction that matters,
    because forced splitting was measured to hurt fertility (see
    train_bpe's morphology_presegment path) while adding extra guaranteed
    single-token vocabulary entries can only help or do nothing (see
    train_lexicon_seeded_bpe's docstring).

    Combinations generated, each respecting a rough grammatical-category
    compatibility check so that, e.g., a verb suffix is never glued onto
    a noun root:
      - root + suffix, for every root/suffix pair whose categories agree
        (a suffix whose grammatical_category contains "verb" only
        combines with roots categorized "verb"; every other suffix is
        treated as a noun/nominal suffix and only combines with roots
        categorized "noun").
      - prefix + root, for every derivational prefix combined with every
        noun root (prefixes in this lexicon are nominal/adjectival
        derivational prefixes, so they are not combined with verb roots).
      - prefix + root + suffix, for every derivational prefix combined
        with every noun root and every noun suffix (the three-way
        combination a fully productive derivation would allow).

    This deliberately grows the tokenizer's final vocabulary well beyond
    what the training corpus or the plain word list would propose on
    their own — every root/prefix/suffix combination is a *guess* that a
    form is plausible Nepali, not evidence that it is attested or
    frequent. That is the explicit tradeoff: more single-token coverage
    for inflected/derived forms the training corpus never happened to
    contain, at the cost of a larger vocabulary (and, in a real model, a
    larger embedding table). The caller records vocab-size-before/after
    so this tradeoff is visible rather than hidden; see
    research_report.md Section 12.1 for the measured numbers.
    """
    def _is_verb_suffix(meta: dict) -> bool:
        return "verb" in meta.get("grammatical_category", "")

    roots_by_cat: dict[str, list[str]] = {"noun": [], "verb": []}
    for row in lexicon.raw_rows.get("roots", []):
        cat = row.get("grammatical_category", "")
        if cat in roots_by_cat:
            roots_by_cat[cat].append(row["surface_form"])

    suffix_rows = lexicon.raw_rows.get("suffixes", [])
    noun_suffixes = [r["surface_form"] for r in suffix_rows if not _is_verb_suffix(r)]
    verb_suffixes = [r["surface_form"] for r in suffix_rows if _is_verb_suffix(r)]
    prefixes = [r["surface_form"] for r in lexicon.raw_rows.get("prefixes", [])]

    candidates: set[str] = set()

    for root in roots_by_cat["noun"]:
        for suf in noun_suffixes:
            candidates.add(root + suf)
        for pre in prefixes:
            candidates.add(pre + root)
            for suf in noun_suffixes:
                candidates.add(pre + root + suf)

    for root in roots_by_cat["verb"]:
        for suf in verb_suffixes:
            candidates.add(root + suf)

    candidates.discard("")
    return candidates


def train_lexicon_seeded_bpe(
    lines: list[str], config: TokenizerConfig, lexicon: Lexicon,
    min_train_frequency: int = 2,
    combinatorial_expansion: bool = False,
) -> tuple[Tokenizer, dict]:
    """
    A second, different attempt at reducing token fertility, tried after
    the morphology-aware pre-segmentation approach (`train_bpe` with
    `morphology_presegment=True`) was measured and found not to reduce
    fertility at a matched vocabulary budget (see research_report.md,
    Sections 12/17): forcing a split at every morpheme boundary only
    blocks merges BPE could otherwise learn, it never adds one.

    This function instead trains a completely standard BPE tokenizer
    first (identical to the baseline — no pre-segmentation, nothing
    forced), and then *adds* a set of strings directly as atomic
    vocabulary entries on top of it, via `Tokenizer.add_tokens`, which
    inserts them as literal-match entries that the tokenizer's
    pre-tokenization layer matches directly — bypassing BPE's learned
    merge sequence entirely for those strings. The seed set is the union
    of:
      - every word in the lexicon's `words.csv` (whole analyzed forms —
        prior linguistic knowledge, not derived from this corpus),
      - every word that occurs at least `min_train_frequency` times in
        the *training* split, counted on its punctuation-stripped form
        so that a word's frequency is no longer undercounted just
        because some of its occurrences happened to fall at the end of a
        sentence with a danda glued on (ordinary use of training data;
        no test data is ever read here),
      - the exact literal (punctuation-attached) training tokens too, so
        sentence-final occurrences keep getting a direct single-token
        match as well as the stripped form, and
      - (only if `combinatorial_expansion=True`) every root+affix
        combination `generate_combinatorial_lexicon_seeds` proposes —
        see that function's docstring for the mechanism and the
        vocabulary-size tradeoff it introduces.

    Why this can help where forced pre-segmentation could not: this
    never removes a merge opportunity BPE already uses, it only adds
    guaranteed single-token entries for words whose BPE-learned
    representation would otherwise cost more than one token. It can only
    help or do nothing; it cannot make fertility worse the way boundary
    insertion did (reversibility/character-preservation is unaffected
    either way, since `add_tokens` entries still reproduce their own
    exact text).

    Measured, honest caveat: on a very small, lexically repetitive
    training corpus, almost every word a held-out test set will contain
    may already have occurred (and cleared the frequency threshold) in
    training, which inflates the apparent benefit relative to how much
    of it would transfer to genuinely unseen words on a larger, more
    varied corpus. The notebook and research report report the measured
    train/test word overlap alongside the fertility result so this is
    visible rather than hidden. The combinatorial seeds are a partial
    answer to that gap (they cover inflected forms never literally
    observed in training) but are themselves unvalidated guesses about
    which combinations are real Nepali words, not corpus evidence.

    Returns (tokenizer, info) where `info` records how many strings were
    proposed as seeds and how many were actually new (not already a
    single BPE token), broken down by source.
    """
    tokenizer = train_bpe(lines, config, morphology_presegment=False)
    vocab_before = tokenizer.get_vocab_size()

    from collections import Counter
    literal_tokens = [w for line in lines for w in normalize_text(line).split(" ") if w]
    literal_counts = Counter(literal_tokens)
    stripped_counts: Counter = Counter()
    for w, c in literal_counts.items():
        stripped_counts[_strip_edge_punctuation(w)] += c
    stripped_counts.pop("", None)

    frequent_literal = {w for w, c in literal_counts.items() if c >= min_train_frequency}
    frequent_stripped = {w for w, c in stripped_counts.items() if c >= min_train_frequency}
    lexicon_words = {row["surface_form"] for row in lexicon.raw_rows.get("words", [])}

    combinatorial_seeds: set[str] = set()
    if combinatorial_expansion:
        combinatorial_seeds = generate_combinatorial_lexicon_seeds(lexicon)

    seeds = frequent_literal | frequent_stripped | lexicon_words | combinatorial_seeds
    seeds.discard("")

    n_added = tokenizer.add_tokens(sorted(seeds))
    info = {
        "vocab_before_seeding": vocab_before,
        "vocab_after_seeding": tokenizer.get_vocab_size(),
        "seed_candidates_proposed": len(seeds),
        "seed_tokens_actually_added": n_added,
        "min_train_frequency": min_train_frequency,
        "combinatorial_expansion_enabled": combinatorial_expansion,
        "combinatorial_candidates_proposed": len(combinatorial_seeds),
        "frequent_literal_candidates": len(frequent_literal),
        "frequent_stripped_candidates": len(frequent_stripped),
        "lexicon_word_candidates": len(lexicon_words),
    }
    return tokenizer, info


def tokenize_word_baseline(tokenizer: Tokenizer, word: str) -> list[str]:
    """Tokenize a single (already-normalized) word with the baseline
    tokenizer: no pre-segmentation, fed to BPE as-is."""
    return tokenizer.encode(word).tokens


def tokenize_word_morphology_aware(
    tokenizer: Tokenizer, lexicon: Lexicon, word: str,
) -> list[str]:
    """
    Tokenize a single word with the morphology-aware tokenizer: first
    apply the same pre-segmentation used during training, then run BPE on
    the resulting marker-joined string, then strip the boundary marker
    back out of the returned subword tokens' surface text for reporting
    (the marker itself is a training-time device, not meant to be shown
    to a user as if it were a real token the model id-maps to — BPE may
    or may not have kept it as part of a merged token internally, and
    that internal detail is preserved in `tokens_raw` for transparency).
    """
    presegmented = morphology_presegment_line(lexicon, word)
    encoding = tokenizer.encode(presegmented)
    tokens_raw = encoding.tokens
    tokens_display = [t.replace(MORPH_BOUNDARY, "") for t in tokens_raw]
    tokens_display = [t for t in tokens_display if t != ""]
    return tokens_display


def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


# ---------------------------------------------------------------------------
# Pretrained tokenizer comparison (Section 14) — optional, network-gated.
# ---------------------------------------------------------------------------

PRETRAINED_MODELS = {
    "mbert": {"model_id": "bert-base-multilingual-cased", "revision": "main",
              "algorithm": "WordPiece"},
    "xlmr": {"model_id": "xlm-roberta-base", "revision": "main",
             "algorithm": "SentencePiece-Unigram"},
}


def try_load_pretrained_tokenizer(name: str):
    """
    Attempt to load a pretrained tokenizer (mBERT or XLM-R) by its exact
    HuggingFace model identifier, using only `transformers`'
    `AutoTokenizer` (tokenizer files only — no model weights are
    downloaded or needed). Returns (tokenizer_or_None, info_dict).
    `info_dict` always records the exact model_id/revision/library
    version attempted and, on failure, the exact exception message, so
    that unavailability is documented rather than silently skipped, per
    the spec's explicit requirement.
    """
    import transformers

    spec = PRETRAINED_MODELS[name]
    info = {
        "model_id": spec["model_id"], "revision": spec["revision"],
        "algorithm": spec["algorithm"], "library": "transformers",
        "library_version": transformers.__version__, "status": "unknown",
        "error": None,
    }
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            spec["model_id"], revision=spec["revision"],
        )
        info["status"] = "loaded"
        return tok, info
    except Exception as e:  # network unavailable, not on allowlist, etc.
        info["status"] = "unavailable"
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info
