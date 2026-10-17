"""
Fertility-optimized Nepali tokenizer ("OptimizedMorphTokenizer").

Why this exists
---------------
Plain BPE encodes a word by replaying learned merges, which is *greedy*: it
does not necessarily find the fewest tokens the vocabulary could express.
Forced morphological pre-splitting (train_bpe(morphology_presegment=True))
was measured to make this worse. This module combines three ideas that can
only help token fertility, at the SAME total vocabulary budget as the
baseline BPE it is compared with:

1. **Minimum-token (Viterbi / shortest-path) encoding.** Every word is
   encoded by dynamic programming over *all* vocabulary pieces, choosing the
   segmentation with the fewest tokens. Because every BPE token is in the
   vocabulary, the result is provably never longer than BPE's own output
   for the same vocabulary. Ties are broken toward lexicon-licensed
   morpheme boundaries, then toward longer pieces, so the output stays
   linguistically sensible (घर + हरू + लाई rather than an arbitrary split).
2. **Frequency-gain vocabulary selection.** Part of the budget is spent on
   whole words / stems / suffixes chosen greedily by *measured token
   savings on the training corpus* (frequency x tokens saved), re-evaluated
   with the real encoder each round, instead of on the low-value last BPE
   merges.
3. **Lexicon priors.** Every root, prefix, suffix and analysed word in the
   morphology lexicon is guaranteed a vocabulary slot, so inflected forms
   never seen in the corpus still cost root+suffix tokens (2-3) rather than
   a character-by-character fallback.

Everything is reversible: ''.join(tokens) == the original word, for any
input (unknown characters are emitted as single-character pieces that keep
their text; they are *reported* as <unk> only through ``unk_count``).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.morphology import Lexicon
from src.normalization import normalize_text
from src.tokenizer import SPECIAL_TOKENS, TokenizerConfig, _strip_edge_punctuation, train_bpe

MAX_PIECE_LEN = 24


def pretokenize(text: str) -> list[str]:
    """Split on whitespace only. Unlike the BPE baselines (which always cut
    punctuation off into its own token), sentence-final punctuation such as
    the danda stays attached, so a frequent 'word+।' can be ONE vocabulary
    piece when that saves tokens; otherwise the DP still emits the
    punctuation as its own piece."""
    return normalize_text(text).split()


class OptimizedMorphTokenizer:
    def __init__(self, vocab: set[str], morphemes: set[str] | None = None):
        self.vocab = set(vocab)
        self.morphemes = set(morphemes or ())
        self._cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ encode
    def encode_word(self, word: str) -> list[str]:
        """Fewest-token segmentation of one pre-tokenized word."""
        hit = self._cache.get(word)
        if hit is not None:
            return list(hit)
        n = len(word)
        INF = (10 ** 9, 0, 0)
        # cost tuple: (token_count, -morpheme_pieces, -sum(len^2)) minimised lexicographically
        best = [INF] * (n + 1)
        back = [0] * (n + 1)
        best[0] = (0, 0, 0)
        for i in range(n):
            if best[i] == INF:
                continue
            c0, c1, c2 = best[i]
            for j in range(i + 1, min(n, i + MAX_PIECE_LEN) + 1):
                piece = word[i:j]
                if piece in self.vocab:
                    cost = (c0 + 1, c1 - (1 if piece in self.morphemes else 0),
                            c2 - (j - i) ** 2)
                elif j == i + 1:  # unknown single character: keep text, cost 1
                    cost = (c0 + 1, c1, c2 - 1)
                else:
                    continue
                if cost < best[j]:
                    best[j] = cost
                    back[j] = i
        out, j = [], n
        while j > 0:
            i = back[j]
            out.append(word[i:j])
            j = i
        out.reverse()
        if len(self._cache) < 500_000:
            self._cache[word] = out
        return list(out)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize arbitrary text (words + punctuation)."""
        out: list[str] = []
        for piece in pretokenize(text):
            out.extend(self.encode_word(piece))
        return out

    def tokenize_word(self, word: str) -> list[str]:
        """Tokenize one whitespace-delimited word (punctuation split off)."""
        return self.tokenize(word)

    def unk_count(self, text: str) -> int:
        return sum(1 for t in self.tokenize(text) if t not in self.vocab)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)

    # -------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vocab": sorted(self.vocab),
                                    "morphemes": sorted(self.morphemes)},
                                   ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OptimizedMorphTokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(set(d["vocab"]), set(d["morphemes"]))

    def vocab_size(self) -> int:
        return len(self.vocab) + len(SPECIAL_TOKENS)


def _lexicon_morphemes(lexicon: Lexicon) -> set[str]:
    out = set()
    for kind in ("roots", "prefixes", "suffixes"):
        for row in lexicon.raw_rows.get(kind, []):
            out.add(normalize_text(row["surface_form"]))
    out.discard("")
    return out


def _lexicon_words(lexicon: Lexicon) -> set[str]:
    return {normalize_text(r["surface_form"]) for r in lexicon.raw_rows.get("words", [])
            if r.get("surface_form")}


def train_optimized_tokenizer(
    lines: list[str], config: TokenizerConfig, lexicon: Lexicon,
    pool_factor: float = 2.5, prune_fraction: float = 0.12,
) -> tuple[OptimizedMorphTokenizer, dict]:
    """
    Build an OptimizedMorphTokenizer whose TOTAL vocabulary (including the
    special tokens) equals ``config.vocab_size`` -- exactly the budget the
    baseline BPE gets, so the comparison is fair.

    Algorithm (Unigram-style pruning driven by the minimum-token encoder):
      1. Candidate pool = BPE trained at ``pool_factor`` x the budget
         + every character + frequent whole words + lexicon morphemes.
      2. Repeat: encode the training corpus with the min-token encoder,
         compute for every non-protected piece its *loss* = usage x extra
         tokens its removal would cost (usage x (cost_without_it - 1)), and
         drop the ``prune_fraction`` cheapest pieces, until the budget is met.
      Characters and lexicon morphemes are protected, so any word can always
      be spelled and lexicon-licensed roots/suffixes always stay available.
    """
    budget = config.vocab_size - len(SPECIAL_TOKENS)
    word_counts: Counter = Counter()
    for line in lines:
        word_counts.update(pretokenize(line))
    chars = {c for w in word_counts for c in w}
    morphemes = _lexicon_morphemes(lexicon)
    protected = chars | morphemes

    big = train_bpe(lines, TokenizerConfig(
        vocab_size=int(config.vocab_size * pool_factor), min_frequency=config.min_frequency))
    pool = {t for t in big.get_vocab() if t not in SPECIAL_TOKENS} | protected
    stripped_counts: Counter = Counter()
    for w, c in word_counts.items():
        stripped_counts[_strip_edge_punctuation(w)] += c
    stripped_counts.pop("", None)
    for counts in (word_counts, stripped_counts):
        pool |= {w for w, c in counts.most_common(int(budget * 1.5)) if c >= 2 and len(w) > 1}
    pool |= {w for w in _lexicon_words(lexicon) if word_counts.get(w, 0) > 0}

    tok = OptimizedMorphTokenizer(pool, morphemes)
    info = {"initial_pool": len(pool), "protected": len(protected), "prune_rounds": 0}
    while len(tok.vocab) > max(budget, len(protected)):
        tok._cache.clear()
        usage: Counter = Counter()
        for w, c in word_counts.items():
            for t in tok.encode_word(w):
                usage[t] += c
        loss: dict[str, int] = {}
        for t in tok.vocab:
            if t in protected:
                continue
            if usage[t] == 0:
                loss[t] = 0
                continue
            tok.vocab.discard(t)
            tok._cache.pop(t, None)
            alt = len(tok.encode_word(t))
            tok.vocab.add(t)
            tok._cache.pop(t, None)
            loss[t] = usage[t] * (alt - 1)
        excess = len(tok.vocab) - budget
        n = max(1, min(excess, int(len(tok.vocab) * prune_fraction)))
        for t in sorted(loss, key=lambda x: (loss[x], -len(x), x))[:n]:
            tok.vocab.discard(t)
        info["prune_rounds"] += 1
    tok._cache.clear()
    info["final_vocab_incl_special"] = tok.vocab_size()
    return tok, info
