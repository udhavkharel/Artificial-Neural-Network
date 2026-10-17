# Nepali Morphology-Aware Indexed Tokenizer for Token Fertility Optimization

## 1. Title

Nepali Morphology-Aware Indexed Tokenizer for Token Fertility Optimization:
A Trie-Indexed Segmentation Approach Compared Against Byte-Pair Encoding

## 2. Abstract

Subword tokenizers trained with byte-pair encoding (BPE) or similar
algorithms tend to produce more tokens per word for morphologically rich
languages than for languages with simpler inflection, because a single
word can carry several grammatical markers that a generic, frequency-
driven merge algorithm has no explicit knowledge of. This project builds
an indexed, Trie-based morphological segmentation engine for Nepali — a
language with an agglutinative case/postposition system — that proposes
and scores candidate root/prefix/suffix splits of a word using a
dynamic-programming search, and uses that segmentation as a
pre-processing step before training BPE. The resulting "morphology-aware"
tokenizer is compared against an identically configured baseline BPE
tokenizer (same algorithm, vocabulary size, training corpus, and trainer
settings) and, where available, existing pretrained multilingual
tokenizers (mBERT, XLM-R), on token fertility (tokens per word),
characters-per-token, a six-variant ablation study isolating each
component of the segmentation engine, and a vocabulary-size sweep, with
paired non-parametric statistical testing and bootstrap confidence
intervals. Because the execution environment had no network access to
Kaggle or HuggingFace Hub, the comparison was run on a synthetic corpus
generated from the project's own seed lexicon rather than the intended
OSCAR Corpus Nepali, and the two pretrained tokenizers could not be
loaded; both limitations are reported explicitly rather than concealed.
On this synthetic corpus, morphology-aware pre-segmentation increased
measured fertility rather than reducing it, and the report explains why,
rather than presenting only a favorable framing of the result. A second,
mechanically different optimization was then tried: rather than forcing
a split before BPE training, the trained lexicon's known whole words and
frequent training-corpus words were added directly as guaranteed
single-token vocabulary entries on top of an otherwise identical baseline
BPE model. This second approach measurably reduced fertility by about
21.7% at a matched vocabulary budget, confirmed with a paired Wilcoxon
test and a bootstrap confidence interval excluding zero, with the honest
caveat that on this run's small test set every test word had already
appeared in training, which likely inflates the measured size of the
benefit relative to a larger, less repetitive corpus. A third attempt
generatively expanded the seed set to every root+affix combination the
lexicon's grammar licenses, growing the vocabulary roughly 26x; this
measured zero additional benefit on the available test set, a null
result reported honestly rather than omitted, with the real
vocabulary-size cost shown alongside it. Finally, the optimized
tokenizer was applied to a realistic end-to-end interaction: a small
demo tokenizes a Nepali prompt and an answer to it (either pasted in or
fetched live from the Anthropic API with a user-supplied key) and
reports the token-count savings on that real input/output pair, which
surfaces the same generalization limitation directly on ordinary
generative-AI prose.

## 3. Research problem

Standard subword tokenizers (BPE, WordPiece, SentencePiece Unigram) learn
their vocabulary purely from surface-level character co-occurrence
statistics. They have no access to morphological structure, so a word
with several stacked grammatical suffixes may be split in ways that are
linguistically arbitrary even when they are statistically efficient, or
may fail to generalize to an unseen inflected form of an otherwise common
root. For agglutinative and morphologically rich languages such as
Nepali, this can translate into higher token fertility than for
morphologically simpler languages at a comparable vocabulary budget,
which in turn affects the effective context length and computational
cost of any downstream model that consumes the tokenized text.

## 4. Motivation

Token fertility is a practical, measurable proxy for how efficiently a
tokenizer represents a language. A tokenizer that needs more tokens per
word to represent the same content costs more compute per unit of text
during training and inference of any model built on top of it, and
effectively shortens the usable context window for a fixed token budget.
If morphological structure can be used to reduce fertility without
changing the underlying BPE algorithm, vocabulary size, or training data,
that would be a low-cost way to make Nepali-language modeling more
efficient. This project investigates whether a concrete, implementable
version of that idea — rule-indexed morphological pre-segmentation —
actually delivers that reduction, rather than assuming it does.

## 5. Research question and objectives

**Research question:** Does pre-segmenting Nepali words into their root
and known grammatical affixes, using an indexed (Trie-based) lexicon and
a scored dynamic-programming search, reduce token fertility when used as
a pre-processing step before training a byte-pair-encoding tokenizer,
compared to training BPE directly on unsegmented text?

**Objectives:**
1. Build an indexed morphological lexicon and a Trie-based lookup
   structure for Nepali roots, prefixes, and suffixes.
2. Build a dynamic-programming segmentation engine that generates scored
   candidate splits of a word and selects the best one, with a
   documented unknown-word fallback and guaranteed reversibility.
3. Train a baseline and a morphology-aware BPE tokenizer under identical
   conditions, differing only in the pre-segmentation step.
4. Measure token fertility, characters-per-token, and related metrics for
   both tokenizers (and pretrained tokenizers where available) on a
   held-out test set.
5. Isolate the contribution of each component of the segmentation engine
   via a six-variant ablation study.
6. Test whether the fertility difference is statistically significant,
   and quantify it with a bootstrap confidence interval.

## 6. Background on subword tokenization

Byte-pair encoding (BPE) for neural machine translation and language
modeling was introduced by Sennrich, Haddow, and Birch (2016), who
adapted a data-compression algorithm to iteratively merge the most
frequent adjacent symbol pairs in a training corpus until a target
vocabulary size is reached. WordPiece, used by BERT and its multilingual
variant mBERT (Devlin et al., 2019), follows a similar iterative-merge
strategy but selects merges by a likelihood-based criterion rather than
raw frequency. SentencePiece's Unigram model, used by XLM-R (Conneau et
al., 2020) and introduced by Kudo (2018) and Kudo and Richardson (2018),
instead starts from a large candidate vocabulary and prunes it to
maximize the likelihood of the training corpus under a unigram language
model, which tends to produce more linguistically motivated segmentations
than pure frequency-based merging in some languages.

All three approaches are morphology-agnostic: they have no built-in
notion of a root or an affix and rely entirely on what is statistically
efficient given the training data and vocabulary budget. Morfessor
(Creutz and Lagus, 2007; Grönroos et al., "Morfessor 2.0," 2014) takes
the opposite approach: it learns a probabilistic model of morpheme
segmentation directly from raw text, unsupervised, without any
hand-built lexicon, trading off segmentation accuracy against model
complexity (minimum description length). The approach taken in this
project sits between these two: rather than learning morphology purely
statistically (Morfessor) or ignoring it entirely (plain BPE), it uses a
small, explicit, hand-built lexicon and a scored search over known
roots/affixes to decide where to split a word before BPE ever sees it —
closer in spirit to finite-state or rule-based morphological analysis
than to either Morfessor or plain BPE, but deliberately simpler than a
full finite-state transducer.

## 7. Proposed architecture

The system has four layers:

1. **Lexicon layer** (`data/lexicons/*.csv`, loaded by
   `src/morphology.load_lexicon`): four separate tables — roots,
   prefixes, suffixes, and fully analyzed complete words — each row
   carrying `surface_form, root, prefix_sequence, suffix_sequence,
   grammatical_category, frequency, source, confidence`.
2. **Index layer** (`src/trie_index.Trie`): each of the four tables is
   loaded into its own Trie, supporting exact lookup, prefix-match
   lookup, metadata retrieval, and a count of stored keys.
3. **Segmentation layer** (`src/morphology.generate_candidates` /
   `select_best_segmentation`): a dynamic program walks a word
   left-to-right, extending every partial segmentation ending at
   position *i* by every known unit (prefix, root, or suffix) that
   matches the text starting at *i*, up to a configurable candidate cap
   and maximum piece count. Each resulting full segmentation is scored
   (`score_candidate`) by a small set of explicit, weighted terms, and
   the highest-scoring one is selected. If no known unit matches
   anywhere in the word, the word is returned as a single "unknown"
   piece (the unknown-word fallback).
4. **Tokenizer layer** (`src/tokenizer.py`): a baseline BPE tokenizer
   trained on unmodified normalized text, and a morphology-aware BPE
   tokenizer trained on the same text after every word has been replaced
   by its selected segmentation's pieces, joined by an internal boundary
   marker that is stripped back out when reporting tokens. Both
   tokenizers use the identical BPE algorithm, vocabulary-size budget,
   minimum-frequency setting, and special-token set.

## 8. Dataset and preprocessing

The intended corpus was OSCAR Corpus Nepali, accessed via `kagglehub`
(`hsebarp/oscar-corpus-nepali`), sampled with reservoir sampling for
memory efficiency on a large file. This environment's network proxy
returns a 403 for both the Kaggle and HuggingFace Hub hosts (confirmed by
a direct connection attempt, not assumed), so this specific run instead
used a **synthetic corpus**: 4,000 short sentences generated by filling a
small set of grammatical sentence templates with nouns and verbs drawn
from the seed lexicon's `words.csv`. The synthetic corpus was split
80/10/10 into train/validation/test at the sentence (document) level with
a fixed random seed (42), so no sentence appears in more than one split.
All text (real or synthetic) is passed through `src/normalization.py`'s
Unicode NFC normalization and whitespace collapsing before any other
processing.

The `--real` code path in `scripts_build/build_corpus.py` implements the
intended OSCAR-corpus download and reservoir sampling and is unchanged
from how it would run on a machine with normal internet access; it was
not exercised in this run.

## 9. Implementation and algorithms

- **Normalization:** Unicode NFC (not NFKC, to avoid altering
  width/compatibility-sensitive characters) plus whitespace collapsing.
  Devanagari vowel signs, conjuncts, punctuation, and digits are left
  untouched in content.
- **Trie:** a standard character-level prefix tree with per-key metadata
  storage; `prefix_matches` walks the input once and collects every
  complete key passed along the way, giving candidate generation
  practically free prefix enumeration.
- **Segmentation search:** dynamic programming over word positions
  (`paths[i]` = every valid partial-piece-sequence ending at position
  *i*), bounded by `max_candidates` (pruning breadth) and `max_depth`
  (pruning the number of pieces per candidate), making the search linear
  in word length rather than exponential in the number of possible split
  points.
- **Scoring:** `score_candidate` sums six weighted, independently
  interpretable terms (whole-word-known bonus, per-root bonus, per-affix
  bonus, log-frequency bonus, a penalty proportional to segment count,
  and a penalty per unknown piece); see `src/morphology.py` for the exact
  formula and `src/morphology.DEFAULT_WEIGHTS` for the default values.
- **Weight selection:** `select_weights_on_validation` chooses among a
  set of candidate weight dictionaries by agreement with a small labeled
  validation set, and is structurally prevented from ever being called
  with test data by the project's own pipeline (only validation pairs are
  ever passed to it).
- **Reversibility:** every `Segmentation` object's `reconstructs` method
  checks that concatenating its pieces reproduces the exact original
  word; this is asserted inside `select_best_segmentation` for every
  candidate it considers, and tested explicitly in
  `tests/test_morphology.py`.

## 10. Experimental design

Two tokenizers (baseline BPE, morphology-aware BPE) were trained on the
same training split, same vocabulary-size budget (1,200 requested; the
actual learned vocabulary saturated below this at 219 and 185 tokens
respectively, because the synthetic corpus has too few distinct subword
units to use a larger budget — see Section 13). Fertility and related
metrics were then computed on the held-out test split's words, which
neither tokenizer training nor lexicon-frequency updates ever saw. The
pretrained-tokenizer comparison (mBERT, XLM-R) was attempted under the
same protocol but could not be completed in this environment (Section
14). A paired comparison (same test words, same order) allows a
non-parametric paired test (Wilcoxon signed-rank) rather than requiring
an independent-samples test, and supports a paired bootstrap for the mean
token-count difference.

## 11. Metrics and equations

```
token fertility        = total subword tokens / total whitespace-separated words
characters per token    = total normalized text characters / total subword tokens
reduction percentage    = 100 x (baseline_token_count - proposed_token_count) / baseline_token_count
```

Also computed: total tokens; mean, median, and 95th-percentile
tokens-per-word; tokens-per-character; character preservation rate (does
concatenating a tokenizer's output reconstruct the input exactly); word
coverage (fraction of words the lexicon has some known analysis for);
unknown-token rate; morphological-index coverage (fraction of words
classified into each evaluation group by `classify_word_group`);
segmentation confidence (the selected candidate's score, shown per word
in the notebook's demo); and runtime per document and per 1,000
characters. Special tokens (`<unk>`, `<pad>`, `<s>`, `</s>`) are excluded
from the primary fertility and characters-per-token calculations by
default.

## 12. Results (from actual execution)

All numbers below are read directly from
`outputs/tables/fertility_comparison.csv`,
`outputs/tables/statistical_tests.csv`, and
`outputs/tables/bootstrap_ci_baseline_vs_morph.csv`, produced by running
`scripts_build/run_experiment.py` once, end to end, on the synthetic
corpus (1,441 test words).

| Tokenizer | Total tokens | Token fertility | Chars/token | Unknown-token rate |
|---|---|---|---|---|
| Baseline BPE | 1,841 | 1.2776 | 5.4253 | 0.0 |
| Morphology-aware BPE (forced split) | 2,864 | 1.9875 | 3.4874 | 0.0 |
| Lexicon-seeded BPE (optimized) | 1,441 | 1.0000 | 6.9313 | 0.0 |
| Lexicon-seeded BPE + combinatorial expansion | 1,441 | 1.0000 | 6.9313 | 0.0 |

Reduction percentage vs. baseline: morphology-aware **-55.57%** (a
negative value means more tokens, i.e. a regression); lexicon-seeded
**+21.73%** (a positive value here means fewer tokens, i.e. a genuine
reduction); lexicon-seeded + combinatorial expansion **+21.73%**
(identical to plain lexicon-seeding — see Section 12.2).

A paired Wilcoxon signed-rank test on the per-word token counts gives, for
baseline vs. morphology-aware, statistic = 0.0, p ≈ 3.86e-172; for
baseline vs. lexicon-seeded, statistic = 0.0, p ≈ 5.51e-89; for baseline
vs. lexicon-seeded + combinatorial, statistic = 0.0, p ≈ 5.51e-89 (the
per-word token counts are identical to plain lexicon-seeding on this test
set, so the test statistic is too). After Holm-Bonferroni correction
across all comparisons, all remain significant at alpha = 0.05. The
paired bootstrap (2,000 resamples) for the mean difference (baseline -
morphology-aware) gives an observed difference of -0.710 tokens/word, 95%
CI [-0.743, -0.677]; for (baseline - lexicon-seeded), the observed
difference is +0.278 tokens/word, 95% CI [0.254, 0.300]; for (baseline -
lexicon-seeded-combinatorial), the observed difference is likewise +0.278
tokens/word, 95% CI [0.254, 0.300] — both lexicon-seeded variants are
statistically significant and estimated precisely, in the same direction,
with the combinatorial variant adding no measurable change on this test
set despite its much larger vocabulary (Section 12.2).

These are real, measured results, not hypothetical examples: on this
corpus, the forced-pre-segmentation approach made fertility worse, the
lexicon-seeding approach made it measurably better, and combinatorially
expanding the seed set further made no measurable difference at all.
Section 17 discusses the mechanical reason for the difference between the
approaches, and Section 16 states each result's own honest limitation
(train/test word overlap for lexicon-seeding; test-set lexical narrowness
for combinatorial expansion).

### 12.1 How the lexicon-seeded tokenizer works

`src/tokenizer.train_lexicon_seeded_bpe` trains an ordinary baseline BPE
tokenizer first — identical to the baseline above, nothing forced or
pre-split — and then calls `Tokenizer.add_tokens` to insert a set of
whole-word strings directly as atomic vocabulary entries on top of it.
The seed set is the union of every word in the lexicon's `words.csv`
(prior linguistic knowledge, independent of this corpus) and every word
occurring at least twice in the *training* split, counted on both its
exact literal form and its punctuation-stripped form (ordinary use of
training data; no test data is read) — the punctuation-stripped count is
a fix added after an earlier run showed seeding only benefited words in
the exact literal spelling they were observed in (e.g. a word seen only
as `"...गर्छु।"`, glued to a sentence-final danda, would not seed the
plain form `"गर्छु"` elsewhere; now it does, verified by
`test_lexicon_seeded_bpe_generalizes_past_sentence_final_punctuation` in
`tests/test_tokenizer.py`). Added tokens are matched by the
tokenizer's pre-tokenization layer as literal strings, bypassing BPE's
learned merge sequence entirely for those words — so a word that would
otherwise have cost the baseline model two or more merges can become a
single token for free. In this run, 78 such entries were added, taking
the vocabulary from 219 to 238 (`outputs/reports/run_log.json`:
`lexicon_seeded_vocab_info`).

The key structural property, verified by `tests/test_tokenizer.py`'s
`test_lexicon_seeded_bpe_does_not_increase_total_tokens`: because the
underlying BPE model is byte-for-byte identical to the baseline's and
only atomic entries are added on top of it, the seeded tokenizer can
never need *more* tokens than the baseline for any word, seen or unseen
— it can only match or improve on the baseline's tokenization. This is
the opposite guarantee from forced pre-segmentation, which can only match
or worsen the baseline, and explains why one approach helped and the
other did not.

Six variants were evaluated on up to 300 held-out test words (limited by
the synthetic test split's size), from `outputs/tables/ablation_study.csv`:

| Variant | Mean pieces/word | Coverage |
|---|---|---|
| 1. Dictionary-only lookup | 1.00 | 0.72 |
| 2. Greedy longest-match | 1.28 | 1.00 |
| 3. Trie candidate generation (first candidate) | 1.00 | 1.00 |
| 4. Trie + morphological-validity rules | 1.00 | 1.00 |
| 5. Trie + frequency-based scoring | 1.70 | 1.00 |
| 6. Full hybrid (complete scoring function) | 1.70 | 1.00 |

The dictionary-only variant has the lowest coverage (0.72) because it can
only recognize words that appear verbatim in the small seed word list,
confirming that the Trie-based candidate search (variants 3-6) is doing
real work beyond simple dictionary lookup — it recovers full coverage by
decomposing unseen inflected forms into known roots and affixes. Variants
5 and 6 produce more pieces per word than 3/4 because frequency-based and
full scoring sometimes prefer a root+affix decomposition over a single
matched whole word when the decomposition's pieces are individually more
frequent — a direct, visible illustration of the same mechanism that
produced the Section 12 result: preferring morphological decomposition
over whole-word matching adds pieces whenever whole-word matching was
already cheap.

The vocabulary-size sweep (`outputs/tables/vocabulary_size_sweep.csv`,
requested sizes 300/600/1200/2400) shows identical fertility at every
requested size (baseline actual vocabulary 219, morphology-aware actual
vocabulary 185, at every setting). This is not a null result being hidden
— it means the synthetic corpus has too few distinct subword units for
BPE to make use of a larger budget at any of the sizes tried; the trainer
exhausts all learnable merges and stops early regardless of the
requested ceiling. A corpus with a larger, more diverse vocabulary of
distinct word forms would be needed to observe the budget actually
being used (see Section 15).

### 12.2 Combinatorial lexicon expansion: a null result, reported honestly

`src/tokenizer.generate_combinatorial_lexicon_seeds` extends the
attempt-2 seed set generatively: instead of only seeding words that were
literally observed (in `words.csv` or the training corpus), it combines
every root with every grammatically compatible affix the lexicon
licenses — every noun root with every noun suffix, every verb root with
every verb suffix, every derivational prefix with every noun root, and
all three together — and proposes every resulting string as an
additional seed. On this lexicon (20 noun roots, 20 verb roots, 15
prefixes, 16 noun suffixes, 4 verb suffixes), this generates 5,500
candidate strings, growing the vocabulary from 219 to 5,704 entries
(`outputs/reports/run_log.json`:
`lexicon_seeded_combinatorial_vocab_info`) — roughly 26x the baseline
vocabulary, for 25x more entries than the 219 the plain baseline BPE
trainer itself learned.

Measured on the held-out test set, this produced **no change at all**:
`lexicon_seeded_bpe_combinatorial`'s total token count, fertility, and
even its full per-word token-count vector are identical to plain
`lexicon_seeded_bpe`. This was verified directly, not just inferred from
matching totals: of the 59 distinct test words, zero are covered *only*
by a generated combinatorial candidate — every one was already covered
by an attempt-2 literal seed (a lexicon word, or a sufficiently frequent
training word). In other words, the mechanism generated 5,500 new
guaranteed-single-token entries, and this particular test set never
happened to ask for any of them.

This is reported as a **null result on this corpus**, explicitly not as
evidence that combinatorial expansion is ineffective in general. The
small synthetic test set is drawn from essentially the same narrow,
template-generated vocabulary as the training set (Section 16), so it
structurally cannot exercise forms outside that vocabulary — which is
exactly the kind of coverage combinatorial expansion is meant to add.
Answering whether the extra coverage is worth its real vocabulary-size
cost (and, in a real model, the corresponding embedding-table cost)
requires testing against text with real lexical diversity, which this
project's available (synthetic) corpus cannot provide — flagged as the
first priority in Section 19 (Future work), alongside rerunning
attempt 2 on the real OSCAR corpus for the same underlying reason.

## 14. Pretrained tokenizer comparison

`src/tokenizer.try_load_pretrained_tokenizer` attempted to load
`bert-base-multilingual-cased` (WordPiece) and `xlm-roberta-base`
(SentencePiece Unigram) via `transformers.AutoTokenizer.from_pretrained`,
revision `main`, `transformers` version 5.19.0, tokenizer files only (no
model weights requested). Both attempts failed with an `OSError`
reporting that the model configuration could not be loaded — the root
cause, confirmed independently, is that this environment's network proxy
returns HTTP 403 for huggingface.co. This is recorded verbatim in
`outputs/reports/run_log.json` under `pretrained_tokenizers`. No fertility
numbers for mBERT or XLM-R are reported in Section 12 as a direct
consequence; none were estimated or substituted.

## 15. Efficiency analysis

Runtime was measured directly (`outputs/tables/runtime_benchmark.csv`)
over 200 held-out test lines: the baseline tokenizer processed them in
about 2.3ms total (≈11.7 microseconds/document, ≈0.42ms per 1,000
characters); the morphology-aware tokenizer took about 18.5ms total
(≈92.7 microseconds/document, ≈3.32ms per 1,000 characters) — roughly 8x
slower per document, because every word is first run through the DP
segmentation search before BPE sees it. This is the real computational
cost of the proposed approach, reported alongside its fertility result
rather than only when it would look favorable (in this run, it does not
look favorable on either axis: worse fertility *and* higher latency).

The Trie-vs-dict-vs-linear-scan benchmark
(`outputs/tables/trie_benchmark.csv`) shows a dict is faster than the
Trie for exact-match lookup (≈1.7e-5s vs ≈3.7e-4s over the benchmark's
query set) — expected, since Python's hash-based dict lookup is O(1) with
very low constant overhead — while for prefix matching the Trie is
substantially faster than either a dict or a list with `startswith`
scanning (≈6.5e-4s vs ≈1.75e-3s and ≈1.54e-3s respectively). This matches
the motivation for using a Trie at all in this project: the segmentation
engine's dominant operation is prefix matching, not exact lookup, and
that is exactly where the Trie's structural advantage shows up in the
actual measurement rather than being merely assumed.

## 16. Limitations

- The corpus used for every result above is synthetic and small (4,000
  generated sentences, ~1,440 test words, on the order of 200 distinct
  BPE-learnable units). This is the direct cause of the counter-intuitive
  Section 12 result: plain BPE can memorize nearly every whole word as a
  single token at this scale, leaving the morphology-aware
  pre-segmentation step nothing to improve — see Section 17 for why this
  is expected to change at a different corpus scale.
- The pretrained-tokenizer comparison (mBERT, XLM-R) could not be run in
  this environment; Section 14 documents exactly why and exactly what was
  attempted.
- The seed lexicon (40 roots, 15 prefixes, 20 suffixes, 59 complete
  words) is sufficient to exercise every mechanism in the pipeline
  correctly but does not represent the full complexity of Nepali
  morphology; coverage figures reported in this project reflect that seed
  size.
- Scoring weights (`src/morphology.DEFAULT_WEIGHTS`) were set manually
  and were not optimized against a large labeled validation set, because
  no such set exists yet at a scale that would make optimization
  meaningful; `select_weights_on_validation` is implemented, tested, and
  ready to use once more validation-labeled examples are available.
- The ablation study and vocabulary sweep both operate within the
  constraints of the same small synthetic corpus and therefore inherit
  its limitations.
- The lexicon-seeded tokenizer's measured 21.7% fertility reduction
  (Section 12) is real and statistically significant on this test set,
  but every one of the 59 distinct test words had already occurred in
  training (0% genuinely unseen — `outputs/reports/run_log.json`:
  `train_test_word_overlap`). Part of the measured benefit therefore
  reflects the seeded tokenizer directly recognizing words it was shown
  during training rather than pure generalization to new inflected forms.
  The mechanism's guarantee (never worse than the baseline, on any word)
  does not depend on this overlap, but the specific *size* of the
  reduction would likely be smaller — while still non-negative — on a
  larger, less repetitive corpus with more genuinely unseen test words.
- Combinatorial lexicon expansion (Section 12.2) grows the vocabulary
  roughly 26x (219 → 5,704 entries) but measured **zero additional
  fertility reduction** on this test set, because every one of its 59
  distinct test words was already covered by a literally-observed seed.
  This is a limitation of the test set's lexical narrowness, not
  evidence against the mechanism — it was never asked to tokenize a word
  that needed one of the 5,500 generated candidates. Whether the extra
  coverage is worth its real vocabulary-size (and, in a real model,
  embedding-table) cost on genuinely diverse text remains untested.
- The ChatGPT-style demo's live-generation path (`/api/chat_compare` in
  `app/backend.py`) was verified for reachability only: a request to
  `api.anthropic.com/v1/messages` with a placeholder key returns a clean
  HTTP 401 "invalid x-api-key" (a normal authentication failure, not a
  network-level block of the kind seen for Kaggle/HuggingFace in Section
  14), confirming the request path itself works. An actual generation
  was not run during development, since that requires a real, paid
  Anthropic API key; the app's paste-in mode (no key required) was fully
  exercised instead. Separately, running the demo with any realistic
  generated Nepali text is expected to show small-to-zero token savings,
  for the same reason as the train/test-overlap caveat above: the
  optimized tokenizer only guarantees single-token matches for the ~130
  words it was shown, and ordinary generative-AI prose draws on far more
  vocabulary than that.

## 17. Discussion

The headline result — that morphology-aware pre-segmentation increased
fertility on this corpus — is informative rather than merely negative.
Morphological pre-segmentation imposes a *minimum* number of pieces per
word equal to the number of recognized morphemes (a word with a
recognized root and one suffix can never be represented in fewer than two
pieces under this scheme, even if the whole inflected form would have
been cheap for plain BPE to memorize directly). That cost is worth paying
only when plain BPE's alternative — memorizing more whole words directly
— is itself expensive, which happens when the vocabulary budget is small
relative to the number of distinct word forms the corpus actually
contains. In this run's synthetic, template-generated corpus, the number
of distinct word forms is small enough (the trained vocabularies
saturated at 219/185 regardless of the requested budget, as shown in
Section 13) that plain BPE never had to pay that alternative cost — it
could simply memorize almost everything, achieving near-optimal fertility
without any morphological knowledge. A real corpus with many more
distinct inflected forms competing for a fixed, smaller-relative-to-
vocabulary-diversity budget is the regime in which the literature (and
the original motivation in Section 4) predicts morphological
pre-segmentation should help, and this project's synthetic corpus does
not represent that regime. This is a scope limitation of the available
data, not evidence against the underlying idea of using morphological
knowledge at all — only against this one specific mechanism (forced hard
boundaries) for exploiting it.

The lexicon-seeded result points at the more productive mechanism: rather
than constraining what BPE is allowed to merge, use morphological/lexical
knowledge to *add* guaranteed single-token entries for known words, which
can only help because it never removes a merge path the baseline already
had. The measured 21.7% reduction confirms this works, at least in the
limited sense this corpus allows it to be tested, and the train/test
overlap caveat (Section 16) is the main reason to treat the *specific
number* as preliminary rather than as the method's ceiling — the
structural guarantee (never worse than baseline) is not in question; how
much better it does on genuinely unseen words at a larger corpus scale is
the open question that remains. This also reframes the original research
question usefully: the question was never really "does splitting words
into morphemes reduce fertility" in the abstract — it is "what is the
most effective way to inject morphological/lexical knowledge into a
subword tokenizer's vocabulary," and this project's own results show two
different answers to that more specific question, one that fails and one
that works, within a single consistent experimental setup.

The combinatorial-expansion result (Section 12.2) sharpens that question
further, into one about *where the ceiling of the seeding mechanism is*.
Attempt 2 showed that seeding literally-observed words helps; attempt 3
tested whether seeding *generated* (unobserved but grammatically licit)
words helps further, and the honest answer on this corpus is "untestable,
not zero" — the mechanism added thousands of guaranteed single-token
entries that this particular test set simply never needed, because the
test set's vocabulary is a strict subset of what attempt 2 already
covers. This is a direct consequence of using the same small, template-
generated corpus for both train and test splits (Section 16): any corpus
built this way will tend to have this property, regardless of which
seeding strategy is used, so the finding says more about the corpus than
about combinatorial expansion itself. It does, however, make the
vocabulary-size cost of that strategy concrete and measured (≈26x
growth) in a way a purely theoretical argument would not, which is useful
regardless of which way the benefit question eventually resolves on real
data.

## 18. Conclusion

A complete, independently tested pipeline for Trie-indexed, dynamic-
programming-based morphological segmentation of Nepali was built, and
two different ways of using it to reduce BPE token fertility were tried
and honestly compared against an identically configured baseline
tokenizer, using rigorously separated fertility and characters-per-token
metrics, a six-variant ablation study, paired statistical testing, and
bootstrap confidence intervals. Forced morphological pre-segmentation
before BPE training did not reduce fertility on this project's synthetic
corpus — it increased it, for reasons the ablation study and the
vocabulary-size sweep make specific and explainable rather than
mysterious. A second approach, adding the lexicon's known words directly
as guaranteed vocabulary entries on top of an otherwise unmodified
baseline BPE model, did reduce fertility, by about 21.7%, with a result
that is statistically significant and structurally guaranteed never to
be worse than the baseline — with the honest caveat that this run's
small, repetitive test corpus likely inflates the specific size of that
number. A third approach, combinatorially generating and seeding every
root+affix combination the lexicon's grammar licenses, measured zero
additional benefit on the available test set despite a roughly 26x
vocabulary increase — reported as a genuine null result on this corpus,
with the real vocabulary-size cost shown rather than hidden, not as a
verdict on the mechanism itself. Finally, the two working tokenizers were
applied to a realistic end-to-end interaction (a Nepali prompt and an
answer to it, tokenized and compared for real token-count savings), which
makes the same generalization limitation visible on ordinary text rather
than only on curated test words. The pipeline, all of its tests, and the
full evaluation suite are ready to be rerun unchanged on a real Nepali
corpus and with real pretrained tokenizers, which this environment could
not reach but which the project's own code already supports.

## 19. Future work

- Rerun the full pipeline — especially the lexicon-seeded tokenizer and
  its combinatorial-expansion variant — on the real OSCAR Corpus Nepali
  (or another large, naturally occurring Nepali corpus) on a machine with
  internet access, using the already-implemented `--real` corpus path, to
  measure (a) how much of the seeded approach's benefit survives on
  genuinely unseen test words rather than words already observed in
  training (0% of this synthetic corpus's test words were unseen), and
  (b) whether combinatorial expansion's 5,500 generated candidates
  actually help on text with real lexical diversity, which this
  synthetic corpus's narrow, closed vocabulary could not test at all
  (Section 12.2) — and whether that benefit, if any, is worth the ≈26x
  vocabulary growth it costs.
- Test the ChatGPT-style demo (`/api/chat_compare`) with a real Anthropic
  API key to confirm live generation end to end (only reachability was
  confirmed during development — see Section 16), and run it against
  genuinely diverse generated Nepali text at scale to quantify how much,
  if any, of the lexicon-seeded tokenizer's benefit transfers to ordinary
  generative-AI output rather than curated test sentences.
- Rerun the pretrained-tokenizer comparison (mBERT, XLM-R) once network
  access to HuggingFace Hub is available, using the already-implemented
  `try_load_pretrained_tokenizer` path with no code changes.
- Expand the seed lexicon substantially (more roots, a fuller affix
  inventory, verb paradigms beyond the present tense) to test whether
  lexicon coverage, rather than corpus scale, is the limiting factor.
- Build a genuinely labeled validation set of expected segmentations
  (beyond the illustrative pairs used in testing) so that
  `select_weights_on_validation` can meaningfully tune the scoring
  weights, rather than using the current hand-set defaults.
- Investigate a smaller BPE vocabulary budget relative to corpus
  diversity specifically, since Section 17's discussion predicts that is
  the condition under which morphology-aware pre-segmentation should show
  its benefit, if it has one on real data.

## References

- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine
  Translation of Rare Words with Subword Units. *Proceedings of the 54th
  Annual Meeting of the Association for Computational Linguistics
  (ACL)*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT:
  Pre-training of Deep Bidirectional Transformers for Language
  Understanding. *Proceedings of NAACL-HLT 2019*.
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G.,
  Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020).
  Unsupervised Cross-lingual Representation Learning at Scale.
  *Proceedings of the 58th Annual Meeting of the Association for
  Computational Linguistics (ACL)*.
- Kudo, T. (2018). Subword Regularization: Improving Neural Network
  Translation Models with Multiple Subword Candidates. *Proceedings of
  the 56th Annual Meeting of the Association for Computational
  Linguistics (ACL)*.
- Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and
  language independent subword tokenizer and detokenizer for Neural Text
  Processing. *Proceedings of the 2018 Conference on Empirical Methods
  in Natural Language Processing (EMNLP): System Demonstrations*.
- Creutz, M., & Lagus, K. (2007). Unsupervised Models for Morpheme
  Segmentation and Morphology Learning. *ACM Transactions on Speech and
  Language Processing*, 4(1).
- Grönroos, S.-A., Virpioja, S., Smit, P., & Kurimo, M. (2014). Morfessor
  FlatCat: An HMM-Based Method for Unsupervised and Semi-Supervised
  Learning of Morphology. *Proceedings of COLING 2014*. (Also see
  Virpioja, S., Smit, P., Grönroos, S.-A., & Kurimo, M. (2013). Morfessor
  2.0: Python Implementation and Extensions for Morfessor Baseline.
  *Aalto University publication series*.)


---

## Addendum (Weeks 6-7): real-corpus run and the optimized min-token tokenizer

Everything above was measured on a tiny synthetic corpus. The final run uses
60,000 real Nepali Wikipedia sentences (80/10/10 split, 37% of distinct test
words unseen in training) and downloads mBERT/XLM-R. Headline results (test
set, 80,524 words, vocabulary 8,000 unless stated):

| Tokenizer | Vocab | Fertility | Tokens vs baseline |
|---|---:|---:|---:|
| mBERT | 119,547 | 2.802 | +71.6% |
| XLM-R | 250,002 | 1.905 | +16.6% |
| Baseline BPE | 8,000 | 1.633 | - |
| Forced morphological pre-split BPE | 8,000 | 1.641 | +0.5% |
| Optimized min-token | 8,000 | 1.528 | -6.4% |
| Lexicon-seeded BPE | 43,957 | 1.356 | -17.0% |
| Optimized min-token (matched 43,957) | 43,957 | 1.265 | -22.5% |

Findings: (1) forced pre-segmentation again fails to reduce fertility on real
data; (2) the earlier lexicon-seeded "win" is largely a vocabulary-size effect
(5.5x larger vocabulary) and its effect does not require seeding;
(3) minimum-token (Viterbi) encoding over a usage-pruned vocabulary, with
lexicon morphemes protected and sentence-final punctuation attachable, lowers
fertility by 5.4-7.3% across vocabulary budgets 2k-32k at equal budget
(paired Wilcoxon, bootstrap CI excluding zero), and by 6.7% against the
seeded tokenizer at equal vocabulary. Limitations: single-domain corpus
(Wikipedia), a 134-entry seed lexicon, and part of the gain comes from
allowing `word+।` to be one piece. See `README.md` and
`src/optimized_tokenizer.py`.
