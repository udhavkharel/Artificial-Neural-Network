# Nepali Morphology-Aware Indexed Tokenizer for Token Fertility Optimization

This project builds a morphology-aware subword tokenization pipeline for
Nepali and tests whether splitting a word into its root and known
grammatical affixes (prefixes/suffixes) *before* training a byte-pair-
encoding (BPE) tokenizer reduces **token fertility** (tokens produced per
word) compared to training BPE on the raw text directly.

The pipeline:

1. A small, hand-assembled **seed lexicon** of Nepali roots, prefixes,
   suffixes, and complete words (`data/lexicons/*.csv`), explicitly
   labeled as seed data rather than a comprehensive dictionary.
2. A **Trie** index (`src/trie_index.py`) over that lexicon, for both
   exact lookup and prefix matching.
3. A **dynamic-programming segmentation engine** (`src/morphology.py`)
   that proposes every valid root/affix split of a word, scores each one
   with an explicit, inspectable scoring function, and selects the best
   one, with a documented unknown-word fallback and a guaranteed
   reversibility property (the pieces always reconstruct the original
   word).
4. Three **BPE tokenizers** (`src/tokenizer.py`): a baseline, a
   morphology-aware variant trained on forcibly pre-split text, and a
   lexicon-seeded variant that adds known whole words directly as
   guaranteed single-token vocabulary entries on top of an otherwise
   identical baseline model. See "Two optimization attempts" below —
   only the third one actually reduces fertility.
5. An **evaluation layer** (`src/evaluation.py`) computing fertility,
   characters-per-token, coverage, unknown-token rate, a 6-variant
   ablation study, bootstrap confidence intervals, and a paired
   Wilcoxon + Holm-corrected statistical comparison.

Everything above is exercised end to end in
`notebooks/nepali_morphology_tokenizer.ipynb`, which also contains an
interactive (`ipywidgets`, with a plain-function fallback) demo.

## Final result (Weeks 6-7): the optimized min-token tokenizer

Run on a **real** corpus: 60,000 Nepali Wikipedia sentences (HuggingFace
`wikimedia/wikipedia`, `20231101.ne`), split 80/10/10 by sentence. The
held-out test set has 80,524 words, **37% of whose distinct forms never
occur in training** (so this is a real generalization test, unlike the
earlier synthetic corpus where 0% were unseen). mBERT and XLM-R are
downloaded and compared directly.

| Tokenizer | Vocab | Tokens (test) | Fertility | vs baseline |
|---|---:|---:|---:|---:|
| mBERT | 119,547 | 225,601 | 2.802 | +71.6% |
| XLM-R | 250,002 | 153,391 | 1.905 | +16.6% |
| Baseline BPE | 8,000 | 131,494 | 1.633 | - |
| Morphology-aware BPE (forced split) | 8,000 | 132,114 | 1.641 | +0.5% (worse) |
| **Optimized min-token (ours)** | **8,000** | **123,060** | **1.528** | **-6.4%** |
| Lexicon-seeded BPE (adds tokens) | 43,957 | 109,207 | 1.356 | -17.0% |
| **Optimized min-token, matched 43,957 vocab** | 43,957 | 101,899 | **1.265** | **-22.5%** |

* Same 8,000-entry budget as the baseline: **6.4% fewer tokens** than BPE
  (paired Wilcoxon p < 1e-300, Holm-significant; bootstrap 95% CI for the per-word saving
  [0.102, 0.108] tokens), **45% fewer than mBERT, 20% fewer than XLM-R**
  (with 15x / 31x smaller vocabularies).
* Same vocabulary as the lexicon-seeded tokenizer: **6.7% fewer tokens**
  than it (the seeded tokenizer's 17% gain over baseline comes mostly from
  a 5.5x larger vocabulary; this is stated rather than hidden).
* The gain over baseline BPE holds across budgets (see
  `outputs/tables/vocabulary_size_sweep.csv`): 5.4% at 2k, 6.1% at 4k,
  6.4% at 8k, 6.9% at 16k, 7.3% at 32k.

**How it works** (`src/optimized_tokenizer.py`):

1. *Minimum-token (Viterbi) encoding*: each word is segmented by dynamic
   programming into the fewest vocabulary pieces; ties prefer lexicon
   morpheme boundaries (घर+हरू+लाई), then longer pieces. Because all BPE
   pieces are in the vocabulary, it can never need more tokens than BPE
   with the same vocabulary.
2. *Usage-pruned vocabulary*: start from a large pool (BPE at 2.5x budget +
   frequent words + lexicon morphemes) and repeatedly drop the pieces whose
   removal costs the fewest extra training-corpus tokens, until exactly the
   budget remains. Characters and lexicon morphemes are protected, so any
   input is representable and unseen inflected forms still split into
   root+suffix.
3. *Attachable punctuation*: whitespace-delimited chunks are encoded
   directly, so a frequent sentence-final `word+।` can be one piece
   (BPE baselines always emit the danda as a separate token). Part of the
   6.4% comes from this; it is a legitimate use of the same vocabulary budget.

Tokenization is lossless: `"".join(tokens) == word` for any input.

Earlier attempts (forced morphological pre-splitting, lexicon-seeded
vocabulary, combinatorial expansion) are kept, reported in
`research_report.md`, and still run by `scripts_build/run_experiment.py`.
Forced splitting did not reduce fertility (it slightly increases it);
seeding reduces it only by enlarging the vocabulary.

> Caveat: the lexicon is a 134-entry hand-built seed list and Wikipedia is
> a single domain; numbers on other text (news, social media) will differ.
> `build_corpus.py --real` (OSCAR via Kaggle) is still supported. The
> `research_report.md` and the "attempts" numbers in older notebook text
> refer to the earlier tiny synthetic corpus run.

## Project layout

```
nepali_morph_tokenizer/
├── README.md
├── research_report.md
├── requirements.txt
├── config.json
├── data/
│   ├── raw/                 # (empty here; real-mode downloads land in a system cache dir)
│   ├── processed/           # train.txt / val.txt / test.txt / corpus_meta.json
│   └── lexicons/            # roots.csv, prefixes.csv, suffixes.csv, words.csv (seed data)
├── src/
│   ├── normalization.py     # Unicode NFC normalization, whitespace/word utilities
│   ├── trie_index.py        # Trie + lookup-method benchmark
│   ├── morphology.py        # lexicon loading, DP segmentation, scoring, evaluation groups
│   ├── tokenizer.py         # baseline/morphology-aware BPE training + pretrained comparison
│   └── evaluation.py        # fertility metrics, ablation scoring, bootstrap CIs, Holm correction
├── tests/                   # pytest suite — normalization, Trie, morphology, tokenizer, evaluation
├── notebooks/
│   └── nepali_morphology_tokenizer.ipynb   # main entry point, executed end to end
├── models/                  # saved tokenizer JSON files (written by scripts_build/run_experiment.py)
├── outputs/
│   ├── figures/             # 9 PNG figures
│   ├── tables/              # every CSV result table
│   └── reports/             # run_log.json (full run metadata, including failures)
└── scripts_build/           # the scripts that build the lexicon, corpus, notebook, and run the experiment
    ├── make_lexicons.py
    ├── build_corpus.py
    ├── run_experiment.py
    └── build_notebook.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 -m ipykernel install --user --name python3   # needed once, to run the notebook
```

## Running everything from scratch

All paths in every script are built with `pathlib` relative to the
project root — nothing is hard-coded to a particular machine.

```bash
# 1) Build the seed lexicon CSVs
python3 scripts_build/make_lexicons.py

# 2) Build the corpus.
#    This machine (no internet to Kaggle):
python3 scripts_build/build_corpus.py --synthetic --sample_size 4000
#    Real Nepali Wikipedia (default for the delivered results; no credentials):
python3 scripts_build/build_corpus.py --wikipedia --sample_size 60000
#    A machine with internet access and Kaggle credentials
#    (~/.kaggle/kaggle.json), to use the real OSCAR Corpus Nepali instead:
python3 scripts_build/build_corpus.py --real --sample_size 100000

# 3) Run the full experiment: trains both tokenizers, attempts the
#    pretrained-tokenizer comparison, computes every metric, runs the
#    ablation study and the vocabulary-size sweep, and writes every
#    table/figure under outputs/.
python3 scripts_build/run_experiment.py

# 4) (Re)build the notebook from the template script, if you've changed
#    anything in scripts_build/build_notebook.py:
python3 scripts_build/build_notebook.py

# 5) Run the tests
python3 -m pytest tests/ -q

# 6) Open and run the notebook
jupyter notebook notebooks/nepali_morphology_tokenizer.ipynb
```

## Running the comparison web app (a UI, no Jupyter needed)

`app/` is a small local web app with a browser UI that loads all three
trained tokenizers from `models/` and the result tables from
`outputs/tables/`, and lets you type any Nepali text to see, live, how
the baseline, the morphology-aware (forced-split), and the lexicon-seeded
(optimized) tokenizers each handle it, side by side, with token counts
and the difference between them. It also displays the project's actual
measured overall numbers rather than recomputing anything.

It requires steps 1-3 above to have been run at least once (so that
`models/baseline_bpe.json`, `models/morphology_aware_bpe.json`,
`models/lexicon_seeded_bpe.json`, `models/lexicon_seeded_bpe_combinatorial.json`,
and `outputs/tables/*.csv` exist) — they already have been in the
delivered project, so you can run the app immediately.

```bash
pip install -r requirements.txt    # includes fastapi + uvicorn
cd app
uvicorn backend:app --port 8000
```

Then open **http://localhost:8000** in a browser. Use `--reload` instead
of a plain port if you plan to edit `app/backend.py` or
`app/static/index.html` and want the server to pick up changes
automatically: `uvicorn backend:app --reload --port 8000`.

**What the app shows, and why it's not a one-sided "win" screen:** the
live comparison panel shows real tokenizer output for whatever you type,
for all three tokenizers. The "Overall measured result" panel shows the
project's actual test-set numbers: the morphology-aware (forced-split)
tokenizer really did increase fertility (not a decrease), and the app
says so plainly with the mechanical reason, rather than hiding it — see
`research_report.md`, Sections 12 and 17. The lexicon-seeded tokenizer
really did reduce fertility by about 21.7%, with its own honest caveat
shown directly in the app: on this run's small corpus, every distinct
test word had already appeared in training, which inflates how much of
that number is true generalization versus memorization of training
vocabulary. The app's third panel shows a separate, additional win:
vocabulary efficiency — the 134-entry seed lexicon achieves full coverage
of inflected forms it was never shown as whole words, while a plain
dictionary lookup only recognizes exact forms it has memorized. A fourth
panel reports attempt 3 (combinatorial expansion) the same way — the
25x vocabulary growth and the measured zero-additional-benefit result,
side by side, not hidden because the result was flat.

**ChatGPT-style demo panel.** The app's last panel lets you type a
Nepali prompt and either paste in an answer you already have (from
claude.ai, ChatGPT, or anywhere else) or paste your own Anthropic API
key to fetch a real answer live from `api.anthropic.com` — the key is
sent directly to Anthropic for that one request only; this app never
stores it, logs it, or writes it to disk. Either way, both the prompt
and the answer are tokenized with the baseline tokenizer and with the
optimized (lexicon-seeded) tokenizer, and the panel reports the actual
token-count savings for the input, the output, and the total — a
realistic end-to-end demonstration, not just an isolated test sentence.
Backed by `/api/chat_compare` in `app/backend.py`. **Honest note**: this
sandbox confirmed `api.anthropic.com` is reachable (a request with a
placeholder key returns a clean HTTP 401, not a network-level block),
but live generation itself was not exercised during development, since
that needs a real, paid API key — the paste-in mode needs no key at all
and was fully tested. Also expect savings on realistic generated answers
to be small or zero most of the time: the optimized tokenizer's seeded
vocabulary only guarantees single-token matches for the ~130 words it
was shown, and ordinary generative-AI prose uses far more vocabulary
than that — the same generalization limitation documented above, now
visible on a real example instead of a curated one.

Running the whole sequence (1)-(5) reproduces every table and figure
under `outputs/` exactly (fixed random seed 42 throughout). The notebook
itself was executed top-to-bottom from a freshly started kernel as a
verification step before this project was delivered, with zero cell
errors and all 56 tests passing from inside the notebook.

## On the pretrained-tokenizer comparison

`src/tokenizer.py`'s `try_load_pretrained_tokenizer` uses
`transformers.AutoTokenizer.from_pretrained` with the exact model
identifiers `bert-base-multilingual-cased` and `xlm-roberta-base`,
revision `main`, downloading tokenizer files only (no model weights are
needed or fetched). On a machine with normal internet access this
succeeds automatically and the fertility tables/figures include mBERT and
XLM-R as additional rows with no code changes required.

## On the lexicon

`data/lexicons/*.csv` is seed data: 40 roots, 15 prefixes, 20 suffixes,
59 fully analyzed complete words, assembled and checked by hand (see the
docstring in `scripts_build/make_lexicons.py` for the exact list and
reasoning). It is enough to exercise every mechanism in the pipeline
correctly end to end, but it does not claim to cover Nepali morphology
comprehensively. `src/morphology.update_frequencies_from_corpus` replaces
the seed (placeholder) frequency numbers with real corpus counts wherever
a lexicon entry is actually observed in the training corpus, so frequency
-based scoring reflects the corpus actually used, not just the seed
author's guesses, for any entry the corpus covers.

## Troubleshooting

- `ModuleNotFoundError` for any package in `requirements.txt`: run
  `pip install -r requirements.txt` inside the virtual environment you're
  using to run these scripts/the notebook.
- `No such kernel named python3` when opening the notebook: run
  `python3 -m ipykernel install --user --name python3` once.
- The pretrained-tokenizer cells report `unavailable`: this means there
  is no network access to HuggingFace Hub from wherever this is running
  — check your internet connection, or run on a machine that has one; the
  rest of the notebook is unaffected and continues with the two local BPE
  tokenizers.
- `build_corpus.py --real` fails with a kagglehub/auth error: make sure
  `~/.kaggle/kaggle.json` (or the Windows equivalent) contains valid
  Kaggle API credentials.
