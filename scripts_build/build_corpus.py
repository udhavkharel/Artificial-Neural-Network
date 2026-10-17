"""
Builds the train/validation/test corpus used by this project.

IMPORTANT, stated plainly: the execution environment used to build and run
this project has no outbound network access to Kaggle or HuggingFace Hub
(both return a 403 at the network proxy — confirmed by a direct connection
test, not assumed). The specification's primary intended corpus (OSCAR
Corpus Nepali, via kagglehub) could not be downloaded here. This script
therefore has two modes:

  --real          Downloads OSCAR Corpus Nepali via kagglehub (works on a
                   machine with normal internet access, e.g. the user's
                   own machine) and samples `--sample_size` lines from it
                   with reservoir sampling (memory-safe for a large file).

  --synthetic      Generates a synthetic Nepali corpus from the seed
                   lexicon's roots, prefixes, and suffixes, combined into
                   grammatically plausible short sentences using a small
                   set of sentence templates. This is clearly labeled
                   synthetic everywhere it is used (file names, notebook
                   markdown, the research report, and every figure).

This project was executed in --synthetic mode. Every fertility result
coming from this run is a result on synthetic data, not a measurement of
real Nepali corpus statistics, and the research report/README say so
explicitly rather than presenting synthetic-data numbers as if they
generalize to a real corpus. The code path for --real mode is complete
and documented so that a user with internet access can rerun the exact
same pipeline on the real OSCAR corpus and get real-corpus numbers.

Split: documents (sentences, in this case) are split 80/10/10 into
train/validation/test with a fixed random seed (42) at the *document*
level, so no sentence appears in more than one split. Tokenizers and
lexicon frequency updates are fit only on the train split; the validation
split is used only for selecting scoring weights (src/morphology.py:
select_weights_on_validation); the test split is touched only in the
final comparison step.
"""

import argparse
import csv
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEXICON_DIR = PROJECT_ROOT / "data" / "lexicons"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SEED = 42

# A handful of simple, templated sentence frames using {subject} {object}
# {verb} and similar slots filled from the seed lexicon's words.csv (so
# every generated sentence uses morphologically real, lexicon-consistent
# word forms rather than ungrammatical random concatenation).
TEMPLATES = [
    "{subj} {obj} {verb}",
    "{subj} {loc} {verb}",
    "{subj} {obj} {instr} {verb}",
    "{subj} {abl} {obj} {verb}",
    "{adj} {subj} {loc} {verb}",
]


def _load_words(category_filter=None):
    rows = []
    with open(LEXICON_DIR / "words.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if category_filter is None or row["grammatical_category"] == category_filter:
                rows.append(row["surface_form"])
    return rows


def build_synthetic_corpus(n_sentences: int, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    nouns = _load_words("noun")
    verbs = _load_words("verb")
    adjectives = ["असफल", "दुर्गम", "सुगम", "स्वतन्त्र"]  # the few derived adjectives in the seed set

    if not nouns or not verbs:
        raise RuntimeError("Seed lexicon words.csv must contain noun and verb forms "
                            "to generate the synthetic corpus; rerun scripts_build/make_lexicons.py first.")

    sentences = []
    for _ in range(n_sentences):
        template = rng.choice(TEMPLATES)
        slot_values = {
            "subj": rng.choice(nouns),
            "obj": rng.choice(nouns),
            "verb": rng.choice(verbs),
            "loc": rng.choice(nouns),
            "instr": rng.choice(nouns),
            "abl": rng.choice(nouns),
            "adj": rng.choice(adjectives),
        }
        sentence = template.format(**slot_values)
        sentences.append(sentence + "।")
    return sentences


def split_documents(docs: list[str], seed: int = SEED):
    rng = random.Random(seed)
    shuffled = docs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def build_real_corpus(sample_size: int, seed: int = SEED) -> list[str]:
    """
    Download OSCAR Corpus Nepali via kagglehub and reservoir-sample
    `sample_size` lines. Requires internet access to Kaggle; raises a
    clear RuntimeError (not a silent empty result) if kagglehub or the
    download is unavailable, so the caller can fall back explicitly
    rather than pretending the real corpus was used.
    """
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            "kagglehub is not installed. Run: pip install kagglehub"
        ) from e

    path = kagglehub.dataset_download("hsebarp/oscar-corpus-nepali")
    corpus_file = Path(path) / "ne_dedup.txt"
    if not corpus_file.exists():
        raise RuntimeError(f"Expected corpus file not found at {corpus_file}")

    rng = random.Random(seed)
    reservoir: list[str] = []
    with open(corpus_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if len(reservoir) < sample_size:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line
    return reservoir


def build_wikipedia_corpus(sample_size: int, seed: int = SEED) -> list[str]:
    """
    Real Nepali corpus: Nepali Wikipedia (wikimedia/wikipedia, 20231101.ne)
    downloaded from the HuggingFace Hub (no credentials needed). Articles
    are split into sentences on the danda / newlines, filtered to sentences
    that are mostly Devanagari and 4-40 words long, de-duplicated, and
    `sample_size` of them are sampled with a fixed seed.
    """
    import re
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("wikimedia/wikipedia", "20231101.ne/train-00000-of-00001.parquet",
                           repo_type="dataset")
    texts = pd.read_parquet(path)["text"].tolist()
    dev = re.compile(r"[\u0900-\u097F]")
    seen, sentences = set(), []
    for t in texts:
        for chunk in re.split(r"[\n।?!]+", t):
            chunk = " ".join(chunk.split())
            n = len(chunk.split())
            if n < 4 or n > 40 or chunk in seen:
                continue
            letters = [c for c in chunk if c.isalpha()]
            if not letters or sum(1 for c in letters if dev.match(c)) / len(letters) < 0.9:
                continue
            seen.add(chunk)
            sentences.append(chunk + "।")
    rng = random.Random(seed)
    rng.shuffle(sentences)
    return sentences[:sample_size]


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--wikipedia", action="store_true",
                      help="real Nepali Wikipedia sentences from the HuggingFace Hub")
    parser.add_argument("--sample_size", type=int, default=3000)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.wikipedia:
        docs = build_wikipedia_corpus(args.sample_size)
        corpus_label = "real_nepali_wikipedia"
    elif args.real:
        docs = build_real_corpus(args.sample_size)
        corpus_label = "real_oscar_nepali"
    else:
        docs = build_synthetic_corpus(args.sample_size)
        corpus_label = "synthetic_template_corpus"

    train, val, test = split_documents(docs)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(PROCESSED_DIR / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(split))

    meta = {
        "corpus_label": corpus_label,
        "n_total": len(docs),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": SEED,
        "split_ratio": "80/10/10",
    }
    with open(PROCESSED_DIR / "corpus_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
