"""
Runs the full experiment: trains baseline and morphology-aware BPE,
attempts to load pretrained tokenizers (mBERT, XLM-R), computes fertility
for every tokenizer on the held-out test words, runs the 6-variant
ablation study, computes the Trie-vs-dict-vs-linear-scan benchmark, runs
the statistical tests (Wilcoxon + Holm), computes bootstrap confidence
intervals, and saves every table (CSV) and figure (PNG) under outputs/.

This script is what the notebook calls into (via `import`), so that the
notebook's cells show real executed output and so the same code is never
duplicated between the notebook and a standalone script.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    bootstrap_ci_mean_difference,
    compute_fertility,
    exclude_special,
    holm_correct,
    reduction_percentage,
)
from src.morphology import classify_word_group, load_lexicon, select_best_segmentation
from src.normalization import normalize_text, whitespace_words
from src.optimized_tokenizer import OptimizedMorphTokenizer, train_optimized_tokenizer
from src.tokenizer import (
    PRETRAINED_MODELS,
    TokenizerConfig,
    load_tokenizer,
    save_tokenizer,
    tokenize_word_baseline,
    tokenize_word_morphology_aware,
    train_bpe,
    train_lexicon_seeded_bpe,
    try_load_pretrained_tokenizer,
)
from src.trie_index import Trie, benchmark_lookup_methods

LEXICON_DIR = PROJECT_ROOT / "data" / "lexicons"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

for d in (MODELS_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _default_vocab_size() -> int:
    """1200 for the tiny synthetic corpus, 8000 for a real corpus."""
    try:
        label = json.loads((PROCESSED_DIR / "corpus_meta.json").read_text())["corpus_label"]
    except Exception:
        return 1200
    return 1200 if label.startswith("synthetic") else 8000


VOCAB_SIZE = _default_vocab_size()
SEED = 42


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [normalize_text(l) for l in f.read().splitlines() if l.strip()]


def save_table(rows: list[dict], name: str):
    import csv
    path = TABLES_DIR / name
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    log = {"steps": []}

    # ---- Step 1: load lexicon and corpus -------------------------------
    lexicon = load_lexicon(LEXICON_DIR)
    train_lines = read_lines(PROCESSED_DIR / "train.txt")
    val_lines = read_lines(PROCESSED_DIR / "val.txt")
    test_lines = read_lines(PROCESSED_DIR / "test.txt")
    corpus_meta = json.loads((PROCESSED_DIR / "corpus_meta.json").read_text())
    log["corpus_meta"] = corpus_meta
    print(f"corpus: {corpus_meta}")

    test_words = [w for line in test_lines for w in whitespace_words(line) if w]
    val_words = [w for line in val_lines for w in whitespace_words(line) if w]
    print(f"test words: {len(test_words)}  val words: {len(val_words)}")

    # ---- Step 2: update lexicon frequencies from train corpus -----------
    from collections import Counter
    train_words = [w for line in train_lines for w in whitespace_words(line) if w]
    train_word_counts = Counter(train_words)
    from src.morphology import update_frequencies_from_corpus
    update_frequencies_from_corpus(lexicon, dict(train_word_counts))

    # ---- Step 3: train baseline and morphology-aware BPE -----------------
    config = TokenizerConfig(vocab_size=VOCAB_SIZE, min_frequency=1)

    t0 = time.perf_counter()
    baseline_tok = train_bpe(train_lines, config, morphology_presegment=False)
    baseline_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    morph_tok = train_bpe(train_lines, config, morphology_presegment=True, lexicon=lexicon)
    morph_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    seeded_tok, seed_info = train_lexicon_seeded_bpe(
        train_lines, config, lexicon, min_train_frequency=2, combinatorial_expansion=False,
    )
    seeded_train_time = time.perf_counter() - t0

    # Third attempt (an extension of the second): the same add_tokens
    # seeding mechanism, but the seed set is additionally expanded with
    # every root+affix combination the lexicon's grammar licenses (see
    # generate_combinatorial_lexicon_seeds), not just words literally
    # observed in training or in the fixed word list. Reported as its own
    # tokenizer, not a silent replacement, so the vocabulary-size cost of
    # combinatorial expansion is visible side by side with its benefit.
    t0 = time.perf_counter()
    combo_tok, combo_info = train_lexicon_seeded_bpe(
        train_lines, config, lexicon, min_train_frequency=2, combinatorial_expansion=True,
    )
    combo_train_time = time.perf_counter() - t0

    # The seeded tokenizers *add* thousands of whole words on top of the
    # base BPE budget, so their real vocabulary is much larger than
    # VOCAB_SIZE. For a fair comparison, also train the optimized tokenizer
    # at exactly the seeded tokenizer's vocabulary size.
    matched_cfg = TokenizerConfig(vocab_size=seeded_tok.get_vocab_size(), min_frequency=1)
    opt_matched_tok, opt_matched_info = train_optimized_tokenizer(train_lines, matched_cfg, lexicon)
    opt_matched_tok.save(MODELS_DIR / "optimized_morph_tokenizer_matched_vocab.json")

    # Final optimized tokenizer: same vocabulary budget as the baseline,
    # min-token (Viterbi) encoding + usage-pruned vocabulary + lexicon priors.
    t0 = time.perf_counter()
    opt_tok, opt_info = train_optimized_tokenizer(train_lines, config, lexicon)
    opt_train_time = time.perf_counter() - t0
    opt_tok.save(MODELS_DIR / "optimized_morph_tokenizer.json")

    save_tokenizer(baseline_tok, MODELS_DIR / "baseline_bpe.json")
    save_tokenizer(morph_tok, MODELS_DIR / "morphology_aware_bpe.json")
    save_tokenizer(seeded_tok, MODELS_DIR / "lexicon_seeded_bpe.json")
    save_tokenizer(combo_tok, MODELS_DIR / "lexicon_seeded_bpe_combinatorial.json")
    log["tokenizer_training"] = {
        "vocab_size": VOCAB_SIZE,
        "baseline_train_seconds": baseline_train_time,
        "morphology_aware_train_seconds": morph_train_time,
        "lexicon_seeded_train_seconds": seeded_train_time,
        "lexicon_seeded_combinatorial_train_seconds": combo_train_time,
        "baseline_actual_vocab": baseline_tok.get_vocab_size(),
        "morphology_aware_actual_vocab": morph_tok.get_vocab_size(),
        "lexicon_seeded_vocab_info": seed_info,
        "lexicon_seeded_combinatorial_vocab_info": combo_info,
        "optimized_train_seconds": opt_train_time,
        "optimized_vocab_info": opt_info,
        "optimized_matched_vocab_info": opt_matched_info,
    }
    print("trained baseline + morphology-aware + lexicon-seeded + combinatorial BPE:",
          log["tokenizer_training"])

    # Train/test word overlap -- this matters for interpreting the
    # lexicon-seeded tokenizer's result honestly: seeding adds frequent
    # TRAIN words as atomic tokens, so if most TEST words already occurred
    # in TRAIN, the measured benefit partly reflects that overlap rather
    # than pure generalization to unseen words. Reported, not hidden.
    distinct_test_words = set(test_words)
    unseen_in_train = sum(1 for w in distinct_test_words if w not in train_word_counts)
    overlap_info = {
        "distinct_test_words": len(distinct_test_words),
        "distinct_test_words_unseen_in_train": unseen_in_train,
        "distinct_test_words_unseen_fraction": round(
            unseen_in_train / len(distinct_test_words), 4
        ) if distinct_test_words else None,
    }
    log["train_test_word_overlap"] = overlap_info
    print("train/test word overlap:", overlap_info)

    # ---- Step 4: pretrained tokenizer comparison (network-gated) ---------
    pretrained_info = {}
    pretrained_tokenizers = {}
    for name in PRETRAINED_MODELS:
        tok, info = try_load_pretrained_tokenizer(name)
        pretrained_info[name] = info
        if tok is not None:
            pretrained_tokenizers[name] = tok
        print(f"pretrained[{name}]: {info['status']}" +
              (f" ({info['error']})" if info["error"] else ""))
    log["pretrained_tokenizers"] = pretrained_info

    # ---- Step 5: fertility for baseline + morphology-aware on test words -
    def _baseline_fn(w):
        return tokenize_word_baseline(baseline_tok, w)

    def _morph_fn(w):
        return tokenize_word_morphology_aware(morph_tok, lexicon, w)

    def _seeded_fn(w):
        return tokenize_word_baseline(seeded_tok, w)

    def _combo_fn(w):
        return tokenize_word_baseline(combo_tok, w)

    def _opt_fn(w):
        return opt_tok.tokenize(w)

    fertility_rows = []
    per_word_counts = {}
    vocab_sizes = {
        "baseline_bpe": baseline_tok.get_vocab_size(),
        "morphology_aware_bpe": morph_tok.get_vocab_size(),
        "lexicon_seeded_bpe": seeded_tok.get_vocab_size(),
        "lexicon_seeded_bpe_combinatorial": combo_tok.get_vocab_size(),
        "optimized_morph_tokenizer": opt_tok.vocab_size(),
        "optimized_morph_tokenizer_matched_vocab": opt_matched_tok.vocab_size(),
    }

    for tok_name, fn in [("baseline_bpe", _baseline_fn), ("morphology_aware_bpe", _morph_fn),
                          ("lexicon_seeded_bpe", _seeded_fn),
                          ("lexicon_seeded_bpe_combinatorial", _combo_fn),
                          ("optimized_morph_tokenizer", _opt_fn),
                          ("optimized_morph_tokenizer_matched_vocab", opt_matched_tok.tokenize)]:
        report = compute_fertility(test_words, fn)
        per_word_counts[tok_name] = report.per_word_token_counts
        fertility_rows.append({
            "tokenizer": tok_name,
            "vocab_size": vocab_sizes[tok_name],
            "total_tokens": report.total_tokens,
            "total_words": report.total_words,
            "token_fertility": round(report.token_fertility, 4),
            "chars_per_token": round(report.chars_per_token, 4),
            "mean_tokens_per_word": round(report.mean_tokens_per_word, 4),
            "median_tokens_per_word": report.median_tokens_per_word,
            "p95_tokens_per_word": report.p95_tokens_per_word,
            "unknown_token_rate": round(report.unknown_token_rate, 4),
        })

    for name, tok in pretrained_tokenizers.items():
        def _fn(w, tok=tok):
            return tok.tokenize(w)
        report = compute_fertility(test_words, _fn)
        per_word_counts[name] = report.per_word_token_counts
        fertility_rows.append({
            "tokenizer": name,
            "vocab_size": len(tok),
            "total_tokens": report.total_tokens,
            "total_words": report.total_words,
            "token_fertility": round(report.token_fertility, 4),
            "chars_per_token": round(report.chars_per_token, 4),
            "mean_tokens_per_word": round(report.mean_tokens_per_word, 4),
            "median_tokens_per_word": report.median_tokens_per_word,
            "p95_tokens_per_word": report.p95_tokens_per_word,
            "unknown_token_rate": round(report.unknown_token_rate, 4),
        })

    save_table(fertility_rows, "fertility_comparison.csv")

    baseline_total = next(r["total_tokens"] for r in fertility_rows if r["tokenizer"] == "baseline_bpe")
    morph_total = next(r["total_tokens"] for r in fertility_rows if r["tokenizer"] == "morphology_aware_bpe")
    seeded_total = next(r["total_tokens"] for r in fertility_rows if r["tokenizer"] == "lexicon_seeded_bpe")
    combo_total = next(r["total_tokens"] for r in fertility_rows if r["tokenizer"] == "lexicon_seeded_bpe_combinatorial")
    reduction_morph = reduction_percentage(baseline_total, morph_total)
    reduction_seeded = reduction_percentage(baseline_total, seeded_total)
    reduction_combo = reduction_percentage(baseline_total, combo_total)
    log["reduction_percentage_morph_vs_baseline"] = reduction_morph
    log["reduction_percentage_seeded_vs_baseline"] = reduction_seeded
    log["reduction_percentage_seeded_combinatorial_vs_baseline"] = reduction_combo
    opt_total = next(r["total_tokens"] for r in fertility_rows if r["tokenizer"] == "optimized_morph_tokenizer")
    log["reduction_percentage_optimized_vs"] = {
        r["tokenizer"]: round(reduction_percentage(r["total_tokens"], opt_total), 2)
        for r in fertility_rows if r["tokenizer"] != "optimized_morph_tokenizer"
    }
    om_total = next(r["total_tokens"] for r in fertility_rows
                    if r["tokenizer"] == "optimized_morph_tokenizer_matched_vocab")
    log["reduction_percentage_optimized_matched_vocab_vs_seeded"] = round(
        reduction_percentage(seeded_total, om_total), 2)
    print("matched-vocab optimized vs lexicon_seeded_bpe reduction %:",
          log["reduction_percentage_optimized_matched_vocab_vs_seeded"])
    print("OPTIMIZED token reduction % vs each tokenizer:", log["reduction_percentage_optimized_vs"])
    print(f"reduction %  (morphology-aware pre-segmentation): {reduction_morph:.2f}")
    print(f"reduction %  (lexicon-seeded vocabulary):          {reduction_seeded:.2f}")
    print(f"reduction %  (lexicon-seeded + combinatorial):     {reduction_combo:.2f}")

    # ---- Step 6: statistical tests (paired Wilcoxon + Holm) --------------
    # Two separate comparisons against the baseline: the (unsuccessful)
    # forced pre-segmentation approach, and the (successful, at matched
    # budget) lexicon-seeded-vocabulary approach.
    a = per_word_counts["baseline_bpe"]
    b = per_word_counts["morphology_aware_bpe"]
    s = per_word_counts["lexicon_seeded_bpe"]
    c = per_word_counts["lexicon_seeded_bpe_combinatorial"]
    stat_rows = []
    p_values = []
    comparisons = []
    if len(a) == len(b) and len(a) > 0 and any(x != y for x, y in zip(a, b)):
        stat, p = wilcoxon(a, b)
        p_values.append(p)
        comparisons.append(("baseline_bpe", "morphology_aware_bpe", stat))
    if len(a) == len(s) and len(a) > 0 and any(x != y for x, y in zip(a, s)):
        stat, p = wilcoxon(a, s)
        p_values.append(p)
        comparisons.append(("baseline_bpe", "lexicon_seeded_bpe", stat))
    if len(a) == len(c) and len(a) > 0 and any(x != y for x, y in zip(a, c)):
        stat, p = wilcoxon(a, c)
        p_values.append(p)
        comparisons.append(("baseline_bpe", "lexicon_seeded_bpe_combinatorial", stat))
    om = per_word_counts["optimized_morph_tokenizer_matched_vocab"]
    if any(x != y for x, y in zip(s, om)):
        stat, p = wilcoxon(s, om)
        p_values.append(p)
        comparisons.append(("lexicon_seeded_bpe", "optimized_morph_tokenizer_matched_vocab", stat))
    o = per_word_counts["optimized_morph_tokenizer"]
    if any(x != y for x, y in zip(a, o)):
        stat, p = wilcoxon(a, o)
        p_values.append(p)
        comparisons.append(("baseline_bpe", "optimized_morph_tokenizer", stat))
    for name in pretrained_tokenizers:
        pc = per_word_counts[name]
        if len(pc) == len(o) and any(x != y for x, y in zip(pc, o)):
            stat, p = wilcoxon(pc, o)
            p_values.append(p)
            comparisons.append((name, "optimized_morph_tokenizer", stat))

    if p_values:
        holm_results = holm_correct(p_values, alpha=0.05)
        for (name_a, name_b, stat), hr in zip(comparisons, holm_results):
            stat_rows.append({
                "comparison": f"{name_a}_vs_{name_b}",
                "wilcoxon_statistic": stat,
                "p_value": hr["p_value"],
                "holm_threshold": hr["threshold"],
                "significant_after_holm": hr["rejected"],
            })
    save_table(stat_rows, "statistical_tests.csv")

    # ---- Step 7: bootstrap CIs for the mean token-count differences ------
    ci_morph = bootstrap_ci_mean_difference(a, b, n_boot=2000, seed=SEED)
    ci_seeded = bootstrap_ci_mean_difference(a, s, n_boot=2000, seed=SEED)
    ci_combo = bootstrap_ci_mean_difference(a, c, n_boot=2000, seed=SEED)
    ci_morph["comparison"] = "baseline_minus_morphology_aware"
    ci_seeded["comparison"] = "baseline_minus_lexicon_seeded"
    ci_combo["comparison"] = "baseline_minus_lexicon_seeded_combinatorial"
    ci_opt = bootstrap_ci_mean_difference(a, o, n_boot=2000, seed=SEED)
    ci_opt["comparison"] = "baseline_minus_optimized_morph_tokenizer"
    rows_ci = [ci_morph, ci_seeded, ci_combo, ci_opt]
    for name in pretrained_tokenizers:
        ci_p = bootstrap_ci_mean_difference(per_word_counts[name], o, n_boot=2000, seed=SEED)
        ci_p["comparison"] = f"{name}_minus_optimized_morph_tokenizer"
        rows_ci.append(ci_p)
    save_table(rows_ci, "bootstrap_ci_baseline_vs_morph.csv")

    # ---- Step 8: word-length / frequency-group breakdown -----------------
    def length_group(w):
        n = len(w)
        if n <= 3:
            return "short(<=3)"
        if n <= 6:
            return "medium(4-6)"
        return "long(7+)"

    breakdown_rows = []
    for group_name, group_fn in [("length", length_group), ("eval_group", lambda w: classify_word_group(lexicon, w))]:
        groups = {}
        for w, fa, fb in zip(test_words, a, b):
            g = group_fn(w)
            groups.setdefault(g, {"baseline": [], "morph": []})
            groups[g]["baseline"].append(fa)
            groups[g]["morph"].append(fb)
        for g, vals in groups.items():
            breakdown_rows.append({
                "grouping": group_name,
                "group": g,
                "n_words": len(vals["baseline"]),
                "baseline_mean_tokens": round(float(np.mean(vals["baseline"])), 4),
                "morph_mean_tokens": round(float(np.mean(vals["morph"])), 4),
            })
    save_table(breakdown_rows, "fertility_by_group.csv")

    # ---- Step 9: 6-variant ablation study ---------------------------------
    def ablation_dict_only(word):
        return [word] if lexicon.words.exists(word) else None

    def ablation_greedy_longest(word):
        pieces = []
        remaining = word
        while remaining:
            m = lexicon.words.longest_prefix_match(remaining) or \
                lexicon.roots.longest_prefix_match(remaining) or \
                lexicon.suffixes.longest_prefix_match(remaining) or \
                lexicon.prefixes.longest_prefix_match(remaining)
            if not m:
                pieces.append(remaining)
                break
            pieces.append(m)
            remaining = remaining[len(m):]
        return pieces

    def ablation_trie_candidates_only(word):
        from src.morphology import generate_candidates
        cands = generate_candidates(lexicon, word)
        return cands[0].as_strings() if cands else [word]

    def ablation_trie_plus_rules(word):
        from src.morphology import generate_candidates
        cands = generate_candidates(lexicon, word)
        valid = [c for c in cands if c.pieces and c.pieces[0].kind != "suffix"]
        chosen = valid[0] if valid else (cands[0] if cands else None)
        return chosen.as_strings() if chosen else [word]

    def ablation_trie_plus_frequency(word):
        from src.morphology import generate_candidates, score_candidate
        weights = {"whole_word_known": 0.0, "root_known": 0.0, "affix_known": 0.0,
                   "frequency_log": 1.0, "segment_count_penalty": 0.0, "unknown_penalty": 1.0}
        cands = generate_candidates(lexicon, word)
        scored = sorted(cands, key=lambda c: score_candidate(c, lexicon, weights)[0], reverse=True)
        return scored[0].as_strings() if scored else [word]

    def ablation_full_hybrid(word):
        seg, _, _, _ = select_best_segmentation(lexicon, word)
        return seg.as_strings()

    ablation_variants = {
        "1_dictionary_only": ablation_dict_only,
        "2_greedy_longest_match": ablation_greedy_longest,
        "3_trie_candidate_generation": ablation_trie_candidates_only,
        "4_trie_plus_morph_rules": ablation_trie_plus_rules,
        "5_trie_plus_frequency_scoring": ablation_trie_plus_frequency,
        "6_full_hybrid": ablation_full_hybrid,
    }

    ablation_rows = []
    sample_words = test_words[:300] if len(test_words) > 300 else test_words
    for variant_name, fn in ablation_variants.items():
        token_counts = []
        coverage = 0
        for w in sample_words:
            result = fn(w)
            if result is None:
                token_counts.append(1)  # falls through unsegmented
            else:
                token_counts.append(len(result))
                coverage += 1
        ablation_rows.append({
            "variant": variant_name,
            "n_words": len(sample_words),
            "mean_pieces_per_word": round(float(np.mean(token_counts)), 4),
            "coverage_rate": round(coverage / len(sample_words), 4) if sample_words else 0.0,
        })
    save_table(ablation_rows, "ablation_study.csv")

    # ---- Step 10: Trie vs dict vs linear-scan benchmark -------------------
    vocabulary = lexicon.words.all_keys() + lexicon.roots.all_keys()
    queries = test_words[:500] if len(test_words) > 500 else test_words
    bench = benchmark_lookup_methods(vocabulary, queries, repeats=3)
    bench_rows = []
    for mode, methods in bench.items():
        for method, seconds in methods.items():
            bench_rows.append({"query_type": mode, "method": method, "seconds": seconds})
    save_table(bench_rows, "trie_benchmark.csv")

    # ---- Step 11: figures --------------------------------------------------
    plt.figure(figsize=(6, 4))
    names = [r["tokenizer"] for r in fertility_rows]
    ferts = [r["token_fertility"] for r in fertility_rows]
    plt.bar(names, ferts, color="#4C72B0")
    plt.ylabel("Token fertility (tokens/word)")
    plt.title("Token fertility by tokenizer (test set)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_fertility_by_tokenizer.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    totals = [r["total_tokens"] for r in fertility_rows]
    plt.bar(names, totals, color="#DD8452")
    plt.ylabel("Total tokens on test corpus")
    plt.title("Total tokens produced on the held-out test set")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_total_tokens_test_corpus.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(a, bins=range(1, max(max(a), max(b)) + 2), alpha=0.6, label="baseline_bpe")
    plt.hist(b, bins=range(1, max(max(a), max(b)) + 2), alpha=0.6, label="morphology_aware_bpe")
    plt.xlabel("Tokens per word")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Distribution of tokens-per-word")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_tokens_per_word_distribution.png", dpi=130)
    plt.close()

    length_rows = [r for r in breakdown_rows if r["grouping"] == "length"]
    plt.figure(figsize=(6, 4))
    x = np.arange(len(length_rows))
    width = 0.35
    plt.bar(x - width/2, [r["baseline_mean_tokens"] for r in length_rows], width, label="baseline")
    plt.bar(x + width/2, [r["morph_mean_tokens"] for r in length_rows], width, label="morphology-aware")
    plt.xticks(x, [r["group"] for r in length_rows])
    plt.ylabel("Mean tokens per word")
    plt.title("Fertility by word-length group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_fertility_by_word_length_group.png", dpi=130)
    plt.close()

    eval_group_rows = [r for r in breakdown_rows if r["grouping"] == "eval_group"]
    plt.figure(figsize=(6, 4))
    plt.bar([r["group"] for r in eval_group_rows], [r["n_words"] for r in eval_group_rows], color="#55A868")
    plt.ylabel("Number of words")
    plt.title("Unknown-word evaluation groups (test set)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_evaluation_group_coverage.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    exact_methods = list(bench["exact"].keys())
    exact_times = [bench["exact"][m] for m in exact_methods]
    prefix_methods = list(bench["prefix"].keys())
    prefix_times = [bench["prefix"][m] for m in prefix_methods]
    x = np.arange(len(exact_methods))
    width = 0.35
    plt.bar(x - width/2, exact_times, width, label="exact match")
    plt.bar(x + width/2, prefix_times, width, label="prefix match")
    plt.xticks(x, exact_methods)
    plt.ylabel("Seconds (best of 3 runs)")
    plt.title("Lookup-method runtime: Trie vs dict vs linear scan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_lookup_runtime_comparison.png", dpi=130)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar([r["variant"] for r in ablation_rows], [r["mean_pieces_per_word"] for r in ablation_rows], color="#8172B2")
    plt.ylabel("Mean pieces per word")
    plt.title("Ablation study: mean segmentation pieces per word")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_ablation_study.png", dpi=130)
    plt.close()

    # Note: the installed matplotlib fonts in this environment have no
    # Devanagari glyph coverage, so Devanagari text rendered directly into
    # a PNG figure would show as missing-glyph boxes. Figure 8 therefore
    # uses numeric word IDs ("W1".."W8") as the chart's labels and token
    # *counts* as the plotted bars, while the actual Devanagari words and
    # their real tokens are written out in full, readable text to
    # outputs/tables/sample_segmentation_examples.csv (and printed directly
    # in the notebook, where Devanagari does render correctly in text/HTML
    # cells, unlike this raster figure).
    sample_compare_words = [w for w in test_words if len(w) > 4][:8] or test_words[:8]
    baseline_tokenized = [_baseline_fn(w) for w in sample_compare_words]
    morph_tokenized = [_morph_fn(w) for w in sample_compare_words]
    seeded_tokenized = [_seeded_fn(w) for w in sample_compare_words]
    combo_tokenized = [_combo_fn(w) for w in sample_compare_words]
    sample_rows = []
    for i, (w, bt, mt, st, ct) in enumerate(
        zip(sample_compare_words, baseline_tokenized, morph_tokenized, seeded_tokenized, combo_tokenized),
        start=1,
    ):
        sample_rows.append({
            "word_id": f"W{i}", "word": w,
            "baseline_tokens": " | ".join(bt), "baseline_token_count": len(bt),
            "morph_tokens": " | ".join(mt), "morph_token_count": len(mt),
            "seeded_tokens": " | ".join(st), "seeded_token_count": len(st),
            "combinatorial_tokens": " | ".join(ct), "combinatorial_token_count": len(ct),
        })
    save_table(sample_rows, "sample_segmentation_examples.csv")

    plt.figure(figsize=(10, 4.5))
    x = np.arange(len(sample_rows))
    width = 0.2
    plt.bar(x - 1.5 * width, [r["baseline_token_count"] for r in sample_rows], width, label="baseline BPE")
    plt.bar(x - 0.5 * width, [r["morph_token_count"] for r in sample_rows], width, label="morphology-aware BPE (forced split)")
    plt.bar(x + 0.5 * width, [r["seeded_token_count"] for r in sample_rows], width, label="lexicon-seeded BPE (optimized)")
    plt.bar(x + 1.5 * width, [r["combinatorial_token_count"] for r in sample_rows], width, label="lexicon-seeded + combinatorial")
    plt.xticks(x, [r["word_id"] for r in sample_rows])
    plt.ylabel("Token count")
    plt.title("Sample word segmentation: four tokenizers compared\n"
              "(see outputs/tables/sample_segmentation_examples.csv for the actual words/tokens)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_sample_segmentation_comparison.png", dpi=130)
    plt.close()

    print("saved figures to", FIGURES_DIR)

    # ---- Step 12: runtime benchmark (per document / per 1000 chars) ------
    from src.evaluation import benchmark_tokenizer_runtime
    runtime_rows = []
    for name, fn in [("baseline_bpe", lambda t: tokenize_word_baseline(baseline_tok, t)),
                      ("morphology_aware_bpe", lambda t: tokenize_word_morphology_aware(morph_tok, lexicon, t)),
                      ("lexicon_seeded_bpe", lambda t: tokenize_word_baseline(seeded_tok, t)),
                      ("lexicon_seeded_bpe_combinatorial", lambda t: tokenize_word_baseline(combo_tok, t)),
                      ("optimized_morph_tokenizer", lambda t: opt_tok.tokenize(t))]:
        r = benchmark_tokenizer_runtime(fn, test_lines[:200] if len(test_lines) > 200 else test_lines)
        r["tokenizer"] = name
        runtime_rows.append(r)
    save_table(runtime_rows, "runtime_benchmark.csv")

    # ---- Step 13: vocabulary-size sweep ------------------------------------
    # Trains both tokenizer families at several vocabulary sizes and reports
    # fertility at each, so the baseline-vs-morphology-aware comparison can
    # be read as a curve (does the gap change with vocabulary budget) rather
    # than a single operating point.
    sweep_sizes = [300, 600, 1200, 2400] if VOCAB_SIZE <= 1200 else [2000, 4000, 8000, 16000, 32000]
    sweep_rows = []
    for size in sweep_sizes:
        cfg = TokenizerConfig(vocab_size=size, min_frequency=1)
        b_tok = train_bpe(train_lines, cfg, morphology_presegment=False)
        m_tok = train_bpe(train_lines, cfg, morphology_presegment=True, lexicon=lexicon)
        b_report = compute_fertility(test_words, lambda w: tokenize_word_baseline(b_tok, w))
        m_report = compute_fertility(test_words, lambda w: tokenize_word_morphology_aware(m_tok, lexicon, w))
        o_tok, _ = train_optimized_tokenizer(train_lines, cfg, lexicon)
        o_report = compute_fertility(test_words, o_tok.tokenize)
        sweep_rows.append({
            "optimized_fertility": round(o_report.token_fertility, 4),
            "optimized_reduction_vs_baseline_pct": round(
                reduction_percentage(b_report.total_tokens, o_report.total_tokens), 4),
            "requested_vocab_size": size,
            "baseline_actual_vocab": b_tok.get_vocab_size(),
            "morph_actual_vocab": m_tok.get_vocab_size(),
            "baseline_fertility": round(b_report.token_fertility, 4),
            "morph_fertility": round(m_report.token_fertility, 4),
            "reduction_percentage": round(
                reduction_percentage(b_report.total_tokens, m_report.total_tokens), 4
            ),
        })
    save_table(sweep_rows, "vocabulary_size_sweep.csv")

    plt.figure(figsize=(6, 4))
    xs = [r["requested_vocab_size"] for r in sweep_rows]
    plt.plot(xs, [r["baseline_fertility"] for r in sweep_rows], marker="o", label="baseline BPE")
    plt.plot(xs, [r["morph_fertility"] for r in sweep_rows], marker="o", label="morphology-aware BPE")
    plt.plot(xs, [r["optimized_fertility"] for r in sweep_rows], marker="o", label="optimized (min-token)")
    plt.xlabel("Requested vocabulary size")
    plt.ylabel("Token fertility (tokens/word)")
    plt.title("Fertility vs. vocabulary size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig9_vocabulary_size_sweep.png", dpi=130)
    plt.close()
    log["vocabulary_sweep"] = sweep_rows
    print("vocabulary sweep:", sweep_rows)

    # ---- Save run log -------------------------------------------------------
    with open(REPORTS_DIR / "run_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
