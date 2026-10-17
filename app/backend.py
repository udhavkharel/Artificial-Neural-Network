"""
A small local web app that shows, live, how three tokenizers built in
this project handle the same Nepali text, side by side:

  - baseline_bpe:          plain BPE, no morphological knowledge at all
  - morphology_aware_bpe:  BPE trained on text forcibly pre-split at
                            every recognized root/suffix boundary (the
                            project's first attempt — measured to
                            *increase* fertility, kept here for
                            transparency, not hidden)
  - lexicon_seeded_bpe:     plain BPE with the lexicon's known whole
                            words, plus frequent training words (counted
                            on their punctuation-stripped form too, so a
                            word seeds regardless of what punctuation
                            follows it), added directly as guaranteed
                            single-token vocabulary entries (the
                            project's second attempt — this one
                            measurably reduces fertility at a matched
                            vocabulary budget; see outputs/tables/ and
                            research_report.md)
  - lexicon_seeded_bpe_combinatorial: the same seeding mechanism, with
                            the seed set additionally expanded to every
                            root+affix combination the lexicon's grammar
                            licenses (not just words literally observed).
                            Loaded and reported in /api/overall_results
                            for transparency, but NOT shown as a fourth
                            live-demo column, because on this project's
                            small synthetic test set it ties the plain
                            seeded tokenizer on every word (a measured,
                            honestly-reported limitation — see that
                            endpoint's "combinatorial" section).

This app does not recompute the project's research results — it loads
the already-trained tokenizers (models/*.json) and the already-computed
tables (outputs/tables/*.csv), and adds two live, interactive features on
top: (1) typing any Nepali text and seeing all three tokenizers' output
for it immediately, word by word, and (2) a ChatGPT-style demo endpoint
(/api/chat_compare) that tokenizes a Nepali prompt and an answer (either
pasted in, or fetched live from the Anthropic API using a key the person
supplies themselves — no key is stored anywhere by this app) and reports
how many tokens the baseline vs. the optimized tokenizer would need for
that real input/output pair.

Run with:
    cd app
    uvicorn backend:app --reload --port 8000
Then open http://localhost:8000 in a browser.
"""

import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.morphology import classify_word_group, load_lexicon, select_best_segmentation
from src.normalization import normalize_text, whitespace_words
from src.optimized_tokenizer import OptimizedMorphTokenizer
from src.tokenizer import (
    load_tokenizer,
    tokenize_word_baseline,
    tokenize_word_morphology_aware,
)

LEXICON_DIR = PROJECT_ROOT / "data" / "lexicons"
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

app = FastAPI(title="Nepali Token Fertility Comparison")

lexicon = load_lexicon(LEXICON_DIR)
baseline_tok = load_tokenizer(MODELS_DIR / "baseline_bpe.json")
morph_tok = load_tokenizer(MODELS_DIR / "morphology_aware_bpe.json")
seeded_tok = load_tokenizer(MODELS_DIR / "lexicon_seeded_bpe.json")
_combo_path = MODELS_DIR / "lexicon_seeded_bpe_combinatorial.json"
combo_tok = load_tokenizer(_combo_path) if _combo_path.exists() else None

optimized_tok = OptimizedMorphTokenizer.load(MODELS_DIR / "optimized_morph_tokenizer.json")

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_DEFAULT_MODEL = "claude-3-5-haiku-20241022"


def read_csv_rows(name: str) -> list[dict]:
    path = TABLES_DIR / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TokenizeRequest(BaseModel):
    text: str


class ChatCompareRequest(BaseModel):
    prompt: str
    api_key: str | None = None       # the person's own Anthropic API key; never stored
    model: str | None = None         # defaults to ANTHROPIC_DEFAULT_MODEL
    manual_response: str | None = None  # paste-in fallback if no key is supplied


@app.get("/")
def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.get("/api/overall_results")
def overall_results():
    """The project's actual measured summary numbers, read straight from
    outputs/tables/ and outputs/reports/ — not recomputed here."""
    import json
    run_log_path = REPORTS_DIR / "run_log.json"
    run_log = json.loads(run_log_path.read_text(encoding="utf-8")) if run_log_path.exists() else {}
    return {
        "fertility_comparison": read_csv_rows("fertility_comparison.csv"),
        "vocabulary_sweep": read_csv_rows("vocabulary_size_sweep.csv"),
        "ablation_study": read_csv_rows("ablation_study.csv"),
        "bootstrap_ci": read_csv_rows("bootstrap_ci_baseline_vs_morph.csv"),
        "statistical_tests": read_csv_rows("statistical_tests.csv"),
        "train_test_word_overlap": run_log.get("train_test_word_overlap", {}),
        "lexicon_seeded_vocab_info": run_log.get("tokenizer_training", {}).get("lexicon_seeded_vocab_info", {}),
        "lexicon_seeded_combinatorial_vocab_info": run_log.get("tokenizer_training", {}).get(
            "lexicon_seeded_combinatorial_vocab_info", {}
        ),
        "reduction_percentage_optimized_vs": run_log.get("reduction_percentage_optimized_vs", {}),
        "corpus_meta": run_log.get("corpus_meta", {}),
        "reduction_percentage_seeded_vs_baseline": run_log.get("reduction_percentage_seeded_vs_baseline"),
        "reduction_percentage_seeded_combinatorial_vs_baseline": run_log.get(
            "reduction_percentage_seeded_combinatorial_vs_baseline"
        ),
        "lexicon_sizes": {
            "roots": lexicon.roots.count(),
            "prefixes": lexicon.prefixes.count(),
            "suffixes": lexicon.suffixes.count(),
            "words": lexicon.words.count(),
            "total_entries": (
                lexicon.roots.count() + lexicon.prefixes.count()
                + lexicon.suffixes.count() + lexicon.words.count()
            ),
        },
    }


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest):
    """Live, word-by-word comparison of all three tokenizers for whatever
    text is typed in."""
    text = normalize_text(req.text)
    words = [w for w in whitespace_words(text) if w]

    rows = []
    total_baseline = 0
    total_morph = 0
    total_seeded = 0
    total_optimized = 0
    for w in words:
        b_tokens = tokenize_word_baseline(baseline_tok, w)
        m_tokens = tokenize_word_morphology_aware(morph_tok, lexicon, w)
        s_tokens = tokenize_word_baseline(seeded_tok, w)
        o_tokens = optimized_tok.tokenize(w)
        total_optimized += len(o_tokens)
        seg, score, breakdown, _ = select_best_segmentation(lexicon, w)
        group = classify_word_group(lexicon, w)
        total_baseline += len(b_tokens)
        total_morph += len(m_tokens)
        total_seeded += len(s_tokens)
        rows.append({
            "word": w,
            "baseline_tokens": b_tokens,
            "baseline_count": len(b_tokens),
            "morph_tokens": m_tokens,
            "morph_count": len(m_tokens),
            "seeded_tokens": s_tokens,
            "seeded_count": len(s_tokens),
            "optimized_tokens": o_tokens,
            "optimized_count": len(o_tokens),
            "segmentation": seg.as_strings(),
            "segmentation_score": round(score, 2),
            "evaluation_group": group,
        })

    n_words = len(words) or 1
    return {
        "words": rows,
        "summary": {
            "n_words": len(words),
            "total_baseline_tokens": total_baseline,
            "total_morph_tokens": total_morph,
            "total_seeded_tokens": total_seeded,
            "total_optimized_tokens": total_optimized,
            "optimized_fertility": round(total_optimized / n_words, 4),
            "optimized_delta": total_optimized - total_baseline,
            "optimized_reduction_pct": round(
                100 * (total_baseline - total_optimized) / total_baseline, 2) if total_baseline else 0.0,
            "baseline_fertility": round(total_baseline / n_words, 4),
            "morph_fertility": round(total_morph / n_words, 4),
            "seeded_fertility": round(total_seeded / n_words, 4),
            "morph_delta": total_morph - total_baseline,
            "seeded_delta": total_seeded - total_baseline,
        },
    }


def _count_optimized(text: str) -> int:
    return len(optimized_tok.tokenize(normalize_text(text)))


def _count_tokens(tok, text: str) -> int:
    """Total subword tokens `tok` needs for `text`, summed word by word
    (same denominator convention used throughout this project)."""
    text = normalize_text(text)
    words = [w for w in whitespace_words(text) if w]
    return sum(len(tokenize_word_baseline(tok, w)) for w in words)


def _call_anthropic(prompt: str, api_key: str, model: str) -> tuple[str | None, str | None]:
    """
    Calls the real Anthropic Messages API with the key the person
    supplied themselves (never read from this app's own environment or
    files, and never written to disk or logged). Returns
    (response_text, error_message) -- exactly one of the two is not None.

    This project's sandbox confirmed api.anthropic.com is reachable and
    that a request with an invalid key correctly comes back as a clean
    HTTP 401 "invalid x-api-key" (not a network-level block), so the
    request path itself is verified; a real generation was not run from
    inside the project's development sandbox, since doing so would have
    required the developer's own Anthropic API key. With a real key
    supplied by whoever is running the app, this performs an ordinary,
    live call.
    """
    body = json.dumps({
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_MESSAGES_URL,
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            text = "".join(
                block.get("text", "") for block in payload.get("content", [])
                if block.get("type") == "text"
            )
            return text, None
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8"))
            msg = detail.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        return None, f"Anthropic API error ({e.code}): {msg}"
    except Exception as e:
        return None, f"Could not reach the Anthropic API: {e}"


@app.post("/api/chat_compare")
def chat_compare(req: ChatCompareRequest):
    """
    The "ChatGPT-style" demo: tokenize a real Nepali prompt and a real
    answer to it (either generated live via the Anthropic API using a
    key supplied in this request, or pasted in manually), and report how
    many tokens the baseline tokenizer vs. this project's optimized
    (min-token, same-vocabulary-budget) tokenizer would need for that exact input and
    output -- i.e. the token-count savings this project's optimization
    would give a real generative-AI interaction, not just an isolated
    test sentence.

    Nothing about the API key is stored: it exists only for the duration
    of this one request, is never written to a file or a log, and is
    never sent anywhere except directly to api.anthropic.com.
    """
    prompt = req.prompt.strip()
    if not prompt:
        return {"error": "Prompt is empty."}

    response_text = None
    source = None
    error = None

    if req.manual_response and req.manual_response.strip():
        response_text = req.manual_response.strip()
        source = "manual_paste"
    elif req.api_key and req.api_key.strip():
        model = (req.model or ANTHROPIC_DEFAULT_MODEL).strip()
        response_text, error = _call_anthropic(prompt, req.api_key.strip(), model)
        source = "anthropic_api_live" if response_text is not None else None
    else:
        return {
            "error": (
                "No API key and no pasted response were given. Either paste "
                "your own Anthropic API key (used only for this one request, "
                "never stored) to generate a real answer live, or paste in an "
                "answer you already got elsewhere (e.g. from claude.ai or "
                "ChatGPT) and this will still compare the token counts."
            )
        }

    if response_text is None:
        return {"error": error or "No response text was produced."}

    input_baseline = _count_tokens(baseline_tok, prompt)
    input_seeded = _count_optimized(prompt)
    output_baseline = _count_tokens(baseline_tok, response_text)
    output_seeded = _count_optimized(response_text)

    def _savings(baseline: int, seeded: int) -> dict:
        diff = baseline - seeded
        pct = round(100 * diff / baseline, 2) if baseline else 0.0
        return {"baseline_tokens": baseline, "optimized_tokens": seeded,
                "tokens_saved": diff, "percent_saved": pct}

    total_baseline = input_baseline + output_baseline
    total_seeded = input_seeded + output_seeded

    return {
        "source": source,
        "prompt": prompt,
        "response_text": response_text,
        "input": _savings(input_baseline, input_seeded),
        "output": _savings(output_baseline, output_seeded),
        "total": _savings(total_baseline, total_seeded),
    }


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
