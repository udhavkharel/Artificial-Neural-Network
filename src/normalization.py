"""
Text normalization for Nepali (Devanagari-script) text.

The goal of this module is narrow and deliberate: put text into one
consistent Unicode form before anything downstream (lexicon lookup, the
Trie, the tokenizers) looks at it, without changing what the text says.

What this module does:
    - Applies Unicode NFC normalization. NFC (not NFKC) is used on purpose:
      NFKC also collapses "compatibility" characters (for example certain
      presentation forms, full-width digits, or ligatures) into forms that
      can change width/spacing-sensitive characters or, for some scripts,
      alter characters that carry real meaning. Devanagari text in the
      corpora this project uses is essentially all composed of ordinary
      NFC-stable codepoints, so NFC is the safer, minimal choice and is
      used uniformly everywhere in the project.
    - Collapses runs of ASCII/Unicode whitespace to a single space and
      strips leading/trailing whitespace, so whitespace differences never
      masquerade as token differences.
    - Leaves Devanagari dependent vowel signs (matras), the virama/halant,
      nukta, candrabindu/anusvara/visarga, conjunct consonant sequences,
      punctuation (both Devanagari danda '।' and ASCII punctuation), and
      digits (both Devanagari '०-९' and ASCII '0-9') untouched in content —
      normalization changes Unicode *representation*, not *content*.
    - Does NOT transliterate Devanagari to Latin script or vice versa, does
      NOT strip punctuation, and does NOT lowercase (Devanagari has no case,
      and any Latin-script words that appear in mixed text are left as-is
      except for whitespace collapsing).

What this module explicitly does not attempt:
    - Spelling correction or dialectal spelling normalization.
    - Removal of diacritics/matras (that would change the word).
    - Word segmentation (that is the job of src/morphology.py).

The same `normalize_text` function is used everywhere in the project —
lexicon loading, corpus cleaning, tokenizer training, and evaluation — so
that a word normalized at one stage is guaranteed to look identical at
every other stage.
"""

from __future__ import annotations

import re
import unicodedata

# Devanagari Unicode block: U+0900 - U+097F
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_WHITESPACE_RE = re.compile(r"\s+")

# Devanagari danda / double danda, treated as sentence punctuation alongside
# ASCII punctuation when splitting text into words.
DEVANAGARI_DANDA = "।॥"


def normalize_text(text: str) -> str:
    """
    Normalize a string of (possibly mixed Nepali/English) text.

    Steps, in order:
      1. Unicode NFC normalization.
      2. Collapse all whitespace runs to a single ASCII space.
      3. Strip leading/trailing whitespace.

    An empty string in returns an empty string out. This function never
    raises on empty or punctuation-only input.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def is_devanagari_word(word: str) -> bool:
    """
    True if `word` contains at least one Devanagari codepoint and no Latin
    letters. Used to separate "Nepali" words from "English"/Latin-script
    words in mixed-script text; a word with digits or punctuation only
    (no letters at all) returns False, since it is not a Devanagari *word*.
    """
    if not word:
        return False
    has_deva = bool(_DEVANAGARI_RE.search(word))
    has_latin = bool(re.search(r"[A-Za-z]", word))
    return has_deva and not has_latin


def is_mixed_script(word: str) -> bool:
    """True if `word` contains both a Devanagari codepoint and a Latin letter."""
    return bool(_DEVANAGARI_RE.search(word)) and bool(re.search(r"[A-Za-z]", word))


def strip_punctuation_for_tokenization(text: str) -> str:
    """
    Remove punctuation characters (ASCII punctuation and the Devanagari
    danda marks) for contexts that need a punctuation-free string — for
    example, computing per-word fertility, where punctuation tokens would
    distort the "tokens per word" ratio. This is a separate, opt-in step;
    `normalize_text` itself never removes punctuation, since punctuation is
    part of the text's content.
    """
    pattern = re.compile(r"[^\w\sऀ-ॿ]|" + f"[{DEVANAGARI_DANDA}]")
    return _WHITESPACE_RE.sub(" ", pattern.sub(" ", text)).strip()


def whitespace_words(text: str) -> list[str]:
    """
    Split already-normalized text into whitespace-separated words. This is
    the exact denominator used for the "token fertility" metric defined in
    evaluation.py (total subword tokens / total whitespace-separated
    words), so this function is intentionally simple and literal: no
    punctuation stripping, no filtering. Call `normalize_text` first.
    """
    if not text:
        return []
    return text.split(" ") if text else []


def character_count(text: str) -> int:
    """
    Character count used as the denominator for "characters per token".
    Counts every character in the normalized string, including spaces and
    punctuation, matching the definition in evaluation.py. This is reported
    separately from token fertility and the two are never conflated.
    """
    return len(text)
