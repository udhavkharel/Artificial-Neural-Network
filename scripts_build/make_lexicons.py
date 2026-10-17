"""
Builds the seed lexicon CSVs under data/lexicons/.

These are SEED lexicons: a small, manually assembled, manually checked set
of Nepali morphological units used to bootstrap and test the indexed
segmentation engine. They are explicitly NOT a comprehensive dictionary of
Nepali morphology — Nepali has thousands of roots and a much larger set of
inflectional/derivational patterns than are enumerated here. Every row's
`source` field says "manual_seed" and every row's `confidence` field is a
value in (0.0, 1.0] reflecting how certain the entry is, not a measured
statistic. Frequencies are placeholder integers reflecting relative
commonness as judged by the person assembling the seed data (not corpus
counts) until the corpus-count-based frequency update step
(src/morphology.py: `update_frequencies_from_corpus`) runs on real data and
overwrites them.

Columns used by every lexicon file (src/morphology.py reads this schema):
    surface_form        the literal string this row represents (a root, a
                         prefix, a suffix, or a complete word)
    root                 for a complete word: its root. For a root entry:
                         itself. Empty for prefix/suffix rows.
    prefix_sequence      '|'-separated list of prefixes attached (words.csv
                         only); empty otherwise
    suffix_sequence      '|'-separated list of suffixes attached (words.csv
                         only); empty otherwise
    grammatical_category noun / verb / adjective / postposition-suffix /
                         derivational-prefix / pronoun / particle / other
    frequency            placeholder relative-frequency integer (seed) —
                         overwritten by real corpus counts when available
    source               always "manual_seed" in these files
    confidence           0.0-1.0, how confident the manual entry is
"""
import csv
from pathlib import Path

LEX_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"
LEX_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "surface_form", "root", "prefix_sequence", "suffix_sequence",
    "grammatical_category", "frequency", "source", "confidence",
]

# ---------------------------------------------------------------------------
# Roots: a small set of common Nepali verb and noun roots.
# ---------------------------------------------------------------------------
ROOTS = [
    # verb roots
    ("गर्", "verb", 950, 0.95),     # do/make
    ("जा्", "verb", 900, 0.80),     # go (bound form; "जानु" is the infinitive)
    ("आउ", "verb", 850, 0.85),      # come
    ("खा्", "verb", 800, 0.80),     # eat
    ("पढ्", "verb", 750, 0.95),     # read/study
    ("लेख्", "verb", 740, 0.95),    # write
    ("बोल्", "verb", 700, 0.90),    # speak
    ("हेर्", "verb", 720, 0.90),    # see/watch
    ("सुन्", "verb", 710, 0.90),    # hear
    ("खेल्", "verb", 690, 0.90),    # play
    ("दिन्", "verb", 650, 0.70),    # give (bound)
    ("बस्", "verb", 680, 0.90),     # sit/stay/live
    ("हुन्", "verb", 600, 0.60),    # be/happen (bound)
    ("राख्", "verb", 640, 0.90),    # keep/put
    ("बुझ्", "verb", 630, 0.90),    # understand
    ("सिक्", "verb", 600, 0.90),    # learn
    ("सिकाउ", "verb", 500, 0.85),   # teach
    ("मिल्", "verb", 560, 0.85),    # match/get along
    ("चल्", "verb", 600, 0.90),     # move/operate
    ("उठ्", "verb", 550, 0.90),     # get up
    # noun roots
    ("घर", "noun", 900, 0.95),      # house
    ("विद्यालय", "noun", 700, 0.90), # school
    ("किताब", "noun", 650, 0.90),   # book
    ("पानी", "noun", 800, 0.90),    # water
    ("मान्छे", "noun", 850, 0.90),  # person
    ("साथी", "noun", 700, 0.90),    # friend
    ("शिक्षक", "noun", 600, 0.90),  # teacher
    ("विद्यार्थी", "noun", 650, 0.90), # student
    ("समय", "noun", 700, 0.90),     # time
    ("काम", "noun", 750, 0.90),     # work
    ("देश", "noun", 650, 0.90),     # country
    ("भाषा", "noun", 600, 0.90),    # language
    ("शब्द", "noun", 580, 0.90),    # word
    ("गाउँ", "noun", 600, 0.90),    # village
    ("सहर", "noun", 580, 0.90),     # city
    ("नदी", "noun", 500, 0.90),     # river
    ("पहाड", "noun", 480, 0.90),    # mountain/hill
    ("खेत", "noun", 500, 0.90),     # field
    ("बजार", "noun", 520, 0.90),    # market
    ("किसान", "noun", 450, 0.90),   # farmer
]

# ---------------------------------------------------------------------------
# Prefixes: Sanskrit-origin derivational prefixes productive in Nepali.
# ---------------------------------------------------------------------------
PREFIXES = [
    ("अ", 700, 0.85),       # negation ("a-", e.g. अज्ञान "ignorance")
    ("अन्", 500, 0.80),     # negation before vowels (अनन्त)
    ("सु", 600, 0.85),      # good/well (सुगम "easy")
    ("दुर्", 450, 0.80),    # bad/difficult (दुर्गम "inaccessible")
    ("प्र", 600, 0.75),     # forward/intensive (प्रयोग "use")
    ("वि", 650, 0.75),      # apart/special (विशेष "special")
    ("निर्", 500, 0.75),    # without/out (निर्माण "construction")
    ("उप", 480, 0.80),      # near/sub (उपसचिव "under-secretary")
    ("अति", 520, 0.85),     # excessive (अतिरिक्त "extra")
    ("अधि", 420, 0.75),     # over/super (अधिकार "right/authority")
    ("अनु", 460, 0.75),     # following/post (अनुभव "experience")
    ("परा", 300, 0.70),     # reverse (पराजय "defeat")
    ("सम्", 400, 0.70),     # together (सम्मिलित "combined")
    ("स्व", 400, 0.80),     # self (स्वतन्त्र "independent")
    ("पुन", 450, 0.80),     # again (पुनर्निर्माण "reconstruction")
]

# ---------------------------------------------------------------------------
# Suffixes: postpositions and inflectional suffixes (case markers, plural,
# tense markers) productive on Nepali nouns/pronouns and verb stems.
# ---------------------------------------------------------------------------
SUFFIXES = [
    ("हरू", "plural", 950, 0.95),
    ("लाई", "dative/accusative", 900, 0.95),
    ("को", "genitive", 900, 0.95),
    ("का", "genitive-plural/adjectival", 850, 0.90),
    ("की", "genitive-feminine", 700, 0.90),
    ("ले", "instrumental/ergative", 880, 0.95),
    ("मा", "locative", 900, 0.95),
    ("बाट", "ablative", 800, 0.90),
    ("सम्म", "terminative", 650, 0.90),
    ("सँग", "associative", 700, 0.90),
    ("संग", "associative-variant", 600, 0.85),
    ("देखि", "ablative-temporal", 600, 0.90),
    ("हरूलाई", "plural+dative", 500, 0.85),
    ("हरूको", "plural+genitive", 500, 0.85),
    ("हरूमा", "plural+locative", 450, 0.85),
    ("हरूबाट", "plural+ablative", 350, 0.80),
    ("न्छ", "verb-present-3sg", 500, 0.75),
    ("न्छु", "verb-present-1sg", 400, 0.75),
    ("एको", "verb-perfective-participle", 550, 0.75),
    ("दै", "verb-progressive", 450, 0.75),
]

# ---------------------------------------------------------------------------
# Words: complete surface forms assembled from (root + prefix/suffix), for
# training/evaluating the "whole word known" lookup and for the
# root/inflected morphology pairs used in evaluation. Each must be
# reconstructable as root + prefix_sequence(optional) + suffix_sequence.
# ---------------------------------------------------------------------------
WORD_ROWS = [
    # (surface_form, root, prefix_seq, suffix_seq, category)
    ("घरहरू", "घर", "", "हरू", "noun"),
    ("घरलाई", "घर", "", "लाई", "noun"),
    ("घरको", "घर", "", "को", "noun"),
    ("घरमा", "घर", "", "मा", "noun"),
    ("घरबाट", "घर", "", "बाट", "noun"),
    ("घरहरूलाई", "घर", "", "हरूलाई", "noun"),
    ("घरहरूको", "घर", "", "हरूको", "noun"),
    ("साथीहरू", "साथी", "", "हरू", "noun"),
    ("साथीलाई", "साथी", "", "लाई", "noun"),
    ("साथीसँग", "साथी", "", "सँग", "noun"),
    ("मान्छेहरू", "मान्छे", "", "हरू", "noun"),
    ("मान्छेलाई", "मान्छे", "", "लाई", "noun"),
    ("किताबहरू", "किताब", "", "हरू", "noun"),
    ("किताबको", "किताब", "", "को", "noun"),
    ("विद्यालयमा", "विद्यालय", "", "मा", "noun"),
    ("विद्यालयबाट", "विद्यालय", "", "बाट", "noun"),
    ("विद्यार्थीहरू", "विद्यार्थी", "", "हरू", "noun"),
    ("विद्यार्थीहरूलाई", "विद्यार्थी", "", "हरूलाई", "noun"),
    ("शिक्षकले", "शिक्षक", "", "ले", "noun"),
    ("शिक्षकहरू", "शिक्षक", "", "हरू", "noun"),
    ("देशमा", "देश", "", "मा", "noun"),
    ("देशहरूको", "देश", "", "हरूको", "noun"),
    ("गाउँमा", "गाउँ", "", "मा", "noun"),
    ("सहरबाट", "सहर", "", "बाट", "noun"),
    ("नदीसम्म", "नदी", "", "सम्म", "noun"),
    ("पहाडमा", "पहाड", "", "मा", "noun"),
    ("खेतमा", "खेत", "", "मा", "noun"),
    ("बजारसम्म", "बजार", "", "सम्म", "noun"),
    ("किसानहरूले", "किसान", "", "हरू|ले", "noun"),
    ("समयमा", "समय", "", "मा", "noun"),
    ("कामको", "काम", "", "को", "noun"),
    ("भाषामा", "भाषा", "", "मा", "noun"),
    ("शब्दहरू", "शब्द", "", "हरू", "noun"),
    # derivational (prefix) examples
    ("असफल", "सफल", "अ", "", "adjective"),
    ("अज्ञान", "ज्ञान", "अ", "", "noun"),
    ("सुगम", "गम", "सु", "", "adjective"),
    ("दुर्गम", "गम", "दुर्", "", "adjective"),
    ("निर्माण", "माण", "निर्", "", "noun"),
    ("स्वतन्त्र", "तन्त्र", "स्व", "", "adjective"),
    ("पुनर्निर्माण", "माण", "पुनर्निर्", "", "noun"),
    # verb inflections (root + suffix on bound verb stems)
    ("गर्छ", "गर्", "", "छ", "verb"),
    ("गर्छु", "गर्", "", "छु", "verb"),
    ("गरेको", "गर्", "", "एको", "verb"),
    ("गर्दै", "गर्", "", "दै", "verb"),
    ("पढ्छ", "पढ्", "", "छ", "verb"),
    ("पढेको", "पढ्", "", "एको", "verb"),
    ("लेख्छ", "लेख्", "", "छ", "verb"),
    ("लेखेको", "लेख्", "", "एको", "verb"),
    ("बोल्छ", "बोल्", "", "छ", "verb"),
    ("हेर्छ", "हेर्", "", "छ", "verb"),
    ("सुन्छ", "सुन्", "", "छ", "verb"),
    ("खेल्छ", "खेल्", "", "छ", "verb"),
    ("बस्छ", "बस्", "", "छ", "verb"),
    ("राख्छ", "राख्", "", "छ", "verb"),
    ("बुझ्छ", "बुझ्", "", "छ", "verb"),
    ("सिक्छ", "सिक्", "", "छ", "verb"),
    ("मिल्छ", "मिल्", "", "छ", "verb"),
    ("चल्छ", "चल्", "", "छ", "verb"),
    ("उठ्छ", "उठ्", "", "छ", "verb"),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    root_rows = [
        {
            "surface_form": s, "root": s, "prefix_sequence": "", "suffix_sequence": "",
            "grammatical_category": cat, "frequency": freq, "source": "manual_seed",
            "confidence": conf,
        }
        for s, cat, freq, conf in ROOTS
    ]
    write_csv(LEX_DIR / "roots.csv", root_rows)

    prefix_rows = [
        {
            "surface_form": s, "root": "", "prefix_sequence": s, "suffix_sequence": "",
            "grammatical_category": "derivational-prefix", "frequency": freq,
            "source": "manual_seed", "confidence": conf,
        }
        for s, freq, conf in PREFIXES
    ]
    write_csv(LEX_DIR / "prefixes.csv", prefix_rows)

    suffix_rows = [
        {
            "surface_form": s, "root": "", "prefix_sequence": "", "suffix_sequence": s,
            "grammatical_category": cat, "frequency": freq, "source": "manual_seed",
            "confidence": conf,
        }
        for s, cat, freq, conf in SUFFIXES
    ]
    write_csv(LEX_DIR / "suffixes.csv", suffix_rows)

    word_rows = [
        {
            "surface_form": surf, "root": root, "prefix_sequence": pfx,
            "suffix_sequence": sfx, "grammatical_category": cat,
            "frequency": 100, "source": "manual_seed", "confidence": 0.9,
        }
        for surf, root, pfx, sfx, cat in WORD_ROWS
    ]
    write_csv(LEX_DIR / "words.csv", word_rows)

    print(f"roots: {len(root_rows)}  prefixes: {len(prefix_rows)}  "
          f"suffixes: {len(suffix_rows)}  words: {len(word_rows)}")


if __name__ == "__main__":
    main()
