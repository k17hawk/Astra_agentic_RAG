#!/usr/bin/env python3
"""
Train the bilingual (Nepali + English) NepBPE tokenizer.

Pipeline:
  1. Build the word-frequency dictionary once, streaming the merged corpus.
  2. Phase 3 (constrained)  : Devanagari + punctuation -> dev_budget slots.
  3. Phase 4 (unconstrained): Latin + digits only      -> lat_budget slots.
  4. Save the vocab to TSV.

Prereq: rebuild the extension after the Phase-4 changes:
    maturin develop --release

Run:
    python train_bilingual.py
"""

import time
import sys
from tiny_llm_scratch_with_tokenizer import PyNepBPETokenizer

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_PATH = "dataset_merged/whole_train_corpus.txt"

# Budget split is an explicit CHOICE. Nepali is the priority language and its
# long tail is the dominant fertility term; English saturates with far fewer
# slots (its high-frequency subword core is small).
DEV_BUDGET = 44_000       # ← CHANGED: 40k → 44k to reclaim Devanagari capacity
LAT_BUDGET =  4_000       # ← CHANGED: 8k → 4k (English needs fewer slots)

THETA           = 100     # V_ambiguous frequency gate (Phase 3 only)
MIN_WORD_FREQ   = 2       # ← CHANGED: 3 → 2 (attacks the long tail)
PROGRESS_LINES  = 500_000
PROGRESS_MERGES = 1_000
OUT_VOCAB       = "vocab_nepbpe/vocab_nepbpe_bilingual.tsv"

# FROZEN AT TRAINING TIME. Whatever you train with, you must load with — the
# folding policy is baked into the vocabulary and cannot be changed after.
FOLDING_RULES = [
    ("सङ्ग", "संग"),   # explicit nasal conjunct -> anusvara
    ("सँग", "संग"),    # chandrabindu            -> anusvara
]

# Base vocabulary. Latin a-z A-Z 0-9 are seeded automatically inside
# initialize_vocab() as LAT-scripted tokens — do NOT add them here.
DEVANAGARI  = [chr(c) for c in range(0x0900, 0x0980)]          # U+0900..U+097F
PUNCTUATION = list(".,!?;:()[]{}\"'`-–—…/\\@#%&*+=<>|~") + ["।", "॥"]

# ----------------------------------------------------------------------------
# Seed morphemes (injected directly into the vocabulary)
# Contains: administrative terms, banks, political parties, titles,
#           education qualifications, media, international organisations
# ----------------------------------------------------------------------------
SEED_MORPHEMES = [
    # Original prompts
    "गा.वि.स.", "वि.सं.", "ज.ब.रा.", "बि.स.", "इ.सं.", "ई.सं.",
    "प्र.", "डा.", "श्री", "रु.",

    # Administration & government
    "न.पा.", "उ.मा.वि.", "मा.वि.", "प्रा.वि.", "नि.मा.वि.",
    "जि.वि.स.", "गा.पा.", "प्र.जि.अ.", "जि.प्र.का.", "जि.प्र.शा.",
    "स.प्र.", "ने.प्र.", "ने.से.", "स.प्र.नि.", "व.प्र.अ.", "प्र.अ.",
    "ना.प्र.", "ह.प्र.", "जि.अ.", "उ.अ.", "स.अ.", "पुन.अ.", "मु.स.",
    "स.स.", "उ.स.", "स.अ.", "ना.सु.", "ख.सु.", "सु.", "का.मु.",
    "अ.प्र.",

    # Banks (Nepali abbreviations)
    "ने.रा.बैं.", "रा.ब.बैं.", "ने.बैं.लि.", "कृ.वि.बैं.",
    "ना.बि.बैं.", "ए.रे.बैं.", "हि.बि.बैं.", "सि.बि.बैं.",
    "प्र.ब.बैं.", "म.बि.बैं.", "ग्लो.बैं.", "ल.बि.बैं.",
    "ने.इ.बि.बैं.", "एन.आइ.सि.", "सि.बैं.", "कु.बि.बैं.",
    "सा.बि.बैं.", "ने.वि.प्रा.",

    # Media & telecom
    "ने.दू.सं.", "ने.टे.", "ने.ते.नि.", "ने.रे.", "ने.टि.भी.",

    # Political parties
    "ने.का.", "ने.क.पा.", "ने.क.पा.ए.मा.ले.", "ने.क.पा.मा.के.",
    "ने.क.पा.ए.स.", "रा.प्र.पा.", "रा.ज.पा.", "ज.म.पा.", "सं.पा.",

    # Titles and designations
    "प्रा.", "प्रा.लि.", "प.लि.", "लि.", "प्र.म.", "उ.प्र.म.",
    "स.प्र.म.नि.", "ने.प्र.म.नि.", "प्र.से.", "र.से.", "उ.र.से.",
    "स.से.", "म.न.पा.", "उ.म.न.पा.", "जि.स.", "न.स.", "गा.स.",
    "व.स.", "नि.प्रा.", "शा.अ.",

    # Education / degrees
    "एस.एल.सी.", "एस.इ.इ.", "पि.एच.डी.", "बि.ए.", "बि.कम.",
    "एम.ए.", "एम.कम.", "एम.बि.ए.", "बि.बि.ए.", "आइ.ए.",
    "आइ.कम.", "सि.ए.", "बि.एस्सी.", "एम.एस्सी.", "एम.डी.",
    "एम.बि.बि.एस.", "बि.डि.एस.",

    # Miscellaneous official / written
    "इ.का.", "म.स.", "ख.", "अ.स.", "का.खा.", "टि.का.", "द्र.",
    "उ.", "प्र.स.", "स.चि.", "सहा.स.", "प्र.आ.", "व.प्र.आ.",
    "स.प्र.अ.", "जि.पं.", "ने.म.सं.", "रा.स्वा.", "अ.ना.",
    "प्र.ले.", "म.ले.", "सि.डि.ओ.",

    # International organisations
    "ए.डि.बि.", "एन.जि.ओ.", "आइ.एन.जि.ओ.", "यु.एन.",
    "डब्लु.एच.ओ.", "यु.एन.डि.पि.",
]

# Digit seeds (years + common numbers)
_YEARS = [str(y) for y in range(1900, 2100)]
_COMMON_NUMBERS = [
    "10", "20", "30", "40", "50", "60", "70", "80", "90",
    "100", "200", "500", "1000", "2000",
]
DIGIT_SEEDS = _YEARS + _COMMON_NUMBERS
SEED_MORPHEMES.extend(DIGIT_SEEDS)

V_STRICT    = []   # unconditionally frozen (terminal — never left-extend)
V_AMBIGUOUS = []   # frequency-gated by THETA


def main() -> None:
    tok = PyNepBPETokenizer(folding_rules=FOLDING_RULES)

    base = tok.initialize_vocab(
        DEVANAGARI,
        SEED_MORPHEMES,
        PUNCTUATION,
        V_STRICT,
        V_AMBIGUOUS,
    )
    print(f"base vocab: {base} tokens "
          f"(128 Devanagari + {len(SEED_MORPHEMES)} seeds + "
          f"{len(PUNCTUATION)} punct + 62 Latin/digit + 256 bytes + ZWNJ)",
          flush=True)

    # Sanity check: Latin base tokens MUST exist, or Phase 4 will merge nothing.
    for probe in ("a", "Z", "7"):
        if not tok.vocab_contains(probe):
            print(f"  FATAL: Latin base token '{probe}' missing from vocab. "
                  f"Phase 4 will produce ZERO merges. Is the rebuilt .so loaded?",
                  file=sys.stderr)
            sys.exit(1)
    print("  Latin base alphabet present (a/Z/7) -> Phase 4 can merge\n", flush=True)

    print(f"training on {DATA_PATH}", flush=True)
    print(f"  budget: {DEV_BUDGET} DEV + {LAT_BUDGET} LAT = "
          f"{DEV_BUDGET + LAT_BUDGET} total", flush=True)
    print(f"  theta={THETA}  min_word_freq={MIN_WORD_FREQ}\n", flush=True)

    t0 = time.perf_counter()
    final = tok.train_bilingual_from_file(
        DATA_PATH,
        DEV_BUDGET,
        LAT_BUDGET,
        THETA,
        MIN_WORD_FREQ,
        PROGRESS_LINES,
        PROGRESS_MERGES,
    )
    wall = time.perf_counter() - t0

    print(f"\n=== training complete ===", flush=True)
    print(f"final vocab : {final}", flush=True)
    print(f"wall clock  : {wall:.1f}s  ({wall/60:.1f} min)", flush=True)

    # Save. Escape tab/newline/backslash so the TSV stays parseable; the Rust
    # loader (load_vocab_tsv) reverses exactly this escaping.
    n = tok.vocab_size()
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        for i in range(n):
            s = tok.get_token_surface(i)
            s = s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
            f.write(f"{i}\t{s}\n")
    print(f"vocab saved : {OUT_VOCAB}", flush=True)

    # Smoke test: Nepali, English, mixed, digits, and a seeded abbreviation.
    print("\n=== smoke test ===", flush=True)
    samples = [
        "नेपालको इतिहास धेरै पुरानो छ",
        "the study of mathematics in Nepal",
        "काठमाडौं Nepal मा UNIFIL छ",
        "सन् 2020 मा गा.वि.स.को निर्णय",
    ]
    for s in samples:
        ids = tok.encode(s)
        pieces = [tok.get_token_surface(i) for i in ids]
        ok = tok.decode(ids) == tok.normalize(s)
        words = max(1, len(tok.normalize(s).split()))
        shown = " ".join("·" if p == "\u0120" else p for p in pieces)
        print(f"  {s}")
        print(f"    {len(ids)} tok ({len(ids)/words:.2f}/word) | "
              f"roundtrip={'OK' if ok else 'FAIL'}")
        print(f"    {shown}")


if __name__ == "__main__":
    main()