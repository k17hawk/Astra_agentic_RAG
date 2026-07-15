#!/usr/bin/env python3
"""
Fertility diagnostic: measure tokens/word on any corpus, BROKEN DOWN by content
type, so you learn WHY the number is what it is — not just what it is.

The key split: Devanagari-only lines vs lines containing Latin/digits. If the
pure-Devanagari fertility is good and the mixed fertility is bad, your Devanagari
tokenizer is fine and you have a SCRIPT COVERAGE problem (no Latin/digit merges)
— which more Nepali data and a bigger vocab will NOT fix.

Usage:
    python fertility_report.py file1.txt [file2.txt ...]
    python fertility_report.py --limit 5000 file.txt
"""

import sys
import re
import statistics
import unicodedata
from collections import Counter
from tiny_llm_scratch_with_tokenizer import PyNepBPETokenizer

VOCAB_TSV = 'nepbpe_vocab_bilingual_new.tsv'
#"nepbpe_vocab.tsv"
FOLDING_RULES = [("सङ्ग", "संग"), ("सँग", "संग")]
SPACE_PIECE = "\u0120"

HAS_LATIN  = re.compile(r"[A-Za-z]")
HAS_DIGIT  = re.compile(r"[0-9]")          # ASCII digits
HAS_DEVNUM = re.compile(r"[\u0966-\u096F]")  # Devanagari digits ०-९
DEV_CHAR   = re.compile(r"[\u0900-\u097F]")


def classify_word(w: str) -> str:
    if HAS_LATIN.search(w):
        return "latin"
    if HAS_DIGIT.search(w):
        return "digit"
    if HAS_DEVNUM.search(w):
        return "dev_digit"
    if DEV_CHAR.search(w):
        return "devanagari"
    return "other"


class Bucket:
    __slots__ = ("tokens", "words", "spaces", "n")
    def __init__(self):
        self.tokens = self.words = self.spaces = self.n = 0
    def add(self, tokens, words, spaces):
        self.tokens += tokens; self.words += words
        self.spaces += spaces; self.n += 1
    def row(self, label):
        if self.words == 0:
            return f"  {label:<22} (none)"
        raw = self.tokens / self.words
        ex  = (self.tokens - self.spaces) / self.words
        return (f"  {label:<22} words={self.words:>9,}  "
                f"tok/word={raw:>6.3f}  ex-space={ex:>6.3f}")


def main():
    args = [a for a in sys.argv[1:]]
    limit = 20000
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return

    tok = PyNepBPETokenizer(folding_rules=FOLDING_RULES)
    n = tok.load_vocab_tsv(VOCAB_TSV)
    space_id = tok.vocab_get_id(SPACE_PIECE)
    print(f"loaded {n} tokens from {VOCAB_TSV}\n")

    for path in args:
        print(f"{'='*66}\n{path}\n{'='*66}")

        overall = Bucket()
        dev_only_lines = Bucket()     # lines with NO latin/digits
        mixed_lines = Bucket()        # lines WITH latin/digits
        by_wordtype = {k: Bucket() for k in
                       ("devanagari", "latin", "digit", "dev_digit", "other")}
        worst = []                    # worst-fertility words, for inspection

        lines = 0
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    norm = tok.normalize(line)
                    words = norm.split()
                    if not words:
                        continue

                    ids = tok.encode(line)
                    sp = ids.count(space_id) if space_id is not None else 0
                    overall.add(len(ids), len(words), sp)

                    if HAS_LATIN.search(norm) or HAS_DIGIT.search(norm):
                        mixed_lines.add(len(ids), len(words), sp)
                    else:
                        dev_only_lines.add(len(ids), len(words), sp)

                    # Per-word: encode each word alone (no space token inside).
                    for w in words:
                        t = len(tok.encode(w))
                        by_wordtype[classify_word(w)].add(t, 1, 0)
                        if t >= 5:
                            worst.append((t, w))

                    lines += 1
                    if lines >= limit:
                        break
        except FileNotFoundError:
            print(f"  file not found — skipping\n")
            continue

        print(f"\nlines={lines:,}\n")
        print("BY LINE TYPE")
        print(overall.row("ALL lines"))
        print(dev_only_lines.row("Devanagari-only"))
        print(mixed_lines.row("contains Latin/digit"))
        pct = 100 * mixed_lines.n / max(1, lines)
        print(f"  -> {mixed_lines.n:,}/{lines:,} lines ({pct:.1f}%) contain Latin or digits")

        print("\nBY WORD TYPE  (word encoded alone; no space tokens)")
        for k in ("devanagari", "latin", "digit", "dev_digit", "other"):
            print(by_wordtype[k].row(k))

        # The verdict.
        dev_ex = ((dev_only_lines.tokens - dev_only_lines.spaces)
                  / max(1, dev_only_lines.words))
        mix_ex = ((mixed_lines.tokens - mixed_lines.spaces)
                  / max(1, mixed_lines.words))
        print("\nVERDICT")
        if mixed_lines.words and dev_only_lines.words:
            gap = mix_ex - dev_ex
            print(f"  Devanagari-only ex-space fertility : {dev_ex:.3f}")
            print(f"  Mixed-line      ex-space fertility : {mix_ex:.3f}   (gap {gap:+.3f})")
            lat = by_wordtype["latin"]
            dig = by_wordtype["digit"]
            if lat.words:
                print(f"  Latin words cost {lat.tokens/lat.words:.1f} tokens EACH "
                      f"({lat.words:,} of them)")
            if dig.words:
                print(f"  Digit words cost {dig.tokens/dig.words:.1f} tokens EACH "
                      f"({dig.words:,} of them)")
            if gap > 0.15:
                print("  => SCRIPT COVERAGE problem. A Latin/digit BPE pass is the fix;")
                print("     more Nepali data and a bigger vocab will NOT help these.")
            else:
                print("  => Not a script problem. The Devanagari long tail itself is")
                print("     fragmenting -> bigger vocab budget / lower min_word_freq.")

        if worst:
            print("\nWORST-FRAGMENTING WORDS (>=5 tokens, most common first)")
            cnt = Counter(w for _, w in worst)
            for w, c in cnt.most_common(15):
                t = len(tok.encode(w))
                pieces = " ".join(tok.get_token_surface(i) for i in tok.encode(w))
                print(f"  {t:>2} tok  x{c:<5} {w:<24} {pieces}")
        print()


if __name__ == "__main__":
    main()