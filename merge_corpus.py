#!/usr/bin/env python3
"""
Merge English + Nepali corpora into a single shuffled training file for BPE.

- Splits each input into documents (blank-line delimited, matching your
  dedup script's convention).
- Subsamples English down to a target byte budget (random doc sampling,
  not truncation, so topic/domain diversity is preserved).
- Uses the full Nepali corpus (or subsamples it too, if you set a cap).
- Shuffles at the DOCUMENT level (not line level) before writing, so the
  BPE trainer doesn't see one giant English block followed by one giant
  Nepali block.
- Optionally tags each doc with a language marker line, useful if you
  want to inspect/debug merge behavior per language later (does not
  affect BPE training itself unless your trainer strips/uses it).

Usage:
  python merge_corpus.py \
      --en dataset_en/en_wikipedia_combined_dedup.txt \
      --ne dataset_ne/new_dedup_docs.txt \
      --out dataset_merged/train_corpus.txt \
      --en-target-gb 7.5 \
      --ne-target-gb 11.0 \
      --seed 42
"""

import argparse
import random
import sys
from pathlib import Path


def iter_documents(path):
    """Yield documents as strings; a document is text between blank lines."""
    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if buf:
                    yield "\n".join(buf)
                    buf = []
            else:
                buf.append(line)
    if buf:
        yield "\n".join(buf)


def human_gb(n_bytes):
    return n_bytes / (1024 ** 3)


def collect_docs_to_budget(path, target_bytes, seed, label):
    """
    Reservoir-style pass: read all docs (streaming), keep a running byte
    total. If target_bytes is None, keep everything. If set, use random
    sampling across the whole file (not just the first N bytes) so we
    don't bias toward whatever topics happen to appear early.
    """
    print(f"[{label}] scanning {path} ...", file=sys.stderr)
    docs = []
    sizes = []
    total_bytes = 0
    n = 0
    for doc in iter_documents(path):
        b = len(doc.encode("utf-8")) + 2  # +2 for the blank-line separator on write
        docs.append(doc)
        sizes.append(b)
        total_bytes += b
        n += 1
        if n % 200_000 == 0:
            print(f"    [{label}] {n:,} docs scanned, {human_gb(total_bytes):.2f} GB so far",
                  file=sys.stderr)

    print(f"[{label}] total: {n:,} docs, {human_gb(total_bytes):.2f} GB", file=sys.stderr)

    if target_bytes is None or total_bytes <= target_bytes:
        if target_bytes is not None:
            print(f"[{label}] corpus already under target ({human_gb(total_bytes):.2f} GB "
                  f"<= {human_gb(target_bytes):.2f} GB) — using all of it", file=sys.stderr)
        return docs

    # Random shuffle indices, then greedily take docs until we hit budget.
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    kept = []
    running = 0
    for i in idx:
        if running + sizes[i] > target_bytes:
            continue
        kept.append(docs[i])
        running += sizes[i]
        if running >= target_bytes:
            break

    print(f"[{label}] subsampled to {len(kept):,} docs, {human_gb(running):.2f} GB "
          f"(target was {human_gb(target_bytes):.2f} GB)", file=sys.stderr)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", required=True, help="Path to cleaned English corpus")
    ap.add_argument("--ne", required=True, help="Path to cleaned Nepali corpus")
    ap.add_argument("--out", required=True, help="Output path for merged training corpus")
    ap.add_argument("--en-target-gb", type=float, default=None,
                     help="Cap English at this many GB (subsampled by doc, random). "
                          "Omit to use the full English corpus.")
    ap.add_argument("--ne-target-gb", type=float, default=None,
                     help="Cap Nepali at this many GB. Omit to use the full Nepali corpus.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag-lang", action="store_true",
                     help="Prefix each doc with a __LANG__ marker line (en/ne). "
                          "Off by default — most BPE trainers don't want this mixed "
                          "into the training text itself.")
    args = ap.parse_args()

    en_target = int(args.en_target_gb * (1024 ** 3)) if args.en_target_gb else None
    ne_target = int(args.ne_target_gb * (1024 ** 3)) if args.ne_target_gb else None

    en_docs = collect_docs_to_budget(args.en, en_target, args.seed, "EN")
    ne_docs = collect_docs_to_budget(args.ne, ne_target, args.seed, "NE")

    if args.tag_lang:
        en_docs = ["__LANG_EN__\n" + d for d in en_docs]
        ne_docs = ["__LANG_NE__\n" + d for d in ne_docs]

    merged = en_docs + ne_docs
    rng = random.Random(args.seed + 1)  # different seed than subsampling shuffle
    rng.shuffle(merged)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[MERGE] writing {len(merged):,} shuffled docs to {out_path} ...", file=sys.stderr)
    total_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(merged):
            f.write(doc)
            f.write("\n\n")
            total_written += len(doc.encode("utf-8")) + 2
            if (i + 1) % 200_000 == 0:
                print(f"    [MERGE] {i+1:,}/{len(merged):,} docs written, "
                      f"{human_gb(total_written):.2f} GB", file=sys.stderr)

    print(f"[MERGE] done. final corpus: {human_gb(total_written):.2f} GB, "
          f"{len(merged):,} documents", file=sys.stderr)
    print(f"[MERGE]   EN docs: {len(en_docs):,} | NE docs: {len(ne_docs):,}", file=sys.stderr)


if __name__ == "__main__":
    main()