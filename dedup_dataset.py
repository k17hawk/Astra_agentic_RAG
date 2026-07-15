#!/usr/bin/env python3
"""
Deduplication checker & cleaner for a single large text file.

Designed for `winki_bank_other.txt` (~11.8 GB). Streams the file so memory
stays flat regardless of size. Three passes:

  PASS 1 - Exact line-level dedup (SHA256). 5-10 min for 11.8 GB.
  PASS 2 - Exact document-level dedup (optional, if you split on blank lines
           or a delimiter). Useful if one doc = multiple lines.
  PASS 3 - Near-duplicate detection via MinHash + LSH. Catches lightly
           edited copies (e.g. syndicated news, reposts with byline changes).

Usage:
  # 1) Quick check only (no output file, just stats):
  python dedup_winki_bank.py check winki_bank_other.txt

  # 2) Write a cleaned (exact-deduped) file:
  python dedup_winki_bank.py clean winki_bank_other.txt winki_bank_other.dedup.txt

  # 3) Run near-duplicate detection (slower, ~30-60 min for 11.8 GB):
  python dedup_winki_bank.py near winki_bank_other.txt --threshold 0.7

  # 4) Full pipeline: exact + near dedup, write cleaned output:
  python dedup_winki_bank.py full winki_bank_other.txt winki_bank_other.clean.txt

Requirements:
  pip install datasketch      # only needed for `near` and `full` modes
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK = 1 << 20  # 1 MB read chunks


def human_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def iter_lines(path, encoding="utf-8"):
    """Stream lines without loading the whole file."""
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            yield line.rstrip("\n")


def sha256(s):
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).digest()


# ---------------------------------------------------------------------------
# PASS 1: Exact line-level dedup
# ---------------------------------------------------------------------------

def pass_exact_lines(path, out_path=None):
    print(f"\n[PASS 1] Exact line-level dedup")
    print(f"  Input:  {path}  ({human_size(os.path.getsize(path))})")
    t0 = time.time()
    seen = set()
    total = 0
    kept = 0
    out = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for line in iter_lines(path):
            total += 1
            if not line:
                # Preserve blank lines (they often delimit documents)
                if out:
                    out.write("\n")
                continue
            h = sha256(line)
            if h in seen:
                continue
            seen.add(h)
            kept += 1
            if out:
                out.write(line + "\n")
            if total % 1_000_000 == 0:
                el = time.time() - t0
                print(f"    {total:,} lines processed | "
                      f"{kept:,} unique | {el:.0f}s | "
                      f"{total/el:,.0f} lines/s")
    finally:
        if out:
            out.close()
    el = time.time() - t0
    dupes = total - kept
    print(f"\n  Done in {el:.0f}s")
    print(f"  Total lines:  {total:,}")
    print(f"  Unique lines: {kept:,}")
    print(f"  Duplicates:   {dupes:,} ({100*dupes/max(total,1):.2f}%)")
    return {"total": total, "unique": kept, "duplicates": dupes,
            "dup_rate": 100*dupes/max(total,1)}


# ---------------------------------------------------------------------------
# PASS 2: Document-level dedup (split on blank lines)
# ---------------------------------------------------------------------------

def iter_documents(path):
    """Yield documents as strings, where a document is text between blank lines."""
    buf = []
    for line in iter_lines(path):
        if line == "":
            if buf:
                yield "\n".join(buf)
                buf = []
        else:
            buf.append(line)
    if buf:
        yield "\n".join(buf)


def pass_exact_docs(path, out_path=None):
    print(f"\n[PASS 2] Exact document-level dedup (split on blank lines)")
    t0 = time.time()
    seen = set()
    total = 0
    kept = 0
    out = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for doc in iter_documents(path):
            total += 1
            h = sha256(doc)
            if h in seen:
                continue
            seen.add(h)
            kept += 1
            if out:
                out.write(doc + "\n\n")
            if total % 100_000 == 0:
                el = time.time() - t0
                print(f"    {total:,} docs | {kept:,} unique | {el:.0f}s")
    finally:
        if out:
            out.close()
    el = time.time() - t0
    dupes = total - kept
    print(f"\n  Done in {el:.0f}s")
    print(f"  Total docs:  {total:,}")
    print(f"  Unique docs: {kept:,}")
    print(f"  Duplicates:  {dupes:,} ({100*dupes/max(total,1):.2f}%)")
    return {"total": total, "unique": kept, "duplicates": dupes,
            "dup_rate": 100*dupes/max(total,1)}


# ---------------------------------------------------------------------------
# PASS 3: Near-duplicate detection via MinHash + LSH
# ---------------------------------------------------------------------------

def shingles(text, k=5):
    """Word-level k-shingles. Falls back to char-shingles for very short text."""
    words = text.split()
    if len(words) < k:
        # char-level fallback
        chars = text.replace(" ", "")
        return {chars[i:i+k] for i in range(max(1, len(chars)-k+1))}
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}


def pass_near_dup(path, threshold=0.7, num_perm=256, out_path=None,
                  doc_mode=False):
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("ERROR: install datasketch first:  pip install datasketch")
        sys.exit(1)

    print(f"\n[PASS 3] Near-duplicate detection (MinHash LSH)")
    print(f"  threshold={threshold}  num_perm={num_perm}  doc_mode={doc_mode}")
    t0 = time.time()

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    items = iter_documents(path) if doc_mode else iter_lines(path)

    total = 0
    near_dup = 0
    out = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for text in items:
            total += 1
            if not text or len(text) < 20:
                if out and text:
                    out.write(text + ("\n\n" if doc_mode else "\n"))
                continue
            sh = shingles(text, k=5)
            if not sh:
                continue
            m = MinHash(num_perm=num_perm)
            for s in sh:
                m.update(s.encode("utf-8"))
            # Query before insert: if any near-neighbor exists, treat as dup
            neighbors = lsh.query(m)
            if neighbors:
                near_dup += 1
                continue
            key = f"doc_{total}"
            lsh.insert(key, m)
            if out:
                out.write(text + ("\n\n" if doc_mode else "\n"))
            if total % 100_000 == 0:
                el = time.time() - t0
                print(f"    {total:,} processed | {near_dup:,} near-dups "
                      f"removed | {el:.0f}s")
    finally:
        if out:
            out.close()
    el = time.time() - t0
    kept = total - near_dup
    print(f"\n  Done in {el:.0f}s")
    print(f"  Total:           {total:,}")
    print(f"  Near-dups found: {near_dup:,} ({100*near_dup/max(total,1):.2f}%)")
    print(f"  Kept:            {kept:,}")
    return {"total": total, "near_duplicates": near_dup,
            "kept": kept, "dup_rate": 100*near_dup/max(total,1)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Dedup checker & cleaner for a large single .txt file.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("check", help="Quick exact line-dup report only.")
    pc.add_argument("input")

    pclean = sub.add_parser("clean", help="Exact line-level dedup, write output.")
    pclean.add_argument("input")
    pclean.add_argument("output")

    pdocs = sub.add_parser("docs", help="Exact doc-level dedup (split on blanks).")
    pdocs.add_argument("input")
    pdocs.add_argument("output", nargs="?")

    pnear = sub.add_parser("near", help="Near-duplicate detection (MinHash LSH).")
    pnear.add_argument("input")
    pnear.add_argument("--threshold", type=float, default=0.7)
    pnear.add_argument("--num-perm", type=int, default=256)
    pnear.add_argument("--doc-mode", action="store_true",
                       help="Treat blank-line-separated blocks as docs.")
    pnear.add_argument("--output", help="Optional cleaned output file.")

    pfull = sub.add_parser("full", help="Exact + near dedup, write output.")
    pfull.add_argument("input")
    pfull.add_argument("output")
    pfull.add_argument("--threshold", type=float, default=0.7)

    a = p.parse_args()

    if a.cmd == "check":
        pass_exact_lines(a.input)
    elif a.cmd == "clean":
        pass_exact_lines(a.input, a.output)
    elif a.cmd == "docs":
        pass_exact_docs(a.input, a.output)
    elif a.cmd == "near":
        pass_near_dup(a.input, a.threshold, a.num_perm, a.output, a.doc_mode)
    elif a.cmd == "full":
        import tempfile
        # 1) exact line dedup to temp file
        tmp = a.output + ".exact.tmp"
        pass_exact_lines(a.input, tmp)
        # 2) near-dup on the temp file
        pass_near_dup(tmp, a.threshold, out_path=a.output, doc_mode=False)
        os.remove(tmp)
        print(f"\nFinal cleaned output: {a.output}")
        print(f"Original size: {human_size(os.path.getsize(a.input))}")
        print(f"Cleaned size:  {human_size(os.path.getsize(a.output))}")


if __name__ == "__main__":
    main()
