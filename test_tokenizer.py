import sys
import statistics
import time
from tiny_llm_scratch_with_tokenizer import PyNepBPETokenizer

VOCAB_TSV = 'nepbpe_vocab_bilingual_new.tsv'
#"dataset_ne/nepbpe_vocab_new.tsv"

# MUST be identical to what you trained with (train.py). If these differ,
# normalization drifts and surface lookups miss.
FOLDING_RULES = [
    ("सङ्ग", "संग"),
    ("सँग", "संग"),
]

# 'Ġ' (U+0120) is the byte-alphabet surface for space (0x20). Without
# Ġ-prefixing, each inter-word space is its own token.
SPACE_PIECE = "\u0120"


def show_piece(p: str) -> str:
    """Render a piece for display: space as ·, ZWNJ as <ZWNJ>."""
    if p == SPACE_PIECE:
        return "·"
    if p == "\u200c":
        return "<ZWNJ>"
    return p


SAMPLES = [
    "तिम्रो मुस्कानमा बिहानको उज्यालो भेटेँ",
    "तिम्रो आँखामा आफ्नै संसार देखेँ",
    "शब्दले भन्न नसक्ने भावना",
    "मुटुले चुपचाप तिमीलाई लेखेँ",
    "हावाले तिम्रो नाम बिस्तारै बोलाउँछ",
    "चन्द्रमाले तिम्रो यादमा रात सजाउँछ",
    "टाढा भए पनि मन नजिकै रहन्छ",
    "साँचो माया समयसँग कहिल्यै नहराउँछ",
    "तिमीसँग बितेको प्रत्येक पल",
    "जीवनको सबैभन्दा सुन्दर गीत बन्यो",
    "दुःखका बादल आए पनि",
    "तिम्रो साथले हरेक आँसु मुस्कान बन्यो",
    "माया भनेको केवल शब्द होइन",
    "एकअर्काको सपना बोक्ने यात्रा हो",
    "विश्वास, सम्मान र साथको डोरीले",
    "दुई आत्मालाई सधैं जोड्ने कथा हो",
    "यदि अर्को जन्मको कथा लेखियो भने",
    "फेरि पनि तिमी नै मेरो रोजाइ हुनेछौ",
    "यस जन्मझैं, त्यो जन्ममा पनि",
    "मेरो हरेक प्रार्थनाको उत्तर तिमी नै हुनेछौ",
]


def main(test_file=None) -> None:
    tok = PyNepBPETokenizer(folding_rules=FOLDING_RULES)
    n = tok.load_vocab_tsv(VOCAB_TSV)
    print(f"loaded {n} tokens from {VOCAB_TSV}\n")

    raw_rates, content_rates = [], []
    tok_total = word_total = space_total = fails = 0

    print("=== sample tokenization ===")
    for idx, s in enumerate(SAMPLES, 1):
        print(f"  [{idx}/{len(SAMPLES)}] processing...", end="", flush=True)

        ids = tok.encode(s)
        pieces = [tok.get_token_surface(i) for i in ids]
        norm = tok.normalize(s)
        words = max(1, len(norm.split()))
        spaces = sum(1 for p in pieces if p == SPACE_PIECE)
        content = len(ids) - spaces
        ok = tok.decode(ids) == norm

        raw_rates.append(len(ids) / words)
        content_rates.append(content / words)
        tok_total += len(ids)
        word_total += words
        space_total += spaces
        if not ok:
            fails += 1

        shown = " ".join(show_piece(p) for p in pieces)
        print(f"\r  {s}")
        print(
            f"    {len(ids)} tok = {content} content + {spaces} space | "
            f"{len(ids)/words:.2f}/word ({content/words:.2f} ex-space) | "
            f"roundtrip={'OK' if ok else 'FAIL'}"
        )
        print(f"    {shown}")
        if not ok:
            print(f"    DECODED : {tok.decode(ids)!r}")
            print(f"    EXPECTED: {norm!r}")

    print("\n=== sample summary ===")
    print(
        f"  tokens/word   : mean={statistics.mean(raw_rates):.2f}  "
        f"median={statistics.median(raw_rates):.2f}"
    )
    print(
        f"  ex-space/word : mean={statistics.mean(content_rates):.2f}  "
        f"median={statistics.median(content_rates):.2f}   <- real subword fertility"
    )
    print(
        f"  micro/word    : {tok_total/max(1,word_total):.2f}  "
        f"(space tokens = {space_total}/{tok_total} = "
        f"{100*space_total/max(1,tok_total):.0f}%)"
    )
    print(f"  roundtrip     : {len(SAMPLES)-fails}/{len(SAMPLES)} OK")

    # Optional: fertility over a held-out file (fast, uses the Rust encode path).
    if test_file:
        print(f"\n=== fertility over {test_file} ===")
        space_id = tok.vocab_get_id(SPACE_PIECE)  # int (or None), computed once
        tt = ww = ss = lines = 0
        t0 = time.perf_counter()
        try:
            with open(test_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    if line_num % 1000 == 0:
                        elapsed = time.perf_counter() - t0
                        print(
                            f"  ... line {line_num}: {tt} tokens in {elapsed:.1f}s "
                            f"({tt/max(1,elapsed):.0f} tok/s)",
                            flush=True,
                        )

                    w = len(tok.normalize(line).split())
                    if w == 0:
                        continue

                    ids = tok.encode(line)
                    sp = ids.count(space_id) if space_id is not None else 0
                    tt += len(ids)
                    ss += sp
                    ww += w
                    lines += 1

                    if lines >= 20000:
                        print(f"  Reached {lines} lines limit", flush=True)
                        break

        except FileNotFoundError:
            print(f"  Error: File '{test_file}' not found. Skipping fertility analysis.")
            return
        except KeyboardInterrupt:
            print(f"\n  Interrupted after {lines} lines", flush=True)
            return

        dt = time.perf_counter() - t0
        print(
            f"  lines={lines} | tokens={tt} | tokens/word={tt/max(1,ww):.3f} | "
            f"ex-space/word={(tt-ss)/max(1,ww):.3f} | space-frac={ss/max(1,tt):.3f} | "
            f"{dt:.1f}s ({tt/max(1,dt):.0f} tok/s)"
        )


if __name__ == "__main__":
    # Handle both command-line and Jupyter environments.
    try:
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--f="):
            main(sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)