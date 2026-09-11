"""Run counting questions on windows from a real document."""

import argparse
import csv
import json
import os
import random
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "realdocs")
SRC = os.environ.get("YAU_INDEX_PATH", os.path.join(HERE, "MASTER_INDEX.tsv"))

COLS = ["year", "subject", "medal", "region", "school"]
# These recreate the windows used in the saved reasoning run.
SEED_OFFSETS = {"year_inv": 48, "subj_dup": 655, "region_ct": 888}


def load_rows(path=SRC):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")]
    keep = []
    for r in rows:
        if r.get("year") and r["year"].strip().isdigit():
            keep.append({c: (r.get(c) or "").strip() or "unknown" for c in COLS})
    return keep


def render(window):
    """Turn a group of rows into a plain text table."""
    head = " | ".join(c for c in COLS)
    lines = [head, "-" * len(head)]
    for i, r in enumerate(window, 1):
        lines.append(f"{i}. " + " | ".join(r[c] for c in COLS))
    return "\n".join(lines)


# Ground truth
def truth(kind, window):
    if kind == "year_inv":  # pairs listed out of year order
        ys = [int(r["year"]) for r in window]
        return sum(1 for i in range(len(ys)) for j in range(i) if ys[j] > ys[i])
    if kind == "subj_dup":  # pairs sharing a subject
        ss = [r["subject"] for r in window]
        return sum(1 for i in range(len(ss)) for j in range(i) if ss[j] == ss[i])
    if kind == "region_ct":  # rows from the mainland
        return sum(1 for r in window if r["region"] == "mainland")
    raise ValueError(kind)


def question(kind, n):
    """Write the question with a clear definition of a pair."""
    npairs = n * (n - 1) // 2
    if kind == "year_inv":
        return (
            f"The table has {n} rows, so there are {npairs} ordered pairs of rows to "
            "consider. Counting over all of those pairs, how many pairs are listed out "
            "of year order? A pair counts when the row appearing EARLIER in the table "
            "has a LATER year than the row appearing after it. This counts PAIRS of "
            f"rows, not rows, so the answer can be anywhere from 0 to {npairs}."
        )
    if kind == "subj_dup":
        return (
            f"The table has {n} rows, so there are {npairs} pairs of rows to consider. "
            "How many pairs of rows have the same value in the subject column? This "
            "counts PAIRS, not rows: if three rows A, B and C all shared a subject, "
            "that is 3 pairs (A-B, A-C, B-C), not 3 rows. The answer can be anywhere "
            f"from 0 to {npairs}."
        )
    if kind == "region_ct":
        return (
            f"The table has {n} rows. How many of those rows have the value 'mainland' "
            f"in the region column? The answer can be anywhere from 0 to {n}."
        )
    raise ValueError(kind)


KIND = {"year_inv": "pair", "subj_dup": "pair", "region_ct": "item"}


def prompt(kind, window):
    return (
        f"{render(window)}\n\n{question(kind, len(window))}\n\n"
        "Reply with the number only, no words and no explanation."
    )


# Scoring
RANGE_RE = re.compile(r"(-?\d[\d,]*)\s*(?:to|-|\u2013|~)\s*(-?\d[\d,]*)")
INT_RE = re.compile(r"-?\d[\d,]*")


def parse_answer(text):
    if not text:
        return None, False
    t = text.strip()
    m = RANGE_RE.search(t)
    if m:
        a, b = (int(x.replace(",", "")) for x in m.groups())
        return (a + b) / 2.0, True
    m = INT_RE.search(t)
    if m:
        hedged = bool(
            re.search(
                r"(\babout\b|approx\w*|\baround\b|\broughly\b|estimat\w*|~|\bat least\b|\bover\b)",
                t,
                re.I,
            )
        )
        return float(m.group(0).replace(",", "")), hedged
    return None, False


def window_seed(kind, n):
    return 4242 + n + SEED_OFFSETS[kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    ap.add_argument(
        "--docs", type=int, default=20, help="windows per (question, length)"
    )
    ap.add_argument("--kinds", nargs="+", default=["year_inv", "subj_dup", "region_ct"])
    ap.add_argument("--model", default="pro")
    ap.add_argument("--thinking", action="store_true", help="the CoT arm")
    ap.add_argument(
        "--dry", action="store_true", help="build + verify data, no API calls"
    )
    ap.add_argument("--source", default=SRC, help="path to MASTER_INDEX.tsv")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    rows = load_rows(args.source)
    print(f"{len(rows)} real rows from MASTER_INDEX.tsv", flush=True)

    if not args.dry:
        from ds_client import client, ask

        c = client()

    arm = "cot" if args.thinking else "onepass"
    out = {
        "model": args.model,
        "arm": arm,
        "source": os.path.basename(args.source),
        "cells": {},
    }
    t_start = time.time()

    for kind in args.kinds:
        for n in args.lengths:
            if n > len(rows):
                continue

            rng = random.Random(window_seed(kind, n))
            wins = []
            for _ in range(args.docs):
                st = rng.randint(0, len(rows) - n)
                w = rows[st : st + n][:]
                rng.shuffle(w)
                wins.append(w)
            gts = [truth(kind, w) for w in wins]
            key = f"{kind}_n{n}"

            if args.dry:
                print(
                    f"  {key:20s} docs={len(wins):3d}  truth range {min(gts)}..{max(gts)}  "
                    f"mean {sum(gts) / len(gts):.1f}"
                )
                out["cells"][key] = {"truth": gts}
                continue

            recs = []
            for w, gt in zip(wins, gts):
                try:
                    r = ask(
                        c, prompt(kind, w), model=args.model, thinking=args.thinking
                    )
                except RuntimeError as e:
                    print(f"    stopped: {e}", flush=True)
                    raise
                val, hedged = parse_answer(r["text"])
                recs.append(
                    {
                        "truth": gt,
                        "pred": val,
                        "hedged": hedged,
                        "raw": r["text"][:80],
                        "in_tok": r["in_tokens"],
                        "out_tok": r["out_tokens"],
                        "rt": r["reasoning_tokens"],
                        "sec": r["seconds"],
                    }
                )
            mu = sum(gts) / len(gts)  # what a guesser who ignores the table says
            null_mae = sum(abs(mu - g) for g in gts) / len(gts)
            null_exact = sum(1 for g in gts if round(mu) == g) / len(gts)
            answered = [x for x in recs if x["pred"] is not None]
            errs = [abs(x["pred"] - x["truth"]) for x in answered]
            exact = sum(1 for x in answered if round(x["pred"]) == x["truth"])
            out["cells"][key] = {
                "n": n,
                "kind": kind,
                "docs": len(recs),
                "answer_rate": len(answered) / len(recs),
                "hedge_rate": sum(1 for x in recs if x["hedged"]) / len(recs),
                "exact_rate": exact / len(recs),
                "mae": (sum(errs) / len(errs)) if errs else None,
                "mean_truth": mu,
                "null_mae": null_mae,
                "null_exact": null_exact,
                "in_tok": sum(x["in_tok"] for x in recs) / len(recs),
                "out_tok": sum(x["out_tok"] for x in recs) / len(recs),
                "sec": sum(x["sec"] for x in recs) / len(recs),
                "records": recs,
            }
            cc = out["cells"][key]
            print(
                f"  {key:20s} exact={cc['exact_rate'] * 100:5.1f}%  "
                f"MAE={cc['mae'] if cc['mae'] is None else round(cc['mae'], 1)}  "
                f"truth~{cc['mean_truth']:.0f} (null MAE {cc['null_mae']:.0f})  "
                f"ans={cc['answer_rate'] * 100:.0f}%  "
                f"out_tok={cc['out_tok']:.0f}  {cc['sec']:.1f}s",
                flush=True,
            )

    if not args.dry:
        p = os.path.join(RES, f"real_{args.model}_{arm}.json")
        json.dump(out, open(p, "w"), indent=2)
        print(f"\nwrote {p}  ({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
