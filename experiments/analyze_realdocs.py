"""Summarize the saved real-document experiment results."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "realdocs")
NICE = {
    "year_inv": "pairs out of year order",
    "subj_dup": "pairs sharing a subject",
    "region_ct": "rows from mainland",
}
KINDS = ["year_inv", "subj_dup", "region_ct"]
LENS = [16, 32, 64, 128, 256]


def load():
    op = json.load(open(os.path.join(RES, "real_pro_onepass.json")))["cells"]
    ct = json.load(open(os.path.join(RES, "real_pro_cot.json")))["cells"]
    return op, ct


def main():
    op, ct = load()
    hdr = (
        "question",
        "n",
        "1pass ex",
        "CoT ex",
        "CoT ans",
        "CoT tok",
        "CoT s",
        "xtok",
        "xsec",
    )
    print("%-26s %4s %9s %7s %8s %8s %7s %7s %6s" % hdr)
    print("-" * 92)
    for kind in KINDS:
        for n in LENS:
            k = "%s_n%d" % (kind, n)
            if k not in op or k not in ct:
                continue
            o, c = op[k], ct[k]
            xt = c["out_tok"] / max(o["out_tok"], 1)
            xs = c["sec"] / max(o["sec"], 0.01)
            print(
                "%-26s %4d %8.0f%% %6.0f%% %7.0f%% %8.0f %7.1f %6.0fx %5.0fx"
                % (
                    NICE[kind],
                    n,
                    o["exact_rate"] * 100,
                    c["exact_rate"] * 100,
                    c["answer_rate"] * 100,
                    c["out_tok"],
                    c["sec"],
                    xt,
                    xs,
                )
            )
        print()

    print("Incorrect chain-of-thought answers:")
    tot_ans = tot_wrong = 0
    for kind in KINDS:
        for n in LENS:
            k = "%s_n%d" % (kind, n)
            if k not in ct:
                continue
            for r in ct[k]["records"]:
                if r["pred"] is not None:
                    tot_ans += 1
                    if round(r["pred"]) != r["truth"]:
                        tot_wrong += 1
    print("  answered %d, wrong %d" % (tot_ans, tot_wrong))

    print("\nCoT output tokens as the document grows:")
    for kind in KINDS:
        ts = [
            ct["%s_n%d" % (kind, n)]["out_tok"]
            for n in LENS
            if "%s_n%d" % (kind, n) in ct
        ]
        print("  %-26s %s" % (NICE[kind], " -> ".join("%.0f" % t for t in ts)))

    print("\nOne pass compared with the simple baseline:")
    for kind in KINDS:
        for n in LENS:
            k = "%s_n%d" % (kind, n)
            if k not in op or op[k]["mae"] is None:
                continue
            o = op[k]
            verdict = "worse than null" if o["mae"] > o["null_mae"] else "beats null"
            print(
                "  %-26s n=%-4d MAE %9.1f  null %7.1f  %s (truth ~%.0f)"
                % (NICE[kind], n, o["mae"], o["null_mae"], verdict, o["mean_truth"])
            )


if __name__ == "__main__":
    main()
