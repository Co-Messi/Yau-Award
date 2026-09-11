"""Make the updated Qwen result figure."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update(
    {"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5, "figure.dpi": 150}
)

ARMS = [
    ("qwen_valuehead_s*.json", "value head (standard readout)", "tab:orange", "o"),
    ("qwen_tallyrand_s*.json", "tally head (random init)", "tab:gray", "^"),
    ("qwen_tallyinit_s*.json", "tally head (active-region init)", "tab:blue", "s"),
]


def load_arm(pattern):
    runs = []
    paths = sorted(glob.glob(os.path.join(RES, pattern)))
    for path in paths:
        runs.append(json.load(open(path))["results"])
    return runs


def series(runs, key):
    if not runs:
        return None, None, None, None
    ns = sorted(int(n) for n in runs[0])
    rows = []
    for n in ns:
        seed_values = []
        for run in runs:
            seed_values.append(run[str(n)][key])
        rows.append(seed_values)
    vals = np.array(rows)
    return ns, np.median(vals, 1), vals.min(1), vals.max(1)


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

ax = axes[0]
for pat, lbl, c, mk in ARMS:
    ns, med, lo, hi = series(load_arm(pat), "within10")
    if ns is None:
        continue
    ax.plot(ns, 100 * med, mk + "-", color=c, ms=4, lw=1.7, label=lbl)
    ax.fill_between(ns, 100 * lo, 100 * hi, color=c, alpha=0.15)
ax.axvspan(4, 16, color="green", alpha=0.07)
ax.text(4.3, 8, "train\nrange", fontsize=7, color="darkgreen")
ax.set_xscale("log", base=2)
ax.set_xlabel("list length $n$")
ax.set_ylabel("predictions within 10% of truth (%)")
ax.set_title("How close to the true count")
ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.42))

ax = axes[1]
for pat, lbl, c, mk in ARMS:
    ns, med, lo, hi = series(load_arm(pat), "mae")
    if ns is None:
        continue
    med = np.maximum(med, 1e-2)
    lo = np.maximum(lo, 1e-2)
    hi = np.maximum(hi, 1e-2)
    ax.plot(ns, med, mk + "-", color=c, ms=4, lw=1.7, label=lbl)
    ax.fill_between(ns, lo, hi, color=c, alpha=0.15)
ax.axvspan(4, 16, color="green", alpha=0.07)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("list length $n$")
ax.set_ylabel("counting error MAE (log)")
ax.set_title("Counting error vs. length")

fig.tight_layout()
out = os.path.join(FIG, "qwen_v3.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

print("\n--- numbers for the paper text ---")
for pat, lbl, _, _ in ARMS:
    runs = load_arm(pat)
    ns, med, lo, hi = series(runs, "mae")
    if ns is None:
        print(f"{lbl}: (no results yet)")
        continue
    _, w, wl, wh = series(runs, "within10")
    dmed, dlo, dhi = dict(zip(ns, med)), dict(zip(ns, lo)), dict(zip(ns, hi))
    dw = dict(zip(ns, w))
    print(
        f"{lbl}: n={ns[-1]} MAE median {dmed[ns[-1]]:.1f} "
        f"(seed range {dlo[ns[-1]]:.1f}-{dhi[ns[-1]]:.1f}); "
        f"within-10% median {100 * dw[ns[-1]]:.0f}%  [{len(runs)} seeds]"
    )
