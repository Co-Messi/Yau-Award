"""Make the first version of the ablation-ladder figure."""

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
    {"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "figure.dpi": 150}
)


def series(tag, arm, key="mae"):
    fs = sorted(glob.glob(os.path.join(RES, f"qwen{tag}_{arm}_s*.json")))
    if not fs:
        return None, None
    runs = [json.load(open(f))["results"] for f in fs]
    ns = sorted(int(n) for n in runs[0])
    m = np.array([np.median([r[str(n)][key] for r in runs]) for n in ns])
    return ns, m


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
SCALES = [("", "0.6B", "tab:red", "o"), ("_4b", "4B", "darkred", "s")]
TSCALES = [("", "0.6B", "tab:blue", "o"), ("_4b", "4B", "navy", "s")]


ax = axes[0]
for tag, lbl, c, mk in SCALES:
    ns, m = series(tag, "valuehead")
    if ns:
        ax.plot(
            ns,
            np.maximum(m, 1e-2),
            mk + "-",
            color=c,
            ms=4,
            lw=1.8,
            label=f"{lbl} model",
        )
ax.axvspan(4, 16, color="green", alpha=0.07)
ax.text(4.6, 2600, "trained\nhere", fontsize=7.5, color="darkgreen")
ax.annotate(
    "0.6B and 4B fall\non ONE curve:\nscaling 7$\\times$ does\nnot move the wall",
    xy=(64, 840),
    xytext=(9, 220),
    fontsize=7.5,
    color="black",
    arrowprops=dict(arrowstyle="->", lw=0.8),
)
ax.set_yscale("log")
ax.set_xscale("log", base=2)
ax.set_xlabel("list length $n$")
ax.set_ylabel("counting error, MAE (log)")
ax.set_title("Standard readout: collapses at every size")
ax.legend(frameon=False, loc="lower right", title="value head")


ax = axes[1]
for tag, _label, c, _marker in SCALES:  # faint value-head curves for contrast
    ns, m = series(tag, "valuehead")
    if ns:
        ax.plot(ns, np.maximum(m, 1e-2), "-", color=c, lw=0.8, alpha=0.3)
for tag, lbl, c, mk in TSCALES:
    ns, m = series(tag, "tallyinit")
    if ns:
        ax.plot(
            ns,
            np.maximum(m, 1e-2),
            mk + "-",
            color=c,
            ms=4,
            lw=1.8,
            label=f"{lbl} model",
        )
ax.axvspan(4, 16, color="green", alpha=0.07)
ax.text(70, 1600, "(faint red:\nvalue head)", fontsize=7, color="tab:red", alpha=0.8)
ax.annotate(
    "the fix stays low\nat both sizes",
    xy=(96, 11),
    xytext=(9, 80),
    fontsize=7.5,
    color="black",
    arrowprops=dict(arrowstyle="->", lw=0.8),
)
ax.set_yscale("log")
ax.set_xscale("log", base=2)
ax.set_xlabel("list length $n$")
ax.set_ylabel("counting error, MAE (log)")
ax.set_title("Tally head: holds at every size")
ax.legend(frameon=False, loc="lower right", title="tally head")

fig.tight_layout()
out = os.path.join(FIG, "ladder.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)


for tag, lbl, _, _ in SCALES:
    ns, mv = series(tag, "valuehead")
    _, mt = series(tag, "tallyinit")
    if ns is None:
        continue
    d = dict(zip(ns, mv))
    t = dict(zip(ns, mt))
    print(
        f"{lbl}: value-head MAE n=8/96 = {d.get(8):.0f}/{d.get(96):.0f}; "
        f"tally MAE n=96 = {t.get(96):.1f}"
    )
