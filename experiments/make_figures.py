"""Make the main PGFPlots figures from saved results."""

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
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
    }
)

ARMS = ["vanilla", "entmax", "logn", "deep", "fraction", "mlp", "tally", "tally-fixed"]
LABEL = {
    "vanilla": "softmax (4L)",
    "entmax": "sparsemax (4L)",
    "logn": "log-n scaled (4L)",
    "deep": "softmax deep (6L, wider)",
    "fraction": "fraction readout (outside class)",
    "mlp": "no-attention MLP",
    "tally": "tally head (learned)",
    "tally-fixed": "tally head (exact construction)",
}
COLOR = {
    "vanilla": "tab:red",
    "entmax": "tab:orange",
    "logn": "tab:brown",
    "deep": "tab:pink",
    "fraction": "tab:purple",
    "mlp": "tab:gray",
    "tally": "tab:cyan",
    "tally-fixed": "tab:blue",
}
STYLE = {a: "-" for a in ARMS}
STYLE["tally-fixed"] = "-"
STYLE["tally"] = "--"


def load(arm):
    runs = []
    for f in sorted(glob.glob(os.path.join(RES, f"toy_{arm}_s*.json"))):
        runs.append(json.load(open(f))["results"])
    return runs


def agg(runs, key):
    ns = sorted(int(n) for n in runs[0])
    mean, lo, hi = [], [], []
    for n in ns:
        vals = [r[str(n)][key] for r in runs]
        mean.append(np.mean(vals))
        lo.append(np.min(vals))
        hi.append(np.max(vals))
    return ns, np.array(mean), np.array(lo), np.array(hi)


def fig_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for arm in ARMS:
        runs = load(arm)
        if not runs:
            continue
        ns, m, lo, hi = agg(runs, "mae")
        m = np.maximum(m, 1e-3)
        lo = np.maximum(lo, 1e-3)
        hi = np.maximum(hi, 1e-3)
        ax.plot(ns, m, STYLE[arm], color=COLOR[arm], label=LABEL[arm], lw=1.6)
        ax.fill_between(ns, lo, hi, color=COLOR[arm], alpha=0.15)
    ax.axvspan(8, 64, color="green", alpha=0.07)
    ax.text(11, 2e4, "train\nrange", fontsize=7, color="darkgreen")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("list length $n$")
    ax.set_ylabel("MAE (log)")
    ax.set_title("Counting error vs. length")
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax = axes[1]
    for arm in ARMS:
        runs = load(arm)
        if not runs:
            continue
        ns, m, lo, hi = agg(runs, "exact")
        ax.plot(ns, 100 * m, STYLE[arm], color=COLOR[arm], lw=1.6)
        ax.fill_between(ns, 100 * lo, 100 * hi, color=COLOR[arm], alpha=0.15)
    ax.axvspan(8, 64, color="green", alpha=0.07)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("list length $n$")
    ax.set_ylabel("exact-match (%)")
    ax.set_title("Exact counting vs. length")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "toy_collapse.pdf"), bbox_inches="tight")
    print("toy_collapse.pdf")


def fig_twin():
    runs = load("twin")
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ks = [1, 2, 4, 8]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(ks)))
    for k, c in zip(ks, cmap):
        ns = sorted(int(n) for n in runs[0])
        rows = []
        for n in ns:
            seed_values = []
            for run in runs:
                seed_values.append(run[str(n)][f"acc_k{k}"])
            rows.append(seed_values)
        vals = np.array(rows)
        ax.plot(
            ns, 100 * vals.mean(1), "-o", color=c, ms=3, label=f"$k={k}$ swaps", lw=1.4
        )
        ax.fill_between(ns, 100 * vals.min(1), 100 * vals.max(1), color=c, alpha=0.15)
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.text(280, 51, "chance", fontsize=7, color="gray")
    ax.axvspan(8, 64, color="green", alpha=0.07)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("list length $n$")
    ax.set_ylabel("detection accuracy (%)")
    ax.set_title("Boolean twin: any inversion?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "toy_twin.pdf"), bbox_inches="tight")
    print("toy_twin.pdf")


def fig_qwen():
    arms = ["fewshot", "lora", "valuehead", "tally"]
    label = {
        "fewshot": "Qwen 4-shot (no training)",
        "lora": "Qwen + LoRA (digits out)",
        "valuehead": "Qwen + value head (class $\\mathcal{T}$)",
        "tally": "Qwen + tally head",
    }
    color = {
        "fewshot": "tab:gray",
        "lora": "tab:red",
        "valuehead": "tab:orange",
        "tally": "tab:blue",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, key, tname, ylab in [
        (axes[0], "exact", "Exact-match vs. length", "exact-match (%)"),
        (axes[1], "mae", "MAE vs. length (log)", "MAE (log)"),
    ]:
        for arm in arms:
            runs = []
            for f in sorted(glob.glob(os.path.join(RES, f"qwen_{arm}_s*.json"))):
                runs.append(json.load(open(f))["results"])
            if not runs:
                continue
            ns = sorted(int(n) for n in runs[0])
            rows = []
            for n in ns:
                seed_values = []
                for run in runs:
                    seed_values.append(run[str(n)][key])
                rows.append(seed_values)
            vals = np.array(rows)
            m = vals.mean(1)
            if key == "exact":
                m = 100 * m
                ax.plot(ns, m, "-o", ms=3, color=color[arm], label=label[arm], lw=1.6)
                ax.fill_between(
                    ns,
                    100 * vals.min(1),
                    100 * vals.max(1),
                    color=color[arm],
                    alpha=0.15,
                )
            else:
                m = np.maximum(m, 1e-2)
                ax.plot(ns, m, "-o", ms=3, color=color[arm], lw=1.6)
                ax.fill_between(
                    ns,
                    np.maximum(vals.min(1), 1e-2),
                    np.maximum(vals.max(1), 1e-2),
                    color=color[arm],
                    alpha=0.15,
                )
                ax.set_yscale("log")
        ax.axvspan(4, 16, color="green", alpha=0.07)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("list length $n$")
        ax.set_ylabel(ylab)
        ax.set_title(tname)
    axes[0].text(5, 80, "train\nrange", fontsize=7, color="darkgreen")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "qwen.pdf"), bbox_inches="tight")
    print("qwen.pdf")


def fig_boundedness():
    """Plot predictions against targets at length 512."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    f = os.path.join(RES, "scatter_n512.json")
    if not os.path.exists(f):
        print("scatter_n512.json is missing, so the optional scatter plot was skipped")
        return
    d = json.load(open(f))
    ax.scatter(
        d["y"], d["vanilla"], s=4, alpha=0.4, color="tab:red", label="softmax (4L)"
    )
    ax.scatter(
        d["y"],
        d["tally_fixed"],
        s=4,
        alpha=0.4,
        color="tab:blue",
        label="tally (exact)",
    )
    lim = max(d["y"]) * 1.05
    ax.plot([0, lim], [0, lim], "k:", lw=1)
    ax.set_xlabel("true inversion count ($n=512$)")
    ax.set_ylabel("prediction")
    ax.legend(frameon=False)
    ax.set_title("Bounded box vs. growing target")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scatter.pdf"), bbox_inches="tight")
    print("scatter.pdf")


if __name__ == "__main__":
    fig_collapse()
    fig_twin()
    fig_qwen()
    fig_boundedness()
