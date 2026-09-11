"""Make PDF figures from the saved experiment results."""

import glob
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "paper", "figures", "mpl")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.relpath(p, HERE))


# Collapse
ARMS = [
    "vanilla",
    "entmax",
    "logn",
    "deep",
    "fraction",
    "mlp",
    "tally-norm",
    "tally",
    "tally-fixed",
]
LABEL = {
    "vanilla": "softmax (4L)",
    "entmax": "sparsemax (4L)",
    "logn": "log-$n$ scaled (4L)",
    "deep": "softmax deep (6L, wider)",
    "fraction": "fraction readout (outside class)",
    "mlp": "no-attention MLP",
    "tally-norm": "tally head, denominator back on",
    "tally": "tally head (learned)",
    "tally-fixed": "tally head (exact construction)",
}
COLOR = {
    "vanilla": "#d62728",
    "entmax": "#ff7f0e",
    "logn": "#8c564b",
    "deep": "#e377c2",
    "fraction": "#7f7f7f",
    "mlp": "#bcbd22",
    "tally-norm": "#9467bd",
    "tally": "#17becf",
    "tally-fixed": "#1f77b4",
}
MARK = {
    "vanilla": "o",
    "entmax": "s",
    "logn": "^",
    "deep": "v",
    "fraction": "D",
    "mlp": "P",
    "tally-norm": "X",
    "tally": "*",
    "tally-fixed": "h",
}

BANDED = {"vanilla", "tally-norm"}


def load_arm(arm):
    """{n: [per-seed cell]} from results/toy_<arm>_s*.json"""
    cells = {}
    for f in sorted(glob.glob(os.path.join(RES, f"toy_{arm}_s*.json"))):
        d = json.load(open(f))
        r = d.get("results", d)
        r = r.get("trained", r) if isinstance(r, dict) else r
        for n, cell in r.items():
            cells.setdefault(int(n), []).append(cell)
    return dict(sorted(cells.items()))


def fig_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    got = []
    for arm in ARMS:
        cells = load_arm(arm)
        if not cells:
            continue
        got.append(arm)
        ns = list(cells)
        exact_arm = arm == "tally-fixed"
        lw = 1.6 if exact_arm else 0.9
        for ax, key, floor in ((axes[0], "mae", 1e-3), (axes[1], "exact", None)):
            med = [st.median([c[key] for c in cells[n]]) for n in ns]
            if floor is not None:
                med = [max(v, floor) for v in med]
            else:
                med = [v * 100 for v in med]
            ax.plot(
                ns,
                med,
                color=COLOR[arm],
                marker=MARK[arm],
                markersize=3.2,
                linewidth=lw,
                label=LABEL[arm],
                zorder=3 if exact_arm else 2,
            )
            if arm in BANDED and len(cells[ns[0]]) > 1:
                lo = [min(c[key] for c in cells[n]) for n in ns]
                hi = [max(c[key] for c in cells[n]) for n in ns]
                if floor is not None:
                    lo = [max(v, floor) for v in lo]
                    hi = [max(v, floor) for v in hi]
                else:
                    lo, hi = [v * 100 for v in lo], [v * 100 for v in hi]
                ax.fill_between(ns, lo, hi, color=COLOR[arm], alpha=0.13, linewidth=0)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("list length $n$")
        ax.axvspan(8, 64, color="#2ca02c", alpha=0.07, linewidth=0, zorder=0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("mean absolute error")
    axes[0].set_title("error vs. length", fontsize=9)
    axes[1].set_ylabel("exact match (%)")
    axes[1].set_ylim(-3, 103)
    axes[1].set_title("exact match vs. length", fontsize=9)
    axes[0].text(
        0.03,
        0.95,
        "training\nlengths",
        transform=axes[0].transAxes,
        fontsize=6.5,
        va="top",
        color="#2ca02c",
    )

    handles = [
        Line2D(
            [],
            [],
            color=COLOR[a],
            marker=MARK[a],
            markersize=3.2,
            linewidth=1.6 if a == "tally-fixed" else 0.9,
            label=LABEL[a],
        )
        for a in got
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=7.2,
    )
    save(fig, "fig_collapse.pdf")


# Denominator
def fig_denominator():
    d = json.load(open(os.path.join(RES, "howmany_v1", "h2_denominator_control.json")))
    res = d["results"]
    lengths = d["lengths"]
    NICE = {
        "inv": "out of order",
        "drop5": "drop by $\\geq5$",
        "dup": "duplicate pairs",
        "sum_s_20": "sum to $s$",
        "count_v_0": "equal $v$",
        "gt_c_8": "above $c$",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    cmap = plt.get_cmap("tab10")
    for i, (t, rec) in enumerate(res.items()):
        c = cmap(i)
        lab = NICE.get(t, t)
        norm_r2 = [rec["cells"][str(n)]["best"]["r2"] for n in lengths]
        norm_ex = [rec["cells"][str(n)]["best"]["exact"] * 100 for n in lengths]
        axes[0].plot(
            lengths,
            norm_r2,
            color=c,
            marker="o",
            markersize=3.2,
            linewidth=1.0,
            label=lab,
        )
        axes[1].plot(
            lengths, norm_ex, color=c, marker="o", markersize=3.2, linewidth=1.0
        )

    axes[0].plot(
        lengths,
        [1.0] * len(lengths),
        color="black",
        linewidth=2.0,
        zorder=5,
        label="sum (all six)",
    )
    axes[1].plot(
        lengths, [100.0] * len(lengths), color="black", linewidth=2.0, zorder=5
    )

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("list length $n$")
        ax.axvspan(8, 64, color="#2ca02c", alpha=0.07, linewidth=0, zorder=0)
    axes[0].axhline(0, color="black", linewidth=0.6, linestyle=":")
    axes[0].set_ylabel("$R^2$")
    axes[0].set_ylim(-4.0, 1.35)
    axes[0].set_title(
        "same weights, same gate: only the division differs", fontsize=8.5
    )
    axes[1].set_ylabel("exact match (%)")
    axes[1].set_ylim(-4, 108)
    axes[1].set_title("exact match", fontsize=8.5)
    axes[0].text(
        0.03,
        0.90,
        "sum: $R^2=1.00$ at every length",
        transform=axes[0].transAxes,
        fontsize=6.8,
        va="top",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=7.2,
    )
    save(fig, "fig_denominator.pdf")


# Scatter
def fig_scatter():
    d = json.load(open(os.path.join(RES, "scatter_n512.json")))
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    lim = max(d["y"]) * 1.05
    ax.plot([0, lim], [0, lim], ls=":", color="black", linewidth=0.8, zorder=1)
    ax.scatter(
        d["y"],
        d["vanilla"],
        s=4,
        color="#d62728",
        alpha=0.55,
        linewidths=0,
        label="softmax (4L)",
        zorder=2,
    )
    ax.scatter(
        d["y"],
        d["tally_fixed"],
        s=4,
        color="#1f77b4",
        alpha=0.55,
        linewidths=0,
        label="tally head (exact)",
        zorder=3,
    )
    ax.set_xlabel("true inversion count at $n=512$")
    ax.set_ylabel("prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    band = max(d["vanilla"]) - min(d["vanilla"])
    cx = st.median(d["y"])
    cy = st.median(d["vanilla"])
    ax.annotate(
        f"every prediction lives in\na band of width {band:,.0f}",
        xy=(cx, cy),
        xytext=(lim * 0.06, lim * 0.34),
        fontsize=7,
        ha="left",
        color="#d62728",
        arrowprops=dict(
            arrowstyle="->", lw=0.7, color="#d62728", connectionstyle="arc3,rad=-0.2"
        ),
    )
    ax.legend(loc="upper left", fontsize=7.2)
    save(fig, "fig_scatter.pdf")


# Twin
def fig_twin():
    KS = [1, 2, 4, 8]
    cells = {}
    for f in sorted(glob.glob(os.path.join(RES, "toy_twin_s*.json"))):
        r = json.load(open(f))
        r = r.get("results", r)
        for n, cell in r.items():
            cells.setdefault(int(n), []).append(cell)
    ns = sorted(cells)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(KS):
        key = f"acc_k{k}"
        med = [
            st.median([c[key] for c in cells[n]]) * 100
            for n in ns
            if key in cells[n][0]
        ]
        xs = [n for n in ns if key in cells[n][0]]
        ax.plot(
            xs,
            med,
            color=cmap(i / 3.4),
            marker="o",
            markersize=3.2,
            linewidth=1.1,
            label=f"$k={k}$ swaps",
        )
    ax.axhline(50, color="black", ls=":", linewidth=0.8)
    ax.text(max(ns), 51.5, "chance", fontsize=6.5, ha="right", va="bottom")
    ax.axvspan(min(ns), 64, color="#2ca02c", alpha=0.07, linewidth=0, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("list length $n$")
    ax.set_ylabel("detection accuracy (%)")
    ax.set_ylim(40, 103)
    ax.legend(loc="lower left", fontsize=7.2, bbox_to_anchor=(0.0, 0.06))
    save(fig, "fig_twin.pdf")


# Six questions
def fig_sixquestions():
    """Left: toy far-length error / no-model error. Right: frozen-LLM held-out questions."""
    labels = ["inv", "drop", "dup", "sum", "$=v$", "$>c$"]
    softmax = [2.30, 2.23, 1.06, 0.91, 1.67, 1.67]
    tally = [0.01, 0.001, 0.53, 0.43, 0.00, 0.42]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    x = range(len(labels))
    w = 0.38
    axes[0].bar(
        [i - w / 2 for i in x], softmax, w, color="#d62728", alpha=0.85, label="softmax"
    )
    axes[0].bar(
        [i + w / 2 for i in x], tally, w, color="#1f77b4", alpha=0.9, label="tally head"
    )
    axes[0].axhline(1.0, ls="--", color="black", linewidth=0.8)
    axes[0].text(len(labels) - 0.6, 1.06, "no-model guess", fontsize=6.5, ha="right")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("far-length error / no-model error", fontsize=8)
    axes[0].set_title("toy: six questions, $8\\times$ length", fontsize=8.5)
    axes[0].legend(fontsize=7.2, loc="upper right")

    qlab = ["$=3$", "$>2$", "$=7$", "$=8$", "$=9$", "$>4$"]
    qval = [0.94, 0.90, 0.92, 0.89, 0.84, 0.89]
    cols = ["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c", "#2ca02c", "#2ca02c"]
    axes[1].bar(range(6), qval, 0.6, color=cols, alpha=0.88)
    axes[1].set_xticks(range(6))
    axes[1].set_xticklabels(qlab, fontsize=8)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("$R^2$ at $n=96$ (frozen LLM)", fontsize=8)
    axes[1].set_title("one head, question in words", fontsize=8.5)
    axes[1].text(0.5, 0.99, "trained", ha="center", fontsize=7, color="#1f77b4")
    axes[1].text(3.5, 0.95, "never asked", ha="center", fontsize=7, color="#2ca02c")
    save(fig, "fig_sixquestions.pdf")


if __name__ == "__main__":
    fig_collapse()
    fig_denominator()
    fig_scatter()
    fig_twin()
    fig_sixquestions()
