"""Make the toy-model length generalization figure."""

import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "paper", "figures")


ARMS = ["vanilla", "entmax", "logn", "deep", "fraction", "mlp", "tally", "tally-fixed"]
LABEL = {
    "vanilla": "softmax (4L)",
    "entmax": "sparsemax (4L)",
    "logn": "log-$n$ scaled (4L)",
    "deep": "softmax deep (6L, wider)",
    "fraction": "fraction readout (outside class)",
    "mlp": "no-attention MLP",
    "tally": "tally head (learned)",
    "tally-fixed": "tally head (exact construction)",
}

COLOR = {
    "vanilla": "red!75!black",
    "entmax": "orange!90!black",
    "logn": "brown!75!black",
    "deep": "magenta!70!black",
    "fraction": "violet!75!black",
    "mlp": "black!45",
    "tally": "cyan!75!black",
    "tally-fixed": "blue!70!black",
}
MARK = {
    "vanilla": "*",
    "entmax": "square*",
    "logn": "triangle*",
    "deep": "diamond*",
    "fraction": "pentagon*",
    "mlp": "x",
    "tally": "o",
    "tally-fixed": "*",
}
MAE_FLOOR = 1e-3  # keep zero errors visible on the log axis
MAE_YMIN = 2e-4
BAND_ARMS = {"vanilla", "tally-fixed"}


def load(arm):
    return [
        json.load(open(f))["results"]
        for f in sorted(glob.glob(os.path.join(RES, f"toy_{arm}_s*.json")))
    ]


def lengths(runs):
    return sorted(int(n) for n in runs[0])


def coords(runs, key, sc, agg, floor=None):
    out = []
    for n in lengths(runs):
        v = agg([r[str(n)][key] for r in runs]) * sc
        if floor is not None:
            v = max(v, floor)
        out.append(f"({n},{round(v, 4)})")
    return " ".join(out)


def band(tag, runs, key, sc, color, floor=None):
    if len(runs) < 2:
        return ""  # draw only the median when there is one seed
    return (
        f"\\addplot[name path={tag}hi,draw=none,forget plot] coordinates {{{coords(runs, key, sc, max, floor)}}};\n"
        f"\\addplot[name path={tag}lo,draw=none,forget plot] coordinates {{{coords(runs, key, sc, min, floor)}}};\n"
        f"\\addplot[{color},opacity=0.13,forget plot] fill between[of={tag}hi and {tag}lo];\n"
    )


def line(runs, key, sc, color, mark, bold=False, dashed=False, floor=None):
    lw = "1.3pt" if bold else "0.9pt"
    sty = "dashed," if dashed else ""
    return (
        f"\\addplot[{color},{sty}mark={mark},mark size=1.5pt,line width={lw},forget plot] "
        f"coordinates {{{coords(runs, key, sc, st.median, floor)}}};\n"
    )


def legend_block():
    s = ""
    for arm in ARMS:
        bold = arm == "tally-fixed"
        dash = arm == "tally"
        lw = "1.3pt" if bold else "0.9pt"
        sty = "dashed," if dash else ""
        s += f"\\addlegendimage{{{COLOR[arm]},{sty}mark={MARK[arm]},mark size=1.5pt,line width={lw}}}\\addlegendentry{{{LABEL[arm]}}}\n"
    return s


def panel(key, sc, ylabel, title, log_y, ymin, ymax, ytick, floor, legend):
    opts = [
        f"title={{{title}}}",
        f"ylabel={{{ylabel}}}",
        f"ymin={ymin}",
        f"ymax={ymax}",
        f"ytick={{{ytick}}}",
    ]
    if log_y:
        opts.append("ymode=log")
    if legend:
        opts.append("legend cell align=left")
        opts.append("legend columns=4")
        opts.append("legend to name=toycollapseleg")
        opts.append(
            "legend style={font=\\footnotesize,draw=none,/tikz/every even column/.append style={column sep=8pt}}"
        )
    s = f"\\nextgroupplot[{', '.join(opts)}]\n"

    s += f"\\addplot[draw=none,fill=green!45!black,opacity=0.08,forget plot] coordinates {{(8,{ymin}) (64,{ymin}) (64,{ymax}) (8,{ymax})}} \\closedcycle;\n"
    for arm in ARMS:
        runs = load(arm)
        if not runs:
            continue
        bold = arm == "tally-fixed"
        dash = arm == "tally"
        if arm in BAND_ARMS:
            s += band(arm + key, runs, key, sc, COLOR[arm], floor)
        s += line(
            runs, key, sc, COLOR[arm], MARK[arm], bold=bold, dashed=dash, floor=floor
        )
    if legend:
        s += legend_block()
    return s


tex = r"""% Made by experiments/make_toycollapse_pgf.py.
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=1.7cm},
  width=6.7cm, height=5.3cm,
  xmode=log, log basis x=2, xmin=13, xmax=620,
  xtick={16,32,64,128,256,512}, xticklabels={$2^4$,$2^5$,$2^6$,$2^7$,$2^8$,$2^9$},
  xlabel={list length $n$},
  tick label style={font=\footnotesize}, label style={font=\footnotesize},
  title style={font=\small,yshift=-2pt}, axis line style={black!55},
  every axis plot/.append style={line join=round},
]
"""
tex += panel(
    "mae",
    1,
    "MAE (log)",
    "Counting error vs.\\ length",
    True,
    MAE_YMIN,
    1.5e5,
    "0.001,0.1,10,1000,100000",
    MAE_FLOOR,
    legend=True,
)
tex += "\\node[font=\\tiny,green!45!black] at (axis cs:30,7e-4) {train};\n"
tex += panel(
    "exact",
    100,
    "exact-match (\\%)",
    "Exact counting vs.\\ length",
    False,
    -4,
    108,
    "0,20,40,60,80,100",
    None,
    legend=False,
)
tex += "\\node[font=\\tiny,green!45!black] at (axis cs:28,58) {train};\n"
tex += r"""\end{groupplot}
\coordinate (leg) at ($(group c1r1.south)!0.5!(group c2r1.south)$);
\node[anchor=north] at ([yshift=-0.9cm]leg) {\pgfplotslegendfromname{toycollapseleg}};
\end{tikzpicture}
"""

os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "toy_collapse_pgf.tex")
open(out, "w").write(tex)
seedcounts = {a: len(load(a)) for a in ARMS}
print("wrote", out)
print("seeds per arm:", seedcounts)
print("lengths:", lengths(load("vanilla")))
