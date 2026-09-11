"""Make the first Qwen result figure."""

import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "paper", "figures")
LENS = [8, 16, 24, 32, 48, 64, 96]

vh = [
    json.load(open(f))["results"]
    for f in sorted(glob.glob(os.path.join(RES, "qwen_valuehead_s*.json")))
]
ti = [
    json.load(open(f))["results"]
    for f in sorted(glob.glob(os.path.join(RES, "qwen_tallyinit_s*.json")))
]
tr = [
    json.load(open(f))["results"]
    for f in sorted(glob.glob(os.path.join(RES, "qwen_tallyrand_s*.json")))
]

COL = {"vh": "orange!85!black", "ti": "blue!70!black", "tr": "black!45"}


def coords(runs, key, sc, agg):
    return " ".join(
        f"({n},{round(agg([r[str(n)][key] for r in runs]) * sc, 2)})" for n in LENS
    )


def band(tag, runs, key, sc, color):
    return (
        f"\\addplot[name path={tag}hi,draw=none,forget plot] coordinates {{{coords(runs, key, sc, max)}}};\n"
        f"\\addplot[name path={tag}lo,draw=none,forget plot] coordinates {{{coords(runs, key, sc, min)}}};\n"
        f"\\addplot[{color},opacity=0.13,forget plot] fill between[of={tag}hi and {tag}lo];\n"
    )


def line(runs, key, sc, color, mark, dashed=False):
    sty = "dashed," if dashed else ""
    return f"\\addplot[{color},{sty}mark={mark},mark size=1.5pt,line width=0.9pt,forget plot] coordinates {{{coords(runs, key, sc, st.median)}}};\n"


LEGEND = (
    "\\addlegendimage{orange!85!black,mark=*}\\addlegendentry{value head}\n"
    "\\addlegendimage{black!45,dashed,mark=triangle*}\\addlegendentry{tally, random init (4/8)}\n"
    "\\addlegendimage{blue!70!black,mark=square*}\\addlegendentry{tally, active-region init (8/8)}\n"
)


def panel(key, sc, ylabel, title, log_y, ymin, ymax, ytick, legend):
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
        opts.append("legend columns=3")
        opts.append("legend to name=qwenleg")
        opts.append(
            "legend style={font=\\footnotesize,draw=none,/tikz/every even column/.append style={column sep=10pt}}"
        )
    s = f"\\nextgroupplot[{', '.join(opts)}]\n"
    s += f"\\addplot[draw=none,fill=green!45!black,opacity=0.08,forget plot] coordinates {{(4,{ymin}) (16,{ymin}) (16,{ymax}) (4,{ymax})}} \\closedcycle;\n"
    s += band("vh" + key, vh, key, sc, COL["vh"]) + line(vh, key, sc, COL["vh"], "*")
    s += band("ti" + key, ti, key, sc, COL["ti"]) + line(
        ti, key, sc, COL["ti"], "square*"
    )
    s += line(tr, key, sc, COL["tr"], "triangle*", dashed=True)
    if legend:
        s += LEGEND
    return s


tex = r"""% Made by experiments/make_qwen_pgf.py.
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=1.7cm},
  width=6.7cm, height=5.3cm,
  xmode=log, log basis x=2, xmin=4, xmax=120,
  xtick={8,16,32,64}, xticklabels={$2^3$,$2^4$,$2^5$,$2^6$},
  xlabel={list length $n$},
  tick label style={font=\footnotesize}, label style={font=\footnotesize},
  title style={font=\small,yshift=-2pt}, axis line style={black!55},
  every axis plot/.append style={line join=round},
]
"""
tex += panel(
    "within10",
    100,
    "within $10\\%$ of truth (\\%)",
    "How close to the true count",
    False,
    -4,
    108,
    "0,20,40,60,80,100",
    legend=True,
)
tex += "\\node[font=\\tiny,green!45!black] at (axis cs:6.4,9) {train};\n"
tex += panel(
    "mae",
    1,
    "counting error, MAE",
    "Counting error vs.\\ length",
    True,
    0.4,
    4000,
    "1,10,100,1000",
    legend=False,
)
tex += r"""\end{groupplot}
\coordinate (leg) at ($(group c1r1.south)!0.5!(group c2r1.south)$);
\node[anchor=north] at ([yshift=-0.9cm]leg) {\pgfplotslegendfromname{qwenleg}};
\end{tikzpicture}
"""

os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "qwen_v3_pgf.tex")
open(out, "w").write(tex)
print("wrote", out, f"({len(vh)} vh, {len(ti)} ti, {len(tr)} tr seeds)")
