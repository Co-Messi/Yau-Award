"""Make the PGFPlots ablation-ladder figure."""

import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "paper", "figures")


def load(tag, arm):
    fs = sorted(glob.glob(os.path.join(RES, f"qwen{tag}_{arm}_s*.json")))
    return [json.load(open(f))["results"] for f in fs]


VH06, VH4B = load("", "valuehead"), load("_4b", "valuehead")
TI06, TI4B = load("", "tallyinit"), load("_4b", "tallyinit")


COL = {
    "vh06": "red!80!black",
    "vh4b": "red!45!black",
    "ti06": "blue!70!black",
    "ti4b": "blue!30!black",
}


def lengths(runs):
    return sorted(int(n) for n in runs[0])


def coords(runs, agg, key="mae"):
    ns = lengths(runs)
    return " ".join(
        f"({n},{round(agg([r[str(n)][key] for r in runs]), 2)})" for n in ns
    )


def band(tag, runs, color):
    return (
        f"\\addplot[name path={tag}hi,draw=none,forget plot] coordinates {{{coords(runs, max)}}};\n"
        f"\\addplot[name path={tag}lo,draw=none,forget plot] coordinates {{{coords(runs, min)}}};\n"
        f"\\addplot[{color},opacity=0.13,forget plot] fill between[of={tag}hi and {tag}lo];\n"
    )


def line(runs, color, mark):
    return (
        f"\\addplot[{color},mark={mark},mark size=1.5pt,line width=0.9pt,forget plot] "
        f"coordinates {{{coords(runs, st.median)}}};\n"
    )


def faint(runs, color):
    return (
        f"\\addplot[{color},opacity=0.3,line width=0.8pt,forget plot] "
        f"coordinates {{{coords(runs, st.median)}}};\n"
    )


YMIN, YMAX = 0.4, 1e4  # shared y-axis range

tex = r"""% Made by experiments/make_ladder_pgf.py.
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=1.7cm},
  width=6.7cm, height=5.3cm,
  xmode=log, log basis x=2, xmin=4, xmax=170,
  xtick={8,16,32,64,128}, xticklabels={$2^3$,$2^4$,$2^5$,$2^6$,$2^7$},
  xlabel={list length $n$},
  ymode=log, ymin=__YMIN__, ymax=__YMAX__, ytick={1,10,100,1000},
  ylabel={counting error, MAE},
  tick label style={font=\footnotesize}, label style={font=\footnotesize},
  title style={font=\small,yshift=-2pt}, axis line style={black!55},
  legend cell align=left,
  legend style={font=\footnotesize,draw=none,at={(0.97,0.03)},anchor=south east},
  every axis plot/.append style={line join=round},
]
""".replace("__YMIN__", str(YMIN)).replace("__YMAX__", str(YMAX))

# Value-head panel
tex += "\\nextgroupplot[title={Standard readout: collapses at every size},legend style={title={value head}}]\n"
tex += (
    f"\\addplot[draw=none,fill=green!45!black,opacity=0.08,forget plot] "
    f"coordinates {{(4,{YMIN}) (16,{YMIN}) (16,{YMAX}) (4,{YMAX})}} \\closedcycle;\n"
)
tex += band("vh06", VH06, COL["vh06"]) + line(VH06, COL["vh06"], "*")
tex += band("vh4b", VH4B, COL["vh4b"]) + line(VH4B, COL["vh4b"], "square*")
tex += (
    "\\addlegendimage{red!80!black,mark=*}\\addlegendentry{0.6B model}\n"
    "\\addlegendimage{red!45!black,mark=square*}\\addlegendentry{4B model}\n"
)
tex += "\\node[font=\\tiny,green!45!black] at (axis cs:6.6,0.62) {train};\n"
tex += (
    "\\node[font=\\tiny,align=left,anchor=west] at (axis cs:9,260)\n"
    "  {both sizes fall on one curve --\\\\$7\\times$ scaling does not move the wall};\n"
)

# Tally-head panel


tex += (
    "\\nextgroupplot[title={Tally head: holds at every size},"
    "legend style={title={tally head},at={(0.03,0.97)},anchor=north west}]\n"
)
tex += (
    f"\\addplot[draw=none,fill=green!45!black,opacity=0.08,forget plot] "
    f"coordinates {{(4,{YMIN}) (16,{YMIN}) (16,{YMAX}) (4,{YMAX})}} \\closedcycle;\n"
)
tex += faint(VH06, COL["vh06"]) + faint(VH4B, COL["vh4b"])
tex += band("ti06", TI06, COL["ti06"]) + line(TI06, COL["ti06"], "*")
tex += band("ti4b", TI4B, COL["ti4b"]) + line(TI4B, COL["ti4b"], "square*")
tex += (
    "\\addlegendimage{blue!70!black,mark=*}\\addlegendentry{0.6B model}\n"
    "\\addlegendimage{blue!30!black,mark=square*}\\addlegendentry{4B model}\n"
)
tex += "\\node[font=\\tiny,red!80!black,anchor=east] at (axis cs:150,3200) {faint: value head};\n"
tex += (
    "\\node[font=\\tiny,align=left,anchor=west] at (axis cs:9,95)\n"
    "  {the fix stays low\\\\at both sizes};\n"
)

tex += r"""\end{groupplot}
\end{tikzpicture}
"""

os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "ladder_pgf.tex")
open(out, "w").write(tex)
print(f"wrote {out}")
print(
    f"  value head: 0.6B {len(VH06)} seeds n<= {lengths(VH06)[-1]}, "
    f"4B {len(VH4B)} seeds n<= {lengths(VH4B)[-1]}"
)
print(
    f"  tally head: 0.6B {len(TI06)} seeds n<= {lengths(TI06)[-1]}, "
    f"4B {len(TI4B)} seeds n<= {lengths(TI4B)[-1]}"
)
print(
    f"  4B value-head MAE @ n=160 = {st.median([r['160']['mae'] for r in VH4B]):.0f} "
    f"(faint wall fits under ymax={YMAX:.0f})"
)
