"""Make the Boolean-twin control figure."""

import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "paper", "figures")
KS = [1, 2, 4, 8]

runs = [
    json.load(open(f))["results"]
    for f in sorted(glob.glob(os.path.join(RES, "toy_twin_s*.json")))
]
LENS = sorted(int(n) for n in runs[0])


COL = {1: "twinA", 2: "twinB", 4: "twinC", 8: "twinD"}
RGB = {1: (70, 52, 128), 2: (43, 116, 142), 4: (37, 172, 130), 8: (155, 217, 60)}
MARK = {1: "*", 2: "square*", 4: "triangle*", 8: "diamond*"}
LBL = {1: "$k=1$ swap", 2: "$k=2$", 4: "$k=4$", 8: "$k=8$"}


def coords(k, agg):
    return " ".join(
        f"({n},{round(agg([r[str(n)][f'acc_k{k}'] for r in runs]) * 100, 2)})"
        for n in LENS
    )


def line(k):
    return (
        f"\\addplot[{COL[k]},mark={MARK[k]},mark size=1.5pt,line width=0.9pt,forget plot] "
        f"coordinates {{{coords(k, st.median)}}};\n"
    )


DEFS = "".join(
    f"\\definecolor{{{COL[k]}}}{{RGB}}{{{RGB[k][0]},{RGB[k][1]},{RGB[k][2]}}}\n"
    for k in KS
)
LEGEND = "".join(
    f"\\addlegendimage{{{COL[k]},mark={MARK[k]}}}\\addlegendentry{{{LBL[k]}}}\n"
    for k in KS
)

YMIN, YMAX = 44, 104

tex = (
    r"""% Made by experiments/make_toytwin_pgf.py.
\begin{tikzpicture}
"""
    + DEFS
    + r"""\begin{axis}[
  width=7.4cm, height=5.6cm,
  xmode=log, log basis x=2, xmin=14, xmax=580,
  xtick={16,32,64,128,256,512}, xticklabels={$2^4$,$2^5$,$2^6$,$2^7$,$2^8$,$2^9$},
  xlabel={list length $n$}, ylabel={detection accuracy (\%)},
  ymin="""
    + str(YMIN)
    + r""", ymax="""
    + str(YMAX)
    + r""", ytick={50,60,70,80,90,100},
  title={Boolean twin: any inversion?},
  tick label style={font=\footnotesize}, label style={font=\footnotesize},
  title style={font=\small,yshift=-2pt}, axis line style={black!55},
  every axis plot/.append style={line join=round},
  legend cell align=left, legend pos=north east,
  legend style={font=\footnotesize,draw=none,fill opacity=0.85,text opacity=1},
]
"""
)

tex += (
    f"\\addplot[draw=none,fill=green!45!black,opacity=0.08,forget plot] coordinates "
    f"{{(14,{YMIN}) (64,{YMIN}) (64,{YMAX}) (14,{YMAX})}} \\closedcycle;\n"
)
tex += f"\\node[font=\\tiny,green!45!black] at (axis cs:30,{YMIN + 3}) {{train}};\n"
tex += "\\addplot[gray,dashed,line width=1pt,forget plot] coordinates {(14,50) (580,50)};\n"
tex += "\\node[font=\\tiny,gray] at (axis cs:380,46.5) {chance};\n"
for k in KS:
    tex += line(k)
tex += LEGEND
tex += r"""\end{axis}
\end{tikzpicture}
"""

os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "toy_twin_pgf.tex")
open(out, "w").write(tex)
mn = min(min(r[str(n)][f"acc_k{k}"] for r in runs for n in LENS) for k in KS) * 100
mx = max(max(r[str(n)][f"acc_k{k}"] for r in runs for n in LENS) for k in KS) * 100
print(f"wrote {out} ({len(runs)} seeds, lens={LENS}, acc range {mn:.1f}-{mx:.1f}%)")
