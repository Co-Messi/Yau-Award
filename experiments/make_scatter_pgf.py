"""Make the prediction scatter plot for the paper."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "paper", "figures")

d = json.load(open(os.path.join(RES, "scatter_n512.json")))
y = d["y"]
vanilla = d["vanilla"]
tally = d["tally_fixed"]
n_pts = len(y)


COL = {"vanilla": "tabred", "tally": "tabblue"}

lim = round(max(y) * 1.05, 2)


def cloud(xs, ys):
    return " ".join(f"({round(x, 2)},{round(v, 2)})" for x, v in zip(xs, ys))


tex = (
    r"""% Made by experiments/make_scatter_pgf.py.
\begin{tikzpicture}
\definecolor{tabred}{RGB}{214,39,40}
\definecolor{tabblue}{RGB}{31,119,180}
\begin{axis}[
  width=7.5cm, height=6.0cm,
  xmin=0, xmax="""
    + f"{lim}"
    + r""", ymin=0, ymax="""
    + f"{lim}"
    + r""",
  xlabel={true inversion count ($n=512$)}, ylabel={prediction},
  title={Bounded box vs.\ growing target},
  tick label style={font=\footnotesize}, label style={font=\footnotesize},
  title style={font=\small,yshift=-2pt}, axis line style={black!55},
  scaled ticks=false, xtick={0,20000,40000,60000}, ytick={0,20000,40000,60000},
  legend cell align=left,
  legend style={at={(0.03,0.97)},anchor=north west,font=\footnotesize,draw=none,fill=none},
]
"""
)

tex += f"\\addplot[black,dotted,line width=0.9pt,forget plot] coordinates {{(0,0) ({lim},{lim})}};\n"

tex += (
    f"\\addplot[only marks,mark=*,mark size=1.3pt,{COL['vanilla']},"
    f"mark options={{fill opacity=0.55,draw opacity=0.55}},forget plot] "
    f"coordinates {{{cloud(y, vanilla)}}};\n"
)
tex += (
    f"\\addplot[only marks,mark=*,mark size=1.3pt,{COL['tally']},"
    f"mark options={{fill opacity=0.55,draw opacity=0.55}},forget plot] "
    f"coordinates {{{cloud(y, tally)}}};\n"
)

tex += (
    f"\\addlegendimage{{only marks,mark=*,{COL['vanilla']},mark size=2pt}}"
    "\\addlegendentry{softmax (4L)}\n"
)
tex += (
    f"\\addlegendimage{{only marks,mark=*,{COL['tally']},mark size=2pt}}"
    "\\addlegendentry{tally (exact)}\n"
)
tex += r"""\end{axis}
\end{tikzpicture}
"""

os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "scatter_pgf.tex")
open(out, "w").write(tex)
print(
    "wrote",
    out,
    f"({n_pts} pts/cloud, lim={lim}, "
    f"vanilla {min(vanilla):.1f}-{max(vanilla):.1f}, y {min(y):.0f}-{max(y):.0f})",
)
