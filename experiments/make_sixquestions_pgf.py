"""Make the six-question comparison figure."""

import os

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")

LABELS = ["inv", "drop", "dup", "sum", "$=v$", "$>c$"]
SOFTMAX = [2.30, 2.23, 1.06, 0.91, 1.67, 1.67]
TALLY = [0.01, 0.001, 0.53, 0.43, 0.00, 0.42]
QLAB = ["$=3$", "$>2$", "$=7$", "$=8$", "$=9$", "$>4$"]
QVAL = [0.94, 0.90, 0.92, 0.89, 0.84, 0.89]
TRAINED = 2  # first two were seen in training, the rest never asked


def bars(vals, shift=0.0):
    return " ".join("(%g,%g)" % (i + shift, v) for i, v in enumerate(vals))


L = [
    r"% Made by experiments/make_sixquestions_pgf.py.",
    r"\begin{tikzpicture}",
    r"\definecolor{sqred}{RGB}{214,39,40}",
    r"\definecolor{sqblue}{RGB}{31,119,180}",
    r"\definecolor{sqgreen}{RGB}{44,160,44}",
    r"\begin{groupplot}[",
    r"  group style={group size=2 by 1, horizontal sep=1.7cm},",
    r"  width=6.7cm, height=4.8cm,",
    r"  ybar, /pgf/bar width=5pt, enlarge x limits=0.12,",
    r"  tick label style={font=\footnotesize}, label style={font=\footnotesize},",
    r"  title style={font=\small,yshift=-2pt}, axis line style={black!55},",
    r"  ymajorgrids, grid style={black!12},",
    r"]",
    r"\nextgroupplot[title={toy: six questions, $8\times$ length},",
    r"  ylabel={far-length error / no-model error}, ymin=0, ymax=2.6,",
    r"  xtick={0,1,2,3,4,5}, xticklabels={%s}," % ",".join(LABELS),
    r"  legend cell align=left, legend columns=2,",
    r"  legend style={font=\footnotesize,draw=none,at={(0.97,0.97)},anchor=north east}]",
    r"\addplot[draw=sqred,fill=sqred,fill opacity=0.85] coordinates {%s};"
    % bars(SOFTMAX),
    r"\addlegendentry{softmax}",
    r"\addplot[draw=sqblue,fill=sqblue,fill opacity=0.9] coordinates {%s};"
    % bars(TALLY),
    r"\addlegendentry{tally head}",
    r"\draw[dashed,black,line width=0.7pt] (rel axis cs:0,0.3846) -- (rel axis cs:1,0.3846);",
    r"\node[anchor=south east,font=\tiny] at (rel axis cs:0.98,0.395) {no-model guess};",
    r"\nextgroupplot[title={one head, question in words},",
    r"  ylabel={$R^2$ at $n=96$ (frozen LLM)}, ymin=0, ymax=1.16,",
    r"  xtick={0,1,2,3,4,5}, xticklabels={%s}, /pgf/bar width=9pt]" % ",".join(QLAB),
    r"\addplot[draw=sqblue,fill=sqblue,fill opacity=0.88] coordinates {%s};"
    % bars(QVAL[:TRAINED]),
    r"\addplot[draw=sqgreen,fill=sqgreen,fill opacity=0.88] coordinates {%s};"
    % " ".join("(%g,%g)" % (i + TRAINED, v) for i, v in enumerate(QVAL[TRAINED:])),
    r"\node[font=\tiny,sqblue] at (axis cs:0.5,1.05) {trained};",
    r"\node[font=\tiny,sqgreen] at (axis cs:3.5,1.05) {never asked};",
    r"\end{groupplot}",
    r"\end{tikzpicture}",
]

out = os.path.join(FIG, "sixquestions_pgf.tex")
open(out, "w").write("\n".join(L) + "\n")
print("wrote", os.path.normpath(out), "|", len(L), "lines")
