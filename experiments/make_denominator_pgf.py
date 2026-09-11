"""Make the denominator-control figure for the paper."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "paper", "figures")

NICE = {
    "inv": "out of order",
    "drop5": "drop by $\\geq5$",
    "dup": "duplicate pairs",
    "sum_s_20": "sum to $s$",
    "count_v_0": "equal $v$",
    "gt_c_8": "above $c$",
}

TAB10 = [
    "{31,119,180}",
    "{255,127,14}",
    "{44,160,44}",
    "{214,39,40}",
    "{148,103,189}",
    "{140,86,75}",
]
MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*", "o"]


def coords(pairs):
    return " ".join("(%g,%g)" % (x, y) for x, y in pairs)


def main():
    d = json.load(open(os.path.join(RES, "howmany_v1", "h2_denominator_control.json")))
    res, lengths = d["results"], d["lengths"]
    lo, hi = lengths[0] * 0.85, lengths[-1] * 1.18

    L = []
    L.append("% Made by experiments/make_denominator_pgf.py.")
    L.append("\\begin{tikzpicture}")
    for i, c in enumerate(TAB10):
        L.append("\\definecolor{den%d}{RGB}%s" % (i, c))
    L.append("\\begin{groupplot}[")
    L.append("  group style={group size=2 by 1, horizontal sep=1.7cm},")
    L.append("  width=6.7cm, height=5.0cm,")
    L.append("  xmode=log, log basis x=2, xmin=%g, xmax=%g," % (lo, hi))
    L.append(
        "  xtick={%s}, xticklabels={%s},"
        % (
            ",".join(str(n) for n in lengths),
            ",".join("$2^{%d}$" % (n.bit_length() - 1) for n in lengths),
        )
    )
    L.append("  xlabel={list length $n$},")
    L.append(
        "  tick label style={font=\\footnotesize}, label style={font=\\footnotesize},"
    )
    L.append("  title style={font=\\small,yshift=-2pt}, axis line style={black!55},")
    L.append("  every axis plot/.append style={line join=round},")
    L.append("]")

    for panel, (key, ylab, title, ymin, ymax, scale, flat) in enumerate(
        [
            (
                "r2",
                "$R^2$",
                "same weights, same gate: only the division differs",
                -4.0,
                1.35,
                1.0,
                1.0,
            ),
            ("exact", "exact match (\\%)", "exact match", -4, 108, 100.0, 100.0),
        ]
    ):
        opts = "title={%s}, ylabel={%s}, ymin=%g, ymax=%g" % (title, ylab, ymin, ymax)
        if panel == 0:
            opts += (
                ", legend cell align=left, legend columns=4, legend to name=denleg,"
                " legend style={font=\\footnotesize,draw=none,"
                "/tikz/every even column/.append style={column sep=8pt}}"
            )
        L.append("\\nextgroupplot[%s]" % opts)

        L.append(
            "\\addplot[draw=none,fill=green!45!black,opacity=0.08,forget plot] "
            "coordinates {(8,%g) (64,%g) (64,%g) (8,%g)} \\closedcycle;"
            % (ymin, ymin, ymax, ymax)
        )
        if panel == 0:
            L.append(
                "\\addplot[black,dotted,line width=0.5pt,forget plot] "
                "coordinates {(%g,0) (%g,0)};" % (lo, hi)
            )
        for i, (_task, rec) in enumerate(res.items()):
            pts = [(n, rec["cells"][str(n)]["best"][key] * scale) for n in lengths]
            L.append(
                "\\addplot[den%d,mark=%s,mark size=1.4pt,line width=0.85pt,"
                "forget plot] coordinates {%s};" % (i, MARKS[i], coords(pts))
            )
        L.append(
            "\\addplot[black,line width=1.6pt,forget plot] coordinates {%s};"
            % coords([(n, flat) for n in lengths])
        )
        if panel == 0:
            L.append(
                "\\node[anchor=north west,font=\\tiny] at (rel axis cs:0.03,0.97) "
                "{sum: $R^2=1.00$ at every length};"
            )
            for i, t in enumerate(res):
                L.append(
                    "\\addlegendimage{den%d,mark=%s,mark size=1.4pt,line width=0.85pt}"
                    "\\addlegendentry{%s}" % (i, MARKS[i], NICE.get(t, t))
                )
            L.append(
                "\\addlegendimage{black,line width=1.6pt}"
                "\\addlegendentry{sum (all six)}"
            )
    L.append("\\end{groupplot}")
    L.append("\\coordinate (leg) at ($(group c1r1.south)!0.5!(group c2r1.south)$);")
    L.append(
        "\\node[anchor=north] at ([yshift=-0.9cm]leg) "
        "{\\pgfplotslegendfromname{denleg}};"
    )
    L.append("\\end{tikzpicture}")

    out = os.path.join(FIG, "denominator_pgf.tex")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", os.path.normpath(out), "|", len(L), "lines")


if __name__ == "__main__":
    main()
