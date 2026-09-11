"""Add the competition cover page to the paper PDF."""

import os
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.environ.get("YAU_TEMPLATE_PATH", os.path.join(HERE, "paper_template.pdf"))
PAPER = os.path.join(HERE, "paper_v9.pdf")
OUT = os.path.join(HERE, "paper_v9_submission.pdf")

COVER = {
    "参赛学生姓名": "Brayden Siew",
    "中学": "横琴德威国际课程高中项目 (DHHQ)",
    "省份": "Guangdong",
    "国家/地区": "China",
    "指导老师姓名": "Dr. Junaid Khan, Mr. Liang Ren",
    "指导老师单位": "DHHQ",
    "论文题目": "Attention Averages Where Convolution Adds: An Extensivity Dichotomy",
}
POS = {
    "参赛学生姓名": (244, 182),
    "中学": (156, 242),
    "省份": (156, 302),
    "国家/地区": (206, 362),
    "指导老师姓名": (244, 422),
    "指导老师单位": (244, 482),
    "论文题目": (200, 542),
}
LINE_END = 497
SIZE = 15.5  # match the label size, like the reference covers
WRAP_DY = 24  # second line sits clear below the underline

doc = fitz.open(PAPER)
toc = doc.get_toc()
doc.insert_pdf(fitz.open(TEMPLATE), from_page=0, to_page=0, start_at=0)
pg = doc[0]
pg.draw_rect(fitz.Rect(85, 75, 200, 100), color=None, fill=(1, 1, 1))


def has_cjk(t):
    return any(0x4E00 <= ord(c) <= 0x9FFF for c in t)


def text_fits(text, font, width):
    try:
        text_width = fitz.get_text_length(text, fontname=font, fontsize=SIZE)
        return text_width <= width
    except Exception:
        return True


for label, val in COVER.items():
    if not val:
        continue
    x, y = POS[label]
    font = "china-s" if has_cjk(val) else "times-roman"
    width = LINE_END - x - 8
    lines, cur = [], ""
    for w in val.split():
        t = (cur + " " + w).strip()
        if text_fits(t, font, width):
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    for k, ln in enumerate(lines):
        cx = x + 6
        run, run_cjk = "", None
        for ch in ln + "\0":
            c_cjk = has_cjk(ch) if ch != "\0" else None
            if run and c_cjk != run_cjk:
                f = "china-s" if run_cjk else "times-roman"
                pg.insert_text(
                    (cx, y - 5 + k * WRAP_DY), run, fontname=f, fontsize=SIZE
                )
                cx += fitz.get_text_length(run, fontname=f, fontsize=SIZE)
                run = ""
            if ch != "\0":
                run += ch
                run_cjk = c_cjk

doc.set_toc([[level, title, page + 1] for level, title, page in toc])
doc.save(OUT, garbage=3, deflate=True)
print(f"wrote {OUT}: {doc.page_count} pages")
