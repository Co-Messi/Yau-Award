# Attention Averages Where Convolution Adds: An Extensivity Dichotomy

Code and stored run outputs for a Yau Science Award (Computer Science)
submission, 2026. Sole student author: Brayden Siew.

## What is in here

- `experiments/` - every experiment script: the toy models and ablation
  ladder, the frozen-Qwen heads, the probes, the Boolean-twin control, the
  denominator control, and the DeepSeek real-document benchmark
  (`real_docs.py`, `ds_client.py`).
- `experiments/results/` - the stored JSON outputs that every table and
  figure in the paper is built from. Nothing in the paper is hand-entered;
  the figure generators read these files.
- `experiments/make_*_pgf.py` - figure generators. Each one emits a
  PGFPlots `.tex` file from the stored results.
- `paper/` - LaTeX source of the paper, references, and the scripts that
  assemble the submission PDF.

## How to run

Python 3.13, PyTorch 2.12 (MPS backend), on a single MacBook Pro (Apple
M5 Max, 64 GB). Toy experiments take seconds; the frozen-model runs take
minutes. The DeepSeek benchmark needs a `DEEPSEEK_API_KEY` in the
environment; the stored results in `experiments/results/` mean nothing has
to be re-run to check the paper's numbers.

Install the Python packages first:

```bash
python3 -m pip install -r requirements.txt
```

The real-document experiment also needs `MASTER_INDEX.tsv`. Put it inside
`experiments/`, pass its path with `--source`, or set `YAU_INDEX_PATH`.
The PDF cover script looks for `paper/paper_template.pdf`; a different path
can be set with `YAU_TEMPLATE_PATH`.

The fixed document seeds match the saved reasoning run. The older saved
one-pass run used different windows, so rerun that arm before using it as an
exact same-window comparison.

Full reproduction details are in Appendix B of the paper.
