"""Test what happens when the tally head uses a denominator."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from howmany_tasks import KIND, witness_logits, sample, _pair_mask  # noqa: E402
from toy_howmany import V, QTYPES, TOY_PARAMS, TOY_HELD_OUT, fixed_n_metrics, normalizer  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "howmany_v1")

TRAIN_BAND = [8, 16, 32, 48, 64]  # the lengths the toy trains on
EVAL_LENGTHS = [16, 64, 128, 256, 512]  # the lengths the paper reports
BATCH = 256


QSEED = {t: 50_000 + 1_000 * i for i, t in enumerate(QTYPES)}


def gate_sums(name, a, param):
    """Use the same scores with either a sum or a mean."""
    gate = torch.clamp(witness_logits(name, a, param), 0.0, 1.0)
    if KIND[name] == "item":
        s = gate.sum(dim=1)
    else:
        n = a.shape[1]
        s = (gate * _pair_mask(n, a.device).unsqueeze(0)).sum(dim=(1, 2))
    return s, s / normalizer(name, a.shape[1])


def fit_affine(xs, ys):
    """Fit a line using all training lengths together."""
    x = torch.cat(xs).double()
    y = torch.cat(ys).double()
    xm, ym = x.mean(), y.mean()
    var = ((x - xm) ** 2).sum()
    alpha = (((x - xm) * (y - ym)).sum() / var).item() if var > 1e-12 else 0.0
    return alpha, (ym - alpha * xm).item()


def main():
    os.makedirs(RES, exist_ok=True)
    out = {}
    print(
        f"{'question':>10}  {'n':>5}  {'sum MAE':>10}  {'mean MAE':>12}  "
        f"{'mean+cal MAE':>14}  {'best MAE':>12}  {'null MAE':>10}  {'best R2':>9}"
    )
    print("-" * 96)

    for t in QTYPES:
        param = [p for p in TOY_PARAMS[t] if p not in TOY_HELD_OUT[t]][0]

        cal_x, cal_y = [], []
        for n in TRAIN_BAND:
            g = torch.Generator().manual_seed(QSEED[t] + 40_000 + n)
            a, y = sample(t, BATCH, n, V, g, param)
            _, mean_pred = gate_sums(t, a, param)
            cal_x.append(mean_pred)
            cal_y.append(y)
        alpha, b = fit_affine(cal_x, cal_y)

        tname = f"{t}" + (f"_{param}" if param is not None else "")
        out[tname] = {
            "calibration": {"alpha": alpha, "b": b, "band": TRAIN_BAND},
            "cells": {},
        }

        for n in EVAL_LENGTHS:
            g = torch.Generator().manual_seed(QSEED[t] + n)
            a, y = sample(t, BATCH, n, V, g, param)
            s, m = gate_sums(t, a, param)
            cal = alpha * m + b

            m_raw = fixed_n_metrics(m.tolist(), y.tolist())
            m_cal = fixed_n_metrics(cal.tolist(), y.tolist())

            best_is_cal = m_cal["mae"] <= m_raw["mae"]
            cell = {
                "sum": fixed_n_metrics(s.tolist(), y.tolist()),
                "mean": m_raw,
                "mean_cal": m_cal,
                "best": dict(
                    m_cal if best_is_cal else m_raw,
                    readout="calibrated" if best_is_cal else "raw mean",
                ),
                "y_mean": y.mean().item(),
                "y_max": y.max().item(),
                "pred_range_cal": [cal.min().item(), cal.max().item()],
            }
            out[tname]["cells"][n] = cell
            print(
                f"{tname:>10}  {n:>5}  {cell['sum']['mae']:>10.3f}  {cell['mean']['mae']:>12.2f}"
                f"  {cell['mean_cal']['mae']:>14.2f}  {cell['best']['mae']:>12.2f}"
                f"  {cell['sum']['null_mae']:>10.1f}  {cell['best']['r2']:>9.3f}"
            )
        print()

    path = os.path.join(RES, "h2_denominator_control.json")
    json.dump(
        {
            "train_band": TRAIN_BAND,
            "lengths": EVAL_LENGTHS,
            "batch": BATCH,
            "results": out,
        },
        open(path, "w"),
        indent=2,
    )
    print(f"wrote {path}")

    print("\n=== normalized result (better of raw mean and calibrated mean) ===")
    worse_than_null, any_exact, pos_r2 = [], [], []
    for tname, rec in out.items():
        lo, hi = EVAL_LENGTHS[0], EVAL_LENGTHS[-1]
        c_lo, c_hi = rec["cells"][lo], rec["cells"][hi]
        if c_hi["best"]["mae"] > c_hi["sum"]["null_mae"]:
            worse_than_null.append(tname)
        if any(rec["cells"][n]["best"]["exact"] > 0 for n in EVAL_LENGTHS):
            any_exact.append(tname)
        if any(rec["cells"][n]["best"]["r2"] > 0 for n in EVAL_LENGTHS):
            pos_r2.append(tname)
        print(
            f"{tname:>10}: sum exact {c_lo['sum']['exact']:.2f}->{c_hi['sum']['exact']:.2f} "
            f"(MAE {c_hi['sum']['mae']:.3f})   |   normalized exact "
            f"{c_lo['best']['exact']:.2f}->{c_hi['best']['exact']:.2f} "
            f"(R2 {c_lo['best']['r2']:.2f}->{c_hi['best']['r2']:.2f}, "
            f"MAE {c_hi['best']['mae']:.1f} vs null {c_hi['sum']['null_mae']:.1f})"
        )

    nq = len(out)
    print(f"\nsum arm exact at every length on {nq}/{nq} questions")
    print(
        f"normalized arm reaches ANY exact match at any length: "
        f"{len(any_exact)}/{nq} {any_exact}"
    )
    print(
        f"normalized arm reaches positive R2 at any length:     "
        f"{len(pos_r2)}/{nq} {pos_r2}"
    )
    print(
        f"normalized arm WORSE than the content-blind null at n={EVAL_LENGTHS[-1]}: "
        f"{len(worse_than_null)}/{nq} {worse_than_null}"
    )


if __name__ == "__main__":
    main()
