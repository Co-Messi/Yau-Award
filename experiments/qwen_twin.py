"""Compare detection and counting on the same Qwen inputs."""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from qwen_howmany import (
    QHarness,
    GTallyAdapter,
    ValueHead,
    MeanPoolHead,
    active_region_init,
    DEV,
    V,
    QTYPES,
    RES,
)


TIDX = 0
QTEXT = "is any pair out of order"
TRAIN_LO, TRAIN_HI = 4, 16
EVAL_LENGTHS = [16, 32, 48, 96]
EVAL_KS = [1, 2, 4, 8]
CONTROL_KS = [2, 4, 8]  # k=1 is reported but excluded from the control, see above


def make_twin_batch(bs, n, g, k_swaps=None):
    """Balanced sorted vs k-adjacent-swapped. Labels recomputed from the realized Inv."""
    half = bs // 2
    rows, ks = [], []
    for r in range(bs):
        while True:
            a = torch.sort(torch.randint(0, V, (n,), generator=g)).values
            asc = (a[:-1] < a[1:]).nonzero(as_tuple=True)[0]
            if r < half or len(asc) > 0:
                break  # positives need at least one usable ascent
        if r >= half:
            k = (
                k_swaps
                if k_swaps
                else int(torch.randint(1, 9, (1,), generator=g).item())
            )
            k = min(k, len(asc))
            pick = asc[torch.randperm(len(asc), generator=g)[:k]].sort().values
            for pos_value in pick:
                pos = int(pos_value)
                a[pos], a[pos + 1] = a[pos + 1].clone(), a[pos].clone()
            ks.append(k)
        rows.append(a)
    a = torch.stack(rows)
    ai, aj = a.unsqueeze(2), a.unsqueeze(1)
    m = torch.tril(torch.ones(n, n), diagonal=-1)
    inv = ((aj > ai).float() * m.unsqueeze(0)).sum(dim=(1, 2))
    y = (inv > 0).float()  # exact label, not the swap count
    return a, y


def head_logit(head, arm, h, imask, last, qh):
    tidx = torch.zeros(h.shape[0], dtype=torch.long, device=DEV) + TIDX
    if arm == "tally":
        return head(h, imask, tidx, qh)
    if arm == "meanpool":
        return head(h, imask, tidx)
    return head(h[torch.arange(h.shape[0], device=DEV), last], tidx)


def build_head(arm, d_model):
    if arm == "tally":
        return GTallyAdapter(d_model).to(DEV)
    if arm == "meanpool":
        return MeanPoolHead(d_model).to(DEV)
    return ValueHead(d_model).to(DEV)


def featurize(hz, a, layer):
    qt = [QTEXT] * a.shape[0]
    ids, attn, imask, last, qend = hz.encode([r.tolist() for r in a], qt)
    hs = hz.features(ids, attn, [layer])[layer]
    qh = hs[torch.arange(a.shape[0], device=DEV), qend]
    return hs, imask, last, qh


def train(hz, arm, seed, layer, steps, lr, bs):
    torch.manual_seed(7000 + seed)
    g = torch.Generator().manual_seed(7000 + seed)
    head = build_head(arm, hz.d_model)
    if arm == "tally":
        active_region_init(head, hz, layer, [(QTYPES[TIDX], None)])
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01
    )
    t0 = time.time()
    for step in range(steps):
        n = int(torch.randint(TRAIN_LO, TRAIN_HI + 1, (1,), generator=g).item())
        a, y = make_twin_batch(bs, n, g)
        hs, imask, last, qh = featurize(hz, a, layer)
        logit = head_logit(head, arm, hs, imask, last, qh)
        loss = F.binary_cross_entropy_with_logits(logit, y.to(DEV))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in head.parameters():
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], 1.0)
        opt.step()
        sched.step()
        if step % 300 == 0:
            print(
                f"  [twin {arm} s{seed}] step {step} loss {loss.item():.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return head


@torch.no_grad()
def evaluate(hz, head, arm, seed, layer, n_eval=256, bs=32):
    head.eval()
    out = {}
    for n in EVAL_LENGTHS:
        out[n] = {}
        for k in EVAL_KS:
            g = torch.Generator().manual_seed(99_000 + 13 * n + k + 101 * seed)
            correct = tot = 0
            for _ in range(max(1, n_eval // bs)):
                a, y = make_twin_batch(bs, n, g, k_swaps=k)
                hs, imask, last, qh = featurize(hz, a, layer)
                pred = (head_logit(head, arm, hs, imask, last, qh) > 0).float().cpu()
                correct += (pred == y).sum().item()
                tot += len(y)
            out[n][f"acc_k{k}"] = correct / tot
        print(
            f"    n={n:3d} "
            + "  ".join(f"k{k}={out[n][f'acc_k{k}']:.3f}" for k in EVAL_KS),
            flush=True,
        )
    head.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["value", "meanpool", "tally"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    hz = QHarness()
    all_res = {}
    for arm in args.arms:
        all_res[arm] = {}
        for s in range(args.seeds):
            path = os.path.join(RES, f"twin_{arm}_s{s}.json")
            if os.path.exists(path):
                all_res[arm][s] = json.load(open(path))["results"]
                print(f"skip twin {arm} s{s}", flush=True)
                continue
            print(f"=== twin {arm} seed {s} (layer {args.layer}) ===", flush=True)
            head = train(hz, arm, s, args.layer, args.steps, args.lr, args.bs)
            r = evaluate(hz, head, arm, s, args.layer)
            json.dump(
                {
                    "arm": arm,
                    "seed": s,
                    "layer": args.layer,
                    "control_ks": CONTROL_KS,
                    "results": r,
                },
                open(path, "w"),
                indent=2,
            )
            all_res[arm][s] = r

    print("\n=== k in {2,4,8}, median over seeds ===")
    import statistics as st

    summary = {}
    for arm in all_res:
        summary[arm] = {}
        for n in EVAL_LENGTHS:
            vals = [
                all_res[arm][s][n][f"acc_k{k}"]
                if isinstance(list(all_res[arm][s])[0], int)
                else all_res[arm][s][str(n)][f"acc_k{k}"]
                for s in all_res[arm]
                for k in CONTROL_KS
            ]
            summary[arm][n] = st.median(vals)
        row = "  ".join(f"n={n}:{summary[arm][n]:.3f}" for n in EVAL_LENGTHS)
        print(f"  {arm:9s} {row}")
    print("\n=== k=1 (vanishing-margin corner, NOT part of the control) ===")
    for arm in all_res:
        vals = {}
        for n in EVAL_LENGTHS:
            v = [
                all_res[arm][s][n]["acc_k1"]
                if isinstance(list(all_res[arm][s])[0], int)
                else all_res[arm][s][str(n)]["acc_k1"]
                for s in all_res[arm]
            ]
            vals[n] = st.median(v)
        print(f"  {arm:9s} " + "  ".join(f"n={n}:{vals[n]:.3f}" for n in EVAL_LENGTHS))

    json.dump(summary, open(os.path.join(RES, "twin_summary.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(RES, 'twin_summary.json')}")


if __name__ == "__main__":
    main()
