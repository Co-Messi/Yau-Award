"""Test how sensitive the tally head is to its starting weights."""

import json
import os
import time

import torch
import torch.nn.functional as F

import qwen_extensivity as Q
from qwen_extensivity import (
    QwenHarness,
    TallyAdapter,
    gen_training_batch,
    eval_head,
    make_list,
    inv_count,
    DEV,
)

RES = Q.RESULTS
LENGTHS = [8, 16, 24, 32, 48, 64, 96]
SUCCESS_N = 96
SUCCESS_WITHIN10 = 0.9
N_SEEDS = 8


class TallyAdapterB(TallyAdapter):
    """Tally head with one trainable gate bias."""

    def __init__(self, d_model, rank=4, normalize=False):
        super().__init__(d_model, rank)
        self.beta = torch.nn.Parameter(torch.zeros(1))
        self.normalize = normalize

    def _feat(self, h):
        return F.layer_norm(h, (h.shape[-1],)) if self.normalize else h

    def forward(self, h, digit_mask):
        h = self._feat(h)
        q, k = self.Wq(h), self.Wk(h)
        logits = q @ k.transpose(1, 2) + self.beta
        gate = torch.clamp(logits, 0.0, 1.0)
        T = h.shape[1]
        causal = torch.tril(torch.ones(T, T, device=h.device), diagonal=-1)
        pair = digit_mask.unsqueeze(2) & digit_mask.unsqueeze(1)
        y = (gate * causal.unsqueeze(0) * pair.float()).sum(dim=(1, 2))
        return self.c * y + self.b


@torch.no_grad()
def gate_fractions(head, h, dmask):
    """Measure how many gate scores can still receive gradients."""
    hf = head._feat(h)
    q, k = head.Wq(hf), head.Wk(hf)
    logits = q @ k.transpose(1, 2) + head.beta
    T = h.shape[1]
    causal = torch.tril(torch.ones(T, T, device=h.device), diagonal=-1)
    pair = (dmask.unsqueeze(2) & dmask.unsqueeze(1)).float()
    m = causal.unsqueeze(0) * pair
    tot = m.sum().clamp_min(1)
    active = (((logits > 0) & (logits < 1)).float() * m).sum() / tot
    dead_lo = ((logits <= 0).float() * m).sum() / tot
    dead_hi = ((logits >= 1).float() * m).sum() / tot
    return round(active.item(), 3), round(dead_lo.item(), 3), round(dead_hi.item(), 3)


@torch.no_grad()
def pred_stats(hz, head, ns, layer=0, bs=24):
    """Measure the spread of predictions at each length."""
    g = torch.Generator().manual_seed(999)
    out = {}
    for n in ns:
        lists = [make_list(n, g) for _ in range(bs)]
        ys = [inv_count(a) for a in lists]
        ids, mask, dmask, _, _ = hz.encode_batch(lists)
        h = hz.hidden_states(ids, mask, layer=layer)
        p = head(h, dmask).float().cpu()
        out[n] = [
            round(p.mean().item(), 2),
            round(p.std().item(), 2),
            round(sum(ys) / len(ys), 2),
        ]
    return out


@torch.no_grad()
def active_region_init(head, hz, layer, target_std=0.5, center=0.5, seed=12345):
    """Move the starting gate scores near the active range."""
    g = torch.Generator().manual_seed(seed)
    lists, _ = gen_training_batch(32, g)
    ids, mask, dmask, _, _ = hz.encode_batch(lists)
    h = hz.hidden_states(ids, mask, layer=layer)
    hf = head._feat(h)
    logits = head.Wq(hf) @ head.Wk(hf).transpose(1, 2)
    s = logits.std().clamp_min(1e-4)
    scale = (target_std / s).sqrt()
    head.Wq.weight.mul_(scale)
    head.Wq.bias.mul_(scale)
    head.Wk.weight.mul_(scale)
    head.Wk.bias.mul_(scale)
    mu = (
        head.Wq(hf) @ head.Wk(hf).transpose(1, 2)
    ).mean()  # mean of the rescaled logits
    head.beta.fill_(center - mu)  # shift into the active band


def train_tally(
    hz,
    head_seed,
    steps=1500,
    layer=0,
    lr=1e-3,
    bs=16,
    init_fix=False,
    log_every=500,
    target_std=0.5,
    normalize=False,
):
    torch.manual_seed(2000 + head_seed)  # seed the HEAD init (global RNG)
    head = TallyAdapterB(hz.model.config.hidden_size, normalize=normalize).to(DEV)
    if init_fix:
        active_region_init(head, hz, layer, target_std=target_std)
    g = torch.Generator().manual_seed(head_seed)  # seed the training data (separate)
    diag = {"head_seed": head_seed, "init_fix": init_fix, "loss": []}
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01
    )
    t0 = time.time()
    for step in range(steps):
        lists, ys = gen_training_batch(bs, g)
        ids, mask, dmask, _, _ = hz.encode_batch(lists)
        y = torch.tensor(ys, dtype=torch.float32, device=DEV)
        h = hz.hidden_states(ids, mask, layer=layer)
        if step == 0:
            diag["active_init"] = gate_fractions(head, h, dmask)
        pred = head(h, dmask)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0:
            diag["loss"].append([step, round(loss.item(), 2)])
    lists, _ = gen_training_batch(64, g)
    ids, mask, dmask, _, _ = hz.encode_batch(lists)
    h = hz.hidden_states(ids, mask, layer=layer)
    diag["active_end"] = gate_fractions(head, h, dmask)
    diag["beta"] = round(head.beta.item(), 4)
    diag["c"] = round(head.c.item(), 4)
    diag["b"] = round(head.b.item(), 4)
    diag["secs"] = round(time.time() - t0)
    return head, diag


def run_phase(
    hz, init_fix, fname_stem, layer=0, target_std=0.5, steps=1500, normalize=False
):
    rows = []
    for hs in range(N_SEEDS):
        head, diag = train_tally(
            hz,
            hs,
            init_fix=init_fix,
            layer=layer,
            target_std=target_std,
            steps=steps,
            normalize=normalize,
        )
        res = eval_head(hz, head, "tally", LENGTHS, hs, layer=layer)
        diag["pred_stats"] = pred_stats(hz, head, [8, 32, 96], layer=layer)
        eval_results = {}
        metric_names = ("exact", "mae", "medrel", "within10", "within5")
        for n in LENGTHS:
            length_results = {}
            for metric_name in metric_names:
                length_results[metric_name] = round(res[n][metric_name], 4)
            eval_results[str(n)] = length_results
        diag["eval"] = eval_results
        json.dump(
            {"arm": "tally", "seed": hs, "results": res},
            open(os.path.join(RES, f"{fname_stem}_s{hs}.json"), "w"),
            indent=2,
        )
        ok = res[SUCCESS_N]["within10"] >= SUCCESS_WITHIN10
        diag["success"] = ok
        rows.append(diag)
        print(
            f"  >> {fname_stem} hs{hs}: active init={diag['active_init'][0]:.2f} "
            f"end={diag['active_end'][0]:.2f} beta={diag['beta']} c={diag['c']} b={diag['b']} | "
            f"n8 MAE={res[8]['mae']:.1f} n96 MAE={res[96]['mae']:.1f} "
            f"within10={res[96]['within10']:.2f} -> {'SUCCESS' if ok else 'FAIL (dead gate)'}",
            flush=True,
        )
    return rows


def main():
    import argparse

    global N_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--tag", default="", help="filename tag, e.g. _4b (empty = 0.6B)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument(
        "--arms", nargs="+", default=["random", "init"], choices=["random", "init"]
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=0,
        help="hidden-state layer to tap (0=embeddings, -1=final, mid=deep)",
    )
    ap.add_argument(
        "--target_std",
        type=float,
        default=0.5,
        help="active-region init: target gate-logit std",
    )
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="LayerNorm the tapped hidden states (for deep taps)",
    )
    args = ap.parse_args()
    N_SEEDS = args.seeds
    layer_tag = "" if args.layer == 0 else f"_L{args.layer}"
    if args.normalize:
        layer_tag += "n"
    Q.MODEL = args.model
    Q.MODEL_TAG = args.tag
    Q.DTYPE = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    hz = QwenHarness()
    out = {"success_criterion": f"within10>={SUCCESS_WITHIN10} @ n={SUCCESS_N}"}

    if "random" in args.arms:
        print(f"\n=== random init, {N_SEEDS} seeds ({args.model}) ===", flush=True)
        out["random"] = run_phase(
            hz,
            init_fix=False,
            fname_stem=f"qwen{args.tag}{layer_tag}_tallyrand",
            layer=args.layer,
            target_std=args.target_std,
            steps=args.steps,
            normalize=args.normalize,
        )
    if "init" in args.arms:
        print(
            f"\n=== active-region init, {N_SEEDS} seeds ({args.model}) ===", flush=True
        )
        out["fixed"] = run_phase(
            hz,
            init_fix=True,
            fname_stem=f"qwen{args.tag}{layer_tag}_tallyinit",
            layer=args.layer,
            target_std=args.target_std,
            steps=args.steps,
            normalize=args.normalize,
        )

    json.dump(
        out,
        open(os.path.join(RES, f"tally_robustness{args.tag}{layer_tag}.json"), "w"),
        indent=2,
    )

    def summarize(rows, name):
        succ = sum(r["success"] for r in rows)
        maes = sorted(r["eval"]["96"]["mae"] for r in rows)
        median = (
            maes[(len(maes) - 1) // 2] + maes[len(maes) // 2]
        ) / 2  # true median (matches paper)
        print(
            f"{name}: success {succ}/{len(rows)} | n=96 MAE median {median:.1f} "
            f"best {maes[0]:.1f} worst {maes[-1]:.1f}",
            flush=True,
        )

    print(
        f"\n=== SUMMARY ({args.model}; success = {SUCCESS_WITHIN10:.0%} within-10% @ n=96) ===",
        flush=True,
    )
    if "random" in out:
        summarize(out["random"], "random init")
    if "fixed" in out:
        summarize(out["fixed"], "active-region init")


if __name__ == "__main__":
    main()
