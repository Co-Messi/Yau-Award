"""Run the six counting tasks on small models."""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from howmany_tasks import KIND, sample, witness_count, family_scale
from toy_extensivity import Block

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
V = 64
QTYPES = list(KIND.keys())
N_TYPES = len(QTYPES)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "howmany_v1")


TOY_PARAMS = {
    "inv": [None],
    "drop5": [None],
    "dup": [None],
    "sum_s": [20, 30, 40, 50, 60, 70, 80],
    "count_v": list(range(0, 64, 4)),
    "gt_c": [8, 16, 24, 32, 40, 48],
}
TOY_HELD_OUT = {
    "inv": [],
    "drop5": [],
    "dup": [],
    "sum_s": [50, 80],
    "count_v": [24, 44, 60],
    "gt_c": [40],
}


def toy_questions(held=False):
    if held:
        source = TOY_HELD_OUT
    else:
        source = {}
        for question_type in QTYPES:
            source[question_type] = []
            for parameter in TOY_PARAMS[question_type]:
                if parameter not in TOY_HELD_OUT[question_type]:
                    source[question_type].append(parameter)

    questions = []
    for question_type in QTYPES:
        for parameter in source[question_type]:
            questions.append((question_type, parameter))
    return questions


class GTallyHead(nn.Module):
    def __init__(self, V=V, d=32, rank=16, n_types=N_TYPES, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.emb = nn.Embedding(V + n_types + 1, d)  # items + type tokens + null-arg
        self.null_arg = V + n_types
        self.A = nn.Linear(d + 2, rank)
        self.B = nn.Linear(d + 2, rank)
        self.G = nn.Linear(2 * (d + 2), rank)  # trilinear question leg
        self.P = nn.Linear(d + 2, rank)
        self.Q = nn.Linear(2 * (d + 2), rank)  # item-channel question side
        nn.init.zeros_(self.G.weight)
        nn.init.zeros_(self.G.bias)  # g starts at 1

        self.beta1 = nn.Parameter(torch.zeros(n_types))
        self.beta2 = nn.Parameter(torch.zeros(n_types))
        self.c1 = nn.Parameter(torch.ones(n_types))
        self.c2 = nn.Parameter(torch.ones(n_types))
        self.b = nn.Parameter(torch.zeros(n_types))

        self.register_buffer(
            "is_pair", torch.tensor([KIND[t] == "pair" for t in QTYPES])
        )

        pair_types = [i for i, t in enumerate(QTYPES) if KIND[t] == "pair"]
        block = torch.zeros(n_types, rank)
        width = rank // max(len(pair_types), 1)
        for k, ti in enumerate(pair_types):
            block[ti, k * width : (k + 1) * width] = 1.0
        for ti in range(n_types):
            if ti not in pair_types:
                block[ti] = 1.0
        self.register_buffer("block1", block)

    def f(self, e):
        return torch.cat(
            [e, (e * e).mean(-1, keepdim=True), torch.ones_like(e[..., :1])], dim=-1
        )

    def qfeat(self, tvec, avec):
        return torch.cat(
            [self.f(self.emb(tvec)), self.f(self.emb(avec))], dim=-1
        )  # [B, 2(d+2)]

    def forward(self, a, tidx, arg):
        B_, n = a.shape
        fe = self.f(self.emb(a))  # [B,n,d+2]
        u = self.qfeat(
            V + tidx, torch.where(arg >= 0, arg, torch.full_like(arg, self.null_arg))
        )
        q, k = self.A(fe), self.B(fe)  # [B,n,r]
        g = self.block1[tidx] * (1.0 + self.G(u))  # [B,r]
        logits = torch.einsum("bir,br,bjr->bij", q, g, k) + self.beta1[tidx].view(
            -1, 1, 1
        )
        gate = torch.clamp(logits, 0.0, 1.0)
        causal = torch.tril(torch.ones(n, n, device=a.device), diagonal=-1)
        y_pair = (gate * causal.unsqueeze(0)).sum(dim=(1, 2))
        item_logit = (self.P(fe) * self.Q(u).unsqueeze(1)).sum(-1) + self.beta2[
            tidx
        ].view(-1, 1)
        y_item = torch.clamp(item_logit, 0.0, 1.0).sum(dim=1)
        if self.normalize:  # normalization test
            y_pair = y_pair / max(n * (n - 1) / 2.0, 1.0)
            y_item = y_item / n
        routed = torch.where(
            self.is_pair[tidx], self.c1[tidx] * y_pair, self.c2[tidx] * y_item
        )
        return routed + self.b[tidx]


@torch.no_grad()
def active_region_init(head, questions, seed=12345, target_std=0.5, center=0.5):
    """Move the starting gate scores into the useful range."""

    def stats(qs, s_off):
        a, tidx, arg, _, _ = make_gen(qs, seed + s_off)(64)
        fe = head.f(head.emb(a))
        u = head.qfeat(
            V + tidx, torch.where(arg >= 0, arg, torch.full_like(arg, head.null_arg))
        )
        g = head.block1[tidx] * (1.0 + head.G(u))  # same masking as forward
        lg = torch.einsum("bir,br,bjr->bij", head.A(fe), g, head.B(fe))
        il = (head.P(fe) * head.Q(u).unsqueeze(1)).sum(-1)
        return lg, il

    lg, il = stats(questions, 0)
    sc = (target_std / lg.std().clamp_min(1e-4)).sqrt()
    head.A.weight.mul_(sc)
    head.A.bias.mul_(sc)
    head.B.weight.mul_(sc)
    head.B.bias.mul_(sc)
    head.P.weight.mul_((target_std / il.std().clamp_min(1e-4)).item())
    by_type = {}
    for t, p in questions:
        by_type.setdefault(t, []).append((t, p))
    for t, qs in by_type.items():
        lg, il = stats(qs, 1 + QTYPES.index(t))
        head.beta1[QTYPES.index(t)] = center - lg.mean().item()
        head.beta2[QTYPES.index(t)] = center - il.mean().item()


class QSoftmaxCounter(nn.Module):
    """Small Transformer with two question tokens at the start."""

    def __init__(self, d=128, h=4, L=4, n_types=N_TYPES):
        super().__init__()
        self.emb = nn.Embedding(V + n_types + 1, d)
        self.null_arg = V + n_types
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, a, tidx, arg):
        argt = torch.where(arg >= 0, arg, torch.full_like(arg, self.null_arg))
        seq = torch.cat([(V + tidx).unsqueeze(1), argt.unsqueeze(1), a], dim=1)
        x = self.emb(seq)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x[:, -1]))[:, 0]


class MeanMLP(nn.Module):
    def __init__(self, d=128, n_types=N_TYPES):
        super().__init__()
        self.emb = nn.Embedding(V + n_types + 1, d)
        self.null_arg = V + n_types
        self.mlp = nn.Sequential(nn.Linear(3 * d, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, a, tidx, arg):
        argt = torch.where(arg >= 0, arg, torch.full_like(arg, self.null_arg))
        z = torch.cat([self.emb(a).mean(1), self.emb(V + tidx), self.emb(argt)], dim=-1)
        return self.mlp(z)[:, 0]


# Data / train / eval


def make_gen(questions, seed, n_lo=8, n_hi=64, iid_frac=0.3, device=DEV):
    """Sample each question type equally often."""
    g = torch.Generator().manual_seed(seed)
    by_type = {}
    for t, p in questions:
        by_type.setdefault(t, []).append(p)
    types = list(by_type.keys())

    def gen(bs):
        t = types[int(torch.randint(0, len(types), (1,), generator=g).item())]
        ps = by_type[t]
        p = ps[int(torch.randint(0, len(ps), (1,), generator=g).item())]
        n = int(torch.randint(n_lo, n_hi + 1, (1,), generator=g).item())
        a, y = sample(t, bs, n, V, g, p, iid_frac=iid_frac)
        tidx = torch.full((bs,), QTYPES.index(t), dtype=torch.long)
        arg = torch.full((bs,), -1 if p is None else p, dtype=torch.long)
        return a.to(device), tidx.to(device), arg.to(device), y.to(device), (t, p, n)

    return gen


def toy_scales(seed=7):
    g = torch.Generator().manual_seed(seed)
    return {
        t: family_scale(
            t, V, g, [p for p in TOY_PARAMS[t] if p not in TOY_HELD_OUT[t]][0]
        )
        for t in QTYPES
    }


def normalizer(t, n):
    return n * (n - 1) / 2.0 if KIND[t] == "pair" else float(n)


def fixed_n_metrics(p, y):
    p, y = (
        torch.as_tensor(p, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
    )
    ss_res = ((p - y) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    rel = (p - y).abs() / y.clamp(min=1.0)
    nrel = (y - y.mean()).abs() / y.clamp(min=1.0)
    return {
        "mae": (p - y).abs().mean().item(),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-9),
        "within10": (rel <= 0.10).float().mean().item(),
        "exact": (p.round() == y).float().mean().item(),
        "null_mae": (y - y.mean()).abs().mean().item(),
        "null_within10": (nrel <= 0.10).float().mean().item(),
    }


def train_model(model, arm, gen, scales, steps, lr, fraction=False):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01
    )
    t0 = time.time()
    for step in range(steps):
        a, tidx, arg, y, (t, p, n) = gen(64)
        pred = model(a, tidx, arg)

        if arm.startswith("gtally"):
            loss = F.mse_loss(pred / scales[t], y / scales[t])
        elif fraction:
            loss = F.mse_loss(pred, y / normalizer(t, n))
        else:
            loss = F.mse_loss(pred, y / scales[t])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in trainable:  # per-tensor clip (see qwen_howmany train_head)
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(
                f"    step {step} ({t}) loss {loss.item():.5f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return model


@torch.no_grad()
def evaluate(
    model,
    arm,
    questions,
    lengths,
    seed,
    scales,
    fraction=False,
    n_eval=512,
    bs=64,
    qswap=False,
):
    out = {}
    for t, p in questions:
        tname = f"{t}" + (f"_{p}" if p is not None else "")
        out[tname] = {}
        for n in lengths:
            g = torch.Generator().manual_seed(50_000 + seed + n)
            preds, ys = [], []
            for _ in range(n_eval // bs):
                a, y = sample(t, bs, n, V, g, p)
                tidx = torch.full((bs,), QTYPES.index(t), dtype=torch.long, device=DEV)
                arg = torch.full(
                    (bs,), -1 if p is None else p, dtype=torch.long, device=DEV
                )
                if qswap:  # use a different question type for this check
                    wt = (QTYPES.index(t) + 1) % len(QTYPES)
                    tidx = torch.full_like(tidx, wt)
                raw = model(a.to(DEV), tidx, arg)
                pred = (
                    raw * normalizer(t, n)
                    if fraction
                    else raw * scales[t]
                    if arm == "softmax" or arm == "mlp"
                    else raw
                )
                preds += pred.float().cpu().tolist()
                ys += y.tolist()
            out[tname][n] = fixed_n_metrics(preds, ys)
    return out


# Runners


def run_config(arm, questions, seed, scales, steps, lengths, mixture_tag="", rank=16):
    tag = mixture_tag or "_".join(
        f"{t}{'' if p is None else p}" for t, p in questions[:1]
    )
    path = os.path.join(RES, f"toy_{arm}_{tag}_s{seed}.json")
    if os.path.exists(path):
        print(f"skip {arm} {tag} s{seed}")
        return
    print(f"=== {arm} {tag} s{seed} ===", flush=True)
    torch.manual_seed(3000 + seed)
    gen = make_gen(questions, seed)
    fraction = arm == "fraction"
    if arm == "witness":
        res = {}
        for t, p in questions:
            tname = f"{t}" + (f"_{p}" if p is not None else "")
            res[tname] = {}
            for n in lengths:
                g = torch.Generator().manual_seed(50_000 + n)
                a, y = sample(t, 256, n, V, g, p)
                yh = witness_count(t, a, p)
                res[tname][n] = fixed_n_metrics(yh.tolist(), y.tolist())
    else:
        if arm.startswith("gtally"):
            model = GTallyHead(rank=rank, normalize=arm == "gtally-norm").to(DEV)
            active_region_init(model, questions)
            lr = 3e-3
        elif arm in ("softmax", "fraction"):
            model = QSoftmaxCounter().to(DEV)
            lr = 3e-4
        else:
            model = MeanMLP().to(DEV)
            lr = 3e-4
        model = train_model(model, arm, gen, scales, steps, lr, fraction)
        model.eval()
        res = {
            "trained": evaluate(model, arm, questions, lengths, seed, scales, fraction)
        }
        if mixture_tag and arm == "gtally":
            res["heldout"] = evaluate(
                model, arm, toy_questions(held=True), lengths, seed, scales, fraction
            )
            res["qswap"] = evaluate(
                model,
                arm,
                questions[:6],
                lengths[:2],
                seed,
                scales,
                fraction,
                qswap=True,
            )
    json.dump(
        {"arm": arm, "tag": tag, "seed": seed, "results": res},
        open(path, "w"),
        indent=2,
    )
    summ = res["trained"] if "trained" in res else res
    for tname, cells in list(summ.items())[:8]:
        last = max(cells)
        first = min(cells)
        print(
            f"  {tname}: n={first} mae {cells[first]['mae']:.1f} r2 {cells[first]['r2']:.2f}"
            f" | n={last} mae {cells[last]['mae']:.1f} r2 {cells[last]['r2']:.2f}"
            f" (null {cells[last]['null_mae']:.0f})",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["e2", "e3"], required=True)
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["gtally", "witness", "gtally-norm", "softmax", "fraction", "mlp"],
    )
    ap.add_argument("--predicates", nargs="+", default=QTYPES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lengths", nargs="+", type=int, default=[16, 64, 128, 256, 512])
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    scales = toy_scales()
    print("scales:", {k: round(v) for k, v in scales.items()}, flush=True)

    if args.mode == "e2":  # per-task matrix
        seeds_for = {
            "gtally": args.seeds,
            "witness": 1,
            "gtally-norm": 2,
            "softmax": 2,
            "fraction": 2,
            "mlp": 2,
        }
        for t in args.predicates:
            p = [q for q in TOY_PARAMS[t] if q not in TOY_HELD_OUT[t]][0]
            for arm in args.arms:
                for s in range(seeds_for.get(arm, 1)):
                    run_config(arm, [(t, p)], s, scales, args.steps, args.lengths)
    else:  # e3: one question-conditioned head over the full mixture
        qs = toy_questions()
        for s in range(args.seeds):
            run_config(
                "gtally",
                qs,
                s,
                scales,
                args.steps * 2,
                args.lengths,
                mixture_tag="mixture",
                rank=32,
            )
        for s in range(2):
            run_config(
                "softmax",
                qs,
                s,
                scales,
                args.steps * 2,
                args.lengths,
                mixture_tag="mixture",
            )


if __name__ == "__main__":
    main()
