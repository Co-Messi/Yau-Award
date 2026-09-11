"""Compare sum and mean aggregation on frozen Qwen features."""

import argparse
import json
import os

import torch
import torch.nn.functional as F

import qwen_extensivity as Q
from qwen_extensivity import QwenHarness, gen_training_batch, make_list, inv_count, DEV

RES = Q.RESULTS
LENGTHS = [8, 16, 24, 32, 48, 64, 96]


class MeanPoolHead(torch.nn.Module):
    """Masked mean over digit-position hidden states, then a linear readout to a scalar."""

    def __init__(self, d_model, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.lin = torch.nn.Linear(d_model, 1)

    def forward(self, h, digit_mask):
        if self.normalize:
            h = F.layer_norm(
                h, (h.shape[-1],)
            )  # identical feature normalization to the tally arm
        m = digit_mask.float().unsqueeze(-1)  # [B,T,1]
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(
            1.0
        )  # masked mean over the SAME digit positions
        return self.lin(pooled)[:, 0]


def closeness(preds, ys):
    import statistics

    rel = [abs(p - y) / max(y, 1.0) for p, y in zip(preds, ys)]
    return {
        "medrel": statistics.median(rel),
        "within10": sum(r <= 0.10 for r in rel) / len(rel),
    }


def train_meanpool(hz, seed, layer, normalize, steps=1500, lr=1e-3, bs=16):
    torch.manual_seed(3000 + seed)
    head = MeanPoolHead(hz.model.config.hidden_size, normalize=normalize).to(DEV)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01
    )
    for _step in range(steps):
        lists, ys = gen_training_batch(bs, g)
        ids, mask, dmask, _, _ = hz.encode_batch(lists)
        y = torch.tensor(ys, dtype=torch.float32, device=DEV)
        h = hz.hidden_states(ids, mask, layer=layer)
        loss = F.mse_loss(head(h, dmask), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return head


@torch.no_grad()
def eval_meanpool(hz, head, layer, seed, n_eval=128, bs=16):
    g = torch.Generator().manual_seed(10_000 + seed)
    res = {}
    for n in LENGTHS:
        preds, ys = [], []
        for _ in range(n_eval // bs):
            lists = [make_list(n, g) for _ in range(bs)]
            ys += [inv_count(a) for a in lists]
            ids, mask, dmask, _, _ = hz.encode_batch(lists)
            h = hz.hidden_states(ids, mask, layer=layer)
            preds += head(h, dmask).float().cpu().tolist()
        p = torch.tensor(preds)
        y = torch.tensor(ys, dtype=torch.float32)
        res[n] = {
            "mae": (p - y).abs().mean().item(),
            **closeness(p.tolist(), y.tolist()),
        }
        print(
            f"  [meanpool s{seed}] n={n}: mae {res[n]['mae']:.1f} within10 {res[n]['within10'] * 100:.0f}%",
            flush=True,
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    Q.MODEL = args.model
    Q.MODEL_TAG = args.tag
    Q.DTYPE = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    os.makedirs(RES, exist_ok=True)
    hz = QwenHarness()
    ltag = f"_L{args.layer}" + ("n" if args.normalize else "")

    rows = []
    for s in range(args.seeds):
        head = train_meanpool(hz, s, args.layer, args.normalize, steps=args.steps)
        res = eval_meanpool(hz, head, args.layer, s)
        json.dump(
            {"arm": "meanpool", "seed": s, "layer": args.layer, "results": res},
            open(os.path.join(RES, f"qwen{args.tag}{ltag}_meanpool_s{s}.json"), "w"),
            indent=2,
        )
        rows.append(res)
        print(
            f"  >> meanpool s{s}: n8 MAE={res[8]['mae']:.1f} n96 MAE={res[96]['mae']:.1f} "
            f"within10@96={res[96]['within10']:.2f}  (normalization test)",
            flush=True,
        )

    maes = sorted(r[96]["mae"] for r in rows)
    med = (maes[(len(maes) - 1) // 2] + maes[len(maes) // 2]) / 2
    print(
        f"\n=== MEAN-POOL CONTROL (layer {args.layer}, normalize={args.normalize}): "
        f"n96 MAE median {med:.1f}  (vs tally sum on identical features) ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
