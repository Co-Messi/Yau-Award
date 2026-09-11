"""Train simple probes on saved Qwen features."""

import json
import os
import time

import torch
import torch.nn.functional as F

from toy_extensivity import DEV, TallyHead, cn2, gen_batch

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def train_tally(seed, steps=12000, lr=3e-3, bs=64, binarize_after=6000, lam=0.05):
    torch.manual_seed(seed)
    m = TallyHead().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.003
    )
    t0 = time.time()
    for step in range(steps):
        n = int(torch.randint(8, 65, (1,)).item())
        a, y = gen_batch(bs, n)
        e = m.emb(a)
        q, k = m.Wq(e), m.Wk(e)
        logits = q @ k.transpose(1, 2)
        gate = torch.clamp(logits, 0.0, 1.0)
        T = a.shape[1]
        mask = torch.tril(torch.ones(T, T, device=a.device), diagonal=-1)
        pred = m.c * (gate * mask.unsqueeze(0)).sum(dim=(1, 2)) + m.b
        loss = F.mse_loss(pred, y)
        if step >= binarize_after:
            frac = (gate * (1 - gate) * mask.unsqueeze(0)).sum() / (mask.sum() * bs)
            loss = loss + lam * frac * cn2(n)  # scale with pair count
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(
                f"  s{seed} step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return m


@torch.no_grad()
def ev(m, lengths=(16, 32, 64, 128, 256, 512), n_eval=512, bs=64):
    out = {}
    for n in lengths:
        stats = {}
        for mode in ("soft", "hard"):
            preds, ys = [], []
            done = 0
            torch.manual_seed(999)
            while done < n_eval:
                b = min(bs, n_eval - done)
                a, y = gen_batch(b, n)
                p = m(a, hard=(mode == "hard"))
                preds.append(p.float().cpu())
                ys.append(y.float().cpu())
                done += b
            p, y = torch.cat(preds), torch.cat(ys)
            stats[mode] = {
                "exact": (p.round() == y).float().mean().item(),
                "mae": (p - y).abs().mean().item(),
            }
        out[n] = stats
        print(
            f"  n={n}: soft exact {stats['soft']['exact']:.3f} mae {stats['soft']['mae']:.2f} | "
            f"hard exact {stats['hard']['exact']:.3f} mae {stats['hard']['mae']:.2f}",
            flush=True,
        )
    return out


if __name__ == "__main__":
    allres = {}
    for seed in range(3):
        print(f"=== probe tally s{seed} ===", flush=True)
        m = train_tally(seed)
        allres[seed] = ev(m)
        with torch.no_grad():
            print(f"  c={m.c.item():.4f} b={m.b.item():.4f}")
    json.dump(allres, open(os.path.join(RESULTS, "probe_tally.json"), "w"), indent=2)
