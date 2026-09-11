"""Compare counting methods on a small Transformer."""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
V = 64  # alphabet size (integer tokens 0..V-1)
SCALE = 500.0  # fixed scale used during training

# Data


def gen_batch(bs, n, device=DEV, for_twin=False, generator=None, k_swaps=None):
    """Make random integer lists and their inversion counts."""
    a = torch.randint(0, V, (bs, n), device=device, generator=generator)
    if for_twin:
        a, _ = torch.sort(a, dim=1)

        for _ in range(8):
            const = (a[:, 1:] == a[:, :-1]).all(dim=1)
            if not const.any():
                break
            a[const] = torch.sort(
                torch.randint(
                    0, V, (int(const.sum()), n), device=device, generator=generator
                ),
                dim=1,
            )[0]
        flip = torch.rand(bs, device=device, generator=generator) < 0.5
        rows = torch.arange(bs, device=device)
        kmax = int(k_swaps) if k_swaps else 8
        for r in range(kmax):
            if k_swaps is None:
                kr = torch.randint(
                    1, kmax + 1, (bs,), device=device, generator=generator
                )
                active = flip & (torch.tensor(r, device=device) < kr)
            else:
                active = flip
            incr = (a[:, :-1] < a[:, 1:]).float()  # current ascents
            pos = torch.multinomial(incr + 1e-9, 1, generator=generator)[:, 0]
            left = a[rows, pos].clone()
            right = a[rows, pos + 1].clone()
            do = active & (left < right)
            a[rows[do], pos[do]] = right[do]
            a[rows[do], pos[do] + 1] = left[do]

        gt = (a.unsqueeze(2) > a.unsqueeze(1)).float()
        mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)
        y = ((gt * mask.unsqueeze(0)).sum(dim=(1, 2)) > 0).float()
        return a, y

    gt = (a.unsqueeze(2) > a.unsqueeze(1)).float()  # [bs, i, j]: a_i > a_j
    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)  # i < j
    y = (gt * mask.unsqueeze(0)).sum(dim=(1, 2))  # pairs i<j with a_i > a_j
    return a, y


# Attention variants


def sparsemax(z, dim=-1):
    """Project scores with sparsemax."""
    zs, _ = torch.sort(z, descending=True, dim=dim)
    rng = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = -1
    rng = rng.view(shape)
    cs = zs.cumsum(dim) - 1
    cond = zs - cs / rng > 0
    k = cond.to(z.dtype).mul(rng).max(dim=dim, keepdim=True).values
    tau = cs.gather(dim, (k.long() - 1).clamp(min=0)) / k
    return torch.clamp(z - tau, min=0)


def rope_rotate(x, base=10000.0):
    """Apply rotary position embeddings."""
    B, H, T, Dh = x.shape
    half = Dh // 2
    freqs = base ** (
        -torch.arange(0, half, device=x.device, dtype=torch.float32) / half
    )
    t = torch.arange(T, device=x.device, dtype=torch.float32)
    ang = torch.outer(t, freqs)  # [T, half]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, d, h, attn_kind="softmax"):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.attn_kind = attn_kind

    def forward(self, x):
        B, T, d = x.shape
        z = self.ln1(x)
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = rope_rotate(q), rope_rotate(k)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if self.attn_kind == "logn":
            logits = logits * math.log(max(T, 2))
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        logits = logits + mask
        if self.attn_kind == "entmax":
            w = sparsemax(
                logits.clamp(min=-1e9), dim=-1
            )  # finite mask value: avoids inf-inf NaNs
        else:
            w = logits.softmax(dim=-1)
        o = (w @ v).transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(o)
        x = x + self.mlp(self.ln2(x))
        return x


class SoftmaxCounter(nn.Module):
    """Small Transformer used as the baseline."""

    def __init__(self, d=128, h=4, L=4, attn_kind="softmax", twin=False):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, h, attn_kind) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.twin = twin

    def forward(self, a):
        x = self.emb(a)
        for b in self.blocks:
            x = b(x)
        out = self.head(self.lnf(x[:, -1]))[:, 0]
        return out  # twin: logit; count: scaled prediction


class TallyHead(nn.Module):
    """Learnable head that adds the scores for every pair."""

    def __init__(self, d=32, rank=4, fixed=False, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.emb = nn.Embedding(V, d)
        self.Wq = nn.Linear(d, rank, bias=True)
        self.Wk = nn.Linear(d, rank, bias=True)
        self.c = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        if fixed:
            with torch.no_grad():
                self.emb.weight.zero_()
                self.emb.weight[:, 0] = torch.arange(V, dtype=torch.float32)
                self.emb.weight[:, 1] = 1.0
                self.Wq.weight.zero_()
                self.Wq.bias.zero_()
                self.Wk.weight.zero_()
                self.Wk.bias.zero_()

                self.Wq.weight[0, 1] = 1.0
                self.Wq.weight[1, 0] = -1.0

                self.Wk.weight[0, 0] = 1.0
                self.Wk.weight[1, 1] = 1.0
                self.c.fill_(1.0)
                self.b.fill_(0.0)
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, a, hard=False):
        e = self.emb(a)  # [B, T, d]
        q, k = self.Wq(e), self.Wk(e)  # [B, T, r]
        logits = q @ k.transpose(1, 2)  # [B, i, j] = <q_i, k_j>
        gate = (logits >= 0.5).float() if hard else torch.clamp(logits, 0.0, 1.0)
        T = a.shape[1]
        mask = torch.tril(torch.ones(T, T, device=a.device), diagonal=-1)  # j < i
        y = (gate * mask.unsqueeze(0)).sum(dim=(1, 2))
        if self.normalize:
            y = y / mask.sum().clamp(
                min=1.0
            )  # mean over causal pairs, so it is bounded
        return self.c * y + self.b


class MLPControl(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.mlp = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, a):
        return self.mlp(self.emb(a).mean(dim=1))[:, 0]


# Train / eval


def cn2(n):
    return n * (n - 1) / 2.0


def train_model(model, kind, steps, seed, twin=False, fraction=False, lr=3e-4, bs=64):
    torch.manual_seed(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or steps == 0:
        return model  # fixed construction: no training
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01
    )
    model.train()
    t0 = time.time()
    for step in range(steps):
        n = int(torch.randint(8, 65, (1,)).item())
        if twin:
            a, y = gen_batch(bs, n, for_twin=True)
            loss = F.binary_cross_entropy_with_logits(model(a), y)
        else:
            a, y = gen_batch(bs, n)
            if kind.startswith("tally"):
                loss = F.mse_loss(model(a), y)  # tally predicts raw count
            else:
                target = (y / cn2(n)) if fraction else (y / SCALE)
                loss = F.mse_loss(model(a), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(
                f"  [{kind} s{seed}] step {step} loss {loss.item():.5f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return model


@torch.no_grad()
def evaluate(model, kind, lengths, twin=False, fraction=False, n_eval=512, bs=64):
    model.eval()
    out = {}
    for n in lengths:
        if twin:
            out[n] = {}
            for k in (1, 2, 4, 8):
                correct = 0
                done = 0
                while done < n_eval:
                    b = min(bs, n_eval - done)
                    a, y = gen_batch(b, n, for_twin=True, k_swaps=k)
                    p = (model(a) > 0).float()
                    correct += (p.cpu() == y.cpu()).sum().item()
                    done += b
                out[n][f"acc_k{k}"] = correct / n_eval
            continue
        preds, ys = [], []
        done = 0
        while done < n_eval:
            b = min(bs, n_eval - done)
            a, y = gen_batch(b, n)
            raw = model(a)
            if kind.startswith("tally"):
                p = raw
            elif fraction:
                p = raw * cn2(n)  # n-dependent UN-scaling outside the model
            else:
                p = raw * SCALE
            preds.append(p.float().cpu())
            ys.append(y.float().cpu())
            done += b
        p = torch.cat(preds)
        y = torch.cat(ys)
        ss_res = ((p - y) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        out[n] = {
            "r2": 1.0 - ss_res / max(ss_tot, 1e-9),
            "mae": (p - y).abs().mean().item(),
            "exact": (p.round() == y).float().mean().item(),
        }
    return out


CONFIGS = {
    "vanilla": dict(cls="soft", attn="softmax", L=4, d=128),
    "entmax": dict(cls="soft", attn="entmax", L=4, d=128),
    "logn": dict(cls="soft", attn="logn", L=4, d=128),
    "deep": dict(cls="soft", attn="softmax", L=6, d=192),
    "fraction": dict(cls="soft", attn="softmax", L=4, d=128, fraction=True),
    "tally-fixed": dict(cls="tally", fixed=True),
    "tally": dict(cls="tally", fixed=False),
    "tally-norm": dict(cls="tally", fixed=False, normalize=True),
    "mlp": dict(cls="mlp"),
    "twin": dict(cls="soft", attn="softmax", L=4, d=128, twin=True),
}


def build(cfg):
    if cfg["cls"] == "soft":
        return SoftmaxCounter(
            d=cfg.get("d", 128),
            L=cfg.get("L", 4),
            attn_kind=cfg.get("attn", "softmax"),
            twin=cfg.get("twin", False),
        ).to(DEV)
    if cfg["cls"] == "tally":
        return TallyHead(
            fixed=cfg.get("fixed", False), normalize=cfg.get("normalize", False)
        ).to(DEV)
    return MLPControl().to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument(
        "--lengths", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512]
    )
    args = ap.parse_args()

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    for name in args.configs:
        cfg = CONFIGS[name]
        for seed in range(args.seeds):
            tag = f"{name}_s{seed}"
            path = os.path.join(os.path.dirname(__file__), "results", f"toy_{tag}.json")
            if os.path.exists(path):
                print(f"skip {tag} (exists)")
                continue
            print(f"=== {tag} ===", flush=True)
            torch.manual_seed(seed)
            model = build(cfg)
            steps = 0 if cfg.get("fixed") else args.steps
            model = train_model(
                model,
                name,
                steps,
                seed,
                twin=cfg.get("twin", False),
                fraction=cfg.get("fraction", False),
            )
            res = evaluate(
                model,
                name,
                args.lengths,
                twin=cfg.get("twin", False),
                fraction=cfg.get("fraction", False),
            )
            json.dump(
                {"config": name, "seed": seed, "results": res},
                open(path, "w"),
                indent=2,
            )
            print(json.dumps(res, indent=2), flush=True)
            if cfg.get("fixed"):
                break  # deterministic; one "seed" enough


if __name__ == "__main__":
    main()
