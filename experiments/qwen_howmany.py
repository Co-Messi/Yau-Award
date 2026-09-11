"""Train different counting heads on frozen Qwen features."""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from howmany_tasks import (
    KIND,
    PARAMS,
    HELD_OUT,
    QUESTION_TEXT,
    sample,
    family_scale,
)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = "Qwen/Qwen3-0.6B-Base"
V = 10
QTYPES = list(KIND.keys())  # ['inv','drop5','dup','sum_s','count_v','gt_c']
N_TYPES = len(QTYPES)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "howmany_v1")


def train_questions():
    """List the questions used for training."""
    out = []
    for t in QTYPES:
        for p in PARAMS[t]:
            if p not in HELD_OUT[t]:
                out.append((t, p))
    return out


def heldout_questions():
    return [(t, p) for t in QTYPES for p in HELD_OUT[t]]


def question_text(t, p, para=0):
    return QUESTION_TEXT[t][para].format(p=p)


class QHarness:
    def __init__(self, model=MODEL, dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(DEV)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.d_model = self.model.config.hidden_size

        self.digit_seg = {}
        for d in range(10):
            ids = self.tok(f" {d}", add_special_tokens=False)["input_ids"]
            assert len(ids) == 2, f"' {d}' is {len(ids)} tokens"
            self.digit_seg[d] = ids
        self.pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        self._prefix_cache = {}
        print(f"loaded {model} ({dtype}); hidden={self.d_model}", flush=True)

    def _prefix(self, qtext):
        if qtext not in self._prefix_cache:
            self._prefix_cache[qtext] = self.tok(
                f"Question: {qtext}?\nList:", add_special_tokens=False
            )["input_ids"]
        return self._prefix_cache[qtext]

    def encode(self, lists, qtexts):
        """Tokenize the question, list items, and answer prompt separately."""
        suffix = self.tok("\nAnswer:", add_special_tokens=False)["input_ids"]
        rows, masks, qends = [], [], []
        for a, qt in zip(lists, qtexts):
            pre = self._prefix(qt)
            qends.append(len(pre) - 1)
            ids = list(pre)
            item_pos = []
            for d in a:
                seg = self.digit_seg[int(d)]
                ids.extend(seg)
                item_pos.append(len(ids) - 1)  # the digit token of this item
            ids.extend(suffix)
            m = [False] * len(ids)
            for p in item_pos:
                m[p] = True
            rows.append(ids)
            masks.append(m)
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), self.pad_id, dtype=torch.long)
        attn = torch.zeros(len(rows), L, dtype=torch.long)
        imask = torch.zeros(len(rows), L, dtype=torch.bool)
        for b, (r, m) in enumerate(zip(rows, masks)):
            ids[b, : len(r)] = torch.tensor(r)
            attn[b, : len(r)] = 1
            imask[b, : len(m)] = torch.tensor(m)
        last = attn.sum(1) - 1
        return (
            ids.to(DEV),
            attn.to(DEV),
            imask.to(DEV),
            last.to(DEV),
            torch.tensor(qends, device=DEV),
        )

    @torch.no_grad()
    def features(self, ids, attn, layers):
        out = self.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        return {L: out.hidden_states[L].float() for L in layers}


# The head


class GTallyAdapter(torch.nn.Module):
    """Head with separate pair and item counting channels."""

    def __init__(self, d_model, rank=16, n_types=N_TYPES, quad=2, rank_item=4):
        super().__init__()
        self.quad = quad
        d_aug = d_model + quad + 1
        self.uq = (
            torch.nn.Parameter(torch.randn(quad, d_model) * 0.02) if quad else None
        )
        self.Wq = torch.nn.Linear(d_aug, rank)
        self.Wk = torch.nn.Linear(d_aug, rank)
        self.wu = torch.nn.Linear(d_aug, rank_item)

        self.Gq = torch.nn.Linear(d_model, rank)
        self.Gu = torch.nn.Linear(d_model, rank_item)
        torch.nn.init.zeros_(self.Gq.weight)
        torch.nn.init.zeros_(self.Gq.bias)
        torch.nn.init.zeros_(self.Gu.weight)
        torch.nn.init.zeros_(self.Gu.bias)

        pair_types = [i for i, t in enumerate(QTYPES) if KIND[t] == "pair"]
        block = torch.zeros(n_types, rank)
        width = rank // max(len(pair_types), 1)
        for k, ti in enumerate(pair_types):
            block[ti, k * width : (k + 1) * width] = 1.0
        for ti in range(n_types):
            if ti not in pair_types:
                block[ti] = 1.0  # item types never use this channel
        self.register_buffer("block1", block)

        self.beta1 = torch.nn.Parameter(torch.zeros(n_types))
        self.beta2 = torch.nn.Parameter(torch.zeros(n_types))
        self.c1 = torch.nn.Parameter(torch.ones(n_types))
        self.c2 = torch.nn.Parameter(torch.ones(n_types))
        self.b = torch.nn.Parameter(torch.zeros(n_types))

        self.register_buffer(
            "is_pair", torch.tensor([KIND[t] == "pair" for t in QTYPES])
        )

    def _feat(self, h):
        z = F.layer_norm(h, (h.shape[-1],))
        if not self.quad:
            return torch.cat([z, torch.ones_like(z[..., :1])], dim=-1)
        qs = (z @ self.uq.T) ** 2
        return torch.cat([z, qs, torch.ones_like(z[..., :1])], dim=-1)

    def channels(self, h, imask, tidx, qh):
        hf = self._feat(h)
        zq = F.layer_norm(qh, (qh.shape[-1],))
        g1 = self.block1[tidx] * (1.0 + self.Gq(zq))  # [B, rank]
        g2 = 1.0 + self.Gu(zq)  # [B, rank_item]
        q, k = self.Wq(hf), self.Wk(hf)
        logits = (q * g1.unsqueeze(1)) @ k.transpose(1, 2) + self.beta1[tidx].view(
            -1, 1, 1
        )
        gate = torch.clamp(logits, 0.0, 1.0)
        T = h.shape[1]
        causal = torch.tril(torch.ones(T, T, device=h.device), diagonal=-1)
        pair = (imask.unsqueeze(2) & imask.unsqueeze(1)).float() * causal.unsqueeze(0)
        y_pair = (gate * pair).sum(dim=(1, 2))
        item_logit = (self.wu(hf) * g2.unsqueeze(1)).sum(-1) + self.beta2[tidx].view(
            -1, 1
        )
        y_item = (torch.clamp(item_logit, 0.0, 1.0) * imask.float()).sum(dim=1)
        return y_pair, y_item

    def forward(self, h, imask, tidx, qh):
        y_pair, y_item = self.channels(h, imask, tidx, qh)
        routed = torch.where(
            self.is_pair[tidx], self.c1[tidx] * y_pair, self.c2[tidx] * y_item
        )
        return routed + self.b[tidx]


class ValueHead(torch.nn.Module):
    """Read the answer from the final token."""

    def __init__(self, d_model, n_types=N_TYPES):
        super().__init__()
        self.lin = torch.nn.Linear(d_model, n_types)

    def forward(self, h_last, tidx):
        z = F.layer_norm(h_last, (h_last.shape[-1],))
        return self.lin(z)[torch.arange(len(tidx), device=h_last.device), tidx]


class MeanPoolHead(torch.nn.Module):
    """Control head that takes the mean of the item features."""

    def __init__(self, d_model, n_types=N_TYPES, frac=False):
        super().__init__()
        self.lin = torch.nn.Linear(d_model, n_types)
        self.frac = frac
        self.register_buffer(
            "is_pair", torch.tensor([KIND[t] == "pair" for t in QTYPES])
        )

    def forward(self, h, imask, tidx):
        z = F.layer_norm(h, (h.shape[-1],))
        m = imask.float().unsqueeze(-1)
        pooled = (z * m).sum(1) / m.sum(1).clamp_min(1.0)
        out = self.lin(pooled)[torch.arange(len(tidx), device=h.device), tidx]
        if self.frac:
            n = imask.float().sum(1)
            mult = torch.where(self.is_pair[tidx], n * (n - 1) / 2, n)
            out = out * mult
        return out


@torch.no_grad()
def active_region_init(
    head, hz, layer, questions, seed=12345, target_std=0.5, center=0.5
):
    """Move each question type's starting scores into the active range."""

    def item_stats(qs, s_off, ti):
        """Measure the starting scores for one question type."""
        gen = make_mixture_gen(qs, seed + s_off)
        lists, qtexts, tidx, y, _ = gen(32)
        ids, attn, imask, _, qend = hz.encode(lists, qtexts)
        h = hz.features(ids, attn, [layer])[layer]
        hf = head._feat(h)
        pairm = imask.unsqueeze(2) & imask.unsqueeze(1)
        q = head.Wq(hf) * head.block1[ti]
        lg = (q @ head.Wk(hf).transpose(1, 2))[pairm]
        il = head.wu(hf).sum(-1)[imask]
        qh = h[torch.arange(h.shape[0], device=h.device), qend]
        return lg, il, (h, imask, tidx, qh, y)

    by_type = {}
    for t, p in questions:
        by_type.setdefault(t, []).append((t, p))

    lg_stds, il_stds = [], []
    for t, qs in by_type.items():
        lg, il, _ = item_stats(qs, 1 + QTYPES.index(t), QTYPES.index(t))
        if KIND[t] == "pair":
            lg_stds.append(lg.std().item())
        il_stds.append(il.std().item())
    sc = (target_std / max(sum(lg_stds) / max(len(lg_stds), 1), 1e-4)) ** 0.5
    head.Wq.weight.mul_(sc)
    head.Wq.bias.mul_(sc)
    head.Wk.weight.mul_(sc)
    head.Wk.bias.mul_(sc)
    head.wu.weight.mul_(target_std / max(sum(il_stds) / len(il_stds), 1e-4))

    for t, qs in by_type.items():
        lg, il, _ = item_stats(qs, 1 + QTYPES.index(t), QTYPES.index(t))
        head.beta1[QTYPES.index(t)] = center - lg.mean().item()
        head.beta2[QTYPES.index(t)] = center - il.mean().item()

    for t, qs in by_type.items():
        _, _, (h, imask, tidx, qh, y) = item_stats(
            qs, 7 + QTYPES.index(t), QTYPES.index(t)
        )
        y_pair, y_item = head.channels(h, imask, tidx, qh)
        ti = QTYPES.index(t)
        ch = y_pair if KIND[t] == "pair" else y_item
        cm = ch.mean().clamp_min(1e-4).item()
        scale_c = max(y.mean().item(), 1.0) / cm
        if KIND[t] == "pair":
            head.c1.data[ti] = scale_c
        else:
            head.c2.data[ti] = scale_c
        head.b.data[ti] = 0.0


# Data plumbing


def make_mixture_gen(questions, seed, n_lo=4, n_hi=16, iid_frac=0.3, para_mix=True):
    """Make one training batch for a question."""
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
        para = int(torch.randint(0, 2, (1,), generator=g).item()) if para_mix else 0
        qtext = question_text(t, p, para)
        tidx = torch.full((bs,), QTYPES.index(t), dtype=torch.long, device=DEV)
        return [r.tolist() for r in a], [qtext] * bs, tidx, y.to(DEV), (t, p, n)

    def gen_mixed(per_type, scales):
        """Make a mixed batch containing every question type."""
        n = int(torch.randint(n_lo, n_hi + 1, (1,), generator=g).item())
        lists, qtexts, tidxs, ys, srows = [], [], [], [], []
        for t in types:
            ps = by_type[t]
            p = ps[int(torch.randint(0, len(ps), (1,), generator=g).item())]
            a, y = sample(t, per_type, n, V, g, p, iid_frac=iid_frac)
            para = int(torch.randint(0, 2, (1,), generator=g).item()) if para_mix else 0
            qtext = question_text(t, p, para)
            lists += [r.tolist() for r in a]
            qtexts += [qtext] * per_type
            tidxs += [QTYPES.index(t)] * per_type
            ys.append(y)
            srows += [scales[t]] * per_type
        return (
            lists,
            qtexts,
            torch.tensor(tidxs, dtype=torch.long, device=DEV),
            torch.cat(ys).to(DEV),
            torch.tensor(srows, dtype=torch.float32, device=DEV),
        )

    gen.mixed = gen_mixed
    return gen


def type_scales(seed=7):
    g = torch.Generator().manual_seed(seed)
    sc = {}
    for t in QTYPES:
        p = [q for q in PARAMS[t] if q not in HELD_OUT[t]][0]
        sc[t] = family_scale(t, V, g, p, n_lo=4, n_hi=16)
    return sc


def fixed_n_metrics(preds, ys):
    p, y = (
        torch.tensor(preds, dtype=torch.float64),
        torch.tensor(ys, dtype=torch.float64),
    )
    ss_res = ((p - y) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    rel = (p - y).abs() / y.clamp(min=1.0)
    return {
        "mae": (p - y).abs().mean().item(),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-9),
        "within10": (rel <= 0.10).float().mean().item(),
        "exact": (p.round() == y).float().mean().item(),
        "null_mae": (y - y.mean()).abs().mean().item(),
        "null_within10": (((y - y.mean()).abs() / y.clamp(min=1.0)) <= 0.10)
        .float()
        .mean()
        .item(),
    }


@torch.no_grad()
def eval_arm(
    hz,
    head,
    kind_of_head,
    questions,
    lengths,
    seed,
    layer,
    n_eval=128,
    bs=16,
    shuffle_probe=False,
):
    """Calculate the result metrics for each question and length."""
    out = {}
    for t, p in questions:
        qtext = question_text(t, p, 0)
        tname = f"{t}" + (f"_{p}" if p is not None else "")
        out[tname] = {}
        for n in lengths:
            g = torch.Generator().manual_seed(10_000 + seed + n)
            preds, ys = [], []
            for _ in range(max(1, n_eval // bs)):
                a, y = sample(t, bs, n, V, g, p)
                lists = [r.tolist() for r in a]
                ids, attn, imask, last, qend = hz.encode(lists, [qtext] * bs)
                h = hz.features(ids, attn, [layer])[layer]
                tidx = torch.full((bs,), QTYPES.index(t), dtype=torch.long, device=DEV)
                if kind_of_head == "tally":
                    qh = h[torch.arange(bs, device=DEV), qend]
                    pred = head(h, imask, tidx, qh)
                elif kind_of_head == "meanpool":
                    pred = head(h, imask, tidx)
                else:  # value head
                    hl = h[torch.arange(bs, device=DEV), last]
                    pred = head(hl, tidx)
                preds += pred.float().cpu().tolist()
                ys += y.tolist()
            out[tname][n] = fixed_n_metrics(preds, ys)
            if shuffle_probe and n == lengths[-1]:
                pr = torch.tensor(preds)
                yy = torch.tensor(ys)
                c = (
                    torch.corrcoef(torch.stack([pr, yy]))[0, 1].item()
                    if yy.std() > 0
                    else 0.0
                )
                out[tname][n]["pred_truth_corr"] = c
    return out


# Training


def train_head(
    hz, arm, questions, seed, layer, steps, lr, bs, scales, rank=16, warmstart=False
):
    torch.manual_seed(2000 + seed)
    if arm == "tally":
        head = GTallyAdapter(hz.d_model, rank=rank).to(DEV)
    elif arm == "value":
        head = ValueHead(hz.d_model).to(DEV)
    elif arm == "meanpool":
        head = MeanPoolHead(hz.d_model).to(DEV)
    elif arm == "fracpool":
        head = MeanPoolHead(hz.d_model, frac=True).to(DEV)
    else:
        raise ValueError(arm)
    gen = make_mixture_gen(questions, seed)
    if arm == "tally":
        active_region_init(head, hz, layer, questions)
    if arm == "tally" and warmstart:
        for t in [t for t in QTYPES if KIND[t] == "pair"]:
            qs_t = [(tt, pp) for tt, pp in questions if tt == t]
            if not qs_t:
                continue
            gen_t = make_mixture_gen(qs_t, seed)
            optw = torch.optim.AdamW(head.parameters(), lr=lr)
            schedw = torch.optim.lr_scheduler.CosineAnnealingLR(
                optw, T_max=1500, eta_min=lr * 0.01
            )
            for _step in range(1500):
                lists, qtexts, tidx, y, _ = gen_t(bs)
                ids, attn, imask, last, qend = hz.encode(lists, qtexts)
                h = hz.features(ids, attn, [layer])[layer]
                qh = h[torch.arange(len(lists), device=DEV), qend]
                loss = F.mse_loss(head(h, imask, tidx, qh) / scales[t], y / scales[t])
                optw.zero_grad(set_to_none=True)
                loss.backward()
                for p in head.parameters():
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_([p], 1.0)
                optw.step()
                schedw.step()
            print(
                f"  [tally s{seed}] warm-start {t} done (loss {loss.item():.4f})",
                flush=True,
            )
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01
    )
    t0 = time.time()
    for step in range(steps):
        if arm == "tally":  # joint batches: all types in every optimizer step
            lists, qtexts, tidx, y, srow = gen.mixed(4, scales)
        else:
            lists, qtexts, tidx, y, (t, p, n) = gen(bs)
        ids, attn, imask, last, qend = hz.encode(lists, qtexts)
        h = hz.features(ids, attn, [layer])[layer]
        if arm == "value":
            hl = h[torch.arange(len(lists), device=DEV), last]
            pred = head(hl, tidx)
        elif arm == "tally":
            qh = h[torch.arange(len(lists), device=DEV), qend]
            pred = head(h, imask, tidx, qh)
        else:
            pred = head(h, imask, tidx)
        loss = (
            F.mse_loss(pred / srow, y / srow)
            if arm == "tally"
            else F.mse_loss(pred / scales[t], y / scales[t])
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()

        for p in head.parameters():
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], 1.0)
        opt.step()
        sched.step()
        if step % 300 == 0:
            tag = "joint" if arm == "tally" else t
            print(
                f"  [{arm} s{seed}] step {step} ({tag}) loss {loss.item():.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return head


# E1 probes


def run_probes(hz, layers=(7, 14, 21), n_train=320, seed=0):
    """Train probes at several Qwen layers."""
    results = {"match_auc": {}, "count_probe": {}}
    g = torch.Generator().manual_seed(seed)

    def item_bits(t, p, a):
        n = a.shape[1]
        if t == "count_v":
            return (a == p).float()
        if t == "gt_c":
            return (a > p).float()
        ai, aj = a.unsqueeze(2), a.unsqueeze(1)
        hit = {
            "inv": aj > ai,
            "drop5": aj >= ai + 5,
            "dup": aj == ai,
            "sum_s": aj + ai == p,
        }[t]
        m = torch.tril(torch.ones(n, n), diagonal=-1)
        return (
            (hit.float() * m.unsqueeze(0)).sum(dim=2) > 0
        ).float()  # item closes a pair

    for t in QTYPES:
        p = [q for q in PARAMS[t] if q not in HELD_OUT[t]][len(PARAMS[t]) // 3]
        qtext = question_text(t, p, 0)
        feats = {L: [] for L in layers}
        bits = []
        lastf = {L: [] for L in layers}
        counts, ns = [], []
        for _phase, nlist, total in (
            ("train", None, n_train),
            ("eval", [16, 48, 96], 96),
        ):
            for i in range(total // 16):
                n = (
                    int(torch.randint(4, 17, (1,), generator=g).item())
                    if nlist is None
                    else nlist[i % len(nlist)]
                )
                a, y = sample(t, 16, n, V, g, p)
                ids, attn, imask, last, _ = hz.encode(
                    [r.tolist() for r in a], [qtext] * 16
                )
                hs = hz.features(ids, attn, list(layers))
                bt = item_bits(t, p, a)
                for L in layers:
                    hf = F.layer_norm(hs[L], (hs[L].shape[-1],))
                    feats[L].append(hf[imask].half().cpu())
                    lastf[L].append(hf[torch.arange(16, device=DEV), last].half().cpu())
                bits.append(bt.reshape(-1).cpu())
                counts += y.tolist()
                ns += [n] * 16
        bits = torch.cat(bits)
        n_tr_items = sum(f.shape[0] for f in feats[layers[0]][: n_train // 16])
        results["match_auc"][t] = {}
        results["count_probe"][t] = {}
        for L in layers:
            X = torch.cat(feats[L]).float()
            Xtr, Xte = X[:n_tr_items], X[n_tr_items:]
            btr, bte = bits[:n_tr_items], bits[n_tr_items:]
            w = torch.zeros(X.shape[1], requires_grad=True)
            b0 = torch.zeros(1, requires_grad=True)
            o = torch.optim.Adam([w, b0], lr=0.05)
            for _ in range(300):
                lo = Xtr @ w + b0
                ls = F.binary_cross_entropy_with_logits(lo, btr)
                o.zero_grad()
                ls.backward()
                o.step()
            with torch.no_grad():
                s = Xte @ w + b0
                pos, neg = s[bte == 1], s[bte == 0]
                auc = (
                    (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()
                    if len(pos) and len(neg)
                    else float("nan")
                )
            results["match_auc"][t][L] = round(auc, 3)

            Xl = torch.cat(lastf[L]).float()
            yc = torch.tensor(counts, dtype=torch.float32)
            nn_ = torch.tensor(ns, dtype=torch.float32)
            tr = torch.arange(len(yc)) < (n_train // 16) * 16
            A = Xl[tr]
            yv = yc[tr]
            lam = 10.0
            W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ yv)
            per_n = {}
            for n in (16, 48, 96):
                m = (~tr) & (nn_ == n)
                if m.any():
                    per_n[n] = fixed_n_metrics((Xl[m] @ W).tolist(), yc[m].tolist())[
                        "r2"
                    ]
            results["count_probe"][t][L] = {k: round(v, 3) for k, v in per_n.items()}
        print(
            f"[probe] {t}: match AUC {results['match_auc'][t]} | "
            f"count-probe R2 {results['count_probe'][t]}",
            flush=True,
        )
    return results


# Main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "scout", "train"])
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--arm", default="tally")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--warmstart", action="store_true")
    ap.add_argument("--outtag", default="")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    hz = QHarness()
    scales = type_scales()
    print("type scales:", {k: round(v, 1) for k, v in scales.items()}, flush=True)

    if args.cmd == "probe":
        res = run_probes(hz, layers=(7, args.layer, 21))
        json.dump(res, open(os.path.join(RES, "e1_probes.json"), "w"), indent=2)

    elif args.cmd == "scout":
        for t in QTYPES:
            p = [q for q in PARAMS[t] if q not in HELD_OUT[t]][len(PARAMS[t]) // 3]
            path = os.path.join(RES, f"scout_{t}.json")
            if os.path.exists(path):
                print(f"skip scout {t}")
                continue
            head = train_head(
                hz,
                "tally",
                [(t, p)],
                0,
                args.layer,
                args.steps,
                args.lr,
                16,
                scales,
                rank=args.rank,
            )
            r = eval_arm(
                hz,
                head,
                "tally",
                [(t, p)],
                [16, 48, 96],
                0,
                args.layer,
                shuffle_probe=True,
            )
            json.dump(r, open(path, "w"), indent=2)
            print(f"[scout {t}] {json.dumps(r, indent=1)}", flush=True)
        path = os.path.join(RES, "scout_mixture.json")
        if not os.path.exists(path):
            qs = train_questions()
            head = train_head(
                hz,
                "tally",
                qs,
                0,
                args.layer,
                args.steps * 2,
                args.lr,
                16,
                scales,
                rank=args.rank,
            )
            evalqs = [
                (
                    t,
                    (
                        [q for q in PARAMS[t] if q not in HELD_OUT[t]][
                            len(PARAMS[t]) // 3
                        ]
                    ),
                )
                for t in QTYPES
            ]
            r = {
                "trained": eval_arm(
                    hz,
                    head,
                    "tally",
                    evalqs,
                    [16, 48, 96],
                    0,
                    args.layer,
                    shuffle_probe=True,
                ),
                "heldout": eval_arm(
                    hz, head, "tally", heldout_questions(), [16, 48, 96], 0, args.layer
                ),
            }
            json.dump(r, open(path, "w"), indent=2)
            print(f"[scout mixture] {json.dumps(r, indent=1)}", flush=True)

    elif args.cmd == "train":
        qs = train_questions()
        evalqs = [
            (t, ([q for q in PARAMS[t] if q not in HELD_OUT[t]][len(PARAMS[t]) // 3]))
            for t in QTYPES
        ]
        for s in range(args.seeds):
            path = os.path.join(RES, f"e4_{args.arm}{args.outtag}_s{s}.json")
            if os.path.exists(path):
                print(f"skip {args.arm} s{s}")
                continue
            head = train_head(
                hz,
                args.arm,
                qs,
                s,
                args.layer,
                args.steps,
                args.lr,
                16,
                scales,
                rank=args.rank,
                warmstart=args.warmstart,
            )
            kind = (
                "value"
                if args.arm == "value"
                else ("meanpool" if args.arm in ("meanpool", "fracpool") else "tally")
            )
            r = {
                "trained": eval_arm(
                    hz,
                    head,
                    kind,
                    evalqs,
                    [8, 16, 24, 48, 96],
                    s,
                    args.layer,
                    shuffle_probe=True,
                )
            }
            if args.arm == "tally":
                r["heldout"] = eval_arm(
                    hz,
                    head,
                    "tally",
                    heldout_questions(),
                    [8, 16, 24, 48, 96],
                    s,
                    args.layer,
                )
            json.dump(r, open(path, "w"), indent=2)
            print(f"[{args.arm} s{s}] done", flush=True)


if __name__ == "__main__":
    main()
