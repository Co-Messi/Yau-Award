"""Run the first counting experiments with frozen Qwen features."""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = "Qwen/Qwen3-0.6B-Base"  # overridden by --model
MODEL_TAG = ""  # "" for 0.6B (keeps existing filenames), "_4b"/"_14b" else
DTYPE = torch.float32  # overridden to bfloat16 for big models
SCALE_Q = 50.0
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def make_list(n, g=None):
    return torch.randint(0, 10, (n,), generator=g).tolist()


def inv_count(a):
    return sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > a[j])


def build_prompt(a):
    return "List:" + "".join(f" {x}" for x in a) + "\nInversions:"


class QwenHarness:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)
        self.model.eval()
        print(f"loaded {MODEL} ({DTYPE}); hidden={self.model.config.hidden_size}")

        self.digit_ids = {}
        for d in range(10):
            ids = self.tok(f"{d}", add_special_tokens=False)["input_ids"]
            assert len(ids) == 1, f"digit '{d}' is {len(ids)} tokens"
            self.digit_ids[ids[0]] = d
        print("digit tokens verified single-token (bare digits)")

    def encode_batch(self, lists, answers=None):
        texts = [build_prompt(a) for a in lists]
        if answers is not None:
            texts = [t + f" {y}" + self.tok.eos_token for t, y in zip(texts, answers)]
        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False,
        )
        ids, mask = enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)
        digit_mask = torch.zeros_like(ids, dtype=torch.bool)
        for tid in self.digit_ids:
            digit_mask |= ids == tid
        labels = None
        if answers is not None:
            labels = ids.clone()
            labels[mask == 0] = -100

            colon_id = self.tok("\nInversions:", add_special_tokens=False)["input_ids"][
                -1
            ]
            for b in range(ids.shape[0]):
                pos = (ids[b] == colon_id).nonzero()
                cut = pos[-1].item() if len(pos) else 0
                labels[b, : cut + 1] = -100
                dm = digit_mask[b].clone()
                dm[cut + 1 :] = False  # answer digits are not "list digits"
                digit_mask[b] = dm
        else:
            pass
        last_idx = mask.sum(dim=1) - 1
        return ids, mask, digit_mask, last_idx, labels

    @torch.no_grad()
    def hidden_states(self, ids, mask, layer=0):
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return out.hidden_states[
            layer
        ].float()  # cast bf16 trunk -> fp32 for the small heads


class TallyAdapter(torch.nn.Module):
    """Add the clamped pair scores and apply a linear output."""

    def __init__(self, d_model, rank=4):
        super().__init__()
        self.Wq = torch.nn.Linear(d_model, rank)
        self.Wk = torch.nn.Linear(d_model, rank)
        self.c = torch.nn.Parameter(torch.tensor(1.0))
        self.b = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, h, digit_mask):
        q, k = self.Wq(h), self.Wk(h)  # [B,T,r]
        logits = q @ k.transpose(1, 2)  # [B,T,T] <q_i, k_j>
        gate = torch.clamp(logits, 0.0, 1.0)
        T = h.shape[1]
        causal = torch.tril(torch.ones(T, T, device=h.device), diagonal=-1)  # j < i
        pair = digit_mask.unsqueeze(2) & digit_mask.unsqueeze(1)  # both digits
        y = (gate * causal.unsqueeze(0) * pair.float()).sum(dim=(1, 2))
        return self.c * y + self.b


class ValueHead(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin = torch.nn.Linear(d_model, 1)

    def forward(self, h_last):
        return self.lin(h_last)[:, 0]


def gen_training_batch(bs, g):
    ns = torch.randint(4, 17, (bs,), generator=g).tolist()
    lists = [make_list(n, g) for n in ns]
    ys = [inv_count(a) for a in lists]
    return lists, ys


def train_head(hz, arm, steps, seed, layer=0, lr=1e-3, bs=16):
    g = torch.Generator().manual_seed(seed)
    d_model = hz.model.config.hidden_size
    head = (TallyAdapter(d_model) if arm == "tally" else ValueHead(d_model)).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01
    )
    t0 = time.time()
    for step in range(steps):
        lists, ys = gen_training_batch(bs, g)
        ids, mask, dmask, last_idx, _ = hz.encode_batch(lists)
        y = torch.tensor(ys, dtype=torch.float32, device=DEV)
        if arm == "tally":
            h = hz.hidden_states(ids, mask, layer=layer)
            pred = head(h, dmask)
            loss = F.mse_loss(pred, y)
        else:
            h = hz.hidden_states(ids, mask, layer=-1)
            h_last = h[torch.arange(ids.shape[0], device=DEV), last_idx]
            pred = head(h_last) * SCALE_Q
            loss = F.mse_loss(pred / SCALE_Q, y / SCALE_Q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(
                f"  [{arm} s{seed}] step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return head


def closeness(preds, ys):
    """Calculate relative error and the percentage of close answers."""
    import statistics

    rel = [abs(p - y) / max(y, 1.0) for p, y in zip(preds, ys)]
    if not rel:
        return {"medrel": 1.0, "within10": 0.0, "within5": 0.0}
    return {
        "medrel": statistics.median(rel),  # median relative error (fraction)
        "within10": sum(r <= 0.10 for r in rel)
        / len(rel),  # fraction within 10% of truth
        "within5": sum(r <= 0.05 for r in rel)
        / len(rel),  # fraction within 5% of truth
    }


@torch.no_grad()
def eval_head(hz, head, arm, lengths, seed, layer=0, n_eval=128, bs=16):
    g = torch.Generator().manual_seed(10_000 + seed)
    res = {}
    for n in lengths:
        preds, ys = [], []
        for _ in range(n_eval // bs):
            lists = [make_list(n, g) for _ in range(bs)]
            ys += [inv_count(a) for a in lists]
            ids, mask, dmask, last_idx, _ = hz.encode_batch(lists)
            if arm == "tally":
                h = hz.hidden_states(ids, mask, layer=layer)
                p = head(h, dmask)
            else:
                h = hz.hidden_states(ids, mask, layer=-1)
                p = head(h[torch.arange(ids.shape[0], device=DEV), last_idx]) * SCALE_Q
            preds += p.float().cpu().tolist()
        p = torch.tensor(preds)
        y = torch.tensor(ys, dtype=torch.float32)
        res[n] = {
            "exact": (p.round() == y).float().mean().item(),
            "mae": (p - y).abs().mean().item(),
            **closeness(p.tolist(), y.tolist()),
        }
        print(
            f"  [{arm} s{seed}] n={n}: exact {res[n]['exact']:.3f} mae {res[n]['mae']:.2f} "
            f"medrel {res[n]['medrel'] * 100:.1f}% within10 {res[n]['within10'] * 100:.0f}%",
            flush=True,
        )
    return res


@torch.no_grad()
def eval_generate(hz, model, lengths, seed, fewshot=False, n_eval=64, bs=8):
    g = torch.Generator().manual_seed(20_000 + seed)
    shots = ""
    if fewshot:
        gs = torch.Generator().manual_seed(7)
        for _ in range(4):
            a = make_list(8, gs)
            shots += build_prompt(a) + f" {inv_count(a)}\n\n"
    res = {}
    for n in lengths:
        correct, abs_err, parsed = 0, [], 0
        pp, yy = [], []
        for _ in range(n_eval // bs):
            lists = [make_list(n, g) for _ in range(bs)]
            ys = [inv_count(a) for a in lists]
            texts = [shots + build_prompt(a) for a in lists]
            enc = hz.tok(
                texts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            ).to(DEV)
            out = model.generate(
                **enc,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=hz.tok.eos_token_id,
            )
            gen = hz.tok.batch_decode(
                out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for s, y in zip(gen, ys):
                tokn = s.strip().split()
                try:
                    v = int(tokn[0]) if tokn else None
                except ValueError:
                    v = None
                if v is not None:
                    parsed += 1
                    abs_err.append(abs(v - y))
                    correct += int(v == y)
                    pp.append(float(v))
                    yy.append(float(y))
        res[n] = {
            "exact": correct / n_eval,
            "mae": sum(abs_err) / max(len(abs_err), 1),
            "parse_rate": parsed / n_eval,
            **closeness(pp, yy),
        }
        print(
            f"  [gen] n={n}: exact {res[n]['exact']:.3f} mae {res[n]['mae']:.2f} "
            f"medrel {res[n]['medrel'] * 100:.1f}% within10 {res[n]['within10'] * 100:.0f}% parsed {res[n]['parse_rate']:.2f}",
            flush=True,
        )
    return res


def train_lora(hz, steps, seed, lr=1e-4, bs=8):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV),
        cfg,
    )
    model.train()
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01
    )
    t0 = time.time()
    for step in range(steps):
        lists, ys = gen_training_batch(bs, g)
        ids, mask, _, _, labels = hz.encode_batch(lists, answers=ys)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(
                f"  [lora s{seed}] step {step} loss {out.loss.item():.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    model.eval()
    return model


def main():
    global MODEL, MODEL_TAG, DTYPE
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arms", nargs="+", default=["fewshot", "valuehead", "tally", "lora"]
    )
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lora_steps", type=int, default=1200)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--lengths", nargs="+", type=int, default=[8, 16, 24, 32, 48, 64, 96]
    )
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument(
        "--tag", type=str, default="", help="filename tag, e.g. _4b (empty = 0.6B)"
    )
    ap.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "bfloat16"]
    )
    args = ap.parse_args()

    MODEL = args.model
    MODEL_TAG = args.tag
    DTYPE = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    os.makedirs(RESULTS, exist_ok=True)
    hz = QwenHarness()
    for arm in args.arms:
        n_seeds = 1 if arm == "fewshot" else args.seeds
        for seed in range(n_seeds):
            path = os.path.join(RESULTS, f"qwen{MODEL_TAG}_{arm}_s{seed}.json")
            if os.path.exists(path):
                print(f"skip {arm} s{seed}")
                continue
            print(f"=== qwen {arm} s{seed} ===", flush=True)
            if arm == "fewshot":
                res = eval_generate(hz, hz.model, args.lengths, seed, fewshot=True)
            elif arm == "lora":
                model = train_lora(hz, args.lora_steps, seed)
                res = eval_generate(hz, model, args.lengths, seed, fewshot=False)
                del model
            else:
                head = train_head(hz, arm, args.steps, seed, layer=args.layer)
                res = eval_head(hz, head, arm, args.lengths, seed, layer=args.layer)
            json.dump(
                {"arm": arm, "seed": seed, "results": res}, open(path, "w"), indent=2
            )


if __name__ == "__main__":
    main()
