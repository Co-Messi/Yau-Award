"""Data generators and exact labels for the counting tasks."""

import torch


# Exact labels


def _pair_mask(n, device):
    return torch.tril(torch.ones(n, n, device=device), diagonal=-1)  # [i, j] with j < i


def label(name, a, param=None):
    """Calculate the correct count for a batch of lists."""
    B, n = a.shape
    ai = a.unsqueeze(2)  # broadcast as a_i over rows  [B, n, 1]
    aj = a.unsqueeze(1)  # broadcast as a_j over cols  [B, 1, n]
    if name == "inv":
        hit = aj > ai
    elif name == "drop5":
        hit = aj >= ai + 5
    elif name == "dup":
        hit = aj == ai
    elif name == "sum_s":
        hit = aj + ai == param
    elif name == "count_v":
        return (a == param).float().sum(dim=1)
    elif name == "gt_c":
        return (a > param).float().sum(dim=1)
    else:
        raise ValueError(name)
    m = _pair_mask(n, a.device)

    return (hit.float() * m.unsqueeze(0)).sum(dim=(1, 2))


KIND = {
    "inv": "pair",
    "drop5": "pair",
    "dup": "pair",
    "sum_s": "pair",
    "count_v": "item",
    "gt_c": "item",
}


PARAMS = {
    "inv": [None],
    "drop5": [None],
    "dup": [None],
    "sum_s": [3, 4, 5, 6, 7, 8, 9],
    "count_v": list(range(10)),
    "gt_c": [0, 1, 2, 3, 4, 5],
}
HELD_OUT = {
    "sum_s": [8, 9],
    "count_v": [7, 8, 9],
    "gt_c": [4],
    "inv": [],
    "drop5": [],
    "dup": [],
}

QUESTION_TEXT = {
    "inv": [
        "how many pairs are out of order",
        "how many earlier numbers are bigger than a later one, counted over pairs",
    ],
    "drop5": [
        "how many pairs drop by 5 or more",
        "how many pairs go down by at least 5",
    ],
    "dup": ["how many duplicate pairs", "how many pairs of equal numbers"],
    "sum_s": ["how many pairs sum to {p}", "how many pairs add up to {p}"],
    "count_v": ["how many {p}s", "how many times does {p} appear"],
    "gt_c": ["how many numbers greater than {p}", "how many numbers above {p}"],
}


# Planted samplers


def _partial_sort_spread(bs, n, V, g):
    a = torch.randint(0, V, (bs, n), generator=g)
    a, _ = torch.sort(a, dim=1)
    for b in range(bs):
        k = int(torch.randint(0, n + 1, (1,), generator=g).item())
        if k >= 2:
            pos = torch.randperm(n, generator=g)[:k]
            perm = pos[torch.randperm(k, generator=g)]
            a[b, pos] = a[b, perm]
    flip = torch.rand(bs, generator=g) < 0.5
    a[flip] = a[flip].flip(dims=[1])
    return a


def _distinct_spread(bs, n, V, g):
    a = torch.empty(bs, n, dtype=torch.long)
    for b in range(bs):
        kmax = min(V, n)
        u = torch.rand(1, generator=g).item()
        k = max(1, int(round(kmax**u)))  # log-spread over 1..kmax
        vals = torch.randperm(V, generator=g)[:k]
        a[b] = vals[torch.randint(0, k, (n,), generator=g)]
    return a


def _plant_items(bs, n, V, g, ok_vals, other_vals):
    """Make lists with a chosen number of matching items."""
    a = torch.empty(bs, n, dtype=torch.long)
    ok = torch.tensor(ok_vals)
    oth = torch.tensor(other_vals)
    for b in range(bs):
        m = int(torch.randint(0, n + 1, (1,), generator=g).item())
        pos = torch.randperm(n, generator=g)
        a[b, pos[:m]] = ok[torch.randint(0, len(ok), (m,), generator=g)]
        a[b, pos[m:]] = oth[torch.randint(0, len(oth), (n - m,), generator=g)]
    return a


def _plant_sum_pairs(bs, n, V, g, s):
    a = torch.empty(bs, n, dtype=torch.long)
    lo, hi = max(0, s - (V - 1)), min(V - 1, s)
    xs = [x for x in range(lo, hi + 1) if x < s - x]
    for b in range(bs):
        if not xs or torch.rand(1, generator=g).item() < 0.3:
            a[b] = torch.randint(0, V, (n,), generator=g)
            continue
        xstar = xs[int(torch.randint(0, len(xs), (1,), generator=g).item())]
        ystar = s - xstar
        m = int(torch.randint(0, n + 1, (1,), generator=g).item())
        m1 = int(torch.randint(0, m + 1, (1,), generator=g).item()) if m else 0
        f_hi = [y for y in range(V) if 2 * y > s and y not in (xstar, ystar)]
        f_lo = [y for y in range(V) if 0 <= 2 * y < s and y not in (xstar, ystar)]
        pick_hi = f_hi and (not f_lo or torch.rand(1, generator=g).item() < 0.5)
        fill = torch.tensor(f_hi if pick_hi else (f_lo or f_hi or list(range(V))))
        pos = torch.randperm(n, generator=g)
        a[b, pos[:m1]] = xstar
        a[b, pos[m1:m]] = ystar
        a[b, pos[m:]] = fill[torch.randint(0, len(fill), (n - m,), generator=g)]
    return a


def sample(name, bs, n, V, g, param=None, iid_frac=0.0):
    """Make one batch and calculate its labels."""
    if name in ("inv", "drop5"):
        a = _partial_sort_spread(bs, n, V, g)
    elif name == "dup":
        a = _distinct_spread(bs, n, V, g)
    elif name == "sum_s":
        a = _plant_sum_pairs(bs, n, V, g, param)
    elif name == "count_v":
        a = _plant_items(bs, n, V, g, [param], [v for v in range(V) if v != param])
    elif name == "gt_c":
        a = _plant_items(bs, n, V, g, list(range(param + 1, V)), list(range(param + 1)))
    else:
        raise ValueError(name)
    if iid_frac > 0:
        k = int(bs * iid_frac)
        if k:
            a[:k] = torch.randint(0, V, (k, n), generator=g)
    y = label(name, a, param)
    return a, y


# Hand-set witnesses


def quad_features(a, V):
    """Turn each number into the features 1, a, and a squared."""
    x = a.float()
    return torch.stack([torch.ones_like(x), x, x * x], dim=-1)  # [..., 3]


def witness_logits(name, a, param=None):
    """Calculate the hand-set scores for one counting rule."""
    x = a.float()
    xi = x.unsqueeze(2)  # a_i
    xj = x.unsqueeze(1)  # a_j
    if name == "inv":  # 1[a_j > a_i]: logit = a_j - a_i  (rank 2)
        return xj - xi
    if name == "drop5":  # 1[a_j >= a_i + 5]: logit = a_j - a_i - 4  (rank 2)
        return xj - xi - 4.0
    if name == "dup":  # 1[a_j == a_i]: logit = 1 - (a_i - a_j)^2  (rank 3 on f)
        return 1.0 - (xi - xj) ** 2
    if name == "sum_s":  # 1[a_i + a_j == s]: logit = 1 - (a_i + a_j - s)^2  (rank 3)
        return 1.0 - (xi + xj - float(param)) ** 2
    if name == "count_v":  # 1[a_i == v]: logit = 1 - (a_i - v)^2  (item channel)
        return 1.0 - (x - float(param)) ** 2
    if name == "gt_c":  # 1[a_i > c]: logit = a_i - c  (item channel, rank 1)
        return x - float(param)
    raise ValueError(name)


def witness_count(name, a, param=None):
    """Clamp the scores and add them to get the count."""
    lg = witness_logits(name, a, param)
    gate = torch.clamp(lg, 0.0, 1.0)
    if KIND[name] == "item":
        return gate.sum(dim=1)
    n = a.shape[1]
    return (gate * _pair_mask(n, a.device).unsqueeze(0)).sum(dim=(1, 2))


# Null + gate


def null_stats(name, V, lengths, g, param=None, n_samples=1000, bs=100):
    """Calculate a baseline that ignores the list contents."""
    out = {}
    for n in lengths:
        ys = []
        for _ in range(n_samples // bs):
            _, y = sample(name, bs, n, V, g, param)
            ys.append(y)
        y = torch.cat(ys)
        mu = y.mean()
        rel = (y - mu).abs() / y.clamp(min=1.0)
        out[n] = {
            "mean": mu.item(),
            "null_mae": (y - mu).abs().mean().item(),
            "null_within10": (rel <= 0.10).float().mean().item(),
            "y_std": y.std().item(),
        }
    return out


def null_defeat_gate(V, lengths, seed=0, verbose=True):
    """Check that the simple baseline stays weak on every task."""
    g = torch.Generator().manual_seed(seed)
    ok = True
    for name in KIND:
        param = PARAMS[name][len(PARAMS[name]) // 2]
        st = null_stats(name, V, lengths, g, param)
        worst = max(v["null_within10"] for v in st.values())
        if verbose:
            row = "  ".join(
                f"n={n}:w10={v['null_within10']:.2f}/mae={v['null_mae']:.0f}"
                for n, v in st.items()
            )
            print(
                f"[gate] {name:8s} param={param}  {row}  -> {'OK' if worst < 0.2 else 'FAIL'}"
            )
        ok &= worst < 0.2
    return ok


def family_scale(name, V, g, param=None, n_lo=8, n_hi=64, n_samples=600):
    ys = []
    for _ in range(n_samples // 50):
        n = int(torch.randint(n_lo, n_hi + 1, (1,), generator=g).item())
        _, y = sample(name, 50, n, V, g, param)
        ys.append(y)
    return max(torch.cat(ys).std().item(), 1.0)
