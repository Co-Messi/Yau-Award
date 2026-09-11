"""Check the hand-written counting rules on random lists."""

import torch

from howmany_tasks import KIND, PARAMS, label, sample, witness_count, null_defeat_gate

V = 64
LENGTHS = [16, 64, 128, 256, 512]


def adversarial(bs, n, V, g):
    rows = [
        torch.sort(torch.randint(0, V, (n,), generator=g))[0],
        torch.sort(torch.randint(0, V, (n,), generator=g))[0].flip(0),
        torch.full((n,), int(torch.randint(0, V, (1,), generator=g).item())),
        torch.randint(0, 2, (n,)),
    ]
    while len(rows) < bs:
        rows.append(torch.randint(0, V, (n,), generator=g))
    return torch.stack(rows[:bs])


def main():
    g = torch.Generator().manual_seed(0)
    total = fails = 0
    for name in KIND:
        for param in PARAMS[name]:
            for n in LENGTHS:
                batches = [
                    sample(name, 16, n, V, g, param)[0],
                    torch.randint(0, V, (16, n), generator=g),
                    adversarial(16, n, V, g),
                ]
                for a in batches:
                    y = label(name, a, param)
                    yh = witness_count(name, a, param)
                    bad = (yh - y).abs().max().item()
                    total += 1
                    if bad > 1e-3:
                        fails += 1
                        print(f"FAIL {name} param={param} n={n}: max abs err {bad}")
    print(
        f"\nwitness battery: {total - fails}/{total} exact"
        + (
            "  -- ALL EXACT, zero training"
            if fails == 0
            else "  -- BUG, fix before anything"
        )
    )

    print("\nnull-defeat gate (toy V=64):")
    ok = null_defeat_gate(V, LENGTHS)
    print("gate:", "PASS" if ok else "FAIL -- redesign sampler(s)")

    print("\nnull-defeat gate (Qwen digits V=10, eval lengths):")
    ok10 = null_defeat_gate(10, [8, 16, 24, 48, 96], seed=1)
    print("gate:", "PASS" if ok10 else "FAIL -- redesign sampler(s)")


if __name__ == "__main__":
    main()
