"""Check that the Qwen item masks select the right tokens."""

from qwen_howmany import QHarness, question_text


def main():
    hz = QHarness()
    lists = [[5, 2, 9, 7, 7], [0, 1, 2], [9] * 8]
    qtexts = [
        question_text("count_v", 7, 0),  # question contains the digit 7!
        question_text("sum_s", 9, 0),  # contains 9
        question_text("inv", None, 0),
    ]
    ids, attn, imask, last, qend = hz.encode(lists, qtexts)

    for b, a in enumerate(lists):
        assert int(imask[b].sum()) == len(a), (
            f"row {b}: mask size {int(imask[b].sum())} != {len(a)}"
        )

        toks = [hz.tok.decode([int(i)]) for i in ids[b][imask[b]]]
        assert [int(t) for t in toks] == a, f"row {b}: {toks} != {a}"

        text = hz.tok.decode(ids[b][attn[b] == 1].tolist())
        first_item = int(imask[b].nonzero()[0])
        prefix = hz.tok.decode(ids[b][:first_item].tolist())
        assert "List:" in prefix and "Answer" not in prefix

        assert hz.tok.decode([int(ids[b, last[b]])]).endswith(":")
        print(f"row {b} OK: {text!r}")

    ids2, attn2, imask2, _, _ = hz.encode([[1, 2, 3]], [question_text("count_v", 7, 0)])
    toks2 = [hz.tok.decode([int(i)]) for i in ids2[0][imask2[0]]]
    assert toks2 == ["1", "2", "3"], toks2
    print("digit-laden question leak test OK:", toks2)
    print("\nALL MASK TESTS PASS")


if __name__ == "__main__":
    main()
