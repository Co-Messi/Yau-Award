"""Retry the larger Qwen experiment after download limits."""

import os
import subprocess
import sys
import time

REPO = "Qwen/Qwen3-14B-Base"
CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-14B-Base")
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_MB = 27000  # ~28GB complete
STALL_SEC = 120  # kill a child making no disk progress for this long
GAP_SEC = 900  # 15 min between attempts (gentle)
DEADLINE_H = 8  # give up after this many hours


def dirsize_mb():
    t = 0
    for r, _, fs in os.walk(CACHE):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t / 1e6


def attempt():
    env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"from huggingface_hub import snapshot_download; snapshot_download('{REPO}')",
        ],
        env=env,
    )
    last, last_t = dirsize_mb(), time.time()
    while p.poll() is None:
        time.sleep(15)
        s = dirsize_mb()
        if s > last + 1:
            last, last_t = s, time.time()
        elif time.time() - last_t > STALL_SEC:
            p.kill()
            return False, s
    return p.returncode == 0, dirsize_mb()


def main():
    deadline = time.time() + DEADLINE_H * 3600
    n = 0
    while time.time() < deadline:
        n += 1
        ok, sz = attempt()
        print(
            f"[{time.strftime('%H:%M')}] attempt {n}: {sz:.0f}/{TARGET_MB}MB "
            f"({'complete' if ok and sz > TARGET_MB else 'incomplete'})",
            flush=True,
        )
        if ok and sz > TARGET_MB:
            print("DOWNLOAD_COMPLETE", flush=True)
            break
        time.sleep(GAP_SEC)
    else:
        print(
            "Stopped after",
            DEADLINE_H,
            "h because the download was still limited",
            flush=True,
        )
        sys.exit(1)

    print("running 14B arms...", flush=True)
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "qwen_extensivity.py"),
            "--model",
            REPO,
            "--tag",
            "_14b",
            "--dtype",
            "bfloat16",
            "--arms",
            "valuehead",
            "tally",
            "fewshot",
            "--seeds",
            "2",
            "--steps",
            "1500",
            "--lengths",
            "8",
            "16",
            "24",
            "32",
            "48",
            "64",
            "96",
            "128",
            "160",
        ],
        cwd=HERE,
        check=True,
    )
    print("14B_ARMS_DONE", flush=True)


if __name__ == "__main__":
    main()
