"""Small DeepSeek client for the real-document experiment."""

import os
import time

from openai import OpenAI

MODELS = {"pro": "deepseek-v4-pro", "flash": "deepseek-v4-flash"}


def _env(path=None):
    path = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    out = {}
    if os.path.exists(path):
        with open(path) as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" in line:
                    a, b = line.split("=", 1)
                    out[a.strip()] = b.strip()

    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
        if os.environ.get(name):
            out[name] = os.environ[name]
    return out


def client(path=None):
    k = _env(path)
    key = k.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return OpenAI(
        api_key=key, base_url=k.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )


def ask(
    c, prompt, model="pro", thinking=False, max_tokens=None, temperature=0.0, retries=3
):
    """Send one request and record its token counts."""
    mdl = MODELS.get(model, model)
    kw = {}
    if not thinking:
        kw["extra_body"] = {"thinking": {"type": "disabled"}}
        max_tokens = max_tokens or 16  # a one-pass answer is a number
    else:
        max_tokens = max_tokens or 4096  # reasoning needs room

    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = c.chat.completions.create(
                model=mdl,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kw,
            )
            dt = time.time() - t0
            m = r.choices[0].message
            reasoning = getattr(m, "reasoning_content", None) or ""
            u = r.usage
            rt = getattr(u, "reasoning_tokens", None)
            if rt is None:  # not always exposed; fall back to the text
                rt = len(reasoning)

            if not thinking and rt > 0:
                raise RuntimeError(
                    f"one-pass arm requested but model emitted {rt} reasoning tokens; "
                    "the thinking setting changed, so these results are invalid"
                )

            return {
                "text": (m.content or "").strip(),
                "reasoning_chars": len(reasoning),
                "reasoning_tokens": rt,
                "in_tokens": u.prompt_tokens,
                "out_tokens": u.completion_tokens,
                "seconds": dt,
                "model": mdl,
                "thinking": thinking,
            }
        except RuntimeError:
            raise  # mode violations are never retried
        except Exception as e:  # transient API errors only
            last = e
            time.sleep(2**attempt)
    raise RuntimeError(f"failed after {retries} attempts: {last}")


if __name__ == "__main__":
    c = client()
    for th in (False, True):
        r = ask(c, "Reply with only the number 7.", thinking=th)
        print(
            f"thinking={str(th):5s} -> {r['text']!r:6s} "
            f"reasoning_tokens={r['reasoning_tokens']:4d} "
            f"out={r['out_tokens']:4d} {r['seconds']:.1f}s"
        )
