"""Label community board budget requests with gpt-5.4-nano in Foundry.

Memoization: each distinct request text goes to the model once. The answer is stored
in a SQLite cache keyed on the prompt, the model, the settings and the text, so a
request repeated across budget cycles reuses its label instead of being paid for again.

Usage (repo folder, venv active):
    python classify.py dev                   # label the 35 dev-set requests with the newest prompt
    python classify.py dev --no-cache        # ask the model again even if cached (stability check)
    python classify.py dev --prompt v1       # use an older prompt from prompts/classify_v1.md
"""
import argparse
import csv
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from openai import BadRequestError
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
DEFAULT_PROMPT = "v2"
# Frozen Nov 5, 2026 after the dev loop. A frozen prompt is never edited; changes go in a new version.
FROZEN_PROMPTS = {"v2": "8fd196398a8202ae69bf49e8812cdd9ebc25f73835d59fcbbd59951df2a41b16"}
CACHE_PATH = ROOT / "data" / "label_cache.db"
DEV_PATH = ROOT / "labels" / "dev_set_labels.csv"
RESULTS_DIR = ROOT / "results"

BASE_URL = os.environ.get("AZURE_OPENAI_BASE_URL", "https://nyc-ev-gap-js.openai.azure.com/openai/v1/")
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano")
MODEL_VERSION = "2026-03-17"  # pinned in Foundry with NoAutoUpgrade
REASONING_EFFORT = "none"     # accepted by this model (smoke test, Nov 4, 2026)
MAX_OUTPUT_TOKENS = 300

# Dollars per 1M tokens, Global Standard, East US 2 (Azure retail price list, checked Nov 1, 2026).
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.20, 0.02, 1.25


# The structured answer. Pydantic turns this class into the JSON schema sent with each request
# (a docstring here would be sent to the model too, so the notes stay in comments).
# Reason comes first so the model states the ask before it chooses a label.
class Label(BaseModel):
    reason: str
    label: Literal["ev_charging", "parking_or_curb", "clean_air", "other"]
    parking_concern: Literal["yes", "no"]


class TooManyCalls(Exception):
    """Raised before a run would exceed its call limit: the code's spending brake."""


def normalize(text):
    """Collapse runs of spaces, tabs and newlines so trivial spacing differences match."""
    return " ".join(text.split())


def model_input(request, explanation):
    return f"Request: {normalize(request)}\nExplanation: {normalize(explanation)}"


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(prompt, text):
    """Changing the prompt, model, settings or text gives a new key, so stale labels are never reused."""
    return sha256(json.dumps([sha256(prompt), DEPLOYMENT, MODEL_VERSION, REASONING_EFFORT, text]))


def open_cache(path=CACHE_PATH):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS labels (
        key TEXT PRIMARY KEY, text TEXT, label TEXT, parking_concern TEXT, reason TEXT,
        prompt_sha TEXT, deployment TEXT, model_version TEXT, effort TEXT,
        input_tokens INTEGER, output_tokens INTEGER, cached_input_tokens INTEGER, created_at TEXT)""")
    return conn


def cache_get(conn, key):
    row = conn.execute("SELECT label, parking_concern, reason FROM labels WHERE key = ?", (key,)).fetchone()
    return dict(zip(("label", "parking_concern", "reason"), row)) if row else None


def cache_put(conn, key, text, prompt, result, usage):
    conn.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        key, text, result["label"], result["parking_concern"], result["reason"],
        sha256(prompt), DEPLOYMENT, MODEL_VERSION, REASONING_EFFORT,
        usage.input_tokens, usage.output_tokens, usage.input_tokens_details.cached_tokens or 0,
        datetime.now(timezone.utc).isoformat()))
    conn.commit()  # saved after every call, so an interrupted run resumes where it stopped


def make_client():
    # Imported here so the tests, which use a fake client, don't need Azure sign-in.
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import OpenAI
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
    return OpenAI(base_url=BASE_URL, api_key=token_provider, max_retries=5)


def ask_model(client, prompt, text):
    """One model call. Returns (result dict, usage)."""
    response = client.responses.parse(
        model=DEPLOYMENT,
        instructions=prompt,
        input=text,
        text_format=Label,
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    if response.output_parsed is None:
        raise ValueError(f"no parsed answer: {response.output_text!r}")
    return response.output_parsed.model_dump(), response.usage


def classify_texts(client, conn, prompt, texts, use_cache=True, max_calls=100):
    """Label each distinct text once. Returns ({text: result}, stats).

    With use_cache=False every text goes to the model and nothing is written to the
    cache, so a stability rerun can't overwrite the first run's labels.
    """
    distinct = list(dict.fromkeys(texts))  # drops repeats, keeps first-seen order
    results = {}
    stats = {"distinct": len(distinct), "cache_hits": 0, "calls": 0, "errors": 0,
             "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for text in distinct:
        key = cache_key(prompt, text)
        if use_cache:
            hit = cache_get(conn, key)
            if hit:
                results[text] = {**hit, "from_cache": True, "error": ""}
                stats["cache_hits"] += 1
                continue
        if stats["calls"] >= max_calls:
            raise TooManyCalls(f"stopped at {max_calls} calls; raise --max-calls to go further")
        stats["calls"] += 1
        try:
            result, usage = ask_model(client, prompt, text)
        except (BadRequestError, ValueError) as err:  # e.g. a content filter block; not cached
            results[text] = {"label": "", "parking_concern": "", "reason": "", "from_cache": False,
                             "error": str(err)[:300]}
            stats["errors"] += 1
            continue
        stats["input_tokens"] += usage.input_tokens
        stats["cached_input_tokens"] += usage.input_tokens_details.cached_tokens or 0
        stats["output_tokens"] += usage.output_tokens
        if use_cache:
            cache_put(conn, key, text, prompt, result, usage)
        results[text] = {**result, "from_cache": False, "error": ""}
    return results, stats


def cost(stats):
    fresh = stats["input_tokens"] - stats["cached_input_tokens"]
    return (fresh * PRICE_IN + stats["cached_input_tokens"] * PRICE_CACHED
            + stats["output_tokens"] * PRICE_OUT) / 1_000_000


def prompt_path(version):
    return PROMPTS_DIR / f"classify_{version}.md"


def load_prompt(version):
    """Read a prompt file and refuse to use a frozen one that has been edited."""
    prompt = prompt_path(version).read_text(encoding="utf-8")
    if version in FROZEN_PROMPTS and sha256(prompt) != FROZEN_PROMPTS[version]:
        raise SystemExit(f"prompts/classify_{version}.md was edited after it was frozen. "
                         "Restore it with git, or put the change in a new version.")
    return prompt


def run_dev(use_cache, max_calls, version=DEFAULT_PROMPT):
    prompt = load_prompt(version)
    with open(DEV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [model_input(r["request"], r["explanation"]) for r in rows]

    conn = open_cache()
    results, stats = classify_texts(make_client(), conn, prompt, texts, use_cache, max_calls)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"dev_{version}{'' if use_cache else '_nocache'}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dev_id", "pred_label", "pred_parking_concern", "reason", "from_cache", "error"])
        for r, text in zip(rows, texts):
            p = results[text]
            writer.writerow([r["dev_id"], p["label"], p["parking_concern"], p["reason"], p["from_cache"], p["error"]])

    print(f"prompt {version}: {len(rows)} rows, {stats['distinct']} distinct texts: {stats['calls']} model calls, "
          f"{stats['cache_hits']} cache hits, {stats['errors']} errors")
    print(f"tokens: {stats['input_tokens']} in ({stats['cached_input_tokens']} cached), "
          f"{stats['output_tokens']} out; cost ${cost(stats):.4f}")
    print(f"wrote {out.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=["dev"], help="which requests to label")
    parser.add_argument("--no-cache", action="store_true", help="ask the model even if a label is cached")
    parser.add_argument("--max-calls", type=int, default=100, help="stop before making more calls than this")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt version, e.g. v1 or v2")
    args = parser.parse_args()
    run_dev(use_cache=not args.no_cache, max_calls=args.max_calls, version=args.prompt)


if __name__ == "__main__":
    main()
