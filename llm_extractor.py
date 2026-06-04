#!/usr/bin/env python3
"""Local LLM atomic fact extractor using Ollama (Qwen2.5-1.5B-Instruct).

Extracts self-contained declarative facts from a single conversational turn.
Uses only stdlib (urllib.request + json) — no extra dependencies required.
Returns an empty list silently if Ollama is not running or returns invalid JSON.

Model : qwen2.5:1.5b  (Apache-2.0, ~986 MB Q4_K_M, 32 K context)
Install: ollama pull qwen2.5:1.5b
Endpoint: http://localhost:11434/api/chat
"""

import json
import urllib.error
import urllib.request

_OLLAMA_URL = "http://localhost:11434/api/chat"
_MODEL = "qwen2.5:1.5b"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Design goals:
#   - Extract ONLY grounded, explicitly stated facts (no inference/speculation).
#   - Each fact is a self-contained declarative sentence using the speaker's
#     name as grammatical subject (third-person singular).
#   - Preserve: exact names, places, dates, numbers, negations.
#   - Skip: questions, greetings, fillers, acknowledgments, opinions about
#     what the OTHER speaker might enjoy/prefer/think.
#   - Return JSON only; {"facts":[]} when nothing is extractable.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You extract atomic biographical and episodic facts from a single conversational turn.

Rules:
1. Extract ONLY facts that are explicitly and directly stated by the speaker.
2. Do NOT infer, speculate, or add information not present in the text.
3. Each fact MUST be a self-contained declarative sentence with the speaker's name as subject.
4. Preserve exact names, places, dates, numbers, and negations word-for-word.
5. SKIP: questions ("?"), greetings, one-word or filler responses, acknowledgments under 5 words, and statements about what the OTHER person might like/think.
6. Output JSON only — no prose, no markdown, no explanation.
   Format: {"facts": [{"text": "<declarative sentence>"}]}
   If nothing qualifies, output: {"facts": []}

Examples:
Input: "Caroline: I've been researching climate policy for the last three years at the university."
Output: {"facts": [{"text": "Caroline has been researching climate policy at the university for three years."}]}

Input: "John: I visited Tokyo in March 2022 and loved the food there."
Output: {"facts": [{"text": "John visited Tokyo in March 2022."}, {"text": "John enjoyed the food in Tokyo."}]}

Input: "Melanie: Did you hear about the new park opening?"
Output: {"facts": []}

Input: "Alice: Yeah, totally."
Output: {"facts": []}

Input: "Sam: I don't own a car — I take the subway everywhere."
Output: {"facts": [{"text": "Sam does not own a car."}, {"text": "Sam takes the subway for transportation."}]}

Input: "Evan: My wife and I got married last October in Hawaii."
Output: {"facts": [{"text": "Evan got married in October in Hawaii."}]}
"""


def extract_atomic_facts(curr_line: str, timeout: int = 60) -> list[str]:
    """Extract atomic declarative facts from a single 'Speaker: text' conversational line.

    Args:
        curr_line: A conversational turn in 'Speaker: text' format.
        timeout:   HTTP request timeout in seconds (default 60, raised from 10
                   after benchmarking: Qwen2.5-1.5B needs ~10-40s on first call).

    Returns:
        List of fact strings (may be empty).  Never raises — all errors are
        silently swallowed and return [] so callers never lose raw storage.
    """
    if not curr_line or not curr_line.strip():
        return []

    # Pre-filter: skip trivially short turns (fillers, acks, one-liners < 4 words)
    # "Yeah, okay." / "Hmm." / "Sure." → never have extractable facts
    _body = curr_line.split(":", 1)[-1].strip() if ":" in curr_line else curr_line
    if len(_body.split()) < 4:
        return []

    payload = {
        "model": _MODEL,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 256,
        },
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": curr_line.strip()},
        ],
    }

    try:
        req = urllib.request.Request(
            _OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            content = body["message"]["content"]
            data = json.loads(content)
            facts = data.get("facts", [])
            return [
                f["text"].strip()
                for f in facts
                if isinstance(f, dict) and isinstance(f.get("text"), str) and f["text"].strip()
            ]
    except Exception:
        # Ollama not running, model not loaded, invalid JSON, timeout — all silently ignored.
        return []


def extract_batch_facts(
    curr_lines: list[str],
    workers: int = 4,
    timeout: int = 60,
) -> list[list[str]]:
    """Extract atomic facts from a batch of turns in parallel.

    Sends up to `workers` concurrent requests to Ollama (thread-per-request,
    I/O-bound).  Each request uses the same 60 s timeout as the single-turn
    function.  Results are returned in INPUT ORDER regardless of completion
    order, so callers can zip(curr_lines, results) safely.

    This is the fast path for benchmark reingest:
      - Saturates Ollama's GPU queue instead of draining it one call at a time.
      - Expected throughput: ~4× versus sequential at workers=4.
      - Deterministic: all futures are joined before the function returns, so
        callers can rely on all facts being available before eval starts.

    Args:
        curr_lines: List of 'Speaker: text' conversational turns.
        workers:    Max concurrent Ollama requests (default 4; limited by GPU).
        timeout:    Per-request HTTP timeout in seconds.

    Returns:
        List of fact-lists, one per input line, in input order.
        On any per-turn error the corresponding entry is [].
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed  # stdlib only

    if not curr_lines:
        return []

    results: list[list[str]] = [[] for _ in curr_lines]

    def _task(idx: int, line: str) -> tuple[int, list[str]]:
        return idx, extract_atomic_facts(line, timeout=timeout)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, i, ln): i for i, ln in enumerate(curr_lines)}
        for fut in as_completed(futures):
            try:
                idx, facts = fut.result()
                results[idx] = facts
            except Exception:
                pass  # already returns [] per-turn on error

    return results
