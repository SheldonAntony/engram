#!/usr/bin/env python3
"""Hybrid classifier: keyword scoring with LLM fallback.

CLI usage:
    classifier.py '{"prompt": "...", "useLLM": true, "apiKey": "sk-..."}'

Returns:
    {"type": "<category>"}   — category is one of: bug, feature, refactor,
                               test, docs, performance, other, or null
"""

import json
import re
import sys

# ─── Keyword table (token → weight) ──────────────────────────────────────────

KEYWORDS: dict[str, dict[str, int]] = {
    "bug": {
        "fix": 2, "bug": 2, "error": 2, "crash": 2, "fail": 2,
        "broken": 1, "exception": 2, "issue": 1,
    },
    "feature": {
        "add": 1, "build": 2, "implement": 2, "create": 1, "feature": 2,
    },
    "refactor": {
        "refactor": 3, "restructure": 2, "cleanup": 2,
        "reorganize": 2, "improve": 1,
    },
    "test": {
        "test": 2, "coverage": 2, "spec": 2, "unit": 2, "integration": 2,
    },
    "docs": {
        "explain": 1, "document": 1, "describe": 1, "why": 1,
    },
    "performance": {
        "optimize": 2, "slow": 2, "performance": 3, "bottleneck": 2,
        "lag": 1, "latency": 2, "speed": 1, "fast": 1,
    },
}


# ─── Keyword scoring ──────────────────────────────────────────────────────────

def _word_match(text: str, word: str) -> bool:
    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def keyword_classify(prompt: str) -> tuple[str, int, int]:
    """Return (best_category, best_score, second_score)."""
    scores = {k: 0 for k in KEYWORDS}
    for task_type, kws in KEYWORDS.items():
        for kw, weight in kws.items():
            if _word_match(prompt, kw):
                scores[task_type] += weight

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    return best, best_score, second_score


# ─── LLM fallback ────────────────────────────────────────────────────────────

def llm_classify(prompt: str, api_key: str | None) -> str | None:
    """Call Claude Haiku for classification when keyword scoring is ambiguous."""
    # Guard: never attempt the API call without a key
    if not api_key:
        return None
    try:
        from anthropic import Anthropic  # lazy import — only needed when LLM is used

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=(
                "You are a task classifier for a coding assistant. "
                "Classify the user's prompt into exactly one of: "
                "bug, feature, refactor, test, docs, performance, other. "
                "Respond with only the category name in lowercase. No explanation."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text.strip().lower()
        if result in KEYWORDS or result == "other":
            return result
        return "other"
    except Exception as e:
        print(f"LLM_CLASSIFY_ERROR: {e}", file=sys.stderr)
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

def classify(
    prompt: str,
    use_llm: bool = True,
    api_key: str | None = None,
) -> str | None:
    """Classify a prompt.

    Zero-config mode (no API key): keyword-only scoring — always returns a result.
    Enhanced mode (API key present): LLM fallback when keyword scoring is ambiguous.
    """
    best, best_score, second_score = keyword_classify(prompt)

    # Confident keyword result — skip the LLM regardless of key availability
    if best_score > 0 and (best_score >= 3 or best_score > 2 * second_score):
        return best

    # No API key (null, empty, or disabled) → graceful degradation: return keyword result
    if not use_llm or not api_key:
        return best if best_score > 0 else None

    # Ambiguous and key is available — escalate to LLM
    llm_result = llm_classify(prompt, api_key)
    if llm_result:
        return llm_result

    # LLM failed or returned unexpected value → fall back to keyword result
    return best if best_score > 0 else None


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing JSON argument"}), file=sys.stderr)
        sys.exit(1)

    data = json.loads(sys.argv[1])
    prompt   = data["prompt"]
    use_llm  = data.get("useLLM", True)
    api_key  = data.get("apiKey")

    result = classify(prompt, use_llm, api_key)
    print(json.dumps({"type": result}))
