#!/usr/bin/env python3
"""LLM council: extract memorable facts from session summaries.

The "council" reviews what happened in a session and decides which facts
are worth storing in long-term memory — 1 to 3 concise, reusable statements.

CLI usage:
    extractor.py '{"text": "...", "apiKey": "sk-..."}'

Returns:
    {"facts": ["fact1", "fact2"]}    — extracted facts (array, may be empty)
    {"facts": []}                    — if text is too short or trivial
"""

import json
import re
import sys


# ─── Keyword-based fallback ───────────────────────────────────────────────────

# Minimum token count before we bother trying to extract anything
_MIN_WORDS = 15

# Markers that hint at something memorable happened
_SIGNAL_PATTERNS = [
    "decided", "chose", "fixed", "solved", "discovered", "realized",
    "switched", "migrated", "refactored", "added", "removed", "implemented",
    "use ", "using ", "framework", "database", "api", "endpoint",
    "auth", "error", "bug", "pattern", "architecture",
]


def keyword_extract(text: str) -> list[str]:
    """Very cheap extraction: return the text itself if it looks signal-rich."""
    words = text.split()
    if len(words) < _MIN_WORDS:
        return []
    lower = text.lower()
    hits = sum(1 for p in _SIGNAL_PATTERNS if p in lower)
    if hits < 2:
        return []
    # Trim to a single concise sentence if very long
    sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
    return [sentences[0][:200]] if sentences else []


# ─── Phase 3: spaCy entity + identifier extraction ───────────────────────────

_nlp = None
_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"}
# Matches camelCase, PascalCase, snake_case, and ALL_CAPS identifiers
_IDENTIFIER_RE = re.compile(
    r'\b([A-Z][a-z]+[A-Z]\w*|[a-z]{2,}_[a-z]\w*|[A-Z]{2,}\w*)\b'
)


def _get_nlp():
    """Lazy-load spaCy en_core_web_sm. Returns None if not installed."""
    global _nlp
    if _nlp is None:
        try:
            import spacy  # noqa: PLC0415
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = False  # sentinel: skip retry on subsequent calls
    return _nlp if _nlp else None


def extract_entities(text: str, max_chars: int = 2000) -> list[str]:
    """Extract named entities and code identifiers from text.

    Uses spaCy labels PERSON/ORG/GPE/PRODUCT/WORK_OF_ART (DATE excluded —
    relative dates cause false positives across unrelated facts) plus regex
    for snake_case/camelCase/PascalCase identifiers. Returns deduplicated
    list of strings with length > 2.
    """
    text = text[:max_chars]
    seen: set[str] = set()
    results: list[str] = []

    nlp = _get_nlp()
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in _ENTITY_LABELS:
                    val = ent.text.strip()
                    if len(val) > 2 and val not in seen:
                        seen.add(val)
                        results.append(val)
        except Exception:
            pass

    for m in _IDENTIFIER_RE.finditer(text):
        val = m.group(1)
        if len(val) > 2 and val not in seen:
            seen.add(val)
            results.append(val)

    return results


def _warmup_nlp() -> None:
    """Pre-load the spaCy model at server startup to avoid first-call latency."""
    _get_nlp()


# ─── LLM council ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a memory curator for a software developer's AI coding assistant.
Given a summary of what happened during a coding session, extract 1-3 concise
facts that are worth remembering for future sessions.

Rules:
- Each fact must be a single sentence, max 120 characters.
- Only include facts that would be useful context in a future unrelated session.
- Skip implementation details that won't matter next time.
- Skip questions the user asked. Focus on decisions, findings, and outcomes.
- If nothing is worth remembering, return an empty list.

Respond with a JSON object: {"facts": ["...", "..."]}
Do not add any text outside the JSON."""


def llm_extract(text: str, api_key: str | None) -> list[str]:
    """Ask Claude Haiku to council what facts deserve long-term storage."""
    if not api_key or len(text.split()) < _MIN_WORDS:
        return keyword_extract(text)
    try:
        from anthropic import Anthropic  # lazy import

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text[:2000]}],
        )
        raw = response.content[0].text.strip()
        parsed = json.loads(raw)
        facts = parsed.get("facts", [])
        # Sanitize: only strings, max 3, max 160 chars each
        return [str(f)[:160] for f in facts if isinstance(f, str)][:3]
    except Exception:
        # Council unavailable — fall back to keyword extraction
        return keyword_extract(text)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: extractor.py '{\"text\": \"...\", \"apiKey\": \"...\"}'"}))
        sys.exit(1)

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"JSON parse error: {exc}"}))
        sys.exit(1)

    text    = payload.get("text", "").strip()
    api_key = payload.get("apiKey") or None

    facts = llm_extract(text, api_key)
    print(json.dumps({"facts": facts}))


if __name__ == "__main__":
    main()
