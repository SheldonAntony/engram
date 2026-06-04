"""Session parsing and document preparation.

Indexing strategy:
- Primary doc: user turns only (keeps embeddings focused, avoids truncation)
- Assistant doc: assistant turns as a separate searchable document (same session ID)
- Preference doc: synthetic document from extracted preference patterns
- Topic doc: synthetic document from extracted nouns/topics for vocabulary bridging
"""

import re
from typing import Optional

# Preference patterns — detect how people express preferences in conversation.
PREFERENCE_PATTERNS = [
    r"i(?:'ve been| have been) having (?:trouble|issues?|problems?) with ([^,\.!?]{5,80})",
    r"i(?:'ve been| have been) feeling ([^,\.!?]{5,60})",
    r"i(?:'ve been| have been) (?:struggling|dealing) with ([^,\.!?]{5,80})",
    r"i(?:'ve been| have been) (?:worried|concerned) about ([^,\.!?]{5,80})",
    r"i(?:'m| am) (?:worried|concerned) about ([^,\.!?]{5,80})",
    r"i prefer ([^,\.!?]{5,60})",
    r"i usually ([^,\.!?]{5,60})",
    r"i(?:'ve been| have been) (?:trying|attempting) to ([^,\.!?]{5,80})",
    r"i(?:'ve been| have been) (?:considering|thinking about) ([^,\.!?]{5,80})",
    r"lately[,\s]+(?:i've been|i have been|i'm|i am) ([^,\.!?]{5,80})",
    r"recently[,\s]+(?:i've been|i have been|i'm|i am) ([^,\.!?]{5,80})",
    r"i(?:'ve been| have been) (?:working on|focused on|interested in) ([^,\.!?]{5,80})",
    r"i want to ([^,\.!?]{5,60})",
    r"i(?:'m| am) looking (?:to|for) ([^,\.!?]{5,60})",
    r"i(?:'m| am) thinking (?:about|of) ([^,\.!?]{5,60})",
    r"i(?:'ve been| have been) (?:noticing|experiencing) ([^,\.!?]{5,80})",
    # Memory/nostalgia patterns
    r"i still remember ([^,\.!?]{5,80})",
    r"i used to ([^,\.!?]{5,60})",
    r"when i was (?:in |at )?(?:high school|college|university|a kid|young) ([^,\.!?]{5,80})",
    r"growing up[,\s]+([^,\.!?]{5,80})",
    r"i(?:'ve| have) always (?:loved|enjoyed|liked|wanted|been) ([^,\.!?]{5,80})",
    r"i don'?t (?:like|enjoy|want|care for) ([^,\.!?]{5,60})",
    r"i(?:'m| am) (?:passionate|excited|enthusiastic) about ([^,\.!?]{5,60})",
    r"i(?:'ve| have) been ([^,\.!?]{5,60}) for (?:years|months|a while|a long time)",
    # Ownership / possession patterns (catches "my photography setup", "my garden")
    r"my ([a-z][^,\.!?]{3,40})",
    # Activity patterns (catches "I grow tomatoes", "I bake cookies")
    r"i (?:grow|bake|cook|make|build|play|practice|collect"
    r"|photograph|paint|write|brew|craft|sew|knit) ([^,\.!?]{3,60})",
    r"i(?:'ve| have) (?:got|set up|installed|bought|built|started) (?:a |an |my )?([^,\.!?]{3,60})",
]

# Topic extraction — pull key nouns/phrases that help bridge vocabulary gaps
_TOPIC_NOUN_RE = re.compile(
    r"\b(?:my |our |the )?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"  # Proper nouns
    r"|\b(?:my |our )([a-z][a-z ]{3,30})\b",  # "my <thing>" possessives
)


def extract_preferences(turns: list) -> list:
    """Extract preference expressions from conversation turns."""
    mentions = []
    for turn in turns:
        if turn.get("role") != "user":
            continue
        text = turn["content"]
        for pat in PREFERENCE_PATTERNS:
            for match in re.findall(pat, text, re.IGNORECASE):
                clean = match.strip().rstrip(".,;!? ")
                if 5 <= len(clean) <= 80:
                    mentions.append(clean)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for m in mentions:
        ml = m.lower()
        if ml not in seen:
            seen.add(ml)
            unique.append(m)
    return unique[:15]


def extract_topics(turns: list) -> list:
    """Extract key topic nouns from user turns for vocabulary bridging."""
    topics = set()
    stop = {
        "I",
        "My",
        "We",
        "You",
        "The",
        "This",
        "That",
        "It",
        "He",
        "She",
        "They",
        "What",
        "How",
        "When",
        "Where",
        "Why",
        "Can",
        "Could",
        "Would",
        "Should",
        "Will",
        "Do",
        "Does",
        "Did",
        "Have",
        "Has",
        "Had",
        "Is",
        "Are",
        "Was",
        "Were",
        "Been",
        "Also",
        "Just",
        "Very",
        "Really",
        "Actually",
        "Well",
        "Sure",
        "Yes",
        "No",
        "Thanks",
        "Thank",
        "Please",
        "Hello",
        "Hi",
        "Hey",
    }
    for turn in turns:
        if turn.get("role") != "user":
            continue
        text = turn["content"]
        # Extract "my X" possessive phrases
        patt = r"\bmy ([a-z][a-z ]{2,30}?)(?:[,\.!?\n]|\band\b|\bor\b)"
        for match in re.findall(patt, text.lower()):
            clean = match.strip()
            if len(clean) >= 3:
                topics.add(clean)
        # Extract proper nouns
        for match in re.findall(r"\b([A-Z][a-z]{2,15})\b", text):
            if match not in stop:
                topics.add(match)
    return list(topics)[:20]


def is_assistant_reference(question: str) -> bool:
    """Detect questions that ask about what the AI previously said."""
    q = question.lower()
    triggers = [
        "you suggested",
        "you told me",
        "you mentioned",
        "you said",
        "you recommended",
        "remind me what you",
        "you provided",
        "you listed",
        "you gave me",
        "you described",
        "what did you",
        "you came up with",
        "you helped me",
        "you explained",
        "can you remind me",
        "you identified",
        "our previous conversation",
        "our previous chat",
        "our last conversation",
        "follow up on our",
        "going back to our",
        "looking back at our",
    ]
    return any(t in q for t in triggers)


def _format_timestamp_prefix(timestamp: str) -> str:
    """Format a timestamp string as a prefix for document text.

    This embeds the date into the text so both dense and BM25 retrieval
    can match temporal queries (e.g. "When did X happen in July?").
    """
    if not timestamp:
        return ""
    return f"[{timestamp}] "


def _chunk_turns(turns: list, max_turns: int = 6, overlap: int = 1) -> list:
    """Split a turn list into overlapping chunks.

    Long sessions dilute embeddings — a 20-turn session buries individual
    facts. Chunking at ~6 turns keeps embeddings focused while overlap
    preserves conversational context across boundaries.

    Returns list of turn sublists.
    """
    if len(turns) <= max_turns:
        return [turns]

    chunks = []
    start = 0
    while start < len(turns):
        end = min(start + max_turns, len(turns))
        chunks.append(turns[start:end])
        # Advance by at least 1 to avoid infinite loop at the tail
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


def _render_turn(turn: dict, speaker_names: Optional[dict]) -> str:
    """Render a turn as text, prepending the speaker name if provided.

    First-person turns ("I got a PS5") don't contain the speaker's name,
    so entity-attribute queries ("What console does Nate own?") can't
    match. Injecting the name ("Nate: I got a PS5") bridges this gap for
    both dense and BM25 retrieval.
    """
    content = turn["content"]
    if not speaker_names:
        return content
    name = speaker_names.get(turn.get("role"))
    if not name:
        return content
    return f"{name}: {content}"


def session_to_documents(
    session: list,
    session_id: str,
    timestamp: str = "",
    include_assistant: bool = False,
    generate_preference_doc: bool = True,
    generate_assistant_doc: bool = True,
    generate_topic_doc: bool = True,
    chunk_max_turns: int = 6,
    speaker_names: Optional[dict] = None,
) -> list:
    """Convert a conversation session into indexable documents.

    Returns list of dicts with keys: id, text, metadata, is_synthetic.

    Strategy:
    - Chunk long sessions into overlapping segments (~6 turns) so facts
      don't get diluted in long-document embeddings
    - Prepend timestamp to text for temporal retrieval
    - Assistant turns as separate doc (catches assistant-reference questions)
    - Synthetic preference and topic docs for vocabulary bridging
    - Optional speaker_names (e.g. {"user": "Nate"}) prepends the speaker
      to each turn so entity-attribute queries match first-person facts
    """
    docs = []
    ts_prefix = _format_timestamp_prefix(timestamp)

    assistant_text = "\n".join(
        _render_turn(t, speaker_names) for t in session if t.get("role") == "assistant"
    )

    # Primary documents: chunk long sessions into smaller segments
    if include_assistant:
        all_turns = session
    else:
        all_turns = [t for t in session if t.get("role") == "user"]

    chunks = _chunk_turns(all_turns, max_turns=chunk_max_turns)

    for ci, chunk in enumerate(chunks):
        chunk_text = "\n".join(_render_turn(t, speaker_names) for t in chunk)
        if not chunk_text.strip():
            continue

        doc_text = ts_prefix + chunk_text
        chunk_id = session_id if len(chunks) == 1 else f"{session_id}_c{ci}"

        docs.append(
            {
                "id": chunk_id,
                "text": doc_text,
                "metadata": {
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "type": "session",
                },
                "is_synthetic": False,
            }
        )

    # Assistant document: separate searchable doc with assistant turns
    if generate_assistant_doc and assistant_text.strip() and not include_assistant:
        docs.append(
            {
                "id": f"{session_id}_asst",
                "text": ts_prefix + assistant_text,
                "metadata": {
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "type": "assistant",
                },
                "is_synthetic": True,
            }
        )

    # Synthetic preference document
    if generate_preference_doc:
        prefs = extract_preferences(session)
        if prefs:
            speaker_label = (speaker_names or {}).get("user") or "User"
            pref_text = ts_prefix + f"{speaker_label} has mentioned: " + "; ".join(prefs)
            docs.append(
                {
                    "id": f"{session_id}_pref",
                    "text": pref_text,
                    "metadata": {
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "type": "preference",
                    },
                    "is_synthetic": True,
                }
            )

    # Synthetic topic document for vocabulary bridging
    if generate_topic_doc:
        topics = extract_topics(session)
        if topics:
            topic_text = ts_prefix + "Topics discussed: " + ", ".join(topics)
            docs.append(
                {
                    "id": f"{session_id}_topic",
                    "text": topic_text,
                    "metadata": {
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "type": "topic",
                    },
                    "is_synthetic": True,
                }
            )

    return docs
