"""MCP server for engram — persistent semantic memory for AI agents.

Usage:
    python server.py
    # or with the mcp CLI:
    mcp run server.py
"""
from mcp.server.fastmcp import FastMCP
from memory import (
    DB_PATH, init_db, retrieve_facts, store_fact,
    memory_release, get_history, get_graph, consolidate_memories,
    store_slot_fill, retrieve_slot_fills,
)

try:
    init_db()
except Exception:
    pass  # schema already exists, migration warnings are harmless
mcp = FastMCP("engram", instructions="Long-term memory for AI agents. Stores and retrieves facts, decisions, preferences, and conversation context.")


@mcp.tool()
def search(project_id: str, query: str, top_n: int = 3) -> str:
    """Search memory for facts matching the query.
    
    Returns ranked facts with content, type, and relevance scores.
    """
    result = retrieve_facts(project_id, "", query, top_n=top_n, threshold=0.0, include_budget_info=True)
    facts = result if isinstance(result, list) else result.get("facts", [])
    if not facts:
        return "No matching facts found."
    lines = []
    for f in facts[:top_n]:
        lines.append(f"[{f.get('fact_type','note')}] {f.get('content','')} (score={f.get('score',0):.3f})")
    return "\n".join(lines)


@mcp.tool()
def remember(project_id: str, session_id: str, content: str, fact_type: str = "note") -> str:
    """Store a fact in memory.
    
    fact_type: note | decision | preference | finding | snippet | summary
    """
    fid = store_fact(project_id, session_id, content, fact_type)
    return f"Stored as fact #{fid}"


@mcp.tool()
def forget(fact_id: int, session_id: str = "") -> str:
    """Delete a fact from memory by its ID."""
    result = memory_release(fact_id, session_id)
    return f"Fact #{fact_id} released."


@mcp.tool()
def history(fact_id: int) -> str:
    """Show the edit history of a fact."""
    entries = get_history(fact_id)
    if not entries:
        return "No history found."
    return "\n".join(
        f"[{e.get('operation','?')}] {e.get('content','')[:200]}" for e in entries
    )


@mcp.tool()
def graph(project_id: str, query: str, depth: int = 1) -> str:
    """Explore related facts around a topic (graph traversal)."""
    result = get_graph(project_id, query, depth)
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])
    lines = [f"Found {len(nodes)} facts, {len(edges)} relations:"]
    for n in nodes:
        lines.append(f"  #{n['id']}: {n['content'][:120]}")
    for e in edges:
        lines.append(f"  #{e['source']} --[{e.get('relation','')}]--> #{e['target']}")
    return "\n".join(lines)


@mcp.tool()
def consolidate(project_id: str, session_id: str = "") -> str:
    """Merge redundant facts and remove stale ones."""
    result = consolidate_memories(project_id)
    removed = result.get("removed", 0)
    merged = result.get("merged", 0)
    return f"Consolidated: {removed} removed, {merged} merged."


@mcp.tool()
def remember_slot(project_id: str, slot_name: str, value: str) -> str:
    """Store a key-value slot (structured config)."""
    store_slot_fill(project_id, slot_name, value)
    return f"Slot '{slot_name}' = '{value}' stored."


@mcp.tool()
def get_slots(project_id: str) -> str:
    """Retrieve all stored key-value slots for a project."""
    slots = retrieve_slot_fills(project_id)
    if not slots:
        return "No slots found."
    return "\n".join(f"{s['slot_name']} = {s['value']}" for s in slots)


if __name__ == "__main__":
    mcp.run(transport="stdio")
