"""Engram CLI — local-first, cloud-ready memory retrieval."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def cmd_init(args):
    """Initialize an Engram memory store."""
    from .backends.faiss_backend import FaissBackend

    path = Path(args.path).resolve()
    backend = FaissBackend(path=path, dimension=1024)
    print(f"Initialized Engram store at {path}")
    print("  Backend: FAISS (local)")
    print(f"  Documents: {backend.count()}")


def cmd_ingest(args):
    """Ingest conversation files into the memory store."""
    from .backends.base import Document
    from .backends.faiss_backend import FaissBackend
    from .ingestion.parser import session_to_documents
    from .retrieval.embedder import Embedder

    store_path = Path(args.store).resolve()
    backend = FaissBackend(path=store_path, dimension=1024)
    embedder = Embedder(args.embed_model)

    input_path = Path(args.input)
    if input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
    else:
        print(f"Unsupported file format: {input_path.suffix}")
        sys.exit(1)

    # Parse sessions — supports two formats:
    # 1. List of {id, timestamp, turns} objects
    # 2. List of turn-lists (raw format)
    all_docs = []
    if isinstance(data, list):
        for i, session in enumerate(data):
            if isinstance(session, dict) and "turns" in session:
                parsed = session_to_documents(
                    session=session["turns"],
                    session_id=session.get("id", f"session_{i}"),
                    timestamp=session.get("timestamp", ""),
                    include_assistant=True,
                )
                all_docs.extend(parsed)
            elif isinstance(session, list):
                parsed = session_to_documents(
                    session=session,
                    session_id=f"session_{i}",
                    include_assistant=True,
                )
                all_docs.extend(parsed)

    if not all_docs:
        print("No documents to ingest.")
        return

    # Embed
    print(f"Embedding {len(all_docs)} documents with {args.embed_model}...")
    texts = [d["text"] for d in all_docs]
    embeddings = embedder.encode_documents(texts)

    # Store
    documents = []
    for i, doc_info in enumerate(all_docs):
        documents.append(
            Document(
                id=doc_info["id"],
                text=doc_info["text"],
                embedding=embeddings[i].tolist(),
                metadata=doc_info["metadata"],
            )
        )

    backend.add(documents)
    print(f"Ingested {len(documents)} documents. Total: {backend.count()}")


def cmd_search(args):
    """Search the memory store."""
    from .backends.faiss_backend import FaissBackend
    from .retrieval.embedder import Embedder

    store_path = Path(args.store).resolve()
    backend = FaissBackend(path=store_path)
    embedder = Embedder(args.embed_model)

    query_vec = embedder.encode_query(args.query)
    results = backend.query(
        embedding=query_vec.tolist(),
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if not results:
        print(f'No relevant results for: "{args.query}"')
        return

    print(f'\nResults for: "{args.query}"\n{"=" * 60}')
    for i, doc in enumerate(results, 1):
        sim = round(doc.score, 3)
        meta = doc.metadata or {}
        print(f"\n  [{i}] Score: {sim}")
        print(f"      Session: {meta.get('session_id', '?')}")
        print(f"      Time: {meta.get('timestamp', '?')}")
        preview = doc.text[:200].replace("\n", " ")
        print(f"      {preview}...")
        print(f"  {'─' * 56}")


def cmd_info(args):
    """Show store info."""
    from .backends.faiss_backend import FaissBackend

    store_path = Path(args.store).resolve()
    backend = FaissBackend(path=store_path)
    print(f"Engram store: {store_path}")
    print(f"  Documents: {backend.count()}")
    print("  Backend: FAISS (local)")


def main():
    parser = argparse.ArgumentParser(
        description=f"Engram v{__version__} — high-recall memory retrieval",
    )
    parser.add_argument("--version", action="version", version=f"engram {__version__}")

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Initialize a memory store")
    p_init.add_argument("path", help="Directory for the store")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest conversations")
    p_ingest.add_argument("input", help="Input file (JSON)")
    p_ingest.add_argument("--store", default="./engram_store", help="Store path")
    p_ingest.add_argument("--embed-model", default="bge-large", help="Embedding model")

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--store", default="./engram_store", help="Store path")
    p_search.add_argument("--embed-model", default="bge-large", help="Embedding model")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search.add_argument(
        "--min-score",
        type=float,
        default=0.45,
        help="Minimum similarity score (0.0-1.0, default: 0.45)",
    )

    # info
    p_info = sub.add_parser("info", help="Show store info")
    p_info.add_argument("--store", default="./engram_store", help="Store path")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "ingest": cmd_ingest,
        "search": cmd_search,
        "info": cmd_info,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
