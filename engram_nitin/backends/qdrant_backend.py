"""Qdrant-backed cloud vector store."""

from __future__ import annotations

from typing import List, Optional

from .base import Document, VectorBackend


class QdrantBackend(VectorBackend):
    """Cloud vector storage using Qdrant.

    Supports both Qdrant Cloud (managed) and self-hosted Qdrant instances.
    Enterprise-ready: handles scaling, filtering, and persistence automatically.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "engram",
        dimension: int = 1024,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("qdrant-client not installed. Run: pip install engram[cloud]")

        self._collection = collection_name
        self._dim = dimension
        self._client = QdrantClient(url=url, api_key=api_key)

        # Create collection if it doesn't exist
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def add(self, docs: List[Document]) -> None:
        if not docs:
            return
        from qdrant_client.models import PointStruct

        points = []
        for doc in docs:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
            payload = {"text": doc.text}
            if doc.metadata:
                payload.update(doc.metadata)
            points.append(
                PointStruct(
                    id=hash(doc.id) & 0xFFFFFFFFFFFFFFFF,  # Qdrant needs int or UUID
                    vector=doc.embedding,
                    payload={"_doc_id": doc.id, **payload},
                )
            )

        self._client.upsert(collection_name=self._collection, points=points)

    def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
    ) -> List[Document]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qfilter = None
        if metadata_filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in metadata_filter.items()
            ]
            qfilter = Filter(must=conditions)

        results = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )

        docs = []
        for point in results.points:
            payload = point.payload or {}
            doc_id = payload.pop("_doc_id", str(point.id))
            text = payload.pop("text", "")
            docs.append(
                Document(
                    id=doc_id,
                    text=text,
                    metadata=payload,
                    score=point.score,
                )
            )
        return docs

    def delete(self, ids: List[str]) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(must=[FieldCondition(key="_doc_id", match=MatchAny(any=ids))]),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def clear(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
        )
