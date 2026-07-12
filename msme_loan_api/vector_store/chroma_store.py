import hashlib
from pathlib import Path

from rag.retriever import RetrievedChunk


class ChromaRetriever:
    """Persistent local ChromaDB retriever with a TF-IDF style interface."""

    def __init__(self, collection_name: str, persist_dir: Path) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Install requirements or switch RAG_BACKEND=tfidf."
            ) from exc

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def index_documents(self, documents: list[str], metadatas: list[dict]) -> int:
        if not documents:
            return 0

        self._collection.delete(where={"type": metadatas[0].get("type", "unknown")})

        ids = []
        for idx, text in enumerate(documents):
            digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
            ids.append(f"{metadatas[idx].get('type', 'doc')}_{idx}_{digest}")

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(documents)

    def search(self, query: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[RetrievedChunk]:
        query_args = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            query_args["where"] = metadata_filter

        result = self._collection.query(**query_args)

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        output = []
        for text, meta, distance in zip(docs, metas, distances):
            score = max(0.0, 1.0 - float(distance))
            output.append(RetrievedChunk(text=text, score=round(score, 4), metadata=meta or {}))

        return output
