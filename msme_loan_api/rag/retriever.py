from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


class Retriever(Protocol):
    def index_documents(self, documents: list[str], metadatas: list[dict]) -> int:
        ...

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...


class TfidfRetriever:
    """Lightweight local retriever that avoids external vector DB setup for MVP."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.matrix = None

    def index_documents(self, documents: list[str], metadatas: list[dict]) -> int:
        self.documents = documents
        self.metadatas = metadatas
        if not documents:
            self.matrix = None
            return 0
        self.matrix = self.vectorizer.fit_transform(documents)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.documents or self.matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        if scores.size == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievedChunk] = []

        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(
                    text=self.documents[idx],
                    score=round(score, 4),
                    metadata=self.metadatas[idx],
                )
            )

        return results
