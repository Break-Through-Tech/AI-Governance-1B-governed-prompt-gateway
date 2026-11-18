from threading import RLock

import faiss
import numpy as np


class SemanticCache:
    def __init__(self, threshold=0.90):
        self.encoder = None
        self.index = faiss.IndexFlatIP(384)
        self.queries = []
        self.results = []
        self.scopes = []
        self.threshold = threshold
        self._lock = RLock()

    def get_embedding(self, query):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = self.encoder.encode([query], normalize_embeddings=True)
        return np.asarray(embedding, dtype="float32")

    def lookup(self, query):
        """Preserve the standalone cache's result-only API."""
        result, _ = self.lookup_with_score(query)
        return result

    def lookup_with_score(self, query, *, scope=None, threshold=None):
        """Return the best eligible match and its cosine score, even below threshold."""
        with self._lock:
            if scope not in self.scopes:
                return None, None
            embedding = self.get_embedding(query)
            scores, indices = self.index.search(embedding, self.index.ntotal)
            cutoff = self.threshold if threshold is None else threshold
            for score, index in zip(scores[0], indices[0]):
                if index >= 0 and self.scopes[index] == scope and np.isfinite(score):
                    similarity = max(-1.0, min(1.0, float(score)))
                    return (self.results[index] if similarity >= cutoff else None), similarity
            return None, None

    def store(self, query, result, *, scope=None):
        with self._lock:
            embedding = self.get_embedding(query)
            self.index.add(embedding)
            self.queries.append(query)
            self.results.append(result)
            self.scopes.append(scope)

    def clear(self):
        with self._lock:
            self.index.reset()
            self.queries.clear()
            self.results.clear()
            self.scopes.clear()
