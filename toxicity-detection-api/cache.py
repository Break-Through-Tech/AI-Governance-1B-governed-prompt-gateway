import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCache:
    def __init__(self, threshold=0.90):
        # Load the embedding model
        self.encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Store query embeddings
        self.index = faiss.IndexFlatIP(384)

        # Store original queries and their results
        self.queries = []
        self.results = []

        # Minimum similarity required for a cache hit
        self.threshold = threshold

    def get_embedding(self, query):
        embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True
        )

        return np.asarray(embedding, dtype="float32")

    def lookup(self, query):
        if self.index.ntotal == 0:
            return None

        embedding = self.get_embedding(query)

        scores, indices = self.index.search(embedding, 1)

        similarity = float(scores[0][0])
        match_index = int(indices[0][0])

        if (
            similarity >= self.threshold
            and 0 <= match_index < len(self.results)
        ):
            return self.results[match_index]

        return None

    def store(self, query, result):
        embedding = self.get_embedding(query)

        self.index.add(embedding)

        self.queries.append(query)
        self.results.append(result)