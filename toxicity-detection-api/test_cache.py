"""Manual smoke check for the standalone semantic-cache API."""
from cache import SemanticCache


if __name__ == "__main__":
    cache = SemanticCache()
    cache.store("How do I file a home insurance claim?", {"label": "safe"})
    print(cache.lookup_with_score("How do I submit a home insurance claim?"))
