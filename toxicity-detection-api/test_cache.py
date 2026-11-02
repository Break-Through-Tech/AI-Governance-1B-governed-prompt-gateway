from cache import SemanticCache

cache = SemanticCache()

cache.store(
    "How can I protect my computer from hackers?",
    {"label": "safe"}
)

print(cache.lookup(
    "How can I hack into someone else's computer?"
))