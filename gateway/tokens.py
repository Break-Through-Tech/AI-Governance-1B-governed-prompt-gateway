"""Token counting shared by every node, selected by optimizer.token_counter in config.yaml.

- estimate: the same heuristic the UI uses before a real tokenizer loads (no dependencies).
- spacy:    spaCy's word/punctuation tokens (linguistic, not what models bill by).
- hf:<repo>: a model's real tokenizer from Hugging Face (needs the `tokenizers` package).
"""
import re
from functools import lru_cache

from gateway import settings

_EST = re.compile(r"(\n+)|( {2,}|\t+)|([A-Za-z]+)|(\d+)|([^\sA-Za-z\d])\5*")


def estimate_tokens(text: str) -> int:
    """Port of estTokens() in website/site/app.js."""
    n = 0
    for m in _EST.finditer(text):
        if m.group(3):
            n += -(-len(m.group(3)) // 8)
        elif m.group(4):
            n += -(-len(m.group(4)) // 3)
        else:
            n += 1
    return n


@lru_cache(maxsize=1)
def _spacy():
    import spacy
    return spacy.blank("en")


@lru_cache(maxsize=4)
def _hf(repo: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained(repo)


def count_tokens(text: str) -> int:
    counter = settings.get("optimizer")["token_counter"]
    if counter == "estimate":
        return estimate_tokens(text)
    if counter == "spacy":
        return sum(1 for t in _spacy().make_doc(text) if not t.is_space)
    if counter.startswith("hf:"):
        return len(_hf(counter[3:]).encode(text, add_special_tokens=False).ids)
    raise ValueError(f"Unknown optimizer.token_counter: {counter!r}")
