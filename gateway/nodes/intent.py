"""Node 1: classify_intent. Owner: TBD.

Labels what kind of customer-service question the prompt is, for two regulated industries:
insurance (claims, coverage, policies, ...) and retail banking (cards, transfers, loans, ...).
Labels look like "insurance.claims.file_claim" or "banking.card.manage"; anything else is "unknown".
This is *not* the risk check: whether a prompt is safe is node 2's job (screen_risk).

How it works:
  1. Pull out the user's actual question. `original_prompt` often carries a system prompt and policy
     context before a "### Question" header; classifying all of that would make every prompt look
     like "policy question".
  2. Score the question against every intent (how depends on the backend, below).
  3. The top score is `intent_confidence`. Below `intent.min_confidence` the intent is "unknown",
     which route_model treats as needing at least the Medium tier (README §6).

Backends (`intent.backend` in config.yaml):
  - finetuned: a roberta-base classifier fine-tuned on Bitext Insurance + Retail Banking by
               "training scripts/train_intent.py", downloaded from `intent.finetuned_model` (a Hugging
               Face repo or a local folder). Scores = the model's softmax probabilities. It can also
               have a trained "unknown" class (from CLINC150); when that wins, the intent is "unknown".
               Loaded on first call.
  - embedding: no training. Cosine similarity between the question and the example questions in
               config.yaml (`intent.intents.<name>.examples`), using `intent.embedding_model`.
               An intent's score is its best-matching example. Loaded on first call.
  - lexical:   like embedding, but compares shared words. No dependencies; works offline and in CI.
"""
import math
import re
from collections import Counter
from functools import lru_cache

from gateway import settings
from gateway.nodes._base import decision, safe_node
from gateway.tokens import count_tokens, estimate_tokens

UNKNOWN = "unknown"

_QUESTION_HEADER = re.compile(r"^\s*#+\s*(?:user\s+)?question\s*:?\s*$", re.I | re.M)
_WORD = re.compile(r"[a-z]+")
_STOPWORDS = frozenset("""
    a about all am an and any are as at be been being but by can could did do does doing for from had has
    have having he her here hers him his how i if in into is it its just me might my myself no nor of
    off on once only or our ours out over own please so some such than that the their them then there
    these they this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours
    um uh hi hello hey thanks thank basically just wondering so much advance trouble kind kindly ok okay
    really also maybe need want like know get got going
""".split())


def extract_question(prompt: str) -> str:
    """The text after the last "### Question" header, or the whole prompt if there is none."""
    headers = list(_QUESTION_HEADER.finditer(prompt))
    if headers:
        question = prompt[headers[-1].end():].strip()
        if question:
            return question
    return prompt.strip()


# ---------------------------------------------------------------------------------------------
# Lexical backend
# ---------------------------------------------------------------------------------------------
def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _bag(text: str) -> Counter:
    return Counter(_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS)


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def _lexical_scores(question: str, intents: dict) -> dict[str, float]:
    q = _bag(question)
    return {name: max((_cosine(q, _bag(ex)) for ex in spec["examples"]), default=0.0)
            for name, spec in intents.items()}


# ---------------------------------------------------------------------------------------------
# Embedding backend (loaded lazily, never at import time)
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _example_matrix(model_name: str, examples: tuple[tuple[str, str], ...]):
    """Embeds every (intent, example) pair once per model + example set."""
    vectors = _encoder(model_name).encode([ex for _, ex in examples], normalize_embeddings=True)
    return [name for name, _ in examples], vectors


def _embedding_scores(question: str, intents: dict, model_name: str) -> dict[str, float]:
    pairs = tuple((name, ex) for name, spec in intents.items() for ex in spec["examples"])
    labels, matrix = _example_matrix(model_name, pairs)
    q = _encoder(model_name).encode([question], normalize_embeddings=True)[0]
    sims = matrix @ q
    scores = {name: 0.0 for name in intents}
    for label, sim in zip(labels, sims):
        scores[label] = max(scores[label], float(sim))
    return scores


# ---------------------------------------------------------------------------------------------
# Fine-tuned backend (loaded lazily, never at import time)
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _finetuned(model_name: str, max_length: int):
    """Loads the fine-tuned classifier once. Returns predict(texts) -> (probabilities, labels)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    @torch.no_grad()
    def predict(texts):
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        return torch.softmax(model(**enc).logits, dim=-1).numpy(), labels
    return predict


def _finetuned_scores(question: str, cfg: dict) -> dict[str, float]:
    probs, labels = _finetuned(cfg["finetuned_model"], cfg["finetuned_max_length"])([question])
    return {label: float(p) for label, p in zip(labels, probs[0])}


# ---------------------------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------------------------
def score_intents(question: str) -> dict[str, float]:
    """A 0..1 score for each intent: similarity to its examples, or the fine-tuned model's probability."""
    cfg = settings.get("intent")
    if cfg["backend"] == "finetuned":
        scores = _finetuned_scores(question, cfg)
    elif cfg["backend"] == "lexical":
        scores = _lexical_scores(question, cfg["intents"])
    elif cfg["backend"] == "embedding":
        scores = _embedding_scores(question, cfg["intents"], cfg["embedding_model"])
    else:
        raise ValueError(f"Unknown intent.backend: {cfg['backend']!r}")
    return {k: min(max(v, 0.0), 1.0) for k, v in scores.items()}


def _fallback(state):
    return {"intent": UNKNOWN, "intent_confidence": 0.0,
            "original_tokens": estimate_tokens(state.get("original_prompt", ""))}


@safe_node("classify_intent", fallback=_fallback)
def classify_intent(state):
    cfg = settings.get("intent")
    prompt = state["original_prompt"]
    n = count_tokens(prompt)
    question = extract_question(prompt)

    ranked = sorted(score_intents(question).items(), key=lambda kv: kv[1], reverse=True)
    (top, conf), (runner_up, runner_conf) = ranked[0], ranked[1] if len(ranked) > 1 else (None, 0.0)
    conf, runner_conf = round(conf, 3), round(runner_conf, 3)

    if top == UNKNOWN:                    # a fine-tuned model with a trained "unknown" class chose it
        intent = UNKNOWN
        reason = f"model predicted unknown ({conf}); runner-up {runner_up} ({runner_conf})"
    elif conf < cfg["min_confidence"]:
        intent = UNKNOWN
        reason = f"best match {top} ({conf}) < min_confidence {cfg['min_confidence']}"
    else:
        intent = top
        reason = f"closest to {top} examples ({conf}); runner-up {runner_up} ({runner_conf})"

    return {
        "intent": intent,
        "intent_confidence": conf,
        "original_tokens": n,
        "decision_log": [decision(intent, reason, backend=cfg["backend"], intent_confidence=conf,
                                  runner_up=runner_up, runner_up_confidence=runner_conf,
                                  question_chars=len(question),
                                  original_tokens=n)],
    }
