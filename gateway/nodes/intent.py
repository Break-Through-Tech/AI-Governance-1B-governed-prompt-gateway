"""Node 1: classify_intent. Owner: TBD.

Labels what kind of customer-service question the prompt is, for two regulated industries:
insurance (claims, coverage, policies, ...) and retail banking (cards, transfers, loans, ...).
Labels look like "insurance.claims.file_claim" or "banking.card.manage"; anything else is "unknown".
This is *not* the risk check: whether a prompt is safe is node 2's job (screen_risk).

How it works:
  1. Pull out the user's actual question. `original_prompt` often carries a system prompt and policy
     context before a "### Question" header; classifying all of that would make every prompt look
     like "policy question".
  2. Ask the fine-tuned model for a probability per label. The model is roberta-base fine-tuned on
     Bitext Insurance + Retail Banking by "training scripts/train_intent.py", with a 24th class,
     "unknown", trained on CLINC150. It is downloaded from `intent.finetuned_model` (a Hugging Face
     repo or a local folder) on the first call, never at import time.
  3. The top probability is `intent_confidence`. The intent is "unknown" when the model picks its
     unknown class, or when the top probability is below `intent.min_confidence`. route_model treats
     "unknown" as needing at least the Medium tier (README §6).
"""
import re
from functools import lru_cache

from gateway import settings
from gateway.nodes._base import decision, safe_node
from gateway.tokens import count_tokens, estimate_tokens

UNKNOWN = "unknown"

_QUESTION_HEADER = re.compile(r"^\s*#+\s*(?:user\s+)?question\s*:?\s*$", re.I | re.M)


def extract_question(prompt: str) -> str:
    """The text after the last "### Question" header, or the whole prompt if there is none."""
    headers = list(_QUESTION_HEADER.finditer(prompt))
    if headers:
        question = prompt[headers[-1].end():].strip()
        if question:
            return question
    return prompt.strip()


@lru_cache(maxsize=2)
def _finetuned(model_name: str, max_length: int):
    """Loads the fine-tuned model once. Returns predict(texts) -> (probabilities, labels)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    @torch.no_grad()
    def predict(texts):
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        return torch.softmax(model(**enc).logits, dim=-1).tolist(), labels
    return predict


def score_intents(question: str) -> dict[str, float]:
    """The model's probability for each label (they add up to 1)."""
    cfg = settings.get("intent")
    probs, labels = _finetuned(cfg["finetuned_model"], cfg["finetuned_max_length"])([question])
    return {label: float(p) for label, p in zip(labels, probs[0])}


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

    if top == UNKNOWN:
        intent = UNKNOWN
        reason = f"model predicted unknown ({conf}); runner-up {runner_up} ({runner_conf})"
    elif conf < cfg["min_confidence"]:
        intent = UNKNOWN
        reason = f"best label {top} ({conf}) < min_confidence {cfg['min_confidence']}"
    else:
        intent = top
        reason = f"model predicted {top} ({conf}); runner-up {runner_up} ({runner_conf})"

    return {
        "intent": intent,
        "intent_confidence": conf,
        "original_tokens": n,
        "decision_log": [decision(intent, reason, intent_confidence=conf, runner_up=runner_up,
                                  runner_up_confidence=runner_conf, question_chars=len(question),
                                  original_tokens=n)],
    }
