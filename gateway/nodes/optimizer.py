"""Nodes 4–5: optimize_prompt and verify_fidelity. Owner: Riya.

optimize_prompt is a Python port of the optimizer engine in website/site/app.js: the same seven
techniques, the same rules, run in the same order. verify_fidelity then checks that the shorter
prompt still means the same thing; if it doesn't, the graph retries in "safe" mode, and if that
also fails, the original prompt is used (README §6).
"""
import math
import re
from collections import Counter
from functools import lru_cache

from gateway import settings
from gateway.nodes._base import decision, safe_node
from gateway.tokens import count_tokens, estimate_tokens

# ---------- techniques (always run in this order) ----------
TECHNIQUES = [
    {"id": "whitespace", "name": "Whitespace normalization", "risk": False},
    {"id": "formatting", "name": "Markup stripping", "risk": False},
    {"id": "filler", "name": "Filler phrase removal", "risk": False},
    {"id": "verbose", "name": "Concise rewriting", "risk": False},
    {"id": "dedup", "name": "Deduplication", "risk": False},
    {"id": "prune", "name": "Context pruning", "risk": True},
    {"id": "telegraphic", "name": "Article dropping", "risk": True},
]
TECHNIQUE_IDS = [t["id"] for t in TECHNIQUES]

I, M = re.IGNORECASE, re.MULTILINE
# Python's `re` has no \p{Extended_Pictographic}; these ranges cover the common emoji blocks.
_EMOJI = "[\U0001F000-\U0001FAFF☀-➿⬀-⯿]️?\\s?"

SPAN_RULES = {
    "whitespace": [
        (r"(?<= ) +", "", 0),
        (r"[ \t]+(?=\n)", "", 0),
        (r"^[ \t]+", "", M),          # app.js uses (?<=\n), which misses lines that start a unit
        (r"(?<=\n)\n+", "", 0),
        (r"\t+", " ", 0),
    ],
    "formatting": [
        (r"^#{1,6}[ \t]+", "", M),
        (r"\*\*|__", "", 0),
        (r"`+", "", 0),
        (r"^[ \t]*([-=*_])\1{2,}[ \t]*$", "", M),
        (_EMOJI, "", 0),
        (r"[¶·]{2,}|(?:¶\s*)+", "", 0),
    ],
    "filler": [
        (r"^\s*(?:hi|hello|hey)(?: there)?[!,.]*\s*", "", I | M),
        (r"\b(?:um+|uh+|erm|hmm+)\b[,.]?\s*", "", I),
        (r"\bso basically,?\s*", "", I),
        (r"\bbasically,?\s*", "", I),
        (r"\bI was (?:just )?wondering,?\s*(?:if|whether)?\s*", "", I),
        (r",?\s*(?:if )?(?:it is|it's) not too much trouble,?\s*", "", I),
        (r"\bif you don't mind,?\s*", "", I),
        (r"\s*\b(?:thanks|thank you)(?: so much| very much| a lot)?(?: in advance)?[!.]*", "", I),
        (r"\bI hope (?:this|that) makes sense[.!]*\s*", "", I),
        (r"\b(?:actually|literally|honestly|really|just|kindly),?\s+", "", I),
    ],
    "verbose": [
        (r"\bin order to\b", "to", I),
        (r"\bdue to the fact that\b", "because", I),
        (r"\bat this point in time\b", "now", I),
        (r"\bin the event that\b", "if", I),
        (r"\b(?:is|are) able to\b", "can", I),
        (r"\bhas the ability to\b", "can", I),
        (r"\ba large number of\b", "many", I),
        (r"\bthe majority of\b", "most", I),
        (r"\bwith (?:regard|respect) to\b", "about", I),
        (r"\bfor the purpose of\b", "for", I),
        (r"\bprior to\b", "before", I),
        (r"\bmake sure that\b", "ensure", I),
        (r"\bwhether or not\b", "whether", I),
        (r"\beach and every\b", "every", I),
        (r"\bfirst and foremost\b", "first", I),
        (r"\b(?:it is important to note that|please note that|I would like you to|could you please)\s*", "", I),
        (r"\bwhat do I need to do\b", "what are the steps", I),
    ],
    "telegraphic": [
        (r"\b(?:the|a|an)\s+(?=[A-Za-z§])", "", I),
    ],
}
SPAN_RULES = {k: [(re.compile(p, f), r) for p, r, f in v] for k, v in SPAN_RULES.items()}

STOP = set((
    "a an the and or but if then so of to in on at by for with from as is are was were be been being it its this that "
    "these those i me my you your we our they them their he she his her what which who how when where why do does did "
    "doing done can could will would should must may might shall have has had just also only always very all any some "
    "such into about up out please need want there here than too own same other more most each few both while over "
    "under again further once").split())


def _stem(w: str) -> str:
    return re.sub(r"(?:ing|ed|es|s|e|d)$", "", w, count=1) if len(w) > 4 else w


def keywords(text: str) -> list[str]:
    """Content words (no stopwords), lightly stemmed, with repeats. Used by dedup, prune and similarity."""
    return [_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 1]


# ---------- units: sentences, or whole lines for list items ----------
_BULLET = re.compile(r"^[ \t]*(?:[-*•]|\d+[.)])\s")


def split_units(text: str) -> list[dict]:
    """Port of splitUnits() in app.js. Joining every unit's text gives back the original exactly."""
    units = []
    for line in re.split(r"(?<=\n)", text):
        if not line:
            continue
        if _BULLET.match(line):
            units.append({"text": line, "bullet": True})
            continue
        start = i = 0
        while i < len(line):
            if line[i] in ".!?":
                j = i
                while j + 1 < len(line) and line[j + 1] in ".!?":
                    j += 1
                if j + 1 == len(line) or line[j + 1].isspace():
                    k = j + 1
                    while k < len(line) and line[k].isspace():
                        k += 1
                    units.append({"text": line[start:k], "bullet": False})
                    start = i = k
                    continue
                i = j + 1
                continue
            i += 1
        if start < len(line):
            units.append({"text": line[start:], "bullet": False})
    merged = []
    for u in units:
        if not u["text"].strip() and merged:
            merged[-1]["text"] += u["text"]
        else:
            merged.append(u)
    for u in merged:
        u["removed"] = False
    return merged


def _match_case(src: str, repl: str) -> str:
    return repl[0].upper() + repl[1:] if repl and src[:1].isupper() else repl


def _apply_span(units, rules, changes):
    for u in units:
        if u["removed"]:
            continue
        for pattern, repl in rules:
            def sub(m):
                new = _match_case(m.group(0), repl)
                if m.group(0) and new != m.group(0):
                    changes.append({"from": m.group(0), "to": new})
                return new
            u["text"] = pattern.sub(sub, u["text"])


def _cut(u, changes):
    changes.append({"from": u["text"].strip(), "to": ""})
    u["text"] = re.search(r"\n*$", u["text"]).group(0)   # keep the line break so lines don't merge
    u["removed"] = True


def _apply_dedup(units, changes):
    seen = []
    for u in units:
        if u["removed"]:
            continue
        kw = set(keywords(u["text"]))
        if len(kw) < 2:
            continue
        dup = any(len(kw & prev) / min(len(kw), len(prev)) >= 0.6 and min(len(kw), len(prev)) >= 3 for prev in seen)
        if dup:
            _cut(u, changes)
        else:
            seen.append(kw)


def question_unit(units):
    """The user's question: the last unit containing '?', else the last non-empty unit."""
    live = [u for u in units if not u["removed"] and u["text"].strip()]
    for u in reversed(live):
        if "?" in u["text"]:
            return u
    return live[-1] if live else None


def _apply_prune(units, changes):
    q = question_unit(units)
    if q is None:
        return
    qk = set(keywords(q["text"]))
    if len(qk) < 2:
        return
    for u in units:
        if u is q or u["removed"] or not u["bullet"] or not u["text"].strip():
            continue
        if not set(keywords(u["text"])) & qk:
            _cut(u, changes)


def _render(units, tidy: bool) -> str:
    text = "".join(u["text"] for u in units)
    if tidy:   # cuts leave line breaks and trailing spaces behind; clean them when whitespace cleanup is on
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def optimize_text(text: str, technique_ids) -> tuple[str, list[dict]]:
    """Run the chosen techniques (in the fixed TECHNIQUES order). Returns (optimized text, per-step report)."""
    unknown = set(technique_ids) - set(TECHNIQUE_IDS)
    if unknown:
        raise ValueError(f"unknown optimizer techniques: {sorted(unknown)}")
    units = split_units(text)
    tidy = "whitespace" in technique_ids
    steps, before = [], count_tokens(text)
    for tech in TECHNIQUES:
        if tech["id"] not in technique_ids:
            continue
        changes = []
        if tech["id"] in SPAN_RULES:
            _apply_span(units, SPAN_RULES[tech["id"]], changes)
        elif tech["id"] == "dedup":
            _apply_dedup(units, changes)
        elif tech["id"] == "prune":
            _apply_prune(units, changes)
        after = count_tokens(_render(units, tidy))
        steps.append({"id": tech["id"], "name": tech["name"], "risk": tech["risk"],
                      "tokens_before": before, "tokens_after": after, "changes": changes})
        before = after
    return _render(units, tidy), steps


# ---------- node 4: optimize_prompt ----------
def _optimize_fallback(state):
    # Fail open: the original prompt goes forward unchanged.
    prompt = state.get("original_prompt", "")
    return {"optimized_prompt": prompt, "optimized_tokens": estimate_tokens(prompt),
            "techniques_applied": [], "optimization_attempts": state.get("optimization_attempts", 0) + 1}


@safe_node("optimize_prompt", fallback=_optimize_fallback)
def optimize_prompt(state):
    cfg = settings.get("optimizer")
    mode = state.get("optimizer_mode", "full")
    original = state["original_prompt"]
    optimized, steps = optimize_text(original, cfg["techniques"][mode])
    before = state.get("original_tokens") or count_tokens(original)
    after = count_tokens(optimized)
    pct = round((before - after) / before * 100, 1) if before else 0.0
    used = [s["id"] for s in steps if s["tokens_after"] < s["tokens_before"] or s["changes"]]
    return {
        "optimizer_mode": mode,
        "optimized_prompt": optimized,
        "optimized_tokens": after,
        "techniques_applied": steps,
        "optimization_attempts": state.get("optimization_attempts", 0) + 1,
        "decision_log": [decision(f"{mode} mode", f"{before} → {after} tokens (−{pct}%)",
                                  tokens_before=before, tokens_after=after, reduction_pct=pct,
                                  techniques_with_changes=used)],
    }


# ---------- node 5: verify_fidelity ----------
_ENTITY = re.compile(r"(?:[A-Za-z]+-)?[A-Za-z$€£§#]*\d[\w$%.,/:#§-]*")


def entities(text: str) -> set[str]:
    """Numbers, amounts, dates, section and policy IDs: anything that contains a digit."""
    return {e.rstrip(".,:;-").lower() for e in _ENTITY.findall(text)}


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


@lru_cache(maxsize=1)
def _encoder(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def similarity(original: str, optimized: str) -> float:
    cfg = settings.get("optimizer")["fidelity"]
    if not optimized.strip():
        return 0.0
    if cfg["similarity_backend"] == "lexical":
        return round(_cosine(Counter(keywords(original)), Counter(keywords(optimized))), 4)
    if cfg["similarity_backend"] == "embedding":
        a, b = _encoder(cfg["embedding_model"]).encode([original, optimized], normalize_embeddings=True)
        return round(float(a @ b), 4)
    raise ValueError(f"unknown similarity_backend: {cfg['similarity_backend']!r}")


def check_fidelity(original: str, optimized: str) -> dict:
    """Run every fidelity check. `passed` is True only if all of them pass."""
    cfg = settings.get("optimizer")["fidelity"]
    sim = similarity(original, optimized)
    orig_ents, opt_ents = entities(original), entities(optimized)
    missing = sorted(orig_ents - opt_ents)
    recall = 1.0 if not orig_ents else round(len(orig_ents & opt_ents) / len(orig_ents), 4)
    q = question_unit(split_units(original))
    missing_q = sorted(entities(q["text"]) - opt_ents) if q else []
    checks = {
        "similarity": sim,
        "similarity_backend": cfg["similarity_backend"],
        "similarity_ok": sim >= cfg["min_similarity"],
        "entity_recall": recall,
        "missing_entities": missing,
        "entity_recall_ok": recall >= cfg["min_entity_recall"],
        "missing_question_entities": missing_q,
        "question_ok": not missing_q or not cfg["require_question_entities"],
    }
    checks["passed"] = checks["similarity_ok"] and checks["entity_recall_ok"] and checks["question_ok"]
    return checks


def _verify_fallback(state):
    # Can't confirm the meaning survived, so don't trust the compression.
    return {"fidelity_passed": False, "fidelity_score": 0.0, "optimizer_mode": "safe",
            "prompt_for_model": state.get("original_prompt", ""),
            "model_input_tokens": state.get("original_tokens") or estimate_tokens(state.get("original_prompt", ""))}


@safe_node("verify_fidelity", fallback=_verify_fallback)
def verify_fidelity(state):
    original, optimized = state["original_prompt"], state["optimized_prompt"]
    checks = check_fidelity(original, optimized)
    out = {"fidelity_score": checks["similarity"], "fidelity_passed": checks["passed"], "fidelity_checks": checks}
    if checks["passed"]:
        out.update(prompt_for_model=optimized, model_input_tokens=state["optimized_tokens"])
        return {**out, "decision_log": [decision("pass", "optimized prompt keeps the original meaning",
                                                 **_summary(checks))]}

    failed = [name for name in ("similarity_ok", "entity_recall_ok", "question_ok") if not checks[name]]
    out.update(prompt_for_model=original,
               model_input_tokens=state.get("original_tokens") or count_tokens(original))
    retry = state.get("optimization_attempts", 1) < settings.get("optimizer")["max_attempts"]
    if retry:
        out["optimizer_mode"] = "safe"
    verdict = "fail → retry in safe mode" if retry else "fail → using original prompt"
    return {**out, "decision_log": [decision(verdict, f"failed checks: {', '.join(failed)}", **_summary(checks))]}


def _summary(checks):
    return {k: checks[k] for k in ("similarity", "entity_recall", "missing_entities", "missing_question_entities")}
