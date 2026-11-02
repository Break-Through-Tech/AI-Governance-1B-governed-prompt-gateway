"""Unit tests for optimize_prompt / verify_fidelity (no LangGraph needed)."""
import pytest

from gateway import settings
from gateway.graph import SAMPLE_PROMPT
from gateway.nodes.optimizer import (TECHNIQUE_IDS, check_fidelity, entities, optimize_prompt, optimize_text,
                                     split_units, verify_fidelity)


def opt(text, *ids):
    return optimize_text(text, ids)[0]


# ---------- units ----------
def test_split_units_round_trips():
    assert "".join(u["text"] for u in split_units(SAMPLE_PROMPT)) == SAMPLE_PROMPT


def test_bullets_are_whole_units():
    units = split_units("- First. Second.\nPlain one. Plain two.")
    assert [u["bullet"] for u in units] == [True, False, False]


# ---------- each technique ----------
def test_whitespace():
    assert opt("a    b\t\tc  \n\n\n   d", "whitespace") == "a b c\nd"


def test_formatting():
    assert opt("### Title\n**bold** and `code`\n---\nok 🎉 done", "formatting") == "Title\nbold and code\n\nok done"


def test_filler():
    out = opt("Um, so basically I was just wondering if the claim is open? Thanks so much in advance!!", "filler")
    assert out == "the claim is open?"


def test_verbose_keeps_capitalization():
    assert opt("In order to file, make sure that you are able to log in.", "verbose") == "To file, ensure you can log in."


def test_dedup_removes_repeated_instruction():
    out = opt("Never give legal advice. Only answer from policy context. "
              "Remember: never give legal advice, and only answer from policy context.", "dedup")
    assert out == "Never give legal advice. Only answer from policy context."


def test_prune_drops_unrelated_bullets_only():
    text = "- Water damage is covered.\n- Update your mailing address in Profile.\nIs water damage covered?"
    out = opt(text, "whitespace", "prune")
    assert "mailing address" not in out and "Water damage is covered" in out and out.endswith("covered?")


def test_telegraphic():
    assert opt("File the claim within a day.", "telegraphic") == "File claim within day."


def test_unknown_technique_is_rejected():
    with pytest.raises(ValueError):
        optimize_text("hi", ["not_a_technique"])


def test_safe_mode_has_no_risky_techniques():
    safe = settings.get("optimizer")["techniques"]["safe"]
    assert not {"prune", "telegraphic"} & set(safe)
    assert set(safe) <= set(TECHNIQUE_IDS)


def test_sample_prompt_full_vs_safe():
    full, _ = optimize_text(SAMPLE_PROMPT, settings.get("optimizer")["techniques"]["full"])
    safe, steps = optimize_text(SAMPLE_PROMPT, settings.get("optimizer")["techniques"]["safe"])
    assert len(full) < len(safe) < len(SAMPLE_PROMPT)
    assert [s["id"] for s in steps] == ["whitespace", "formatting", "filler", "verbose", "dedup"]
    assert "§9.4" in safe and "§9.4" not in full


# ---------- fidelity ----------
def test_entities():
    assert entities("Home §4.2: pay $1,500 within 60 days (policy HO-12345).") == {"§4.2", "$1,500", "60", "ho-12345"}


def test_unchanged_prompt_passes():
    assert check_fidelity(SAMPLE_PROMPT, SAMPLE_PROMPT)["passed"]


def test_dropped_number_fails():
    c = check_fidelity("Is my $500 deductible waived for claim 77?", "Is my deductible waived for claim 77?")
    assert not c["passed"] and c["missing_entities"] == ["$500"] and c["missing_question_entities"] == ["$500"]


def test_empty_optimized_prompt_fails():
    assert not check_fidelity("Thanks so much!", "")["passed"]


def test_unrelated_rewrite_fails_similarity():
    assert not check_fidelity("Is water damage from a burst pipe covered?", "What is the weather today?")["similarity_ok"]


# ---------- nodes ----------
def test_optimize_node_counts_attempts_and_mode():
    out = optimize_prompt({"original_prompt": SAMPLE_PROMPT, "optimizer_mode": "safe", "optimization_attempts": 1})
    assert out["optimization_attempts"] == 2 and out["optimizer_mode"] == "safe"
    assert out["optimized_tokens"] < 239


def test_verify_node_switches_to_safe_mode_when_retry_left():
    out = verify_fidelity({"original_prompt": "Pay $500 by Friday?", "optimized_prompt": "Pay by Friday?",
                           "optimized_tokens": 4, "original_tokens": 6, "optimization_attempts": 1})
    assert out["fidelity_passed"] is False
    assert out["optimizer_mode"] == "safe"
    assert out["prompt_for_model"] == "Pay $500 by Friday?" and out["model_input_tokens"] == 6
