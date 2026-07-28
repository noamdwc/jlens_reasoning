from __future__ import annotations

import hashlib

from jlens_reasoning.benchmarks.flenqa_prompts import (
    build_prompt_text,
    compute_prompt_id,
)


def test_pir_prompt_matches_authors_template_byte_for_byte() -> None:
    prompt = build_prompt_text(
        task="PIR",
        question="Is the key in the study?",
        mixin="The key is in the study.\nThe lamp is in the hall.",
        rule=None,
    )

    assert prompt == (
        "The key is in the study.\nThe lamp is in the hall.\n"
        "True/False Question: Is the key in the study?\n"
        "Answer only True or False.\n"
    )


def test_monorel_prompt_matches_authors_template_byte_for_byte() -> None:
    prompt = build_prompt_text(
        task="MonoRel",
        question="Is Ada older than Bea?",
        mixin="Ada is older than Cy.\nCy is older than Bea.",
        rule=None,
    )

    assert prompt == (
        "Here are some facts. Answer the exact following question based on the "
        "text: Is Ada older than Bea? Answer the question as it appears exactly.\n"
        "Ada is older than Cy.\nCy is older than Bea.\n"
        "Is Ada older than Bea?\n"
        "Answer only True or False.\n"
    )


def test_ruletaker_prompt_preserves_raw_rule_typo_and_trailing_newline() -> None:
    prompt = build_prompt_text(
        task="Simplified RuleTaker",
        question="The cow is blue.",
        mixin="The cow is young.\nThe cow is kind.",
        rule=["If someone is young then they are blue."],
    )

    assert prompt == (
        'Answer whether the statement The cow is blue. can be derived from the '
        'rule and the facts. Answer with either "True" or "False".\n'
        "Rule: ['If someone is young then they are blue.']\n"
        "Facts: The cow is young.\nThe cow is kind.\n"
        'Answer with either "True or "False".\n'
    )


def test_prompt_id_is_full_sha256_of_final_text_only() -> None:
    text = "same final prompt\n"

    assert compute_prompt_id(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert len(compute_prompt_id(text)) == 64
