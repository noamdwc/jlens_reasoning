from __future__ import annotations

from experiments.flenqa_length_drift.preflight import (
    pad_to_token_count,
    run_preflight,
)


class WordTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[list[int]]]:
        self.calls.append(kwargs)
        count = len(text.split())
        return {"input_ids": [list(range(count))]}


def test_preflight_padding_targets_exact_token_counts_not_words() -> None:
    tokenizer = WordTokenizer()

    for target in (250, 1000, 3000):
        padded = pad_to_token_count(
            "original prompt",
            filler="padding",
            target_tokens=target,
            tokenizer=tokenizer,
        )
        assert padded.endswith("original prompt")
        assert len(padded.split()) == target

    assert all(call["truncation"] is False for call in tokenizer.calls)


def test_preflight_padding_rejects_oversized_prompt() -> None:
    tokenizer = WordTokenizer()

    try:
        pad_to_token_count(
            "already too long",
            filler="padding",
            target_tokens=2,
            tokenizer=tokenizer,
        )
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("oversized preflight prompt was accepted")


def test_run_preflight_requires_jacobian_to_beat_logit_at_every_length() -> None:
    tokenizer = WordTokenizer()

    passed = run_preflight(
        prompt="spider case",
        filler="padding",
        target_tokens=(250, 1000, 3000),
        tokenizer=tokenizer,
        evaluate=lambda prompt: (1, 3),
    )

    assert passed.passed is True
    assert [result.actual_tokens for result in passed.results] == [250, 1000, 3000]

    failed = run_preflight(
        prompt="spider case",
        filler="padding",
        target_tokens=(250, 1000, 3000),
        tokenizer=tokenizer,
        evaluate=lambda prompt: (4, 3) if len(prompt.split()) == 3000 else (1, 3),
    )
    assert failed.passed is False
