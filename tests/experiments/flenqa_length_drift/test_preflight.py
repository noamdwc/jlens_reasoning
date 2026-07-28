from __future__ import annotations

import torch

from experiments.flenqa_length_drift.preflight import (
    evaluate_lens_preflight,
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

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [2] if text == " spider" else [8, 9]


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


def test_lens_preflight_uses_identical_untruncated_ids_and_best_token_rank() -> None:
    tokenizer = WordTokenizer()

    class Runner:
        def __init__(self, target_logit: float) -> None:
            self.target_logit = target_logit
            self.positions = None

        def run(self, prompt, *, positions, max_seq_len):
            self.positions = tuple(positions)
            ids = tokenizer(prompt, truncation=False)["input_ids"]
            logits = torch.arange(10, dtype=torch.float32).repeat(len(positions), 1)
            logits[:, 2] = self.target_logit
            return type(
                "Result",
                (),
                {
                    "input_ids": ids,
                    "model_logits": torch.zeros(len(positions), 10),
                    "logits_by_layer": {3: logits},
                },
            )()

    jacobian = Runner(20)
    logit = Runner(2)

    ranks = evaluate_lens_preflight(
        "one two three",
        tokenizer=tokenizer,
        jacobian_runner=jacobian,
        logit_runner=logit,
        target_surfaces=(" spider", "two tokens"),
    )

    assert ranks == (1, 8)
    assert jacobian.positions == logit.positions == (0, 1, 2)
