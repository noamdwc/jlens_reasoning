from __future__ import annotations

import torch

import experiments.flenqa_length_drift.scoring as scoring_module
from experiments.flenqa_length_drift.scoring import (
    extract_generated_verdict,
    score_binary_answer,
)


class VerdictTokenizer:
    pieces = {
        "True": [1],
        " True": [2],
        "true": [3],
        " true": [4],
        "TRUE": [5],
        " TRUE": [6],
        "False": [7],
        " False": [8],
        "false": [9],
        " false": [10],
        "FALSE": [11],
        " FALSE": [12],
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces.get(text, [99, 100])


def test_binary_target_ranks_use_shared_best_token_rank(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []
    real = scoring_module.best_token_rank

    def recording(logits: torch.Tensor, target_ids: tuple[int, ...]) -> int:
        calls.append(target_ids)
        return real(logits, target_ids)

    monkeypatch.setattr(scoring_module, "best_token_rank", recording)
    logits = torch.zeros(13)
    logits[2] = 4.0
    logits[8] = 2.0

    result = score_binary_answer(
        logits,
        tokenizer=VerdictTokenizer(),
        label=True,
        generated_text="True.",
    )

    assert len(calls) == 2
    assert set(calls[0]) == {1, 2, 3, 4, 5, 6}
    assert set(calls[1]) == {7, 8, 9, 10, 11, 12}
    assert result.predicted is True
    assert result.correct is True
    assert result.extracted is True
    assert result.generated_correct is True
    assert result.agrees is True


def test_generated_verdict_is_front_loaded_and_gold_blind() -> None:
    assert extract_generated_verdict(" false. Explanation later.") is False
    assert extract_generated_verdict("TRUE\nMore") is True
    assert extract_generated_verdict("Maybe true later") is None
    assert extract_generated_verdict("") is None


def test_binary_score_records_disagreement_without_changing_logit_verdict() -> None:
    logits = torch.zeros(13)
    logits[8] = 5.0
    logits[2] = 1.0

    result = score_binary_answer(
        logits,
        tokenizer=VerdictTokenizer(),
        label=False,
        generated_text="True",
    )

    assert result.predicted is False
    assert result.correct is True
    assert result.extracted is True
    assert result.generated_correct is False
    assert result.agrees is False
