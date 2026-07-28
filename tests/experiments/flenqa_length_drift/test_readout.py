from __future__ import annotations

import torch

import experiments.flenqa_length_drift.readout as readout_module
from experiments.flenqa_length_drift.anchors import Anchor
from experiments.flenqa_length_drift.readout import (
    TokenCandidate,
    deterministic_topk,
    reduce_readout,
)


def test_deterministic_topk_breaks_logit_ties_by_lower_token_id() -> None:
    logits = torch.tensor([1.0, 5.0, 5.0, 4.0, 5.0])

    ranked = deterministic_topk(logits, k=3)

    assert [(item.rank, item.token_id, item.logit) for item in ranked] == [
        (1, 1, 5.0),
        (2, 2, 5.0),
        (3, 4, 5.0),
    ]


def test_topk_does_not_call_best_token_rank_per_output_token(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("top-k must not scan ranks token by token")

    monkeypatch.setattr(readout_module, "best_token_rank", forbidden)

    assert len(deterministic_topk(torch.arange(100, dtype=torch.float32), k=25)) == 25


def test_reduce_readout_uses_best_token_rank_for_bridge_targets(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []
    real = readout_module.best_token_rank

    def recording(logits: torch.Tensor, target_ids: tuple[int, ...]) -> int:
        calls.append(target_ids)
        return real(logits, target_ids)

    monkeypatch.setattr(readout_module, "best_token_rank", recording)
    logits = {
        7: torch.tensor(
            [
                [0.0, 5.0, 2.0, 1.0],
                [3.0, 0.0, 4.0, 2.0],
            ]
        )
    }

    reduction = reduce_readout(
        prompt_id="abc",
        lens_kind="jacobian",
        logits_by_layer=logits,
        positions=(3, 9),
        anchors=(Anchor("fact_a_end", 3), Anchor("final_prompt", 9)),
        candidates=(TokenCandidate(" bridge", 3),),
        top_k=2,
    )

    assert calls == [(3,), (3,)]
    assert [item.layer for item in reduction.targets] == [7, 7]
    assert [item.position for item in reduction.targets] == [3, 9]
    assert reduction.targets[0].rank == 3
    assert {item.anchor_label for item in reduction.topk} == {
        "fact_a_end",
        "final_prompt",
    }
    assert {item.position for item in reduction.summary} == {3, 9}


def test_reduce_readout_rejects_layer_rows_that_do_not_match_positions() -> None:
    try:
        reduce_readout(
            prompt_id="abc",
            lens_kind="logit",
            logits_by_layer={2: torch.zeros(1, 4)},
            positions=(1, 2),
            anchors=(Anchor("final_prompt", 2),),
            candidates=(),
            top_k=2,
        )
    except ValueError as exc:
        assert "positions" in str(exc)
    else:
        raise AssertionError("mismatched lens rows were accepted")
