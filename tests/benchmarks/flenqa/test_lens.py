from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch

from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    SourceProvenance,
)
from jlens_reasoning.benchmarks.flenqa.lens import (
    LensPassResult,
    LensRunners,
    deterministic_topk,
    run_prompt,
)
from jlens_reasoning.benchmarks.flenqa.positions import PreparedPrompt


class RecordingRunner:
    def __init__(self, result: LensPassResult) -> None:
        self.result = result
        self.requested_positions: list[tuple[int, ...]] = []

    def run(
        self,
        prompt: str,
        *,
        positions: Sequence[int],
        max_seq_len: int,
    ) -> LensPassResult:
        self.requested_positions.append(tuple(positions))
        return self.result


def _prepared() -> PreparedPrompt:
    prompt = FlenqaPrompt(
        canonical_index=0,
        prompt_id="a" * 64,
        problem_id=1,
        task="Simplified RuleTaker",
        text="abc",
        question="c",
        key_texts=("a", "b"),
        rule="r",
        label=True,
        mixin="ab",
        provenance=(SourceProvenance(3, 500, "same", "last"),),
    )
    return PreparedPrompt(
        prompt=prompt,
        input_ids=(10, 11, 12),
        offsets=((0, 1), (1, 2), (2, 3)),
        positions={
            "sampled_padding": (0,),
            "question_end": (2,),
            "final_prompt": (2,),
        },
    )


def _pass(
    *,
    layers: tuple[int, ...] = (3, 9),
    positions: int = 2,
    vocab_size: int = 5,
    model_logits: torch.Tensor | None = None,
    input_ids: object = ((10, 11, 12),),
) -> LensPassResult:
    by_layer: Mapping[int, torch.Tensor] = {
        layer: torch.arange(
            positions * vocab_size,
            dtype=torch.float32,
        ).reshape(positions, vocab_size)
        + layer
        for layer in layers
    }
    return LensPassResult(
        logits_by_layer=by_layer,
        model_logits=(
            torch.arange(
                positions * vocab_size,
                dtype=torch.float32,
            ).reshape(positions, vocab_size)
            if model_logits is None
            else model_logits
        ),
        input_ids=input_ids,
    )


def _runners(
    *,
    jacobian: LensPassResult | None = None,
    logit: LensPassResult | None = None,
) -> LensRunners:
    return LensRunners(
        jacobian=RecordingRunner(_pass() if jacobian is None else jacobian),
        logit=RecordingRunner(_pass() if logit is None else logit),
    )


def _run(
    runners: LensRunners,
    *,
    top_k: int = 25,
    logits_rtol: float = 1e-5,
    logits_atol: float = 1e-6,
):
    return run_prompt(
        _prepared(),
        runners=runners,
        top_k=top_k,
        max_seq_len=4096,
        logits_rtol=logits_rtol,
        logits_atol=logits_atol,
    )


def test_run_prompt_executes_duplicate_labels_once() -> None:
    runners = _runners()

    result = _run(runners, top_k=2)

    assert runners.jacobian.requested_positions == [(0, 2)]
    assert runners.logit.requested_positions == [(0, 2)]
    assert result.returned_layers == (3, 9)
    assert result.batches["positions"].num_rows == 3
    assert result.batches["topk"].num_rows == 2 * 2 * 2 * 2


def test_run_prompt_records_maximum_model_logit_difference() -> None:
    jacobian_logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    logit_logits = torch.tensor([[1.0, 2.0 + 5e-7], [3.0, 4.0]])

    result = _run(
        _runners(
            jacobian=_pass(vocab_size=2, model_logits=jacobian_logits),
            logit=_pass(vocab_size=2, model_logits=logit_logits),
        )
    )

    expected = (jacobian_logits - logit_logits).abs().max().item()
    assert result.max_abs_logit_diff == pytest.approx(expected)
    assert result.batches["prompts"].to_pydict()["max_abs_logit_diff"] == pytest.approx(
        [expected]
    )


def test_run_prompt_rejects_logits_outside_tolerance() -> None:
    runners = _runners(
        jacobian=_pass(
            vocab_size=1,
            model_logits=torch.tensor([[1.0], [1.0]]),
        ),
        logit=_pass(
            vocab_size=1,
            model_logits=torch.tensor([[1.01], [1.0]]),
        ),
    )

    with pytest.raises(RuntimeError, match="allclose"):
        _run(runners)


def test_run_prompt_rejects_input_ids_different_from_preparation() -> None:
    wrong = _pass(input_ids=((99,),))

    with pytest.raises(RuntimeError, match="input IDs"):
        _run(_runners(jacobian=wrong))


def test_run_prompt_rejects_different_layer_keys() -> None:
    with pytest.raises(RuntimeError, match="layer keys"):
        _run(
            _runners(
                jacobian=_pass(layers=(4, 11)),
                logit=_pass(layers=(4, 12)),
            )
        )


def test_run_prompt_rejects_model_logits_with_wrong_position_rows() -> None:
    wrong = _pass(model_logits=torch.zeros(1, 5))

    with pytest.raises(RuntimeError, match="model-logit rows"):
        _run(_runners(jacobian=wrong, logit=wrong))


def test_deterministic_topk_breaks_logit_ties_by_lower_token_id() -> None:
    logits = torch.tensor([1.0, 5.0, 5.0, 4.0, 5.0])

    ranked = deterministic_topk(logits, k=3)

    assert [(item.rank, item.token_id, item.logit) for item in ranked] == [
        (1, 1, 5.0),
        (2, 2, 5.0),
        (3, 4, 5.0),
    ]
