from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest
import torch

from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    SourceProvenance,
)
from jlens_reasoning.benchmarks.flenqa.positions import (
    LabeledPosition,
    PreparedPrompt,
)
from jlens_reasoning.benchmarks.flenqa.runner import (
    LensPassResult,
    LensRunners,
    RunConfig,
    deterministic_topk,
    run_prompt,
)


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


def _prepared(
    *,
    positions: tuple[LabeledPosition, ...] = (
        LabeledPosition("sampled_padding", 0),
        LabeledPosition("question_end", 2),
        LabeledPosition("final_prompt", 2),
    ),
) -> PreparedPrompt:
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
        token_signature="signature",
        context_char_span=None,
        context_token_span=None,
        paragraph_payload_spans=(),
        diagnostics=(),
        bridge=None,
        positions=positions,
        special_token_ids=frozenset(),
    )


def _pass(
    *,
    layers: tuple[int, ...] = (3, 9),
    positions: int = 2,
    vocab_size: int = 5,
    model_logits: torch.Tensor | None = None,
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
        input_ids=[[10, 11, 12]],
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


def _config(**overrides: Any) -> RunConfig:
    config = RunConfig(
        model_name="model",
        lens_revision="lens",
        tokenizer_name="tokenizer",
        code_revision="code",
        expected_source_rows=1,
        expected_bridge_problems=0,
    )
    return replace(config, **overrides)


def test_run_prompt_executes_duplicate_labels_once() -> None:
    runners = _runners()

    batches = run_prompt(
        _prepared(),
        runners=runners,
        config=_config(top_k=2),
    )

    assert runners.jacobian.requested_positions == [(0, 2)]
    assert runners.logit.requested_positions == [(0, 2)]
    assert batches["positions"].num_rows == 3
    assert batches["topk"].num_rows == 2 * 2 * 2 * 2


def test_run_prompt_accepts_allclose_logits_and_records_exact_max_diff() -> None:
    jacobian_logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    logit_logits = torch.tensor([[1.0, 2.0 + 5e-7], [3.0, 4.0]])
    runners = _runners(
        jacobian=_pass(vocab_size=2, model_logits=jacobian_logits),
        logit=_pass(vocab_size=2, model_logits=logit_logits),
    )

    batches = run_prompt(_prepared(), runners=runners, config=_config())

    expected = (
        jacobian_logits - logit_logits
    ).abs().max().item()
    assert batches["prompts"].to_pydict()[
        "max_abs_logit_diff"
    ] == pytest.approx([expected])


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
        run_prompt(_prepared(), runners=runners, config=_config())


def test_run_prompt_preserves_returned_layer_keys() -> None:
    result = _pass(layers=(4, 11))

    batches = run_prompt(
        _prepared(),
        runners=_runners(jacobian=result, logit=result),
        config=_config(top_k=1),
    )

    assert set(batches["topk"].to_pydict()["layer"]) == {4, 11}


def test_run_prompt_rejects_different_layer_keys() -> None:
    with pytest.raises(RuntimeError, match="layer keys"):
        run_prompt(
            _prepared(),
            runners=_runners(
                jacobian=_pass(layers=(4, 11)),
                logit=_pass(layers=(4, 12)),
            ),
            config=_config(),
        )


def test_deterministic_topk_breaks_logit_ties_by_lower_token_id() -> None:
    logits = torch.tensor([1.0, 5.0, 5.0, 4.0, 5.0])

    ranked = deterministic_topk(logits, k=3)

    assert [(item.rank, item.token_id, item.logit) for item in ranked] == [
        (1, 1, 5.0),
        (2, 2, 5.0),
        (3, 4, 5.0),
    ]
