from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

import jlens_reasoning.benchmarks.flenqa.runner as runner_module
from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    FlenqaRow,
    SourceProvenance,
)
from jlens_reasoning.benchmarks.flenqa.positions import (
    LabeledPosition,
    PreparedPrompt,
    ResolvedSpan,
)
from jlens_reasoning.benchmarks.flenqa.runner import (
    LensPassResult,
    LensRunners,
    PromptShard,
    RunConfig,
    deterministic_topk,
    run_benchmark,
    run_prompt,
    run_shard,
)
from jlens_reasoning.benchmarks.flenqa.storage import (
    REQUIRED_TABLES,
    TABLE_SCHEMAS,
    is_shard_complete,
)
from jlens_reasoning.experiments_utils.spans import CharSpan


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
        context=ResolvedSpan("context", "ab", CharSpan(0, 2), CharSpan(0, 2)),
        paragraph_payload_spans=(),
        facts=(
            ResolvedSpan("fact_a_end", "a", CharSpan(0, 1), CharSpan(0, 1)),
            ResolvedSpan("fact_b_end", "b", CharSpan(1, 2), CharSpan(1, 2)),
        ),
        bridges=(),
        question=ResolvedSpan(
            "question_end",
            "c",
            CharSpan(2, 3),
            CharSpan(2, 3),
        ),
        rule=None,
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

    expected = (jacobian_logits - logit_logits).abs().max().item()
    assert batches["prompts"].to_pydict()["max_abs_logit_diff"] == pytest.approx(
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


def test_run_prompt_rejects_model_logits_with_wrong_position_rows() -> None:
    wrong = _pass(model_logits=torch.zeros(1, 5))

    with pytest.raises(RuntimeError, match="model-logit rows"):
        run_prompt(
            _prepared(),
            runners=_runners(jacobian=wrong, logit=wrong),
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


def test_incomplete_shard_is_rebuilt_from_the_beginning(tmp_path: Path) -> None:
    partial = tmp_path / "topk" / "shard-00000.parquet"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    manifest = run_shard(
        PromptShard(0, (0,)),
        (_prepared(),),
        output_dir=tmp_path,
        runners=_runners(),
        config=_config(),
    )

    assert manifest.prompt_ids == ("a" * 64,)
    assert is_shard_complete(
        tmp_path,
        shard_id=0,
        schemas=TABLE_SCHEMAS,
        required_tables=REQUIRED_TABLES,
    )


class FailingRunner:
    def run(
        self,
        prompt: str,
        *,
        positions: Sequence[int],
        max_seq_len: int,
    ) -> LensPassResult:
        raise AssertionError("completed shard must not rerun")


def test_completed_shard_resumes_without_rerunning(tmp_path: Path) -> None:
    shard = PromptShard(0, (0,))
    first = run_shard(
        shard,
        (_prepared(),),
        output_dir=tmp_path,
        runners=_runners(),
        config=_config(),
    )

    resumed = run_shard(
        shard,
        (_prepared(),),
        output_dir=tmp_path,
        runners=LensRunners(FailingRunner(), FailingRunner()),
        config=_config(),
    )

    assert resumed == first


class CharTokenizer:
    all_special_ids: tuple[int, ...] = ()

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        return {
            "input_ids": [[*range(len(text))]],
            "offset_mapping": [[(index, index + 1) for index in range(len(text))]],
        }


class DynamicRunner:
    def __init__(self, *, model_logit_offset: float = 0.0) -> None:
        self.model_logit_offset = model_logit_offset

    def run(
        self,
        prompt: str,
        *,
        positions: Sequence[int],
        max_seq_len: int,
    ) -> LensPassResult:
        rows = len(positions)
        logits = torch.arange(rows * 5, dtype=torch.float32).reshape(rows, 5)
        return LensPassResult(
            logits_by_layer={4: logits},
            model_logits=logits + self.model_logit_offset,
            input_ids=[[*range(len(prompt))]],
        )


def _ruletaker_row() -> FlenqaRow:
    return FlenqaRow(
        source_row_id=0,
        problem_id=0,
        sample_id=0,
        task="Simplified RuleTaker",
        label=True,
        key_texts=("The cow is young.", "The cow is kind."),
        rule="If someone is young then they are blue.",
        question="The cow is blue.",
        mixin="The cow is young.\nThe cow is kind.",
        ctx_size_declared=250,
        padding_type_declared="books",
        dispersion_declared="first",
    )


class RecordingProgress:
    def __init__(self, **kwargs: object) -> None:
        self.options = kwargs
        self.updates: list[int] = []
        self.closed = False

    def __enter__(self) -> RecordingProgress:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def update(self, amount: int = 1) -> None:
        self.updates.append(amount)


def test_run_benchmark_tracks_individual_prompts_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_bars: list[RecordingProgress] = []

    def recording_progress(**kwargs: object) -> RecordingProgress:
        progress = RecordingProgress(**kwargs)
        progress_bars.append(progress)
        return progress

    monkeypatch.setattr(runner_module, "tqdm", recording_progress, raising=False)
    rows = (
        _ruletaker_row(),
        replace(
            _ruletaker_row(),
            source_row_id=1,
            problem_id=1,
            question="The cow is green.",
        ),
    )

    run_benchmark(
        rows,
        output_dir=tmp_path,
        tokenizer=CharTokenizer(),
        runners=LensRunners(DynamicRunner(), DynamicRunner()),
        config=_config(expected_source_rows=2, shard_size=1),
    )

    assert len(progress_bars) == 1
    assert progress_bars[0].options == {
        "total": 2,
        "desc": "FLenQA prompts",
        "unit": "prompt",
        "disable": False,
    }
    assert progress_bars[0].updates == [1, 1]
    assert progress_bars[0].closed


def test_run_benchmark_counts_completed_prompts_when_resuming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _ruletaker_row()
    config = _config()
    run_benchmark(
        (row,),
        output_dir=tmp_path,
        tokenizer=CharTokenizer(),
        runners=LensRunners(DynamicRunner(), DynamicRunner()),
        config=config,
    )
    (tmp_path / "run-manifest.json").unlink()
    progress_bars: list[RecordingProgress] = []

    def recording_progress(**kwargs: object) -> RecordingProgress:
        progress = RecordingProgress(**kwargs)
        progress_bars.append(progress)
        return progress

    monkeypatch.setattr(runner_module, "tqdm", recording_progress, raising=False)

    run_benchmark(
        (row,),
        output_dir=tmp_path,
        tokenizer=CharTokenizer(),
        runners=LensRunners(FailingRunner(), FailingRunner()),
        config=config,
    )

    assert progress_bars[0].updates == [1]
    assert progress_bars[0].closed


def test_run_manifest_summarizes_max_logit_difference(tmp_path: Path) -> None:
    offset = 2**-20
    manifest = run_benchmark(
        (_ruletaker_row(),),
        output_dir=tmp_path,
        tokenizer=CharTokenizer(),
        runners=LensRunners(
            DynamicRunner(),
            DynamicRunner(model_logit_offset=offset),
        ),
        config=_config(),
    )

    assert manifest.max_abs_logit_diff == pytest.approx(offset)
    assert manifest.returned_layers == (4,)


def test_completed_run_resumes_without_rerunning_lenses(tmp_path: Path) -> None:
    config = _config()
    first = run_benchmark(
        (_ruletaker_row(),),
        output_dir=tmp_path,
        tokenizer=CharTokenizer(),
        runners=LensRunners(DynamicRunner(), DynamicRunner()),
        config=config,
    )

    resumed = run_benchmark(
        (_ruletaker_row(),),
        output_dir=tmp_path,
        tokenizer=CharTokenizer(),
        runners=LensRunners(FailingRunner(), FailingRunner()),
        config=config,
    )

    assert resumed == first


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    run_benchmark(
        (_ruletaker_row(),),
        output_dir=tmp_path,
        tokenizer=CharTokenizer(),
        runners=LensRunners(DynamicRunner(), DynamicRunner()),
        config=_config(),
    )

    with pytest.raises(RuntimeError, match="configuration"):
        run_benchmark(
            (_ruletaker_row(),),
            output_dir=tmp_path,
            tokenizer=CharTokenizer(),
            runners=LensRunners(DynamicRunner(), DynamicRunner()),
            config=_config(top_k=1),
        )
