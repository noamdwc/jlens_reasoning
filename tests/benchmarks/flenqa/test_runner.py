from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

import jlens_reasoning.benchmarks.flenqa.runner as runner_module
from jlens_reasoning.benchmarks.flenqa.dataset import FlenqaRow
from jlens_reasoning.benchmarks.flenqa.lens import LensPassResult, LensRunners
from jlens_reasoning.benchmarks.flenqa.runner import (
    RunConfig,
    RunSummary,
    run_benchmark,
)
from jlens_reasoning.benchmarks.flenqa.storage import REQUIRED_TABLES


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
        self.calls = 0

    def run(
        self,
        prompt: str,
        *,
        positions: Sequence[int],
        max_seq_len: int,
    ) -> LensPassResult:
        self.calls += 1
        rows = len(positions)
        logits = torch.arange(rows * 5, dtype=torch.float32).reshape(rows, 5)
        return LensPassResult(
            logits_by_layer={4: logits},
            model_logits=logits + self.model_logit_offset,
            input_ids=[[*range(len(prompt))]],
        )


class FailingRunner:
    def run(
        self,
        prompt: str,
        *,
        positions: Sequence[int],
        max_seq_len: int,
    ) -> LensPassResult:
        raise AssertionError("lens must not run")


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


def _row(*, source_row_id: int = 0, problem_id: int = 0) -> FlenqaRow:
    return FlenqaRow(
        source_row_id=source_row_id,
        problem_id=problem_id,
        sample_id=0,
        task="Simplified RuleTaker",
        label=True,
        key_texts=("The cow is young.", "The cow is kind."),
        rule="If someone is young then they are blue.",
        question=f"The cow is color {problem_id}.",
        mixin="The cow is young.\nThe cow is kind.",
        ctx_size_declared=250,
        padding_type_declared="books",
        dispersion_declared="first",
    )


def _config(**overrides: object) -> RunConfig:
    return replace(RunConfig(expected_source_rows=1), **overrides)


def test_run_benchmark_streams_compatible_parquet_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_bars: list[RecordingProgress] = []

    def recording_progress(**kwargs: object) -> RecordingProgress:
        progress = RecordingProgress(**kwargs)
        progress_bars.append(progress)
        return progress

    monkeypatch.setattr(runner_module, "tqdm", recording_progress)
    offset = 2**-20
    rows = (_row(), _row(source_row_id=1, problem_id=1))

    summary = run_benchmark(
        rows,
        output_dir=tmp_path,
        tokenizer=CharTokenizer(),
        runners=LensRunners(
            DynamicRunner(),
            DynamicRunner(model_logit_offset=offset),
        ),
        config=_config(expected_source_rows=2, shard_size=1),
    )

    assert summary == RunSummary(
        prompt_count=2,
        returned_layers=(4,),
        max_abs_logit_diff=pytest.approx(offset),
    )
    for table in REQUIRED_TABLES:
        files = sorted((tmp_path / table).glob("*.parquet"))
        assert [path.name for path in files] == [
            "shard-00000.parquet",
            "shard-00001.parquet",
        ]
        assert all(pq.read_table(path).num_rows > 0 for path in files)
    assert not (tmp_path / "manifests").exists()
    assert not (tmp_path / "run-meta.json").exists()
    assert not (tmp_path / "run-manifest.json").exists()
    assert progress_bars[0].options == {
        "total": 2,
        "desc": "FLenQA prompts",
        "unit": "prompt",
        "disable": False,
    }
    assert progress_bars[0].updates == [1, 1]
    assert progress_bars[0].closed


def test_run_benchmark_rejects_populated_output_before_running_lenses(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "topk" / "old.parquet"
    existing.parent.mkdir()
    existing.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="already contains FLenQA output"):
        run_benchmark(
            (_row(),),
            output_dir=tmp_path,
            tokenizer=CharTokenizer(),
            runners=LensRunners(FailingRunner(), FailingRunner()),
            config=_config(),
        )


def test_run_summary_keeps_layers_when_top_k_is_zero(tmp_path: Path) -> None:
    summary = run_benchmark(
        (_row(),),
        output_dir=tmp_path,
        tokenizer=CharTokenizer(),
        runners=LensRunners(DynamicRunner(), DynamicRunner()),
        config=_config(top_k=0),
        show_progress=False,
    )

    assert summary.returned_layers == (4,)
    assert pq.read_table(tmp_path / "topk" / "shard-00000.parquet").num_rows == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_k": -1}, "top_k"),
        ({"shard_size": 0}, "shard_size"),
        ({"max_seq_len": 0}, "max_seq_len"),
        ({"expected_source_rows": 0}, "expected_source_rows"),
        ({"logits_rtol": -1.0}, "tolerances"),
    ],
)
def test_run_benchmark_rejects_invalid_config(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(
            (_row(),),
            output_dir=tmp_path,
            tokenizer=CharTokenizer(),
            runners=LensRunners(FailingRunner(), FailingRunner()),
            config=_config(**overrides),
        )


def test_run_benchmark_rejects_unexpected_source_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected 2 source rows; found 1"):
        run_benchmark(
            (_row(),),
            output_dir=tmp_path,
            tokenizer=CharTokenizer(),
            runners=LensRunners(FailingRunner(), FailingRunner()),
            config=_config(expected_source_rows=2),
        )
