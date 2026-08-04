from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jlens_reasoning.benchmarks.flenqa.accuracy import (
    AccuracyRunConfig,
    load_accuracy_results,
    run_accuracy,
)
from jlens_reasoning.benchmarks.flenqa.dataset import FlenqaRow
from jlens_reasoning.evaluation import ModelOutput


class FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[list[int]]]:
        assert kwargs["truncation"] is False
        return {"input_ids": [[*range(len(text))]]}


class RecordingGenerator:
    def __init__(self, output: ModelOutput | None = None) -> None:
        self.output = output or ModelOutput(
            "Final answer: True",
            token_ids=(21, 22),
            token_pieces=(" True", ""),
            finish_reason="eos",
        )
        self.calls: list[tuple[str, int]] = []

    def __call__(self, prompt: str, *, max_new_tokens: int) -> ModelOutput:
        self.calls.append((prompt, max_new_tokens))
        return self.output


def _row() -> FlenqaRow:
    return FlenqaRow(
        source_row_id=0,
        problem_id=0,
        sample_id=0,
        task="PIR",
        label=True,
        key_texts=("Ava is in the blue room.", "The blue room has marble."),
        rule=None,
        question="Is Ava in a marble room?",
        mixin="Ava is in the blue room.\nThe blue room has marble.",
        ctx_size_declared=250,
        padding_type_declared="books",
        dispersion_declared="random",
    )


def _config(**overrides: object) -> AccuracyRunConfig:
    return replace(
        AccuracyRunConfig(
            model_name="model",
            tokenizer_name="tokenizer",
            code_revision="code",
            shard_size=1,
            expected_source_rows=1,
            expected_prompts=1,
        ),
        **overrides,
    )


def test_run_accuracy_generates_once_per_unique_prompt(tmp_path: Path) -> None:
    generator = RecordingGenerator()
    rows = (_row(), replace(_row(), source_row_id=1, sample_id=1))
    manifest = run_accuracy(
        rows,
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generate=generator,
        config=_config(expected_source_rows=2),
    )
    table = load_accuracy_results(tmp_path, manifest)

    assert len(generator.calls) == 1
    assert generator.calls[0][1] == 64
    assert table.num_rows == 1
    assert table.column("correct").to_pylist() == [True]
    assert table.column("n_input_tokens").to_pylist() == [len(generator.calls[0][0])]
    assert len(table.column("provenance").to_pylist()[0]) == 2


def test_completed_run_resumes_without_generation(tmp_path: Path) -> None:
    config = _config()
    first = run_accuracy(
        (_row(),),
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generate=RecordingGenerator(),
        config=config,
    )

    def fail(*args: object, **kwargs: object) -> ModelOutput:
        raise AssertionError("completed run must not generate")

    resumed = run_accuracy(
        (_row(),),
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generate=fail,
        config=config,
    )

    assert resumed == first


def test_config_mismatch_is_rejected_before_shards_change(tmp_path: Path) -> None:
    run_accuracy(
        (_row(),),
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generate=RecordingGenerator(),
        config=_config(),
    )
    shard = tmp_path / "results" / "shard-00000.parquet"
    before = shard.read_bytes()

    with pytest.raises(RuntimeError, match="configuration"):
        run_accuracy(
            (_row(),),
            output_dir=tmp_path,
            tokenizer=FakeTokenizer(),
            generate=RecordingGenerator(),
            config=_config(max_new_tokens=8),
        )

    assert shard.read_bytes() == before


def test_failed_generation_aborts_incomplete_shard(tmp_path: Path) -> None:
    def fail(*args: object, **kwargs: object) -> ModelOutput:
        raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        run_accuracy(
            (_row(),),
            output_dir=tmp_path,
            tokenizer=FakeTokenizer(),
            generate=fail,
            config=_config(),
        )

    assert not (tmp_path / "manifests" / "shard-00000.json").exists()
    assert not (tmp_path / "results" / "shard-00000.parquet").exists()


def test_over_limit_prompt_fails_before_generation(tmp_path: Path) -> None:
    generator = RecordingGenerator()

    with pytest.raises(ValueError, match="maximum sequence length"):
        run_accuracy(
            (_row(),),
            output_dir=tmp_path,
            tokenizer=FakeTokenizer(),
            generate=generator,
            config=_config(max_seq_len=1),
        )

    assert generator.calls == []


def test_result_preserves_generation_metadata_and_nominal_length(
    tmp_path: Path,
) -> None:
    output = ModelOutput(
        "False, then TRUE",
        token_ids=(4, 5, 6),
        token_pieces=("False", ", then", " TRUE"),
        finish_reason="eos",
    )
    manifest = run_accuracy(
        (_row(),),
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generate=RecordingGenerator(output),
        config=_config(),
    )
    result = load_accuracy_results(tmp_path, manifest).to_pydict()

    assert result["ctx_size"] == [250]
    assert result["generated_token_ids"] == [[4, 5, 6]]
    assert result["generated_token_pieces"] == [["False", ", then", " TRUE"]]
    assert result["generated_text"] == ["False, then TRUE"]
    assert result["generation_status"] == ["complete"]
    assert result["finish_reason"] == ["eos"]
    assert result["verdict"] == [True]


def test_corrupt_completed_shard_is_rebuilt(tmp_path: Path) -> None:
    config = _config()
    run_accuracy(
        (_row(),),
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generate=RecordingGenerator(),
        config=config,
    )
    (tmp_path / "results" / "shard-00000.parquet").write_bytes(b"corrupt")
    generator = RecordingGenerator()

    manifest = run_accuracy(
        (_row(),),
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generate=generator,
        config=config,
    )

    assert len(generator.calls) == 1
    assert load_accuracy_results(tmp_path, manifest).num_rows == 1


def test_expected_prompt_count_is_checked_before_generation(tmp_path: Path) -> None:
    generator = RecordingGenerator()

    with pytest.raises(ValueError, match="unique prompts"):
        run_accuracy(
            (_row(),),
            output_dir=tmp_path,
            tokenizer=FakeTokenizer(),
            generate=generator,
            config=_config(expected_prompts=2),
        )

    assert generator.calls == []


def test_result_loading_rejects_run_configuration_mismatch(tmp_path: Path) -> None:
    manifest = run_accuracy(
        (_row(),),
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generate=RecordingGenerator(),
        config=_config(),
    )
    metadata_path = tmp_path / "run-meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config_hash"] = "different"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="configuration"):
        load_accuracy_results(tmp_path, manifest)
