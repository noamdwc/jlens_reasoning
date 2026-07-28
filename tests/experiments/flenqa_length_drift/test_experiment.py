from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import torch

from experiments.flenqa_length_drift.experiment import (
    LensPassResult,
    LensRunners,
    PromptShard,
    plan_shards,
    run_prompt,
    run_shard,
)
from experiments.flenqa_length_drift.tables import (
    REQUIRED_SHARD_TABLES,
    TABLE_SCHEMAS,
)
from jlens_reasoning.benchmarks.flenqa import FlenqaPrompt, FlenqaRow
from jlens_reasoning.benchmarks.flenqa_preparation import prepare_prompt
from jlens_reasoning.benchmarks.flenqa_prompts import (
    build_prompt_text,
    compute_prompt_id,
)
from jlens_reasoning.experiments_utils.storage import is_shard_complete


class ExperimentTokenizer:
    verdict_ids = {
        "True": 1,
        " True": 2,
        "true": 3,
        " true": 4,
        "TRUE": 5,
        " TRUE": 6,
        "False": 7,
        " False": 8,
        "false": 9,
        " false": 10,
        "FALSE": 11,
        " FALSE": 12,
        "Bob": 13,
        " Bob": 14,
        "bob": 15,
        " bob": 16,
        "BOB": 17,
        " BOB": 18,
    }

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, Any]:
        ids = [50, *(100 + index for index in range(len(text)))]
        offsets = [(0, 0), *((index, index + 1) for index in range(len(text)))]
        return {"input_ids": [ids], "offset_mapping": [offsets]}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        token_id = self.verdict_ids.get(text)
        return [token_id] if token_id is not None else [90, 91]


def _source_row(index: int = 0) -> FlenqaRow:
    return FlenqaRow(
        source_row_id=index,
        problem_id=index,
        sample_id=index,
        task="PIR",
        label=True,
        key_texts=("Ada called Bob.", "Bob greeted Ada."),
        rule=None,
        question="Did Ada meet someone?",
        mixin=f"Ada called Bob.\nPadding text {index}.\nBob greeted Ada.",
        ctx_size_declared=500,
        padding_type_declared="books",
        dispersion_declared="middle",
    )


def _prepared(index: int = 0):
    row = _source_row(index)
    text = build_prompt_text(
        task=row.task,
        question=row.question,
        mixin=row.mixin,
        rule=row.rule,
    )
    prompt = FlenqaPrompt(
        canonical_index=index,
        prompt_id=compute_prompt_id(text),
        problem_id=row.problem_id,
        task=row.task,
        text=text,
        question=row.question,
        key_texts=row.key_texts,
        rule=None,
        label=row.label,
        mixin=row.mixin,
        ctx_size_declared=row.ctx_size_declared,
        source_row_ids=(index,),
        padding_type_declared=("books",),
        dispersion_declared=("middle",),
    )
    tokenizer = ExperimentTokenizer()
    return prepare_prompt(prompt, tokenizer, bridge="Bob"), tokenizer


class FakeRunner:
    def __init__(
        self,
        input_ids: Sequence[int],
        *,
        replacement_ids: Sequence[int] | None = None,
    ) -> None:
        self.input_ids = tuple(
            input_ids if replacement_ids is None else replacement_ids
        )
        self.calls: list[tuple[str, tuple[int, ...], int]] = []

    def run(
        self,
        prompt: str,
        *,
        positions: Sequence[int],
        max_seq_len: int,
    ) -> LensPassResult:
        selected = tuple(positions)
        self.calls.append((prompt, selected, max_seq_len))
        rows = len(selected)
        logits = torch.zeros(rows, 128)
        logits[:, 13] = 4.0
        logits[:, 2] = 5.0
        logits[:, 8] = 1.0
        return LensPassResult(
            logits_by_layer={3: logits},
            model_logits=logits.clone(),
            input_ids=self.input_ids,
        )


def test_run_prompt_calls_both_lenses_with_identical_prompt_positions_and_ids() -> None:
    prepared, tokenizer = _prepared()
    jacobian = FakeRunner(prepared.input_ids)
    logit = FakeRunner(prepared.input_ids)

    batches = run_prompt(
        prepared,
        source_rows={0: _source_row()},
        jacobian_runner=jacobian,
        logit_runner=logit,
        tokenizer=tokenizer,
        generate=lambda prompt: "True.",
    )

    assert len(jacobian.calls) == len(logit.calls) == 1
    assert jacobian.calls == logit.calls
    prompt, positions, max_seq_len = jacobian.calls[0]
    assert prompt == prepared.prompt.text
    assert positions
    assert max_seq_len == 4096
    assert tuple(table for table, _batch in batches) == REQUIRED_SHARD_TABLES
    for table, batch in batches:
        assert isinstance(batch, pa.RecordBatch)
        assert batch.schema == TABLE_SCHEMAS[table]
    assert dict(batches)["prompts"].num_rows == 1
    assert dict(batches)["source_rows"].num_rows == 1
    assert dict(batches)["spans"].num_rows == len(prepared.diagnostics)
    assert dict(batches)["anchors"].num_rows > 0
    assert dict(batches)["topk"].num_rows > 0
    assert dict(batches)["bridge"].num_rows > 0
    assert dict(batches)["summary"].num_rows > 0
    assert dict(batches)["scoring"].num_rows == 1


@pytest.mark.parametrize("which", ["jacobian", "logit"])
def test_run_prompt_rejects_lens_token_id_mismatch_before_emitting(
    which: str,
) -> None:
    prepared, tokenizer = _prepared()
    wrong = (*prepared.input_ids[:-1], prepared.input_ids[-1] + 1)
    jacobian = FakeRunner(
        prepared.input_ids,
        replacement_ids=wrong if which == "jacobian" else None,
    )
    logit = FakeRunner(
        prepared.input_ids,
        replacement_ids=wrong if which == "logit" else None,
    )

    with pytest.raises(RuntimeError, match="token IDs"):
        run_prompt(
            prepared,
            source_rows={0: _source_row()},
            jacobian_runner=jacobian,
            logit_runner=logit,
            tokenizer=tokenizer,
        )


def test_run_prompt_rejects_different_model_logits_between_lens_calls() -> None:
    prepared, tokenizer = _prepared()
    jacobian = FakeRunner(prepared.input_ids)

    class DifferentRunner(FakeRunner):
        def run(self, *args, **kwargs):
            result = super().run(*args, **kwargs)
            return LensPassResult(
                logits_by_layer=result.logits_by_layer,
                model_logits=result.model_logits + 1,
                input_ids=result.input_ids,
            )

    with pytest.raises(RuntimeError, match="model logits"):
        run_prompt(
            prepared,
            source_rows={0: _source_row()},
            jacobian_runner=jacobian,
            logit_runner=DifferentRunner(prepared.input_ids),
            tokenizer=tokenizer,
        )


def test_run_prompt_padding_anchors_are_members_of_explicit_padding_set() -> None:
    prepared, tokenizer = _prepared()
    runner = FakeRunner(prepared.input_ids)

    batches = dict(
        run_prompt(
            prepared,
            source_rows={0: _source_row()},
            jacobian_runner=runner,
            logit_runner=FakeRunner(prepared.input_ids),
            tokenizer=tokenizer,
        )
    )

    anchors = batches["anchors"].to_pydict()
    padding_positions = {
        position
        for label, position in zip(
            anchors["anchor_label"],
            anchors["position"],
            strict=True,
        )
        if label == "sampled_padding"
    }
    context_text = prepared.prompt.text
    assert padding_positions
    assert {
        context_text[prepared.offsets[position][0] : prepared.offsets[position][1]]
        for position in padding_positions
    } <= set("Padding text 0.")


def test_plan_shards_uses_original_canonical_order() -> None:
    prepared = [_prepared(index)[0] for index in range(5)]

    assert plan_shards(prepared, shard_size=2) == (
        PromptShard(shard_id=0, prompt_indices=(0, 1)),
        PromptShard(shard_id=1, prompt_indices=(2, 3)),
        PromptShard(shard_id=2, prompt_indices=(4,)),
    )


def test_run_shard_resumes_only_from_validated_complete_manifest(
    tmp_path: Path,
) -> None:
    prepared_and_tokenizers = [_prepared(index) for index in range(3)]
    prepared = [item[0] for item in prepared_and_tokenizers]
    tokenizer = prepared_and_tokenizers[0][1]
    source_rows = {index: _source_row(index) for index in range(3)}
    runners = LensRunners(
        jacobian=FakeRunner(prepared[0].input_ids),
        logit=FakeRunner(prepared[0].input_ids),
    )
    shards = plan_shards(prepared, shard_size=1)

    run_shard(
        shards[0],
        prepared,
        output_dir=tmp_path,
        source_rows=source_rows,
        runners=runners,
        tokenizer=tokenizer,
    )
    remaining = tuple(
        shard
        for shard in shards
        if not is_shard_complete(
            tmp_path,
            shard_id=shard.shard_id,
            schemas=TABLE_SCHEMAS,
            required_tables=REQUIRED_SHARD_TABLES,
        )
    )

    assert remaining == shards[1:]
    assert remaining[0].prompt_indices == (1,)
    assert remaining[1].prompt_indices == (2,)


@pytest.mark.parametrize("fail_table", REQUIRED_SHARD_TABLES)
def test_run_shard_crash_keeps_shard_incomplete_and_reruns_every_prompt(
    tmp_path: Path,
    fail_table: str,
) -> None:
    first, tokenizer = _prepared(0)
    second, _ = _prepared(1)
    prepared = [first, second]
    source_rows = {index: _source_row(index) for index in range(2)}
    shard = plan_shards(prepared, shard_size=2)[0]
    calls: list[str] = []

    def fake_run_prompt(prompt, **kwargs):
        calls.append(prompt.prompt.prompt_id)
        return run_prompt(prompt, **kwargs)

    def crash_after_table(table: str) -> None:
        if table == fail_table:
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        run_shard(
            shard,
            prepared,
            output_dir=tmp_path,
            source_rows=source_rows,
            runners=LensRunners(
                jacobian=FakeRunner(first.input_ids),
                logit=FakeRunner(first.input_ids),
            ),
            tokenizer=tokenizer,
            run_prompt_fn=fake_run_prompt,
            after_append=crash_after_table,
        )

    assert not is_shard_complete(
        tmp_path,
        shard_id=shard.shard_id,
        schemas=TABLE_SCHEMAS,
        required_tables=REQUIRED_SHARD_TABLES,
    )

    calls.clear()

    class PerPromptRunner:
        def run(self, prompt, *, positions, max_seq_len):
            expected = next(
                candidate for candidate in prepared if candidate.prompt.text == prompt
            )
            return FakeRunner(expected.input_ids).run(
                prompt,
                positions=positions,
                max_seq_len=max_seq_len,
            )

    manifest = run_shard(
        shard,
        prepared,
        output_dir=tmp_path,
        source_rows=source_rows,
        runners=LensRunners(
            jacobian=PerPromptRunner(),
            logit=PerPromptRunner(),
        ),
        tokenizer=tokenizer,
        run_prompt_fn=fake_run_prompt,
    )

    assert calls == [first.prompt.prompt_id, second.prompt.prompt_id]
    assert manifest.prompt_ids == tuple(calls)
