import json
from dataclasses import replace
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from experiments.flenqa_probe_jgain.experiment import (
    measure_prompt,
    select_pairs,
    summarize_conditions,
)
from jlens_reasoning.benchmarks.flenqa.dataset import FlenqaRow, prepare_prompts
from jlens_reasoning.inference import InferenceConfig
from jlens_reasoning.probe_jlens import prompt_hidden_map
from jlens_reasoning.probing import ProbeConfig

NOTEBOOK = Path("experiments/flenqa_probe_jgain/flenqa_probe_jgain.ipynb")


def paired_prompts(task="PIR", problem=1, label=True):
    row = FlenqaRow(
        source_row_id=problem * 2,
        problem_id=problem,
        task=task,
        label=label,
        key_texts=("cat",),
        rule="cat",
        question="cat",
        mixin=f"cat {problem}",
        ctx_size_declared=500,
        padding_type_declared="same",
        dispersion_declared="first",
    )
    return prepare_prompts(
        (
            row,
            replace(
                row,
                source_row_id=problem * 2 + 1,
                ctx_size_declared=3000,
                mixin=f"cat cat {problem}",
            ),
        )
    )


def test_cohort_preserves_splits_and_reports_missing_pairs():
    prompts, partitions = [], {}
    for task in ("PIR", "MonoRel", "Simplified RuleTaker"):
        for label in (False, True):
            for partition in (
                "validation",
                "validation",
                "validation",
                "test",
                "train",
            ):
                problem = len(partitions) + 1
                partitions[problem] = partition
                prompts.extend(paired_prompts(task, problem, label))
    partitions[99] = "test"  # An explicitly missing problem remains in coverage.
    dev, _ = select_pairs(
        tuple(prompts),
        partitions,
        phase="development",
        seed=1729,
        short_ctx=500,
        long_ctx=3000,
    )
    assert len(dev) == 12
    assert dev.groupby(["task", "label"]).size().eq(2).all()
    assert all(partitions[p] == "validation" for p in dev.problem_id)
    test, coverage = select_pairs(
        tuple(prompts),
        partitions,
        phase="evaluation",
        seed=1729,
        short_ctx=500,
        long_ctx=3000,
    )
    assert len(test) == 6
    assert all(partitions[p] == "test" for p in test.problem_id)
    assert coverage.set_index("problem_id").loc[99, "reason"] == "no_matched_pair"
    assert set(dev.problem_id).isdisjoint(test.problem_id)


@pytest.mark.parametrize("task", ["PIR", "MonoRel", "Simplified RuleTaker"])
def test_notebook_measurement_and_export_cells_use_real_utilities(
    model, tokenizer, tmp_path, task
):
    model.requires_grad_(False)
    config = ProbeConfig(InferenceConfig.direct(max_new_tokens=1, max_input_tokens=64))
    prompts = paired_prompts(task)
    probe = {
        "weight": torch.arange(1, 9).float(),
        "bias": torch.tensor(0.1),
        "training_mean": torch.zeros(8),
    }
    settings = {
        "layer": 0,
        "relative_strengths": [0.001],
        "control_seeds": [11],
        "fd_relative_steps": [0.001, 0.003],
        "fd_rtol": 0.05,
        "fd_atol": 1e-4,
    }
    cells = {
        cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells
    }
    namespace = dict(
        model=model,
        tokenizer=tokenizer,
        probe=probe,
        blocks=model.model.layers,
        PROBE_CONFIG=config,
        settings=settings,
        PHASE="development",
        pd=pd,
        np=np,
        measure_prompt=measure_prompt,
        prompt_by_id={p.prompt_id: p for p in prompts},
        selected_ids=[p.prompt_id for p in prompts],
        RESULT_DIR=tmp_path,
        display=lambda *args: None,
        tqdm=lambda xs, **kwargs: xs,
        write_results=lambda path, value: path.write_text(json.dumps(value)),
    )
    for cell in ("measure-first-prompt", "measure-cohort"):
        exec(compile(cells[cell], f"{NOTEBOOK}:{cell}", "exec"), namespace)
    results = pd.read_parquet(tmp_path / "conditions.parquet")
    assert len(results) == 12  # two prompts, six conditions each
    assert results.status.eq("ok").all()
    assert results.input_sha256.str.len().eq(64).all()
    assert set(results.condition) == {
        "clean",
        "identity",
        "gain_plus",
        "gain_minus",
        "amplify",
        "random",
    }
    routing = results[results.condition.isin(["gain_plus", "gain_minus", "random"])]
    np.testing.assert_allclose(
        routing.probe_score, routing.clean_probe_score, atol=1e-6
    )
    assert len(namespace["diagnostic_rows"]) == 4
    assert all(parameter.grad is None for parameter in model.parameters())
    summary, contrasts = summarize_conditions(results, seed=1729)
    assert summary.n_problems.eq(1).all()
    assert contrasts.n_problems.eq(1).all()
    assert contrasts.low.isna().all()  # Do not invent uncertainty from one problem.


def test_zero_content_is_reported_not_counted_as_a_wrong_answer(model, tokenizer):
    model.requires_grad_(False)
    prompt = paired_prompts()[0]
    config = ProbeConfig(InferenceConfig.direct(max_new_tokens=1))
    mapping = prompt_hidden_map(
        model, tokenizer, prompt.text, layer=0, blocks=model.model.layers, config=config
    )
    probe = {
        "weight": torch.ones(8),
        "bias": torch.tensor(0.0),
        "training_mean": mapping.hidden.cpu(),
    }
    records, checks = measure_prompt(
        model,
        tokenizer,
        prompt,
        probe,
        layer=0,
        blocks=model.model.layers,
        config=config,
        relative_strengths=[0.001],
        control_seeds=[11],
        fd_relative_steps=[0.001],
    )
    undefined = [row for row in records if row["status"] == "zero_content"]
    assert len(undefined) == 3
    assert all("correct" not in row for row in undefined)
    assert checks == []
    assert records[0]["condition"] == "clean" and "correct" in records[0]


def test_evaluation_rejects_changed_development_artifacts(tmp_path):
    cells = {
        cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells
    }
    (tmp_path / "run_metadata.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "frozen_settings.json").write_text(
        json.dumps({"development_hashes": {"conditions.parquet": "expected"}})
    )
    with pytest.raises(ValueError, match="Development artifact changed"):
        exec(
            cells["load-frozen-settings"],
            dict(
                PHASE="evaluation",
                DEVELOPMENT_DIR=tmp_path,
                json=json,
                file_sha256=lambda path: "changed",
            ),
        )


def test_failed_first_prompt_blocks_later_cells():
    cells = {
        cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells
    }
    # Simulate a rerun with previous results still in the notebook namespace.
    namespace = dict(first_prompt_passed=True, cohort_complete=True)
    with pytest.raises(NameError):
        exec(cells["measure-first-prompt"], namespace)
    for cell, message in (
        ("measure-cohort", "First-prompt checks must succeed"),
        ("summaries", "Complete the cohort before summarizing"),
        ("freeze-settings", "Complete the cohort before freezing"),
    ):
        namespace["FREEZE_STRENGTH"] = 0.001
        with pytest.raises(RuntimeError, match=message):
            exec(cells[cell], namespace)


def test_accuracy_contrasts_average_random_seeds_within_problems():
    rows = []
    for problem in range(3):
        for ctx in (500, 3000):
            for condition, random_seed, correct in [
                ("clean", -1, False),
                ("gain_plus", -1, True),
                ("random", 11, True),
                ("random", 29, False),
            ]:
                rows.append(
                    dict(
                        prompt_id=f"{problem}-{ctx}",
                        problem_id=problem,
                        ctx_size=ctx,
                        condition=condition,
                        random_seed=random_seed,
                        correct=correct,
                        relative_strength=0 if condition == "clean" else 0.001,
                        sensitivity=1.0,
                        clean_sensitivity=1.0,
                        probe_score=2.0,
                        clean_probe_score=2.0,
                        status="ok",
                    )
                )
    _, contrasts = summarize_conditions(pd.DataFrame(rows), seed=1)
    random = contrasts[
        (contrasts.condition == "gain_plus") & (contrasts.reference == "random")
    ]
    assert random.n_problems.eq(3).all()
    assert random[random.comparison != "long_minus_short"]["mean"].eq(0.5).all()
    assert random[random.comparison == "long_minus_short"]["mean"].eq(0).all()


def test_freezing_requires_development_derivatives_and_preserved_scores(tmp_path):
    cells = {
        cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells
    }
    results = pd.DataFrame(
        [
            dict(
                prompt_id="p",
                condition=condition,
                relative_strength=0.001,
                gain_norm=1.0,
                clean_content_norm=1.0,
                status="ok",
                probe_score=2.0,
                clean_probe_score=2.0,
                sensitivity=sensitivity,
                clean_sensitivity=1.0,
            )
            for condition, sensitivity in [("gain_plus", 1.1), ("gain_minus", 0.9)]
        ]
    )
    saved = []
    namespace = dict(
        PHASE="development",
        cohort_complete=True,
        FREEZE_STRENGTH=0.001,
        settings={
            "relative_strengths": [0.001, 0.003],
            "probe_score_atol": 1e-4,
            "probe_score_rtol": 1e-4,
        },
        results=results,
        checks=pd.DataFrame(
            [dict(prompt_id="p", sensitivity_matches=True, gain_matches=True)]
        ),
        np=np,
        RESULT_DIR=tmp_path,
        input_hashes={"probes.pt": "checkpoint"},
        implementation_hashes={"code": "hash"},
        file_sha256=lambda path: "digest",
        write_results=lambda path, value: saved.append(value),
    )
    exec(cells["freeze-settings"], namespace)
    assert saved[0]["settings"]["relative_strengths"] == [0.001]
    namespace["PHASE"] = "evaluation"
    with pytest.raises(ValueError, match="tested in development"):
        exec(cells["freeze-settings"], namespace)
    namespace["PHASE"] = "development"
    results.loc[0, "probe_score"] = 3.0
    with pytest.raises(ValueError, match="preservation failed"):
        exec(cells["freeze-settings"], namespace)
    results.loc[0, "probe_score"] = 2.0
    namespace["checks"]["gain_matches"] = False
    with pytest.raises(ValueError, match="Derivative checks"):
        exec(cells["freeze-settings"], namespace)
