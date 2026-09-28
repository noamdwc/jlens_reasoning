from dataclasses import replace

import pandas as pd
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
