"""FLenQA cohort selection, condition measurements, and paired summaries.

Derivative, intervention, and uncertainty mechanics live in jlens_reasoning.
This module keeps the notebook's experiment stages short and inspectable.
"""

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from experiments.flenqa_probe_jlens.analysis import matched_prompt_pairs
from jlens_reasoning.evaluation import evaluate_paper_binary
from jlens_reasoning.experiments_utils.controls import random_orthogonal_direction
from jlens_reasoning.experiments_utils.interventions import add_prompt_delta
from jlens_reasoning.experiments_utils.statistics import paired_problem_bootstrap
from jlens_reasoning.inference import generate_chat
from jlens_reasoning.probe_jlens import (
    j_gain,
    j_sensitivity,
    probe_component,
    prompt_hidden_map,
)
from jlens_reasoning.probing import score_probe, unit_probe_direction


def select_pairs(prompts, problem_to_split, *, phase, seed, short_ctx, long_ctx):
    """One random rendering per problem; two per task/label for development.

    Evaluation includes every eligible test problem. Selection never uses
    correctness, probe scores, or derivatives. Missing pairs remain in coverage.
    """
    if phase not in ("development", "evaluation"):
        raise ValueError("Phase must be development or evaluation")
    partition = "validation" if phase == "development" else "test"
    eligible = tuple(p for p in prompts if problem_to_split[p.problem_id] == partition)
    pairs = pd.DataFrame(
        matched_prompt_pairs(eligible, short_ctx=short_ctx, long_ctx=long_ctx)
    )
    if pairs.empty:
        raise ValueError("No matched prompts in the requested partition")
    selected = pairs.sample(frac=1, random_state=seed).drop_duplicates("problem_id")
    if phase == "development":
        selected = selected.groupby(["task", "label"], group_keys=False).head(2)
    selected = selected.sort_values("problem_id").reset_index(drop=True)
    counts = pairs.groupby("problem_id").size()
    selected_ids = set(selected.problem_id)
    coverage = pd.DataFrame(
        [
            {
                "problem_id": problem,
                "partition": partition,
                "available_pairs": int(counts.get(problem, 0)),
                "selected": problem in selected_ids,
                "reason": "selected"
                if problem in selected_ids
                else (
                    "no_matched_pair"
                    if problem not in counts
                    else "development_subsample"
                ),
            }
            for problem, split in sorted(problem_to_split.items())
            if split == partition
        ]
    )
    return selected, coverage


def measure_prompt(
    model,
    tokenizer,
    prompt,
    probe,
    *,
    layer,
    blocks,
    config,
    relative_strengths,
    control_seeds,
    fd_relative_steps,
):
    """Measure all conditions for one FLenQA prompt; empty FD steps skip checks.

    Strength is delta norm / clean hidden norm. Undefined content/gain produces
    explicit rows without invented answers. Unsupported derivatives propagate
    errors: no prompt is silently skipped and no fallback objective is used.
    """
    started = time.perf_counter()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    clean_answer = generate_chat(model, tokenizer, prompt.text, config=config.inference)
    mapping = prompt_hidden_map(
        model,
        tokenizer,
        prompt.text,
        layer=layer,
        blocks=blocks,
        config=config,
        input_record={
            "input_sha256": clean_answer.input_sha256,
            "n_input_tokens": clean_answer.input_token_count,
        },
    )
    hidden = mapping.hidden
    clean_probe_score = float(score_probe(probe, hidden))
    axis = unit_probe_direction(probe).to(hidden)
    component = probe_component(probe, hidden)
    content_norm, hidden_norm = float(component.norm()), float(hidden.norm())
    gain_result = (
        j_gain(mapping.final_hidden, hidden, component) if content_norm else None
    )
    gain_norm = float(gain_result.gain.norm()) if gain_result else 0.0
    clean_s = gain_result.sensitivity if gain_result else np.nan
    gain_unit = gain_result.gain / gain_norm if gain_norm else None
    content_unit = component / content_norm if content_norm else None
    diagnostics = []
    if content_norm and gain_norm:
        for relative_step in fd_relative_steps:
            eps = relative_step * hidden_norm
            plus = mapping.final_hidden(hidden + (eps * content_unit).to(hidden))
            minus = mapping.final_hidden(hidden - (eps * content_unit).to(hidden))
            fd_s = float(((plus - minus) / (2 * eps)).norm())
            upper = j_sensitivity(
                mapping.final_hidden, hidden + (eps * gain_unit).to(hidden), component
            )
            lower = j_sensitivity(
                mapping.final_hidden, hidden - (eps * gain_unit).to(hidden), component
            )
            fd_gain = (upper**2 - lower**2) / (4 * eps)
            diagnostics.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "relative_step": relative_step,
                    "analytic_sensitivity": clean_s,
                    "fd_sensitivity": fd_s,
                    "analytic_gain_derivative": gain_norm,
                    "fd_gain_derivative": fd_gain,
                }
            )

    directions = [
        ("gain_plus", -1, gain_unit),
        ("gain_minus", -1, -gain_unit if gain_unit is not None else None),
        ("amplify", -1, content_unit),
    ]
    directions.extend(
        ("random", seed, random_orthogonal_direction(axis, seed=seed))
        for seed in control_seeds
    )
    conditions = [
        ("clean", 0.0, -1, torch.zeros_like(hidden)),
        ("identity", 0.0, -1, torch.zeros_like(hidden)),
    ]
    conditions.extend(
        (
            name,
            strength,
            seed,
            direction * (strength * hidden_norm) if direction is not None else None,
        )
        for strength in relative_strengths
        for name, seed, direction in directions
    )
    records = []
    for name, strength, seed, delta in conditions:
        row = {
            "prompt_id": prompt.prompt_id,
            "problem_id": prompt.problem_id,
            "task": prompt.task,
            "label": int(prompt.label),
            "ctx_size": prompt.provenance[0].ctx_size,
            "layer": layer,
            "condition": name,
            "relative_strength": strength,
            "random_seed": seed,
            **mapping.input_record,
            "clean_sensitivity": clean_s,
            "clean_probe_score": clean_probe_score,
            "clean_hidden_norm": hidden_norm,
            "clean_content_norm": content_norm,
            "gain_norm": gain_norm,
            "status": "ok"
            if delta is not None
            else ("zero_content" if not content_norm else "zero_gain"),
        }
        if delta is not None:
            edited = hidden + delta.to(hidden)
            realized = edited - hidden
            if name == "clean":
                answer = clean_answer
            else:
                with add_prompt_delta(
                    blocks[layer],
                    delta,
                    prompt_length=clean_answer.input_token_count,
                    token_position=config.token_position,
                ):
                    answer = generate_chat(
                        model, tokenizer, prompt.text, config=config.inference
                    )
            if (
                name == "identity"
                and answer.output.token_ids != clean_answer.output.token_ids
            ):
                raise ValueError("Zero intervention changed generated tokens")
            sensitivity = (
                clean_s
                if name in ("clean", "identity")
                else (
                    j_sensitivity(mapping.final_hidden, edited, component)
                    if content_norm
                    else np.nan
                )
            )
            verdict = evaluate_paper_binary(answer.output, expected=prompt.label)
            row.update(
                probe_score=float(score_probe(probe, edited)),
                signed_projection=float(
                    (edited - probe["training_mean"].to(hidden)) @ axis
                ),
                content_norm=float(probe_component(probe, edited).norm()),
                sensitivity=sensitivity,
                delta_norm=float(realized.norm()),
                requested_delta_norm=float(delta.norm()),
                hidden_norm=float(edited.norm()),
                answer=answer.raw_text,
                generated_token_ids=list(answer.output.token_ids),
                generation_status=answer.output.generation_status.value,
                correct=verdict.correct,
                parsed_answer=verdict.verdict,
            )
        records.append(row)
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(model.device)
        if model.device.type == "cuda"
        else None
    )
    for row in records:
        row.update(prompt_elapsed_seconds=elapsed, peak_cuda_bytes=peak)
    return records, diagnostics


def summarize_conditions(results, *, seed):
    """Return per-condition diagnostics and problem-paired accuracy contrasts."""
    measured = results[results.status == "ok"].copy()
    clean = measured[measured.condition == "clean"].set_index("prompt_id")
    baseline_correct = measured.prompt_id.map(clean.correct).astype(bool)
    measured["accuracy_delta"] = measured.correct.astype(
        float
    ) - baseline_correct.astype(float)
    measured["sensitivity_delta"] = measured.sensitivity - measured.clean_sensitivity
    measured["probe_score_drift"] = (
        measured.probe_score - measured.clean_probe_score
    ).abs()
    measured["repaired"] = measured.correct.astype(bool) & ~baseline_correct
    measured["damaged"] = ~measured.correct.astype(bool) & baseline_correct
    keys = ["condition", "relative_strength", "ctx_size"]
    per_problem = measured.groupby([*keys, "problem_id"], as_index=False).agg(
        accuracy_delta=("accuracy_delta", "mean"),
        sensitivity_delta=("sensitivity_delta", "mean"),
        probe_score_drift=("probe_score_drift", "max"),
        repaired=("repaired", "mean"),
        damaged=("damaged", "mean"),
    )
    summary = per_problem.groupby(keys, as_index=False).agg(
        n_problems=("problem_id", "nunique"),
        n_sensitivity_problems=("sensitivity_delta", "count"),
        accuracy_delta=("accuracy_delta", "mean"),
        sensitivity_delta=("sensitivity_delta", "mean"),
        max_probe_score_drift=("probe_score_drift", "max"),
        repair_rate=("repaired", "mean"),
        damage_rate=("damaged", "mean"),
    )
    contrasts = []
    for (condition, strength), group in per_problem.groupby(
        ["condition", "relative_strength"]
    ):
        for reference in ("clean", "random"):
            if condition in ("clean", "identity") or condition == reference:
                continue
            paired = group.copy()
            if reference == "random":
                random = per_problem[
                    (per_problem.condition == "random")
                    & (per_problem.relative_strength == strength)
                ]
                paired = paired.merge(
                    random[["problem_id", "ctx_size", "accuracy_delta"]],
                    on=["problem_id", "ctx_size"],
                    suffixes=("", "_random"),
                    validate="one_to_one",
                )
                paired["accuracy_delta"] -= paired.accuracy_delta_random
            for ctx, subset in paired.groupby("ctx_size"):
                interval = (
                    asdict(
                        paired_problem_bootstrap(
                            subset.problem_id.tolist(),
                            subset.accuracy_delta.tolist(),
                            seed=seed,
                        )
                    )
                    if len(subset) >= 2
                    else {
                        "mean": float(subset.accuracy_delta.mean()),
                        "low": None,
                        "high": None,
                        "n_problems": len(subset),
                    }
                )
                contrasts.append(
                    {
                        "condition": condition,
                        "relative_strength": strength,
                        "reference": reference,
                        "comparison": str(ctx),
                        **interval,
                    }
                )
            wide = paired.pivot(
                index="problem_id", columns="ctx_size", values="accuracy_delta"
            )
            if len(wide.columns) == 2:
                short, long = sorted(wide.columns)
                matched = wide.dropna()
                if len(matched) >= 2:
                    interval = paired_problem_bootstrap(
                        matched.index.tolist(),
                        (matched[long] - matched[short]).tolist(),
                        seed=seed,
                    )
                    contrasts.append(
                        {
                            "condition": condition,
                            "relative_strength": strength,
                            "reference": reference,
                            "comparison": "long_minus_short",
                            **asdict(interval),
                        }
                    )
    return summary, pd.DataFrame(contrasts)
