from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from jlens import JacobianLens

from experiments.flenqa_probe_jlens.analysis import (
    matched_prompt_pairs,
    static_probe_projection,
    validate_split,
)
from jlens_reasoning.benchmarks.flenqa.dataset import FlenqaPrompt, SourceProvenance
from jlens_reasoning.experiments_utils.interventions import jlens_vector


def split_fixture():
    rows = [
        SimpleNamespace(problem_id=i, task="PIR", label=bool(i % 2))
        for i in range(300)
        for _ in range(2)
    ]
    split = {
        "format_version": 1,
        "seed": 1729,
        "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "stratified_by": ["task", "label"],
        "problems": {
            "train": list(range(180)),
            "validation": list(range(180, 240)),
            "test": list(range(240, 300)),
        },
        "problem_metadata": {
            str(i): {"task": "PIR", "label": bool(i % 2)} for i in range(300)
        },
    }
    return rows, split


def test_split_keeps_related_variants_in_one_partition():
    rows, split = split_fixture()
    mapping = validate_split(split, rows)
    assert mapping[0] == "train"
    assert mapping[180] == "validation"
    assert mapping[299] == "test"
    assert len(mapping) == 300


@pytest.mark.parametrize(
    "corruption",
    ["leak", "duplicate", "missing", "metadata", "seed", "strata", "variant"],
)
def test_split_rejects_leakage_and_stale_assets(corruption):
    rows, split = split_fixture()
    if corruption == "leak":
        split["problems"]["test"][0] = 0
    elif corruption == "duplicate":
        split["problems"]["train"][1] = 0
    elif corruption == "missing":
        split["problems"]["test"].pop()
    elif corruption == "metadata":
        split["problem_metadata"]["0"]["label"] = True
    elif corruption == "seed":
        split["seed"] = 7
    elif corruption == "strata":
        split["stratified_by"] = ["label"]
    else:
        rows[0].label = True
    with pytest.raises(ValueError):
        validate_split(split, rows)


def prompt(identifier, ctx, *, problem_id=1, label=True, dispersions=("first",)):
    return FlenqaPrompt(
        canonical_index=0,
        prompt_id=identifier,
        problem_id=problem_id,
        task="PIR",
        text=identifier,
        question="Q",
        key_texts=("fact",),
        rule=None,
        label=label,
        mixin=identifier,
        provenance=tuple(
            SourceProvenance(i, ctx, "books", dispersion)
            for i, dispersion in enumerate(dispersions)
        ),
    )


def test_matching_uses_all_provenance_and_never_crosses_problems():
    prompts = [
        prompt("short", 500, dispersions=("first", "last")),
        prompt("long-first", 3000),
        prompt("long-last", 3000, dispersions=("last",)),
        prompt("unmatched", 500, problem_id=2),
    ]
    pairs = matched_prompt_pairs(prompts, short_ctx=500, long_ctx=3000)
    assert {(p["short_id"], p["long_id"]) for p in pairs} == {
        ("short", "long-first"),
        ("short", "long-last"),
    }
    assert {p["problem_id"] for p in pairs} == {1}


@pytest.mark.parametrize("corruption", ["label", "ambiguous", "question"])
def test_matching_rejects_inconsistent_pairs(corruption):
    short = prompt("short", 500)
    long = prompt("long", 3000)
    prompts = [short, long]
    if corruption == "label":
        prompts[1] = replace(long, label=False)
    elif corruption == "ambiguous":
        prompts.append(prompt("different-short", 500))
    else:
        prompts[1] = replace(long, question="Different problem")
    with pytest.raises(ValueError):
        matched_prompt_pairs(prompts, short_ctx=500, long_ctx=3000)


def test_projection_matches_transport_and_existing_jlens_vectors():
    # Nonsymmetric J catches an accidental transpose. ||w|| = 5.
    jacobian = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    unembedding = torch.tensor([[2.0, -1.0], [0.0, 3.0], [-1.0, 0.0]])
    lens = JacobianLens(jacobians={0: jacobian}, n_prompts=1, d_model=2)
    projected, scores = static_probe_projection(
        jacobian, torch.tensor([3.0, 4.0]), unembedding
    )
    torch.testing.assert_close(projected, torch.tensor([2.2, 5.0]))
    torch.testing.assert_close(scores, torch.tensor([-0.6, 15.0, -2.2]))
    direction = torch.tensor([0.6, 0.8])
    torch.testing.assert_close(projected, lens.transport(direction, 0))
    for token in range(3):
        pulled_back = jlens_vector(lens, unembedding, layer=0, token_id=token)
        torch.testing.assert_close(scores[token], pulled_back @ direction)
    # The same sign on gold output margin and probe direction cancels for False.
    for gold_sign in (-1, 1):
        h = torch.zeros(2, requires_grad=True)
        margin = gold_sign * ((unembedding[0] - unembedding[1]) @ jacobian @ h)
        (grad,) = torch.autograd.grad(margin, h)
        torch.testing.assert_close(
            grad @ (gold_sign * direction), scores[0] - scores[1]
        )


def test_zero_probe_direction_is_rejected():
    with pytest.raises(ValueError):
        static_probe_projection(torch.eye(2), torch.zeros(2), torch.eye(2))
