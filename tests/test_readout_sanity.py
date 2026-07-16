import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jlens_reasoning.experiments.readout_sanity import (
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    LensCoordinateSwapper,
    MODEL_NAME,
    READOUT_CASES,
    SWAP_CASES,
    ReadoutCase,
    SwapCase,
    TokenVariant,
    aggregate_capability_checks,
    analyze_case,
    best_target_rank,
    concept_token_variants,
    coordinate_swap,
    find_last_subsequence,
    jlens_vector,
    positions_after_literal,
    prepare_scoring_input,
    run_readout_sanity,
    single_token_surface,
    summarize_swap_logits,
    top_tokens,
    validate_model_lens,
    workspace_layers,
    write_results,
)


def test_released_artifact_coordinates_match_upstream_walkthrough() -> None:
    assert MODEL_NAME == "Qwen/Qwen3.5-4B"
    assert LENS_REPO == "neuronpedia/jacobian-lens"
    assert LENS_REVISION == "qwen-n1000"
    assert LENS_FILE.endswith("Qwen3.5-4B_jacobian_lens_n1000.pt")


def test_cases_cover_released_read_and_swap_examples() -> None:
    read_cases = {case.key: case for case in READOUT_CASES}
    swap_cases = {case.key: case for case in SWAP_CASES}

    assert read_cases["spider"].prompt == (
        "The number of legs on the animal that spins webs is"
    )
    assert read_cases["spider"].expected_answers == ("8", "eight")
    assert read_cases["spider"].target_concepts == ("spider",)

    assert [(case.key, case.target_answers[0]) for case in SWAP_CASES] == [
        ("spider", "6"),
        ("france_capital", "Beijing"),
        ("france_language", "Chinese"),
        ("france_continent", "Asia"),
        ("france_currency", "Yuan"),
    ]
    assert swap_cases["spider"].source_surface == " spider"
    assert swap_cases["spider"].target_surface == " ant"
    france_swaps = [case for case in SWAP_CASES if case.key.startswith("france_")]
    assert all(case.source_surface == " France" for case in france_swaps)
    assert all(case.target_surface == " China" for case in france_swaps)


def test_single_token_surface_is_strict() -> None:
    tokenizer = FakeTokenizer()

    assert single_token_surface(tokenizer, " France") == TokenVariant(
        token_id=17,
        surface=" France",
    )
    with pytest.raises(ValueError, match="exactly one token"):
        single_token_surface(tokenizer, " FRANCE")


class FakeTokenizer:
    def __init__(self) -> None:
        self.pieces = {
            "France": [7],
            " France": [17],
            "france": [8],
            " france": [18],
            "FRANCE": [9, 10],
            " FRANCE": [19, 20],
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces.get(text, [99, 100])

    def decode(
        self, token_ids: list[int], *, clean_up_tokenization_spaces: bool = False
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return f"token-{token_ids[0]}"


def test_concept_variants_keep_single_tokens_and_deduplicate() -> None:
    variants = concept_token_variants(FakeTokenizer(), ("France", "france"))

    assert [(variant.token_id, variant.surface) for variant in variants] == [
        (7, "France"),
        (17, " France"),
        (8, "france"),
        (18, " france"),
    ]


def test_find_last_subsequence_and_positions_after_literal() -> None:
    assert find_last_subsequence([1, 17, 2, 17, 3], ([7], [17])) == (3, 4)
    assert positions_after_literal(
        FakeTokenizer(), torch.tensor([[1, 17, 2, 3]]), "France"
    ) == [2, 3]


def test_rank_is_one_based_best_variant_and_stable_for_ties() -> None:
    logits = torch.tensor([0.0, 3.0, 3.0, 1.0])

    assert best_target_rank(logits, (2, 3)) == 2
    assert best_target_rank(logits, (3,)) == 3


def test_jlens_vector_composes_jacobian_and_unembedding() -> None:
    lens = SimpleNamespace(
        jacobians={1: torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    )
    unembedding = torch.tensor([[0.0, 0.0], [5.0, 6.0]])

    assert torch.equal(
        jlens_vector(lens, unembedding, layer=1, token_id=1),
        torch.tensor([23.0, 34.0]),
    )


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (0.0, [1.0, 0.0, 7.0]),
        (1.0, [0.0, 1.0, 7.0]),
        (2.0, [-1.0, 2.0, 7.0]),
    ],
)
def test_coordinate_swap_strength_and_orthogonal_component(
    alpha: float,
    expected: list[float],
) -> None:
    hidden = torch.tensor([1.0, 0.0, 7.0])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])

    actual = coordinate_swap(hidden, source, target, alpha=alpha)

    assert torch.allclose(actual, torch.tensor(expected))


def test_coordinate_swap_preserves_shape_and_dtype() -> None:
    hidden = torch.tensor(
        [[[1.0, 0.0], [0.5, 0.25]]],
        dtype=torch.bfloat16,
    )

    actual = coordinate_swap(
        hidden,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        alpha=1.0,
    )

    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype


class TensorBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TupleBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
        return hidden, "cache"


def test_swapper_patches_all_positions_and_preserves_tuple_members() -> None:
    blocks = nn.ModuleList([TensorBlock(), TupleBlock()])
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        1: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    with LensCoordinateSwapper(blocks, vectors, alpha=1.0):
        first = blocks[0](hidden)
        second, cache = blocks[1](hidden)

    assert torch.equal(first, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    assert torch.equal(second, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    assert cache == "cache"
    assert all(not block._forward_hooks for block in blocks)


def test_swapper_removes_hooks_after_exception() -> None:
    blocks = nn.ModuleList([TensorBlock()])
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }

    with pytest.raises(RuntimeError, match="stop"):
        with LensCoordinateSwapper(blocks, vectors, alpha=1.0):
            raise RuntimeError("stop")

    assert not blocks[0]._forward_hooks


def test_top_tokens_preserve_token_ids_and_logits() -> None:
    assert top_tokens(torch.tensor([0.0, 2.0, 1.0]), FakeTokenizer(), k=2) == [
        {"token_id": 1, "token": "token-1", "logit": 2.0},
        {"token_id": 2, "token": "token-2", "logit": 1.0},
    ]


def test_workspace_layers_use_inclusive_ceil_and_floor_bounds() -> None:
    assert workspace_layers(20, range(20)) == list(range(7, 17))
    assert workspace_layers(20, [0, 6, 7, 12, 16, 17, 19]) == [7, 12, 16]


def test_results_round_trip_without_tensors(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    write_results(output, {"rank": torch.tensor(3), "layers": torch.tensor([7, 8])})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "layers": [7, 8],
        "rank": 3,
    }


class RunnerTokenizer(FakeTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.pieces.update(
            {
                "spider": [2],
                " spider": [2],
                "Spider": [2],
                " Spider": [2],
                "SPIDER": [2],
                " SPIDER": [2],
                "8": [4],
                " 8": [4],
                "eight": [4],
                " eight": [4],
                "Eight": [4],
                " Eight": [4],
                "EIGHT": [4],
                " EIGHT": [4],
            }
        )


def test_swap_summary_uses_best_strength_and_clean_rank() -> None:
    tokenizer = RunnerTokenizer()
    tokenizer.pieces.update(
        {
            "6": [5],
            " 6": [5],
            "six": [5],
            " six": [5],
            "Six": [5],
            " Six": [5],
            "SIX": [5],
            " SIX": [5],
        }
    )
    clean = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    alpha_1 = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, 3.5])
    alpha_2 = torch.tensor([1.0, 0.0, -1.0, -2.0, -3.0, 5.0])

    result = summarize_swap_logits(
        clean,
        {1.0: alpha_1, 2.0: alpha_2},
        clean_answers=("8", "eight"),
        target_answers=("6", "six"),
        tokenizer=tokenizer,
        top_k=3,
    )

    assert result["clean"]["expected_rank"] == 5
    assert result["clean"]["expected_top1"] is False
    assert result["clean"]["target_rank"] == 6
    assert result["interventions"]["1.0"]["target_rank"] == 2
    assert result["interventions"]["2.0"]["target_rank"] == 1
    assert result["best_intervened_rank"] == 1
    assert result["improved"] is True
    assert result["target_top1"] is True


class FormattingTokenizer(RunnerTokenizer):
    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return " " if token_ids[0] == 0 else f"token-{token_ids[0]}"


def test_scoring_input_appends_only_bounded_clean_formatting_tokens() -> None:
    calls: list[list[int]] = []

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        calls.append(input_ids[0].tolist())
        logits = torch.zeros(6)
        logits[0 if input_ids.shape[1] == 1 else 4] = 5.0
        return logits

    scoring_input, prefix = prepare_scoring_input(
        torch.tensor([[9]]),
        forward_next_token=forward_next_token,
        tokenizer=FormattingTokenizer(),
        max_formatting_tokens=2,
    )

    assert scoring_input.tolist() == [[9, 0]]
    assert prefix == [{"token_id": 0, "token": " "}]
    assert calls == [[9], [9, 0]]


def test_capability_gate_requires_three_improvements_and_one_top1() -> None:
    read_results = [
        {"key": "spider", "checks": {"baseline_top1": True, "read_capability": True}},
        {"key": "france_capital", "checks": {"baseline_top1": True}},
        {"key": "france_language", "checks": {"baseline_top1": True}},
        {"key": "france_continent", "checks": {"baseline_top1": True}},
        {"key": "france_currency", "checks": {"baseline_top1": True}},
    ]
    swap_results = [
        {"improved": True, "target_top1": True},
        {"improved": True, "target_top1": False},
        {"improved": True, "target_top1": False},
        {"improved": False, "target_top1": False},
        {"improved": False, "target_top1": False},
    ]

    checks, failures = aggregate_capability_checks(read_results, swap_results)

    assert checks == {
        "clean_baselines": True,
        "spider_read": True,
        "swap_rank_improvements": True,
        "swap_target_top1": True,
    }
    assert failures == []


def test_capability_gate_reports_aggregate_swap_failures() -> None:
    read_results = [
        {"key": "spider", "checks": {"baseline_top1": True, "read_capability": True}},
    ]
    swap_results = [
        {"improved": True, "target_top1": False},
        {"improved": True, "target_top1": False},
        {"improved": False, "target_top1": False},
    ]

    checks, failures = aggregate_capability_checks(read_results, swap_results)

    assert checks["swap_rank_improvements"] is False
    assert checks["swap_target_top1"] is False
    assert failures == [
        "coordinate swaps improved 2/3 target ranks; need at least 3",
        "no coordinate swap placed its target answer at top-1",
    ]


class FakeLens:
    d_model = 4
    source_layers = [0, 1, 2, 3]
    n_prompts = 1000

    def apply(self, model, prompt, *, use_jacobian=True, **kwargs):
        del model, prompt, kwargs
        input_ids = torch.tensor([[0, 1, 3]])
        model_logits = torch.zeros(3, 6)
        model_logits[-1, 4] = 9.0
        layer_logits = {}
        for layer in self.source_layers:
            logits = torch.zeros(3, 6)
            logits[:, 2] = 8.0 if use_jacobian and layer == 2 else -1.0
            layer_logits[layer] = logits
        return layer_logits, model_logits, input_ids


def test_validate_model_lens_rejects_width_and_layer_mismatches() -> None:
    with pytest.raises(ValueError, match="residual width"):
        validate_model_lens(SimpleNamespace(n_layers=4, d_model=5), FakeLens())

    lens = FakeLens()
    lens.source_layers = [0, 4]
    with pytest.raises(ValueError, match="fitted layers"):
        validate_model_lens(SimpleNamespace(n_layers=4, d_model=4), lens)


def test_analyze_case_grades_baseline_and_spider_readout() -> None:
    case = ReadoutCase(
        key="spider",
        prompt="prompt",
        expected_answers=("8", "eight"),
        target_concepts=("spider",),
    )

    result = analyze_case(
        case,
        model=SimpleNamespace(n_layers=4, d_model=4),
        lens=FakeLens(),
        tokenizer=RunnerTokenizer(),
        top_k=3,
    )

    assert result["checks"] == {
        "paper_top1_hit": True,
        "read_capability": True,
    }
    assert result["summary"]["jacobian_lens"]["best_rank"] == 1
    assert result["summary"]["jacobian_lens"]["layer"] == 2
    assert result["summary"]["logit_lens"]["best_rank"] > 1
    assert result["passed"] is True


def test_run_readout_sanity_keeps_failed_case_details() -> None:
    case = ReadoutCase(
        key="wrong-baseline",
        prompt="prompt",
        expected_answers=("missing",),
        target_concepts=("spider",),
    )
    tokenizer = RunnerTokenizer()
    tokenizer.pieces["missing"] = [5]
    tokenizer.pieces[" missing"] = [5]

    result = run_readout_sanity(
        model=SimpleNamespace(n_layers=4, d_model=4),
        lens=FakeLens(),
        tokenizer=tokenizer,
        cases=(case,),
        top_k=3,
    )

    assert result["passed"] is False
    assert result["cases"][0]["checks"]["baseline_top1"] is False
    assert result["failures"] == ["wrong-baseline: baseline top-1 mismatch"]
