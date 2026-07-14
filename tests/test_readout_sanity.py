import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jlens_reasoning.experiments.readout_sanity import (
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    READOUT_CASES,
    ReadoutCase,
    analyze_case,
    best_target_rank,
    concept_token_variants,
    find_last_subsequence,
    positions_after_literal,
    run_readout_sanity,
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


def test_cases_cover_exact_spider_and_france_prompts() -> None:
    cases = {case.key: case for case in READOUT_CASES}

    assert cases["spider"].prompt == (
        "The number of legs on the animal that spins webs is"
    )
    assert cases["spider"].expected_answers == ("8", "eight")
    assert cases["spider"].target_concepts == ("spider",)
    assert cases["spider"].literal_argument is None

    france = [case for case in READOUT_CASES if case.key.startswith("france_")]
    assert len(france) == 4
    assert {case.expected_answers[0] for case in france} == {
        "Paris",
        "French",
        "Europe",
        "Euro",
    }
    assert all(case.target_concepts == ("France",) for case in france)
    assert all(case.literal_argument == "France" for case in france)


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

    assert result["checks"] == {"baseline_top1": True, "target_top_k": True}
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
