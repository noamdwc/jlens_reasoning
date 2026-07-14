import json
from pathlib import Path

import torch

from jlens_reasoning.experiments.readout_sanity import (
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    READOUT_CASES,
    best_target_rank,
    concept_token_variants,
    find_last_subsequence,
    positions_after_literal,
    top_tokens,
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
