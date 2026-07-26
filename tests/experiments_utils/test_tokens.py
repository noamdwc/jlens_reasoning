import torch

from jlens_reasoning.evaluation_utils import best_token_rank, top_token_values
from jlens_reasoning.experiments_utils.tokens import (
    TokenVariant,
    concept_surfaces,
    concept_token_variants,
    find_last_subsequence,
    positions_after_literal,
    positions_from_literal,
    prepare_scoring_input,
    single_token_surface,
)


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
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return f"token-{token_ids[0]}"


class FormattingTokenizer(FakeTokenizer):
    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return " " if token_ids[0] == 0 else f"token-{token_ids[0]}"


def test_concept_surfaces_has_no_experiment_case_dependency() -> None:
    assert concept_surfaces("France") == (
        "France",
        " France",
        "france",
        " france",
        "FRANCE",
        " FRANCE",
    )


def test_single_token_surface_is_strict() -> None:
    tokenizer = FakeTokenizer()

    assert single_token_surface(tokenizer, " France") == TokenVariant(
        token_id=17,
        surface=" France",
    )
    try:
        single_token_surface(tokenizer, " FRANCE")
    except ValueError as exc:
        assert "exactly one token" in str(exc)
    else:
        raise AssertionError("multi-token surface was accepted")


def test_concept_variants_keep_single_tokens_and_deduplicate() -> None:
    variants = concept_token_variants(FakeTokenizer(), ("France", "france"))

    assert [(variant.token_id, variant.surface) for variant in variants] == [
        (7, "France"),
        (17, " France"),
        (8, "france"),
        (18, " france"),
    ]


def test_find_last_subsequence_and_literal_positions() -> None:
    tokenizer = FakeTokenizer()
    input_ids = torch.tensor([[1, 17, 2, 3]])

    assert find_last_subsequence([1, 17, 2, 17, 3], ([7], [17])) == (3, 4)
    assert positions_after_literal(tokenizer, input_ids, "France") == [2, 3]
    assert positions_from_literal(tokenizer, input_ids, "France") == [1, 2, 3]


def test_rank_is_one_based_best_variant_and_stable_for_ties() -> None:
    logits = torch.tensor([0.0, 3.0, 3.0, 1.0])

    assert best_token_rank(logits, (2, 3)) == 2
    assert best_token_rank(logits, (3,)) == 3


def test_top_tokens_preserve_token_ids_and_logits() -> None:
    assert top_token_values(
        torch.tensor([0.0, 2.0, 1.0]),
        FakeTokenizer(),
        k=2,
    ) == ((1, "token-1", 2.0), (2, "token-2", 1.0))


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
