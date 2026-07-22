import math
from types import SimpleNamespace

import pytest
import torch

from jlens_reasoning.evaluation_utils import log_rank_gain
from jlens_reasoning.experiments_utils.controls import (
    build_random_target_exclusions,
    derive_subseed,
    matched_random_vectors,
    mean,
    percentile,
    percentile_label,
    require_exact_case_keys,
    select_random_targets,
    strict_percentile_gate,
)

SEEDS = (11, 29, 47, 71, 101, 131, 167, 199, 239, 281, 331, 379, 431, 487, 547, 607)
VECTOR_NAMESPACE = "jlens-control-v1"
TARGET_NAMESPACE = "jlens-random-target-v1"
VECTOR_OPTIONS = {
    "namespace": VECTOR_NAMESPACE,
    "norm_atol": 1e-6,
    "norm_rtol": 1e-5,
    "low_precision_norm_atol": 1e-2,
    "low_precision_norm_rtol": 1e-2,
    "max_attempts": 1024,
}


def test_log_rank_gain_uses_natural_logarithms() -> None:
    assert log_rank_gain(100, 10) == pytest.approx(math.log(10.0))
    assert log_rank_gain(10, 100) == pytest.approx(-math.log(10.0))
    assert log_rank_gain(7, 7) == 0.0


@pytest.mark.parametrize(("clean_rank", "intervened_rank"), [(0, 1), (1, 0)])
def test_log_rank_gain_requires_positive_ranks(
    clean_rank: int, intervened_rank: int
) -> None:
    with pytest.raises(ValueError, match="positive"):
        log_rank_gain(clean_rank, intervened_rank)


def test_mean_and_percentile_reject_empty_sequences() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean([])
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.95)


def test_percentile_uses_linear_interpolation() -> None:
    values = [float(value) for value in range(16)]
    assert percentile(values, 0.95) == pytest.approx(14.25)
    assert percentile_label(0.95) == "95th-percentile"


def test_percentile_gate_is_strict_and_caller_described() -> None:
    values = [0.0] * 15 + [4.0]
    interpretation = "deterministic sanity check; not statistical significance"
    threshold = percentile(values, 0.95)

    equal = strict_percentile_gate(
        threshold,
        values,
        quantile=0.95,
        interpretation=interpretation,
    )
    greater = strict_percentile_gate(
        threshold + 1e-12,
        values,
        quantile=0.95,
        interpretation=interpretation,
    )

    assert equal["passed"] is False
    assert greater["passed"] is True
    assert equal["interpretation"] == interpretation


def test_subseeds_are_stable_layer_and_role_specific() -> None:
    assert (
        derive_subseed(11, 7, "source", namespace=VECTOR_NAMESPACE)
        == 3688398498245801101
    )
    assert derive_subseed(
        11, 7, "source", namespace=VECTOR_NAMESPACE
    ) != derive_subseed(11, 7, "target", namespace=VECTOR_NAMESPACE)
    assert derive_subseed(
        11, 7, "source", namespace=VECTOR_NAMESPACE
    ) != derive_subseed(11, 8, "source", namespace=VECTOR_NAMESPACE)


def test_subseed_rejects_unknown_role_and_empty_namespace() -> None:
    with pytest.raises(ValueError, match="Role"):
        derive_subseed(11, 7, "other", namespace=VECTOR_NAMESPACE)
    with pytest.raises(ValueError, match="namespace"):
        derive_subseed(11, 7, "source", namespace="")


def test_random_vectors_are_deterministic_and_order_independent() -> None:
    real = {
        7: (torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])),
        9: (torch.tensor([1.0, 0.0]), torch.tensor([5.0, 12.0])),
    }

    first, _ = matched_random_vectors(real, base_seed=SEEDS[0], **VECTOR_OPTIONS)
    second, _ = matched_random_vectors(
        dict(reversed(list(real.items()))),
        base_seed=SEEDS[0],
        **VECTOR_OPTIONS,
    )
    different, _ = matched_random_vectors(real, base_seed=SEEDS[1], **VECTOR_OPTIONS)

    for layer in real:
        assert torch.equal(first[layer][0], second[layer][0])
        assert torch.equal(first[layer][1], second[layer][1])
    assert not torch.equal(first[7][0], different[7][0])


def test_random_vectors_match_norm_device_and_dtype() -> None:
    real = {
        2: (
            torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
            torch.tensor([0.0, 2.0], dtype=torch.bfloat16),
        ),
        3: (torch.zeros(2), torch.tensor([5.0, 12.0])),
    }

    generated, norms = matched_random_vectors(
        real, base_seed=SEEDS[1], **VECTOR_OPTIONS
    )

    for layer in sorted(real):
        for role_index, role in enumerate(("source", "target")):
            real_vector = real[layer][role_index]
            random_vector = generated[layer][role_index]
            assert random_vector.device == real_vector.device
            assert random_vector.dtype == real_vector.dtype
            assert norms[str(layer)][role]["matched"] is True
            assert norms[str(layer)][role]["device_matches"] is True
            assert norms[str(layer)][role]["dtype_matches"] is True
    assert torch.equal(generated[3][0], torch.zeros(2))


def test_random_vectors_remain_finite_after_low_precision_cast() -> None:
    real = {
        0: (
            torch.tensor([60000.0, 60000.0], dtype=torch.float16),
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
    }

    generated, norms = matched_random_vectors(real, base_seed=29, **VECTOR_OPTIONS)

    assert torch.isfinite(generated[0][0]).all()
    assert norms["0"]["source"]["matched"] is True


def test_random_vectors_require_positive_attempt_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        matched_random_vectors(
            {0: (torch.ones(2), torch.ones(2))},
            base_seed=11,
            **{**VECTOR_OPTIONS, "max_attempts": 0},
        )


class VocabularyTokenizer:
    def __init__(self) -> None:
        self.vocab = {f"token-{token_id}": token_id for token_id in range(40)}
        self.pieces = {
            "spider": [1],
            "France": [2],
            "ant": [3],
            "China": [4],
            "8": [5],
            "Paris": [6],
            "six": [7],
            "New Yuan": [10, 11],
        }
        self.all_special_ids = [0, 13, 14]
        self.added_tokens_decoder = {
            15: SimpleNamespace(special=True),
            16: SimpleNamespace(special=False),
        }
        self.decode_calls: list[tuple[tuple[int, ...], bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces[text]

    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        self.decode_calls.append((tuple(token_ids), clean_up_tokenization_spaces))
        token_id = token_ids[0]
        if token_id == 8:
            return " "
        if token_id == 12:
            return "\n"
        return f"token-{token_id}"

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)


def test_random_target_exclusions_are_token_id_based_and_complete() -> None:
    tokenizer = VocabularyTokenizer()

    exclusions = build_random_target_exclusions(
        tokenizer,
        source_surfaces=("spider", "France"),
        target_surfaces=("ant", "China"),
        clean_answer_surfaces=("8", "Paris"),
        intended_answer_surfaces=("six", "New Yuan"),
        formatting_token_ids=(8,),
        existing_excluded_ids=(9,),
    )

    assert exclusions == {
        "sources": [1, 2],
        "targets": [3, 4],
        "clean_answers": [5, 6],
        "intended_answers": [7, 10, 11],
        "formatting": [8],
        "reserved_special": [0, 13, 14, 15, 16],
        "decoded_formatting": [8, 12],
        "existing_filter": [9],
        "all": list(range(17)),
    }
    assert all(cleanup is False for _, cleanup in tokenizer.decode_calls)


def test_random_target_selection_is_deterministic_unique_and_bounded() -> None:
    tokenizer = VocabularyTokenizer()
    excluded_ids = range(16)

    first = select_random_targets(
        tokenizer,
        excluded_ids=excluded_ids,
        seeds=SEEDS,
        output_vocab_size=32,
        namespace=TARGET_NAMESPACE,
    )
    second = select_random_targets(
        tokenizer,
        excluded_ids=reversed(list(excluded_ids)),
        seeds=SEEDS,
        output_vocab_size=32,
        namespace=TARGET_NAMESPACE,
    )

    assert [item["token_id"] for item in first] == [
        20,
        26,
        21,
        18,
        19,
        28,
        30,
        27,
        22,
        17,
        25,
        16,
        24,
        29,
        31,
        23,
    ]
    assert first == second
    assert len({item["token_id"] for item in first}) == 16
    assert all(0 <= item["token_id"] < 32 for item in first)


def test_random_target_selection_rejects_invalid_input() -> None:
    tokenizer = VocabularyTokenizer()
    with pytest.raises(ValueError, match="eligible"):
        select_random_targets(
            tokenizer,
            excluded_ids=range(32),
            seeds=SEEDS,
            output_vocab_size=32,
            namespace=TARGET_NAMESPACE,
        )
    with pytest.raises(ValueError, match="namespace"):
        select_random_targets(
            tokenizer,
            excluded_ids=(),
            seeds=SEEDS,
            output_vocab_size=32,
            namespace="",
        )


def test_exact_case_keys_are_caller_owned() -> None:
    results = [{"key": "a"}, {"key": "b"}]
    require_exact_case_keys(results, expected_keys=("a", "b"))
    with pytest.raises(ValueError, match="Expected exact case keys"):
        require_exact_case_keys(results, expected_keys=("b", "a"))
