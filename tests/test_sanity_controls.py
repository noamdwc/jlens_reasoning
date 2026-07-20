import math

import pytest
import torch

from jlens_reasoning.experiments.sanity_controls import (
    CONTROL_SEEDS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    derive_subseed,
    log_rank_gain,
    matched_random_vectors,
    percentile,
    strict_percentile_gate,
)


def test_shared_control_definitions_are_fixed() -> None:
    assert CONTROL_SEEDS == (
        11,
        29,
        47,
        71,
        101,
        131,
        167,
        199,
        239,
        281,
        331,
        379,
        431,
        487,
        547,
        607,
    )
    assert IDENTITY_ATOL == 1e-6
    assert IDENTITY_RTOL == 1e-5


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


def test_percentile_uses_documented_linear_interpolation() -> None:
    values = [float(value) for value in range(16)]
    assert percentile(values, 0.95) == pytest.approx(14.25)


def test_percentile_gate_is_strict_and_not_significance_claim() -> None:
    values = [0.0] * 15 + [4.0]
    threshold = percentile(values, 0.95)
    equal = strict_percentile_gate(threshold, values, quantile=0.95)
    greater = strict_percentile_gate(threshold + 1e-12, values, quantile=0.95)

    assert equal["passed"] is False
    assert greater["passed"] is True
    assert equal["interpretation"] == (
        "deterministic sanity check; not statistical significance"
    )


def test_subseeds_are_stable_layer_and_role_specific() -> None:
    assert derive_subseed(11, 7, "source") == 3688398498245801101
    assert derive_subseed(11, 7, "source") != derive_subseed(11, 7, "target")
    assert derive_subseed(11, 7, "source") != derive_subseed(11, 8, "source")


def test_subseed_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="Role"):
        derive_subseed(11, 7, "other")


def test_random_vectors_are_deterministic_for_seed_and_order_independent() -> None:
    real = {
        7: (torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])),
        9: (torch.tensor([1.0, 0.0]), torch.tensor([5.0, 12.0])),
    }

    first, _ = matched_random_vectors(real, base_seed=CONTROL_SEEDS[0])
    second, _ = matched_random_vectors(
        dict(reversed(list(real.items()))), base_seed=CONTROL_SEEDS[0]
    )
    different, _ = matched_random_vectors(real, base_seed=CONTROL_SEEDS[1])

    for layer in real:
        assert torch.equal(first[layer][0], second[layer][0])
        assert torch.equal(first[layer][1], second[layer][1])
    assert not torch.equal(first[7][0], different[7][0])


def test_random_vectors_match_each_per_layer_role_norm() -> None:
    real = {
        2: (torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])),
        3: (torch.zeros(2), torch.tensor([5.0, 12.0])),
    }

    generated, norms = matched_random_vectors(real, base_seed=CONTROL_SEEDS[1])

    for layer in sorted(real):
        for role_index, role in enumerate(("source", "target")):
            real_vector = real[layer][role_index]
            random_vector = generated[layer][role_index]
            assert torch.linalg.vector_norm(random_vector).item() == pytest.approx(
                torch.linalg.vector_norm(real_vector).item(), abs=1e-6, rel=1e-5
            )
            assert norms[str(layer)][role]["matched"] is True
    assert torch.equal(generated[3][0], torch.zeros(2))


def test_random_vectors_restore_real_device_and_dtype() -> None:
    real = {
        2: (
            torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
            torch.tensor([0.0, 2.0], dtype=torch.bfloat16),
        )
    }

    generated, norms = matched_random_vectors(real, base_seed=CONTROL_SEEDS[2])

    for role_index, role in enumerate(("source", "target")):
        assert generated[2][role_index].device == real[2][role_index].device
        assert generated[2][role_index].dtype == real[2][role_index].dtype
        assert norms["2"][role]["device_matches"] is True
        assert norms["2"][role]["dtype_matches"] is True
