import math

import pytest

from jlens_reasoning.experiments.sanity_controls import (
    CONTROL_SEEDS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    derive_subseed,
    log_rank_gain,
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
