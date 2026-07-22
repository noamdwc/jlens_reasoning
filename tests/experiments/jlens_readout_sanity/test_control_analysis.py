import math

import pytest

from experiments.jlens_readout_sanity.constants import IDENTITY_ATOL, IDENTITY_RTOL

EXPECTED_CASE_KEYS = (
    "spider",
    "france_capital",
    "france_language",
    "france_continent",
    "france_currency",
)


def test_real_rank_gain_cases_selects_numeric_control_alpha() -> None:
    from experiments.jlens_readout_sanity.control_analysis import (
        real_rank_gain_cases,
    )

    swaps = [
        {
            "key": "spider",
            "clean": {"target_rank": 20},
            "interventions": {
                "2.0": {"target_rank": 1, "top1_id": 9},
                "1": {"target_rank": 5, "top1_id": 8},
            },
        }
    ]

    cases, real_mean = real_rank_gain_cases(swaps, expected_keys=("spider",))

    assert cases == [
        {
            "key": "spider",
            "clean_rank": 20,
            "intervened_rank": 5,
            "intervened_top1_id": 8,
            "log_rank_gain": pytest.approx(math.log(4)),
        }
    ]
    assert real_mean == pytest.approx(math.log(4))


def test_assemble_control_results_preserves_the_serialized_envelope() -> None:
    from experiments.jlens_readout_sanity.control_analysis import (
        assemble_control_results,
    )

    passing = {"passed": True}
    result = assemble_control_results(
        expected_keys=EXPECTED_CASE_KEYS,
        identity=passing,
        matched_random_vector=passing,
        wrong_concept=passing,
        random_target=passing,
    )

    assert set(result) == {
        "seeds",
        "definitions",
        "thresholds",
        "tolerances",
        "identity",
        "matched_random_vector",
        "wrong_concept",
        "random_target",
        "passed",
    }
    assert result["definitions"]["expected_case_keys"] == list(EXPECTED_CASE_KEYS)
    assert result["thresholds"]["percentile_quantile"] == 0.95
    assert result["tolerances"]["identity_logits"] == {
        "atol": IDENTITY_ATOL,
        "rtol": IDENTITY_RTOL,
    }
    assert result["passed"] is True
