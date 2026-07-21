from experiments.jlens_readout_sanity.constants import (
    CONTROL_ALPHA,
    CONTROL_CASE_KEYS,
    CONTROL_CHECK_MAP,
    CONTROL_SEEDS,
    DEFAULT_INTERVENTION_STRENGTHS,
    DEFAULT_MAX_FORMATTING_TOKENS,
    DEFAULT_MINIMUM_IMPROVEMENTS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    LOW_PRECISION_NORM_ATOL,
    LOW_PRECISION_NORM_RTOL,
    MAX_RANDOM_VECTOR_ATTEMPTS,
    MODEL_NAME,
    NORM_ATOL,
    NORM_RTOL,
    PERCENTILE_INTERPRETATION,
    PERCENTILE_QUANTILE,
    RANDOM_TARGET_NAMESPACE,
    RANDOM_VECTOR_NAMESPACE,
    READOUT_CASES,
    SPIDER_READ_MAX_RANK,
    SWAP_CASES,
    TOP_K,
    WORKSPACE_LAYER_LOWER_FRACTION,
    WORKSPACE_LAYER_UPPER_FRACTION,
    WRONG_CONCEPT_REQUIRED_CASE_WINS,
)
from experiments.jlens_readout_sanity.types import ReadoutCase, SwapCase


def test_artifact_coordinates_and_readout_policy_are_fixed() -> None:
    assert MODEL_NAME == "Qwen/Qwen3.5-4B"
    assert LENS_REPO == "neuronpedia/jacobian-lens"
    assert LENS_REVISION == "qwen-n1000"
    assert LENS_FILE == (
        "qwen3.5-4b/jlens/Salesforce-wikitext/"
        "Qwen3.5-4B_jacobian_lens_n1000.pt"
    )
    assert TOP_K == 25
    assert WORKSPACE_LAYER_LOWER_FRACTION == 0.35
    assert WORKSPACE_LAYER_UPPER_FRACTION == 0.80
    assert DEFAULT_INTERVENTION_STRENGTHS == (1.0, 2.0)
    assert DEFAULT_MINIMUM_IMPROVEMENTS == 3
    assert DEFAULT_MAX_FORMATTING_TOKENS == 2
    assert SPIDER_READ_MAX_RANK == 5


def test_control_policy_is_fixed_and_namespaced() -> None:
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
    assert CONTROL_ALPHA == 1.0
    assert IDENTITY_ATOL == 1e-6
    assert IDENTITY_RTOL == 1e-5
    assert NORM_ATOL == 1e-6
    assert NORM_RTOL == 1e-5
    assert LOW_PRECISION_NORM_ATOL == 1e-2
    assert LOW_PRECISION_NORM_RTOL == 1e-2
    assert PERCENTILE_QUANTILE == 0.95
    assert PERCENTILE_INTERPRETATION == (
        "deterministic sanity check; not statistical significance"
    )
    assert WRONG_CONCEPT_REQUIRED_CASE_WINS == 4
    assert MAX_RANDOM_VECTOR_ATTEMPTS == 1024
    assert RANDOM_VECTOR_NAMESPACE == "jlens-control-v1"
    assert RANDOM_TARGET_NAMESPACE == "jlens-random-target-v1"
    assert CONTROL_CHECK_MAP == (
        ("identity", "identity_control"),
        ("matched_random_vector", "matched_random_vector_control"),
        ("wrong_concept", "wrong_concept_control"),
        ("random_target", "random_target_control"),
    )


def test_fixed_cases_preserve_order_and_derive_control_keys() -> None:
    assert all(isinstance(case, ReadoutCase) for case in READOUT_CASES)
    assert all(isinstance(case, SwapCase) for case in SWAP_CASES)
    assert tuple(case.key for case in READOUT_CASES) == (
        "spider",
        "france_capital",
        "france_language",
        "france_continent",
        "france_currency",
    )
    assert SWAP_CASES == (
        SwapCase("spider", " spider", " ant", ("6", "six")),
        SwapCase("france_capital", " France", " China", ("Beijing",)),
        SwapCase("france_language", " France", " China", ("Chinese",)),
        SwapCase("france_continent", " France", " China", ("Asia",)),
        SwapCase("france_currency", " France", " China", ("Yuan",)),
    )
    assert CONTROL_CASE_KEYS == tuple(case.key for case in SWAP_CASES)
