import inspect

from jlens_reasoning.experiments import (
    readout_constants,
    readout_sanity,
    readout_utils,
    sanity_constants,
    sanity_controls,
)


def test_readout_constants_have_focused_ownership_and_legacy_aliases() -> None:
    expected = {
        "MODEL_NAME": "Qwen/Qwen3.5-4B",
        "LENS_REPO": "neuronpedia/jacobian-lens",
        "LENS_REVISION": "qwen-n1000",
        "LENS_FILE": (
            "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
        ),
        "TOP_K": 25,
        "WORKSPACE_LAYER_LOWER_FRACTION": 0.35,
        "WORKSPACE_LAYER_UPPER_FRACTION": 0.80,
        "DEFAULT_INTERVENTION_STRENGTHS": (1.0, 2.0),
        "DEFAULT_MINIMUM_IMPROVEMENTS": 3,
        "DEFAULT_MAX_FORMATTING_TOKENS": 2,
        "SPIDER_READ_MAX_RANK": 5,
    }
    for name, value in expected.items():
        assert getattr(readout_constants, name) == value

    for name in ("MODEL_NAME", "LENS_REPO", "LENS_REVISION", "LENS_FILE", "TOP_K"):
        assert getattr(readout_sanity, name) is getattr(readout_constants, name)
    assert readout_utils.TOP_K is readout_constants.TOP_K

    run_defaults = inspect.signature(readout_sanity.run_readout_sanity).parameters
    assert (
        run_defaults["alphas"].default
        is readout_constants.DEFAULT_INTERVENTION_STRENGTHS
    )
    assert (
        run_defaults["minimum_improvements"].default
        is readout_constants.DEFAULT_MINIMUM_IMPROVEMENTS
    )


def test_sanity_constants_have_focused_ownership_and_legacy_aliases() -> None:
    assert len(sanity_constants.CONTROL_SEEDS) == 16
    assert sanity_constants.CONTROL_CASE_KEYS == (
        "spider",
        "france_capital",
        "france_language",
        "france_continent",
        "france_currency",
    )
    assert sanity_constants.IDENTITY_ATOL == 1e-6
    assert sanity_constants.IDENTITY_RTOL == 1e-5
    assert sanity_constants.NORM_ATOL == 1e-6
    assert sanity_constants.NORM_RTOL == 1e-5
    assert sanity_constants.LOW_PRECISION_NORM_ATOL == 1e-2
    assert sanity_constants.LOW_PRECISION_NORM_RTOL == 1e-2
    assert sanity_constants.PERCENTILE_QUANTILE == 0.95
    assert sanity_constants.CONTROL_ALPHA == 1.0
    assert sanity_constants.WRONG_CONCEPT_REQUIRED_CASE_WINS == 4
    assert sanity_constants.MAX_RANDOM_VECTOR_ATTEMPTS == 1024

    legacy_names = (
        "CONTROL_SEEDS",
        "CONTROL_CASE_KEYS",
        "IDENTITY_ATOL",
        "IDENTITY_RTOL",
        "NORM_ATOL",
        "NORM_RTOL",
        "PERCENTILE_QUANTILE",
        "PERCENTILE_INTERPRETATION",
        "CONTROL_CHECK_MAP",
    )
    for name in legacy_names:
        assert getattr(sanity_controls, name) is getattr(sanity_constants, name)
