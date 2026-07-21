import inspect

from jlens_reasoning.experiments import (
    readout_constants,
    readout_sanity,
    readout_utils,
)


def test_readout_constants_have_focused_ownership_and_legacy_aliases() -> None:
    expected = {
        "MODEL_NAME": "Qwen/Qwen3.5-4B",
        "LENS_REPO": "neuronpedia/jacobian-lens",
        "LENS_REVISION": "qwen-n1000",
        "LENS_FILE": (
            "qwen3.5-4b/jlens/Salesforce-wikitext/"
            "Qwen3.5-4B_jacobian_lens_n1000.pt"
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
