from jlens_reasoning.experiments.readout_sanity import (
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    READOUT_CASES,
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
