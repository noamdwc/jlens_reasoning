from dataclasses import FrozenInstanceError

import pytest

from experiments.jlens_readout_sanity.experiment import (
    Case,
    InterventionSpec,
    ReadoutSpec,
    validate_cases,
)


def spider_case() -> Case:
    return Case(
        key="spider",
        prompt="The number of legs on the animal that spins webs is",
        expected_answers=("8", "eight"),
        readout=ReadoutSpec(
            concepts=("spider",),
            require_capability_gate=True,
        ),
        intervention=InterventionSpec(
            source_surface=" spider",
            target_surface=" ant",
            target_answers=("6", "six"),
        ),
    )


def test_case_composes_readout_and_intervention_without_kind_dispatch() -> None:
    case = spider_case()

    assert case.readout is not None
    assert case.readout.require_capability_gate
    assert case.intervention is not None
    assert case.intervention.alphas == (1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        case.key = "ant"  # type: ignore[misc]


@pytest.mark.parametrize(
    "cases",
    [
        (),
        (Case("", "prompt", ("answer",), readout=ReadoutSpec(("x",))),),
        (Case("x", "", ("answer",), readout=ReadoutSpec(("x",))),),
        (Case("x", "prompt", (), readout=ReadoutSpec(("x",))),),
        (Case("x", "prompt", ("answer",)),),
        (
            Case("x", "prompt", ("answer",), readout=ReadoutSpec(("x",))),
            Case("x", "other", ("answer",), readout=ReadoutSpec(("x",))),
        ),
    ],
)
def test_validate_cases_rejects_invalid_collections(cases: tuple[Case, ...]) -> None:
    with pytest.raises(ValueError):
        validate_cases(cases)


def test_validate_cases_accepts_readout_swap_and_combined_cases() -> None:
    cases = (
        Case("read", "p1", ("a",), readout=ReadoutSpec(("concept",))),
        Case(
            "swap",
            "p2",
            ("b",),
            intervention=InterventionSpec(" source", " target", ("c",)),
        ),
        spider_case(),
    )

    validate_cases(cases)
