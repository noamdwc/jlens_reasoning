from dataclasses import FrozenInstanceError

import pytest

from experiments.jlens_readout_sanity.experiment import (
    Case,
    InterventionSpec,
    ReadoutSpec,
    generate_and_evaluate,
    validate_cases,
)
from jlens_reasoning.evaluation import (
    AnswerStatus,
    GenerationStatus,
    ModelOutput,
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


def test_generate_and_evaluate_uses_full_raw_output_and_think_parser() -> None:
    case = spider_case()
    generated = ModelOutput(
        text="<think>A spider has eight legs.</think>\n 8.",
        token_ids=(1, 2, 3),
        token_pieces=("<think>reason</think>", " ", "8."),
        generation_status=GenerationStatus.COMPLETE,
        finish_reason="eos",
    )
    seen: list[str] = []

    def generate_output(prompt: str) -> ModelOutput:
        seen.append(prompt)
        return generated

    result = generate_and_evaluate(case, generate_output)

    assert seen == [case.prompt]
    assert result.raw_output is generated
    assert result.extracted_answer == "8"
    assert result.answer_status is AnswerStatus.CORRECT
    assert result.passed
