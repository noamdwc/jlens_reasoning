from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.jlens_readout_sanity.experiment import (
    Case,
    ExperimentRuntime,
    InterventionSpec,
    ReadoutSpec,
    generate_and_evaluate,
    run_intervention,
    run_readout,
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


class OperationTokenizer:
    def encode(self, surface: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        token_ids = {
            "spider": 2,
            "ant": 3,
            "8": 4,
            "eight": 4,
            "6": 5,
            "six": 5,
        }
        normalized = surface.strip().casefold()
        return [token_ids[normalized]] if normalized in token_ids else [0, 1]

    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return {0: "zero", 1: "one", 2: " spider", 3: " ant", 4: "8", 5: "6"}[
            token_ids[0]
        ]


class FakeReadoutLens:
    source_layers = [2]

    def apply(
        self,
        model: object,
        prompt: str,
        *,
        use_jacobian: bool = True,
        **kwargs: object,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        del model, prompt, kwargs
        input_ids = torch.tensor([[0, 1]])
        model_logits = torch.zeros(2, 6)
        model_logits[-1, 4] = 9.0
        readout = torch.zeros(2, 6)
        readout[:, 2] = 8.0 if use_jacobian else -1.0
        return {2: readout}, model_logits, input_ids


class TensorBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TinySwapModel:
    def __init__(self) -> None:
        self.layers = nn.ModuleList([TensorBlock() for _ in range(4)])

    def encode(self, prompt: str) -> torch.Tensor:
        del prompt
        return torch.tensor([[0, 1]])


def test_run_readout_returns_typed_rank_locations_and_capability_gate() -> None:
    runtime = ExperimentRuntime(
        model=SimpleNamespace(),
        lens=FakeReadoutLens(),
        tokenizer=OperationTokenizer(),
        unembedding_weight=torch.zeros(6, 2),
        forward_next_token=lambda _: torch.zeros(6),
        generate_output=lambda _: ModelOutput("8"),
    )
    case = spider_case()

    result = run_readout(case, case.readout, runtime, layers=(2,), top_k=3)

    assert result.jacobian_lens.best_rank == 1
    assert result.jacobian_lens.layer == 2
    assert result.logit_lens.best_rank > 1
    assert result.workspace_layers == (2,)
    assert result.paper_top1_hit is True
    assert result.capability_passed is True


def test_run_intervention_uses_evaluation_for_each_configured_alpha() -> None:
    model = TinySwapModel()
    lens = SimpleNamespace(jacobians={2: torch.eye(2)})
    unembedding = torch.zeros(6, 2)
    unembedding[2] = torch.tensor([1.0, 0.0])
    unembedding[3] = torch.tensor([0.0, 1.0])

    def forward_next_token(input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        for block in model.layers:
            hidden = block(hidden)
        logits = torch.zeros(6)
        logits[4] = hidden[0, -1, 0]
        logits[5] = hidden[0, -1, 1]
        return logits

    runtime = ExperimentRuntime(
        model=model,
        lens=lens,
        tokenizer=OperationTokenizer(),
        unembedding_weight=unembedding,
        forward_next_token=forward_next_token,
        generate_output=lambda _: ModelOutput("8"),
    )
    case = spider_case()

    result, prepared = run_intervention(
        case,
        case.intervention,
        runtime,
        layers=(2,),
        top_k=3,
    )

    assert prepared.case is case
    assert result.source.token_id == 2
    assert result.target.token_id == 3
    assert result.clean_target.accepted_references == ("6", "six")
    assert tuple(condition.alpha for condition in result.conditions) == (1.0, 2.0)
    assert all(condition.comparison.improved for condition in result.conditions)
    assert all(condition.comparison.reached_top1 for condition in result.conditions)
