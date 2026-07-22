import inspect
import math
from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from jlens_reasoning.evaluation import (
    AnswerStatus,
    EvaluationResult,
    GenerationStatus,
    ModelOutput,
    NextTokenEvaluation,
    RankedToken,
    ReasoningStatus,
    SimpleFactualEvaluator,
    compare_token_ranks,
    evaluate,
    evaluate_next_token,
)
from jlens_reasoning.evaluation_utils import (
    answer_token_variants,
    best_token_rank,
    log_rank_gain,
    parse_think_tags,
    top_token_values,
)


def test_model_output_preserves_raw_token_artifact() -> None:
    output = ModelOutput(
        text=" 8.",
        token_ids=(220, 23, 13),
        token_pieces=(" ", "8", "."),
        finish_reason="eos",
    )

    assert output.text == " 8."
    assert output.token_ids == (220, 23, 13)
    assert output.token_pieces == (" ", "8", ".")
    assert output.generation_status is GenerationStatus.COMPLETE


def test_model_output_rejects_mismatched_token_metadata() -> None:
    with pytest.raises(ValueError, match="same length"):
        ModelOutput(text="8", token_ids=(23,), token_pieces=())


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (GenerationStatus.GENERATION_ERROR, None),
        (GenerationStatus.GENERATION_ERROR, ""),
        (GenerationStatus.GENERATION_ERROR, "   "),
        (GenerationStatus.COMPLETE, "boom"),
        (GenerationStatus.TRUNCATED, "boom"),
    ],
)
def test_model_output_rejects_inconsistent_generation_error(
    status: GenerationStatus, message: str | None
) -> None:
    with pytest.raises(ValueError, match="must agree"):
        ModelOutput(
            text="",
            generation_status=status,
            generation_error=message,
        )


def test_spider_regression() -> None:
    result = evaluate(
        " 8.\n\nThis conclusion is based on...",
        ("8", "eight"),
    )

    assert result.raw_output.text == " 8.\n\nThis conclusion is based on..."
    assert result.evaluation_text == "8.\n\nThis conclusion is based on..."
    assert result.extracted_answer == "8"
    assert result.normalized_answer == "8"
    assert result.answer_status is AnswerStatus.CORRECT
    assert result.passed


@pytest.mark.parametrize(
    ("output", "references", "normalized", "matched"),
    [
        (" EIGHT! ", "eight", "eight", "eight"),
        ("Paris? Explanation", ("Lyon", "paris."), "paris", "paris."),
        ("Cafe\u0301", "Café", "café", "Café"),
    ],
)
def test_minimal_normalization(
    output: str,
    references: str | tuple[str, ...],
    normalized: str,
    matched: str,
) -> None:
    result = evaluate(output, references)

    assert result.normalized_answer == normalized
    assert result.matched_reference == matched
    assert result.answer_status is AnswerStatus.CORRECT


@pytest.mark.parametrize(
    ("output", "reference"),
    [
        ("Cote d'Ivoire", "Côte d'Ivoire"),
        ("New-York", "New York"),
        ("the Paris", "Paris"),
    ],
)
def test_normalization_does_not_change_meaning(output: str, reference: str) -> None:
    assert evaluate(output, reference).answer_status is AnswerStatus.INCORRECT


def test_extraction_does_not_search_for_reference() -> None:
    result = evaluate("6. The answer is 8.", "8")

    assert result.extracted_answer == "6"
    assert result.normalized_answer == "6"
    assert result.answer_status is AnswerStatus.INCORRECT


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_complete_output_is_unparseable(text: str) -> None:
    result = evaluate(text, "8")

    assert result.evaluation_text == ""
    assert result.extracted_answer is None
    assert result.normalized_answer is None
    assert result.answer_status is AnswerStatus.UNPARSEABLE
    assert not result.passed


def test_answer_after_inline_thinking_is_graded() -> None:
    raw_text = "<think>A spider has eight legs.</think>\n 8."
    result = evaluate(
        raw_text,
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=parse_think_tags),
    )

    assert result.raw_output.text == raw_text
    assert result.reasoning_status is ReasoningStatus.PARSED
    assert result.evaluation_text == "8."
    assert result.extracted_answer == "8"
    assert result.passed


def test_answer_only_inside_thinking_does_not_count() -> None:
    result = evaluate(
        "<think>The answer is 8.</think>\n6",
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=parse_think_tags),
    )

    assert result.evaluation_text == "6"
    assert result.extracted_answer == "6"
    assert result.answer_status is AnswerStatus.INCORRECT


def test_multiple_thinking_spans_are_removed() -> None:
    result = evaluate(
        "<think>first</think> 8.<think>second</think>",
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=parse_think_tags),
    )

    assert result.reasoning_status is ReasoningStatus.PARSED
    assert result.evaluation_text == "8."
    assert result.answer_status is AnswerStatus.CORRECT


def test_absent_thinking_is_not_present() -> None:
    result = evaluate(
        "8.",
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=parse_think_tags),
    )

    assert result.reasoning_status is ReasoningStatus.NOT_PRESENT
    assert result.answer_status is AnswerStatus.CORRECT


@pytest.mark.parametrize(
    "text",
    [
        "<think>unfinished",
        "stray</think>8",
        "<think>outer <think>nested</think></think>8",
    ],
)
def test_malformed_thinking_is_not_graded(text: str) -> None:
    result = evaluate(
        text,
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=parse_think_tags),
    )

    assert result.raw_output.text == text
    assert result.evaluation_text == ""
    assert result.extracted_answer is None
    assert result.normalized_answer is None
    assert result.reasoning_status is ReasoningStatus.MALFORMED
    assert result.answer_status is AnswerStatus.NOT_GRADED
    assert not result.passed


def test_generation_error_is_not_graded() -> None:
    output = ModelOutput(
        text="partial raw output",
        generation_status=GenerationStatus.GENERATION_ERROR,
        generation_error="device failure",
    )

    result = evaluate(output, "8")

    assert result.raw_output is output
    assert result.raw_output.text == "partial raw output"
    assert result.generation_error == "device failure"
    assert result.evaluation_text == ""
    assert result.answer_status is AnswerStatus.NOT_GRADED
    assert not result.passed


def test_punctuation_only_truncation_is_not_graded() -> None:
    output = ModelOutput("...", generation_status=GenerationStatus.TRUNCATED)

    result = evaluate(output, "8")

    assert result.evaluation_text == "..."
    assert result.extracted_answer is None
    assert result.normalized_answer is None
    assert result.answer_status is AnswerStatus.NOT_GRADED


@pytest.mark.parametrize("text", ["8 or", "8", "partial answer"])
def test_ambiguous_truncation_is_not_graded(text: str) -> None:
    output = ModelOutput(text, generation_status=GenerationStatus.TRUNCATED)

    result = evaluate(output, "8")

    assert result.raw_output.text == text
    assert result.evaluation_text == ""
    assert result.extracted_answer is None
    assert result.normalized_answer is None
    assert result.generation_status is GenerationStatus.TRUNCATED
    assert result.answer_status is AnswerStatus.NOT_GRADED
    assert not result.passed


@pytest.mark.parametrize(
    ("text", "evaluation_text"),
    [
        ("8.\nThis sentence is incom", "8."),
        ("8! trailing frag", "8!"),
        ("8\ntrailing frag", "8"),
    ],
)
def test_complete_front_loaded_answer_survives_truncation(
    text: str, evaluation_text: str
) -> None:
    output = ModelOutput(text, generation_status=GenerationStatus.TRUNCATED)

    result = evaluate(output, "8")

    assert result.raw_output.text == text
    assert result.evaluation_text == evaluation_text
    assert result.extracted_answer == "8"
    assert result.answer_status is AnswerStatus.CORRECT
    assert result.passed


def test_empty_safe_truncation_prefix_is_not_graded() -> None:
    output = ModelOutput("\npartial", generation_status=GenerationStatus.TRUNCATED)

    result = evaluate(output, "8")

    assert result.evaluation_text == ""
    assert result.answer_status is AnswerStatus.NOT_GRADED


@pytest.mark.parametrize("references", [(), [], "", "...", " ? "])
def test_empty_or_normalized_empty_references_are_rejected(
    references: str | list[str] | tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="reference"):
        evaluate("8", references)


def test_any_empty_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="reference"):
        evaluate("8", ("8", "   "))


def test_model_output_rejects_mutable_token_collections() -> None:
    with pytest.raises(TypeError, match="tuples"):
        ModelOutput(text="8", token_ids=[23], token_pieces=["8"])  # type: ignore[arg-type]


def test_frozen_dataclasses_and_tuple_fields_are_immutable() -> None:
    output = ModelOutput("8", token_ids=(23,), token_pieces=("8",))
    result = evaluate(output, ["8", "eight"])

    assert result.accepted_references == ("8", "eight")
    with pytest.raises(FrozenInstanceError):
        output.text = "6"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.answer_status = AnswerStatus.INCORRECT  # type: ignore[misc]


def test_result_stores_only_final_text_processing_artifacts() -> None:
    names = {field.name for field in fields(EvaluationResult)}

    assert names == {
        "raw_output",
        "evaluation_text",
        "extracted_answer",
        "normalized_answer",
        "reasoning_status",
        "answer_status",
        "accepted_references",
        "matched_reference",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"matched_reference": None},
        {
            "answer_status": AnswerStatus.INCORRECT,
            "extracted_answer": None,
            "normalized_answer": None,
        },
        {"answer_status": AnswerStatus.UNPARSEABLE},
    ],
)
def test_result_rejects_inconsistent_fields(changes: dict[str, object]) -> None:
    result = evaluate("8", "8")

    with pytest.raises((TypeError, ValueError)):
        replace(result, **changes)


def test_pass_rule_covers_each_status_dimension() -> None:
    correct = evaluate("8", "8")
    malformed = evaluate(
        "<think>unfinished",
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=parse_think_tags),
    )
    generation_error = evaluate(
        ModelOutput(
            "",
            generation_status=GenerationStatus.GENERATION_ERROR,
            generation_error="boom",
        ),
        "8",
    )

    assert correct.passed
    assert not evaluate("6", "8").passed
    assert not malformed.passed
    assert not generation_error.passed


def test_result_records_reference_audit_data() -> None:
    result = evaluate("Paris.", ("Lyon", "paris."))

    assert result.accepted_references == ("Lyon", "paris.")
    assert result.matched_reference == "paris."


def test_runner_accepts_a_custom_factual_evaluator() -> None:
    expected = evaluate("8", "8")

    def custom(
        output: ModelOutput, accepted_references: tuple[str, ...]
    ) -> EvaluationResult:
        assert output.text == "ignored"
        assert accepted_references == ("unused",)
        return expected

    assert evaluate("ignored", "unused", evaluator=custom) is expected


class RankTokenizer:
    pieces = {
        "Paris": [2],
        " Paris": [4],
        "paris": [2],
        " paris": [4],
        "PARIS": [8, 9],
        " PARIS": [8, 9],
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces.get(text, [10, 11])

    def decode(
        self, token_ids: list[int], *, clean_up_tokenization_spaces: bool = False
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return {0: "zero", 1: "one", 2: "Paris", 3: "three", 4: " Paris"}.get(
            token_ids[0], f"token-{token_ids[0]}"
        )


def test_next_token_evaluation_resolves_variants_and_stable_ranks() -> None:
    result = evaluate_next_token(
        torch.tensor([0.0, 4.0, 3.0, 1.0, 2.0]),
        ("Paris",),
        RankTokenizer(),
        top_k=3,
    )

    assert result.accepted_references == ("Paris",)
    assert result.accepted_token_ids == (2, 4)
    assert result.target_rank == 2
    assert result.top1_id == 1
    assert result.top1_token == "one"
    assert result.top_tokens == (
        RankedToken(1, "one", 4.0),
        RankedToken(2, "Paris", 3.0),
        RankedToken(4, " Paris", 2.0),
    )


def test_rank_comparison_uses_positive_improvement_convention() -> None:
    baseline = NextTokenEvaluation(
        accepted_references=("Paris",),
        accepted_token_ids=(2,),
        top1_id=1,
        top1_token="one",
        target_rank=10,
        top_tokens=(),
    )
    candidate = replace(
        baseline,
        top1_id=2,
        top1_token="Paris",
        target_rank=1,
    )

    comparison = compare_token_ranks(baseline, candidate)

    assert comparison.baseline_rank == 10
    assert comparison.candidate_rank == 1
    assert comparison.rank_gain == 9
    assert comparison.log_rank_gain == pytest.approx(math.log(10))
    assert comparison.improved
    assert comparison.reached_top1


@pytest.mark.parametrize("references", [(), ("",), ("two tokens",)])
def test_next_token_evaluation_rejects_unscorable_answers(
    references: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="accepted|Accepted|single-token"):
        evaluate_next_token(
            torch.tensor([0.0, 1.0, 2.0]),
            references,
            RankTokenizer(),
        )


@pytest.mark.parametrize(
    "function",
    (answer_token_variants, best_token_rank, top_token_values, log_rank_gain),
)
def test_rank_helpers_document_their_outputs(function: object) -> None:
    docstring = inspect.getdoc(function)

    assert docstring is not None
    assert "\nReturns " in docstring
