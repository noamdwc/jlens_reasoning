from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .evaluation_utils import (
    ReasoningStatus,
    extract_answer,
    match_reference,
    no_reasoning,
    normalize_text,
    safe_truncated_text,
)


class GenerationStatus(StrEnum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    GENERATION_ERROR = "generation_error"


class AnswerStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNPARSEABLE = "unparseable"
    NOT_GRADED = "not_graded"


@dataclass(frozen=True, slots=True)
class ModelOutput:
    text: str
    token_ids: tuple[int, ...] = ()
    token_pieces: tuple[str, ...] = ()
    generation_status: GenerationStatus = GenerationStatus.COMPLETE
    finish_reason: str | None = None
    generation_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple) or not isinstance(
            self.token_pieces, tuple
        ):
            raise TypeError("token IDs and pieces must be tuples")
        if len(self.token_ids) != len(self.token_pieces):
            raise ValueError("token IDs and pieces must have the same length")
        has_error = bool(self.generation_error and self.generation_error.strip())
        expects_error = self.generation_status is GenerationStatus.GENERATION_ERROR
        if has_error != expects_error:
            raise ValueError("generation_error status and message must agree")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    raw_output: ModelOutput
    evaluation_text: str
    extracted_answer: str | None
    normalized_answer: str | None
    reasoning_status: ReasoningStatus
    answer_status: AnswerStatus
    accepted_references: tuple[str, ...]
    matched_reference: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.accepted_references, tuple):
            raise TypeError("accepted references must be a tuple")
        graded = self.answer_status in (AnswerStatus.CORRECT, AnswerStatus.INCORRECT)
        has_answer = (
            self.extracted_answer is not None and self.normalized_answer is not None
        )
        if graded != has_answer:
            raise ValueError("graded status and answer must agree")
        has_match = self.matched_reference is not None
        if (self.answer_status is AnswerStatus.CORRECT) != has_match:
            raise ValueError("correct status and matched reference must agree")
        if has_match and self.matched_reference not in self.accepted_references:
            raise ValueError("matched reference must be accepted")

    @property
    def generation_status(self) -> GenerationStatus:
        return self.raw_output.generation_status

    @property
    def generation_error(self) -> str | None:
        return self.raw_output.generation_error

    @property
    def passed(self) -> bool:
        return self.answer_status is AnswerStatus.CORRECT and not (
            self.reasoning_status is ReasoningStatus.MALFORMED
            or self.generation_status is GenerationStatus.GENERATION_ERROR
        )


class FactualEvaluator(Protocol):
    def __call__(
        self, output: ModelOutput, accepted_references: tuple[str, ...]
    ) -> EvaluationResult: ...


@dataclass(frozen=True, slots=True)
class SimpleFactualEvaluator:
    reasoning_parser: Callable[[str], tuple[str, ReasoningStatus]] = no_reasoning

    def __call__(
        self, output: ModelOutput, accepted_references: tuple[str, ...]
    ) -> EvaluationResult:
        if not accepted_references or any(
            not isinstance(reference, str) or not normalize_text(reference)
            for reference in accepted_references
        ):
            raise ValueError("accepted references must normalize to non-empty text")
        if output.generation_status is GenerationStatus.GENERATION_ERROR:
            return EvaluationResult(
                raw_output=output,
                evaluation_text="",
                extracted_answer=None,
                normalized_answer=None,
                reasoning_status=ReasoningStatus.NOT_PRESENT,
                answer_status=AnswerStatus.NOT_GRADED,
                accepted_references=accepted_references,
                matched_reference=None,
            )
        evaluation_text, reasoning_status = self.reasoning_parser(output.text)
        if output.generation_status is GenerationStatus.TRUNCATED:
            evaluation_text = safe_truncated_text(evaluation_text)
        if reasoning_status is ReasoningStatus.MALFORMED or not evaluation_text:
            status = (
                AnswerStatus.NOT_GRADED
                if reasoning_status is ReasoningStatus.MALFORMED
                or output.generation_status is GenerationStatus.TRUNCATED
                else AnswerStatus.UNPARSEABLE
            )
            return EvaluationResult(
                raw_output=output,
                evaluation_text="",
                extracted_answer=None,
                normalized_answer=None,
                reasoning_status=reasoning_status,
                answer_status=status,
                accepted_references=accepted_references,
                matched_reference=None,
            )
        evaluation_text = evaluation_text.strip()
        answer = extract_answer(evaluation_text)
        normalized = normalize_text(answer) if answer is not None else None
        if answer is None:
            status = (
                AnswerStatus.NOT_GRADED
                if output.generation_status is GenerationStatus.TRUNCATED
                else AnswerStatus.UNPARSEABLE
            )
            matched = None
        else:
            matched = match_reference(normalized, accepted_references)
            status = (
                AnswerStatus.CORRECT if matched is not None else AnswerStatus.INCORRECT
            )
        return EvaluationResult(
            raw_output=output,
            evaluation_text=evaluation_text,
            extracted_answer=answer,
            normalized_answer=normalized,
            reasoning_status=reasoning_status,
            answer_status=status,
            accepted_references=accepted_references,
            matched_reference=matched,
        )


def evaluate(
    output: str | ModelOutput,
    accepted_references: str | Sequence[str],
    evaluator: FactualEvaluator | None = None,
) -> EvaluationResult:
    model_output = ModelOutput(output) if isinstance(output, str) else output
    if isinstance(accepted_references, str):
        references = (accepted_references,)
    else:
        references = tuple(accepted_references)
    return (evaluator or SimpleFactualEvaluator())(model_output, references)
