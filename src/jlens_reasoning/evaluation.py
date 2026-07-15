import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class GenerationStatus(StrEnum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    GENERATION_ERROR = "generation_error"


class ReasoningStatus(StrEnum):
    NOT_PRESENT = "not_present"
    PARSED = "parsed"
    MALFORMED = "malformed_reasoning"


class AnswerStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNPARSEABLE = "unparseable"
    NOT_GRADED = "not_graded"


@dataclass(frozen=True, slots=True)
class ComponentId:
    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("component name and version must be non-empty")


@dataclass(frozen=True, slots=True)
class ModelOutput:
    text: str
    token_ids: tuple[int, ...] = ()
    token_pieces: tuple[str, ...] = ()
    generation_status: GenerationStatus = GenerationStatus.COMPLETE
    finish_reason: str | None = None
    generation_error: str | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_pieces):
            raise ValueError("token IDs and pieces must have the same length")
        has_error = bool(self.generation_error)
        expects_error = self.generation_status is GenerationStatus.GENERATION_ERROR
        if has_error != expects_error:
            raise ValueError("generation_error status and message must agree")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    evaluator: ComponentId
    reasoning_parser: ComponentId
    extractor: ComponentId
    normalizer: ComponentId
    accepted_references: tuple[str, ...]
    generation_status: GenerationStatus
    reasoning_status: ReasoningStatus
    answer_status: AnswerStatus
    generation_error: str | None
    raw_output: ModelOutput
    evaluation_text: str
    extracted_answer: str | None
    normalized_answer: str | None
    matched_reference: str | None

    def __post_init__(self) -> None:
        if (
            self.generation_status is not self.raw_output.generation_status
            or self.generation_error != self.raw_output.generation_error
        ):
            raise ValueError("generation fields must match raw output")
        if (
            self.matched_reference is not None
            and self.matched_reference not in self.accepted_references
        ):
            raise ValueError("matched reference must be accepted")
        if self.answer_status is AnswerStatus.CORRECT and (
            self.extracted_answer is None
            or self.normalized_answer is None
            or self.matched_reference is None
        ):
            raise ValueError("correct result requires an answer and match")

    @property
    def passed(self) -> bool:
        return (
            self.answer_status is AnswerStatus.CORRECT
            and self.reasoning_status is not ReasoningStatus.MALFORMED
            and self.generation_status is not GenerationStatus.GENERATION_ERROR
        )


ReasoningFunction = Callable[[str], tuple[str, ReasoningStatus]]


@dataclass(frozen=True, slots=True)
class ReasoningParser:
    component_id: ComponentId
    parse: ReasoningFunction

    def __call__(self, text: str) -> tuple[str, ReasoningStatus]:
        return self.parse(text)


def _no_reasoning(text: str) -> tuple[str, ReasoningStatus]:
    return text, ReasoningStatus.NOT_PRESENT


SIMPLE_FACTUAL = ComponentId("simple_factual", "v1")
FRONT_LOADED = ComponentId("front_loaded_segment", "v1")
MINIMAL_TEXT = ComponentId("minimal_text", "v1")
NO_REASONING_ID = ComponentId("none", "v1")
NO_REASONING_PARSER = ReasoningParser(NO_REASONING_ID, _no_reasoning)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().casefold().rstrip(".!?")


def _extract(evaluation_text: str) -> str | None:
    answer = re.split(r"[.!?\n]", evaluation_text, maxsplit=1)[0].strip()
    return answer or None


class FactualEvaluator(Protocol):
    def __call__(
        self, output: ModelOutput, accepted_references: tuple[str, ...]
    ) -> EvaluationResult: ...


@dataclass(frozen=True, slots=True)
class SimpleFactualEvaluator:
    reasoning_parser: ReasoningParser = NO_REASONING_PARSER

    def __call__(
        self, output: ModelOutput, accepted_references: tuple[str, ...]
    ) -> EvaluationResult:
        evaluation_text, reasoning_status = self.reasoning_parser(output.text)
        evaluation_text = evaluation_text.strip()
        answer = _extract(evaluation_text)
        normalized = _normalize(answer) if answer is not None else None
        matched = next(
            (
                reference
                for reference in accepted_references
                if _normalize(reference) == normalized
            ),
            None,
        )
        status = (
            AnswerStatus.UNPARSEABLE
            if answer is None
            else AnswerStatus.CORRECT
            if matched is not None
            else AnswerStatus.INCORRECT
        )
        return EvaluationResult(
            evaluator=SIMPLE_FACTUAL,
            reasoning_parser=self.reasoning_parser.component_id,
            extractor=FRONT_LOADED,
            normalizer=MINIMAL_TEXT,
            accepted_references=accepted_references,
            generation_status=output.generation_status,
            reasoning_status=reasoning_status,
            answer_status=status,
            generation_error=output.generation_error,
            raw_output=output,
            evaluation_text=evaluation_text,
            extracted_answer=answer,
            normalized_answer=normalized,
            matched_reference=matched,
        )


def evaluate(
    output: str | ModelOutput,
    accepted_references: str | Sequence[str],
    evaluator: FactualEvaluator | None = None,
) -> EvaluationResult:
    model_output = ModelOutput(output) if isinstance(output, str) else output
    references = (
        (accepted_references,)
        if isinstance(accepted_references, str)
        else tuple(accepted_references)
    )
    return (evaluator or SimpleFactualEvaluator())(model_output, references)
