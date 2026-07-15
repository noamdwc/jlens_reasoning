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
        if not self.name.strip() or not self.version.strip():
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
        if not isinstance(self.accepted_references, tuple):
            raise TypeError("accepted references must be a tuple")
        if (
            self.generation_status is not self.raw_output.generation_status
            or self.generation_error != self.raw_output.generation_error
        ):
            raise ValueError("generation fields must match raw output")
        graded = self.answer_status in (AnswerStatus.CORRECT, AnswerStatus.INCORRECT)
        has_answer = (
            self.extracted_answer is not None and self.normalized_answer is not None
        )
        if graded != has_answer:
            raise ValueError("graded status and answer must agree")
        has_match = self.matched_reference is not None
        if (self.answer_status is AnswerStatus.CORRECT) != has_match:
            raise ValueError("correct status and match must agree")
        if has_match and self.matched_reference not in self.accepted_references:
            raise ValueError("matched reference must be accepted")

    @property
    def passed(self) -> bool:
        return self.answer_status is AnswerStatus.CORRECT and not (
            self.reasoning_status is ReasoningStatus.MALFORMED
            or self.generation_status is GenerationStatus.GENERATION_ERROR
        )


@dataclass(frozen=True, slots=True)
class ReasoningParser:
    component_id: ComponentId
    parse: Callable[[str], tuple[str, ReasoningStatus]]


def _no_reasoning(text: str) -> tuple[str, ReasoningStatus]:
    return text, ReasoningStatus.NOT_PRESENT


SIMPLE_FACTUAL = ComponentId("simple_factual", "v1")
FRONT_LOADED = ComponentId("front_loaded_segment", "v1")
MINIMAL_TEXT = ComponentId("minimal_text", "v1")
NO_REASONING_ID = ComponentId("none", "v1")
NO_REASONING_PARSER = ReasoningParser(NO_REASONING_ID, _no_reasoning)


_THINK_SPAN = re.compile(r"<think>(?:(?!</?think>).)*</think>", re.DOTALL)


def _parse_think_tags(text: str) -> tuple[str, ReasoningStatus]:
    if "<think>" not in text and "</think>" not in text:
        return text, ReasoningStatus.NOT_PRESENT
    visible = _THINK_SPAN.sub("", text)
    if "<think>" in visible or "</think>" in visible:
        return "", ReasoningStatus.MALFORMED
    return visible, ReasoningStatus.PARSED


THINK_TAGS_ID = ComponentId("think_tags", "v1")
THINK_TAGS_PARSER = ReasoningParser(THINK_TAGS_ID, _parse_think_tags)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().casefold().rstrip(".!?")


def _extract(evaluation_text: str) -> str | None:
    answer = re.split(r"[.!?\n]", evaluation_text, maxsplit=1)[0].strip()
    return answer or None


def _safe_truncated_text(text: str) -> str | None:
    boundary = max(text.rfind(character) for character in ".!?\n")
    return None if boundary < 0 else text[: boundary + 1].strip()


class FactualEvaluator(Protocol):
    def __call__(
        self, output: ModelOutput, accepted_references: tuple[str, ...]
    ) -> EvaluationResult: ...


@dataclass(frozen=True, slots=True)
class SimpleFactualEvaluator:
    reasoning_parser: ReasoningParser = NO_REASONING_PARSER

    def _result(
        self,
        output: ModelOutput,
        references: tuple[str, ...],
        reasoning: ReasoningStatus,
        status: AnswerStatus,
        text: str = "",
        answer: str | None = None,
        normalized: str | None = None,
        matched: str | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            SIMPLE_FACTUAL,
            self.reasoning_parser.component_id,
            FRONT_LOADED,
            MINIMAL_TEXT,
            references,
            output.generation_status,
            reasoning,
            status,
            output.generation_error,
            output,
            text,
            answer,
            normalized,
            matched,
        )

    def __call__(
        self, output: ModelOutput, accepted_references: tuple[str, ...]
    ) -> EvaluationResult:
        if not accepted_references or any(
            not isinstance(reference, str) or not _normalize(reference)
            for reference in accepted_references
        ):
            raise ValueError("accepted references must normalize to non-empty text")
        if output.generation_status is GenerationStatus.GENERATION_ERROR:
            return self._result(
                output,
                accepted_references,
                ReasoningStatus.NOT_PRESENT,
                AnswerStatus.NOT_GRADED,
            )
        evaluation_text, reasoning_status = self.reasoning_parser.parse(output.text)
        if output.generation_status is GenerationStatus.TRUNCATED:
            evaluation_text = _safe_truncated_text(evaluation_text)
        if reasoning_status is ReasoningStatus.MALFORMED or not evaluation_text:
            status = (
                AnswerStatus.NOT_GRADED
                if reasoning_status is ReasoningStatus.MALFORMED
                or output.generation_status is GenerationStatus.TRUNCATED
                else AnswerStatus.UNPARSEABLE
            )
            return self._result(output, accepted_references, reasoning_status, status)
        evaluation_text = evaluation_text.strip()
        answer = _extract(evaluation_text)
        normalized = _normalize(answer) if answer is not None else None
        matched = next(
            (r for r in accepted_references if _normalize(r) == normalized), None
        )
        if answer is None:
            status = AnswerStatus.UNPARSEABLE
        elif matched is None:
            status = AnswerStatus.INCORRECT
        else:
            status = AnswerStatus.CORRECT
        return self._result(
            output,
            accepted_references,
            reasoning_status,
            status,
            evaluation_text,
            answer,
            normalized,
            matched,
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
