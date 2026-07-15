import pytest

from jlens_reasoning.evaluation import (
    AnswerStatus,
    ComponentId,
    EvaluationResult,
    GenerationStatus,
    ModelOutput,
    ReasoningStatus,
    evaluate,
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


@pytest.mark.parametrize(("name", "version"), [("", "v1"), ("parser", "")])
def test_component_id_rejects_empty_values(name: str, version: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ComponentId(name, version)


def test_spider_regression() -> None:
    result = evaluate(
        " 8.\n\nThis conclusion is based on...",
        ("8", "eight"),
    )

    assert result.raw_output.text == " 8.\n\nThis conclusion is based on..."
    assert result.evaluation_text == "8.\n\nThis conclusion is based on..."
    assert result.extracted_answer == "8"
    assert result.normalized_answer == "8"
    assert result.matched_reference == "8"
    assert result.answer_status is AnswerStatus.CORRECT
    assert result.passed


@pytest.mark.parametrize(
    ("output", "references", "normalized", "match"),
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
    match: str,
) -> None:
    result = evaluate(output, references)

    assert result.normalized_answer == normalized
    assert result.matched_reference == match
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
    assert result.matched_reference is None
    assert result.answer_status is AnswerStatus.INCORRECT


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_complete_output_is_unparseable(text: str) -> None:
    result = evaluate(text, "8")

    assert result.evaluation_text == ""
    assert result.extracted_answer is None
    assert result.normalized_answer is None
    assert result.answer_status is AnswerStatus.UNPARSEABLE
    assert not result.passed


def test_result_records_factual_provenance() -> None:
    result = evaluate("Paris.", ("Paris", "PARIS"))

    assert isinstance(result, EvaluationResult)
    assert result.accepted_references == ("Paris", "PARIS")
    assert (result.evaluator.name, result.evaluator.version) == (
        "simple_factual",
        "v1",
    )
    assert (result.reasoning_parser.name, result.reasoning_parser.version) == (
        "none",
        "v1",
    )
    assert (result.extractor.name, result.extractor.version) == (
        "front_loaded_segment",
        "v1",
    )
    assert (result.normalizer.name, result.normalizer.version) == (
        "minimal_text",
        "v1",
    )
    assert result.reasoning_status is ReasoningStatus.NOT_PRESENT
