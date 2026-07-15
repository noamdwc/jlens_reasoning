import pytest

from jlens_reasoning.evaluation import (
    ComponentId,
    GenerationStatus,
    ModelOutput,
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
