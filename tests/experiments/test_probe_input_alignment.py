from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.flenqa_probe_jlens.constants import PROBE_CONFIG
from jlens_reasoning.inference import (
    InferenceConfig,
    InferenceInputError,
    chat_input_fingerprint,
    prepare_chat_inputs,
)
from jlens_reasoning.probing import (
    probe_input_contract,
    validate_input_record,
    validate_probe_input_contract,
)


def notebook_cell(path: str, cell_id: str) -> str:
    notebook = json.loads(Path(path).read_text())
    return "".join(
        next(c for c in notebook["cells"] if c.get("id") == cell_id)["source"]
    )


def test_probe_contract_rejects_legacy_assets_and_changed_template(tokenizer):
    expected = probe_input_contract(tokenizer, config=PROBE_CONFIG)
    validate_probe_input_contract(expected, expected)
    with pytest.raises(ValueError, match="Retrain"):
        validate_probe_input_contract(
            {"format_version": 1, "input_format": "raw_flenqa_prompt"}, expected
        )
    tokenizer.chat_template += " True"
    with pytest.raises(ValueError, match="Retrain"):
        validate_probe_input_contract(
            expected, probe_input_contract(tokenizer, config=PROBE_CONFIG)
        )


def test_contract_detects_changed_vocabulary_but_ignores_runtime_padding(tokenizer):
    original = probe_input_contract(tokenizer, config=PROBE_CONFIG)
    tokenizer(["cat", "cat cat"], padding=True)
    assert probe_input_contract(tokenizer, config=PROBE_CONFIG) == original
    tokenizer.add_tokens(["new"])
    assert probe_input_contract(tokenizer, config=PROBE_CONFIG) != original


def test_input_record_requires_exact_tokens_not_just_length(tokenizer):
    encoded = prepare_chat_inputs(tokenizer, "cat", config=PROBE_CONFIG.inference)
    assert encoded["input_ids"].tolist() == [[1, 3, 7, 2, 4]]
    record = {"input_sha256": chat_input_fingerprint(encoded), "n_input_tokens": 5}
    validate_input_record(record, encoded)
    encoded["input_ids"][0, 2] = 5
    with pytest.raises(ValueError, match="Regenerate"):
        validate_input_record(record, encoded)
    with pytest.raises(ValueError, match="Regenerate"):
        validate_input_record({"n_input_tokens": 5}, encoded)


def test_input_limit_counts_chat_wrapper(tokenizer):
    with pytest.raises(InferenceInputError, match="exceeds configured limit 4"):
        prepare_chat_inputs(
            tokenizer, "cat", config=InferenceConfig.direct(max_input_tokens=4)
        )
