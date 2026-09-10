from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import transformers
from jlens.hooks import ActivationRecorder
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.flenqa_probe_jlens.constants import PROBE_CONFIG
from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaRow,
    build_prompt_text,
    prepare_prompts,
)
from jlens_reasoning.evaluation_utils import answer_token_variants
from jlens_reasoning.inference import (
    InferenceConfig,
    InferenceInputError,
    chat_input_fingerprint,
    generate_chat,
    prepare_chat_inputs,
)
from jlens_reasoning.probe_jlens import probe_sensitivities
from jlens_reasoning.probing import (
    extract_probe_features,
    probe_input_contract,
    token_margin,
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


@pytest.mark.parametrize("task", ["PIR", "MonoRel", "Simplified RuleTaker"])
def test_notebook_training_evaluation_and_gradients_use_generation_inputs(
    tokenizer, monkeypatch, task, tmp_path
):
    torch.manual_seed(1)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=2,
            attn_implementation="eager",
        )
    ).eval()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
    )
    row = FlenqaRow(
        0, 1, task, True, ("cat",), "cat", "cat", "cat", 250, "books", "first"
    )
    prompt = prepare_prompts([row])[0]
    seen = []

    def capture_inputs(module, args, kwargs):
        seen.append(
            {key: kwargs[key].clone() for key in ("input_ids", "attention_mask")}
        )

    with_hook = model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    try:
        generated = generate_chat(
            model,
            tokenizer,
            prompt.text,
            config=InferenceConfig.direct(max_new_tokens=1),
        )
        generated_inputs = seen[0]
        assert generated_inputs["input_ids"][0, -1].item() == 4  # assistant prefix
        ns = dict(
            transformers=transformers,
            torch=torch,
            np=np,
            pd=pd,
            context=SimpleNamespace(device="cpu", runs_dir=tmp_path),
            MODEL_PATH="synthetic",
            selected_rows=[row],
            problem_to_split={1: "train"},
            build_prompt_text=build_prompt_text,
            tqdm=lambda values, **kwargs: values,
            prepare_chat_inputs=prepare_chat_inputs,
            chat_input_fingerprint=chat_input_fingerprint,
            PROBE_CONFIG=PROBE_CONFIG,
            extract_probe_features=extract_probe_features,
            probe_sensitivities=probe_sensitivities,
            token_margin=token_margin,
            probe_input_contract=probe_input_contract,
            validate_probe_input_contract=validate_probe_input_contract,
            validate_input_record=validate_input_record,
        )
        exec(
            notebook_cell(
                "notebooks/flenqa_probe_assets.ipynb", "extract-hidden-states"
            ),
            ns,
        )
        trained_features = ns["layer_features"]
        for key in generated_inputs:
            torch.testing.assert_close(seen[-1][key], generated_inputs[key])

        # Execute generation's record/schema assembly with the result from the
        # real generate_chat call above, avoiding another 400-token generation.
        ns.update(
            pa=pa,
            rows=[row],
            prepare_prompts=prepare_prompts,
            InferenceConfig=InferenceConfig,
            generate_chat=lambda *args, **kwargs: generated,
            MODEL_NAME="synthetic",
            PROJECT_COMMIT="synthetic",
        )
        save_cell = notebook_cell(
            "notebooks/flenqa_full_run.ipynb", "save-model-outputs"
        )
        exec(save_cell[: save_cell.index("assert len(records)")], ns)
        generated_table = pa.Table.from_pylist(
            ns["records"], schema=ns["MODEL_OUTPUT_SCHEMA"]
        )
        ns.update(
            test_prompts=[prompt],
            answer_token_variants=answer_token_variants,
            model_records=generated_table.to_pandas().set_index("prompt_id"),
        )
        exec(
            notebook_cell("notebooks/flenqa_probe_eval.ipynb", "extract-test-states"),
            ns,
        )
        for key in generated_inputs:
            torch.testing.assert_close(seen[-1][key], generated_inputs[key])
        for trained, evaluated in zip(
            trained_features, ns["layer_features"], strict=True
        ):
            torch.testing.assert_close(trained, evaluated)

        weight = torch.arange(1, 9, dtype=torch.float32) / 10
        ns.update(
            ActivationRecorder=ActivationRecorder,
            lens_model=SimpleNamespace(layers=model.model.layers),
            available_layers=[0],
            checkpoint={
                "layers": {
                    layer: {
                        "weight": weight,
                        "training_mean": torch.zeros(8),
                        "bias": torch.tensor(0.3),
                    }
                    for layer in range(2)
                }
            },
            saved_scores=pd.DataFrame(
                [
                    {
                        "prompt_id": prompt.prompt_id,
                        "layer": layer,
                        "probe_score": float(state[0] @ weight + 0.3),
                        "input_sha256": generated.input_sha256,
                        "n_input_tokens": generated.input_token_count,
                        "output_margin": ns["output_margins"][0],
                    }
                    for layer, state in enumerate(trained_features)
                ]
            ).set_index(["prompt_id", "layer"]),
        )
        model.requires_grad_(False)
        prior_hooks = [dict(layer._forward_hooks) for layer in model.model.layers]
        exec(
            notebook_cell(
                "experiments/flenqa_probe_jlens/flenqa_probe_jlens.ipynb",
                "prompt-sensitivity-function",
            ),
            ns,
        )
        rows = ns["prompt_sensitivity"](prompt)
        assert len(rows) == 2 and all(np.isfinite(row["sensitivity"]) for row in rows)
        for key in generated_inputs:
            torch.testing.assert_close(seen[-1][key], generated_inputs[key])
        assert [
            dict(layer._forward_hooks) for layer in model.model.layers
        ] == prior_hooks
        # Equal-length token changes must fail before a sensitivity can be interpreted.
        ns["saved_scores"]["input_sha256"] = "stale"
        with pytest.raises(ValueError, match="Regenerate"):
            ns["prompt_sensitivity"](prompt)
    finally:
        with_hook.remove()
