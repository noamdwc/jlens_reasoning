# Probing

`jlens_reasoning.probing` is the source of truth for probing mechanics in this
project. Experiments call its public API rather than implementing their own
extraction, fitting, score equations, or checkpoint readers in notebooks.

## Responsibility

| Owner | Responsibility |
| --- | --- |
| `inference` | Chat preparation, generation, and hashing the actual input IDs/mask |
| `probing.contracts` | Configurable input/feature identity and compatibility checks |
| `probing.features` | Extracting layer states at the selected wrapped-input token |
| `probing.linear` | Training-only centering, binary logistic fitting, validation selection, scoring, probabilities, predictions and metrics |
| `probing.artifacts` | Validating, saving and loading frozen probe checkpoints and metadata |
| `probing.objectives` | Differentiable output objectives such as token-logit margins |
| [`probe_jlens`](probe_jlens.md) | Combined probe/J-Lens transport and prompt sensitivity analysis |
| Experiment | Dataset rendering, labels, split assignments, fitting settings, cohorts, output objective, aggregation, plots and artifact paths |

Import through `jlens_reasoning.probing`. Add future reusable probing behavior
there. The current implementation supports centered binary logistic probes over
Transformers layer hidden states; other probe families should extend the package
when an experiment needs them.

## A new experiment

Choose an inference profile and feature position explicitly. Input settings are
shared with generation; the probing API never tokenizes raw text separately.

```python
from jlens_reasoning.inference import InferenceConfig
from jlens_reasoning.probing import (
    ProbeConfig,
    evaluate_probe,
    extract_probe_features,
    fit_binary_probe,
    load_probe_checkpoint,
    probe_input_contract,
    save_probe_checkpoint,
)

config = ProbeConfig(InferenceConfig.direct(max_input_tokens=4096))
# For another experiment, choose InferenceConfig.reasoning(...) or a different
# token_position. Both choices change the saved feature/input contract.
model.eval()
contract = probe_input_contract(tokenizer, config=config)
features = extract_probe_features(model, tokenizer, prompt, config=config)
# features.states[layer]: CPU float32 vector at config.token_position.
# features.logits: next-token logits at the final wrapped-input position.
# features.input_record: exact input hash and token count.

# Assemble feature matrices and labels from development splits chosen by the
# experiment. Fit each layer independently; never pass test data to fitting.
probe = fit_binary_probe(
    train_features, train_labels, validation_features, validation_labels,
    c_grid=(0.01, 0.1, 1.0, 10.0, 100.0), seed=1729,
)
evaluation = evaluate_probe(probe, test_features, test_labels)
# scores, predictions, gold_margins, gold_probabilities, correct, metrics

# `probes_by_layer` contains the fitted probe for every declared model layer.
metadata = {
    **contract,
    "model_name": model_name,
    "num_layers": model.config.num_hidden_layers,
    "hidden_dim": model.config.hidden_size,
    # Record the experiment's split, label convention and fitting settings too.
}
save_probe_checkpoint("probes.pt", probes_by_layer, metadata=metadata,
                      metadata_path="metadata.json")
checkpoint = load_probe_checkpoint("probes.pt", metadata_path="metadata.json",
                                   expected_contract=contract, model_name=model_name)
```

`extract_probe_features(..., input_record=saved_answer)` verifies the actual
input hash before extraction. This is needed when generation and probing run
separately and reload saved results. The tokenizer/template fingerprint detects
changed assets; the per-record hash detects changed actual model inputs.

`score_probe` always computes `(features - training_mean) @ weight + bias`.
`evaluate_probe` predicts class 1 only for scores strictly greater than zero,
then orients margins toward the supplied binary labels. Sigmoid values are probe
scores, not a guarantee of calibration under distribution shift. Generated model
answers are still graded through the separate shared evaluation module.

## Combining probes with J-Lens

Direction transport and prompt sensitivity live in the separate
[`jlens_reasoning.probe_jlens`](probe_jlens.md) module. It consumes core probing
APIs; core probing does not import or re-export the combined analysis. Future
routing-framework implementation belongs in that module.

## Existing FLenQA artifacts

The shared package preserves the existing direct-chat final-token contract and
version-2 tensor checkpoint layout. Moving these mechanics does not itself
require retraining chat-v2 probes. Raw-prompt v1 probes remain incompatible.
FLenQA's settings live in `experiments/flenqa_probe_jlens/constants.py`; its
split, context sizes, regularization grid, label orientation, and failure cohorts
remain visible in the experiment notebooks.

Scikit-learn is a declared, locked package dependency. Rebuild/upload the wheel
bundle before running the migrated notebooks in Colab so their imports resolve
against the updated package.
