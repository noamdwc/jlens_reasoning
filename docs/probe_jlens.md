# Probe–J-Lens analysis

`jlens_reasoning.probe_jlens` owns the combination of learned probes and J-Lens
analysis, including the one-axis sensitivity and second-order utilities for
[the probe–J-Lens routing framework](probe_jlens_routing_framework.md).

## Boundary

Core [`probing`](probing.md) owns feature/input contracts, hidden-state extraction,
probe fitting, scoring, evaluation and checkpoints. `probe_jlens` consumes those
APIs and owns direction transport, prompt derivatives, and comparisons between
probe representations and downstream influence. Experiments choose data, layers,
cohorts, objectives, result aggregation and plots.

Dependencies flow from `probe_jlens` to `probing`; core probing does not import
or re-export the J-Lens analysis. Training and evaluating probes therefore do not
load the J-Lens hooks used by sensitivity analysis.

```python
from jlens_reasoning.probe_jlens import (
    ProbeSensitivity,
    probe_sensitivities,
    static_probe_projection,
)
from jlens_reasoning.probing import ProbeConfig, token_margin
```

## Directions and sensitivities

`static_probe_projection(J, weight, unembedding)` computes `J @ unit_weight`,
then `unembedding @ projected_direction`. Rows of J are target coordinates;
columns are source coordinates. These are linear vocabulary scores, not
normalized lens logits.

`probe_sensitivities` takes the model, tokenizer, prompt, frozen probes, model
blocks, `ProbeConfig`, and a differentiable scalar objective over the next-token
logits. For example, `lambda logits: token_margin(logits, positive_ids,
negative_ids)` computes a mean-logit contrast. The experiment chooses token
groups; the shared module owns feature selection, gradients, projection onto the
unit probe, and hook cleanup, including on failures. Supplying `saved_records`
requires matching input hashes and reproducing saved probe scores/output margins
within the explicit tolerances before returning sensitivities.

Layer l always refers to `hidden_states[l + 1]`. The final entry is normalized
and must not be transported through a J-Lens map fitted to a raw block output.
Sensitivity calculations include that final entry and validate earlier block
boundaries against recorded activations. Token positions select the probed
feature; output objectives always use the next-token logits at the input end.

## Relationship to the routing framework

The implemented `probe_sensitivities` measures a scalar output derivative along
the fixed positive-class probe direction:

```text
sensitivity = gradient_h(output_objective) · normalized(probe_weight)
```

The framework's proposed J-sensitivity is a different quantity:

```text
s_j = norm(J_l(x) @ normalized(d_j))
d_j = P_l @ centered_hidden_state
```

It measures propagation into the final hidden representation along the content
currently present in the probe subspace. The current output-margin derivative
must not be labeled as that final-hidden-state norm.

## Second-order routing utilities

The shared utilities implement the one-dimensional probe case. They do not
establish a measured effect on reasoning or implement a multidimensional probe
subspace. Existing output-margin calculations and chat-v2 artifacts are unchanged.

```python
from jlens_reasoning.probe_jlens import (
    j_gain, j_sensitivity, probe_component, prompt_hidden_map,
)

model.eval()
model.requires_grad_(False)
mapping = prompt_hidden_map(
    model, tokenizer, prompt, layer=layer, blocks=blocks, config=config,
    input_record=saved_answer,
)
d = probe_component(probe, mapping.hidden)
m = float(d.norm())
# Zero content has no unit direction: record that case instead of normalizing it.
if m > 0:
    result = j_gain(mapping.final_hidden, mapping.hidden, d)
    s = result.sensitivity
    g = result.gain
    if g.norm() > 0:
        delta = step_size * g / g.norm()
        edited = (mapping.hidden + delta).to(mapping.hidden)
        # Reuse clean d: do not redefine the direction after editing.
        edited_s = j_sensitivity(mapping.final_hidden, edited, d)
```

`probe_component` returns `P @ (hidden - training_mean)` without probe bias.
Its norm is the proposed J-merging magnitude; retain its signed projection and
the ordinary probe score as well. A zero component remains zero.

`prompt_hidden_map` captures the raw source block output at the configured token
and returns a callable mapping a replacement vector to the **final normalized
last-token state**. Other source-layer positions stay fixed. The final probe
layer is post-normalization and is rejected as a raw-block source. The model
must remain frozen and in eval mode while using the map. Input hashes are
checked when supplied; checkpoint/model compatibility is still owned by the
probe loader. For a final-token source, the map caches the fixed causal prefix
once, then differentiates only a single-token continuation. Each call copies
that prefix cache, and recurrent-state writes are discarded because there is
no later decode step. This avoids retaining a full-prompt higher-order graph.
The clean target must reproduce full prefill within `atol=rtol=1e-4` before the
map is returned. Earlier source positions retain the full-prompt, checkpointed
path. Hooks and forwards are restored after each call, including on failure.

`j_sensitivity` and `j_gain` also accept ordinary differentiable vector-to-vector
functions, making their geometry testable independently of a model. They use a
matrix-free Jacobian-vector product. `j_gain` differentiates its squared norm
with the supplied direction detached and normalized, then removes the component
along that axis. It returns **unnormalized** gain; affine maps have zero gain.
The chosen axis is the entire excluded subspace in this one-dimensional API.

Higher-order autograd support depends on the actual model and attention kernels;
unsupported operations fail rather than switching objectives or approximating
silently. Low-precision casts can change the realized step and probe score.
Tiny-model CPU tests validate the derivatives against full-prompt replay. The
cached-prefix map also passes all 48 finite-difference checks across the
24-prompt Qwen development cohort on an A100 40 GB, peaking at 18.63 GiB.
Sensitivity changes as intended, while generated answers remain unchanged in
this pilot; see the [pilot report](../experiments/flenqa_probe_jgain/README.md).

## Shared intervention and analysis helpers

- `experiments_utils.interventions.add_prompt_delta(block, delta,
  prompt_length=..., token_position=...)` is a context manager around one
  `generate_chat` call. It adds the delta at one token on the first block call,
  verifies that call contains one full prompt, and leaves later decode calls
  untouched. It supports tensor or tuple block outputs and always removes its
  hook. Zero deltas provide an identity control. This is an inference edit;
  use the hidden-state map for derivatives.
- `experiments_utils.controls.random_orthogonal_direction(axis, norm=..., seed=...)`
  draws a reproducible direction orthogonal to one probe axis at a requested
  norm. It returns at least float32; measure the realized change after casting
  to the model's activation dtype.
- `experiments_utils.statistics.paired_problem_bootstrap(problem_ids,
  differences, seed=...)` takes already-paired differences. It averages within
  problems, then bootstraps equally weighted problem means. It returns the mean,
  percentile interval, and number of independent problems. Experiments remain
  responsible for pairing, coverage, and within-problem weighting.
