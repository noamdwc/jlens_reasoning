# Probe–J-Lens analysis

`jlens_reasoning.probe_jlens` owns the combination of learned probes and J-Lens
analysis. It is a small module today and the home for future implementation of
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

Future implementation belongs here: probe-subspace content (`d_j`), J-merging
(`m_j`), prompt-specific final-hidden-state J-sensitivity (`s_j`), and second-order
J-gain directions (`g_j`). Those quantities and interventions are not implemented
by this module yet. Add them as the framework's feature boundaries, geometry and
controls are settled; the core probing package remains responsible for probes.

This separation changes imports and ownership only. Existing calculations,
saved-result tolerances, and chat-v2 artifact formats are preserved.
