# Probe–J-Lens Routing Framework for Long-Context Reasoning

**Status:** Working research note  
**Purpose:** Record the current definitions, relevant prior work, current empirical evidence, and the next experimental plan.  
**Important:** Terms such as **J-merging**, **J-sensitivity**, **J-direction**, and **J-gain** are project working terminology, not established names in the literature.

---

## 1. Research goal

The project started from a familiar long-context observation:

> A model can fail even when task-relevant information is still decodable from its hidden state.

Probing alone can tell us whether information is represented, but not whether the model is actually using that information downstream. The Jacobian Lens (J-Lens) gives a model-derived map from an intermediate residual-stream representation to the final hidden-state basis.

The research goal is therefore **not** merely to show that relevant information survives long context, and **not** merely to amplify a probe direction.

The stronger goal is:

> **Identify task-relevant information with probing, measure how strongly that information influences the final hidden state, and intervene on the hidden state so the downstream computation becomes more or less sensitive to that information while leaving the information itself approximately unchanged.**

This is intended as a general mechanism-level intervention framework, not a TRUE/FALSE steering trick. FLenQA is the first testbed.

---

## 2. What we have learned so far

### 2.1 Probe performance changes under long context

At layer 18, probe AUROC drops from:

- short context: **0.994**
- long context: **0.763**

Therefore the issue is not only a fixed classification threshold becoming less appropriate. Long context also reduces label separability.

This is one reason **not** to use the probe's classification threshold as the intervention rule.

### 2.2 Static J-Lens projection does not show simple suppression

For the learned probe direction \(w_l\), the static average-Jacobian projection

\[
\bar J_l \hat w_l
\]

does not appear strongly suppressed in norm at layer 18:

\[
\|\bar J_{18}\hat w_{18}\| \approx 1.147
\]

versus a random-direction baseline of approximately

\[
1.139.
\]

So the current evidence does **not** support a simple story in which the probe direction disappears during propagation.

### 2.3 Static output orientation becomes more answer-related later

The output-space orientation of the probe direction changes across layers. In the current analysis, True–False alignment rises from roughly:

\[
0.033 \quad \text{at layer 18}
\]

to:

\[
0.199 \quad \text{at layer 30}.
\]

Vocabulary projections also become more answer-related later, with terms such as `YES`, `confirmed`, `False`, and `Wrong` appearing among stronger directions.

This suggests a rough picture in which a task-label-related representation becomes increasingly aligned with answer vocabulary later in the network.

However, static \(\bar J_l\) is the same for every prompt at a layer, so it cannot explain why one specific long prompt succeeds and another fails.

### 2.4 Prompt-specific sensitivity is the more interesting signal

For the three initially inspected long-context failures at layer 18:

- the probe margin increased;
- prompt-specific output sensitivity decreased.

This is preliminary because:

- the sample is tiny;
- the saved-answer and probe pipelines currently have an input-format mismatch.

But the pattern is consistent with:

> **relevant information remains represented, while its downstream influence weakens.**

This is the mechanism we now want to test directly.

---

## 3. Core mathematical objects

Let:

- \(h_l\) be the hidden representation at layer \(l\) for a specific prompt;
- \(\tilde h_l\) be the representation after applying exactly the same centering/normalization used by the probe;
- \(\mathcal S_l\) be the task-relevant subspace identified by the probe;
- \(P_l\) be the projector onto \(\mathcal S_l\);
- \(h_{\mathrm{final}}\) be the final hidden representation used by the model before final output decoding.

For a one-dimensional linear probe, \(\mathcal S_l\) may initially be the span of the normalized probe vector \(\hat w_l\). Later, this can be generalized to a multidimensional probe-derived subspace.

### 3.1 Prompt-specific Jacobian

Define the prompt-specific Jacobian:

\[
J_l(x)
=
\frac{\partial h_{\mathrm{final}}}{\partial h_l}.
\]

This is different from the static J-Lens transport matrix:

\[
\bar J_l
=
\mathbb E_x\left[
\frac{\partial h_{\mathrm{final}}}{\partial h_l}
\right].
\]

The static matrix is useful for average geometry and vocabulary interpretation. The prompt-specific Jacobian is required for explaining or intervening on a particular prompt.

---

## 4. Working definitions

### 4.1 \(d_j\): J-relevant component / J-direction

Define:

\[
\boxed{
d_j = P_l \tilde h_l
}
\]

This is the part of the current hidden representation that lies in the task-relevant subspace learned by the probe.

For nonzero \(d_j\), define its normalized direction:

\[
\boxed{
\hat d_j = \frac{d_j}{\|d_j\|}
}
\]

Interpretation:

> **\(d_j\) is the task-relevant information currently active in this prompt at this layer.**

For a binary probe, we do **not** threshold its sign to decide TRUE or FALSE. The two sides of the probe axis are treated as two possible contents of the same task-relevant subspace.

This avoids using the probe as an oracle for the correct answer.

---

### 4.2 \(m_j\): J-merging

Define:

\[
\boxed{
m_j = \|d_j\|
}
\]

Interpretation:

> **J-merging measures how strongly the current hidden representation occupies the probe-identified task-relevant subspace.**

It answers:

> How much task-relevant signal is present here?

It does **not** answer whether the model uses that signal.

For a one-dimensional probe, this is closely related to the magnitude of the projection onto the probe axis, but the exact implementation must respect the probe's preprocessing and geometry.

---

### 4.3 \(s_j\): J-sensitivity

Define:

\[
\boxed{
s_j
=
\left\|
J_l(x)\hat d_j
\right\|
}
\]

Interpretation:

> **J-sensitivity measures how strongly a unit perturbation in the currently represented task-relevant direction at layer \(l\) propagates into the final hidden representation.**

It answers:

> If this relevant information changes slightly here, how much does \(h_{\mathrm{final}}\) change?

This is intentionally about the **final hidden state**, not merely activity inside the current layer and not directly about increasing a particular output token.

That distinction is central to the project.

---

## 5. Why \(m_j\) and \(s_j\) must be separated

A hidden state can have:

\[
m_j \text{ high}, \qquad s_j \text{ high}
\]

meaning the relevant information is present and has strong downstream influence.

Or:

\[
m_j \text{ high}, \qquad s_j \text{ low}
\]

meaning the information is present but downstream computation is relatively insensitive to it.

The second case is especially interesting for long-context failures.

The target mechanism is therefore **not necessarily information loss**. It may instead be a failure of routing, leverage, or downstream use.

---

## 6. Static J-Lens versus prompt-specific J-sensitivity

The static J-Lens gives:

\[
\bar J_l \hat d_j.
\]

This describes how the direction propagates **on average across contexts**.

It is useful for:

- mapping the probe direction into the final hidden-state basis;
- projecting through the unembedding to inspect vocabulary orientation;
- comparing probe directions against known J-Lens vectors;
- identifying candidate layers and broad geometry.

But static \(\bar J_l\) cannot distinguish two prompts at the same layer.

For prompt-level reasoning failures, use:

\[
J_l(x)\hat d_j.
\]

This gives prompt-specific \(s_j\).

A practical research pattern is:

1. use static J-Lens for broad screening and interpretation;
2. use prompt-specific Jacobians only on selected layers/prompts for mechanism tests.

---

## 7. The new intervention target: increase use, not content

A naive probe intervention would do something like:

\[
h_l' = h_l + \alpha d_j.
\]

That strengthens the task-relevant representation itself.

This is useful as a baseline, but it is **not our main goal**.

Our main question is:

> **Can we change the hidden state so the downstream network becomes more sensitive to the task-relevant information already present, without directly increasing that information?**

In our terminology:

\[
m_j' \approx m_j
\]

while:

\[
s_j' > s_j.
\]

For a degradation experiment:

\[
m_j' \approx m_j
\]

while:

\[
s_j' < s_j.
\]

If these interventions improve or damage reasoning respectively, they provide stronger causal evidence for a routing/use mechanism than simply amplifying the probe feature.

---

## 8. \(g_j\): J-gain direction

### 8.1 Freeze the content direction

When defining the intervention, compute \(d_j\) and \(\hat d_j\) from the clean hidden state and **hold \(\hat d_j\) fixed**.

This matters because we want to change the network's sensitivity to the existing information, not redefine the information direction while optimizing it.

Define:

\[
q_j(h_l)
=
\frac{1}{2}
\left\|
J_l(h_l)\hat d_j
\right\|^2.
\]

Since:

\[
s_j = \|J_l(h_l)\hat d_j\|,
\]

increasing \(q_j\) locally increases \(s_j\).

### 8.2 Local second-order direction

The direct local direction is:

\[
\nabla_{h_l} q_j.
\]

Because \(J_l\) is already a first derivative of \(h_{\mathrm{final}}\), this gradient contains **second-order derivative information**.

To avoid directly changing the probe-relevant content, project this gradient outside the probe subspace:

\[
\boxed{
g_j
=
(I-P_l)
\nabla_{h_l}
\frac{1}{2}
\left\|
J_l(h_l)\hat d_j
\right\|^2
}
\]

Call \(g_j\) the **J-gain direction**.

Interpretation:

> **\(g_j\) is a local hidden-state direction that should increase downstream sensitivity to the current task-relevant representation while avoiding a direct move inside the probe subspace.**

Then use a small intervention:

\[
\boxed{
h_l' = h_l + \eta g_j
}
\]

for sensitivity enhancement, or:

\[
\boxed{
h_l' = h_l - \eta g_j
}
\]

for sensitivity suppression.

This is a **single local step**, not an optimization loop.

---

## 9. Why this is second-order J-Lens information

First-order J-Lens asks:

\[
J_l \hat d_j
\]

> Where does this information go?

Our intervention asks:

\[
\nabla_{h_l}
\frac{1}{2}
\|J_l\hat d_j\|^2
\]

> How should the current hidden state change so this information propagates more strongly?

Because:

\[
J_l
=
\frac{\partial h_{\mathrm{final}}}{\partial h_l},
\]

changing \(J_l\) with respect to \(h_l\) requires:

\[
\frac{\partial J_l}{\partial h_l}
=
\frac{\partial^2 h_{\mathrm{final}}}{\partial h_l^2}.
\]

We do **not** need to materialize the full Hessian.

The needed quantity is a contracted second-order derivative and should be computable with autograd using a Jacobian-vector product plus a second backward/differentiation step.

---

## 10. Important local guarantee

Let:

\[
c_j = \nabla_{h_l} q_j
\]

and:

\[
g_j=(I-P_l)c_j,
\]

assuming \(P_l\) is an orthogonal projector in the representation space being used.

Then for small positive \(\eta\):

\[
q_j(h_l+\eta g_j)
\approx
q_j(h_l)
+
\eta\,c_j^\top g_j.
\]

Because:

\[
c_j^\top g_j
=
c_j^\top(I-P_l)c_j
=
\|g_j\|^2
\ge 0,
\]

the step locally increases \(q_j\) whenever \(g_j\neq0\).

At the same time:

\[
P_l g_j=0,
\]

so the intervention does not directly alter the probe-subspace component in the idealized linear geometry.

This is the key reason the projected second-order direction is attractive: it gives a simple local intervention rather than an optimization problem.

---

## 11. Relation to the spider ↔ ant experiment

The J-Lens paper demonstrates that internal reasoning can be redirected by swapping J-Lens coordinates for an inferred intermediate concept.

Example:

> "The number of legs on the animal that spins webs is ..."

The J-Lens surfaces a `spider` representation in intermediate layers. Swapping the `spider` J-Lens coordinate for `ant` changes the model's answer from `8` to `6`.

The important conceptual lesson is that changing an internal intermediate representation can redirect later reasoning rather than merely editing the final output.

Our problem is harder:

- we do not know in advance which vocabulary concept corresponds to "the relevant information";
- the relevant representation may not have a clean single-token name;
- we do not want to replace the content with an alternative answer;
- we want to change how strongly downstream computation uses the existing relevant representation.

The probe therefore serves as a task-specific way to find the relevant representation when no clean `spider`/`ant` token vector is available.

---

## 12. Related work: directly relevant

### 12.1 Gurnee et al. (2026): Jacobian Lens and J-space

**Paper:** *Verbalizable Representations Form a Global Workspace in Language Models*  
https://transformer-circuits.pub/2026/workspace/

The paper defines the J-Lens using an average Jacobian transport from intermediate residual streams to the final-layer basis.

Most relevant findings for this project:

1. J-Lens coordinate swaps can causally redirect inferred intermediate concepts, including spider→ant.
2. The paper goes beyond predefined vocabulary vectors: for two-hop factual prompts it constructs probes for unspoken intermediate concepts.
3. It decomposes those probes into a J-space component and a J-orthogonal remainder.
4. The paper reports that the J-space component often explains only about 10–15% of probe variance, yet carries most of the observed causal effect.
5. Swapping the J-space component of the probes changes the target answer much more reliably than swapping the non-J-space remainder.

**Why it matters here**

This is the closest prior work to our probe–J-Lens combination and must be treated as foundational rather than rediscovered.

Their question is approximately:

> Which part of a probe-derived representation lies in J-space and causally mediates an intermediate concept?

Our proposed extension is different:

> For a task-relevant representation already present in a particular prompt, how sensitive is the downstream computation to it, and can we modify the hidden state to increase or decrease that sensitivity without directly changing the represented content?

This distinction should be tested carefully before making any novelty claim.

---

### 12.2 Giulianelli et al. (2018): probe-guided intervention

**Paper:** *Under the Hood: Using Diagnostic Classifiers to Investigate and Improve how Language Models Track Agreement Information*  
https://aclanthology.org/W18-5426/

They train diagnostic classifiers on LSTM hidden states to track grammatical-number information, identify where that information becomes corrupted, and intervene on internal states. The intervention substantially improves difficult agreement predictions.

**Why it matters**

This establishes that probe information can be used not only diagnostically but also to change model behavior and improve performance.

**Difference from our target**

Their intervention changes the represented feature toward a desired grammatical value. Our primary target is to alter **downstream sensitivity/use** while preserving the task-relevant content as much as possible.

---

### 12.3 Elazar et al. (2021): Amnesic Probing

**Paper:** *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals*  
https://aclanthology.org/2021.tacl-1.10/

The central lesson is that probe decodability does not establish behavioral importance. They remove linearly encoded information and measure the behavioral consequences.

**Why it matters**

This directly motivates the separation between:

\[
m_j
\]

and:

\[
s_j.
\]

High probe signal does not imply that the model is relying on it.

---

### 12.4 Ravfogel et al. (2021): AlterRep

**Paper:** *Counterfactual Interventions Reveal the Causal Effect of Relative Clause Representations on Agreement Prediction*  
https://aclanthology.org/2021.conll-1.15/

AlterRep constructs counterfactual hidden representations by changing how a probed linguistic feature is encoded while attempting to leave other aspects intact, then measures the effect on model behavior.

**Why it matters**

This is a strong causal-probing baseline and provides methodology for controlled representation intervention.

**Difference from our target**

AlterRep edits the **value/content of a feature**. We want to test an intervention that changes the **gain/routing of a feature already present**.

---

### 12.5 Nanda, Lee & Wattenberg (2023): OthelloGPT

**Paper:** *Emergent Linear Representations in World Models of Self-Supervised Sequence Models*  
https://arxiv.org/abs/2309.00941

They show that OthelloGPT contains useful linear representations of board state and that this understanding enables control of model behavior using simple vector arithmetic.

**Why it matters**

It is an important precedent for the broader idea:

\[
\text{find a latent representation}
\rightarrow
\text{edit it}
\rightarrow
\text{change computation/behavior}.
\]

---

### 12.6 Belrose et al. (2023): LEACE

**Paper:** *LEACE: Perfect Linear Concept Erasure in Closed Form*  
https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html

LEACE provides a closed-form method for removing linearly detectable concept information while minimally changing the representation under its stated geometry.

**Why it matters**

It is useful as an established **content-suppression baseline**.

Our main degradation experiment is different: rather than erase the relevant information (\(m_j\downarrow\)), we want to decrease its downstream sensitivity (\(s_j\downarrow\)) while keeping \(m_j\) approximately fixed.

---

### 12.7 Canby et al. (2025): reliability of causal probing

**Paper:** *How Reliable are Causal Probing Interventions?*  
https://aclanthology.org/2025.ijcnlp-long.47/

The paper evaluates causal-probing interventions using two major criteria:

- **completeness:** how fully the targeted representation is transformed;
- **selectivity:** how little unrelated information is changed.

It finds a tradeoff between them and shows that intervention reliability matters for downstream behavioral effects.

**Why it matters**

Our intervention must be validated for **selectivity**, not merely judged by whether accuracy moves in the hoped-for direction.

For our specific method, a key validation target is:

\[
m_j' \approx m_j
\]

while \(s_j\) changes.

We should additionally measure changes to unrelated probe directions / hidden-state geometry.

---

## 13. Related work worth mentioning but not central

### Activation Addition / activation engineering

Turner et al. (2023), *Steering Language Models With Activation Engineering*  
https://arxiv.org/abs/2308.10248

Activation Addition constructs contrastive steering vectors and adds them to intermediate activations at inference time.

Relevant as a simple steering baseline, but it does not use probe-derived task relevance or Jacobian sensitivity.

### Representation Engineering

Zou et al. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency*  
https://arxiv.org/abs/2310.01405

This work studies population-level representations and their monitoring/manipulation, including safety-relevant properties.

Relevant to future generalization beyond FLenQA, especially if the framework is applied to safety-related representations.

---

## 14. What is already known versus what we are testing

### Already established by prior work

We should **not** claim novelty for the following:

- linear probes can decode latent information;
- high probe accuracy does not prove causal use;
- probe-derived representations can be erased or counterfactually changed;
- modifying hidden representations can change downstream behavior;
- probe-derived concepts can overlap with causally important J-space components;
- J-Lens coordinate interventions can redirect intermediate reasoning.

### Current research hypothesis

The specific hypothesis we want to test is:

> **Some long-context reasoning failures occur when task-relevant information remains represented at an intermediate layer but the downstream computation becomes less sensitive to that information.**

And the intervention hypothesis is:

> **A small hidden-state change outside the probe-relevant subspace, chosen from second-order Jacobian information, can increase or decrease downstream sensitivity to the existing relevant representation without directly changing that representation, thereby improving or degrading reasoning.**

This is a hypothesis, not yet an established result.

---

## 15. Experimental plan

### Phase 0 — fix comparability

Before interpreting prompt-specific results:

1. use exactly the same input/chat formatting for:
   - answer generation,
   - activation extraction,
   - probe evaluation,
   - prompt-specific Jacobian computation;
2. reproduce the existing probe metrics;
3. verify the three initial layer-18 failures under the corrected pipeline.

No causal claim should rely on the current mismatched input formats.

---

### Phase 1 — validate \(d_j\), \(m_j\), and \(s_j\)

For each selected layer and prompt:

1. compute the probe-relevant component:
   \[
   d_j=P_l\tilde h_l;
   \]
2. compute:
   \[
   m_j=\|d_j\|;
   \]
3. compute prompt-specific:
   \[
   s_j=\|J_l(x)\hat d_j\|;
   \]
4. compare across:
   - short correct;
   - long correct;
   - long incorrect;
   - ideally matched versions of the same underlying problem.

Primary question:

> Do long failures show systematically different \(s_j\) after controlling for \(m_j\)?

Do not rely on three examples. Use a proper cohort and uncertainty estimates.

---

### Phase 2 — static J-Lens screening

Use \(\bar J_l\) to cheaply characterize all layers:

- \(\|\bar J_l\hat w_l\|\);
- alignment with J-space / J-Lens dictionary;
- vocabulary projections;
- comparison to random directions;
- comparison across short/long probe directions.

Use this only as a layer-selection and interpretation tool.

Do **not** treat static \(\bar J_l\) as a prompt-specific failure explanation.

---

### Phase 3 — implement J-gain \(g_j\)

For a small number of selected prompts/layers:

1. obtain clean \(\hat d_j\);
2. freeze \(\hat d_j\);
3. compute:
   \[
   q_j=\frac12\|J_l(h_l)\hat d_j\|^2;
   \]
4. compute:
   \[
   c_j=\nabla_{h_l}q_j;
   \]
5. remove the direct probe-subspace component:
   \[
   g_j=(I-P_l)c_j;
   \]
6. normalize/scale \(g_j\) using a controlled intervention magnitude;
7. intervene with:
   \[
   h_l'=h_l+\eta g_j
   \]
   and:
   \[
   h_l'=h_l-\eta g_j.
   \]

No iterative optimizer is required for the first experiment.

---

### Phase 4 — verify that the intervention changes the intended quantity

For each intervention, measure:

#### Content preservation

\[
\Delta m_j = m_j'-m_j
\]

should remain small.

Also check:

- probe score/margin changes;
- unrelated validation probes;
- hidden-state norm;
- cosine similarity to the clean state.

#### Sensitivity manipulation

Measure:

\[
\Delta s_j=s_j'-s_j.
\]

Expected:

- \(+g_j\): \(\Delta s_j>0\);
- \(-g_j\): \(\Delta s_j<0\).

The first success criterion is **not accuracy**. It is showing that the intervention changes \(s_j\) in the expected direction while minimally changing \(m_j\).

---

### Phase 5 — test reasoning behavior

Only after Phase 4 works:

Compare behavioral outcomes under:

1. clean run;
2. \(+g_j\) sensitivity-enhancing intervention;
3. \(-g_j\) sensitivity-suppressing intervention;
4. random orthogonal direction with matched norm;
5. naive \(+d_j\) probe amplification baseline;
6. optionally a content-erasure baseline such as LEACE/INLP;
7. optionally a static J-Lens/J-space steering baseline.

Main causal predictions:

\[
s_j\uparrow,\ m_j\approx\text{constant}
\Rightarrow
\text{reasoning improves}
\]

and:

\[
s_j\downarrow,\ m_j\approx\text{constant}
\Rightarrow
\text{reasoning degrades}.
\]

The strongest result would be bidirectional and dose-responsive.

---

## 16. Controls we should require

### Random-direction control

Use random directions matched for:

- norm;
- orthogonality to the probe subspace;
- layer and position.

### Probe amplification control

Compare against:

\[
h_l+\alpha d_j.
\]

This tells us whether changing sensitivity adds value beyond simply strengthening the decoded feature.

### Static-J control

Construct a comparable intervention using only static \(\bar J_l\) information if possible.

This tests whether prompt-specific second-order structure matters.

### Layer control

Apply the same intervention magnitude at:

- candidate low-\(s_j\) layers;
- layers with similar \(m_j\) but higher \(s_j\);
- unrelated layers.

### Sign control

Test both:

\[
+g_j
\]

and:

\[
-g_j.
\]

A bidirectional effect is much stronger evidence than improvement alone.

### Selectivity controls

Following the causal-probing literature, verify that the intervention does not simply distort the entire representation.

---

## 17. Key implementation questions still open

These should be resolved experimentally, not hidden inside notation.

### 17.1 What exactly is \(h_l\)?

Possibilities include:

- final prompt token only;
- the task-relevant token position;
- multiple selected positions;
- a pooled representation.

The probe and Jacobian must use compatible objects.

### 17.2 One probe vector or a subspace?

Start with the current linear \(w_l\) for simplicity.

But a single probe direction should not automatically be interpreted as the full representation. Later work should consider a multidimensional task-relevant subspace.

### 17.3 Which geometry defines \(P_l\)?

If the probe uses normalized/centered hidden states, the projection must be defined in the same representation space.

For a general linear transform or covariance-aware probe, naive Euclidean projection may be wrong.

### 17.4 What final hidden state?

The definition of \(h_{\mathrm{final}}\) should match the J-Lens setup:

- exact layer;
- token position(s);
- normalization point.

This must be explicit before comparing static and prompt-specific Jacobians.

### 17.5 Norm-only sensitivity may be too broad

Current:

\[
s_j=\|J_l\hat d_j\|.
\]

This asks whether the relevant direction changes **anything** in the final hidden state.

Later, we may need a structured sensitivity measure that asks whether the propagated change remains inside a useful final subspace rather than arbitrary directions.

Do not introduce this complexity before testing the simple norm definition.

### 17.6 Second-order cost

The J-gain calculation uses second-order autograd. We should benchmark:

- memory;
- runtime;
- layer-by-layer feasibility;
- selected-position versus all-position computation.

The initial experiment should use a small number of layers and prompts.

---

## 18. Minimal next experiment

The next notebook should be intentionally small.

### Cohort

Use a small, matched set of:

- short correct prompts;
- corresponding long prompts;
- prioritize long failures where probe information remains strong.

### Layer

Start with layer 18 because:

- the current probe analysis is informative there;
- we already have preliminary prompt-specific sensitivity observations there.

Do not assume layer 18 is ultimately optimal.

### Measurements

For each prompt:

\[
m_j,\quad s_j.
\]

Then for a very small subset compute \(g_j\) and test a few \(\eta\) values.

### Required output table

At minimum:

| prompt | condition | correct | \(m_j\) | \(s_j\) | \(\Delta m_j\) | \(\Delta s_j\) | answer before | answer after |
|---|---|---:|---:|---:|---:|---:|---|---|

Conditions:

- clean;
- \(+g_j\);
- \(-g_j\);
- random orthogonal;
- \(+d_j\) probe amplification.

### Decision criterion

Before scaling up, require evidence that:

1. \(+g_j\) reliably increases \(s_j\);
2. \(-g_j\) reliably decreases \(s_j\);
3. \(m_j\) remains approximately stable;
4. random orthogonal directions do not reproduce the effect;
5. the behavioral direction is at least consistent with the sensitivity change.

If these fail, reconsider the definition of \(s_j\) or the level/position at which it is measured before running a large experiment.

---

## 19. Potential contribution if the hypothesis survives

A cautious version of the contribution would be:

> **Probing identifies task-relevant information, while prompt-specific Jacobian analysis measures whether that information has downstream influence. We test a second-order intervention that changes the model's sensitivity to a probe-identified representation while approximately preserving the representation itself, and evaluate whether this can improve or impair long-context reasoning.**

This is stronger than:

> "The model still knows the answer."

And more general than:

> "Add the probe vector."

It could potentially generalize to settings where the relevant hidden representation is not directly tied to a fixed answer token, including safety-related reasoning, constraint awareness, planning state, or other latent task information.

---

## 20. Terminology summary

| Symbol | Working name | Meaning |
|---|---|---|
| \(\mathcal S_l\) | task-relevant probe subspace | representation identified by probing |
| \(P_l\) | probe projector | projection onto \(\mathcal S_l\) |
| \(d_j=P_l\tilde h_l\) | J-relevant component / J-direction | relevant content currently present |
| \(\hat d_j\) | normalized J-direction | unit direction of current relevant content |
| \(m_j=\|d_j\|\) | J-merging | amount of task-relevant signal present |
| \(J_l(x)\) | prompt-specific Jacobian | local map from \(h_l\) to \(h_{\mathrm{final}}\) |
| \(\bar J_l\) | static J-Lens Jacobian | average map used by J-Lens |
| \(s_j=\|J_l(x)\hat d_j\|\) | J-sensitivity | downstream sensitivity to the relevant information |
| \(g_j=(I-P_l)\nabla_{h_l}\frac12 s_j^2\) | J-gain direction | local off-subspace change intended to increase \(s_j\) without directly changing \(m_j\) |

---

## 21. References

1. Gurnee, W. et al. (2026). **Verbalizable Representations Form a Global Workspace in Language Models.** Transformer Circuits.  
   https://transformer-circuits.pub/2026/workspace/

2. Giulianelli, M., Harding, J., Mohnert, F., Hupkes, D., & Zuidema, W. (2018). **Under the Hood: Using Diagnostic Classifiers to Investigate and Improve how Language Models Track Agreement Information.** BlackboxNLP / EMNLP.  
   https://aclanthology.org/W18-5426/

3. Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). **Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals.** TACL 9, 160–175.  
   https://aclanthology.org/2021.tacl-1.10/

4. Ravfogel, S., Prasad, G., Linzen, T., & Goldberg, Y. (2021). **Counterfactual Interventions Reveal the Causal Effect of Relative Clause Representations on Agreement Prediction.** CoNLL.  
   https://aclanthology.org/2021.conll-1.15/

5. Nanda, N., Lee, A., & Wattenberg, M. (2023). **Emergent Linear Representations in World Models of Self-Supervised Sequence Models.**  
   https://arxiv.org/abs/2309.00941

6. Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). **LEACE: Perfect Linear Concept Erasure in Closed Form.** NeurIPS 36.  
   https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html

7. Turner, A. M. et al. (2023). **Steering Language Models With Activation Engineering.**  
   https://arxiv.org/abs/2308.10248

8. Zou, A. et al. (2023). **Representation Engineering: A Top-Down Approach to AI Transparency.**  
   https://arxiv.org/abs/2310.01405

9. Canby, M. E., Davies, A., Rastogi, C., & Hockenmaier, J. (2025). **How Reliable are Causal Probing Interventions?** IJCNLP-AACL.  
   https://aclanthology.org/2025.ijcnlp-long.47/

---

## 22. One-sentence project direction

> **Use probing to identify the information relevant to a task, J-Lens to measure how that information propagates into the model's final hidden state, and second-order Jacobian structure to increase or decrease the model's downstream sensitivity to that information without directly changing the information itself.**
