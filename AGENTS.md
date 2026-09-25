# J-Lens research work

- Keep research notebooks short and readable from top to bottom. Reuse the existing loaders and project utilities; avoid building a framework inside a notebook. Put reusable computation in the existing package only when it has a concrete reuse need.
- Keep shared probing logic in `src/jlens_reasoning/probing/` and probe/J-Lens integration in `src/jlens_reasoning/probe_jlens.py`. Notebooks should call these implementations rather than duplicate their internals.
- Preserve the requested branch, experiment, and artifact scope. Inspect the relevant producer and consumer before changing saved assets or their loading. Do not regenerate expensive assets when compatible saved outputs already answer the task.
- When asked what to rerun, separate semantic input changes from compatible refactors. Give the minimum affected notebook order, prerequisites, reusable assets and safe parallelism; distinguish that advice from completed execution.
- For analysis, distinguish implementation from measured results, association from intervention evidence, and probe decodability from causal model use.
- Preserve scientific contracts when simplifying: dataset population and coverage; problem identity and split separation; prompt/token/position alignment; model/lens/layer compatibility; coordinate/sign conventions and experimental controls. Solve memory constraints without silently narrowing the study population.
- Enforce artifact/schema/alignment checks at their owning loader or public boundary, then let downstream code use that contract. Avoid repeating the same checks in notebooks and internal helpers. Retain checks whose failure would change the experiment's meaning or corrupt its outputs.
- For notebook and artifact changes, read the relevant section of `docs/REPRODUCING.md`. For probes, use `docs/probing.md`; for combined probe/J-Lens work, use `docs/probe_jlens.md`. Read `docs/probe_jlens_routing_framework.md` only for that research direction.
- Use `uv` and the locked project environment. Run affected CPU tests; the full documented checks are in `README.md`. Report model-backed checks as unrun when unavailable.
- Keep generated outputs and disposable process notes out of Git according to `.gitignore`; retain useful research conclusions in project docs.
