# Thesis Experiment Rebuild (2026-03-15)

This workspace contains the isolated rebuild of the thesis experiment pipeline.

Rules for this workspace:

- Do not modify or delete the original experiment scripts or original result files.
- Any rebuilt protocol, component, script, figure, result, or summary must live here.
- Preserve full traceability from old scripts to rebuilt scripts.
- Favor fair and non-degenerate baselines over artificially weak comparisons.
- Do not fabricate data or evaluated case counts.

Directory overview:

- `docs/`: execution notes, protocol specifications, and final summaries.
- `configs/`: experiment configuration files for rebuilt runs.
- `shared/`: reusable builders, evaluators, and helpers.
- `rq1/`: rebuilt parsing and latency experiments.
- `rq2/`: rebuilt graph-quality experiments.
- `rq34/`: rebuilt RCA and end-to-end experiments.
- `edge_profile/`: edge-oriented emulation profiling for local footprint and payload evidence.
- `figures/`: regenerated figures for thesis-facing outputs.
- `reports/`: aggregated summaries and analysis reports.
- `manifests/`: machine-readable run manifests and provenance files.

Current rebuild priorities:

1. Standardize `RQ1` case construction, GT definition, PA evaluation, and Drain baseline protocol.
2. Preserve strong `RQ2` evidence while moving outputs into a cleaner pipeline.
3. Re-run `RQ34` only after the rebuilt upstream protocol is stable.
4. Add explicit edge-oriented profiling so the thesis claim is grounded in measured local footprint and payload reduction, not only narrative framing.
