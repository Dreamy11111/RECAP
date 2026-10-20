# Data inventory

The files in this directory are generated simulation outputs used to inspect communication-aware cooperative planning under delayed and lossy coordination links.

- `long_rollout_*.csv`: long-horizon traces for the evaluated method variants.
- `runtime_trace_*.csv`: per-step runtime component traces.
- `full_loop_rollout_summary.csv`: aggregate full-loop rollout metrics.
- `runtime_budget_summary.csv`: aggregate runtime budget statistics.
- `persistent_stress_trials.csv`: trial-level results under persistent link stress.
- `persistent_stress_summary.csv`: aggregate persistent-stress results.
- `failure_boundary_grid.csv`: packet-loss and delay parameter sweep.
- `failure_boundary_summary.json`: compact failure-boundary summary.
- `real_closed_loop_*.csv`: representative closed-loop traces for different method variants and fleet sizes.
- `real_closed_loop_summary.csv`: aggregate closed-loop metrics.
- `real_care_validator_injection_log.csv`: recovery and validator event log.
- `artifact_manifest.json` and `real_closed_loop_manifest.json`: generated-file metadata.

All CSV files are plain-text, comma-separated tables with column headers. JSON files are UTF-8 encoded.

