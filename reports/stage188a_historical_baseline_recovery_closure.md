# Stage188-A historical baseline recovery closure

**Decision:** `STAGE174D1_EXACT_HISTORICAL_RUN_NOT_RECOVERABLE`

This Markdown documents the JSON closure at `reports/stage188a_historical_baseline_recovery_closure.json`. That JSON, not original Stage174-D1F runtime provenance, is the exact input to builder option `--stage174d1-reference`. Its required schema fixes `decision`, `reference_decision`, `historical_reference_only: true`, `exact_historical_run_recoverable: false`, `baseline_definition: current_commit_default_off_paired_baseline`, and an `experimental_scope` object containing the verified Stage174-D1F reference facts.

The Stage174-D1F closure recovers only the historical design and result context: `v6b_minimal`, Mamba, seed 174, 20 epochs, `data/controlled_v5_v3_without_time_swap.jsonl`, no time-swap main training, internal clean-dev selection, final CE from `output["logits"]`, no `loss_logits` use, and no external evaluation. Its clean-dev metrics remain historical reference values only.

Exact argv, parsed arguments, the full resolved runtime configuration, historical Git commit, and exact optimizer, scheduler, and batching provenance are absent. Therefore neither an exact Stage174-D1 replay claim nor a historical metric-reproduction claim is permitted.

Those missing historical provenance fields are explicitly non-blocking for Stage188-A. A missing or malformed recovery closure JSON itself is blocking and must produce a fail-closed Stage188-A blocked report rather than being treated as recovered runtime provenance.

Stage188-B instead uses `current_commit_default_off_paired_baseline`. Both arms are newly executed after Stage188-A manifest materialization from the same current commit, trainer source/SHA, environment, and explicit common argv. No historical checkpoint or run artifact is reused. Historical Stage174-D1F results may be cited only as contextual reference.
