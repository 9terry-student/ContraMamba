# Stage191-A existing-artifact trajectory feasibility audit

## Purpose

Stage190 closes the direct selected-checkpoint SUPPORT-gradient-conflict hypothesis. All three intervention checkpoints had no SUPPORT conflict, and the margin gradient was predominantly shared rather than head-local. Its local descent direction improved eligible CE and SUPPORT-specific objectives but worsened all-clean-dev CE at all six checkpoints. This makes optimization trajectory, class redistribution, and checkpoint selection the next hypotheses, without proving any of them.

Before replaying training, Stage191-A must determine whether the existing Stage189/Stage190 artifacts already contain an authoritative trajectory. Reusing sufficient artifacts avoids an unnecessary replay and separates observations written during training from reconstructed claims.

## Authoritative writer paths

Static inspection of `scripts/train_controlled_v6b_minimal.py` proves these single-run report paths:

- Margin epochs: `runs.single.compatible_positive_margin.epoch_metrics`
- Clean-dev epoch diagnostics: `runs.single.v7_epoch_diagnostic_history`
- Final epoch: `runs.single.final_epoch`
- Selected epoch: `runs.single.best_epoch`
- Selection metric: `runs.single.select_metric`
- Selection-rule provenance: `runs.single.audit_ledger.active_selection_rules`

The trainer writes epoch, clean-dev macro-F1, prediction distribution, and selected diagnostic accuracies to the clean history. It does not write per-epoch SUPPORT recall, false-entitlement total, polarity-error total, or clean-dev CE there. Top-level selected-checkpoint metrics are not an epoch trajectory and cannot fill those gaps.

## Sufficiency distinction

Metric-only trajectory analysis requires complete authoritative 20-epoch clean outcomes and exact selection provenance. Gradient-at-every-epoch analysis additionally requires exact epoch-addressable checkpoints for epochs 1 through 20. Missing epoch checkpoints does not invalidate otherwise complete metric trajectory evidence; it only prevents gradient-at-every-epoch claims. Missing required epoch-level clean outcomes requires a deterministic replay specification because those outcomes cannot be inferred from selected predictions, Stage189-D deltas, loss histories, interpolation, or nearby epochs.

## Decisions

Decision priority is:

1. `STAGE191A_TRAJECTORY_FEASIBILITY_BLOCKED` for any failed identity, topology, parsing, or Stage190 closure gate.
2. `STAGE191A_EXISTING_TRAJECTORY_FULL_READY` when all six runs have complete margin and clean histories, exact selection provenance, and checkpoints for every epoch.
3. `STAGE191A_EXISTING_TRAJECTORY_METRIC_ONLY_READY` when the same metric requirements pass but checkpoint coverage is incomplete.
4. `STAGE191A_DETERMINISTIC_REPLAY_REQUIRED` only when all identity and closure gates pass but at least one clean trajectory is incomplete.

The recommended next stage is respectively repair, Stage191-B existing-artifact metric and gradient-trajectory analysis, Stage191-B existing-artifact metric trajectory analysis without gradient-at-every-epoch claims, or a Stage191-B deterministic replay specification. The replay-required decision does not authorize training.

## Safety and advancement

Stage191-A reads structured artifacts, hashes checkpoint files without loading them, and inspects source text for writer contracts. It imports no trainer or model framework, loads no model or checkpoint, runs no inference or evaluation, and uses no external, OOD, or bridge artifact. It performs no model advancement.
