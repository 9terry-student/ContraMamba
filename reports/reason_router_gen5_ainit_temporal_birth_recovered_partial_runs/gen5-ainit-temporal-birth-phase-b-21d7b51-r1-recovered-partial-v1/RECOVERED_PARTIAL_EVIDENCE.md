# Gen5 A-init Temporal Birth Phase B — Recovered Partial Evidence

## Status

`RECOVERED_PARTIAL_VALID_EVIDENCE`

This artifact recovers scientifically usable results from the failed source run
`gen5-ainit-temporal-birth-phase-b-21d7b51-r1` at commit `21d7b514f7b88617cff3b679fb97ecaf6c5411ec`. It does not relabel the failed run as a full
Phase B PASS.

## Recovered evidence

- complete t=20 endpoint worker results from both GPUs;
- all 18 same-training-RNG/different-A orientations;
- complete 21-step geometric trajectory;
- geometric birth step: `1`;
- t=20 fixed functional gate: `PASS`;
- t=20 task-row enrichment: `12.474884398`;
- t=20 full-replay max abs: `1.19209289551e-07`.

## Frozen-authentication discrepancy

The frozen precursor absolute task-row energy authentication did not reproduce:

- observed actual task-row energy: `0.000427012909918`;
- frozen target: `0.000323695863574`;
- observed control task-row energy: `3.42298089742e-05`;
- frozen target: `2.63172556629e-05`.

This discrepancy is retained as data, not corrected post hoc.

## Scientific interpretation boundary

The recovered run supports:

1. geometric A-init-specific raw-write separation is already nonzero at optimizer
   step 1 under the executed Phase B geometry;
2. at t=20 the executed finite visible/complement intervention decomposition
   passes the fixed functional gate;
3. at t=20 the executed local task-row energy remains enriched over the fixed
   signed-permutation controls by more than 5x.

The run does **not** establish the functional-freedom birth step because the
functional scan had not started before the t=20 frozen-authentication stop.
Therefore it does not establish `IMMEDIATE_FIRST_UPDATE_BIRTH` or a full Phase B
PASS.

## Provenance

All source run logs, command bytes, start marker, and endpoint worker logs are
bundled alongside the recovered metrics. Source hashes are recorded in
`recovered_partial_provenance.json`.
