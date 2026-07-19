# Stage196-B2-B2 Row-Level Paired Treatment-Path Probe

## Purpose and authority

This stage is an artifact-only paired treatment-path probe under frozen Mamba. It compares the precommitted `joint` and `frame_local_only` arms row by row across 20 epochs. Its authority is limited to describing temporal and directional paths associated with the observed gradient-ownership treatment effect. It does not implement an intervention, loss, router, trainer, or training regime and does not claim formal causal mediation, necessity, sufficiency, safety, unfrozen behavior, external validity, or architectural superiority.

## Required invocation

The analyzer requires, without defaults:

```text
--repo-root
--stage196b2b1-analysis-json
--stage196b2b1-analyzer-git-commit
--stage196b2a-analysis-json
--stage196b2p0-run-root
--current-git-commit
--output-dir
```

The frozen roles are distinct: B2-B1 analyzer `85f1de8f9e0393ccdca5da4bc0725d88d8f427c9`; B2-A analyzer `833752ec25890e76ee3c2dc3f3bc3c3c1b7428a6`; B2-P0 runtime `e9aaff24054f1d409119b70df13b94159a34a8e4`; B1 runtime `9835cbbf86d83aca0964821669e63f7f6deb1c59`; FrameGate origin `5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8`; and the new analyzer commit supplied through `--current-git-commit` and required to equal repository HEAD.

## Source closure

The parent of the B2-B1 analysis JSON must contain exactly its eight named outputs. The analyzer requires decision `STAGE196B2B1_SEED_SPECIFIC_NO_STABLE_BIFURCATION`, next stage `STAGE196B2B2_NO_PROMOTION_ROW_LEVEL_CAUSAL_PROBE`, empty blockers, exactly 23 passed contract rows, exactly 16 row profiles, positive seeds 184 and 185, contrast seed 183, counts 5/6 and 2/3, failed Rules A and B across the positive seeds, false local exact cross-seed transfer, and the explicitly non-authorizing composition view.

The parent of the B2-A analysis JSON must contain exactly its nine named outputs. It must report `STAGE196B2A_SEED_SPECIFIC_MIXED_PROPAGATION`, empty blockers, a wholly passed contract, three seed-summary rows, 60 epoch-propagation rows, and 4320 support-transition rows.

Top-level null or absent normalized provenance/margin fields are replaced from passed contract values with a nonblocking schema warning. A populated top-level value must agree. Missing authoritative contract and top-level values, or any non-null disagreement, fails closed. The normalized margin source is exactly `support_logit - not_entitled_logit`, and every normalized provenance role is emitted at the new JSON top level.

## P0 closure and alignment

The P0 root contains exactly six directories: both arms for seeds 183, 184, and 185. Each directory contains exactly the 20 sidecars `stage196b2p0_epoch_channels_001.jsonl` through `_020.jsonl`. Each file has exactly 720 object rows and the exact 18-field P0 schema. Epoch, training seed, and gradient mode must match the directory. IDs, source-row IDs, positions, labels, intervention type, and population remain identical across epochs and between arms; duplicates and missing rows fail closed.

No checkpoint or model is read. The native composition margin is recomputed independently in each arm as support logit minus not-entitled logit. The exact schema has no derived margin field; if the schema is changed, closure fails rather than accepting an unverified derivative.

The primary rows are exactly the 16 B2-B1 profiles. Each primary identity requires `id`, `source_row_id`, `stable_row_id`, `dev_position`, and `seed`; all identity fields and the SUPPORT gold label are cross-checked in both arms at every epoch. Intervention type and all four recurrent-set memberships are preserved from B2-B1. Seed183 is contrast-only and never enters a primary denominator.

## Paired epoch construction

The paired table has exactly 16 × 20 = 320 rows. Every delta is `frame_local_only - joint` for Frame, predicate, sufficiency, polarity, entitlement, and SUPPORT-vs-NOT_ENTITLED margin. Each row reports both predictions, both SUPPORT indicators, SUPPORT disagreement, the sign of each delta, and local-sign agreement with the same-epoch final-margin-delta sign. Raw delta magnitudes are never ranked across channels because channel scales differ.

The only boundaries are Frame/predicate/sufficiency/entitlement probability at 0.5, polarity SUPPORT-facing margin at zero, and composition margin at zero. Paired boundary transitions are `FAIL_TO_PASS`, `PASS_TO_FAIL`, `STABLE_FAIL`, and `STABLE_PASS`. Tolerance `1e-6` is reserved for serialization equality checks and is not a scientific threshold. There is no search, calibration, epsilon sweep, learned boundary, score optimization, classifier, or clustering.

## Tail-three and event definitions

Per-row summaries use exact epochs 18, 19, and 20. They contain both arms' native values, all six mean deltas, transition frequencies, prediction pattern, composition margin, and exact terminal signs (`negative`, `zero`, `positive`) without a magnitude cutoff.

For each delta, the first terminal-sign-stable epoch is the earliest epoch whose sign and every later sign equal the tail-three terminal sign. It is null for terminal zero or if no such epoch exists. For each local/composition boundary, first persistent divergence is the earliest epoch at which the arms differ and continue differing through epoch 20. Event ordering is descriptive only.

## Fixed path taxonomy

Exactly one class is assigned in this order:

1. `FRAME_ENTITLEMENT_GAIN`: positive tail-three Frame, entitlement, and margin deltas; nonnegative polarity delta; Frame and entitlement stabilization do not follow margin stabilization when those epochs exist.
2. `FRAME_ENTITLEMENT_LOSS`: negative tail-three Frame, entitlement, and margin deltas; nonpositive polarity delta; the same event-order condition.
3. `POLARITY_OVERRIDE_DESPITE_FRAME_GAIN`: positive Frame, negative polarity, and nonpositive margin or non-stable intervention SUPPORT; polarity stabilization does not follow margin stabilization when margin stabilization exists.
4. `ENTITLEMENT_OPPOSES_FRAME`: nonzero Frame and entitlement signs oppose.
5. `COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY`: persistent composition divergence exists and no local persistent divergence exists at or before it.
6. `MULTI_CHANNEL_CONFLICT`: at least one positive and one negative nonzero local sign.
7. `NO_STABLE_DIRECTIONAL_PATH`: no earlier class applies.

These are exhaustive descriptive path classes, not formal mediators.

## Role and group audits

Recovery margin-direction concordance means positive tail-three margin delta; harm concordance means negative tail-three margin delta. The analyzer separately reports prediction-pattern concordance, intervention `STABLE_SUPPORT`, and selected-checkpoint agreement with the tail-three role. Discordant rows remain included.

Each positive seed × role group reports row count; path counts/rates; per-channel sign and transition counts; mean and ordered raw tail-three deltas; median sign-stability and boundary-divergence epochs; margin and prediction concordance; intervention types; and recurrent memberships. Seeds remain separate.

The intervention audit preserves `seed × intervention_type × transition_role × path_class` without sparse-type merging. It explicitly reports whether polarity-flip remains harm-only, paraphrase harm follows the polarity-override class, none/paraphrase recovery shares a path, and paths occur in both positive seeds. These descriptions cannot authorize routing.

## Fixed decisions

The decision rules are applied in order:

- `STAGE196B2B2_FRAME_ENTITLEMENT_PATH_DOMINANT` requires, independently in each positive seed, at least 75% recovery gain, at least 75% harm loss, no other harm class above 25%, and relevant Frame/entitlement stabilization no later than final-margin stabilization.
- `STAGE196B2B2_POLARITY_OVERRIDE_HARM_SUBTYPE` requires failure of the first rule, at least 75% recovery gain in each seed, at least two polarity-override harm rows in some positive seed, none in recovery, and polarity-negative stabilization no later than final-margin direction or terminal degradation.
- `STAGE196B2B2_COMPOSITION_WITHOUT_STABLE_LOCAL_PRECURSOR` requires failure of both earlier rules, at least 75% composition-without-local rows in every seed-role group, and no single local directional path covering at least 50% everywhere.
- Otherwise the complete result is `STAGE196B2B2_SEED_SPECIFIC_MULTIPATH_EFFECT`.
- Source, provenance, schema, alignment, count, trajectory, rendering, or output failure produces `STAGE196B2B2_ANALYSIS_INCOMPLETE`.

The respective next stages are monotonicity-constraint design, Frame-gain-with-polarity-preservation design, inference-only composition activation swap, no-promotion inference-only component swap, or input repair. Design outcomes do not authorize implementation or training.

## Output contract

The new output directory is new/empty and is separated from every frozen source tree. It receives exactly nine files: analysis JSON, Markdown report, 16-row path summary, 320-row epoch table, group summary, event-order summary, intervention-type paths, seed183 contrast summary, and contract CSV. Exclusive creation and a post-write filename check prevent overwrite and extra output.

The JSON records decision and next stage, blockers, current and normalized historical provenance, margin source and warnings, seed roles and populations, path counts, all three rule evaluations, intervention and event summaries, authorized/prohibited interpretations, nine-file count, and the required all-false activity flags except `artifact_only_analysis = true`.

The Markdown generator emits exactly the 22 precommitted sections, from Executive decision through Recommended next stage.

## Static-review boundary

Implementation review is static only. No Python execution, `py_compile`, smoke test, analyzer run, checkpoint/model loading, training, Kaggle command, commit, or push is part of this stage implementation. Historical artifacts remain read-only and are neither overwritten nor modified.
