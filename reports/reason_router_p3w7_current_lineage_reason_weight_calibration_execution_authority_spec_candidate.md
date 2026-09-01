# P3-W7 Current-Lineage Reason-Weight Calibration Execution Authority Specification Candidate

Status: READY.

This candidate is a static current-lineage factorial-readiness and reason-loss
calibration execution-authority materialization only. It does not authorize a
calibration run while unverified or unfrozen. It does not authorize A1, A2, A3,
training, evaluation, model loading, checkpoint loading, forward passes, CUDA,
commit, or push.

Candidate existence is not execution authorization. Any future calibration run
requires independent verifier PASS, an immutable freeze commit for this exact
authority candidate, and explicit subsequent controller authorization.

## Authority Chain Used

Authority precedence consumed:

- Current research-controller instruction.
- Frozen formal P3-W7-A0 execution authority:
  `2737c3c6116ae3766b469801f990e2c45ba9a55e`,
  `reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`.
- Frozen P3-W7 A0 validated-evidence analysis authority:
  `759124743a9441a4c1811912770c9389fe7432f6`.
- Frozen P3-W7 A0 validated-evidence analysis result:
  `34a5df306a00a28492b08f666bf3f7ac06c26944`,
  `reports/reason_router_p3w7_a0_validated_evidence_analysis_report.md`.
- Historical P3 factorial contract and reason-weight calibration chain as
  historical semantic authority only:
  `72cfdd3d551832e33799ca0399a6d6bf0c431901`,
  `efb18e0660e9c886e2c64c0557282313431383d0`,
  `6df1d013b74cb95c658a3ffddbcff8ed0214cdb3`,
  `1d6e63fb830fd061f6a9962a5bdba70312a34d7f`,
  `76d068cd9bc3b888d101e0cf2b7a3ded82578077`,
  `8a587a6f28a84a01237d81a47898ec4d5597ffc`.
- P3-W5/P3-W6 polarity remediation chain including
  `01d983f8d09cacf0eddefd2014fc81a28771cf5e` and
  `49d7c37cd307893bf8fbc96cd2b6730369fcd8d6`, plus later applicable
  P3-W6-F2/P4 artifacts.
- Current-lineage P4-L contract and latest applicable
  materialization/validation/provisioning artifacts resolved from repository
  history and current tracked artifacts.
- `AGENTS.md`.

Initial repository state for candidate materialization:

- HEAD: `34a5df306a00a28492b08f666bf3f7ac06c26944`
- Branch/status: `## p3w7-current-lineage-calibration-authority`
- Initial tracked worktree: clean.

## Current P4-L Data Contract

The immutable current-lineage P4-L identity is the P4-B R1 regenerated dataset
plus the provisioned and validated P4-L sidecar/provenance lineage. The
historical `f552...` dataset and `5bc03c...` sidecar identities are not current
P4-L identities.

| Field | Current binding |
|---|---|
| Regenerated dataset path | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` |
| Dataset physical SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| Dataset semantic SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| Integrity sidecar path | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl` |
| Sidecar physical SHA256 | `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1` |
| Sidecar semantic SHA256 | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |
| Provenance artifact identity | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json` |
| Provenance physical SHA256 | `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2` |
| Row count | `3600` |
| Split seed | `174` |
| Dev ratio | `0.2` |

Latest applicable lineage anchors:

- P4-L authority commit:
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`.
- P4-L materialization/provisioned canonical artifact freeze:
  `a93c291b79974f4aaa0b51f4578c807e8a5d6301`.
- P4-L provisioning-result validation authority freeze:
  `026216aedb3fa3290dfef65bb81f164580992918`.
- P4-V closure result:
  `reports/reason_router_p2_p3w6f2_p4v_canonical_sidecar_path_provenance_schema_correction_execution_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4v_canonical_sidecar_path_provenance_schema_correction_result.json`.
- P4-X trainer rebind implementation commit:
  `8f6defacc1995f263c97000fe43f6034b1ce9324`.
- P4-Y trainer rebind validation evidence:
  `reports/reason_router_p2_p3w6f2_p4y_trainer_rebind_validation_8f6defacc1995f263c97000fe43f6034b1ce9324/p3w6f2_p4y_trainer_rebind_validation_result_candidate.json`.

Local Windows checkout raw-byte SHA256 values differ because checked-out text
files contain CRLF bytes. LF-normalized local bytes match the frozen current
P4-L physical SHA256 values above. The current authority identity remains the
frozen LF-byte identity, not the platform-converted working-copy raw hash.

## Historical Calibration Disposition

Historical P3W2 resolved:

`reason_loss_weight = 0.6518018402446165`

That scalar is historical evidence only. It is not reusable as the current
P3-W7 A1/A3 execution parameter because the historical calibration cohort used:

- Dataset SHA256:
  `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`.
- Sidecar semantic SHA256:
  `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`.
- Historical aggregate:
  `reports/reason_router_p2_p3w2_calibration_e8124806/calibration_aggregate.json`.

The current P4-L cohort uses:

- Dataset physical SHA256:
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`.
- Dataset semantic SHA256:
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`.
- Sidecar semantic SHA256:
  `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`.

Decision: historical scalar transfer is rejected for current P3-W7 A1/A3
execution. No transfer authority is created by this candidate.

## Static Train Readiness Reconstruction

The deterministic split was reconstructed from the current source records using
the trainer-imported `scripts.build_controlled_v5.split_by_pair_id` semantics:
validate records, sort unique `pair_id`, shuffle with `random.Random(174)`,
take `round(300 * 0.2) = 60` dev pairs, and preserve all pair rows together.

Current split:

- Total rows: `3600`.
- Unique pairs: `300`.
- Dev pairs: `60`.
- Train pairs: `240`.
- Dev rows: `720`.
- Train rows: `2880`.
- Ordered train row identity hash:
  `cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784`.

The static P2 reason-supervision reconstruction used current trainer semantics
from `scripts/train_controlled_v6b_minimal.py`:

- source required fields from `P2_SOURCE_REQUIRED_FIELDS`;
- sidecar required fields from `P2_SIDE_CAR_REQUIRED_FIELDS`;
- stable join by source `id` to sidecar `row_id`;
- split and canonical-row checks;
- generator status normalization over schema, dataset-source, grammar,
  canonical, intervention-contract, polarity-contamination, and time-swap
  status fields;
- first-blocker primary reason order:
  `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`;
- sidecar/source frame binary equality;
- final-label/primary-reason and polarity-label/final-label consistency;
- polarity applicability only when reason-supervision eligible,
  `frame_compatible_label == 1`, `predicate_covered_label == 1`,
  `sufficiency_label == 1`, and final label is `REFUTE` or `SUPPORT`.

Authoritative train split readiness table:

| Quantity | Count |
|---|---:|
| Total train rows | 2880 |
| Reason-supervision eligible rows | 1450 |
| Reason-supervision excluded rows | 1430 |
| Primary reason FRAME | 726 |
| Primary reason PREDICATE | 121 |
| Primary reason SUFFICIENCY | 242 |
| Primary reason AUTHORIZED | 361 |
| Frame-applicable target 0 | 726 |
| Frame-applicable target 1 | 724 |
| Predicate-applicable target 0 | 121 |
| Predicate-applicable target 1 | 603 |
| Sufficiency-applicable target 0 | 242 |
| Sufficiency-applicable target 1 | 361 |
| Polarity-applicable rows | 361 |
| Polarity target REFUTE | 119 |
| Polarity target SUPPORT | 242 |

Train exclusion counts under current trainer semantics:

| Exclusion code | Count |
|---|---:|
| `P2_GENERATOR_STATUS_DEFECT` | 121 |
| `P2_INTEGRITY_SOURCE_REQUIRED` | 1309 |
| `P2_POLARITY_INTERVENTION_CONTRACT_FAIL` | 1071 |

Excluded polarity candidates:

| Candidate definition | Count | Exclusion reason |
|---|---:|---|
| structurally polarity-applicable directional rows with F/P/S all passing, but not trainer polarity-applicable | 359 | 121 with `P2_GENERATOR_STATUS_DEFECT`; 238 with `P2_INTEGRITY_SOURCE_REQUIRED` |

Readiness verdict:

- Both polarity target classes are represented: yes.
- Polarity-local readiness gate: PASS.
- All local binary cohort readiness gate: PASS.
- Current trainer normal A1/A3 polarity-local readiness gate would pass.

This readiness is established from current sidecar-eligible trainer semantics,
not raw final labels alone.

## Calibration Implementation Compatibility

Inspected implementation and tests:

- `scripts/train_controlled_v6b_minimal.py`.
- `tests/test_reason_router_p3w1_calibration.py`.
- `scripts/aggregate_reason_router_p3w1_calibration.py`.

Compatibility verdict: PASS.

The existing no-step calibration mode can operate on the current P4-L
dataset/sidecar without implementation changes, subject to future explicit
execution authorization. Current code already binds the canonical P4-L dataset,
sidecar, provenance, semantic hashes, sidecar physical hash, provenance physical
hash, row count, and P2 reason-supervision counts through the P4-X constants
and validation helpers.

Existing calibration semantics preserved:

- Seeds exactly `180`, `181`, `182`.
- Fresh initialization per seed.
- A3 calibration-only invocation with `conditional_first_blocker` measurement
  arm and `explicit_local` measurement ownership.
- Train split only.
- No dev forward in calibration path.
- No A0 prediction, reference, metric, or checkpoint access.
- No backward.
- No optimizer step.
- No scheduler step.
- Zero parameter updates.
- `reason_loss_weight = 0.0` only as calibration placeholder.
- Complete authoritative train split.
- One logical unit per seed.
- Pooled count-weighted estimator.
- One global positive reason-loss weight shared by A1 and A3.

Observed implementation boundary: the future calibration unit necessarily
constructs the model and performs a no-gradient train-split forward to measure
losses. This candidate did not execute that path.

## Future Calibration Execution Contract

Decision: READY for future current-lineage reason-loss calibration, only after
independent verification, immutable candidate freeze, and explicit subsequent
controller authorization.

Exact execution checkout commit for the code/data state audited here:

`34a5df306a00a28492b08f666bf3f7ac06c26944`

This execution checkout commit is distinct from the later immutable authority
freeze commit that must contain this exact candidate. The later freeze commit is
not predicted here.

Required current-lineage input identities are exactly the current P4-L data
contract listed above. Any dataset, sidecar, provenance, semantic hash,
physical hash, row count, split seed, dev ratio, or ordered train identity
change blocks calibration.

Required seeds:

- `180`
- `181`
- `182`

Output namespace:

`reports/reason_router_p3w7_current_lineage_reason_weight_calibration_execution_34a5df306a00a28492b08f666bf3f7ac06c26944`

Required seed artifacts:

- `reports/reason_router_p3w7_current_lineage_reason_weight_calibration_execution_34a5df306a00a28492b08f666bf3f7ac06c26944/seed180/calibration_unit.json`
- `reports/reason_router_p3w7_current_lineage_reason_weight_calibration_execution_34a5df306a00a28492b08f666bf3f7ac06c26944/seed181/calibration_unit.json`
- `reports/reason_router_p3w7_current_lineage_reason_weight_calibration_execution_34a5df306a00a28492b08f666bf3f7ac06c26944/seed182/calibration_unit.json`

Required deterministic aggregate artifact:

`reports/reason_router_p3w7_current_lineage_reason_weight_calibration_execution_34a5df306a00a28492b08f666bf3f7ac06c26944/calibration_aggregate.json`

Each seed unit must include at minimum the current
`reason_router_p3w1_calibration_unit_v1` required fields, including:

- `schema_version`
- `status`
- `seed`
- `unit_index`
- `unit_scope`
- `ordered_train_row_count`
- `ordered_train_row_identity_hash`
- `model_mode`
- `measurement_arm`
- `measurement_gradient_ownership`
- `reason_loss_weight_placeholder`
- `calibration_gate_scope`
- `primary_reason_min_train_count`
- `primary_reason_class_counts`
- `primary_reason_min_count_gate_pass`
- `local_binary_cohort_counts`
- `local_binary_training_readiness`
- `all_local_binary_cohorts_training_ready`
- `polarity_local_training_ready`
- `weight_resolution_measurement_valid`
- `normal_a1_a3_training_ready`
- `training_readiness_separate_from_weight_resolution`
- `fresh_initialization`
- `checkpoint_loaded`
- `gradient_tracking_enabled`
- `before_backward`
- `before_optimizer_step`
- `before_scheduler_step`
- `parameter_update_count`
- `optimizer_step_executed`
- `scheduler_step_executed`
- `dev_forward_executed`
- `calibration_data_scope`
- `train_reason_supervision_built`
- `dev_reason_supervision_built`
- `dev_inputs_accessed_for_calibration`
- `dev_labels_used_for_calibration`
- `dev_counts_used_for_gate`
- `dev_metrics_used_for_calibration`
- `a0_reference_predictions_required`
- `a0_reference_predictions_accessed`
- `a0_predictions_used_for_calibration`
- `a0_logits_used_for_calibration`
- `a0_metrics_used_for_calibration`
- `a0_checkpoint_used_for_calibration`
- `external_eval_executed`
- `normal_training_report_written`
- `causal_checkpoint_written`
- `final_loss_mean`
- `final_applicable_count`
- `final_loss_sum_reconstructed`
- `final_loss_finite`
- `reason_loss_mean`
- `reason_eligible_count`
- `reason_loss_sum_reconstructed`
- `reason_loss_finite`
- `dataset_path`
- `dataset_sha256`
- `sidecar_path`
- `sidecar_semantic_sha256`
- `expected_sidecar_semantic_sha256`
- `sidecar_semantic_sha256_verified`
- `split_seed`
- `dev_ratio`
- `execution_commit`
- `declared_execution_commit`
- `execution_commit_verified`
- `decision`

The aggregate must include at minimum the current
`reason_router_p3w1_calibration_aggregate_v1` required fields, preserve the
same current P4-L dataset/sidecar identities, verify all three seeds, verify
consistent ordered train identity, and report the resolved common A1/A3 weight.

Exact pooled estimator:

```text
mu_final =
  sum_s n_final_s * loss_final_s / sum_s n_final_s
mu_reason =
  sum_s n_reason_s * loss_reason_s / sum_s n_reason_s
resolved_reason_loss_weight =
  mu_final / mu_reason
```

Required gates:

- all three seed units present;
- exactly one unit per seed;
- every unit status PASS;
- execution commit equals
  `34a5df306a00a28492b08f666bf3f7ac06c26944`;
- dataset SHA256 equals
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`;
- sidecar semantic SHA256 equals
  `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`;
- split seed exactly `174`;
- dev ratio exactly `0.2`;
- ordered train row count exactly `2880`;
- ordered train row identity hash exactly
  `cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784`;
- every `final_loss_mean`, `reason_loss_mean`, reconstructed sum, `mu_final`,
  `mu_reason`, and `resolved_reason_loss_weight` finite and positive;
- `final_applicable_count > 0`;
- `reason_eligible_count > 0`;
- `resolved_reason_loss_weight > 0`;
- `normal_a1_a3_training_ready = true`;
- `polarity_local_training_ready = true`.

Forbidden estimator choices:

- no seed-specific weights;
- no simple average of per-seed ratios;
- no dev-selected weight;
- no A0-performance-selected weight;
- no A0 reference, prediction, metric, checkpoint, or selected-epoch input;
- no external-label calibration or threshold selection.

Exact command templates are intentionally not released as runnable commands by
this candidate. They may be materialized only in a subsequent controller-
authorized execution handoff after independent verification and authority
freeze, because this candidate must not be mistaken for immediate execution
authorization.

## Factorial Release Status

Factorial contract preserved:

| Arm | Router | Ownership | Reason weight | Status |
|---|---|---|---|---|
| A0 | `explicit_product` | `joint` | `0` | Not changed by this candidate |
| A1 | `conditional_first_blocker` | `joint` | positive common reason weight | Not released until current-lineage calibration evidence resolves the weight |
| A2 | `explicit_product` | `explicit_local` | `0` | Blocked until the matched A1-A3 factorial execution contract is released |
| A3 | `conditional_first_blocker` | `explicit_local` | same positive common reason weight as A1 | Not released until current-lineage calibration evidence resolves the weight |

A1/A3 are not released by this candidate. A2 is not released alone. No A1, A2,
or A3 execution is authorized.

## Evidence-Level Separation

Calibration code correctness, calibration execution success, calibration
artifact/provenance validity, resolved hyperparameter authority, and later
factorial scientific conclusion remain separate states.

A future successful calibration run would be hyperparameter-resolution evidence
only. It would not be scientific evidence for the P2 mechanism or any A1/A2/A3
factorial conclusion.

## Validation Summary

Validation performed for this candidate:

- HEAD and initial clean status inspected.
- Frozen authority/result commits inspected.
- Current P4-L identities resolved from latest applicable artifacts and
  repository history.
- Current dataset and sidecar semantic hashes recomputed.
- Local raw and LF-normalized physical hashes inspected; LF-normalized hashes
  match frozen P4-L physical identities.
- Deterministic train split reconstructed with seed `174` and dev ratio `0.2`.
- Current trainer P2 sidecar eligibility/exclusion behavior reproduced
  statically.
- Current train polarity class/readiness counts computed.
- Existing calibration-mode implementation and tests audited for compatibility.
- Historical-vs-current calibration identity comparison performed.

No model load, checkpoint load, model forward, backward, optimizer step,
scheduler step, CUDA execution, training, evaluation, prediction generation,
commit, or push occurred.

Final candidate file SHA256 and byte size are computed after materialization and
reported in the completion note rather than embedded self-referentially here.

Final readiness token:

`P3W7_CURRENT_LINEAGE_REASON_WEIGHT_CALIBRATION_EXECUTION_AUTHORITY_CANDIDATE_READY`
