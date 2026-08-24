# P3-W6-F2-P4-X Trainer Rebind Authority Specification Candidate

Authority/version:

`P3W6F2P4X_TRAINER_REBIND_AUTHORITY_SPEC_CANDIDATE_V1`

Stage ID:

`P3-W6-F2-P4-X`

Candidate path:

`reports/reason_router_p2_p3w6f2_p4x_trainer_rebind_authority_spec_candidate.md`

## 1. Authority Basis And Phase

Creation authority:

- Current controller instruction.
- Active predecessor:
  `69c7d7b142171c8a3b21c0984b2b3162da04fe77`.
- P4 lineage closure evidence freeze:
  `69c7d7b142171c8a3b21c0984b2b3162da04fe77`.
- Frozen P4-V closure result:
  `reports/reason_router_p2_p3w6f2_p4v_canonical_sidecar_path_provenance_schema_correction_execution_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4v_canonical_sidecar_path_provenance_schema_correction_result.json`.
- P4-V closure result SHA256:
  `d36960dfcea59dd53c9e9327ada06229c4930fb6da6cd448d9738c3072f11a47`.
- P4-V authority freeze:
  `3e8fa6269a0728d615e03240ab4cd8f15418c178`.
- P4-W authority freeze:
  `20965bece9c72f182f112c3a608c1fa2f2dce42b`.
- Frozen P4-L authority:
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`.
- P4-L implementation anchor:
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- Applicable frozen P2/P3 specifications, manifests, reports, and tests in
  this repository.
- `AGENTS.md`.
- Repository naming/provenance conventions.

Phase:

`TRAINER REBIND AUTHORITY / SPECIFICATION ONLY`

This candidate defines a future bounded implementation required to bind
training inputs to the validated canonical P4-L integrity sidecar. It does not
implement that rebind.

Local read-only authority search found earlier lineage-rebind material,
including P4-H, but no frozen authority applicable to the post-P4-V/P4-W
validated canonical P4-L trainer binding requested here. Search also found no
`P4-X`/`p4x` namespace collision. Therefore `P3-W6-F2-P4-X` is the next unused,
unambiguous P4 lineage stage.

## 2. Current Trainer Surface Audit

Future implementation scope is trainer-local unless the verifier finds a
specific test helper requires a small test-only fixture helper.

Production entry point:

- `scripts/train_controlled_v6b_minimal.py::main`
- `scripts/train_controlled_v6b_minimal.py::build_parser`

Current hard-bound identity constants to rebind:

- `scripts/train_controlled_v6b_minimal.py::_STAGE187_AUTHORITATIVE_DATA`
- `scripts/train_controlled_v6b_minimal.py::_STAGE187_AUTHORITATIVE_SIDECAR`
- `scripts/train_controlled_v6b_minimal.py::_STAGE187_DATASET_SHA256`
- `scripts/train_controlled_v6b_minimal.py::_STAGE187_SIDECAR_SEMANTIC_SHA256`
- `scripts/train_controlled_v6b_minimal.py::_STAGE187_EXPECTED_SIDECAR_ROWS`

Dataset and data-loader construction:

- source load: `scripts/train_controlled_v6b_minimal.py::main`,
  `records = v5.load_jsonl(args.data)`.
- split: `scripts/train_controlled_v5.py::split_by_pair_id`, called from
  `main` with `dev_ratio=args.dev_ratio` and `seed=resolved_split_seed`.
- encoding/data-loader surface:
  `scripts/train_controlled_v5.py::encode_records`,
  `scripts/train_controlled_v5.py::encode_mamba_records`, and
  `scripts/train_controlled_v5.py::move_inputs`, called from `main`.
- batch/sample surface: `scripts/train_controlled_v5.py::sample_indices`,
  called from the training epoch loop in `main`.

P2/P4 sidecar loader and join surface:

- `scripts/train_controlled_v6b_minimal.py::_stage187_file_sha256`
- `scripts/train_controlled_v6b_minimal.py::_stage187_semantic_sidecar_sha256`
- `scripts/train_controlled_v6b_minimal.py::_stage187_load_integrity_sidecar`
  for compatible-positive-margin sidecar access.
- `scripts/train_controlled_v6b_minimal.py::_p2_load_reason_integrity_sidecar`
  for reason-router metadata sidecar access.
- `scripts/train_controlled_v6b_minimal.py::_p2_resolve_canonical_lineage_for_split`
  for per-split canonical lineage.
- `scripts/train_controlled_v6b_minimal.py::_p2_prepare_reason_supervision`
- `scripts/train_controlled_v6b_minimal.py::_p2_prepare_reason_supervision_train_only`

Batch schema affected by sidecar admission:

- source record fields required by `P2_SOURCE_REQUIRED_FIELDS`;
- sidecar fields required by `P2_SIDE_CAR_REQUIRED_FIELDS`;
- tensor fields attached to `train_inputs`/`dev_inputs`:
  `p2_primary_reason_targets_4`,
  `p2_secondary_reason_targets_3`,
  `p2_reason_supervision_eligible`,
  `p2_frame_applicability_mask`,
  `p2_predicate_applicability_mask`,
  `p2_sufficiency_applicability_mask`,
  `p2_polarity_applicability_mask`,
  `p2_polarity_targets_2`.

Reason-router loss surface:

- `scripts/train_controlled_v6b_minimal.py::_p2_reason_router_losses`
- `scripts/train_controlled_v6b_minimal.py::_p2_masked_bce_loss`
- `scripts/train_controlled_v6b_minimal.py::_p2_reason_arm_loss_export`
- `scripts/train_controlled_v6b_minimal.py::_p2_product_arm_loss_export`
- `scripts/train_controlled_v6b_minimal.py::_p2_record_epoch_loss_snapshot`

Model/router/gradient ownership surface:

- `src/contramamba/modeling_v6b_minimal.py::ContraMambaV6BMinimal.forward`
- `src/contramamba/heads/entitlement_decision.py::FinalEntitlementDecisionHead.forward`
- `scripts/train_controlled_v6b_minimal.py::_p2_resolve_arm_contract`
- `scripts/train_controlled_v6b_minimal.py::main`, where
  `model.gradient_ownership_mode` and `model.return_q_diagnostics` are set.
- optimizer construction remains `scripts/train_controlled_v5.py::build_optimizer`,
  called from `main`.

Configuration/CLI surface:

- `--data`
- `--reason-router-arm`
- `--reason-router-mode`
- `--gradient-ownership-mode`
- `--reason-loss-weight`
- `--controlled-integrity-sidecar-path`
- `--expected-integrity-sidecar-semantic-sha256`
- `--compatible-positive-margin-weight`
- `--compatible-positive-margin-logit`
- `--split-seed`
- `--dev-ratio`
- checkpoint metadata emitted through
  `scripts/train_controlled_v6b_minimal.py::_p2_checkpoint_metadata_from_args`
  and report metadata in `main`.

Relevant tests:

- `tests/test_reason_router_p2_contract.py`
- `tests/test_reason_router_p3w1_calibration.py`
- `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py`
- `tests/test_reason_router_p3w6f2_p4b_r1_stage185_compatibility.py`
- `tests/test_reason_router_p3w6f2_p4b_r1_regeneration.py`

## 3. Canonical P4-L Binding Contract

P4 canonical artifact/provenance integrity is ESTABLISHED by the frozen P4-V
closure result. This candidate does not re-execute P4-V and does not require
the canonical sidecar bytes to be tracked in this local checkout. Future
trainer execution must require the canonical files at runtime and fail closed
before any training step if they are missing, malformed, moved, symlinked,
mutated, or identity-mismatched.

Canonical directory identity:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458`

Canonical files:

- `p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
- `p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`

Required identities:

- sidecar physical SHA256:
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- provenance physical SHA256:
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`
- sidecar semantic SHA256:
  `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`
- source dataset physical SHA256:
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- source dataset semantic SHA256:
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- row count: `3600`

Runtime path resolution:

- Future trainer code must resolve `--data` and
  `--controlled-integrity-sidecar-path` against repository root when relative,
  matching the current style.
- The resolved data path must equal the canonical P4-B/P4-L source dataset path
  recorded in the canonical provenance manifest.
- The resolved sidecar path must equal the canonical sidecar file path above.
- The resolved provenance path must be inferred as the sibling canonical
  provenance file unless a future frozen authority explicitly adds a separate
  CLI. This candidate does not require a new CLI flag.
- The canonical directory and both files must be normal filesystem objects, not
  symlinks. Any missing path or symlink is a fail-closed pre-training error.

Provenance/schema/version verification:

- The provenance JSON must parse as one JSON object.
- `schema_version` must equal
  `P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR_PROVENANCE_V1`.
- `sidecar_schema_version` must equal
  `P3W6F2P4L_CURRENT_LINEAGE_EFFECTIVE_INTEGRITY_SIDECAR_V1`.
- `p4l_authority_commit` must equal
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`.
- `builder_source_commit` must equal
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- `row_count` must equal `3600`.
- `training_admission_released`, `implementation_authorized`,
  `artifact_materialization_authorized_by_p4l`, `a0_execution_authorized`,
  `training_authorized`, `evaluation_authorized`, `kaggle_authorized`, and
  `gpu_authorized` must not be interpreted as trainer/training authority. If
  any field is absent or not a JSON boolean, fail closed. If any field claims a
  broader authority than the currently frozen workflow grants, fail closed.

Physical/semantic identity verification:

- Compute physical SHA256 over the exact source dataset bytes, sidecar bytes,
  and provenance bytes.
- Compute sidecar semantic SHA256 with the established
  `_stage187_semantic_sidecar_sha256` semantics: parse JSONL into an ordered
  list, remove only `created_at` per row, JSON serialize with sorted keys,
  compact separators, `ensure_ascii=false`, and hash UTF-8 bytes.
- Compute source dataset semantic SHA256 according to the canonical provenance
  contract. If current trainer code lacks a helper for this, future
  implementation must add a bounded helper rather than comparing physical SHA
  only.
- All required identities above must match before model construction,
  optimizer construction, checkpoint load, calibration export, backward,
  optimizer step, report write, or training-loop entry.

Stable join-key contract:

- The stable join key is source row `id` joined to sidecar row `row_id`.
- This key is derived from P4-L, the P4-L builder, and existing P2 trainer
  tests. P4-L requires the sidecar to use source row `id` as `row_id`.
- The join is exact one-to-one over all 3600 source rows and 3600 sidecar rows.
- Row-position/order equality is required only as defense in depth and must not
  be the only coupling. A future loader must first validate unique non-empty
  keys and exact source/sidecar key-set equality, then validate source-order
  equality as an additional check.

Duplicate-key handling:

- Duplicate source `id` fails closed.
- Duplicate sidecar `row_id` fails closed.
- Duplicate provenance-path identity, if represented through inconsistent
  sidecar/provenance paths, fails closed.

Missing/unmatched handling:

- Missing source `id` or sidecar `row_id` fails closed.
- Missing sidecar row for any trainer source row fails closed.
- Sidecar row with no matching trainer source row fails closed.
- Trainer row not in the canonical 3600-row source universe fails closed for
  P2/P4-L-bound reason-router training. Bridge/external rows must not be
  silently admitted to P2 reason supervision; existing bridge handling remains
  ineligible for reason-specific sidecar supervision.
- Sidecar row outside the canonical universe fails closed.

Malformed or missing required fields:

- Missing `P2_SIDE_CAR_REQUIRED_FIELDS` or P4-L required schema fields fails
  closed before any training step.
- Missing `P2_SOURCE_REQUIRED_FIELDS` fails closed before any training step for
  P2-enabled arms.
- Exact binary fields must be JSON integers `0` or `1`, not booleans.
- `reason_codes` must be a JSON array, sorted and unique.
- `eligible_for_positive_margin` and `p2_reason_supervision_eligible` must be
  JSON booleans if consumed.
- Unsupported `integrity_status` fails closed.

## 4. P2 Semantic Invariants

Future implementation must preserve without reinterpretation:

- primary first-blocker order:
  `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`;
- secondary reasons remain multi-label diagnostic only;
- secondary reasons are not duplicated into external class targets or loss;
- final 3-way CE is router-only;
- F/P/S inputs on the final 3-way CE path are detached for explicit-local/A3;
- polarity input on the final 3-way CE path is detached for explicit-local/A3;
- EMA is observer/baseline only, never teacher, loss target, or novelty
  mechanism;
- A0-A3 ablation semantics are preserved;
- E0 algebraic-equivalence check semantics are preserved.

No row, field, diagnostic, or count from P4-L may be used to change final
external class labels or to weaken the distinction between label correctness
and evidence entitlement.

## 5. Supervision Admission Contract

Admission semantics are inherited from frozen P4-L/P2 authority:

- `p2_reason_supervision_eligible` controls admission to reason-specific
  supervision.
- `integrity_status` is not itself a training target.
- `eligible_for_positive_margin` controls only the compatible-positive-margin
  surface if consumed by that trainer path; it is distinct from P2
  reason-supervision eligibility.

Known counts to preserve:

- `p2_reason_supervision_eligible`: true `1769`, false `1831`.
- `integrity_status`: `ELIGIBLE = 1769`, `INELIGIBLE = 1562`,
  `UNRESOLVED = 269`.
- `eligible_for_positive_margin`: true `724`, false `2876`, if consumed by the
  trainer.

Required reconciliations:

- `1769 + 1562 + 269 = 3600`
- `1769 + 1831 = 3600`
- `724 + 2876 = 3600`

Fail-closed handling:

- Ineligible rows must not enter reason-specific losses.
- Unresolved rows must not enter reason-specific losses.
- Malformed rows must fail the loader rather than become eligible.
- Rows lacking an authorized supervision contract must be masked out or fail
  closed according to the affected field. They must not be silently treated as
  clean.
- If any count sanity check fails, future trainer code must stop before any
  model/backward/optimizer activity.

## 6. Explicit Gradient Ownership Contract

Existing authority settles the material ownership semantics for P2 A0-A3:

- A0: explicit product/router baseline, reason loss inactive.
- A1: conditional first-blocker router with joint ownership.
- A2: explicit product with explicit-local ownership.
- A3: conditional first-blocker router with explicit-local ownership.

Loss ownership:

- final 3-way CE owns the final router/composer path only.
- frame BCE owns `FrameGate`.
- predicate conditional BCE owns `PredicateCoverageHead`.
- sufficiency conditional BCE owns `SufficiencyGate`.
- authorized polarity CE owns `PolarityEnergyHead`.
- primary reason CE owns `FinalEntitlementDecisionHead.reason_bias_3` and, in
  A1 joint ownership only, may backpropagate through F/P/S local paths as
  established by existing tests.
- EMA/teacher observer owns no training loss and introduces no gradient path.
- Diagnostic exports own no trainable component.

Detach obligations:

- In A3 explicit-local mode, final 3-way CE must not send gradients into
  FrameGate, PredicateCoverageHead, SufficiencyGate, or PolarityEnergyHead.
- In A3 explicit-local mode, primary reason CE must not send gradients into
  FrameGate, PredicateCoverageHead, SufficiencyGate, or PolarityEnergyHead.
- In A3 explicit-local mode, local losses still update their respective local
  heads through raw owner tensors.
- In A1 joint mode, the existing joint path is preserved; this candidate does
  not authorize a new detach or a new shared/backbone ownership rule.
- No gradient from final 3-way router CE may enter detached F/P/S or polarity
  paths.
- Reason-specific losses cannot consume unauthorized, ineligible, unresolved,
  malformed, bridge/external, or sidecar-missing rows.
- Secondary reason diagnostics must not create a loss.
- EMA diagnostics must not add teacher loss, target loss, consistency loss, or
  novelty loss.

Shared/backbone parameters:

- Existing P2 tests assert no backbone gradient in the tested dummy/frozen
  setup. This candidate does not create new shared/backbone scientific
  semantics. If future implementation needs to alter optimizer parameter
  groups or shared/backbone trainability, P4-X is BLOCKED and a new authority
  is required.
- Future implementation must preserve the current optimizer construction
  surface unless a verifier proves parameter-group metadata must be updated
  solely to keep existing trainable parameter ownership checks honest.

## 7. Future Validation Contract

After implementation, validation must be bounded and non-training first. The
minimal future validation set is:

- canonical sidecar physical SHA mismatch test;
- canonical provenance physical SHA mismatch test;
- source dataset physical SHA mismatch test;
- sidecar semantic SHA mismatch test;
- source dataset semantic SHA mismatch test;
- missing sidecar artifact test;
- missing provenance artifact test;
- wrong canonical path test;
- symlink path rejection test if the platform supports it;
- provenance schema/version mismatch test;
- sidecar schema/version mismatch test;
- stable key join success test over source `id` and sidecar `row_id`;
- duplicate source `id` test;
- duplicate sidecar `row_id` test;
- source row missing sidecar key test;
- sidecar row unmatched by source key test;
- source-order defense-in-depth mismatch test;
- malformed/missing required source field tests;
- malformed/missing required sidecar field tests;
- malformed boolean/integer exact-binary tests;
- `reason_codes` sorted-unique test;
- admission-mask tests for eligible, ineligible, unresolved, and malformed
  rows;
- count sanity tests for `1769/1831`, `1769/1562/269`, and `724/2876`;
- reason-specific supervision routing tests;
- final 3-way CE router-only tests;
- detach/gradient ownership tests for A1 and A3;
- zero-gradient assertions for prohibited final CE paths under explicit-local;
- zero-gradient assertions for prohibited primary reason CE paths under
  explicit-local;
- positive/nonzero gradient assertions only for established local owners and
  established router parameters;
- secondary-reason non-loss diagnostic tests;
- EMA observer-only tests where teacher observer is enabled;
- preservation tests for A0-A3 CLI resolution;
- preservation tests for E0 algebraic-equivalence semantics;
- checkpoint/report metadata identity tests for new P4-L hashes and paths.

These validations are future implementation validations only. This candidate
does not authorize trainer execution, training, evaluation, A0, GPU use, or
Kaggle execution.

## 8. Planned Implementation Delta

Production code expected to change in the future bounded implementation:

- `scripts/train_controlled_v6b_minimal.py`

Expected production-code edits:

- rebind authoritative dataset/sidecar/provenance constants or equivalent
  canonical identity table from Stage187 historical values to canonical P4-L
  values;
- add bounded canonical provenance loading/verification if not already present;
- add source dataset semantic SHA verification if not already present;
- update sidecar loader/report/checkpoint metadata names to distinguish
  canonical P4-L from historical Stage185;
- preserve current CLI names unless a future verifier proves a new flag is
  required;
- preserve current training/loss/model behavior other than fail-closed
  identity/admission binding.

Tests expected to change in the future bounded implementation:

- `tests/test_reason_router_p2_contract.py`
- optionally a new narrowly named test file under `tests/` if keeping P4-L
  canonical identity fixtures separate is cleaner.

Config/schema changes:

- No standalone config file change is currently required.
- No dataset, sidecar, provenance artifact, README, historical report, or
  canonical P4-L artifact change is authorized.

## 9. Authority Boundary

P4 canonical artifact/provenance integrity = ESTABLISHED.

This stage is trainer-rebind specification only.

This candidate alone does NOT authorize implementation until independently
verified and frozen under repository workflow.

This candidate does NOT authorize:

- trainer execution;
- A0;
- training;
- evaluation;
- GPU execution;
- Kaggle execution;
- sidecar rebuilding;
- canonical P4-L artifact mutation;
- P4 closure result mutation;
- dataset mutation;
- README rewrite;
- scientific conclusion.

README rewrite remains after trainer-rebind implementation and validation.

Research-credit A0 authority remains after README rewrite.

## 10. Stop Conditions For Future Implementation

Future implementation must stop rather than improvise if:

- canonical sidecar/provenance files are absent at runtime;
- any required hash mismatches;
- provenance schema/version is absent or mismatched;
- source semantic hash computation cannot be reproduced;
- source `id` to sidecar `row_id` cannot be validated as an exact one-to-one
  stable join;
- row-position-only coupling would be required;
- required supervision semantics are not established by frozen authority;
- material gradient ownership semantics would need to change;
- implementation would require model/head redesign;
- implementation would require dataset, sidecar, provenance, historical report,
  README, split, seed, label, or promotion-criteria mutation;
- training/evaluation would be required to validate the patch.

## 11. Candidate Self-Check

Trainer surface: PASS.

Canonical binding contract: PASS.

Join key contract: PASS.

Gradient ownership contract: PASS.

Mask admission contract: PASS.

Provenance binding: PASS.

Validation plan: PASS.

Planned implementation delta: production `scripts/train_controlled_v6b_minimal.py`;
tests `tests/test_reason_router_p2_contract.py` and optionally one new
focused test file; no config/schema file change currently required.

Authority boundary: PASS.

Authorized delta for this task: exactly this one untracked Markdown candidate.

Training/evaluation run by this task: NO.
