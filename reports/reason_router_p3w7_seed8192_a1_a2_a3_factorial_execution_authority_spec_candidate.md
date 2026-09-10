# P3-W7 Seed8192 A1/A2/A3 Factorial Execution Authority Candidate

## 1. Status, decision, and authority chain

`PASS_READY_FOR_INDEPENDENT_SEED8192_FACTORIAL_EXECUTION_AUTHORITY_REVERIFICATION`

This is the single REPORT-ONLY candidate for the matched Seed8192 A1/A2/A3 factorial. It releases normal execution only conditionally: independent high-risk verification must pass; these exact candidate bytes must be frozen in a dedicated commit; that commit must be pushed; the remote commit/blob must be authenticated; and runtime `HEAD` must equal that eventual exact full frozen commit. Until every condition holds, A1/A2/A3 execution is forbidden.

```text
FACTORIAL_EXECUTION_AUTHORITY_COMMIT = <THIS_CORRECTED_FACTORIAL_AUTHORITY_FREEZE_COMMIT>
RUNTIME_HEAD_MUST_EQUAL_FACTORIAL_EXECUTION_AUTHORITY_COMMIT=TRUE
AUTHORING_HEAD=e1b5db0188d5da0f14fc41cf477a3e9bcb052c14
AT_AUTHORING_TIME = CANDIDATE_ONLY
FACTORIAL_CANDIDATE_STATUS = READY_FOR_INDEPENDENT_REVERIFICATION_AFTER_PREREQUISITE_INTEGRATION
FACTORIAL_EXECUTION_AUTHORIZED_NOW = NO
PRE_FREEZE_FACTORIAL_EXECUTION_AUTHORIZED = NO
POST_FREEZE_FACTORIAL_EXECUTION_AUTHORIZED = CONDITIONAL_ON_ALL_RUNTIME_AND_PROVENANCE_GATES
AA0270A_IS_FACTORIAL_EXECUTION_COMMIT = FALSE
E1B5DB0_IS_FACTORIAL_EXECUTION_COMMIT = FALSE
CALIBRATION_EXECUTION_COMMIT_IS_FACTORIAL_EXECUTION_COMMIT = FALSE
TRAINING_EVALUATION_ALLOWED_DURING_AUTHORING=NO
```

Governing chain, in precedence order, is: the current controller instruction; current HEAD `e1b5db0188d5da0f14fc41cf477a3e9bcb052c14`; the validated calibration-v2 aggregate evidence freeze; the frozen Seed8192 revised-split A0 N=3 clean-replacement execution authority, commit `55debe94f0d19d16a334395e8561901fed6b52fa`, blob `2fe0301a60243504a8e9a4d5d4cd2e8959da80ec`; its validated-evidence analysis freeze, commit `dd183f59f4040405c178da193fe99c7c7f3ef57f`, blob `6ff707c9807fa03aa9be51aba5bfb3d573b71343`; the calibration-v2 execution/provenance chain named by the aggregate freeze (`510d29a9267832dcae521e0c419c2ae0a95c575`, `850b9e38ce64698885e0f24f132a3ab0f20bd42a`, `53144c36ca629294157d37c677e6cceed1f261b7`, `8dbab7c83eb4bee07e98776007586a69e5287fab`, and `21403f5e6cff6ca813c6df127c7ee0295998c597`); the frozen A0-reference prerequisite bindings below; current trainer/contracts; and `AGENTS.md`.

### Frozen A0-reference prerequisite bindings

```text
A0_REFERENCE_IDENTITY_AUTHORITY_COMMIT = aa0270a8d974a81928aa2025047b778ae641d7a2
A0_REFERENCE_IDENTITY_AUTHORITY_BLOB = d65ec5a90b5bedefd8424b58fbf594149722ce02
A0_REFERENCE_RUNTIME_PROVISIONING_AUTHORITY_COMMIT = e1b5db0188d5da0f14fc41cf477a3e9bcb052c14
A0_REFERENCE_RUNTIME_PROVISIONING_AUTHORITY_BLOB = 5c14dfb05c0c0326ade15975966606c63e0f04fd
A0_REFERENCE_PROVENANCE_PREREQUISITE = SATISFIED
A0_REFERENCE_RUNTIME_AVAILABILITY_PREREQUISITE = SATISFIED_BY_CONDITIONAL_PROVISIONING_AUTHORITY
PER_RUN_A0_REFERENCE_PROVISIONING_AUTHORITY = e1b5db0188d5da0f14fc41cf477a3e9bcb052c14
PROVISIONED_A0_REFERENCE_COUNT_PER_RUN = 1
PROVISIONED_A0_REFERENCE_IN_COLLECTOR_MANIFEST = FALSE
SAME_RUNTIME_RUN_TO_COLLECTION = TRUE
```

`aa0270a8d974a81928aa2025047b778ae641d7a2` freezes the exact A0-reference path, bytes, SHA256, and admissibility. `e1b5db0188d5da0f14fc41cf477a3e9bcb052c14` freezes only the runtime provisioning mechanism. Neither earlier authority independently authorizes factorial execution; this final factorial authority must consume both.

The former missing-frozen-A0-reference-provenance-and-clean-runtime-provisioning verifier blocker is `RESOLVED_HISTORICAL`: it is resolved only as a prerequisite integration matter, not as execution authorization or a scientific result.

After this corrected candidate is independently reverified PASS, frozen by a dedicated commit, pushed, and remotely authenticated, that exact dedicated full 40-hex commit resolves `FACTORIAL_EXECUTION_AUTHORITY_COMMIT`. It is neither aa0270a, e1b5db0, nor the calibration execution commit `850b9e38ce64698885e0f24f132a3ab0f20bd42a`. All nine runs use that one exact common execution-authority commit unless a later explicit recovery authority supersedes a failed individual run.

The historical factorial authority at `reports/reason_router_p3w7_a1_a2_a3_factorial_execution_authority_spec_candidate.md` (historical blob `852354511d93f6eedc1272b77c0419e62c0eda82`, history commit `230088191cdb774cf24a4aaf11a4424bc7165513`) is SEMANTIC/TEMPLATE CONTEXT ONLY. Its split174 bindings, old P4-L sidecar, historical A0 references, historical calibration acceptance, namespaces, and `0.6202430063306562` are inadmissible for active Seed8192 execution. `0.6202430063306562` appears in this candidate only as an explicitly historical, forbidden value; it is never an active Seed8192 execution value.

## 2. Calibration conclusion and scientific boundary

The validated aggregate freeze consumes exactly:

```text
COMMON_REASON_LOSS_WEIGHT=0.6273209029272248
```

This is the sole positive common A1/A3 reason-loss weight in this authority. It is common across seeds, not seed-specific. The calibration evidence is measurement/calibration evidence, not model-performance evidence: it establishes no A1/A3 performance, superiority, promotion, causal claim, dev/test/OOD result, or factorial result.

## 3. Exact revised Seed8192 data and implementation contract

| Binding | Exact identity |
|---|---|
| Canonical dataset path | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` |
| Dataset Git blob / canonical content SHA256 / semantic SHA256 | `2b6829bf04a1333446aac6f7c603d9178b339f36` / `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` / `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| Revised P4-L sidecar path | `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl` |
| Sidecar Git blob / physical SHA256 / semantic SHA256 | `83d119e327acacda7cff6b4e24c6502898294e03` / `9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d` / `2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9` |
| Sidecar provenance path / Git blob / physical SHA256 | `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json` / `6c970033fae82286452f6d635b94f441d0f3d048` / `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8` |
| Current trainer blob | `scripts/train_controlled_v6b_minimal.py`: `b7f8a3df3926e34f30ebfdb75f1e16e7b7da0a60` |
| Current contract-test blobs | P2 `64762e27b3a2035e29bbc4d922fac18adaaca97f`; P4-X rebind `5699ec92459aebda711fdb94470430cdf9349fce`; P4-X prelaunch `e9e1d8bedd0cccb6bc0e3deaf43e320ea5480849`; A0 recovery `d8197efa0685ac4ea9892d2111f9df4092fb957b` |

The active split is exactly `split_seed=8192`, `dev_ratio=0.2`, 300 total pairs, 240 train pairs, 60 dev pairs, 2880 train rows, and 720 dev rows. Identity serialization is UTF-8 `value + "\n"` for pair hashes and UTF-8 `row_id<TAB>pair_id<LF>` for ordered row hashes.

| Identity | SHA256 |
|---|---|
| Pair universe / shuffled pairs | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` / `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| Train pairs / dev pairs | `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` / `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` |
| Ordered train rows / ordered dev rows | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` / `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |

## 4. Exact admissible same-seed A0 references

Only these three mutable-worktree evidence members are admissible after byte authentication. Their membership is determined by the frozen A0 N=3 authority/evidence above, not by their directory names or their untracked presence. They have no Git blobs because the A0 roots are intentionally pre-existing untracked import roots; the owning execution-authority freeze is `55debe94f0d19d16a334395e8561901fed6b52fa` and their later evidence disposition is frozen at `dd183f59f4040405c178da193fe99c7c7f3ef57f`.

| Seed | Admitted member and exact prediction path | Bytes / SHA256 | Disposition |
|---:|---|---|---|
| 180 | `replacement_r1`; `reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0/training_report_predictions.jsonl` | `3937018` / `80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334` | primary admitted replacement |
| 181 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed181/A0/training_report_predictions.jsonl` | `3935282` / `d7a2d79091e2b076610d58b6e3539a347c29706b796c9364b6998c5995b42472` | primary admitted member |
| 182 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed182/A0/training_report_predictions.jsonl` | `3938383` / `ae1a1dfd9a844437e032a506716abb68544b16ea1fcdff7eaba14a040a48e650` | primary admitted member |

Seed180 r2, at `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0`, is excluded: `SEED180_R2_STANDARD_CM_WRAPPER_PROVENANCE=INCOMPLETE`; its exact disposition remains `HISTORICAL_PROVENANCE_INCOMPLETE_EVIDENCE_ONLY`. Do not erase, normalize, or reclassify that caveat. Historical split174 A0 predictions are forbidden. The two pre-existing roots named in this section must not be modified, deleted, normalized, staged, or treated as authoritative merely because they exist.

## 5. Frozen arms and exactly nine runs

Primary reasons remain ordered `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`; secondary reasons are diagnostic-only. Final 3-way CE is router-only. Explicit-local retains its detach semantics. The encoder is frozen.

| Arm | Status | Router / ownership | Reason supervision / weight |
|---|---|---|---|
| A0 | completed matched control; DO NOT RERUN | `explicit_product` / `joint` | disabled / `0.0` |
| A1 | new | `conditional_first_blocker` / `joint` | enabled / `0.6273209029272248` |
| A2 | new | `explicit_product` / `explicit_local` | disabled / `0.0` |
| A3 | new | `conditional_first_blocker` / `explicit_local` | enabled / `0.6273209029272248` |

Exactly these nine runs are authorized after the Section 1 release gates: `seed180:A1,A2,A3`; `seed181:A1,A2,A3`; `seed182:A1,A2,A3`. There is no A0 rerun, E0 execution, A4, or additional seed/arm. A failure in any run stops the batch; do not silently continue.

Fresh namespaces are `reports/reason_router_p3w7_seed8192_a1_a2_a3_factorial_runs/seed{180|181|182}/{A1|A2|A3}/`, distinct from historical split174 and completed Seed8192 A0 namespaces. Provenance-level run names are respectively `p3w7-seed8192-factorial-seed<SEED>-<ARM>-authority-<FULL_COMMIT>` and must be unique/non-recycled.

## 6. Exact trainer command templates

Every command uses the common prefix below, with exactly one concrete seed/reference/arm/output binding from the following matrix. The current trainer requires `--reason-router-a0-reference-predictions` for A1/A2/A3 and automatically writes `training_report_predictions.jsonl` next to `training_report.json`; its explicit outputs are report JSON, clean-dev prediction JSON, and selected checkpoint.

```bash
python scripts/train_controlled_v6b_minimal.py --data reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl --architecture v6b_minimal --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --frame-downstream-gradient-mode joint --epochs 20 --max-length 128 --dev-ratio 0.2 --seed <SEED> --split-seed 8192 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --ranking-weight 0.0 --class-weighting none --stage174c-clean-pairwise-mode off --stage174c-clean-pairwise-weight 0.0 --stage174c-clean-polarity-preservation-weight 0.0 --stage175b-support-anchor-mode off --stage175b-support-anchor-weight 0.0 --stage177c-frame-pairwise-mode off --stage177c-frame-pairwise-weight 0.0 --compatible-positive-margin-logit 0.0 --compatible-positive-margin-weight 0.0 --lr 0.001 --controlled-integrity-sidecar-path reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl --expected-integrity-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --save-selected-checkpoint --selected-checkpoint-filename selected_checkpoint.pt --reason-router-a0-reference-predictions <EXACT_SAME_SEED_A0_PATH> --reason-router-arm <ARM> --reason-router-mode <ROUTER> --gradient-ownership-mode <OWNERSHIP> --reason-loss-weight <WEIGHT> --output-json <OUTDIR>/training_report.json --output-predictions-json <OUTDIR>/clean_dev_predictions.json
```

| Runs | `SEED`; exact A0 reference | Arm suffix (`ARM`; `ROUTER`; `OWNERSHIP`; `WEIGHT`) | `OUTDIR` |
|---|---|---|---|
| 180 A1/A2/A3 | `180`; seed180 replacement path in Section 4 | A1; `conditional_first_blocker`; `joint`; `0.6273209029272248` — A2; `explicit_product`; `explicit_local`; `0.0` — A3; `conditional_first_blocker`; `explicit_local`; `0.6273209029272248` | `reports/reason_router_p3w7_seed8192_a1_a2_a3_factorial_runs/seed180/<ARM>` |
| 181 A1/A2/A3 | `181`; seed181 path in Section 4 | A1; `conditional_first_blocker`; `joint`; `0.6273209029272248` — A2; `explicit_product`; `explicit_local`; `0.0` — A3; `conditional_first_blocker`; `explicit_local`; `0.6273209029272248` | `reports/reason_router_p3w7_seed8192_a1_a2_a3_factorial_runs/seed181/<ARM>` |
| 182 A1/A2/A3 | `182`; seed182 path in Section 4 | A1; `conditional_first_blocker`; `joint`; `0.6273209029272248` — A2; `explicit_product`; `explicit_local`; `0.0` — A3; `conditional_first_blocker`; `explicit_local`; `0.6273209029272248` | `reports/reason_router_p3w7_seed8192_a1_a2_a3_factorial_runs/seed182/<ARM>` |

`FACTORIAL_COMMAND_DELTA_REQUIRED = NO`; `COMMAND_CONTRACT_INVARIANCE = PASS`; `MISSING_EXPLICIT_COMMAND_INVARIANCE_ATTESTATION = RESOLVED`. The prerequisite integration at `aa0270a`/`e1b5db0` changes only runtime availability/provenance gating; it does not alter any of the existing nine trainer command templates. No CLI flag, arm setting, seed, split, A0 reference path, reason-loss weight, output namespace, or trainer invocation change is required. The existing commands remain the authoritative commands for the nine runs, subject to the newly frozen per-run provisioning/runtime gates.

Required output/provenance set per run is `training_report.json`, derived `training_report_predictions.jsonl`, `clean_dev_predictions.json`, `run_provenance.json`, and `selected_checkpoint.pt`, plus an immutable wrapper-sidecar recording the A0 source freeze/commit, source path, bytes, SHA256, provisioned path, and post-copy byte/hash verification. No source evidence is overwritten.

## 7. Runtime, collection, and fail-closed gates

For each individual one of the nine runs, apply the frozen provisioning mechanism at `PER_RUN_A0_REFERENCE_PROVISIONING_AUTHORITY` exactly: begin in a fresh or independently isolated checkout whose exact full `HEAD` equals `FACTORIAL_EXECUTION_AUTHORITY_COMMIT`; provision exactly one matrix-matched same-seed A0 reference before `cm run` start-marker creation; validate its exact path, bytes, and SHA256; and add only that exact input path to checkout-local `.git/info/exclude`. Keep the tracked diff, index, and status clean, do not ignore the factorial output namespace, and revalidate the A0 input immediately before the already-frozen exact A1/A2/A3 command. No full protocol is duplicated here; the referenced frozen provisioning authority controls its transport, clean-gate, marker-ordering, and collector details.

Before model loading, verify remote presence/authentication of `FACTORIAL_EXECUTION_AUTHORITY_COMMIT` and its blob; authority/trainer/test blobs above; dataset/sidecar/provenance and all split identities above; command identity/registered command SHA; unique run name; absent output targets; and exact matrix arguments. After exit 0, revalidate the provisioned A0 input bytes/SHA before collection. Use normal authenticated `cm run`, collector, manifest audit, ZIP creation, ZIP download, and `cm import` provenance: authentic `command.sh`, `run.log`, `run.meta`, commit equality, command hash, exit code, artifact manifest path/size/SHA256, and post-import audit must all match. The provisioned A0 input must be absent from the collector manifest.

After a successful GPU run, collector/handoff creation, manifest audit, ZIP creation, and ZIP download must finish in the SAME uninterrupted runtime before GPU/session shutdown or restart. GPU/session shutdown or restart may occur only after ZIP download. Any nonzero exit, input mismatch, collector contamination, missing/extra artifact, failed import, output collision, or nonfinite loss is `STOP`; do not continue, retry, or change attempt identity without a later authority.

Code correctness, execution success, artifact/provenance validity, and scientific conclusion are separate gates. Successful execution does not itself establish a factorial scientific conclusion; later validated evidence and separate interpretation authority are required.

## 8. Non-actions and authoring boundary

This authoring action performs no scientific execution and establishes no factorial result. It performs no implementation, training, evaluation, inference, model/checkpoint loading, Kaggle execution, staging, commit, or push. It changes no scripts, tests, datasets, sidecars/provenance, calibration/A0 artifacts, controller/registry, existing report, Git index, or either pre-existing untracked A0 root.
