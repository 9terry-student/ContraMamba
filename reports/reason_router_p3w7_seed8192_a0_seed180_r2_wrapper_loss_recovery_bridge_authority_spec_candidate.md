# P3-W7 Seed8192 A0 Seed180 R2 Wrapper-Loss Recovery Bridge Authority Specification Candidate

## Verdict, phase, and narrow purpose

`PASS_READY_FOR_INDEPENDENT_SEED8192_A0_SEED180_R2_WRAPPER_LOSS_RECOVERY_BRIDGE_AUTHORITY_VERIFICATION`

This is a REPORT_ONLY_SEED8192_A0_SEED180_R2_WRAPPER_LOSS_RECOVERY_BRIDGE_AUTHORITY_AUTHORING candidate.  It defines a truthful bounded bridge for the already-completed Seed8192 A0 seed180 r2 artifact set after loss of the authentic standard-cm wrapper files.  It does not make the r2 artifacts fully provenance-valid or scientifically admissible.  It does not authorize an execution merely by existing.

The candidate's required authoring HEAD is `8c515ec3f94fe95df117d9b9dd39626396447ed4`.  Its authority basis was inspected directly from these frozen commits:

| Role | Commit | Frozen report / disposition |
| --- | --- | --- |
| Seed8192 execution authority | `abd85a088c274678004432160625d42208112848` | `reports/reason_router_p3w7_seed8192_revised_split_a0_execution_authority_spec_candidate.md` |
| Recovery-handoff authority | `48d258fc5e8b09e163e9252f33c86936244ef872` | Seed8192 A0 seed180 r2 recovery handoff authority |
| Local-preflight correction | `7e8b909e57c0e716f5154cc1d3083c06ef8f2d5a` | recovery preflight boundary correction |
| Recovery-command freeze | `bdb8202331f20dfb850924912133c3dd8871feb2` | recovery commands freeze |
| Stage-L PowerShell successor | `a8f7a968ce7493002ef04061cccf70e4332f5768` | Stage-L PowerShell successor |
| Stage-L scalar-output correction | `8c515ec3f94fe95df117d9b9dd39626396447ed4` | current frozen scalar-output correction |

Historical precedent `233ed0be080e1d30dd47de2e66136475ec2ede76` was inspected directly only for recovery-authority structure.  Historical reference `b32d73dfa49b6b9dfabf3093802904323cf679cd` is context only.

## R2 disposition and wrapper-loss finding

| Classification | Required status |
| --- | --- |
| Original run | `p3w7-seed8192-a0-seed180-r2` |
| Execution commit | `abd85a088c274678004432160625d42208112848` |
| Execution success | `ESTABLISHED` |
| Training/evaluation under this authority | `NOT AUTHORIZED; MUST NOT BE REPEATED` |
| Scientific conclusion | `NONE` |
| Artifact provenance validity | not fully established; recovery import/audit remains required |
| Standard cm-wrapper provenance | `INCOMPLETE / LOST` |

Forensic search in the current Kaggle evidence state found `FOUND=0` for each authentic original `command.sh`, `run.log`, and `run.meta`.  Therefore:

```text
STANDARD_CM_WRAPPER_PROVENANCE = INCOMPLETE / LOST
WRAPPER_LOSS_STATUS = AUTHENTIC_ORIGINAL_WRAPPERS_NOT_FOUND
```

The five scientific artifacts remain exact frozen identities; wrapper loss is neither training failure nor scientific failure.  It must not be reinterpreted as either.  The consumed r2 attempt must not be overwritten, retried, resumed, or rerun under the same attempt identity.

Historical frozen wrapper records may be cited only as historical records:

| Historical record | Value |
| --- | --- |
| Registered execution-command SHA256 | `82be0c377e305228609e8ce9f75a6b3e8b8a6f83f9999a1543914f9fc517c1f4` |
| `run.log` SHA256 | `a0d4b015cc77e8060184a5035333fe46e101f52d931af3101950382e24407f4e` |
| `run.meta` SHA256 | `23989109951bdf2b3fdfc17f9a7739c34e2875131bc76c0ccb1e313858fcdc27` |
| Wrapper `STARTED_UTC` | `2026-09-08T22:38:29Z` |
| Wrapper `FINISHED_UTC` | `2026-09-08T22:41:44Z` |
| Wrapper `EXIT_CODE` | `0` |

These values MUST NOT be used to synthesize, backfill, edit, backdate, or represent replacement historical wrapper bytes.  In particular, no synthetic `command.sh`, `run.log`, `run.meta`, or `start.marker` is permitted.

## Immutable surviving artifacts

The source namespace is:

```text
reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0
```

Only these five regular-file artifacts are surviving scientific evidence for this bridge.  A future collector or auditor must verify each exact source path, byte count, and SHA256 before copying, parsing, packaging, or relying on it.

| Relative source path | Bytes | SHA256 |
| --- | ---: | --- |
| `training_report.json` | 306244 | `146b7330f6cf479bd339b2eb0af886d5eda3eed72589f57c0879bb5f6d9f5d9c` |
| `training_report_predictions.jsonl` | 3937018 | `80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334` |
| `clean_dev_predictions.json` | 4840320 | `5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d` |
| `run_provenance.json` | 69120 | `057237823c0b56a907b051b1d4018eb9317aacd860f986083aaa2277307ffbbf` |
| `selected_checkpoint.pt` | 518269879 | `0724f5a2e537c932f6692dd74713d57fc70f8182ee2719c33665b09114bd944a` |

The checkpoint is an identified immutable artifact only.  This authority does not authorize checkpoint loading, deserialization, inference, model forward passes, metric recomputation, or any other use of it.

## Frozen Stage-K disposition

The existing frozen Stage-K payload is immutable historical evidence with:

```text
BYTES = 11386
SHA256 = e4e7f9e8a15082b7b0a93faefaf5d60245674a65360432c204f2131b498a309c
FROZEN_V3_STAGE_K = PRESERVED_BUT_INADMISSIBLE_FOR_CURRENT_WRAPPER_LOSS_EVIDENCE_STATE
```

It MUST NOT be modified, superseded in its historical bytes, or executed.  Its v3 design depended on authentic original wrappers; current wrapper loss makes that frozen path inadmissible.  The correct response is a distinct recovery-only mechanism, not relaxation of standard-v3 validation.

## Mandatory independent trainer-level provenance binding

A future recovery collector/auditor must first verify `run_provenance.json` against its exact table hash `057237823c0b56a907b051b1d4018eb9317aacd860f986083aaa2277307ffbbf`.  Only after that hash passes may it parse the file.  It must then fail closed unless the parsed trainer-level bindings include at least all of the following:

```text
schema_version = stage174a_v1
status = completed
source_provenance.git_commit = abd85a088c274678004432160625d42208112848
source_provenance.git_is_dirty = false
source_provenance.git_diff_names = []
source_provenance.trainer_sha256 = 9792f95df934b8b78cffe07bb7613a35984dba79d56ee7ea719d533dd7117d87
training seed = 180
resolved split seed = 8192
split policy = fixed_explicit_split_seed
dev rows = 720
train rows = 2880
loaded rows = 3600
architecture = v6b_minimal
backbone = mamba
model = state-spaces/mamba-130m-hf
device = cuda
freeze_encoder = true
arm = A0
router mode = explicit_product
gradient ownership = joint
reason loss weight = 0.0
A0 reference predictions consumed = none
completed epochs = 20
selected epoch = 20
selected checkpoint SHA256 = 0724f5a2e537c932f6692dd74713d57fc70f8182ee2719c33665b09114bd944a
selected checkpoint bytes = 518269879
clean-dev-only selection = true
external data used = false
external labels used = false
```

The trainer timestamps are distinct facts and MUST NOT be conflated with absent cm-wrapper timestamps:

```text
created_at_utc = 2026-09-08T22:41:17.438232Z
finalized_at_utc = 2026-09-08T22:41:42.246155Z
```

## Exact trainer invocation verification

`run_provenance.json` contains `command_string` and `raw_sys_argv`.  The future recovery verification MUST compare them against the exact expanded seed180/A0 command contract frozen by the Seed8192 execution authority, using exact `raw_sys_argv` comparison where feasible rather than a loose subset test.  It must require the exact dataset/sidecar/provenance and seed180/A0 output paths frozen there, enabled selected-checkpoint saving with `selected_checkpoint.pt`, and all of:

```text
--seed 180
--split-seed 8192
--reason-router-arm A0
--reason-router-mode explicit_product
--gradient-ownership-mode joint
--reason-loss-weight 0.0
--freeze-encoder true
--frame-downstream-gradient-mode joint
--epochs 20
--max-length 128
--dev-ratio 0.2
--device cuda
--select-metric final_macro_f1
--ranking-weight 0.0
--class-weighting none
--lr 0.001
--stage174c-clean-pairwise-mode off
--stage174c-clean-pairwise-weight 0.0
--stage174c-clean-polarity-preservation-weight 0.0
--stage175b-support-anchor-mode off
--stage175b-support-anchor-weight 0.0
--stage177c-frame-pairwise-mode off
--stage177c-frame-pairwise-weight 0.0
--compatible-positive-margin-logit 0.0
--compatible-positive-margin-weight 0.0
```

There must be no `--reason-router-a0-reference-predictions`.  Any argv mismatch, path substitution, auxiliary nonzero weight, enabled auxiliary mode, checkpoint target mismatch, or missing exact comparison when feasible is a recovery blocker.  This trainer-level validation does not convert the unavailable wrapper command bytes into standard-wrapper provenance.

## Dataset, P4-L, and split bindings

The frozen execution authority binds the following exact paths, which a future recovery validation must use rather than a historical split174 substitute:

```text
dataset = reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl
P4-L sidecar = reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl
P4-L provenance = reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json
```

Required exact identities are:

| Binding | SHA256 |
| --- | --- |
| Dataset canonical | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| Dataset semantic | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| P4-L sidecar physical | `9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d` |
| P4-L sidecar semantic | `2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9` |
| P4-L provenance physical | `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8` |

The recovery auditor must also revalidate the fixed-explicit-split-seed Seed8192 contract:

| Split identity | SHA256 |
| --- | --- |
| Pair universe | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` |
| Shuffled pair | `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| Train pair | `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` |
| Dev pair | `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` |
| Ordered train-row | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` |
| Ordered dev-row | `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |

## Required distinct recovery-only package

Any later, separately authorized collector/package implementation must use exactly this recovery schema:

```text
contramamba-seed8192-a0-seed180-r2-wrapper-loss-recovery-v1
```

It is explicitly NOT `contramamba-handoff-v3`, and it must not call standard `cm collect` or standard `cm import`.  Its proposed ZIP root is exactly:

```text
recovery_manifest.json
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/training_report.json
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/training_report_predictions.jsonl
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/clean_dev_predictions.json
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/run_provenance.json
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/selected_checkpoint.pt
```

It MUST NOT include synthetic or substituted `run.log`, `run.meta`, `command.sh`, or `start.marker`.

At minimum, `recovery_manifest.json` must record the recovery schema, original run identity, source execution commit, all five exact artifact paths/sizes/SHA256 values, exact `run_provenance.json` SHA256, `standard_cm_wrapper_provenance = INCOMPLETE`, `wrapper_loss_status = AUTHENTIC_ORIGINAL_WRAPPERS_NOT_FOUND`, recovery evidence basis `trainer-level provenance + immutable artifact hashes`, and a recovery collection UTC.  That recovery UTC is new recovery metadata and MUST NOT claim to be historical execution or wrapper time.

The future package must be a recovery-only capture.  It must fail closed before reading/copying a candidate when source path, regular-file status, byte count, SHA256, trainer-level binding, command contract, dataset/P4-L binding, or split binding differs.  It must not mutate original evidence and must keep the recovery package distinct from any standard handoff namespace.

## Boundaries, non-authorizations, and future chain

This candidate authorizes none of the following:

- recovery collector implementation or recovery execution;
- Kaggle execution, checkpoint/model loading, training, evaluation, metric recomputation, or dataset regeneration;
- standard cm collection or standard cm import;
- artifact promotion, A1/A2/A3, calibration, scientific interpretation, or a Seed8192 N=3 admission decision;
- staging, commit, push, branch change, or modification of an implementation, test, artifact, helper, registry, historical report, or Kaggle state.

The only bounded future sequence is: a separate recovery-only collector/package implementation authority; then a separate recovery execution authority; then a separate local recovery importer/auditor authority.  Only after a successful recovery import/audit may a distinct validated-evidence analysis authority decide whether this r2 artifact set is admissible for the Seed8192 N=3 baseline.

No future mechanism may weaken existing `contramamba-handoff-v3` or standard `cm import` provenance validation to accommodate this loss.  It must instead enforce the distinct schema above.  No future mechanism may create wrapper evidence, overwrite/retry/resume the consumed r2 execution, or present trainer-level facts as authentic wrapper files.

## Historical precedent boundary and control-plane context

The historical recovery authority at `233ed0be080e1d30dd47de2e66136475ec2ede76` is structural precedent only.  This candidate explicitly prohibits reuse of its split174 identities, split174 sidecar hashes, historical result namespace, historical recovery run/package identities, and scientific admission decision.  Seed8192 identities above are the only identities eligible for this bridge.

The following are historical operational context only:

- standard cm collect failed because filesystem start-marker discovery selected zero files;
- the later wrapper-preserving frozen v3 path became unusable because authentic original wrapper files are absent from the current Kaggle evidence state;
- the five source artifacts nevertheless remain exact-hash matches.

These failures do not establish training failure, scientific failure, completed wrapper-loss recovery, or fully established artifact provenance validity.

## Independent verification and stop conditions

Independent verification must confirm this report's exact bytes and Git blob after authoring; inspect the authority chain; verify the immutable tables and classification boundaries; and reject any proposal that fabricates wrapper bytes, changes standard-v3 provenance requirements, invokes Stage-K, or crosses a future implementation/execution/import authority boundary.

Any mismatch in frozen identity, required provenance field, exact command comparison, data/P4-L/split binding, artifact hash/size/path, or truthful wrapper-loss classification is a STOP.  It requires a new authority and may not be repaired by changing evidence, history, wrapper metadata, seed, split, output, registry, helper, or importer.

Authoring status: no implementation, no test, no execution, no Kaggle, no training, no evaluation, no staging, no commit, and no push.
