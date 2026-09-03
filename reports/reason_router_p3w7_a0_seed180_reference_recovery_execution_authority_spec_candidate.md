# P3-W7 A0 Seed180 Reference Recovery Execution Authority Spec Candidate

Status: READY candidate, static recovery execution-authority candidate only.

This candidate authorizes no recovery execution during candidate writing. It creates a future execution authority that is valid only after this exact candidate is independently verified and frozen in Git.

## Authority Basis

Current authoring state:

- HEAD: `812c82d96b2461ed7ae236f6c3ba6d0cf775a182`
- Branch: `p3w7-a0-seed180-reference-recovery-execution-authority-v2`
- Corrected helper implementation commit: `812c82d96b2461ed7ae236f6c3ba6d0cf775a182`
- Corrected helper path: `scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py`
- Corrected helper test path: `tests/test_reason_router_p3w7_a0_seed180_reference_recovery_helper.py`

Corrected helper committed LF identities:

| Artifact | SHA256 | Size | CR | CRLF | Verdict |
|---|---:|---:|---:|---:|---|
| `scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py` | `bf63b73d9aac2f2546dc3182599e41cf611470f65f17b61fadd6d11fab450f30` | 30251 | 0 | 0 | PASS |
| `tests/test_reason_router_p3w7_a0_seed180_reference_recovery_helper.py` | `75343e4676104102143cb3a8237d717cee78896a724f1f8cd6983823184257ea` | 28121 | 0 | 0 | PASS |

The predecessor helper commit `98dfe3ee25c266ad0e12e2215f8ca68ea499fdda` is defective for sidecar semantic hashing and is not admissible for execution. It is retained as provenance only.

Upstream and correction authority chain:

| Commit | Role | Verdict |
|---|---|---|
| `3e1bb765883f2d2bad9a77e67dd58b0a691cfc22` | frozen sidecar-semantic correction authority | present; supersedes defective predecessor semantics |
| `df1cba2ed0833026d7e2293b22f6ab47687229cb` | frozen corrected helper implementation authority | present |
| `ceaee6236340ef7006f7004d910f388ec565db0e` | frozen upstream retained-artifact recovery authority | present |
| `2737c3c6116ae3766b469801f990e2c45ba9a55e` | formal source seed180 execution | present |
| `233ed0be080e1d30dd47de2e66136475ec2ede76` | historical provenance recovery authority | present |
| `80cb034792f03226cf6e22c196c1229ed4e6dd62` | P4-L authority | present |
| `2f9e6076791358922e3ebd70e89533d9cb83b458` | P4-L canonical builder source commit | present |

AGENTS.md was inspected. This is a static/report-only authority-writing task. No implementation change, recovery execution, training, evaluation, dataset regeneration, checkpoint mutation, model load, Kaggle use, commit, or push is authorized by this candidate-writing phase.

## Canonical Dataset

Dataset:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Canonical Git blob identity:

- SHA256: `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- Size: 1879593 bytes
- Rows: 3600
- CR: 0
- CRLF: 0
- Semantic SHA256: `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

Current Windows authoring checkout diagnosis:

- Working-tree SHA256: `eedbf93cf7fc3e141c4a49511750cbe4d8b0443e7de3463ea7e77696aca2c572`
- Working-tree size: 1883193 bytes
- Working-tree rows: 3600
- Working-tree CR: 3600
- Working-tree CRLF: 3600
- `git ls-files --eol`: `i/lf w/crlf`

Verdict: PASS for candidate writing. The Git blob is the canonical dataset identity. The CRLF-expanded authoring working tree is checkout representation only, not canonical dataset drift. Future recovery execution is prohibited in this authoring worktree.

## Canonical Sidecar

Sidecar:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`

Canonical Git blob physical identity:

- Physical SHA256: `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- Size: 8390489 bytes
- Rows: 3600
- CR: 0
- CRLF: 0

Current Windows authoring checkout diagnosis:

- Working-tree physical SHA256: `a04f991554876cd6fea049d8ed494cd4a2f548ee5f69d08c8eacd9db6293389a`
- Working-tree size: 8394089 bytes
- Working-tree CR: 3600
- Working-tree CRLF: 3600
- `git ls-files --eol`: `i/lf w/crlf`

Corrected semantic sidecar probe:

- Semantic SHA256: `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`
- Semantics: preserve row order; exclude exactly `created_at`; canonical JSON array; compact sorted-key serialization; `allow_nan=False`

Verdict: PASS. Physical and semantic identities are distinct and must not be conflated. The corrected helper must use semantic sidecar identity, with no physical-SHA fallback.

## Semantic Self-Binding

This candidate defines:

`RECOVERY_EXECUTION_AUTHORITY_COMMIT`

as the immutable Git commit that first freezes this exact independently verified candidate.

Before freeze, the literal 40-hex commit is unknown. This is an intentional semantic future binding, not a placeholder defect.

After freeze:

- the controller must obtain the literal 40-hex `RECOVERY_EXECUTION_AUTHORITY_COMMIT`;
- the dedicated execution worktree HEAD must equal that exact literal;
- the helper CLI must receive the same exact literal via `--expected-recovery-execution-authority-commit`;
- `812c82d96b2461ed7ae236f6c3ba6d0cf775a182` must be an ancestor of runtime HEAD;
- `812c82d96b2461ed7ae236f6c3ba6d0cf775a182` is not runtime HEAD unless it is also the later freeze commit, which this candidate does not assume;
- neither `98dfe3ee25c266ad0e12e2215f8ca68ea499fdda` nor `ceaee6236340ef7006f7004d910f388ec565db0e` is runtime HEAD.

Verdict: PASS. The self-binding is unambiguous and fail-closed.

## Dedicated LF Execution Worktree Contract

Actual recovery execution must not occur in a CRLF-expanded authoring worktree.

After this exact candidate is independently verified and frozen, future setup must use a distinct dedicated worktree checked out at `RECOVERY_EXECUTION_AUTHORITY_COMMIT`:

```bash
git -C <SOURCE_REPO> -c core.autocrlf=false -c core.eol=lf worktree add --detach <EXECUTION_WORKTREE> <RECOVERY_EXECUTION_AUTHORITY_COMMIT>
```

This candidate-writing task does not create that worktree.

Future pre-helper dedicated-worktree byte preflight must require:

- HEAD equals `RECOVERY_EXECUTION_AUTHORITY_COMMIT`;
- `812c82d96b2461ed7ae236f6c3ba6d0cf775a182` is an ancestor of HEAD;
- corrected helper SHA256 equals `bf63b73d9aac2f2546dc3182599e41cf611470f65f17b61fadd6d11fab450f30`;
- corrected helper semantic sidecar probe equals `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`;
- dataset SHA256 equals `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`;
- dataset size equals 1879593 bytes;
- dataset CR equals 0;
- dataset CRLF equals 0;
- `git ls-files --eol` or equivalent establishes LF-compatible index and working-tree representation;
- tracked/index/worktree state is clean;
- no unexpected untracked files are present.

If any preflight check fails, classify `EXECUTION_ENVIRONMENT_BLOCKED`.

Forbidden eligibility manufacture:

- no manual normalization;
- no restore-over;
- no checkout-over;
- no `git clean`;
- no `git reset`;
- no renormalization;
- no changed expected SHA;
- no weakened helper validation.

Verdict: PASS. Dedicated LF execution-worktree contract is specified fail-closed.

## Retained ZIP

Exact retained ZIP:

`C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip`

Whole-file identity:

- SHA256: `6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966`
- Size: 527420392 bytes
- Member count: 6

Required member table:

| Member | Size | SHA256 |
|---|---:|---|
| `recovery_manifest.json` | 2144 | `69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json` | 306114 | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/selected_checkpoint.pt` | 518269815 | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` |
| `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` |

Verdict: PASS. No alternate ZIP, member, helper, or seed is admissible.

## Destination Pre-State

Destination:

`reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/`

Observed current authoring pre-state: destination base path absent. Therefore all first-execution final paths are absent:

- `training_report.json`: absent
- `clean_dev_predictions.json`: absent
- `training_report_predictions.jsonl`: absent
- `selected_checkpoint.pt`: absent
- `run_provenance.json`: absent
- `A0_REFERENCE_AUDIT.json`: absent

Future execution rule: if any final path is unexpectedly present before first authorized execution, BLOCK and inspect provenance. Never delete existing outputs merely to satisfy authority.

Verdict: PASS.

## Exact Future Command

After freeze, with the literal authority commit substituted:

```bash
python scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py materialize-reference --zip C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip --expected-recovery-execution-authority-commit <RECOVERY_EXECUTION_AUTHORITY_COMMIT>
```

No alternate command semantics are authorized.

## Authorized Future Writes

Only the following repository final paths are authorized:

- `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json`
- `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json`
- `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl`
- `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/selected_checkpoint.pt`
- `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json`
- `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/A0_REFERENCE_AUDIT.json`

Helper-owned staging is authorized only according to frozen corrected helper semantics.

No other repository write is authorized.

## A0_REFERENCE_AUDIT Contract

The future persisted audit must satisfy the complete frozen helper/upstream contract and must explicitly preserve:

| Field | Required value |
|---|---|
| `status` | `PASS` |
| `source_execution_commit` | `2737c3c6116ae3766b469801f990e2c45ba9a55e` |
| `recovery_authority_commit` | `233ed0be080e1d30dd47de2e66136475ec2ede76` |
| `retained_zip_sha256` | `6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966` |
| `manifest_sha256` | `69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed` |
| `standard_cm_wrapper_provenance` | `INCOMPLETE` |
| `provenance_disposition` | `RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE` |
| `recovery_reference_status` | `RECOVERY_REFERENCE_AUDIT_PASS` |
| dataset physical SHA | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| sidecar semantic SHA | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |

Historical standard CM wrapper provenance must not be upgraded.

The audit must be reread after persistence and exact persisted equality must be validated according to corrected helper semantics.

Verdict: PASS, including exact `provenance_disposition`.

## Outcome Classification

`SUCCESS`: helper exit 0 plus all artifact hashes/sizes and persisted-audit reread gates PASS.

`HELPER_BLOCKED`: helper exit 64; preserve diagnostic; no manual overwrite or delete.

`EXECUTION_ENVIRONMENT_BLOCKED`: dedicated-worktree byte/config/preflight fails before helper invocation.

`ENVIRONMENT_FAILURE`: Git, Python, or filesystem failure outside helper provenance rejection.

`PARTIAL_PUBLICATION`: interruption after some exact authorized artifacts are published.

## Retry Semantics

No retry after `SUCCESS`.

Same-authority retry is allowed only when all of the following remain unchanged:

- authority commit;
- corrected helper bytes;
- retained ZIP bytes.

And only when one of the following holds:

- environment failure occurred before public writes;
- interruption left only exact-identical helper-admissible partial outputs and no audit.

Block conditions:

- nonidentical final artifact: BLOCK;
- existing audit: BLOCK;
- changed helper, ZIP, or authority commit: new authority required;
- ambiguous state: BLOCK pending inspection.

## Post-Success Boundary

`SUCCESS` establishes only:

- corrected helper execution success;
- exact retained-artifact materialization;
- seed-local `A0_REFERENCE_AUDIT` PASS;
- preserved historical provenance caveat.

It does not establish a scientific conclusion, change A0 metrics, release A1/A2/A3, authorize training/evaluation, authorize factorial interpretation, or automatically authorize commit/push.

After `SUCCESS`, independent result/artifact verification is required before small-file freeze/import.

Potentially Git-eligible only after separate verification:

- `training_report.json`
- `clean_dev_predictions.json`
- `training_report_predictions.jsonl`
- `run_provenance.json`
- `A0_REFERENCE_AUDIT.json`

Explicitly not Git-eligible:

- `selected_checkpoint.pt`

## Candidate-Writing Validation Summary

Candidate-writing validation performed only read-only repository and artifact inspections plus creation of this single Markdown candidate.

Verdicts:

- READY: yes.
- Corrected helper commit/SHA: PASS.
- Correction-authority chain: PASS.
- Corrected semantic sidecar probe: PASS.
- Physical-vs-semantic sidecar distinction: PASS.
- Canonical Git-blob dataset identity: PASS.
- Authoring CRLF diagnosis: PASS as checkout representation only.
- Semantic self-binding: PASS.
- Dedicated LF execution-worktree contract: PASS.
- Retained ZIP identity/member table: PASS.
- Destination pre-state: PASS.
- Exact command template: PASS.
- `A0_REFERENCE_AUDIT` required fields: PASS, including exact `provenance_disposition`.
- Authorized write set: PASS.
- Outcome/retry semantics: PASS.
- Post-success/scientific boundary: PASS.
- `selected_checkpoint.pt` Git exclusion: PASS.

Explicitly not performed:

- no helper execution;
- no retained ZIP extraction/materialization;
- no destination writes;
- no dataset or sidecar modification;
- no training;
- no evaluation;
- no checkpoint/model load;
- no Kaggle use;
- no commit;
- no push.
