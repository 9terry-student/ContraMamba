# P3-W7 A1 Seed180 Pretrainer Retry Execution Authority Specification Candidate

Authority/version: `P3W7_A1_SEED180_PRETRAINER_RETRY_EXECUTION_AUTHORITY_V1`

## Verdict

Status: `PASS_READY_FOR_INDEPENDENT_VERIFICATION`.

This is a narrow report-only authority correction candidate for the seed180/A1 recovery state after the imported recovery2 pretrainer preflight failure. It does not itself authorize trainer execution. It defines the authority interpretation and future prerequisites under which one later independently verified and frozen commit, based on this candidate, may become the recovery execution authority for exactly one seed180/A1 replacement trainer-process launch.

No training, evaluation, Kaggle execution, code implementation, staging, commit, or push is authorized by authoring this candidate.

## 1. Authority Chain And Current State

Authority precedence consumed:

1. Current workflow-controller instruction.
2. Mandatory worktree: `C:\p3w7-a0-n3-validated-evidence-analysis`.
3. Mandatory starting HEAD: `77bbe71cc1cdd1fc9afb04640fdf37088e11a87c`.
4. Branch: `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`.
5. Current corrected recovery candidate at HEAD `77bbe71cc1cdd1fc9afb04640fdf37088e11a87c`:
   `reports/reason_router_p3w7_a1_seed180_factorial_recovery_execution_authority_spec_candidate.md`.
6. Base factorial execution authority commit:
   `230088191cdb774cf24a4aaf11a4424bc7165513`.
7. Imported recovery2 preflight-failure audit:
   `C:\Users\Home1\.contramamba\imports\p3w7-factorial-a1-seed180-recovery2-auth28cba18_28cba185e6a0_20260905_220954`.
8. Repository `AGENTS.md`.

HEAD `77bbe71cc1cdd1fc9afb04640fdf37088e11a87c` is a corrected recovery candidate revision. It is not itself sufficient trainer execution authority merely because its commit message may say `Freeze`. The corrected recovery candidate body says `PASS_READY_FOR_INDEPENDENT_VERIFICATION` and requires a later exact freeze commit after independent verification. Therefore `77bbe71cc1cdd1fc9afb04640fdf37088e11a87c` must not be treated as seed180/A1 trainer execution authority.

Future execution authority, if any, is a later independently verified, committed, pushed, remotely verified, exact frozen commit based on this candidate. Until that exact future freeze commit exists, the final executable authority identity and final outer run name remain unresolved.

## 2. Authority Collision Search

The current corrected recovery candidate resolves the prior recovery1/orphan ambiguity only up to candidate status. It still requires independent verification and a later exact freeze commit. It also uses the old deterministic post-freeze outer name rule with `recovery2`, which is now consumed by the imported failure described below.

Repository search found references to `p3w7-factorial-a1-seed180-recovery2-auth28cba18` and `28cba185e6a0b0e76364697b37cd31f4fa2fc060` only in the existing recovery candidate. No existing applicable authority was found that already resolves the recovery2 pretrainer preflight failure, consumed outer run name, non-authoritative orphan-byte treatment after import, and future `recovery3` naming state. This candidate is therefore the narrow authority-state correction artifact.

## 3. Recovery2 Imported Audit Verification

Imported audit path inspected:

`C:\Users\Home1\.contramamba\imports\p3w7-factorial-a1-seed180-recovery2-auth28cba18_28cba185e6a0_20260905_220954`

Imported files inspected:

- `manifest.json`
- `import.json`
- `run.meta`
- `run.log`
- `command.sh`

Verified imported identities:

| Field | Value |
|---|---|
| Run name | `p3w7-factorial-a1-seed180-recovery2-auth28cba18` |
| Registered/executed HEAD | `28cba185e6a0b0e76364697b37cd31f4fa2fc060` |
| Command SHA256 | `78464a22732e123858c5c603940dc78945bab1b4885e7b72469832835177ee6a` |
| ZIP SHA256 | `9ad879747a2219170de6c51cd2e1e2d1611ea9f6fc2799ecbc7d0d248d8b7ce4` |
| Exit code | `2` |
| Started UTC | `2026-09-05T13:06:46Z` |
| Finished UTC | `2026-09-05T13:06:47Z` |
| Run log SHA256 | `767b465eb605e071f42484509f1ffc69c52fb2d192a795d4746f0e321c7dca92` |
| Run meta SHA256 | `eef97e89e76415525fb58bd788ab3654dee41714a066709121b817b47ad5bccf` |
| Imported scientific files | `0` |
| Observed failure | `FACTORIAL_RECOVERY_RUN_BLOCKED: required orphan recovery1 provenance is missing` |

The imported `run.log` shows `P4L_SEMANTIC_BINDING_PREFLIGHT=PASS`.

The imported `run.log` does not show `RECOVERY_PREFLIGHT_PASS`.

The imported `run.log` does not show `TRAINER_PROCESS_LAUNCH_BEGIN`.

Therefore the trainer process did not launch in this registered recovery2 run. The one authorized replacement trainer-process launch was not consumed by recovery2. The outer run name `p3w7-factorial-a1-seed180-recovery2-auth28cba18` is consumed and must never be reused.

This imported failure is provenance evidence only. It is not scientific evidence.

## 4. Attempt Consumption Boundary

The scientific replacement trainer-launch budget remains exactly one.

Wrapper or preflight rejection before trainer process launch does not consume the remaining replacement launch. `TRAINER_PROCESS_LAUNCH_BEGIN` or actual trainer process launch consumes it. Once trainer launches, success or failure consumes the only replacement. Any further launched retry after that would require a new separately authorized recovery authority.

Recovery2 stopped before `RECOVERY_PREFLIGHT_PASS`, before `TRAINER_PROCESS_LAUNCH_BEGIN`, and before any trainer process launch. Therefore recovery2 consumed only its outer wrapper/provenance run name, not the replacement trainer launch.

## 5. Orphan Evidence Treatment

The current corrected recovery candidate at HEAD `77bbe71cc1cdd1fc9afb04640fdf37088e11a87c` explicitly establishes that the imported evidence does not independently establish:

- orphan SHA256;
- orphan byte count;
- orphan JSON fields or content;
- orphan creation time;
- origin or lifecycle;
- whether the unknown provenance-producing event launched a trainer.

Therefore SHA256 `5ae39056ef6f0561055391153a86efa0b7708e3bdf52f963bbd4f877eb4c00e1` and byte count `1439` are not execution-authoritative. Future preflight must not bind to those values. Future preflight must not require reconstructed JSON content to validate historical provenance.

The CPU-only Kaggle rematerialization performed after the recovery2 import must not be promoted into historical evidence. If that rematerialized external file still exists, it is `NON_AUTHORITATIVE_SESSION_STATE`. It must not be copied into scientific output as historical evidence, must not be used to resolve the original ambiguity, and must not be overwritten, deleted, or reinterpreted by the future wrapper. If absent in a fresh runtime, absence alone must not block execution. If present, leave it untouched; presence alone must not establish identity or provenance validity.

The authoritative historical basis is the imported registered Attempt-2 collision evidence, not reconstructed external bytes.

The original orphan-producing event remains:

`AMBIGUOUS_POSSIBLY_CONSUMED_PRIOR_SEED180_A1_ATTEMPT`

The orphan external byte identity/content remains:

`UNRESOLVED_NOT_EXECUTION_AUTHORITATIVE`

## 6. Future Wrapper Provenance

A future independently verified and frozen authority may authorize creation of a new run-specific wrapper provenance file for the future pretrainer retry. The future wrapper provenance must:

- bind the new non-recycled run name;
- bind the future exact freeze commit;
- bind seed180/A1 and exact same-seed A0 source;
- record the imported Attempt-1 disposition;
- record the imported Attempt-2 disposition;
- record the imported recovery2 preflight disposition;
- record the original orphan-producing event as `AMBIGUOUS_POSSIBLY_CONSUMED_PRIOR_SEED180_A1_ATTEMPT`;
- explicitly mark orphan external byte identity/content as `UNRESOLVED_NOT_EXECUTION_AUTHORITATIVE`;
- never copy or claim reconstructed orphan JSON as historical evidence.

The future wrapper may create only a new run-specific wrapper provenance file at a new absent path. It must not reuse any prior wrapper provenance filename or run name.

## 7. Recovery3 Naming Semantics

Consumed outer provenance run name:

`p3w7-factorial-a1-seed180-recovery2-auth28cba18`

Future deterministic outer run naming rule:

`p3w7-factorial-a1-seed180-recovery3-auth<short-freeze-sha>`

`recovery3` is the third outer provenance/wrapper recovery name. It is not a third trainer launch. The scientific replacement trainer-launch budget remains exactly one because no trainer launch occurred in recovery2.

The final exact run name is selected only after the future freeze commit exists. The `<short-freeze-sha>` component must derive from that future exact freeze commit, not from HEAD `77bbe71cc1cdd1fc9afb04640fdf37088e11a87c`, not from registered recovery2 HEAD `28cba185e6a0b0e76364697b37cd31f4fa2fc060`, and not from base factorial authority `230088191cdb774cf24a4aaf11a4424bc7165513`.

## 8. Scientific Contract Preservation

This candidate preserves the scientific contract exactly:

| Field | Value |
|---|---|
| Seed | `180` |
| Arm | `A1` |
| Split seed | `174` |
| Epochs | `20` |
| Architecture | `v6b_minimal` |
| Backbone | `mamba` |
| Model | `state-spaces/mamba-130m-hf` |
| Freeze encoder | `true` |
| Router | `conditional_first_blocker` |
| Ownership | `joint` |
| Reason supervision | active |
| Reason-loss weight | exactly `0.6202430063306562` |
| A0 reference | exact same-seed A0 only |

This candidate preserves the exact P4-L data and sidecar identities from the current recovery candidate:

| Artifact | Path | Identity |
|---|---|---|
| Dataset | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` | physical SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`; semantic SHA256 `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| Sidecar | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl` | physical SHA256 `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`; semantic SHA256 `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |
| Sidecar provenance | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json` | physical SHA256 `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2` |

Rows and split remain:

| Field | Value |
|---|---|
| Total rows | `3600` |
| Train rows | `2880` |
| Dev rows | `720` |
| Split seed | `174` |
| Dev ratio | `0.2` |
| Ordered train identity | `cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784` |

Exact seed180 A0 source remains:

| Field | Value |
|---|---|
| Commit | `b32d73dfa49b6b9dfabf3093802904323cf679cd` |
| Path | `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl` |
| SHA256 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| Bytes | `3934123` |

Preserved non-authorizations:

- no A0 rerun;
- no E0;
- no A2/A3;
- no extra seed or arm;
- no aggregation;
- no winner selection;
- no threshold tuning;
- no promotion;
- no mechanism claim;
- no scientific interpretation.

## 9. Output Collision Policy

Scientific namespace remains:

`reports/reason_router_p3w7_a1_a2_a3_factorial_runs/seed180/A1/`

It must be absent before trainer launch. Never delete, overwrite, or reuse an existing scientific output directory.

The future new wrapper provenance path must be absent before creation. Do not reuse any prior wrapper provenance filename or run name.

## 10. Command-Byte Contract

This candidate preserves the currently inspected `cm.ps1` command-byte contract:

- command bytes are UTF-8;
- internal line endings are LF;
- command text has no leading or trailing whitespace;
- the registered command has no final LF;
- command SHA256 is independently precomputed over the exact intended bytes;
- the registry `HASH` must match the independently reviewed SHA256 exactly;
- execution blocks before `cm run` on mismatch;
- unsupported `utf8-final-lf-v1` mode is not authorized.

Any future tooling change that claims final-LF support must be independently inspected and frozen before use.

## 11. Future Execution Prerequisites

Future execution is forbidden unless all of the following are true:

1. this candidate receives independent verifier `PASS`;
2. a later exact freeze commit exists;
3. the exact freeze commit is pushed;
4. remote presence of the exact full SHA is verified;
5. Kaggle bootstrap checks out that exact future freeze SHA;
6. repo and index are clean;
7. trainer semantics are unchanged;
8. exact P4-L physical and semantic identities are verified;
9. exact same-seed A0 source is verified;
10. scientific output namespace is absent;
11. new wrapper provenance path is absent;
12. exact command registration and hash verification pass;
13. GPU is ON only immediately for the actual trainer workload.

Any mismatch blocks execution fail-closed.

## 12. Candidate Materialization Notes

Candidate materialization target:

`reports/reason_router_p3w7_a1_seed180_factorial_pretrainer_retry_execution_authority_spec_candidate.md`

Expected authoring state:

- Worktree: `C:\p3w7-a0-n3-validated-evidence-analysis`
- Branch: `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`
- HEAD: `77bbe71cc1cdd1fc9afb04640fdf37088e11a87c`
- Initial tracked/index state: clean
- Expected delta: exactly one new untracked file, this candidate

Final candidate SHA256, byte count, LF count, CR count, final-LF status, `git diff --check`, `git diff --name-status`, `git diff --cached --name-status`, `git status --short`, and untracked-file verification are intentionally reported outside this file to avoid self-referential candidate content.
