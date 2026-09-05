# P3-W7 A1 Seed180 Factorial Recovery Execution Authority Specification Candidate

Authority/version: `P3W7_A1_SEED180_FACTORIAL_RECOVERY_EXECUTION_AUTHORITY_V1`

## Verdict

Status: `PASS_READY_FOR_INDEPENDENT_VERIFICATION`.

This candidate conservatively resolves the ambiguous seed180/A1 P3-W7 factorial attempt state by authorizing at most one future replacement seed180/A1 trainer-process launch after independent verification, authority freeze, exact command registration, and preflight.

This is provenance and attempt recovery only. It does not change scientific semantics, does not authorize A2/A3, does not authorize any other seed or arm, and does not authorize training, evaluation, Kaggle execution, staging, commit, or push during authoring.

## 1. Authority Chain

Authority precedence consumed:

1. Current workflow-controller instruction.
2. Mandatory worktree: `C:\p3w7-a0-n3-validated-evidence-analysis`.
3. Current recovery candidate materialization commit: `28cba185e6a0b0e76364697b37cd31f4fa2fc060`.
4. Base frozen P3-W7 A1/A2/A3 factorial execution authority:
   - commit: `230088191cdb774cf24a4aaf11a4424bc7165513`
   - path: `reports/reason_router_p3w7_a1_a2_a3_factorial_execution_authority_spec_candidate.md`
   - frozen SHA256: `3fae9f06fd997373760ecbbb0393d53ae2da4e53f2375454d2f80825b217e099`
5. A0 N=3 validated-evidence report parent: `52e024b6a1389fb3dd46d1ec58ad8b4b99c86c6b`.
6. Seed180 immutable A0 source:
   - commit: `b32d73dfa49b6b9dfabf3093802904323cf679cd`
   - path: `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl`
   - SHA256: `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef`
   - bytes: `3934123`
7. Repository `AGENTS.md`.
8. Imported failure audits inspected at:
   - `C:\Users\Home1\.contramamba\imports\p3w7-factorial-a1-seed180-auth2300881-v2_230088191cdb_20260904_172920`
   - `C:\Users\Home1\.contramamba\imports\p3w7-factorial-a1-seed180-auth2300881-v2-transport-recovery1_230088191cdb_20260904_174320`

The original frozen factorial authority remains the scientific contract source for A1 seed180, but it must not be used for another seed180/A1 attempt.

## 2. Prior Failure Disposition

### Attempt 1

| Field | Value |
|---|---|
| Run name | `p3w7-factorial-a1-seed180-auth2300881-v2` |
| Registered HEAD | `230088191cdb774cf24a4aaf11a4424bc7165513` |
| Registered command SHA256 | `0052592a192db5990127f02e3d6540b4094d10744a9989307f54a8730673b525` |
| Exit code | `127` |
| Failure | `COMMAND_REGISTRATION_TRANSPORT_FAILURE` |
| Imported scientific artifact count | `0` |

Disposition: local PowerShell registration text was replayed by Bash. The trainer process was not launched. Scientific execution did not start. The run imported successfully with zero scientific artifacts. This run name is permanently consumed and must not be reused.

### Attempt 2

| Field | Value |
|---|---|
| Run name | `p3w7-factorial-a1-seed180-auth2300881-v2-transport-recovery1` |
| Registered HEAD | `230088191cdb774cf24a4aaf11a4424bc7165513` |
| Registered command SHA256 | `6646e3a37f36530fcd42800073a0e2d61f2e60d800df53bd40f8836349cc96b2` |
| Exit code | `2` |
| Failure | `PRETRAINING_PROVENANCE_COLLISION` |
| Imported scientific artifact count | `0` |

Disposition: the observed registered attempt stopped at `run-name/provenance collision` before its trainer invocation line. Its collected/imported scientific artifact count is zero. This run name is permanently consumed and must not be reused.

## 3. Orphan Runtime Evidence

Independently supported observed collision path:

`/kaggle/working/contramamba_factorial_wrapper_provenance/p3w7-factorial-a1-seed180-auth2300881-v2-transport-recovery1_a0_reference_provenance.json`

Supported registered recovery1 facts:

| Field | Value |
|---|---|
| Run name | `p3w7-factorial-a1-seed180-auth2300881-v2-transport-recovery1` |
| Registered HEAD | `230088191cdb774cf24a4aaf11a4424bc7165513` |
| Registered command SHA256 | `6646e3a37f36530fcd42800073a0e2d61f2e60d800df53bd40f8836349cc96b2` |
| Exit code | `2` |
| Imported scientific artifact count | `0` |

The available imported evidence establishes only that the registered recovery1 run log encountered a wrapper/provenance collision at the path above before its trainer invocation line. Therefore the observed registered Attempt 2 itself did not launch its trainer.

The available imported evidence does not independently establish:

- the orphan file's SHA256;
- the orphan file's byte count;
- the orphan file's JSON fields or content;
- the orphan file's creation time;
- whether the orphan state originated from an earlier unregistered or replayed execution path;
- whether a trainer process associated with the provenance-producing event launched;
- whether a trainer process associated with the provenance-producing event did not launch.

No exact orphan identity value, byte count, JSON content claim, filesystem mtime relationship, persistent IPython history conclusion, or `.bash_history` conclusion is execution-authoritative in this candidate.

Ambiguity basis:

1. Attempt 1 is independently established as a command-transport failure before trainer launch.
2. The observed registered Attempt 2 is independently established as stopping at a provenance-path collision before its trainer invocation.
3. Therefore the observed registered Attempt 2 itself did not launch its trainer.
4. The collision demonstrates that pre-existing external state existed at the expected provenance path.
5. The currently imported evidence does not establish the origin or lifecycle of that pre-existing state.
6. Absence of imported scientific artifacts from the two registered failed runs does not prove that no other prior trainer process associated with the pre-existing external state ever launched.
7. Therefore neither `TRAINER_DEFINITELY_LAUNCHED` nor `TRAINER_DEFINITELY_DID_NOT_LAUNCH` is justified for the unknown provenance-producing event.
8. Fail closed by treating the original seed180/A1 attempt budget as ambiguously possibly consumed.

Frozen attempt state:

`AMBIGUOUS_POSSIBLY_CONSUMED_PRIOR_SEED180_A1_ATTEMPT`

This is provenance uncertainty, not evidence that training actually occurred. This candidate must not claim either `trainer definitely launched` or `trainer definitely did not launch` for the unknown provenance-producing event. The ambiguity itself is the recovery trigger.

## 4. Recovery Authorization

This authority may authorize exactly one replacement seed180/A1 factorial execution after all post-freeze prerequisites pass.

The replacement is recovery evidence, not an extra factorial replicate. No aggregation, factorial-effect computation, winner selection, promotion, mechanism claim, threshold tuning, or scientific interpretation is authorized by this recovery authority.

Only this arm/seed is authorized:

| Field | Value |
|---|---|
| Arm | `A1` |
| Seed | `180` |
| Split seed | `174` |
| Epochs | `20` |
| Architecture | `v6b_minimal` |
| Backbone | `mamba` |
| Model | `state-spaces/mamba-130m-hf` |
| Encoder | frozen |
| Router | `conditional_first_blocker` |
| Ownership | `joint` |
| Reason supervision | active |
| Reason-loss weight | exactly `0.6202430063306562` |
| A0 reference | same-seed only |
| Dataset/sidecar | exact current-lineage P4-L identities in Section 6 |

Non-authorization:

- no A0 rerun
- no E0
- no A2
- no A3
- no extra seed
- no extra arm
- no modification to trainer, tests, data, A0 artifacts, P4-L artifacts, cm tooling, Git history, or Kaggle state

## 5. A0 Reference Contract

Preserve exact seed180 A0 source:

| Field | Value |
|---|---|
| Commit | `b32d73dfa49b6b9dfabf3093802904323cf679cd` |
| Path | `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl` |
| SHA256 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| Bytes | `3934123` |

Preserve caveat exactly:

```text
standard_cm_wrapper_provenance=INCOMPLETE

provenance_disposition=
RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE
```

Do not normalize, erase, reclassify, or upgrade this historical provenance. Existing exact A0 provisioned input may be consumed only after re-verifying same-seed SHA256 and byte count. No trust by path alone is permitted.

## 6. P4-L Exact Contract

Dataset:

| Field | Value |
|---|---|
| Path | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` |
| Physical SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| Semantic SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |

Sidecar:

| Field | Value |
|---|---|
| Path | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl` |
| Physical SHA256 | `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1` |
| Semantic SHA256 | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |

Sidecar provenance:

| Field | Value |
|---|---|
| Path | `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json` |
| Physical SHA256 | `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2` |

Rows and split:

| Field | Value |
|---|---|
| Total rows | `3600` |
| Train rows | `2880` |
| Dev rows | `720` |
| Split seed | `174` |
| Dev ratio | `0.2` |
| Ordered train identity | `cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784` |

Any drift blocks execution.

## 7. Output And Provenance Namespace

The scientific arm namespace remains:

`reports/reason_router_p3w7_a1_a2_a3_factorial_runs/seed180/A1/`

It must be absent before trainer launch. Do not delete, rename, overwrite, or reuse an existing output directory. Any evidence that a seed180/A1 scientific output currently exists blocks replacement execution.

The outer provenance-level run name must be new and non-recycled. This candidate does not freeze a final run name containing a future commit SHA that does not yet exist.

Deterministic post-freeze naming rule:

`p3w7-factorial-a1-seed180-recovery2-auth<short-freeze-sha>`

The controller selects the exact name after authority freeze. The exact name selected under this rule must not equal either consumed prior run name.

Wrapper provenance must use a new filename bound to the new run name. Do not reuse, delete, overwrite, or reinterpret the orphan recovery1 provenance. The expected wrapper-provenance path for the future replacement is:

`/kaggle/working/contramamba_factorial_wrapper_provenance/<NEW_RUN_NAME>_a0_reference_provenance.json`

That path must be absent before wrapper provenance creation.

## 8. Command-Byte Registration Contract

Read-only inspection of `C:\Users\Home1\Downloads\cm.ps1` found the current command registry behavior:

- `cm run save <name>` reads `(Get-Clipboard -Raw).Trim()`.
- If the first line is `%%bash`, the tool drops that line and trims again.
- The registry command SHA256 is computed from `[System.Text.Encoding]::UTF8.GetBytes($runCommand)`.
- `cm run <name>` recomputes the registered command hash before handoff.
- `cm run <name>` base64-encodes the exact registered command bytes.
- The Kaggle run cell decodes `COMMAND_BASE64` with `printf '%s' "$COMMAND_BASE64" | base64 --decode > "$COMMAND_FILE"`.
- Kaggle recomputes `sha256sum "$COMMAND_FILE"` and blocks if it differs from the registered `COMMAND_SHA256`.

Read-only inspection did not find support for `CONTRAMAMBA_RUN_COMMAND_BYTE_MODE=utf8-final-lf-v1`, did not find a `utf8-final-lf-v1` command-byte mode, and did not find a supported policy that preserves exactly one final LF byte through `cm run save`. Therefore this candidate does not authorize use of that unsupported environment variable or a final-LF canonical command policy under the inspected tooling.

Future registration under the currently inspected `cm.ps1` must instead:

1. construct the exact approved shell command as UTF-8 text with LF internal line endings;
2. contain no leading or trailing whitespace bytes because current `cm run save` trims them;
3. contain no final LF in the independently reviewed command bytes because current `cm run save` trims it;
4. independently compute the intended command SHA256 over exactly the same UTF-8 bytes that current `cm run save` will store;
5. copy only the exact shell command to clipboard, with no Markdown fence, no `%%bash`, and no local PowerShell registration text;
6. run `cm run save <NEW_RUN_NAME>` under the exact frozen recovery-authority HEAD;
7. require the registry `HASH` to equal the independently precomputed intended SHA256 exactly;
8. block before `cm run <NEW_RUN_NAME>` on any byte/hash mismatch;
9. never reinterpret any line-ending, trim, final-LF, registry, or transport hash mismatch as harmless after registration.

If a later cm tooling revision claims final-LF support, that support must be independently inspected and frozen before use. If future tooling cannot guarantee exact reviewed bytes from save through Kaggle decode, execution is blocked with:

`BLOCKED_RECOVERY_COMMAND_BYTE_REGISTRATION_CONTRACT`

## 9. Post-Freeze Execution Prerequisites

Execution is forbidden unless all prerequisites pass:

1. this recovery candidate is independently verified;
2. this recovery candidate is committed, pushed, and frozen;
3. exact remote full SHA of the recovery freeze commit is verified;
4. runtime HEAD equals the recovery freeze commit;
5. trainer/source semantics are unchanged except the report-only recovery authority commit;
6. repo and index are clean;
7. exact P4-L physical and semantic identities in Section 6 are verified;
8. exact seed180 A0 Git-object source SHA and bytes in Section 5 are verified;
9. exact output namespace in Section 7 is absent;
10. orphan recovery1 provenance is retained and not reused;
11. exact new wrapper-provenance path is absent;
12. exact approved command is registered against the recovery freeze commit;
13. registry command SHA equals independently reviewed command bytes;
14. GPU is ON only for the actual trainer workload.

## 10. Attempt Consumption Boundary

This recovery authority grants exactly one replacement trainer-process launch.

Before trainer process launch, wrapper or preflight rejection does not consume the replacement attempt.

At trainer process launch, the replacement attempt is consumed regardless of later success or failure.

If that replacement trainer launch fails, no further retry is authorized under this authority. A later retry requires another separately created, independently verified, frozen recovery authority.

## 11. Stop-On-Failure Rules

Return `BLOCKED` and do not auto-repair on:

- any preflight mismatch
- any HEAD, hash, or command mismatch
- any output collision
- any A0 source/destination SHA or byte mismatch
- any provenance collision
- any evidence that seed180/A1 scientific output currently exists
- any need to alter scientific arguments
- any need to modify trainer, code, tests, or data
- any attempt to authorize A2/A3 before replacement seed180/A1 is provenance-validly collected/imported
- any command-byte registration contract drift that cannot be reconciled with exact reviewed bytes

## 12. Status Distinctions

Required later reports must distinguish:

1. code correctness
2. recovery execution success
3. artifact/provenance validity
4. scientific conclusion

A recovery run `PASS` alone establishes no scientific claim.

## 13. Candidate Materialization Notes

Candidate materialization target:

`reports/reason_router_p3w7_a1_seed180_factorial_recovery_execution_authority_spec_candidate.md`

Current materialization state:

- Worktree: `C:\p3w7-a0-n3-validated-evidence-analysis`
- Branch: `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`
- `BASE_FACTORIAL_AUTHORITY_COMMIT`: `230088191cdb774cf24a4aaf11a4424bc7165513`
- `CURRENT_RECOVERY_CANDIDATE_COMMIT`: `28cba185e6a0b0e76364697b37cd31f4fa2fc060`
- The candidate is already tracked in `CURRENT_RECOVERY_CANDIDATE_COMMIT`.
- The current recovery candidate commit is not yet a verified or frozen execution authority merely because it was committed.
- The corrected candidate status is `PASS_READY_FOR_INDEPENDENT_VERIFICATION`.

A later exact recovery freeze commit may become execution authority only after independent verification `PASS_READY_FOR_FREEZE`, commit/push, remote full-SHA verification, runtime bootstrap at that exact SHA, exact new run naming, exact new command review/registration, and all candidate post-freeze preconditions.

Final candidate SHA256, byte count, line-ending facts, `git diff --check`, `git diff --name-status`, `git diff --cached --name-status`, and `git status --short` are intentionally reported outside this file to avoid self-referential candidate content.
