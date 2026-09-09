# P3-W7 Seed8192 A0 Seed180 R2 Provenance Recovery-Handoff Execution Authority Specification Candidate

## Verdict and boundary

`PASS_READY_FOR_INDEPENDENT_SEED8192_A0_SEED180_R2_RECOVERY_HANDOFF_AUTHORITY_VERIFICATION`

This REPORT-ONLY candidate freezes a narrow, recovery-only handoff mechanism.
It authorizes no recovery execution, `cm import`, training, evaluation,
checkpoint loading, Kaggle execution, commit, push, source change, wrapper
change, marker mutation, or import of the failed zero-file ZIP merely by being
authored.  It may become recovery-HANDOFF-generation authority only after
independent verification PASS, exact byte/blob freeze, commit/push, and remote
identity verification.  A later real local import remains a separate authority
boundary.

The decision is that a truthful unmodified-`cm import`-compatible handoff is
representable.  The frozen corrected helper validates v3 schema, run/registry/
command/commit/wrapper/timestamp bindings and every selected file's path, size,
and SHA256, but does not validate or constrain the informational
`artifact_discovery` property.  The recovery manifest must therefore use the
truthful distinct value:

```text
artifact_discovery = "fixed_authorized_path_sha256_recovery"
```

It must not claim `filesystem_start_marker`.  That value describes the failed
standard collector's discovery method, not this recovery's exact authorized
path-and-hash selection.

## Authority and authenticated inputs

| Binding | Exact value |
| --- | --- |
| Current authority HEAD | `4352dfa6fa719c84716bca3e1f5df4efae7b7a4d` |
| R2 execution commit / expected and actual manifest commit | `abd85a088c274678004432160625d42208112848` |
| Execution-authority report | `reports/reason_router_p3w7_seed8192_revised_split_a0_execution_authority_spec_candidate.md` |
| Corrected-import evidence freeze | `reports/reason_router_p3w7_seed8192_a0_seed180_r2_cm_import_utc_timestamp_comparison_correction_implementation_evidence_freeze_candidate.md` |
| Frozen helper | `C:\Users\Home1\.contramamba\cm.ps1` |
| Helper SHA256 / bytes | `09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae` / `91498` |
| Run name | `p3w7-seed8192-a0-seed180-r2` |
| Registered command SHA256 | `82be0c377e305228609e8ce9f75a6b3e8b8a6f83f9999a1543914f9fc517c1f4` |
| `run.log` SHA256 | `a0d4b015cc77e8060184a5035333fe46e101f52d931af3101950382e24407f4e` |
| `run.meta` SHA256 | `23989109951bdf2b3fdfc17f9a7739c34e2875131bc76c0ccb1e313858fcdc27` |
| `STARTED_UTC` | `2026-09-08T22:38:29Z` |
| `FINISHED_UTC` | `2026-09-08T22:41:44Z` |
| `EXIT_CODE` | `0` |

Read-only inspection found the current local registry entry for this exact run.
It has `head=abd85a088c274678004432160625d42208112848`, the exact registered
command SHA256 above, and a recomputable UTF-8 command matching that hash.

The older `reason_router_p3w7_a0_seed180_provenance_recovery_execution_authority_spec_candidate.md`
is historical context only.  Its formal A0 execution lineage is
`2737c3c6116ae3766b469801f990e2c45ba9a55e`, split 174, and different artifact
paths.  It is not authority for this revised-split Seed8192 r2 recovery.

## Fixed recovered artifacts

The authentic original r2 source directory is:

```text
reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0
```

The current authority checkout does not contain that unimported run directory;
this report does not infer local presence from the registry entry.  A later
activated recovery may proceed only in the environment that holds the original
evidence, where every fixed path must pass the checks below.

Only the following five files are eligible.  Selection is exactly source path
plus this SHA256; start-marker mtime, recursive discovery, globbing, and any
other candidate are prohibited.

| Relative path | Required SHA256 |
| --- | --- |
| `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/training_report.json` | `146b7330f6cf479bd339b2eb0af886d5eda3eed72589f57c0879bb5f6d9f5d9c` |
| `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/training_report_predictions.jsonl` | `80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334` |
| `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/clean_dev_predictions.json` | `5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d` |
| `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/run_provenance.json` | `057237823c0b56a907b051b1d4018eb9317aacd860f986083aaa2277307ffbbf` |
| `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/selected_checkpoint.pt` | `0724f5a2e537c932f6692dd74713d57fc70f8182ee2719c33665b09114bd944a` |

`run.log`, `run.meta`, and `command.sh` must be the authentic original r2
wrapper outputs.  They must be copied as bytes and bound by the hashes above
(with `command.sh` bound by the registered command SHA256).  They must never be
synthesized, edited, backdated, or represented as historical if absent.

## Required future v3 package

The future ZIP pathname is exactly:

```text
/kaggle/working/contramamba_handoffs/p3w7-seed8192-a0-seed180-r2_abd85a088c274.zip
```

Its ZIP root must contain exactly this required import namespace; no failed
zero-file ZIP may be supplied as import input:

```text
manifest.json
run.log
run.meta
command.sh
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/training_report.json
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/training_report_predictions.jsonl
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/clean_dev_predictions.json
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/run_provenance.json
files/reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/selected_checkpoint.pt
```

`manifest.json` must have `schema="contramamba-handoff-v3"`, run name above,
`expected_commit` and `actual_commit` both `abd85a088c274678004432160625d42208112848`,
`command_file="command.sh"`, the exact command/wrapper hashes, canonical UTC
timestamps, exit code zero, `file_count=5`, the distinct truthful discovery
value above, a newly generated recovery-time `collected_utc`, and exactly five
file entries.  Each entry's `path` is the path below `files/`, its `sha256` is
the fixed value in the table, and its `size_bytes` is the actual runtime source
byte count obtained only after the corresponding source SHA256 has passed.
`collected_utc` is recovery-generation metadata, not historical wrapper time.

## Fail-closed recovery procedure required for later activation

Before packaging, the separately activated CPU-only recovery command must be
frozen as exact UTF-8 bytes with a final LF and recorded SHA256 before it is
executed.  It must fail closed before ZIP creation unless all of the following
hold:

- current helper identity equals the frozen SHA256 and byte count;
- the local registry entry exists, its command recomputes to the registered
  SHA256, and its head equals `abd85a088c274678004432160625d42208112848`;
- each authentic `run.log`, `run.meta`, and `command.sh` exists and hashes to
  its required value; semantic wrapper fields exactly match run name, commits,
  command SHA256, canonical timestamps, and exit code;
- each of the five fixed source paths exists and passes its fixed SHA256 before
  its size is read, copied, or placed in the manifest;
- the staging namespace is new and contains only the required root members;
- post-copy staged artifacts and wrapper files rehash exactly, and manifest
  paths, count, sizes, hashes, canonical timestamps, and ZIP entry safety are
  independently checked before the ZIP is released.

No recovery step may inspect or use `start.marker` for selection, mutate a
marker, mutate original evidence, use the standard collector's discovery
result, load a checkpoint, import trainer code, perform a model forward pass,
train, evaluate, regenerate data, or use GPU.  The recovery is packaging only.

The corrected unmodified importer remains the final enforcement point: it must
retain all existing schema, registry, command, commit, local-HEAD, wrapper,
timestamp, ZIP/path, size, SHA256, collision, rollback, and audit validations.
No helper or importer change is authorized.

## Separate future local-import boundary

The future real import must not run on this authority-report HEAD.  The
corrected importer requires:

```text
local HEAD == handoff expected commit == abd85a088c274678004432160625d42208112848
```

It therefore requires a separate/detached local worktree pinned to
`abd85a088c274678004432160625d42208112848`, while leaving the current authority
branch intact.  At import time the external helper must still be exactly
`09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae` and
`91498` bytes.  This candidate does not authorize that import.

## Independent-verifier gate and authoring record

An independent verifier must inspect the source/run-registry/helper identities,
the exact generated recovery command bytes and final-LF hash, the ZIP member
namespace, manifest truthfulness, each source and staged SHA256/size, and the
unchanged importer requirements before any activation.  A mismatch is a STOP
requiring new authority; it must not be repaired by artifact, marker, wrapper,
registry, helper, or importer mutation.

Authoring performed only read-only inspection and creation of this single
candidate report.  Recovery execution: NOT PERFORMED.  Real import: NOT
PERFORMED.  Training: NOT PERFORMED.  Evaluation: NOT PERFORMED.  Kaggle:
NOT PERFORMED.  Commit/push: NOT PERFORMED.
