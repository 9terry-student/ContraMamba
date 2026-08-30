# P3-W7-A0 Seed180 Provenance Recovery Execution Authority Specification Candidate

Authority/version:

`P3W7_A0_SEED180_PROVENANCE_RECOVERY_EXECUTION_AUTHORITY_V1`

## Status

CANDIDATE ONLY.

This candidate authorizes nothing by existence alone. It is a recovery authority
specification candidate for the already-completed P3-W7-A0 seed180 execution
only. It does not authorize training, evaluation, checkpoint regeneration,
artifact mutation, cleanup, overwrite, resume, retry, Kaggle execution, A1, A2,
A3, result promotion, or scientific interpretation.

Future recovery execution requires all of:

1. independent verification PASS for this exact candidate;
2. this exact candidate frozen in a new immutable Git commit;
3. any required recovery collector/import implementation separately authorized,
   implemented, verified, and frozen;
4. post-freeze read-only gates;
5. exact recovery command hash verification;
6. exact checkout of the recovery authority and implementation freeze.

## Authority Basis

Current local materialization HEAD required before creating this candidate:

`56bf9e7dca92d1d7e61ab153038a68aeb21c4017`

Formal Week 1 P3-W7-A0 execution authority freeze:

`2737c3c6116ae3766b469801f990e2c45ba9a55e`

Frozen authority path:

`reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`

Seed180 authorized wrapper command SHA256:

`dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e`

Operational material inspected for this candidate:

- `AGENTS.md`
- `docs/RESEARCH_OPERATIONS.md`
- `reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`
- `C:\Users\Home1\.contramamba\cm.ps1`

The files named in the workflow instruction as `04_KAGGLE_RUNBOOK.md`,
`05_FAILURE_RECOVERY.md`, and `06_NAMING_AND_PROVENANCE.md` were not present as
tracked files under those literal names in the current repository materialization
or at the formal A0 execution freeze. This candidate therefore does not infer
additional operational permissions from absent files.

The commit delta:

`2737c3c6116ae3766b469801f990e2c45ba9a55e..56bf9e7dca92d1d7e61ab153038a68aeb21c4017`

is documentation-only and adds `docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md`.
It does not modify trainer code, dataset artifacts, P4-L sidecar/provenance,
the A0 authority file, A0 execution parameters, loss/gradient semantics, or
original seed180 outputs.

## Original Execution Disposition

Formal A0 execution freeze:

`2737c3c6116ae3766b469801f990e2c45ba9a55e`

Seed:

`180`

Attempt disposition:

`CONSUMED`

The seed180 trainer attempt must not be rerun, retried, resumed, or replaced.

Observed trainer completion:

- `20 / 20 epochs`
- trainer `run_provenance` status: `completed`
- trainer `run_provenance` source Git commit:
  `2737c3c6116ae3766b469801f990e2c45ba9a55e`
- trainer `run_provenance` source Git dirty state: `false`

Forensic notebook-history observation:

`AUTHORIZED_EXACT_MATCH_COUNT=2`

This notebook-history observation is supporting forensic evidence only. It is
not standard `cm run` wrapper metadata and must not be transformed into, or
represented as, historical `run.meta`, `run.log`, `start.marker`, or
`command.sh` evidence.

## Immutable Observed Artifact Anchors

Original Kaggle run directory:

`/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0`

Future recovery must treat the following files as existing immutable source
evidence. It must prohibit deletion, rename, overwrite, regeneration, or
in-place modification.

| File | SHA256 | Size |
| --- | --- | ---: |
| `training_report.json` | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` | 306114 |
| `clean_dev_predictions.json` | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` | 4838225 |
| `training_report_predictions.jsonl` | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` | 3934123 |
| `selected_checkpoint.pt` | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` | 518269815 |
| `run_provenance.json` | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` | 68429 |

Also frozen:

- `prediction_export_row_count = 720`
- `selected_checkpoint selected_epoch = 20`
- `selected_checkpoint SHA256 =
  dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da`

## Recovery Status Boundaries

The statuses remain separate:

- execution success: observed;
- standard `cm run` wrapper provenance: incomplete/missing;
- artifact validity: not yet formally recovered/imported;
- scientific conclusion: `NOT ESTABLISHED`.

Recovery verification mismatch is a provenance recovery blocker only. It is not
automatically a model failure, training failure, scientific failure, or evidence
that the observed artifacts are invalid.

No recovery step may repair, infer, backfill, or fabricate historical wrapper
metadata.

## Current `cm` Contract Findings

The inspected local operator implementation is:

`C:\Users\Home1\.contramamba\cm.ps1`

Observed SHA256:

`94a4333037cf434b895fbee08e70dc1254b1e9ea233d6bad424dd4a4b34ecdaf`

`cm run save <name>`:

- reads a command from the clipboard;
- rejects Markdown code fences;
- removes a leading `%%bash` cell marker if present;
- trims the stored command string;
- stores the current local HEAD, commit subject, command, command SHA256, and
  save timestamp in `$HOME\.contramamba\run-registry.json`;
- computes the command SHA256 over the exact stored UTF-8 command bytes with no
  added final LF.

`cm run <name>`:

- requires the current local HEAD to equal the registry HEAD;
- recomputes the stored command hash from the registry command;
- base64-encodes the exact command bytes for Kaggle transport;
- creates a Kaggle wrapper cell that checks Kaggle HEAD and clean worktree;
- writes the decoded command to
  `/kaggle/working/contramamba_run_logs/<run>_<shortcommit>.command.sh`;
- verifies that command file SHA256 equals the registry hash;
- creates `/kaggle/working/contramamba_run_logs/<run>_<shortcommit>.start.marker`
  immediately before command execution;
- writes wrapper metadata to
  `/kaggle/working/contramamba_run_logs/<run>_<shortcommit>.meta`;
- writes wrapper output to
  `/kaggle/working/contramamba_run_logs/<run>_<shortcommit>.log`;
- appends `FINISHED_UTC` and `EXIT_CODE` after the command exits.

`cm collect <name>`:

- requires a matching local run-registry entry;
- requires current local HEAD to equal the registered HEAD;
- requires Kaggle HEAD to equal the expected commit;
- requires the wrapper-created `run.log`, `run.meta`, `command.sh`, and
  `start.marker`;
- parses provenance only from `run.meta`;
- requires `RUN_NAME`, `EXPECTED_COMMIT`, `ACTUAL_COMMIT`, `COMMAND_SHA256`,
  `STARTED_UTC`, `FINISHED_UTC`, and `EXIT_CODE`;
- requires `COMMAND_SHA256` to match the local registry authorization;
- independently hashes `command.sh`, `run.log`, and `run.meta`;
- discovers artifacts by `find . -type f -newer "$START_MARKER"` under the repo;
- packages `manifest.json`, `run.log`, `run.meta`, `command.sh`, and copied
  files under `files/`;
- writes manifest schema `contramamba-handoff-v3` with run name, expected and
  actual commits, command hash, exit code, start/finish timestamps, run log hash,
  run meta hash, artifact discovery method `filesystem_start_marker`, file
  count, and per-file path/size/SHA256 records.

`cm import <zip>`:

- requires a `.zip` input;
- validates ZIP entry paths before extraction;
- requires `manifest.json`, `files/`, `run.log`, `run.meta`, and `command.sh`;
- requires manifest schema `contramamba-handoff-v3`;
- requires valid run name, `command_file = command.sh`, 64-hex command/run-log/
  run-meta hashes, nonempty timestamps, and nonnegative integer exit code;
- requires a local registry entry for the manifest run name;
- requires registry HEAD to equal handoff expected commit;
- recomputes the registry command hash;
- requires registry command hash to equal manifest command hash;
- requires expected and actual commits to be valid 40-hex and equal;
- requires current local HEAD to equal the handoff expected commit;
- requires manifest `file_count` to equal the number of file entries;
- rehashes `run.log`, `run.meta`, and `command.sh`;
- parses `run.meta` and requires its semantic fields to match the manifest;
- validates every artifact path as relative and repo-confined;
- validates every artifact size and SHA256 before copying;
- blocks local path collisions unless the existing destination file is already
  byte-identical;
- copies only after all validation passes and records an import audit outside
  the Git repository.

Conclusion: current `cm collect` and `cm import` cannot truthfully consume a
seed180 direct-trainer recovery package without pretending that standard wrapper
provenance existed before the trainer execution. They must not be weakened or
bypassed for this recovery.

## No-Code Recovery Decision

No-code recovery through existing `cm collect` / `cm import` is not possible.

Reason: the consumed seed180 trainer was launched directly, so the standard
wrapper evidence required by current `cm collect` / `cm import` is missing or
incomplete. Creating `run.meta`, `run.log`, `start.marker`, or `command.sh`
after the fact and representing them as historical originals would fabricate
wrapper provenance.

The truthful path is a separately authorized recovery-only capture/package and
a separately authorized local recovery import/audit mechanism.

## Required Bounded Recovery Mechanism

A future implementation stage, if separately authorized, must add a recovery
collector/import path distinct from `contramamba-handoff-v3`. The new mechanism
must be recovery-only and fail-closed.

Required recovery package schema name:

`contramamba-seed180-a0-provenance-recovery-v1`

Required recovery package destination:

`/kaggle/working/contramamba_recovery_handoffs/seed180_a0_<recovery-shortcommit>.zip`

Required recovery package contents:

- `recovery_manifest.json`
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json`
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json`
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl`
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/selected_checkpoint.pt`
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json`
- optional `forensics/ipython_history_authorized_exact_matches.txt`, only if
  the live Kaggle IPython history is legitimately available at recovery time.

The optional forensic history file must be labeled as forensic supporting
evidence. It must not be used as a substitute for missing wrapper-created
`run.meta`, `run.log`, `start.marker`, or `command.sh`.

The recovery collector must be CPU-only, must not import or execute the trainer,
must not perform model forward passes, must not evaluate metrics, must not
require GPU, and must not mutate the original seed180 output directory.

The recovery importer/auditor must validate the package without copying over or
modifying any existing local file unless a separate import authority explicitly
allows a new recovery-only audit location. It must preserve the distinction
between recovered artifact identity and scientific interpretation.

## Required Future Validation

Future recovery must fail closed unless every artifact path is a regular file
and every size/SHA256 exactly matches the immutable anchors in this candidate.

Future recovery must fail closed unless `run_provenance.json` independently
confirms at least:

- `schema_version = stage174a_v1`
- `status = completed`
- `source_provenance.git_commit =
  2737c3c6116ae3766b469801f990e2c45ba9a55e`
- `source_provenance.git_is_dirty = false`
- seed/training seed `180`
- resolved split seed `174`
- `architecture = v6b_minimal`
- `backbone = mamba`
- `model = state-spaces/mamba-130m-hf`
- `device = cuda`
- `freeze_encoder = true`
- reason-router `arm = A0`
- `router_mode = explicit_product`
- `gradient_ownership_mode = joint`
- resolved reason loss `0.0`
- dataset physical SHA256
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- dataset semantic SHA256
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- P4-L sidecar physical SHA256
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- P4-L sidecar semantic SHA256
  `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`
- P4-L provenance physical SHA256
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`
- completed epochs `20`
- selected epoch `20`
- prediction export row count `720`

Future recovery must perform the strongest feasible exact command validation:

- compare any `run_provenance.json` `raw_sys_argv`, `parsed_args`, and
  `command_string` fields against the seed180 trainer-only suffix from the
  frozen A0 authority;
- prove seed, split seed, A0 arm, router mode, gradient ownership mode, dataset
  path, sidecar path, expected sidecar semantic SHA256, output paths, objective
  neutralization flags, checkpoint-save flag, model/backbone/architecture,
  epoch count, device, learning rate, selection metric, class weighting, and
  omitted reason-loss behavior;
- compare the frozen full wrapper command identity to
  `dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e`;
- never claim unavailable historical wrapper metadata.

If a raw command string is absent from trainer provenance, future recovery may
still proceed only if all parsed identity fields and immutable artifact anchors
match exactly. The manifest must record that wrapper command provenance remains
missing and that validation relied on trainer-level provenance plus immutable
artifact hashes, not on standard `cm run` metadata.

## Future Command Hash Policy

No exact recovery execution command is authorized by this candidate because the
required recovery collector/import implementation does not yet exist.

When a future implementation exists, the recovery command must be a single-line
command encoded as UTF-8 with exactly one final LF byte for SHA256 calculation.
The hash must exclude Markdown fence bytes, display wrapping, leading blank
lines, trailing spaces, and extra final blank lines. Placeholder command hashes
are prohibited.

## Explicit Non-Authorizations

This candidate does not authorize:

- seed180 training rerun;
- seed180 resume;
- seed180 retry;
- deleting or moving seed180 outputs;
- modifying any file in the original seed180 run directory;
- fabricating historical `run.meta`, `run.log`, `start.marker`, or `command.sh`;
- weakening or bypassing current `cm collect` / `cm import`;
- seed181 or seed182 execution;
- A1, A2, or A3 execution;
- scientific interpretation;
- result promotion;
- modification of the original A0 authority freeze or history;
- modification of `C:\Users\Home1\.contramamba\cm.ps1`.

## Failure Semantics

Any mismatch in artifact hash, artifact size, required trainer provenance,
authority freeze, command identity, path identity, row count, epoch count, or
forensic-evidence labeling is a provenance recovery blocker.

No automatic repair is authorized. No metadata may be inferred, fabricated,
renamed into place, backdated, or overwritten.

## Next Required Stage

The next authorized action is independent verification of this candidate.

If that passes and the candidate is frozen, the next separate implementation
authority may specify a narrow recovery-only collector/import mechanism for
`contramamba-seed180-a0-provenance-recovery-v1`. That implementation stage must
remain CPU-only and must not train, evaluate, regenerate checkpoints, mutate
seed180 outputs, or alter standard `cm` handoff semantics unless explicitly and
separately authorized.
