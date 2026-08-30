# P3-W7-A0 Seed180 Provenance Recovery Tooling Implementation Authority Specification Candidate

Authority/version:

`P3W7_A0_SEED180_PROVENANCE_RECOVERY_TOOLING_IMPLEMENTATION_AUTHORITY_V1`

## Status

CANDIDATE ONLY.

This file is an implementation-authority specification candidate for a future,
narrow, recovery-only collector plus local audit-import mechanism implementing
schema:

`contramamba-seed180-a0-provenance-recovery-v1`

This candidate does not authorize implementation by existence alone. It does
not authorize training, evaluation, Kaggle execution, recovery collection,
recovery audit-import, seed180 rerun/retry/resume, checkpoint regeneration,
artifact mutation, A1/A2/A3, seed181/seed182 work, result promotion,
scientific interpretation, commit, push, or modification of standard `cm`
behavior.

The implementation candidate becomes operative implementation authority only
after:

1. this exact candidate receives independent verification PASS;
2. the verified candidate is frozen in an immutable Git commit;
3. the user explicitly authorizes implementation from that frozen authority.

After implementation and validation, recovery execution is still not
automatically authorized. Recovery execution requires a later exact
recovery-execution authority or an explicit transition rule satisfying the
future execution boundary in this file.

## Authority Basis

Current repository HEAD required for creating this candidate:

`233ed0be080e1d30dd47de2e66136475ec2ede76`

Frozen seed180 provenance-recovery authority:

`233ed0be080e1d30dd47de2e66136475ec2ede76`

Frozen recovery authority path:

`reports/reason_router_p3w7_a0_seed180_provenance_recovery_execution_authority_spec_candidate.md`

Original formal A0 execution freeze:

`2737c3c6116ae3766b469801f990e2c45ba9a55e`

Original A0 authority path:

`reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`

Operational controls:

- `AGENTS.md`
- `docs/RESEARCH_OPERATIONS.md`

External `cm` implementation may be inspected read-only only:

`C:\Users\Home1\.contramamba\cm.ps1`

Expected external `cm` SHA256:

`94a4333037cf434b895fbee08e70dc1254b1e9ea233d6bad424dd4a4b34ecdaf`

This candidate preserves the frozen recovery-authority finding that current
standard `cm collect` and `cm import` cannot truthfully recover seed180 because
standard wrapper metadata is incomplete/missing. The future mechanism must be a
separate recovery schema and must not weaken, bypass, impersonate, or modify
`contramamba-handoff-v3`.

## Repository Boundary

Future implementation may proceed only if all of the following are true before
editing:

- `git rev-parse HEAD` is exactly
  `233ed0be080e1d30dd47de2e66136475ec2ede76`, or the later immutable freeze
  commit that contains this candidate after independent verification;
- tracked worktree is clean;
- index is clean;
- the frozen recovery authority path is present in the checked-out Git tree;
- the original A0 authority path is present in the checked-out Git tree;
- `C:\Users\Home1\.contramamba\cm.ps1` still has SHA256
  `94a4333037cf434b895fbee08e70dc1254b1e9ea233d6bad424dd4a4b34ecdaf` if it is
  inspected.

Unrelated untracked work is protected and out of scope. Its presence alone is
not a blocker, and future implementation must not inspect its scientific
content, modify it, stage it, delete it, move it, rename it, stash it, or depend
on it. Protected unrelated untracked work includes:

- the historical root `.patch` files;
- `reports/stage180a_pass2_annotations_completed.csv`;
- `reports/longterm_o0a_native_mamba_state_dynamics_authority_spec_candidate.md`;
- `scripts/observe_longterm_o0a_native_mamba_state_dynamics.py`;
- `tests/test_observe_longterm_o0a_native_mamba_state_dynamics.py`.

The future implementation-authority delta is exactly:

1. new script:
   `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`
2. new tests:
   `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`

No other repository file may be changed unless an independent verifier proves
an unavoidable requirement and a new authority is created first.

The implementation must not modify:

- `C:\Users\Home1\.contramamba\cm.ps1`;
- existing `cm run`, `cm collect`, or `cm import` behavior;
- the original A0 authority;
- the frozen recovery authority;
- trainer code;
- model code;
- dataset files;
- P4-L sidecar or provenance;
- original seed180 outputs;
- seed181 or seed182 state;
- A1/A2/A3 code or authority;
- protected parallel untracked files.

## Immutable Original Seed180 State

Freeze without reinterpretation:

- original execution commit:
  `2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- recovery authority freeze:
  `233ed0be080e1d30dd47de2e66136475ec2ede76`;
- seed: `180`;
- attempt disposition: `CONSUMED`;
- execution success: `OBSERVED` / trainer completed `20/20` epochs;
- standard `cm run` wrapper provenance: `INCOMPLETE / MISSING`;
- artifact/provenance validity: `NOT YET FORMALLY RECOVERED`;
- scientific conclusion: `NOT_ESTABLISHED`.

No future implementation may authorize rerun, retry, resume, replacement,
checkpoint regeneration, trainer execution, model forward execution, evaluation,
or mutation of original seed180 artifacts.

Authorized original wrapper SHA256:

`dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e`

## Script Interface

The future script must provide exactly two explicit subcommands:

- `collect`
- `audit-import`

There must be no default subcommand. Invoking the script without a subcommand
must fail closed with a nonzero exit status.

The script must use only the Python standard library unless a separately
verified implementation need proves otherwise. It must not import or transitively
require trainer, model, GPU, or evaluation dependencies.

## Collector Contract

The future `collect` subcommand is a recovery-only Kaggle-side CPU operation.

It must never import:

- `train_controlled_v6b_minimal`;
- model modules for forward execution;
- `torch`;
- `transformers`;
- `mamba_ssm`.

It must never:

- launch the trainer;
- load a checkpoint into a model;
- perform model forward;
- compute evaluation metrics;
- regenerate predictions;
- mutate any original seed180 artifact;
- require or initialize GPU/CUDA.

It must require an explicit literal:

`--expected-implementation-commit <40hex>`

It must fail closed unless `git rev-parse HEAD` exactly equals the expected
implementation commit. The implementation must reject malformed commit values.

It must require tracked worktree clean and index clean using checks that ignore
untracked evidence files:

- tracked cleanliness check equivalent to
  `git status --porcelain=v1 --untracked-files=no`;
- index cleanliness check equivalent to `git diff --cached --name-status`.

It must require the frozen recovery-authority file and original A0 authority to
be present in the checked-out Git tree.

It must read only this fixed original run directory:

`/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0`

It must not provide a CLI option that can redirect the source run directory.

It must require each source artifact to exist, be a regular file, not be a
symlink, have the exact frozen byte size, and have the exact frozen SHA256:

| File | Size | SHA256 |
| --- | ---: | --- |
| `training_report.json` | 306114 | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` |
| `clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` |
| `training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| `selected_checkpoint.pt` | 518269815 | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` |
| `run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` |

The collector must parse `run_provenance.json` fail-closed with duplicate JSON
key rejection and verify all identities required by the frozen recovery
authority, including:

- `schema_version = stage174a_v1`;
- `status = completed`;
- `source_provenance.git_commit =
  2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- `source_provenance.git_is_dirty = false`;
- seed and training seed `180`;
- resolved split seed `174`;
- `architecture = v6b_minimal`;
- `backbone = mamba`;
- `model = state-spaces/mamba-130m-hf`;
- `device = cuda`;
- `freeze_encoder = true`;
- reason-router `arm = A0`;
- router mode `explicit_product`;
- gradient ownership `joint`;
- effective reason loss `0.0`;
- completed epochs `20`;
- selected epoch `20`;
- prediction export row count `720`;
- dataset physical SHA256
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`;
- dataset semantic SHA256
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`;
- sidecar physical SHA256
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`;
- sidecar semantic SHA256
  `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`;
- P4-L provenance physical SHA256
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`.

The collector must perform exact trainer-command semantic validation against:

- `raw_sys_argv`;
- `parsed_args`;
- `command_string`;
- the frozen A0 seed180 trainer suffix from the original A0 authority.

It must verify all command semantics specified in the recovery authority,
including omission and effective-neutralization behavior. Required command
semantics include seed, split seed, A0 arm, router mode, gradient ownership,
dataset path, sidecar path, expected sidecar semantic SHA256, output paths,
objective-neutralization flags, checkpoint-save flag, model/backbone/
architecture, epoch count, device, learning rate, selection metric, class
weighting, and omitted reason-loss behavior.

It must not claim trainer argv is the same byte object as the authorized full
wrapper command. It must separately anchor the full original wrapper SHA256:

`dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e`

V1 must not require or implement IPython-history collection. The frozen recovery
authority treats that evidence as optional; this V1 excludes it to minimize
provenance surface.

Package destination must be outside the Git repository and constrained to:

`/kaggle/working/contramamba_recovery_handoffs/`

Package filename must be:

`seed180_a0_<implementation-shortcommit>.zip`

The collector must fail if the target already exists. It must not provide an
overwrite option. It must prefer `ZIP_STORED` unless a demonstrated correctness
reason requires otherwise; the 518 MB checkpoint must not be recompressed merely
for convenience.

## Recovery Manifest Contract

The recovery package must contain exactly these entries:

- `recovery_manifest.json`;
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json`;
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json`;
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl`;
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/selected_checkpoint.pt`;
- `files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json`.

The package must not contain standard wrapper artifacts:

- `run.meta`;
- `run.log`;
- `command.sh`;
- `start.marker`.

No extra entries are allowed.

Manifest schema must be exactly:

`contramamba-seed180-a0-provenance-recovery-v1`

The manifest must explicitly contain fields representing at least:

- schema;
- `recovery_scope = P3-W7-A0 seed180 provenance recovery`;
- `recovery_authority_commit =
  233ed0be080e1d30dd47de2e66136475ec2ede76`;
- implementation commit;
- `original_execution_commit =
  2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- `seed = 180`;
- `attempt_disposition = CONSUMED`;
- `execution_status = completed`;
- `standard_cm_wrapper_provenance = missing/incomplete`;
- `original_authorized_wrapper_sha256 =
  dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e`;
- `recovery_capture_created_at_utc`, clearly labeled in nearby field text as
  recovery capture time, not historical execution time;
- source `run_provenance.json` SHA256;
- source trainer Git commit;
- immutable artifact file table with path, size, and SHA256;
- `scientific_conclusion = NOT_ESTABLISHED`.

The manifest must not invent:

- historical wrapper start time;
- historical wrapper finish time;
- historical wrapper exit code;
- historical wrapper log hash;
- historical wrapper meta hash.

JSON parsing and writing must be deterministic enough for audit. Manifest
generation must use deterministic ordering for stable fields where practical.
Manifest parsing must reject duplicate keys.

## Audit-Import Contract

The future `audit-import` subcommand is local validation only.

It must require:

- `--zip <recovery.zip>`;
- `--expected-implementation-commit <40hex>`;
- `--audit-output <path>`.

It must fail closed unless the current repository `HEAD` exactly equals the
expected implementation commit. The implementation must reject malformed commit
values.

It must never invoke current `cm import`. It must never impersonate schema
`contramamba-handoff-v3`.

It must validate the ZIP container before reading artifact content:

- ZIP file exists and is regular;
- entry names are unique;
- exact allowlist only;
- no absolute paths;
- no `..`;
- no backslash path ambiguity;
- no symlink entries;
- no encrypted entries;
- no duplicate logical paths;
- no unexpected directories or files;
- anchored source file sizes prevent decompression-size ambiguity.

It must parse `recovery_manifest.json` with duplicate JSON key rejection.

It must verify every manifest semantic field and every packaged artifact
size/SHA256.

It must re-parse packaged `run_provenance.json` and repeat the same required
semantic identity checks as the collector. It must not trust collector
validation merely because the package exists.

It must validate trainer-command semantics independently again against
`raw_sys_argv`, `parsed_args`, `command_string`, and the frozen A0 seed180
trainer suffix.

It must not copy packaged seed180 artifacts into the Git repository.

`--audit-output` must resolve outside the Git repository. It must:

- fail if output already exists;
- create exactly one new recovery audit JSON;
- use exclusive creation and no overwrite;
- contain package SHA256, manifest SHA256, implementation/recovery/original
  commits, all artifact validation results, trainer provenance validation
  result, trainer command validation result, and final status;
- state `execution_success = OBSERVED`;
- state `recovered_artifact_identity = VALIDATED`;
- state `standard_cm_wrapper_provenance = INCOMPLETE`;
- state `scientific_conclusion = NOT_ESTABLISHED`.

It must not promote results.

## Security And Fail-Closed Requirements

Future implementation must:

- use `pathlib` and defensive path resolution;
- reject symlinks where relevant;
- reject duplicate JSON keys;
- reject malformed, nonfinite, or type-confused required values;
- reject unknown schema;
- reject extra ZIP entries;
- reject missing ZIP entries;
- reject commit mismatch;
- reject artifact size/hash mismatch;
- reject trainer-provenance mismatch;
- reject trainer-command mismatch;
- reject output collision;
- avoid broad exception swallowing;
- avoid automatic repair;
- avoid fallback to standard `contramamba-handoff-v3`.

It must never provide:

- `--force`;
- `--overwrite`;
- `--skip-validation`;
- `--ignore-hash`;
- equivalent weakening switches.

Every mismatch must be reported as:

`PROVENANCE_RECOVERY_BLOCKER`

Such a blocker is not automatically a model failure, training failure,
scientific failure, or scientific conclusion.

## Future Test Contract

The required future test file must use synthetic/temp fixtures. Tests must not
require the real 518 MB checkpoint, GPU, trainer, model forward, network,
Kaggle, or scientific evaluation.

Pass coverage must include at least:

- valid synthetic recovery source fixture;
- valid recovery ZIP;
- valid audit-import;
- audit output outside repo.

Fail coverage must include at least:

- wrong current implementation commit;
- tracked-dirty repo;
- staged-dirty repo;
- missing artifact;
- wrong artifact size;
- wrong artifact SHA256;
- symlink source artifact;
- malformed `run_provenance.json`;
- wrong source Git commit;
- dirty source provenance;
- wrong seed;
- wrong split seed;
- wrong arm/router/gradient ownership;
- nonzero reason-loss semantics;
- wrong dataset identity;
- wrong sidecar identity;
- wrong epoch count;
- wrong prediction row count;
- trainer argv/parsed-args/command mismatch;
- existing output ZIP collision;
- wrong manifest schema;
- duplicate manifest JSON key;
- ZIP traversal;
- ZIP absolute path;
- ZIP backslash ambiguity;
- ZIP symlink entry;
- ZIP encrypted entry;
- duplicate ZIP entry;
- missing ZIP entry;
- extra ZIP entry;
- packaged artifact tampering;
- `run_provenance.json` tampering;
- audit-output inside repo;
- audit-output collision.

Tests must require static verification that the recovery module imports none of:

- `torch`;
- `transformers`;
- `mamba_ssm`;
- `scripts.train_controlled_v6b_minimal`;
- `train_controlled_v6b_minimal`.

## Future Implementation Validation

The future implementation stage must run narrow validation:

```text
python -m py_compile scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py
pytest tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py
git diff --check
```

It must also perform exact static import inspection proving the recovery module
does not import:

- `torch`;
- `transformers`;
- `mamba_ssm`;
- `scripts.train_controlled_v6b_minimal`;
- `train_controlled_v6b_minimal`.

No trainer, evaluation, Kaggle, GPU, real checkpoint, or scientific test is
authorized by the implementation validation.

Because this is provenance/authority logic, future implementation requires an
independent verifier before freeze.

## Future Execution Boundary

Even after implementation PASS and implementation freeze, recovery execution is
not automatically authorized.

A later exact recovery-execution authority, or an explicit transition in the
frozen implementation authority, must require:

- implementation independent verification PASS;
- implementation immutable freeze;
- post-freeze local gates PASS;
- exact recovery command materialized after freeze;
- UTF-8 exact command bytes with exactly one final LF for SHA256;
- exact command hash verification;
- Kaggle checkout of exact implementation freeze;
- GPU off;
- source artifact collision/identity checks.

The implementation authority must not predict future implementation commit SHA
or future command SHA. Placeholders must not be treated as executable authority.

## Explicit Non-Authorizations

This candidate does not authorize:

- any code implementation by its mere existence;
- training;
- evaluation;
- seed180 rerun/retry/resume;
- seed181 or seed182;
- A1/A2/A3;
- Kaggle recovery execution;
- recovery collection;
- recovery audit-import;
- result promotion;
- scientific interpretation;
- modification of standard `cm` behavior;
- modification of `C:\Users\Home1\.contramamba\cm.ps1`;
- mutation of original seed180 outputs;
- cleanup/reset/stash of unrelated work.

## Candidate Transition Rule

Only after independent verification PASS and explicit user commit/push does the
frozen candidate become authority for the exact bounded implementation delta.

The next authorized action after this candidate is created is independent
verification of this candidate. Implementation must wait for a separate
authorized implementation step under the frozen authority.
