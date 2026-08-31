# P3-W7-A0 Seed180 Provenance Recovery Tooling External cm Drift Reconciliation Authority Specification Candidate

Authority/version:

`P3W7_A0_SEED180_PROVENANCE_RECOVERY_TOOLING_EXTERNAL_CM_DRIFT_RECONCILIATION_AUTHORITY_V1`

## Status

CANDIDATE ONLY.

This is a narrow authority-specification candidate for reconciling unrelated
external `cm.ps1` byte drift observed after the P3-W7-A0 seed180 provenance
recovery tooling implementation authority was frozen.

This candidate does not authorize implementation by existence alone. It does
not authorize training, evaluation, Kaggle execution, recovery collection,
recovery audit-import, seed180 rerun/retry/resume, checkpoint regeneration,
artifact mutation, A1/A2/A3, seed181/seed182 work, result promotion,
scientific interpretation, commit, push, standard `cm` behavior changes, or any
modification, restoration, dependency on, or reinterpretation of `cm.ps1`.

## Authority Chain

Authority precedence for this candidate:

1. current user instruction and current repository state;
2. frozen tooling implementation authority commit
   `cdd71ea4f556392eab594ebb5df8258355610e01`;
3. frozen tooling authority path
   `reports/reason_router_p3w7_a0_seed180_provenance_recovery_tooling_implementation_authority_spec_candidate.md`;
4. frozen provenance-recovery authority commit
   `233ed0be080e1d30dd47de2e66136475ec2ede76`;
5. original formal A0 execution freeze
   `2737c3c6116ae3766b469801f990e2c45ba9a55e`;
6. current repository `HEAD`
   `f7241abea9a09b54ff3b8ee66cacbd7f4feebb14`;
7. repository-wide `AGENTS.md` controls.

This candidate is created at current `HEAD`
`f7241abea9a09b54ff3b8ee66cacbd7f4feebb14`, which is three tracked commits
beyond `cdd71ea4f556392eab594ebb5df8258355610e01`.

## Repository Preconditions

This candidate may be created only when:

- `git rev-parse HEAD` is exactly
  `f7241abea9a09b54ff3b8ee66cacbd7f4feebb14`;
- tracked worktree is clean;
- index is clean;
- `cdd71ea4f556392eab594ebb5df8258355610e01` is an ancestor of current `HEAD`;
- post-`cdd71ea4f556392eab594ebb5df8258355610e01` tracked commits do not modify
  seed180 recovery authority or tooling implementation paths;
- the two recovery implementation files remain untracked:
  - `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`;
  - `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- protected unrelated untracked files are untouched.

Unrelated untracked files remain protected. Their presence alone is not a
blocker for this SPEC / AUTHORITY RECONCILIATION task.

## Post-cdd71ea Independence Assessment

Current `HEAD` is three commits beyond
`cdd71ea4f556392eab594ebb5df8258355610e01`:

- `7f52e21` freezes long-term O0a execution authority;
- `9a453dd` freezes O0a isolated Kaggle tooling recovery authority;
- `f7241ab` records long-term O0a native Mamba screening artifacts.

Static path inspection shows these commits add only unrelated O0a authority and
artifact files under `reports/longterm_o0a...` and do not modify:

- `reports/reason_router_p3w7_a0_seed180_provenance_recovery_tooling_implementation_authority_spec_candidate.md`;
- `reports/reason_router_p3w7_a0_seed180_provenance_recovery_execution_authority_spec_candidate.md`;
- `reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`;
- `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`.

These post-`cdd71ea` commits are unrelated O0a authority/artifact work and do
not change the seed180 recovery contract. This assessment does not inspect or
reinterpret the scientific results of those O0a artifacts beyond establishing
repository independence.

## External cm.ps1 Drift Facts

The tooling implementation authority was frozen at
`cdd71ea4f556392eab594ebb5df8258355610e01` while the external `cm.ps1`
identity was:

`94a4333037cf434b895fbee08e70dc1254b1e9ea233d6bad424dd4a4b34ecdaf`

The external `cm.ps1` was subsequently changed by a separate, unrelated
research lane.

The current observed external `cm.ps1` identity is:

`a8b818c22f6597b470fdab567378448837fc65f0fa6c75e381d602fb4bbc93b7`

This drift is external, mutable, out-of-scope state owned by other research
lanes. Resolving this drift here must not restore, modify, depend on, invoke,
or reinterpret `cm.ps1`.

## Current Recovery Implementation Facts

The current recovery implementation:

- does not call `cm.ps1`;
- does not call `cm import`;
- does not call `cm collect`;
- does not call `cm run`;
- does not modify standard `cm` behavior;
- does not impersonate `contramamba-handoff-v3`.

The recovery mechanism remains a separate schema:

`contramamba-seed180-a0-provenance-recovery-v1`

## Narrow Supersession

This candidate supersedes exactly one implementation-time precondition from the
frozen tooling implementation authority:

> `cm.ps1` must still have the historical SHA256 if inspected.

No other requirement from
`cdd71ea4f556392eab594ebb5df8258355610e01` is superseded.

Replacement rule:

- external `cm.ps1` is mutable, out-of-scope state owned by other research
  lanes;
- the current byte identity of external `cm.ps1` is not an implementation or
  verification precondition for this standalone recovery tool;
- recovery implementation and verification must not depend on any behavior or
  content of the current `cm.ps1`;
- recovery implementation and verification must not modify or restore
  `cm.ps1`;
- recovery tooling must not invoke `cm.ps1` or any `cm` command;
- any future proposal to integrate recovery with standard `cm` requires a
  separate authority and is not authorized here.

## Preserved Recovery Semantics

All other authority and recovery semantics remain unchanged, including:

- seed180 attempt disposition is `CONSUMED`;
- no seed180 rerun, retry, or resume is authorized;
- original execution success is `OBSERVED` only;
- standard `cm` wrapper provenance is `INCOMPLETE`;
- scientific conclusion is `NOT_ESTABLISHED`;
- original A0 execution commit is
  `2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- recovery authority commit is
  `233ed0be080e1d30dd47de2e66136475ec2ede76`;
- recovery schema remains
  `contramamba-seed180-a0-provenance-recovery-v1`;
- fixed source directory remains
  `/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0`;
- exact ZIP contract is preserved;
- provenance validation is preserved;
- trainer command validation is preserved;
- fail-closed semantics are preserved;
- no training or evaluation is authorized;
- no seed181 or seed182 work is authorized;
- no A1, A2, or A3 work is authorized;
- no scientific interpretation is authorized;
- no recovery execution authority is granted;
- implementation remains exactly the two previously authorized files:
  - `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`;
  - `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`.

The exact five source artifact identities remain unchanged:

| File | Size | SHA256 |
| --- | ---: | --- |
| `training_report.json` | 306114 | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` |
| `clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` |
| `training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| `selected_checkpoint.pt` | 518269815 | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` |
| `run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` |

This amendment does not authorize implementation by existence alone.

## Security And Non-Retroactivity

The historical finding remains unchanged: the original seed180 attempt did not
possess truthful standard `cm` wrapper provenance, and no present or future
change to `cm.ps1` may retroactively create that historical provenance.

Current `cm.ps1` behavior cannot be used as evidence of what happened during
the historical seed180 attempt. No current `cm.ps1` feature may synthesize or
repair historical `run.meta`, `run.log`, `command.sh`, `start.marker`, or
wrapper execution timestamps.

Standard wrapper provenance remains incomplete permanently for that historical
attempt. This separate recovery mechanism validates surviving artifact identity
and provenance only. This amendment does not weaken standard `cm` provenance
requirements.

## Future Implementation Continuation Transition

The current two-file recovery implementation may resume bounded remediation
only after:

1. this exact reconciliation candidate receives independent verifier PASS;
2. this exact reconciliation candidate is frozen in an immutable commit;
3. the controller explicitly transitions back to implementation remediation.

After that transition, bounded remediation scope is limited to the already
identified unresolved production defects:

A. Atomic path-replacement TOCTOU: after packaging, revalidate that each source
`PATH` still resolves to the same source object and identity expected by the
collection operation, so an atomic `os.replace` cannot silently make the
collection claim current source identity for a different path object.

B. Required provenance identities: fields required by the actual frozen
observed `stage174a_v1` schema, including dataset semantic identity and
selected-checkpoint identity, must not silently fall back or become optional.

No other implementation scope is granted.

Even after remediation PASS and implementation freeze, recovery collect and
audit-import execution remain not authorized until a later execution authority
or transition.

## Explicit Non-Authorizations

This candidate does not authorize:

- modifying, restoring, staging, committing, pushing, invoking, or depending on
  `C:\Users\Home1\.contramamba\cm.ps1`;
- changing standard `cm` behavior;
- implementation code changes;
- test changes;
- training;
- evaluation;
- Kaggle execution;
- recovery collection;
- recovery audit-import;
- seed180 rerun, retry, or resume;
- seed181 or seed182 work;
- A1, A2, or A3 work;
- scientific interpretation;
- result promotion;
- cleanup, reset, stash, checkout, rename, deletion, staging, or mutation of
  unrelated files.

## Candidate Transition Rule

The exact recommended next action after creating this candidate is independent
verification of this candidate.

Implementation remediation must wait until this candidate receives independent
verifier PASS, is frozen in an immutable commit, and the controller explicitly
transitions back to implementation remediation under the narrow scope stated
above.
