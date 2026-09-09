# A0 Seed180 R2 `cm import` UTC Timestamp Correction — Implementation Evidence Freeze Candidate

## Verdict and scope

`PASS_READY_FOR_INDEPENDENT_CM_IMPORT_UTC_TIMESTAMP_CORRECTION_IMPLEMENTATION_FREEZE`

This REPORT-ONLY implementation-evidence freeze records the independently verified external-helper postimage and its static validation evidence. It creates no authority to execute `cm import`, recovery collection, training, evaluation, Kaggle work, commit, or push.

## Authority lineage

| Item | Frozen identity / role |
| --- | --- |
| Activated implementation authority commit | `2cb78db66414bcd193dae0de1d80302b325886b7` |
| Implementation authority report | `reports/reason_router_p3w7_seed8192_a0_seed180_r2_cm_import_utc_timestamp_comparison_correction_implementation_authority_spec_candidate.md` |
| Parent execution-authority lineage (context only) | A0 seed180 r2 execution authority at `abd85a088c274678004432160625d42208112848` |
| Independent verification verdict | `PASS_READY_FOR_CM_IMPORT_UTC_TIMESTAMP_CORRECTION_IMPLEMENTATION_FREEZE` |

The helper is an external/local file, not Git-tracked. This repository report freezes its observed identity and evidence only; helper bytes themselves are not committed.

## Frozen helper identity and preimage lineage

| State | SHA256 | Bytes |
| --- | --- | ---: |
| Authenticated original preimage | `b15d70832e7c76c05fea6a9955bd199edcf9fb633fe0fe34266c44788260f570` | 86385 |
| Intermediate failed-verification postimage | `acac21892845113573d3aa17c317a77f7e72e2f3f15cd3306ea647594808ce36` | 91332 |
| Final independently verified remediation postimage | `09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae` | 91498 |

The resolved final-helper path is `C:\Users\Home1\.contramamba\cm.ps1`. Deterministic static byte observations at freeze authoring: no UTF-8 BOM; 2,442 LF line endings and zero CRLF line endings; final LF present. These observations identify the external local postimage but do not place its bytes under Git control.

## Implemented semantics frozen

- Raw manifest JSON is retained before `ConvertFrom-Json`.
- Timestamp extraction is JSON-aware through `System.Text.Json.JsonDocument`.
- The only selected manifest keys are ordinal, case-sensitive `started_utc` and `finished_utc`. Uppercase and mixed-case aliases do not satisfy either required key.
- Duplicate exact lowercase selected keys fail closed, as do non-string exact selected values.
- Each selected manifest and `run.meta` timestamp must lexically match `^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$` and pass invariant calendar/time validation using `TryParseExact("yyyy-MM-ddTHH:mm:ss'Z'", InvariantCulture, AssumeUniversal|AdjustToUniversal)`.
- Raw manifest and metadata timestamps are compared using `StringComparison.Ordinal`. No `DateTime.ToString()`, normalization, timezone conversion, or alternate timestamp representation participates in equality.
- Duplicate `STARTED_UTC` or `FINISHED_UTC` fields in `run.meta` fail closed.

## Independent verification evidence

The independent verifier recorded the exact final postimage before testing, all Q1–Q8 and A–P expected results below, a complete PowerShell parser result of zero errors, and unchanged post-test helper identity (`09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae`, 91498 bytes). The verifier also recorded a clean repository and no persistent test or backup artifacts. This freeze does not rerun those tests or execute the helper.

### Q1–Q8 selected-key freeze

| Case | Fixture | Required verified outcome |
| --- | --- | --- |
| Q1 | Exact lowercase `started_utc` plus `finished_utc` | PASS |
| Q2 | Uppercase-only `STARTED_UTC` plus `FINISHED_UTC` | FAIL CLOSED: required lowercase keys are absent |
| Q3 | Mixed-case-only `Started_Utc` plus `Finished_Utc` | FAIL CLOSED |
| Q4 | Lowercase `started_utc` plus uppercase `FINISHED_UTC` | FAIL CLOSED: exact lowercase `finished_utc` is absent |
| Q5 | Uppercase `STARTED_UTC` plus lowercase `finished_utc` | FAIL CLOSED: exact lowercase `started_utc` is absent |
| Q6 | Duplicate exact lowercase `started_utc` | FAIL CLOSED |
| Q7 | Duplicate exact lowercase `finished_utc` | FAIL CLOSED |
| Q8 | Non-string exact lowercase `started_utc` | FAIL CLOSED |

### A–P timestamp-validation freeze

| Case | Fixture | Required verified outcome |
| --- | --- | --- |
| A | Manifest `started_utc=2026-09-08T22:38:29Z`; `run.meta` `STARTED_UTC=2026-09-08T22:38:29Z` | PASS |
| B | Manifest `finished_utc=2026-09-08T22:41:44Z`; `run.meta` `FINISHED_UTC=2026-09-08T22:41:44Z` | PASS |
| C | One-second mismatch between a selected manifest timestamp and its metadata counterpart | FAIL CLOSED |
| D | Malformed timestamp | FAIL CLOSED |
| E | Impossible calendar/time value | FAIL CLOSED |
| F | Required timestamp absent | FAIL CLOSED |
| G | Timezone-less timestamp | FAIL CLOSED |
| H | `+00:00` offset representation instead of literal `Z` | FAIL CLOSED |
| I | Non-zero offset representation | FAIL CLOSED |
| J | Fractional-seconds timestamp | FAIL CLOSED |
| K | Duplicate exact lowercase manifest `started_utc` | FAIL CLOSED |
| L | Duplicate exact lowercase manifest `finished_utc` | FAIL CLOSED |
| M | Duplicate `STARTED_UTC` in `run.meta` | FAIL CLOSED |
| N | Duplicate `FINISHED_UTC` in `run.meta` | FAIL CLOSED |
| O | `en-US` culture with an exact valid canonical pair | PASS |
| P | `ko-KR` culture with the same exact valid canonical pair | PASS; identical semantic result to O |

Thus A, B, O, and P passed; C through N failed closed as required.

## Non-regression controls confirmed present

The independently verified implementation retained the following controls: registry run binding; registry command-SHA recomputation; expected/actual commit equality; local-HEAD equality; `command.sh`, `run.log`, and `run.meta` SHA256 binding; other semantic `run.meta` bindings; manifest schema; `file_count`; ZIP/path safety; duplicate artifact-path detection; artifact-path safety; artifact size/SHA256 checks; collision rejection; rollback; and the local import audit record.

The verified delta left `cm run`, `cm collect`, `START_MARKER`, and collector discovery unchanged.

## Explicit limits

This freeze does **not** validate seed180 artifacts, authorize real `cm import`, authorize recovery-collector execution, authorize retraining, authorize seed181, authorize A1/A2/A3, authorize calibration, or establish a scientific conclusion.

## Recovery consequence

Any later actual r2 import must use this exact helper postimage or a separately authorized successor, preserve every existing import guard, use an r2-compatible five-artifact recovery handoff, and satisfy the local `HEAD == handoff expected commit` check. Therefore, absent separate authority, the real import must execute from a worktree pinned to r2 execution commit `abd85a088c274678004432160625d42208112848`, not from the authority-report HEAD.

## Authoring boundary

Training/evaluation: NO. Real import: NO. Recovery: NO. Kaggle: NO. Commit/push: NO. The intended repository delta is exactly this one new report; no existing repository file and no external helper byte is modified.
