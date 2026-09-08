# A0 Seed180 R2 `cm import` UTC Timestamp Comparison Correction — Implementation Authority Specification Candidate

## 1. Verdict and authority

`PASS_READY_FOR_INDEPENDENT_CM_IMPORT_UTC_TIMESTAMP_COMPARISON_CORRECTION_AUTHORITY_VERIFICATION`

This is a REPORT-ONLY / IMPLEMENTATION-AUTHORITY SPECIFICATION candidate. It authorizes no implementation, import, recovery collection, Kaggle action, training, evaluation, checkpoint loading, commit, or push by its creation. It is limited to correcting the lossless UTC timestamp-comparison defect in the external local helper's `cm import` path for a valid `contramamba-handoff-v3` package. It does not establish seed180 artifact validity and does not authorize seed181.

Authority applied: current controller instruction; the active A0 execution authority at `reports/reason_router_p3w7_seed8192_revised_split_a0_execution_authority_spec_candidate.md` (required authority commit `abd85a088c274678004432160625d42208112848`); the read-only external helper; `AGENTS.md`; and `docs/RESEARCH_OPERATIONS.md`. The controller-named literal files `01_WORKFLOW.md`, `04_KAGGLE_RUNBOOK.md`, `05_FAILURE_RECOVERY.md`, and `06_NAMING_AND_PROVENANCE.md` were absent both in the repository and in `C:\Users\Home1\.contramamba` at authoring; no permission or behavior is inferred from their absence. The existing seed180 recovery authority was additionally read as compatible context.

## 2. Current identity freeze

| Item | Frozen value |
| --- | --- |
| Repository branch | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` |
| Repository HEAD | `abd85a088c274678004432160625d42208112848` |
| Helper path | `C:\Users\Home1\.contramamba\cm.ps1` |
| Helper SHA256 | `B15D70832E7C76C05FEA6A9955BD199EDCF9FB633FE0FE34266C44788260F570` |
| Helper byte count | `86385` |
| Relevant implementation identity | `cm import`, lines 1732–1759 (manifest parse/extraction), 1787–1790 (only nonempty timestamp check), 1964–1999 (metadata parse and comparison) |

The future implementer must stop before editing if the helper's SHA256, byte count, or the cited control-flow identity differs. A differing helper requires a new authority, not adaptation by analogy.

## 3. Exact defect and static reproduction

The current helper parses `manifest.json` with `ConvertFrom-Json` (lines 1732–1736). For ISO `Z` timestamp JSON strings, the available PowerShell implementation coerces the values to `System.DateTime`. Lines 1757–1758 then explicitly cast those values to `[string]`. Lines 1994–1995 compare those locale-dependent strings to raw strings independently extracted from authenticated `run.meta`.

Relevant current logic identity:

```powershell
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
...
$startedUtc  = [string]$manifest.started_utc
$finishedUtc = [string]$manifest.finished_utc
...
$metaValues[$matches[1]] = $matches[2]
...
[string]$metaValues["STARTED_UTC"] -ne $startedUtc -or
[string]$metaValues["FINISHED_UTC"] -ne $finishedUtc
```

Static-only reproduction, actually run against the local PowerShell parser (not `cm import`):

```text
JSON: { "started_utc": "2026-09-08T22:38:29Z", "finished_utc": "2026-09-08T22:41:44Z" }
type_started=System.DateTime
string_started=09/08/2026 22:38:29
type_finished=System.DateTime
string_finished=09/08/2026 22:41:44
meta_started=2026-09-08T22:38:29Z
comparison=False
```

Thus the authenticated identical raw values `STARTED_UTC=2026-09-08T22:38:29Z` and `FINISHED_UTC=2026-09-08T22:41:44Z` fail solely because `DateTime.ToString()` is selected implicitly by `[string]`, not because the instant or wrapper metadata differs.

## 4. Selected correction semantics and frozen grammar

**Selected approach: preserve and compare raw canonical strings.** The implementation must retain the raw JSON string values for `started_utc` and `finished_utc` before any date-coercing deserialization, strictly validate those raw strings, strictly validate the corresponding raw `run.meta` values, then compare each pair with ordinal string equality. It must not compare a `DateTime.ToString()` result and must not normalize, reinterpret, or reformat a timestamp before the equality check.

This is narrower and safer than accepting general offset-aware values: v3's collector emits canonical second-precision UTC `Z` strings, and exact authenticated equality verifies the representation as well as the instant. It preserves the original ISO UTC values losslessly and avoids locale, culture, and daylight-saving behavior entirely.

The only accepted grammar for all four timestamp values is exactly:

```text
^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$
```

After lexical acceptance, the implementation must use culture-invariant exact parsing solely to reject impossible calendar/time values, with format `yyyy-MM-ddTHH:mm:ss'Z'`, invariant culture, and UTC semantics. No fractional seconds, whitespace, timezone-less value, lowercase `z`, local timezone, offset (`+00:00` included), or non-UTC offset is authorized. Canonical raw strings—not the parsed values—are compared ordinally.

## 5. Authorized minimal implementation shape

The implementation delta is limited to `C:\Users\Home1\.contramamba\cm.ps1`, inside the v3 `cm import` manifest/timestamp and `run.meta` timestamp-validation/comparison region. It must:

1. retain the original manifest JSON text in a variable before existing normal manifest parsing;
2. obtain the raw JSON string tokens for exactly `started_utc` and `finished_utc` without date coercion and reject absent, non-string, or duplicate selected manifest keys;
3. apply the frozen grammar and exact UTC semantic validation to those two raw manifest strings;
4. while retaining the existing metadata behavior for all other fields, reject duplicate `STARTED_UTC` or `FINISHED_UTC` metadata fields; require both fields exactly once; apply the same grammar/semantic validation to their raw values; and
5. replace only the two timestamp comparisons with ordinal raw-string equality.

The raw-token reader must be JSON-aware (or an equally bounded parser that demonstrably distinguishes JSON string tokens and duplicate selected keys); regex extraction over arbitrary JSON text is not authorized. It must not change the v3 manifest schema, permit alternate timestamp representations, or make timestamp equality conditional on a culture. The existing `ConvertFrom-Json` object may remain for all non-timestamp fields if the raw timestamp reader is separate and agreement is enforced; it must not be used as the source of compared timestamp strings.

Fail closed before any copy for: malformed JSON/timestamp; missing selected timestamp; timezone-less timestamp; any offset-form timestamp; duplicate selected manifest timestamp key; duplicate `STARTED_UTC` or `FINISHED_UTC` in `run.meta`; impossible date/time; or unequal canonical raw strings. An unequal instant, including a one-second difference, remains an import block.

## 6. Non-regression invariants — must not change

The implementation must preserve byte-for-byte where practical, and behaviorally without weakening, the present validations for: registry run-name binding; registry command SHA256 and its recomputation; expected/actual commit equality; local HEAD equality; `command.sh`, `run.log`, and `run.meta` SHA256; semantic wrapper metadata binding other than the corrected timestamp representation path; manifest schema; `file_count`; artifact path safety; artifact size; artifact SHA256; ZIP member/path safety; and local collision rejection.

It must not modify `cm run`, `cm collect`, `START_MARKER`, collector discovery, original r2 wrapper files, the run registry, artifacts, selected checkpoint deserialization behavior, trainer/data/split/seed/scientific outputs, or any repository source/test/authority. It must not import the zero-file ZIP or execute the recovery collector. This correction does not weaken standard provenance requirements or convert direct-trainer recovery into a standard v3 handoff.

## 7. Required implementation validation

After a separately authorized implementation, validation must be static/unit-level and must not import a real recovery package, run a collector, train, evaluate, or use Kaggle. It must include focused tests (or a deterministic isolated PowerShell harness covering the timestamp helper) that prove:

| Case | Required result |
| --- | --- |
| `2026-09-08T22:38:29Z` / `2026-09-08T22:41:44Z` exact manifest-to-meta r2 pairs | pass timestamp validation/equality |
| Either raw metadata timestamp differs by one second | fail closed |
| malformed or impossible timestamp | fail closed |
| missing timestamp | fail closed |
| timezone-less or any offset timestamp | fail closed |
| duplicate selected manifest key or duplicate timestamp metadata field | fail closed |
| same inputs under at least two cultures (including `en-US` and `ko-KR`) | identical result; no locale conversion affects equality |
| existing positive/negative fixtures for registry, commit, wrapper hashes, schema, file count, ZIP/path, artifact hash/size, and collision | unchanged result |

Required commands, once implementation is explicitly authorized, are: `git diff --check`; the narrow timestamp/import regression test command added or identified by that implementation; and the existing narrow `cm import` validation tests/fixture suite, if present. The verifier must record commands actually run and must separately state that no real import, recovery collector, training, evaluation, or Kaggle execution occurred.

## 8. Implementation scope, verifier, and stop conditions

Before any implementation: re-read this frozen report, authenticate the exact helper identity above, inspect the exact import region, inspect all relevant existing helper/import tests, and confirm a clean repository state. The implementation must be independently reviewed by a provenance-focused verifier who did not author the patch. That verifier must inspect the diff, independently reproduce the original coercion, confirm raw-string/ordinal comparison and frozen grammar, run the mandated gates, and explicitly attest that each non-regression invariant remains enforced.

Stop and create no patch if: the stated defect cannot be reproduced from this exact helper; raw canonical string preservation cannot be implemented without a broader parser/schema change; the grammar is ambiguous; helper identity differs; any validation would be weakened; the proposal would accept offsets or timezone-less values; `cm collect`/recovery behavior would change; any file outside the single helper would need modification; or an actual seed180 artifact validation/import or seed181 action would be required.

## 9. Authoring confirmation

Intended new report path:

`reports/reason_router_p3w7_seed8192_a0_seed180_r2_cm_import_utc_timestamp_comparison_correction_implementation_authority_spec_candidate.md`

Authoring is static inspection only. Training/evaluation: `NO`. Kaggle execution: `NO`. Commit/push: `NO`. This report creates exactly one new repository file and modifies no existing repository file. Final authority status remains:

`PASS_READY_FOR_INDEPENDENT_CM_IMPORT_UTC_TIMESTAMP_COMPARISON_CORRECTION_AUTHORITY_VERIFICATION`
