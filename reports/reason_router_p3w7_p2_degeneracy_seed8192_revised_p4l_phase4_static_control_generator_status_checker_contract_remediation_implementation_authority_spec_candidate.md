# Phase-IV generator-status checker-contract remediation implementation-authority candidate

## 1. Candidate decision and authority binding

**Candidate status:**
`PASS_READY_FOR_INDEPENDENT_PHASE_IV_GENERATOR_STATUS_CHECKER_CONTRACT_REMEDIATION_IMPLEMENTATION_AUTHORITY_VERIFICATION`

**Exact next action:**
`FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_GENERATOR_STATUS_CHECKER_CONTRACT_REMEDIATION_IMPLEMENTATION_AUTHORITY_VERIFICATION`

This is a report-only candidate. It authorizes no implementation unless an
independent verifier byte/blob freezes it, commits and pushes it, remotely
authenticates that commit, and activates it as the successor implementation
authority. It is bound to frozen diagnosis commit
`577d006a9a2a13bcc514ea8d00e70359ca478fd8`, parent
`ddb2ae9107e0938606036c24ce379888c5cb77af`, diagnosis path
`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_orion_approval_polarity_flip_generator_status_root_cause_diagnosis_candidate.md`, and diagnosis blob
`9853d2e10f60bf4c57ff2330fa703fdc2109e48a`.

The frozen classifications are exactly:

```text
CHECKER_CONTRACT_MISMATCH = ESTABLISHED
DEFECT_LAYER = PHASE_IV_CHECKER
DEFECT_SCOPE = MULTI_ROW_SYSTEMATIC
LATENT_DOWNSTREAM_BLOCKER_MASKED_BY_EARLIER_SPLIT_IDENTITY_FAILURE = ESTABLISHED
PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = NOT_ESTABLISHED
SCIENTIFIC_CONCLUSION = NONE
```

The frozen observed failure remains
`P4X_GENERATOR_STATUS_DEFECT: orion_approval__polarity_flip`; its immediate
status was `grammar_status = FAIL`. The frozen populations are
reason-ineligible `1831`, integrity `INELIGIBLE + UNRESOLVED` `1831`, and
any-seven-generator-status-non-`PASS` `1831`; these three row-ID sets are
exactly equal.

## 2. Authenticated implementation anchors and opening state

The opening repository state was exact: worktree
`C:\p3w7-a0-n3-validated-evidence-analysis`; branch
`p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`; HEAD
`577d006a9a2a13bcc514ea8d00e70359ca478fd8`; configured upstream
`origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`; upstream tip
`577d006a9a2a13bcc514ea8d00e70359ca478fd8`; ahead/behind `0/0`; tracked
unstaged `0`; staged `0`; and nonignored untracked `0`.

The current focused implementation anchors are checker blob
`90b96a329cd6e92d3c5c79b76d02dc2d31233574` for
`scripts/validate_reason_router_p4x_prelaunch_static_control.py`, and focused
test blob `e310dcdd72f9dc39ca281626ea20dd6bd19a7132` for
`tests/test_reason_router_p4x_prelaunch_static_control.py`. The corrected
split/hash implementation anchor remains
`8e637bdd439d62d429e1c019281efe398ee8c368`.

The current unconditional defect is in `_derive_cohorts`, at the all-seven
predicate `all(side.get(field) == "PASS" for field in statuses)`, which raises
`P4X_GENERATOR_STATUS_DEFECT` before sidecar reason eligibility is reconciled.
The correction is not deletion of that assertion. It is an independently
derived generator-status classification joined to independently derived source
semantics.

## 3. Exact future implementation scope

After this candidate becomes active, implementation may modify exactly:

```text
scripts/validate_reason_router_p4x_prelaunch_static_control.py
tests/test_reason_router_p4x_prelaunch_static_control.py
```

No other existing file may be modified. In particular, the implementation must
not modify trainer, dataset, sidecar, provenance, builder, execution records,
split authority, P4-L authority, frozen reports, A-series code, losses, labels,
gradient ownership, EMA, calibration, or model/checkpoint logic. The delta is
checker/test only; regeneration is not authorized.

## 4. Frozen status domain and fail-closed classification

The governing P4-L artifact contract and frozen builder
`scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
define `STATUS_ENUM` exactly as:

```text
PASS
FAIL
UNRESOLVED
NOT_APPLICABLE
```

The seven authenticated component fields remain exactly:

```text
schema_status
dataset_source_status
grammar_status
canonical_status
intervention_contract_status
polarity_contamination_status
time_swap_status
```

For each joined row, the future checker must require every field to exist, to
have type `str`, and to be one of that exact frozen enum. It must not normalize,
coerce, infer, or extend the enum.

Definitions are exact:

- `GENERATOR_CLEAN`: all seven authenticated component statuses are `PASS`.
- `GENERATOR_DEFECT`: one or more authenticated fields is `FAIL`.
- `GENERATOR_UNRESOLVED`: no field is `FAIL`, and one or more authenticated
  fields is `UNRESOLVED` or `NOT_APPLICABLE`.
- `MALFORMED`: a component field is missing, non-string, or outside the frozen
  enum. This must fail closed with an explicit checker contract error; it is
  never ordinary ineligibility.

`GENERATOR_DEFECT` and `GENERATOR_UNRESOLVED` must not produce an
unconditional `P4X_GENERATOR_STATUS_DEFECT` merely because their rows are
present. They are preserved P4-L exclusions from P2 reason supervision.

## 5. Independent reason-eligibility contract

The checker must preserve, and must independently rederive, source-semantic
eligibility from its existing source contracts: first-blocker consistency;
final-label requirements; polarity/directional consistency; polarity
intervention contract; source/sidecar identity; pair identity; split identity;
and frame binding. It may not trust the sidecar eligibility flag without that
derivation.

Conceptually:

```text
source_semantic_eligible = existing independently derived source-semantic result
generator_clean = all seven authenticated component statuses are PASS
expected_reason_eligible = source_semantic_eligible AND generator_clean
```

The checker must require both:

```text
type(side["p2_reason_supervision_eligible"]) is bool
side["p2_reason_supervision_eligible"] is expected_reason_eligible
```

Thus a source-semantically eligible row with a recognized non-PASS status and a
literal sidecar `false` is an authenticated preserved exclusion; the same row
with sidecar `true` fails closed as a reason-eligibility derivation mismatch.
An all-PASS, source-semantically eligible row with sidecar `false` also fails
closed unless an independently rederived exclusion explicitly authorized by the
frozen checker contract applies. Malformed status always fails closed before it
can be treated as an exclusion.

## 6. Cohorts and aggregates

Only `expected_reason_eligible` rows may enter reason/polarity cohort counts.
Rows excluded by recognized generator defects or unresolved statuses must not
enter those cohorts. Do not alter these frozen cohort constants:

| Split | Family | 0 | 1 |
| --- | --- | ---: | ---: |
| train | frame | 714 | 695 |
| train | predicate | 119 | 576 |
| train | sufficiency | 238 | 338 |
| train | polarity | 100 | 238 |
| dev | frame | 186 | 174 |
| dev | predicate | 31 | 143 |
| dev | sufficiency | 62 | 81 |
| dev | polarity | 19 | 62 |

Do not weaken, delete, or alter aggregate authentication. Preserve exactly:

```text
reason: true = 1769; false = 1831
integrity: ELIGIBLE = 1769; INELIGIBLE = 1562; UNRESOLVED = 269
margin: true = 695; false = 2905
```

The remediation must reconcile row-level eligibility with these frozen
aggregate contracts, rather than changing constants to pass tests.

## 7. Preserved identity, split, and provenance controls

Preserve the exact corrected split/hash semantics and the exact
`SPLIT_IDENTITIES` and `PROVENANCE_SPLIT_IDENTITIES`. Preserve LF-terminated
pair-list hashing and row hashing as `row_id<TAB>pair_id<LF>`. Preserve dataset,
sidecar, and provenance blob/SHA authentication; Phase-II activation/freeze
lineage checks; repository-cleanliness checks; and Git-canonical-byte checks.
The change must not alter source, sidecar, provenance, split authority, or
frozen artifact bytes.

## 8. Required future focused tests

The future patch must add focused coverage without weakening existing split,
provenance, Git identity, symlink, dirty-worktree, or first-blocker tests.

1. **A — preserved FAIL exclusion.** A source-semantically eligible synthetic
   row with one recognized `FAIL` and sidecar reason eligibility `false` does
   not fail merely for generator status and is absent from cohort counting.
2. **B — preserved unresolved exclusion.** A source-semantically eligible row
   with recognized `UNRESOLVED` or `NOT_APPLICABLE`, and literal sidecar `false`,
   is excluded without being classified clean.
3. **C — sidecar false negative.** All statuses `PASS`, source semantics
   eligible, and sidecar eligibility `false` fails the independently derived
   reason-eligibility contract.
4. **D — sidecar false positive.** A recognized non-PASS status and sidecar
   eligibility `true` fails that derivation contract.
5. **E — malformed status.** Missing, non-string, and unknown values each fail
   through an explicit checker contract error.
6. **F — clean cohort regression.** The existing all-PASS eligible fixture
   continues to derive identical applicable cohorts.
7. **G — frozen-artifact population semantics.** A bounded authenticated test
   establishes 1831 reason-ineligible rows, 1831 integrity non-`ELIGIBLE` rows,
   and 1831 rows with any seven-status non-PASS, with exact row-ID set equality.
8. **H — surfaced-row regression.** Authenticate
   `orion_approval__polarity_flip` as `train`, `grammar_status=FAIL`, and
   `p2_reason_supervision_eligible=false`, and establish that it is a preserved
   ineligible row, not an unconditional checker-contract failure.

## 9. Future implementation validation and independent verification

When and only when this candidate is activated, the implementation phase may
perform bounded implementation validation: targeted new/changed tests, then

```text
pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py
```

On Windows, pytest may use external `TEMP`, `TMP`, and `TMPDIR`, plus external
`-o cache_dir=<EXTERNAL_CACHE_DIR>`, so no repository-local pytest artifacts are
created under the frozen cache-isolation authority.

This authorization does **not** authorize a standalone production-checker run.
That remains separately gated until independent implementation verification,
corrected checker/test identity freeze, implementation commit/push remote
authentication, and explicit activation of a new execution-validation
authority.

The future implementation must have one bounded implementer followed by a
fresh, independent high-risk implementation verifier. The verifier must
independently confirm valid non-PASS exclusion rather than abort, malformed
fail-closed behavior, independent reason derivation, unchanged cohort and
aggregate constants, unchanged split/hash correction, no dataset/sidecar/
provenance mutation, and no scientific execution.

## 10. Scientific boundary and non-retroactivity

This candidate and any implementation authority derived from it prohibit A0,
A1, A2, A3, trainer execution, training, evaluation, model/checkpoint loading,
GPU, CUDA, and Kaggle. A future implementation gate PASS does not authorize
A-series execution. `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS` remains
`NOT_ESTABLISHED` until a separately authorized standalone checker execution
actually passes; `SCIENTIFIC_CONCLUSION` remains `NONE`.

All historical evidence is preserved. The historical
`P4X_SPLIT_IDENTITY_MISMATCH` and later
`P4X_GENERATOR_STATUS_DEFECT: orion_approval__polarity_flip` remain frozen, and
must not be rewritten as if the corrected checker existed. The established
latent-downstream-blocker classification remains unchanged.

## 11. Candidate completion record

Only this report is created by this report-only task. No checker, test, builder,
training, evaluation, GPU/CUDA/Kaggle, staging, commit, or push is authorized or
performed. The only remaining blocker is the required fresh independent
high-risk authority verification and its subsequent activation sequence.
