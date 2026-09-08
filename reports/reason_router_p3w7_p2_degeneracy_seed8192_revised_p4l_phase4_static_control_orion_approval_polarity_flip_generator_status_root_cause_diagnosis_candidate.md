# Phase-IV Orion approval polarity-flip generator-status root-cause diagnosis

## Verdict

`PASS_READY_FOR_INDEPENDENT_PHASE_IV_ORION_APPROVAL_POLARITY_FLIP_GENERATOR_STATUS_ROOT_CAUSE_DIAGNOSIS_VERIFICATION`

This is a report-only, Git-canonical diagnosis. No checker, test, builder, dataset,
sidecar, provenance, training, evaluation, CUDA/GPU, Kaggle, staging, commit, or
push was performed during this diagnosis.

## Opening and frozen execution state

The required opening identity was authenticated: worktree
`C:\p3w7-a0-n3-validated-evidence-analysis`; branch
`p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`; HEAD and upstream tip
`ddb2ae9107e0938606036c24ce379888c5cb77af`; configured upstream
`origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`; ahead/behind `0/0`.
The historical pre-authoring repository state recorded by the frozen execution
evidence had tracked unstaged `0`, staged `0`, and nonignored untracked `0`.
The current post-authoring verifier state has tracked unstaged `0`, staged `0`,
and exactly one nonignored untracked file:
`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_orion_approval_polarity_flip_generator_status_root_cause_diagnosis_candidate.md`.

Frozen execution evidence is recorded without reinterpretation: pre-execution
authentication PASS; focused `pytest -q -o cache_dir=<external-cache-dir>
tests/test_reason_router_p4x_prelaunch_static_control.py` exited 0 with `86 passed,
1 skipped in 18.36s` and empty stderr; the repository was clean before the checker.
The standalone checker exited 1, had empty stdout, and stderr exactly
`{"contract": "P4X_GENERATOR_STATUS_DEFECT: orion_approval__polarity_flip", "status": "FAIL"}`.
Post-execution re-authentication PASS. The supplied classifications remain:
implementation delta correctness ESTABLISHED, historical and fresh focused-test
correctness ESTABLISHED, prelaunch static-control execution success NOT_ESTABLISHED,
and scientific conclusion NONE.

## Frozen artifact authentication

At HEAD, Git resolves checker blob `90b96a329cd6e92d3c5c79b76d02dc2d31233574`,
focused-test blob `e310dcdd72f9dc39ca281626ea20dd6bd19a7132`, dataset blob
`2b6829bf04a1333446aac6f7c603d9178b339f36`, sidecar blob
`83d119e327acacda7cff6b4e24c6502898294e03`, and provenance blob
`6c970033fae82286452f6d635b94f441d0f3d048`. Their required Git-canonical raw
SHA256 identities are respectively dataset
`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`, sidecar
`9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d`, and
provenance `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8`.
Frozen provenance records `builder_source_commit`
`149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b`, `builder_source_path`
`scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`, and
`builder_source_sha256` `a24e2c33ac7104dfac7fc8464e1c19e2ac7a77c99ff162b7fdf946066c4f9134`.
Independent Phase-II execution evidence authenticates producer Git blob
`a6f774c4ff79c3d047600d3e559c90c04768088c`, producer Git-canonical SHA256
`58bbda4c136323037868fc3b0b4a6d99d651932383bf6147624d34b0da207e6f`, and
generation-time producer worktree SHA256
`a24e2c33ac7104dfac7fc8464e1c19e2ac7a77c99ff162b7fdf946066c4f9134`.
Thus `a24e2...` is the independently corroborated generation-time
filesystem-byte identity, not the Git-canonical source SHA256; commit and path
authenticate Git lineage separately from the worktree-byte SHA. The frozen
revised P4-L authority `ff181f...` says its SHA binds revised builder source
bytes but does not itself explicitly define the byte domain. Later activated
Phase-II materialization/execution authority explicitly required measuring the
pre-execution producer worktree SHA and matching the generated provenance field
to it; the frozen execution record independently corroborates that relationship.

## Target extraction and consistency

Exactly one Git-canonical source row exists, at physical index 11 (zero-based):

```text
id=orion_approval__polarity_flip
pair_id=orion_approval
claim=Dr Mira Chen, the director, approved the Orion project in Seoul during Monday.
evidence=Dr Mira Chen, the director, did not approved the Orion project in Seoul during Monday.
final_label=REFUTE; frame_compatible_label=1; predicate_covered_label=1;
sufficiency_label=1; polarity_label=REFUTE; primary_failure_type=polarity;
intervention_type=polarity_flip
```

Exactly one sidecar row exists, also physical index 11:

```text
row_id=orion_approval__polarity_flip; pair_id=orion_approval;
canonical_row_id=orion_approval__none; split=train;
frame_compatible_label=1; p2_reason_supervision_eligible=false;
eligible_for_positive_margin=false; integrity_status=INELIGIBLE;
p2_primary_reason=AUTHORIZED; p2_reason_exclusion_codes=[P2_GENERATOR_STATUS_DEFECT];
reason_codes=[DEV_SPLIT_EXCLUDED,DID_NOT_INFLECTED_PREDICATE,GRAMMAR_TEMPLATE_FAIL,P2_GENERATOR_STATUS_DEFECT]
```

The sidecar schema contains only `frame_compatible_label`, not sidecar predicate or
sufficiency label fields; therefore no sidecar F/P/S triplet exists to compare. Its
present frame value equals source F=1. This is not the emitted contract: the checker
also only compares the frame field. Source/sidecar row id and pair id agree.

The seven checker fields are: `schema_status=PASS`, `dataset_source_status=PASS`,
`grammar_status=FAIL`, `canonical_status=PASS`, `intervention_contract_status=PASS`,
`polarity_contamination_status=PASS`, `time_swap_status=PASS`. Thus the sole target
non-PASS pair is **`grammar_status=FAIL`**. The checker’s exact predicate is
`all(side.get(field) == "PASS" for field in statuses)`; this frozen value completely
explains its named failure.

The target split is train, consistent with authenticated Seed8192 membership. Its
canonical source and sidecar target are both `orion_approval__none`; that canonical
source is same-pair, intervention `none`, self-anchored in the sidecar, and has
SUPPORT/1/1/1/SUPPORT/none semantics. The polarity-flip source has matching directional
REFUTE final and polarity labels and the required `polarity_flip` intervention. No
source-to-sidecar identity, pair, present-frame, split, canonical-link, or polarity
semantic mismatch was found.

## Producer and authority trace

The producer is the above builder, with Git lineage identified by commit
`149adf32...`; its authentication also distinguishes the Git-canonical source
identity from the generation-time physical worktree identity described above. Its bridge
copies historical `grammar_status`; when a P4-B compatibility row exists, it assigns
`PASS` only for compatibility `PASS`, and otherwise assigns `FAIL` for compatibility
`FAIL` (else `UNRESOLVED`). The relevant short fragment is
`bridged["grammar_status"] = "FAIL" if status == "FAIL" else "UNRESOLVED"`.
The target has historical reason `DID_NOT_INFLECTED_PREDICATE` / `GRAMMAR_TEMPLATE_FAIL`,
and the frozen bridge deterministically carries the resulting `FAIL`; polarity-flip
rows are not specially converted to PASS. The sidecar records that determination with
`P2_GENERATOR_STATUS_DEFECT` and makes the row ineligible.

The P4-L reconstruction/rebinding authority requires `grammar_status` be preserved or
rederived from the historical bridge plus P4-B compatibility, split-independently; it
does not require it be PASS. The P4-L artifact contract explicitly defines a
`P2_GENERATOR_STATUS_DEFECT` exclusion and treats the component statuses as sidecar
evidence. Conversely the Phase-IV checker unconditionally requires all seven status
fields PASS for every joined source row, before applying eligibility. The frozen
sidecar has 1,831 such non-PASS rows while its aggregate contract deliberately records
1,831 reason-ineligible rows. Therefore category **B** is established: producer output
is valid under its governing P4-L authority, but the Phase-IV checker applies an
incompatible all-row PASS requirement. No contradiction within the P4-L authority was
found, and no source-row defect is established.

## Checker order and historical masking

The corrected checker validates in this order: repository/anchor and Phase-II commit
identity; Phase-II frozen evidence binding; source/sidecar/provenance bytes and
provenance; split recomputation and exact `SPLIT_IDENTITIES`; stable join/canonical
setup; aggregate validation; then `_derive_cohorts(train, ...)`, followed only if it
succeeds by `_derive_cohorts(dev, ...)`. Generator-status validation is inside the
second source-order loop in each cohort derivation, after stable join and frame check.
The target is the first train source row with a non-PASS field, so it raises the stated
contract during **train** derivation; dev is not reached.

At historical implementation `dd34cd00336d04d384767fd533c33253d2c9c6ac`, dataset
and sidecar resolve to the same current blobs `2b6829...` and `83d119...`; the target
status values are therefore byte-identical. The historical checker blob is
`c49725202aac50e65b8b3dd7a1e0cbe53484047e`. Comparing it with the corrected blob shows
only identity-hash serialization and ordered split-row identity changes. The seven-field
all-PASS predicate and its placement in cohort derivation are unchanged. Historical
control flow reaches `P4X_SPLIT_IDENTITY_MISMATCH` before aggregate/cohort derivation.
It did not observe the generator defect.

`LATENT_DOWNSTREAM_BLOCKER_MASKED_BY_EARLIER_SPLIT_IDENTITY_FAILURE = ESTABLISHED`.

## Diagnostic scope and traversal

Read-only parsing finds 1,831 affected rows: failing-field counts are
`grammar_status: 212`, `canonical_status: 1650`,
`intervention_contract_status: 1350`, and `polarity_contamination_status: 1350`.
Exact non-PASS values are respectively `FAIL`, `UNRESOLVED`, `FAIL`, and `FAIL`.
Affected source primary-failure distribution is frame 900, polarity 300, sufficiency
300, none 181, predicate 150. Train/dev is 1,471/360.

Affected intervention counts are polarity_flip 300; paraphrase, entity_swap,
event_swap, location_swap, role_swap, title_name_swap, predicate_swap,
evidence_deletion, evidence_truncation, and irrelevant_evidence 150 each; and none 31.
The exact systematic affected-ID inventory is: every `generated_fact_151` through
`generated_fact_300` for each of those eleven non-`none` intervention forms; every
polarity-flip row for the first 150 pairs (the 30 named pairs and
`generated_fact_031` through `generated_fact_150`); plus `none` rows for generated
facts 151,153,155,158,161,165,166,172,174,176,186,195,207,210,221,225,227,230,238,
240,241,257,262,265,271,273,274,285,287,297,300. The 212 grammar failures comprise
the first 150 polarity-flips and `none` plus `paraphrase` for those last 31 IDs.

Thus the target is not isolated: it is the first defective row in actual train source
physical traversal (source and sidecar index 11); another offending train row follows
at index 23 and the first dev offender is `museum_purchase__polarity_flip` at index 35.

## Required classification and boundary

```text
IMMEDIATE_FAILURE_CAUSE = grammar_status=FAIL violates the checker all-seven-PASS condition
UPSTREAM_CAUSAL_ORIGIN = Phase-IV checker contract applies all-row PASS to P4-L-preserved ineligible statuses
DEFECT_LAYER = PHASE_IV_CHECKER
RECOMMENDED_DEFECT_LAYER_CLASSIFICATION = PHASE_IV_CHECKER_WITH_PROVENANCE_AUTHENTICATION_CAVEAT
DEFECT_SCOPE = MULTI_ROW_SYSTEMATIC
PRODUCER_PROVENANCE_IDENTITY = VALID_AND_AUTHENTICATED
CHECKER_CONTRACT_MISMATCH = ESTABLISHED
LATENT_DOWNSTREAM_BLOCKER_MASKED_BY_EARLIER_SPLIT_IDENTITY_FAILURE = ESTABLISHED
CODE_CORRECTNESS_CURRENT_HASH_SERIALIZATION = ESTABLISHED
FRESH_FOCUSED_TEST_CORRECTNESS = ESTABLISHED
PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = NOT_ESTABLISHED
SCIENTIFIC_CONCLUSION = NONE
REMEDIATION_AUTHORIZED = FALSE
CHECKER_RERUN_AUTHORIZED = FALSE
CANDIDATE_STATUS = PASS_READY_FOR_INDEPENDENT_PHASE_IV_ORION_APPROVAL_POLARITY_FLIP_GENERATOR_STATUS_ROOT_CAUSE_DIAGNOSIS_VERIFICATION
EXACT_NEXT_ACTION = FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_ORION_APPROVAL_POLARITY_FLIP_GENERATOR_STATUS_ROOT_CAUSE_DIAGNOSIS_VERIFICATION
```

Possible future work is a decision branch only: reconcile the Phase-IV all-row status
contract with the P4-L eligibility/exclusion contract, or establish a different
authority interpretation. This report selects neither and authorizes neither.

## Completion controls

`git diff --check` passed. No execution beyond read-only Git inspection and bounded
Python parsing was performed. No staging, commit, or push was performed. No scope
deviation occurred. Additional frozen non-PASS rows are reported above; they are the
remaining blocker together with the un-reconciled checker/P4-L contract.

The raw byte identity is reported from a post-write read-only calculation in the
delivery record; it is not staged.
