# Phase-IV Split-Identity Hash-Serialization Correction Implementation Authority Specification Candidate

## Status, phase, and authority

**Status:** `PASS_READY_FOR_INDEPENDENT_PHASE_IV_SPLIT_IDENTITY_HASH_SERIALIZATION_CORRECTION_IMPLEMENTATION_AUTHORITY_VERIFICATION`

This is a report-only correction implementation-authority specification candidate. It authorizes **no implementation**, tests, standalone-checker execution, training, evaluation, A0/A1/A2/A3, trainer, CUDA/GPU, Kaggle, model/checkpoint loading, dataset/sidecar/provenance regeneration, staging, commit, push, or scientific conclusion.

It is prospective only and is governed by frozen root-cause interpretation commit `182ff454ab44a134a33d6b7a15f16356afb2ed8e`, whose sole parent is activated failed execution-validation authority `436d37499fd66a7d3b67756246c60223aa32dc48`. Its sole changed path is `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_split_identity_hash_input_serialization_root_cause_interpretation_candidate.md`, blob `7605a17b7374e11a37685c38d1a01a7a92f54102`. Other frozen authorities are current implementation `dd34cd00336d04d384767fd533c33253d2c9c6ac`, revised split design `b4fbb5666d796161f95ae23612ce2448c25063ee`, and revised P4-L producer `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b`.

At authoring start, the required opening state was authenticated: worktree `C:\\p3w7-a0-n3-validated-evidence-analysis`; branch `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`; HEAD `182ff454ab44a134a33d6b7a15f16356afb2ed8e`; upstream `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` at that same SHA; ahead/behind `0/0`; tracked unstaged `0`; staged `0`; nonignored untracked `0`. Git-native enumeration was used without recursively traversing ignored pytest-cache directories.

## 1. Frozen classification and failed execution evidence

**PRIMARY:** `SPLIT_IDENTITY_HASH_INPUT_SERIALIZATION_MISMATCH`.

**SECONDARY:** `SPLIT_ALGORITHM_AUTHORITY_MISMATCH`.

**SECONDARY_SCOPE:** audit hash-input serialization layer only.

The exact frozen focused pytest evidence is `pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py`: `83 passed, 1 skipped`, exit `0`, stderr empty. Therefore `CODE_CORRECTNESS_EVIDENCE = ESTABLISHED` for frozen implementation `dd34cd00336d04d384767fd533c33253d2c9c6ac`.

The exact frozen standalone checker evidence is `python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head dd34cd00336d04d384767fd533c33253d2c9c6ac`: exit `1`, stdout empty, stderr exactly `{"contract": "P4X_SPLIT_IDENTITY_MISMATCH", "status": "FAIL"}`. Post-failure re-authentication passed. `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = NOT_ESTABLISHED`. This is not a scientific failure: `SCIENTIFIC_CONCLUSION = NONE`.

## 2. Frozen implementation, inputs, and immutable split identities

The future bounded implementation may modify exactly these two existing paths, and no third implementation/test path:

1. `scripts/validate_reason_router_p4x_prelaunch_static_control.py` — blob `c49725202aac50e65b8b3dd7a1e0cbe53484047e`, SHA256 `bcbf1818cfed1351077d2f3c2db809a70cc6953ef2d50adaa08d5b9e49a409b3`, 25070 bytes.
2. `tests/test_reason_router_p4x_prelaunch_static_control.py` — blob `f9c581d29c4333e55e32dbe8828c1730706be45a`, SHA256 `98fe97715a8b0f67cdb04b4ea8e2a0da5fbf3fe525d8ea5ef532e44ef59e61b5`, 33250 bytes.

The dataset is blob `2b6829bf04a1333446aac6f7c603d9178b339f36`, physical SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`, semantic SHA256 `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`. Provenance is blob `6c970033fae82286452f6d635b94f441d0f3d048`, physical SHA256 `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8`. Neither may change.

`SPLIT_IDENTITIES` must remain exactly: `pair_count=300`; `train_pair_count=240`; `dev_pair_count=60`; `train_row_count=2880`; `dev_row_count=720`; `pair_universe_sha256=41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2`; `shuffled_pair_sha256=ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55`; `train_pair_sha256=f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049`; `dev_pair_sha256=30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4`; `ordered_train_row_sha256=478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8`; `ordered_dev_row_sha256=7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4`.

`PROVENANCE_SPLIT_IDENTITIES` must be exactly those eleven keys plus only `historical_seed174_dev_pair_sha256=259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d`. Historical Seed174 remains provenance-only and absent from `SPLIT_IDENTITIES`. The six incorrect current-checker diagnostic hashes must not replace any frozen constant.

## 3. Exact prospective correction contract

The correction is limited strictly to audit hash-input serialization. It must not change the split algorithm, seed, dev ratio, train/dev membership, split constants, dataset, sidecar, provenance, equality semantics, trainer dependencies, or any model/checkpoint/torch/transformers/CUDA path. It must not use tolerance, subset/superset semantics, hard-coded PASS, fallback to historical Seed174, split reselection, unrelated refactoring, formatting sweeps, or unrelated symbol renaming.

It must additionally preserve: no label, loss, gradient, EMA, or calibration change; no dataset, sidecar, or provenance mutation; no split seed change; no dataset regeneration; and no provenance regeneration. These prohibitions apply to both future implementation scope and its validation authority unless superseded explicitly by a later authorized stage.

For every ordered pair-ID iterable, the hash bytes must be exactly equivalent to `"".join(f"{pair_id}\\n" for pair_id in pair_ids).encode("utf-8")`, then SHA256: exactly one ID per LF-terminated line, including a final LF for every nonempty input; no CR, JSON, other delimiter, prefix, or suffix. Apply this to sorted canonical pair IDs for the universe; the exact shuffled Seed8192 sequence; and canonical sorted selected train and dev pair IDs.

For selected rows in original dataset order, bytes must be exactly equivalent to `"".join(f"{row['id']}\\t{row['pair_id']}\\n" for row in rows).encode("utf-8")`, then SHA256. Every record is `id`, TAB, `pair_id`, LF. Do not hash row ID alone and do not sort selected rows after selection.

The minimal semantic delta is preferred. A conforming implementation may append an LF for each existing pair/list helper input and supply ordered-row values as `"{id}\\t{pair_id}"`, or introduce narrowly named pair-identity and ordered-row-identity helpers. Exact byte equivalence is mandatory; exact code spelling is not.

## 4. Mandatory future regression obligations

The later implementation authority must require focused tests proving all of the following:

A. Pair-list serialization includes its final LF.
B. Previous no-final-LF serialization differs and is not accepted as authoritative.
C. Authenticated current dataset pair universe reproduces `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2`.
D. Exact Seed8192 shuffled sequence reproduces `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55`.
E. Canonical sorted train pairs reproduce `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049`.
F. Canonical sorted dev pairs reproduce `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4`.
G. Dataset-order train rows encoded `id+TAB+pair_id+LF` reproduce `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8`.
H. Dataset-order dev rows encoded `id+TAB+pair_id+LF` reproduce `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4`.
I. Row-ID-only serialization differs from the authoritative ordered-row identity.
J. Genuine pair-universe or train/dev membership mutation still raises `P4X_SPLIT_IDENTITY_MISMATCH`.
K. Genuine row membership/order mutation still raises `P4X_SPLIT_IDENTITY_MISMATCH`, or the existing exact fail-closed split-identity contract reached by that path, without weakening validation.
L. `PROVENANCE_SPLIT_IDENTITIES` exact equality remains unchanged.
M. Historical Seed174 remains provenance-only and absent from `SPLIT_IDENTITIES`.
N. Existing focused protections remain covered: repository/upstream identity, lineage, cleanliness, canonical Git bytes, symlink rejection, dataset semantic validation, cohort and aggregate validation, provenance schema/flags, and trainer/model/CUDA isolation.

## 5. Execution boundary and evidence layers

This authoring task executes no tests. After activation, a future implementation authority may authorize narrowly targeted serialization unit tests and exactly `pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py`. It must prohibit repository-wide pytest, trainer tests/execution, standalone checker execution, training, evaluation, and A0/A1/A2/A3. Focused pytest during implementation may establish only `IMPLEMENTATION_DELTA_CORRECTNESS` and/or `FOCUSED_TEST_CORRECTNESS`; it does not establish `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS` and does not authorize any A-series work.

The future phase must report separately: `IMPLEMENTATION_DELTA_CORRECTNESS`; `FOCUSED_TEST_CORRECTNESS`; `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS`; `ARTIFACT_PROVENANCE_VALIDITY`; and `SCIENTIFIC_CONCLUSION`. Only a separately authorized standalone checker execution can establish `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS`; focused pytest can establish only the implementation/test correctness layer.

## 6. Non-retroactivity, verification, and lifecycle

This authority cannot retroactively make `dd34cd00336d04d384767fd533c33253d2c9c6ac` a successful static-control implementation or reinterpret its frozen `P4X_SPLIT_IDENTITY_MISMATCH` at activated authority `436d37499fd66a7d3b67756246c60223aa32dc48`. Any correction needs a new freeze identity and a later explicit execution-validation authority before checker rerun.

Before activation, a fresh independent high-risk verifier must establish exact authority lineage and root-cause freeze, checker/test-only scope, frozen-constant preservation, exact byte contract, no membership change authorization, no provenance relaxation, sufficient regressions, preserved execution boundary, and no A-series/training authority.

Required lifecycle: (1) author or correct this candidate; (2) fresh independent high-risk authority verification; (3) exact authority-candidate byte/blob freeze; (4) report-only stage; (5) cm ship; (6) implementation-authority activation commit; (7) push; (8) remote verification; (9) bounded checker/test implementation; (10) authorized targeted serialization tests as needed; (11) exact focused pytest: `pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py`; (12) independent high-risk implementation verification; (13) exact implementation byte/blob freeze; (14) implementation-only commit/push; (15) remote verification; (16) separate execution-validation authority authoring/activation; (17) only then standalone checker execution. That later execution-validation authority owns the exact checker command, execution preconditions, and whether a fresh focused pytest is required again. The implementation authority does not authorize post-freeze checker execution. No shortcut is authorized.

## 7. Candidate disposition

`SCIENTIFIC_CONCLUSION = NONE`.

Exact next action: `FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_SPLIT_IDENTITY_HASH_SERIALIZATION_CORRECTION_IMPLEMENTATION_AUTHORITY_VERIFICATION`.

Authoring this candidate authorizes no implementation.
