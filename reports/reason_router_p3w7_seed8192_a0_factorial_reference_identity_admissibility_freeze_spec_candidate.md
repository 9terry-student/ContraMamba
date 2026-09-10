# Seed8192 A0 factorial-reference identity/admissibility freeze — candidate

## Status and purpose

This is a report-only, static-provenance candidate authored at required HEAD
`c31f6614c17fe285ea87860c07d59f49d7efa8e2`. It closes only the missing
identity/admissibility layer for future same-seed Seed8192 A1/A2/A3 use of
the A0 `training_report_predictions.jsonl` reference artifacts. It does not
provision artifacts, authorize factorial execution, training, evaluation,
Kaggle, trainer changes, `cm.ps1` changes, scientific-artifact changes,
Seed180 r2 repair/reinterpretation, or historical split174 evidence.

AT_AUTHORING_TIME = CANDIDATE_ONLY
TRAINING_EVALUATION_ALLOWED = NO
KAGGLE_ALLOWED = NO
RUNTIME_PROVISIONING_ALLOWED = NO
FACTORIAL_EXECUTION_ALLOWED = NO

## Frozen authority lineage and membership

The exclusive primary A0 N=3 membership is frozen by the clean-replacement
execution authority, commit `55debe94f0d19d16a334395e8561901fed6b52fa`,
report `reports/reason_router_p3w7_seed8192_a0_n3_clean_replacement_execution_authority_spec_candidate.md`,
blob `2fe0301a60243504a8e9a4d5d4cd2e8959da80ec`: seed180 replacement_r1,
seed181, and seed182. Its validated-evidence analysis is commit
`dd183f59f4040405c178da193fe99c7c7f3ef57f`, report
`reports/reason_router_p3w7_seed8192_a0_n3_validated_evidence_analysis_report_candidate.md`,
blob `6ff707c9807fa03aa9be51aba5bfb3d573b71343`.

A0_PRIMARY_N3_MEMBERSHIP_AUTHORITY = 55debe94f0d19d16a334395e8561901fed6b52fa
A0_N3_VALIDATED_EVIDENCE_AUTHORITY = dd183f59f4040405c178da193fe99c7c7f3ef57f
SEED180_R2_PRIMARY_ADMISSION = EXCLUDED
SEED180_R2_STANDARD_CM_WRAPPER_PROVENANCE = INCOMPLETE
SEED180_REPLACEMENT_R1_PRIMARY_ADMISSION = ADMITTED
R2_TO_REPLACEMENT_RELOCATION_CLAIM = NONE

The Seed180 recovery bridge (`4d26d5601b10714d0158049357d600901274a054`,
blob `e1f35f6d9cf6eec306f891aa3b30824f87f9749a`) and recovery handoff
(`48d258fc5e8b09e163e9252f33c86936244ef872`, blob
`7bef12bcc38d6b0b45feda49d4b9b9cb9701d2f7`) are consumed only for the
historical disposition of excluded r2. They do not establish copying,
relocation, or materialization of replacement_r1 from r2.

## Common frozen data and split binding

All admitted members bind to dataset
`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`,
Git blob `2b6829bf04a1333446aac6f7c603d9178b339f36`, canonical SHA256
`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`,
and semantic SHA256
`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`.
They bind to sidecar
`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl`,
semantic SHA256 `2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9`.
The split is `split_seed=8192`, `dev_ratio=0.2`, `train_rows=2880`,
`dev_rows=720`, ordered-train-row SHA256
`478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8`,
and ordered-dev-row SHA256
`7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4`.

SEED8192_DATA_SPLIT_BINDING=PASS
HISTORICAL_SPLIT174_REFERENCE_ADMISSIBLE=NO

## Local companion revalidation

Read-only revalidation found all three directories present. Each
`run_provenance.json` records its matching training seed, split seed 8192, the
dataset and sidecar above, `reason_router_arm=A0`,
`reason_router_mode=explicit_product`, `gradient_ownership_mode=joint`,
`reason_loss_weight=0.0`, completed status, and clean frozen execution
provenance at commit `55debe94f0d19d16a334395e8561901fed6b52fa`.

| Member | Directory | clean_dev_predictions.json | run_provenance.json | training_report.json |
| --- | --- | --- | --- | --- |
| seed180 replacement_r1 | `reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0` | 4840320 / `5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d` | 69444 / `a758538e93e6e52ca261cb593285c298344808a3626eed7d9b9664e29a6c1a3d` | 306325 / `2cdf0925e3a0ef1b925f6b00ac4b2095d18a113896a437ded77622f5134b2013` |
| seed181 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed181/A0` | 4838584 / `789d02f9092ce6b051d0ca435272c9e93a3962183dbb0a4d4dbb20cebf2ac3fe` | 69133 / `82f6a511f9b8228c91d3419cc8872f7371c0785a01aa730904d43a8f04f6a98b` | 306101 / `0068aec52a9afb4bd8e79d711ac666a5257186ab142c8c72da890fe64c2c45e8` |
| seed182 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed182/A0` | 4841685 / `029ec6ae31df2f5ca9526d1e631496f7ee272967a6f5e08684f29aa09ad490d4` | 69143 / `934acab332773b4127ffa5c68b09a8bded168ab96ac5b18c29018e3e78b77c66` | 306108 / `ef60c457a28be8ed91e57ef8d9be4d89150b0e93fe90a045eb12a6ea78262a22` |

A0_ADMITTED_MEMBER_COMPANION_EVIDENCE_REVALIDATED=PASS

## Exact admissible A0 reference identities

The canonical reference artifact type is `training_report_predictions.jsonl`.
Each local file below was readable, parsed as exactly 720 JSONL rows, and
matched the listed direct byte count and SHA256.

| Seed / admitted member | Exact destination path | Bytes | SHA256 |
| --- | --- | ---: | --- |
| 180 / seed180 replacement_r1 | `reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0/training_report_predictions.jsonl` | 3937018 | `80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334` |
| 181 / seed181 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed181/A0/training_report_predictions.jsonl` | 3935282 | `d7a2d79091e2b076610d58b6e3539a347c29706b796c9364b6998c5995b42472` |
| 182 / seed182 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed182/A0/training_report_predictions.jsonl` | 3938383 | `ae1a1dfd9a844437e032a506716abb68544b16ea1fcdff7eaba14a040a48e650` |

SEED180_A0_REFERENCE_EXACT_IDENTITY=PASS
SEED181_A0_REFERENCE_EXACT_IDENTITY=PASS
SEED182_A0_REFERENCE_EXACT_IDENTITY=PASS
A0_REFERENCE_ARTIFACT_TYPE = training_report_predictions.jsonl

The historical excluded r2 artifact at
`reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0/training_report_predictions.jsonl`
has the same stated bytes and SHA256 as the admitted seed180 replacement_r1
artifact. That byte equality is content equality only: it does not establish
source/destination provenance. No frozen evidence proves r2-to-replacement_r1
copying or relocation, and this report creates no such claim. replacement_r1
is admissible because frozen authority independently admits replacement_r1 and
its wrapper/import/member provenance; this report freezes its JSONL at its own
exact destination path. The excluded r2 path remains inadmissible for future
factorial consumption.

SEED180_REFERENCE_PROVENANCE_BASIS = INDEPENDENTLY_ADMITTED_REPLACEMENT_R1
SEED180_R2_BYTE_EQUALITY_INTERPRETATION = CONTENT_EQUALITY_ONLY_NOT_RELOCATION_PROVENANCE
SEED180_R2_REFERENCE_CONSUMPTION_ALLOWED = NO

## Future factorial consumption contract

Same-seed binding is mandatory: seed180 A1/A2/A3 consume the admitted seed180
replacement_r1 JSONL above; seed181 A1/A2/A3 consume the seed181 JSONL above;
and seed182 A1/A2/A3 consume the seed182 JSONL above. Cross-seed substitution,
historical Seed180 r2 substitution, and historical split174 A0 references are
forbidden. The following are not substitutes: `clean_dev_predictions.json`,
`training_report.json`, `run_provenance.json`, `selected_checkpoint.pt`,
standalone metrics, logits, or historical references.

SAME_SEED_A0_REFERENCE_BINDING_REQUIRED = TRUE
CROSS_SEED_REFERENCE_SUBSTITUTION_ALLOWED = FALSE
HISTORICAL_R2_SUBSTITUTION_ALLOWED = FALSE
HISTORICAL_SPLIT174_SUBSTITUTION_ALLOWED = FALSE

## Scope layers, clean runtime, and handoff

Layer 1, A0 member scientific/execution provenance, is previously frozen by
the N=3 authorities. Layer 2, exact future factorial-reference path, bytes,
and SHA256, is frozen by this report. Layer 3, availability of these exact
untracked bytes in a future clean Kaggle checkout, is not solved or authorized
here. Layer 4, A1/A2/A3 factorial execution authority, is not authorized here.

A0_MEMBER_PROVENANCE_VALIDITY = PREVIOUSLY_FROZEN
A0_REFERENCE_EXACT_IDENTITY_FREEZE = THIS_REPORT
FUTURE_RUNTIME_REFERENCE_PROVISIONING = NOT_AUTHORIZED_HERE
FACTORIAL_EXECUTION_AUTHORIZED = NO
FUTURE_CLEAN_RUNTIME_HAS_A0_REFERENCES = NO

All three exact reference files are untracked and have no Git blob at the
authoring HEAD. After this identity freeze is independently verified and
frozen, the exact next stage is Seed8192 factorial A0-reference runtime
provisioning authority authoring. That authority must consume this frozen
identity report and specify exact-byte/path, collector-clean provisioning into
a clean pinned factorial runtime before `cm run`; this report does not design
or authorize the mechanism. The calibration aggregate provisioning authority
at `21403f5e6cff6ca813c6df127c7ee0295998c597` is precedent only and is not
reused directly for A0 prediction references.

CALIBRATION_PROVISIONING_AUTHORITY_REUSED_DIRECTLY = NO
FACTORIAL_A0_REFERENCE_PROVISIONING_AUTHORITY_REQUIRED = YES
FACTORIAL_CANDIDATE_STATUS = BLOCKED_PENDING_A0_REFERENCE_PROVENANCE_AND_RUNTIME_PROVISIONING

## Freeze semantics and scientific boundary

This candidate becomes the frozen exact A0 factorial-reference
identity/admissibility authority only after all of the following: independent
high-risk provenance verification PASS; exact candidate bytes, SHA256, and Git
blob are frozen; exactly this candidate is explicitly staged; the user creates
a dedicated commit; push succeeds; and remote commit/blob identity is
authenticated. A commit subject alone is never authority; authority comes from
the frozen report body. No execution is authorized by freezing this document.

This report makes no new scientific claim: it does not establish A1/A2/A3
execution success, improvement over A0, effect of reason supervision, causal
mechanism, promotion, model-quality improvement, or statistical significance.

SCIENTIFIC_CONCLUSION_FROM_THIS_REPORT = NONE

PASS_READY_FOR_INDEPENDENT_SEED8192_A0_FACTORIAL_REFERENCE_IDENTITY_ADMISSIBILITY_FREEZE_VERIFICATION
