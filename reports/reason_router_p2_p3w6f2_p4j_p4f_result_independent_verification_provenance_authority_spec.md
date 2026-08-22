# P3-W6-F2-P4-J P4-F Result Independent Verification Provenance Authority Specification

Authority/version:

`P3W6F2P4J_P4F_RESULT_INDEPENDENT_VERIFICATION_PROVENANCE_AUTHORITY_V1`

Decision scope: specification-only provenance-authority repair for recording,
freezing, independently verifying, and consuming a P4-F Level-3 admission
primary result.

This document is a candidate authority specification. It becomes canonical
only after this P4-J specification independently verifies PASS and is frozen
at an immutable Git commit. It does not claim that any token or provenance
contract newly defined here was historically canonical.

## A. Exact Blocker Being Repaired

P4-J resolves exactly this blocker:

`P4F_RESULT_INDEPENDENT_VERIFICATION_PROVENANCE_CONTRACT_UNRESOLVED`

The current P4-F primary admission result has an independently reported
substantive verification finding:

`P4-F primary result substantive verification = PASS`

P4-J preserves that substantive finding as historical evidence only. It does
not claim that P4-F is already final independently verified authority, does
not backfill the prior verifier run into an attestation, and does not create
the verification attestation.

P4-J does not reopen, re-evaluate, or re-adjudicate:

- Gate-6;
- P4-B;
- P4-D;
- P4-G;
- P4-H;
- P4-I;
- the P4-F primary admission decision.

## B. Namespace And History Search Findings

Candidate creation required repository-content and Git-history searches for:

- `P4-J`;
- `P4J`;
- `P4-F independent verification`;
- `LEVEL3_ADMISSION verification`;
- `P4-F verification attestation`;
- `admission verification attestation`;
- `result independent verification`;
- `verification token`;
- equivalent canonical path/schema/token/provenance contract.

Repository content search found no existing P4-J/P4J canonical namespace, no
P4-F Level-3 admission verification-attestation schema, and no existing
authority that resolves this exact provenance contract. The relevant current
content hits were P4-G/P4-I independent-verification token or attestation
authority patterns and P4-E historical review text; those are adjacent
precedents, not P4-F result verification provenance authority.

Git-history search found no existing P4-J/P4J report/spec authority and no
applicable P4-F admission verification-attestation contract. The `P4J`
history hits were confined to old uploaded or backup runtime artifacts and
did not establish a repository authority namespace. Git-history hits for
`verification token`, `LEVEL3_ADMISSION`, and `verification attestation`
resolved to P4-E, P4-F, P4-G, P4-H, or P4-I authority history, not this P4-F
result-independent-verification provenance contract.

If any later verifier finds an existing applicable frozen authority that
already resolves this exact contract, P4-J must be BLOCKED and must not be
used.

If any later verifier finds that P4-J/P4J collides with an existing canonical
namespace, P4-J must be BLOCKED.

## C. Primary Result Identity Bound By P4-J

P4-J binds the current P4-F primary result candidate as evidence:

- result path:
  `reports/reason_router_p2_p3w6f2_p4f_level3_admission_result_2a17772a4e44bd29a8f40fb6f8fac4ba90bb591a.json`
- candidate SHA256:
  `7283ab2e685d8c64f6e007108510b3e4bdff3a4f6e9e67c7c385bcd0245e5a2f`
- admission commit:
  `2a17772a4e44bd29a8f40fb6f8fac4ba90bb591a`
- primary decision:
  `P3W6F2P4F_LEVEL3_ADMISSION_PASS`
- P4-F specification commit:
  `8fd76261fbad2ef78e915e3442147c971e95aa33`

`admission_commit` is the pre-existing committed repository state against
which the P4-F admission evaluation was performed and whose full SHA is
encoded in the primary result filename.

`result_freeze_commit` is a future immutable Git commit that first freezes the
P4-F primary result file itself. It is not known during P4-J candidate
creation and must not be predicted in this specification or in the P4-F
primary result.

Freezing the P4-F primary result file does not make the PASS final authority.
It only creates one required immutable input for a later independent verifier.

## D. Required Freeze Ordering

The required ordering is exactly:

1. P4-J candidate specification is created.
2. P4-J receives independent verification.
3. P4-J is frozen at an immutable Git commit.
4. The existing P4-F primary result is separately reviewed and frozen at an
   immutable Git commit.
5. A new independent P4-F verifier runs against:
   - frozen P4-J;
   - frozen P4-F primary result;
   - frozen upstream authorities.
6. Only if that new verification PASSes may it create the P4-F independent
   verification attestation.
7. The attestation receives separate review and immutable freeze.
8. Only after the attestation freeze may the P4-F PASS be treated as final,
   independently verified, consumable admission authority.
9. Only then may a later workflow consider explicit authorization of the
   selected P4-H audit.

The previous substantive verifier run must not be backfilled into the final
attestation.

Specification verification, P4-J freeze, P4-F result freeze, and attestation
creation are separate authority steps. No step may predict the future commit
SHA produced by a later step.

## E. Verification Attestation Path Contract

The future P4-F independent verification attestation path is:

`reports/reason_router_p2_p3w6f2_p4f_level3_admission_verification_attestation_<result_freeze_commit>.json`

where:

`<result_freeze_commit>` is the full 40-character immutable Git commit that
froze the P4-F primary result file.

`<result_freeze_commit>` must exist before the attestation is created. P4-J
does not resolve or predict that SHA.

The attestation itself must not require, contain, or forward-predict the Git
commit that first freezes the attestation file.

## F. Verification Attestation Schema

Future successful P4-F verification attestation schema/version:

`P3W6F2P4F_LEVEL3_ADMISSION_VERIFICATION_ATTESTATION_V1`

The attestation must require at least:

- `schema_version`
- `p4f_spec_commit`
- `admission_commit`
- `result_path`
- `result_freeze_commit`
- `primary_result_sha256`
- `primary_decision_token`
- `independent_verification_status`
- `independent_verification_token`
- `admission_prerequisite_verification_status`
- `action_resolution_provenance_verification_status`
- `selected_action_contract_verification_status`
- `authority_binding_status`
- `provenance_verification_status`
- `separate_execution_authorization_required`
- `training_admission_released`
- `verification_contract_authority_path`
- `verification_contract_authority_commit`
- `verification_contract_authority_token`
- `blockers`
- `failure_reasons`

Additional provenance fields may be included, but no required frozen P4-F
field may be removed, renamed, or reinterpreted.

The required P4-J binding fields in a future attestation are:

- `verification_contract_authority_path =
  reports/reason_router_p2_p3w6f2_p4j_p4f_result_independent_verification_provenance_authority_spec.md`
- `verification_contract_authority_commit =` the already-existing immutable
  P4-J freeze commit at attestation creation time
- `verification_contract_authority_token =
  P3W6F2P4J_P4F_RESULT_INDEPENDENT_VERIFICATION_PROVENANCE_AUTHORITY_V1`

This P4-J candidate does not predict its own future freeze SHA.

## G. Successful Verification Status And Token

P4-J defines only the currently proven needed successful combination.

When:

`primary_decision_token = P3W6F2P4F_LEVEL3_ADMISSION_PASS`

and:

`independent_verification_status = PASS`

then the canonical independent verification token is:

`P3W6F2P4F_LEVEL3_ADMISSION_PRIMARY_PASS_INDEPENDENT_VERIFICATION_PASS`

This token becomes canonical only after P4-J:

1. independently verifies PASS; and
2. is frozen at an immutable commit.

P4-J does not define speculative verification tokens for:

- primary `FAIL`;
- primary `BLOCKED`;
- verifier `BLOCKED`;
- verifier `FAIL`;
- other future cases.

The successful attestation contract may use:

`independent_verification_status = PASS`

only.

If a future verifier is BLOCKED or FAIL, no successful attestation is created
under this contract. A later authority may define other recording semantics if
needed.

## H. Required Meaning Of A PASS Attestation

A valid PASS attestation must independently verify at minimum:

1. exact primary result path, SHA256, and freeze provenance;
2. `admission_commit` semantics as the pre-existing committed evaluation
   state, not the result-freeze commit;
3. every frozen P4-F admission prerequisite;
4. Gate-6/P4-B/P4-D identities;
5. Stage184/185 historical-only treatment;
6. independently verified and frozen P4-G action-resolution provenance;
7. exact interpretation of `level3_action_authority_*`;
8. P4-H selected-action source authority;
9. exact earliest action and exact action type;
10. P4-F PASS semantics;
11. no conflicting newer authority;
12. scientific-boundary preservation;
13. `training_admission_released = false`;
14. `separate_execution_authorization_required = true`.

The future verifier must verify that the attested P4-F PASS means only that
the exact selected P4-H static audit becomes eligible for a later explicit
workflow/execution authorization.

## I. Immutable Consumption Semantics

P4-F PASS becomes final independently verified repository authority only when
all of the following exist coherently:

1. frozen P4-F specification;
2. frozen P4-F primary result;
3. primary result SHA binding;
4. frozen P4-J authority;
5. new independent verifier PASS;
6. conforming PASS attestation;
7. immutable attestation freeze commit.

Before item 7, P4-F PASS is not final consumable admission authority.

The immutable provenance tuple sufficient to identify the independently
verified P4-F PASS is:

- P4-F specification path and commit;
- admission commit;
- primary result path;
- primary result SHA256;
- result freeze commit;
- P4-J authority path, freeze commit, and authority token;
- verification attestation path;
- verification attestation SHA256;
- verification attestation freeze commit;
- independent verification token.

The attestation must not predict its own future freeze commit. That commit is
external Git provenance obtained only after attestation freeze.

## J. Execution And Training Boundary

Even after final P4-F independent verification, these values remain mandatory:

`training_admission_released = false`

`separate_execution_authorization_required = true`

Final verified P4-F PASS means only that this selected P4-H action becomes
eligible for a subsequent explicit workflow/execution authorization:

`P3W6F2P4H_A0_CURRENT_LINEAGE_EXECUTION_REBIND_AUDIT`

P4-J does not itself authorize or execute:

- P4-H audit;
- implementation;
- trainer changes;
- data or sidecar generation;
- A0/A1/A2/A3;
- training;
- evaluation;
- model or checkpoint operations;
- Stage185 execution;
- Kaggle;
- GPU;
- promotion.

No false-to-true transition of `training_admission_released` is authorized by
P4-J, by future P4-F result freeze, or by future P4-F independent verification
attestation.

## K. Scientific Boundary

P4-J is verification-provenance-only.

It must not modify, reinterpret, or reopen:

- 119-pair Gate-6 result;
- F2 remediation;
- P4-B regeneration;
- Gate-5 integrity;
- Stage185 compatibility;
- Reason Router design;
- reason ordering;
- loss or gradient semantics;
- split or seed semantics;
- label semantics;
- dataset identity;
- promotion criteria.

P4-J must not change any frozen authority or any current result file.

## L. Fail-Closed Conditions

P4-J must fail closed or be BLOCKED for:

- HEAD mismatch during candidate creation;
- tracked worktree or index dirt during candidate creation;
- existing applicable verification provenance authority found;
- P4-J/P4J namespace collision;
- any need to modify frozen P4-F;
- any need to modify the P4-F primary result candidate;
- multiple equally authoritative recording contracts remaining unresolved;
- any attempt to retroactively treat the prior verifier run as the final
  attestation;
- any attestation that lacks a frozen P4-J authority binding;
- any attestation path that does not encode the exact `result_freeze_commit`;
- any missing or mismatched primary result SHA256;
- any mismatch between `admission_commit` and result filename semantics;
- any attempt to treat primary result freeze alone as final P4-F authority;
- any attempt to authorize P4-H, training, evaluation, implementation, Kaggle,
  GPU, promotion, or Stage185 execution;
- any attempt to set `training_admission_released=true`;
- any scientific-boundary reinterpretation.

Missing, ambiguous, conflicting, or incomplete verification provenance is
BLOCKED, not substantive admission FAIL.

## M. Candidate Creation Environment

Environment:

`LOCAL_STATIC_READ_ONLY`

Allowed for this candidate:

- repository and Git history search;
- static authority inspection;
- hash/provenance comparison;
- creation of this single P4-J candidate specification.

Forbidden:

- Python project runtime;
- pytest;
- validators;
- Kaggle;
- GPU;
- training or evaluation;
- P4-H audit;
- modifying any existing file;
- modifying the P4-F result candidate;
- staging;
- commit;
- push.

Training/Evaluation allowed:

`NO`

Commit/Push:

`NO`

Final candidate readiness token:

`P3W6F2P4J_P4F_RESULT_INDEPENDENT_VERIFICATION_PROVENANCE_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
