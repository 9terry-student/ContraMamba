# P3-W6-F2-P4-I P4-G Verification Token Namespace Authority Specification

Authority/version:

`P3W6F2P4I_P4G_VERIFICATION_TOKEN_NAMESPACE_AUTHORITY_V1`

Decision scope: specification-only authority repair for the missing P4-G PASS-primary independent-verification token namespace.

This document is a candidate authority specification. It becomes canonical only after this P4-I specification independently verifies PASS and is frozen at an immutable Git commit. It does not claim that any token newly defined here was historically canonical.

## A. Exact Blocker Being Repaired

The current independent verification attempt for the frozen P4-G PASS primary result found:

- primary-result substance: `PASS`
- primary SHA/provenance: `PASS`
- candidate universe: `PASS`
- candidate counts: `PASS`
- P4-H admissibility: `PASS`
- precedence/non-domination: `PASS`
- training boundary: `PASS`

That attempt stopped before attestation creation solely because:

`CANONICAL_PASS_PRIMARY_INDEPENDENT_VERIFICATION_TOKEN_UNRESOLVED`

No attestation was created by that attempt.

P4-I resolves only this missing token namespace binding. It does not re-run, reinterpret, or re-adjudicate the P4-G primary action-resolution decision.

## B. Existing Historical Token Preservation

The already-frozen historical P4-G BLOCKED-primary attestation contains this independent verification token:

`P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_PRIMARY_BLOCKED_INDEPENDENT_VERIFICATION_PASS`

Historical attestation path:

`reports/reason_router_p2_p3w6f2_p4g_level3_action_resolution_verification_attestation_ebeb6f77c46a23c0b6bc29aeddb5fca4cc69aabf.json`

Historical attestation freeze commit:

`9bec86eea8082d8d0dd68419542dc8565374c5e9`

This P4-I specification preserves the historical BLOCKED-primary token unchanged. It does not rename, supersede, reinterpret, or invalidate that historical token.

## C. New Token Definition

When:

`primary_decision_token = P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_PASS`

and:

`independent_verification_status = PASS`

then the canonical independent verification token is:

`P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_PRIMARY_PASS_INDEPENDENT_VERIFICATION_PASS`

This authority defines no speculative tokens for:

- primary `FAIL`
- verifier `BLOCKED`
- verifier `FAIL`
- other future cases

Those cases are outside the currently proven need and remain unresolved unless a later authority explicitly defines them.

## D. No Retroactive Attestation

P4-I itself:

- does not create an attestation;
- does not convert the previous BLOCKED verifier attempt into a completed attestation;
- does not alter the frozen primary result;
- does not change `verification_status` inside the primary result;
- does not rewrite P4-G;
- does not claim P4-G independently verified until a later verifier creates and freezes a valid attestation.

The prior blocked verifier run must not be backfilled.

## E. Future Attestation Binding

A future independent verifier may use the new canonical token only after P4-I itself has:

1. passed independent verification;
2. been frozen at an immutable Git commit.

The future P4-G PASS attestation remains at:

`reports/reason_router_p2_p3w6f2_p4g_level3_action_resolution_verification_attestation_4699e280afefd0b8a2e9d9eaceb3e9a4d63f7a54.json`

The existing P4-G required attestation fields remain unchanged. The frozen P4-G schema defines minimum required fields, so this P4-I authority may require additive provenance fields without removing or reinterpreting any existing P4-G field.

Because the PASS-primary token is defined by later P4-I authority, a future P4-G PASS attestation that uses it must add these explicit provenance extension fields:

- `verification_token_authority_path`
- `verification_token_authority_commit`
- `verification_token_authority_token`

with:

`verification_token_authority_path = reports/reason_router_p2_p3w6f2_p4i_p4g_verification_token_namespace_authority_spec.md`

`verification_token_authority_commit = the already-existing immutable future P4-I freeze commit`

`verification_token_authority_token = P3W6F2P4I_P4G_VERIFICATION_TOKEN_NAMESPACE_AUTHORITY_V1`

This candidate specification does not predict the future P4-I freeze SHA.

## F. Existing Immutable Provenance

P4-I binds the following existing provenance without changing it:

- P4-G spec commit: `dc62e63da045435429dc3927f8dc5b2c0d4de59c`
- `resolution_state_commit`: `368d3b6991389aa6b6fd80f421c73565b562e290`
- primary result freeze commit: `4699e280afefd0b8a2e9d9eaceb3e9a4d63f7a54`
- primary result SHA256: `678f09ce25aef101cea12f6c49b1182e839d988a023afdb10530d88420e1530a`
- primary decision: `P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_PASS`
- selected action: `P3W6F2P4H_A0_CURRENT_LINEAGE_EXECUTION_REBIND_AUDIT`
- P4-H authority commit: `368d3b6991389aa6b6fd80f421c73565b562e290`

## G. Authority Effect

P4-I may only resolve the token namespace ambiguity for the PASS-primary/PASS-verifier combination defined above.

P4-I explicitly does not:

- alter the primary P4-G decision;
- perform independent verification itself;
- create or freeze the P4-G attestation;
- execute P4-H audit;
- run P4-F admission evaluation;
- modify implementation;
- generate sidecars/data;
- authorize A0/A1/A2/A3;
- authorize training/evaluation;
- authorize model/checkpoint loading;
- authorize Kaggle/GPU;
- set `training_admission_released=true`.

Mandatory invariant:

`training_admission_released = false`

## H. Workflow Ordering

The required ordering is:

1. P4-I candidate specification created.
2. P4-I independently verified.
3. P4-I frozen at immutable commit.
4. A new independent P4-G verifier run may then consume frozen P4-I.
5. Only that new verifier may create the P4-G PASS attestation using the canonical token.
6. Attestation must then be separately reviewed/frozen.
7. Only after verified/frozen P4-G PASS may P4-F be re-evaluated.
8. No P4-H audit execution occurs before the required P4-F admission path.

The prior blocked verifier run cannot simply be backfilled.

## I. Environment

Environment:

`LOCAL_STATIC_READ_ONLY`

Allowed:

- Git/file/history/search/hash inspection;
- creation of this single P4-I specification candidate.

Forbidden:

- Python project runtime;
- pytest;
- validators;
- model/Torch/checkpoints;
- dataset/sidecar generation;
- training/evaluation;
- Kaggle/GPU;
- modification of any existing file.

Training/Evaluation allowed:

`NO`

Commit/Push:

`NO`

## J. Namespace Search Findings Required For Candidate Creation

P4-I candidate creation requires confirming:

- HEAD is exactly `4699e280afefd0b8a2e9d9eaceb3e9a4d63f7a54`;
- tracked worktree is clean;
- no P4-I/P4I namespace conflict exists;
- no existing canonical token already resolves the exact PASS-primary/PASS-verifier case;
- no higher authority already defines the missing PASS-primary token;
- the prior BLOCKED-primary token exists and is preserved.

If any existing applicable canonical PASS-primary independent-verification token is found, P4-I must not be created and the process must stop.

## K. Compatibility Assessment

The frozen P4-G attestation schema lists minimum required fields and does not prohibit additional fields. Therefore, P4-I's future requirement for `verification_token_authority_path`, `verification_token_authority_commit`, and `verification_token_authority_token` is additive provenance and is compatible with P4-G so long as no existing P4-G field is removed, renamed, or reinterpreted.

If a later verifier finds that these additive provenance fields conflict with frozen P4-G authority, attestation creation must block rather than modify frozen P4-G or the primary result.

## L. Fail-Closed Boundaries

This P4-I specification must fail closed for:

- HEAD mismatch during candidate creation;
- tracked worktree dirt during candidate creation;
- existing applicable canonical PASS-primary token found;
- P4-I/P4I namespace collision;
- additive attestation provenance-field conflict with frozen P4-G authority;
- any need to modify frozen P4-G or the primary result;
- any implied training/evaluation authority.

Final candidate readiness token:

`P3W6F2P4I_P4G_VERIFICATION_TOKEN_NAMESPACE_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
