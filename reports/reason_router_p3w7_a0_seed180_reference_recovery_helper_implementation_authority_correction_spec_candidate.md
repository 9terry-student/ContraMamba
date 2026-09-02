# P3-W7 A0 Seed180 Reference Recovery Helper Implementation Authority Correction Specification Candidate

Authority/version:

`P3W7_A0_SEED180_REFERENCE_RECOVERY_HELPER_IMPLEMENTATION_AUTHORITY_CORRECTION_V1_CANDIDATE`

## Status

READY.

This is a static authority-correction candidate only. It authorizes no implementation, no source or test modification, no retained ZIP extraction, no seed180 artifact materialization, no `A0_REFERENCE_AUDIT.json` creation, no training, no evaluation, no model load, no checkpoint deserialization, no Kaggle execution, no run-registry mutation, no commit, and no push.

If independently verified and frozen, this correction candidate supersedes only the defective runtime freeze-binding clause and dependent CLI/example/test wording in the frozen helper implementation authority V1. It does not discard, rewrite, amend, or replace the V1 commit as provenance.

## Authority Basis

Frozen helper implementation authority V1 commit containing the defect:

`1344323726bda6f6526d374f0dedafb9b33aabf2`

Defective V1 authority file:

`reports/reason_router_p3w7_a0_seed180_reference_recovery_helper_implementation_authority_spec_candidate.md`

Frozen upstream retained-artifact reference-recovery authority commit:

`ceaee6236340ef7006f7004d910f388ec565db0e`

Upstream authority file:

`reports/reason_router_p3w7_a0_seed180_retained_artifact_reference_recovery_authority_spec_candidate.md`

Repository-wide `AGENTS.md` applies.

## Exact Defect

The frozen V1 helper implementation authority incorrectly treats `ceaee6236340ef7006f7004d910f388ec565db0e` as both the upstream retained-artifact recovery authority and the helper runtime authority freeze. It requires future CLI execution to satisfy `HEAD == ceaee6236340ef7006f7004d910f388ec565db0e` while authorizing the helper script to be added only in a later implementation commit.

That contract is impossible:

- at `ceaee6236340ef7006f7004d910f388ec565db0e`, the future helper script does not exist;
- at any later helper implementation commit containing the helper script, `HEAD` cannot equal `ceaee6236340ef7006f7004d910f388ec565db0e`.

No scientific, provenance, ZIP, artifact, audit, destination, transaction, implementation-scope, test-matrix, or non-authorization semantics are defective merely because of this freeze-binding error.

## Corrected Authority Lifecycle

The executable corrected lifecycle is:

1. Upstream retained-artifact recovery authority: `ceaee6236340ef7006f7004d910f388ec565db0e`.
2. Helper implementation authority V1: `1344323726bda6f6526d374f0dedafb9b33aabf2`, retained as immutable provenance with a known defective freeze-binding clause.
3. This correction candidate: if independently verified and frozen, supersedes only the defective freeze-binding clause and dependent wording.
4. Future helper implementation commit: a distinct later commit containing exactly the authorized helper script and helper test delta.
5. Still-later recovery execution authority freeze: a distinct later immutable authority that binds the exact helper implementation commit, exact retained ZIP, and exact recovery operation/wrapper, and is independently verified and committed before real materialization.
6. Recovery execution: the helper may run only when current `HEAD` equals the supplied recovery execution authority freeze commit.

The helper script is present in the future execution tree by construction because the still-later recovery execution authority must be committed after, and must bind, the distinct helper implementation commit that adds `scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py`.

## Corrected Identity Semantics

`ceaee6236340ef7006f7004d910f388ec565db0e` remains only the frozen upstream retained-artifact recovery authority. It is not the helper runtime `HEAD` value and must not be hardcoded as the helper runtime `HEAD`.

`1344323726bda6f6526d374f0dedafb9b33aabf2` remains immutable defective V1 helper implementation authority provenance. It is not discarded, rewritten, amended, or mechanically used as the helper runtime `HEAD`.

The future helper implementation commit is a separate later implementation commit. It must contain exactly the authorized helper source/test delta and must not itself authorize retained ZIP extraction, materialization, training, evaluation, model load, checkpoint deserialization, Kaggle execution, commit, or push.

The future recovery execution authority freeze is the only valid runtime `HEAD` binding for real helper execution. It must bind the exact helper implementation commit, exact retained ZIP, and exact recovery operation/wrapper before materialization.

## Selected Final CLI Argument

Selected final production CLI name:

`--expected-recovery-execution-authority-commit`

Rationale: the name states the actual authority identity being supplied at runtime. It avoids preserving the defective ambiguity in V1's `--expected-authority-freeze-commit`, where "authority" was incorrectly read as the upstream retained-artifact authority instead of the later execution authority.

Corrected CLI shape:

```text
python scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py materialize-reference --zip C:\Users\Home1\Downloads\seed180_a0_6189be22715e.zip --expected-recovery-execution-authority-commit <40-hex-lowercase-recovery-execution-authority-freeze-commit>
```

Corrected argument semantics:

- `--expected-recovery-execution-authority-commit` is the later recovery execution authority's immutable freeze commit.
- The helper must validate the value is lowercase 40-hex.
- The helper must resolve the value to a Git commit.
- The helper must require current `HEAD` to equal the supplied commit.
- The helper must not hardcode `ceaee6236340ef7006f7004d910f388ec565db0e` as the runtime `HEAD`.
- The helper must not hardcode `1344323726bda6f6526d374f0dedafb9b33aabf2` as the runtime `HEAD`.
- The later recovery execution authority, not the helper implementation authority, will supply and freeze the exact runtime value.

No alternative production CLI name remains open.

## Complete V1 Occurrence Review

The complete V1 document was searched for:

- `ceaee6236340ef7006f7004d910f388ec565db0e`;
- `expected-authority-freeze-commit`;
- `authority freeze commit`;
- `HEAD`;
- `implementation commit`;
- `recovery execution authority`.

Defective or dependent V1 clauses found:

| V1 location | V1 text or semantic role | Correction |
| --- | --- | --- |
| Authority Basis, "Authority freeze commit for this candidate" | Identifies `ceaee6236340ef7006f7004d910f388ec565db0e` as the helper implementation authority freeze. | Superseded for helper implementation authority identity. `ceaee623...` remains upstream retained-artifact recovery authority only; V1 commit `134432...` is the frozen helper implementation authority provenance containing the defect. |
| Authority identities list item 1 | "authority freeze commit: `ceaee623...`" | Superseded. Runtime authority identities must distinguish upstream retained-artifact authority, frozen defective V1 helper implementation authority, future helper implementation commit, and later recovery execution authority freeze. |
| Authority identities list item 2 | "future implementation commit: the commit that implements this helper after this candidate is frozen" | Preserved, with clarification that it is distinct from `ceaee623...`, `134432...`, and the later recovery execution authority freeze. |
| Authority identities list item 3 | "later recovery execution authority: a separate authority that may authorize running the helper against the retained ZIP" | Preserved and strengthened: this later authority must bind the exact helper implementation commit, exact retained ZIP, and exact operation/wrapper, and its immutable freeze commit is the runtime `HEAD` value. |
| Required Helper CLI Contract example | Uses `--expected-authority-freeze-commit ceaee6236340ef7006f7004d910f388ec565db0e`. | Superseded. The production CLI must use `--expected-recovery-execution-authority-commit <40-hex-lowercase-recovery-execution-authority-freeze-commit>`. |
| Required Helper CLI Contract command bullet | Requires current `HEAD` to equal `--expected-authority-freeze-commit`. | Superseded. Current `HEAD` must equal `--expected-recovery-execution-authority-commit`, whose value is the later recovery execution authority freeze commit. |
| Required Helper CLI Contract command bullet | "reject malformed commit values" | Preserved and clarified as lowercase 40-hex plus Git commit resolution. |
| Required Future Test Matrix | "CLI rejects wrong authority freeze commit" | Superseded only in naming/semantics. Test must reject a wrong recovery execution authority freeze commit. |
| Required Future Test Matrix | "CLI rejects malformed authority freeze commit" | Superseded only in naming/semantics. Test must reject malformed recovery execution authority commit values, including non-lowercase, non-40-hex, and non-commit objects. |
| Validation Performed For This Candidate | `git rev-parse HEAD` listed as static inspection performed. | Preserved as historical V1 validation text; it is not a runtime helper `HEAD` binding. |

No other V1 occurrence of the searched terms depends on the incorrect interpretation of `ceaee6236340ef7006f7004d910f388ec565db0e` as the helper runtime authority freeze.

## Preserved V1 Semantics

All V1 clauses not listed as superseded above remain authoritative if this correction is independently verified and frozen.

Preserved exactly:

- selected implementation shape: one new narrow helper script;
- future source file: `scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py`;
- future test file: `tests/test_reason_router_p3w7_a0_seed180_reference_recovery_helper.py`;
- no trainer modification;
- exact retained ZIP path and SHA256;
- exact retained ZIP entry identities, sizes, and SHA256 values;
- historical `standard_cm_wrapper_provenance = missing/incomplete`;
- all manifest and `run_provenance.json` gates;
- exact destination namespace;
- no-overwrite and collision semantics;
- transaction strategy;
- raw checkpoint hashing only, with no checkpoint deserialization;
- canonical dataset, sidecar, and split contract;
- exact dev identity algorithm;
- normal `A0_REFERENCE_AUDIT.json` PASS contract;
- additive recovery provenance fields;
- future test matrix except for the corrected CLI argument name and runtime authority semantics above;
- no A1/A2/A3 release;
- no scientific conclusion;
- later artifact freeze/import requirement;
- later factorial authority requirement;
- all explicit non-authorizations.

## Explicit Non-Authorizations

This correction candidate does not authorize:

- implementation;
- changing the frozen V1 authority file;
- changing scripts;
- changing tests;
- modifying the trainer;
- changing the retained ZIP;
- extracting or materializing seed180 artifacts;
- creating `A0_REFERENCE_AUDIT.json`;
- changing artifacts, dataset, sidecar, split semantics, or run registry;
- changing Git history;
- training;
- evaluation;
- model load;
- checkpoint deserialization;
- Kaggle;
- commit;
- push.

## Validation Performed For This Candidate

Static validation performed:

- `git status --short --branch`;
- `git rev-parse HEAD`;
- direct complete inspection of `reports/reason_router_p3w7_a0_seed180_reference_recovery_helper_implementation_authority_spec_candidate.md`;
- direct inspection of `reports/reason_router_p3w7_a0_seed180_retained_artifact_reference_recovery_authority_spec_candidate.md`;
- complete V1 occurrence search for `ceaee6236340ef7006f7004d910f388ec565db0e`, `expected-authority-freeze-commit`, `authority freeze commit`, `HEAD`, `implementation commit`, and `recovery execution authority`.

Validation required after writing this candidate:

- `git diff --check`;
- exactly one new untracked Markdown correction candidate;
- candidate SHA256, byte size, newline checks, final LF check, and trailing-whitespace check.

Not performed:

- implementation;
- source/test modification;
- retained ZIP extraction;
- seed180 artifact materialization;
- `A0_REFERENCE_AUDIT.json` creation;
- training;
- evaluation;
- model load;
- checkpoint deserialization;
- Kaggle;
- commit;
- push.
