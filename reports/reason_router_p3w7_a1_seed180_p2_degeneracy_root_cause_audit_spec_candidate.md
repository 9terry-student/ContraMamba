# ContraMamba P3-W7 A1 seed180 P2 Degeneracy Root-Cause Audit Spec Candidate

Status: PASS_READY_FOR_INDEPENDENT_VERIFICATION

Verdict: PASS_ROOT_CAUSE_RESOLVED

Primary classification: EXPECTED_DATA_SPLIT_CONTRACT_REJECTION

Polarity absence classification: RANDOM_SPLIT_DEGENERACY

## Authority

1. Current workflow-controller instruction.
2. Worktree: C:\p3w7-a0-n3-validated-evidence-analysis
3. HEAD: 15a9103a34efaf290d365d069c8c741994805330
4. Branch: p3w7-a1-a2-a3-factorial-execution-authority-n3-v2
5. Frozen incident correction at HEAD.
6. Previously independently verified root-cause evidence.

## Phase

Report-only artifact re-materialization.

No implementation, training, evaluation, GPU work, Kaggle work, staging, commit, or push is authorized by this candidate.

## Frozen Split Evidence

Exact frozen split:

- total rows 3600
- train rows 2880
- dev rows 720
- train pairs 240
- dev pairs 60
- split seed 174
- dev ratio 0.2

Applicable binary counts:

| Reason | Split | Applicable | Class 0 | Class 1 |
|---|---:|---:|---:|---:|
| frame | train | 1450 | 726 | 724 |
| frame | dev | 319 | 174 | 145 |
| predicate | train | 724 | 121 | 603 |
| predicate | dev | 145 | 29 | 116 |
| sufficiency | train | 603 | 242 | 361 |
| sufficiency | dev | 116 | 58 | 58 |
| polarity | train | 361 | 119 | 242 |
| polarity | dev | 58 | 0 | 58 |

Polarity support:

| Split | REFUTE | SUPPORT |
|---|---:|---:|
| train | 119 | 242 |
| dev | 0 | 58 |

Exact failure:

```text
P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE:
{'dev': {'polarity': {0: 0, 1: 58}}}
```

## Contract Findings

- REFUTE -> class0.
- SUPPORT -> class1.
- Polarity class0 is semantically possible.
- Train contains valid polarity class0 examples.
- Normal A1/A3 dev degeneracy rejection is intentional.
- P3-W1 spec preserves that gate.
- test_normal_a1_a3_path_still_applies_dev_cohort_degeneracy_gate verifies it.
- Normal P2 full helper rejects degenerate polarity supervision.

## Defect Matrix

| Candidate Defect | Finding |
|---|---|
| Code defect | NO |
| Data defect | NO |
| Split implementation defect | NO |
| Label-semantics defect | NO |
| Applicable-cohort construction defect | NO |
| Validation-scope defect | NO |
| Frozen configuration/gate incompatibility | YES |
| Pre-execution feasibility-validation omission | YES |
| Deterministic under frozen inputs | YES |

## Root-Cause Conclusion

The P2 failure is not explained by a code defect, data defect, split implementation defect, label-semantics defect, applicable-cohort construction defect, or validation-scope defect. The observed failure is the expected consequence of the frozen data split contract producing a dev applicable polarity cohort with no class0 examples under split seed 174 and dev ratio 0.2.

The correct primary classification is EXPECTED_DATA_SPLIT_CONTRACT_REJECTION. The correct polarity absence classification is RANDOM_SPLIT_DEGENERACY.

Because the trainer and normal P2 full helper reject degenerate polarity supervision, the observed stop is contract-preserving behavior under the frozen inputs. This resolves the root cause as a frozen configuration/gate incompatibility plus a pre-execution feasibility-validation omission, not as a defect in the trainer guard or reason-label semantics.

## Correction Boundary

- Do not weaken trainer guard.
- Do not change split seed/dev ratio.
- Do not modify dataset.
- Do not change labels/applicability.
- Those are scientific-contract changes.
- Future static cohort-feasibility preflight may be separately authorized.
- No implementation here.

## Incident Boundary

- recovery3 remains UNAUTHORIZED_TRAINER_LAUNCH_PROVENANCE_INCIDENT.
- No scientific evidence.
- No automatic retry.
- No run-name reuse.
- A2/A3 blocked.
- Replicate budget undecided.
- Future execution requires new explicit independently verified/frozen authority.

## Artifact-Recovery Disclosure

Prior lost candidate identities:

- 5fe9782596aa5643827e7166ef02dc9b1453944ca71dddb1808a8a69b352590c
- 2f30584da3ab2d2ed52950780f0a494fd476d4376f95471af8715d886b5a98f7

Both were untracked candidate revisions whose substantive content was verified, but neither was frozen because the worktree artifact later disappeared. This new candidate revision does not claim byte identity with either prior lost candidate.

## Execution Boundary

Training/Evaluation: NO

GPU/Kaggle: NO

Commit/Push: NO

Staging: NO

## Expected Delta

Exactly one new untracked Markdown file:

- reports/reason_router_p3w7_a1_seed180_p2_degeneracy_root_cause_audit_spec_candidate.md

No tracked modifications.

No staging.
