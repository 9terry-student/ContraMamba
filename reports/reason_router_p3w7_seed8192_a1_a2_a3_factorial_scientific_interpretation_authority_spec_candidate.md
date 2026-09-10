# P3-W7 Seed8192 A0/A1/A2/A3 Factorial Scientific-Interpretation Authority Candidate

## 1. Status and authority chain

This is a report-only scientific-interpretation authority candidate.  It defines
the permissible future interpretation of already frozen evidence; it performs no
interpretation and selects no factorial conclusion.

```text
INTERPRETATION_AUTHORITY_STATUS = CANDIDATE_ONLY
VALIDATED_EVIDENCE_FREEZE_COMMIT = 6dcef9520af2cb88691628a77b72f3fdd7042cd8
VALIDATED_EVIDENCE_REPORT_BLOB = 17be9bead782f129de925148580b034d981079cd
UPSTREAM_EXECUTION_AUTHORITY_COMMIT = 3a76c6cd3f6bd8b011317f37938677822ce9191d
UPSTREAM_EXECUTION_AUTHORITY_FILE = reports/reason_router_p3w7_seed8192_a1_a2_a3_factorial_execution_authority_spec_candidate.md
TRAINING_EVALUATION_AUTHORIZED = NO
IMPLEMENTATION_AUTHORIZED = NO
E0_EXECUTION_AUTHORIZED = NO
A4_EXECUTION_AUTHORIZED = NO
ADDITIONAL_SEED_OR_ARM_AUTHORIZED = NO
FINAL_FACTORIAL_SCIENTIFIC_CONCLUSION_IN_THIS_TASK = NONE
```

Authority precedence is the current task instruction, the frozen validated
evidence report at the commit and blob above, the upstream execution authority
at the commit and file above, applicable historical scientific-interpretation
report patterns (including O0b) as context only, and `AGENTS.md`.  The frozen
validated-evidence report is the numerical authority; the execution authority
defines the completed design and its execution constraints.

This candidate becomes usable only after all of the following occur: independent
verification reports PASS; these exact candidate bytes are frozen in a dedicated
commit; that commit is pushed; and remote authentication of the commit and blob
succeeds.  Until then it is not an active interpretation authority.

## 2. Frozen scientific design

The completed matched 2x2 factorial has these factors:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Router | `explicit_product` | `conditional_first_blocker` |
| Gradient ownership | `joint` | `explicit_local` |

| Cell | Router | Gradient ownership | Reason-loss weight |
|---|---|---|---:|
| A0 | `explicit_product` | `joint` | 0 |
| A1 | `conditional_first_blocker` | `joint` | 0.6273209029272248 |
| A2 | `explicit_product` | `explicit_local` | 0 |
| A3 | `conditional_first_blocker` | `explicit_local` | 0.6273209029272248 |

Frozen run seeds are 180, 181, and 182.  The split is 8192.  Interpretation
must preserve the reason order `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`,
the diagnostic-only role of secondary reasons, router-only final 3-way CE, and
the frozen encoder.  The global `frame_downstream_gradient_mode=joint` is a
separate setting from `gradient_ownership_mode`; it must not be conflated with
the factorial ownership factor.

## 3. Sole admissible scientific evidence

Later interpretation must use
`reports/reason_router_p3w7_seed8192_a1_a2_a3_factorial_validated_evidence_analysis_report_candidate.md`
at commit `6dcef9520af2cb88691628a77b72f3fdd7042cd8` and blob
`17be9bead782f129de925148580b034d981079cd` as its primary numerical authority.
Upstream imported JSON evidence may be read only for traceability or verification;
it may not silently redefine frozen validated numbers or introduce new metrics.

No checkpoint loading, model execution, new inference, or Kaggle execution is
admissible.  Historical split174 results are inadmissible as active scientific
evidence.  Historical seed180 r2 is excluded.  The seed180 A3 `-retry1` suffix
is controller-history context only, not a scientific premise, consistent with
the validated-evidence report.

## 4. Frozen primary scientific questions and contrasts

| Question | Frozen question | Primary contrast |
|---|---|---|
| Q1 | Under joint ownership, does replacing `explicit_product` with `conditional_first_blocker` show a reproducible beneficial, neutral, mixed, or harmful descriptive effect? | A1 - A0 |
| Q2 | Under `explicit_product`, does changing ownership from `joint` to `explicit_local` show a reproducible beneficial, neutral, mixed, or harmful descriptive effect? | A2 - A0 |
| Q3 | With `explicit_local` ownership active, does `conditional_first_blocker` improve or rescue performance relative to `explicit_product`? | A3 - A2 |
| Q4 | With `conditional_first_blocker`, does `explicit_local` ownership improve or degrade the configuration? | A3 - A1 |
| Q5 | Is the combined A3 effect compatible with additive independent factor effects, or is there descriptive interaction? | A3 - A2 - A1 + A0 |
| Q6 | When final macro-F1 or accuracy changes, which external-class F1 and structural diagnostics account for the change? | Class and structural decomposition |

Q6 must examine `NOT_ENTITLED`, `REFUTE`, and `SUPPORT` F1; frame, predicate,
sufficiency, and entitled-polarity accuracy; and relevant pairwise diagnostic
pass rates.

## 5. Outcome hierarchy and evidence discipline

The primary outcome is final macro-F1.  Final accuracy is the co-primary
descriptive safeguard.  Every contrast must also decompose `NOT_ENTITLED` F1,
`REFUTE` F1, `SUPPORT` F1, and prediction distribution.  Required structural
diagnostics are frame accuracy, predicate accuracy, sufficiency accuracy,
entitled-polarity accuracy, and pairwise diagnostics.

Structural diagnostics may explain patterns but must not override contradictory
final-task metrics without explicit justification.  No post-hoc scalar composite,
weighted aggregate score, or unfreezed metric may be constructed.

## 6. Bounded interpretation vocabulary and cross-seed rules

For each contrast, the only permitted descriptive labels are:

```text
DESCRIPTIVELY_BENEFICIAL
DESCRIPTIVELY_HARMFUL
MIXED_OR_SEED_DEPENDENT
NEAR_NEUTRAL
NOT_INTERPRETABLE
```

These labels are descriptive, not significance claims.  A later report must not
use “statistically significant,” “proven,” “causal,” “universally better,” or
“superior architecture” unless a later authority supplies evidence that supports
those terms.

N=3 is descriptive only.  The later report must give each seed separately, the
cross-seed mean, range/SD as frozen in validated evidence, and sign consistency.
It must not hide sign reversals.  A materially different sign across seeds cannot
be described as reproducibly beneficial or harmful without explicit qualification
of heterogeneity.  A common sign in all three frozen seeds may support the bounded
phrase “directionally consistent across the three frozen seeds,” but not a claim
of statistical significance or generalization across random seeds or populations.

No numeric PASS threshold may be invented.  Magnitude must be described relative
to observed frozen contrasts and the baseline-cell scale, distinguishing small
near-baseline change, material degradation/improvement, and seed-sensitive
change.  Decimal differences must not become post-hoc promotion thresholds.

## 7. Interaction rules

The interaction is separately assessed as:

```text
I = A3 - A2 - A1 + A0
```

It is distinct from every main contrast.  A nonzero descriptive interaction does
not establish a causal mechanism, and sign heterogeneity across seeds must be
retained.  The later report may discuss the factors as approximately additive,
antagonistic, synergistic, or heterogeneous only in descriptive factorial terms;
it must not make a mechanistic causal claim from interaction alone.

## 8. Required falsification and alternative-explanation matrix

| ID | Required consideration |
|---|---|
| F1 | If A1 is approximately A0 across seeds, `conditional_first_blocker` alone does not establish practical improvement despite reason-specific supervision. |
| F2 | If A2 < A0 consistently, `explicit_local` ownership is descriptively harmful under `explicit_product`, not beneficial isolation. |
| F3 | If A3 < A1 consistently, `explicit_local` ownership remains harmful with `conditional_first_blocker`. |
| F4 | If A3 does not exceed A2, conditional routing does not rescue `explicit_local` ownership. |
| F5 | If class-level gains are offset by `SUPPORT` or `REFUTE` collapse, aggregate accuracy alone cannot justify promotion. |
| F6 | If structural diagnostic accuracy improves while final macro-F1 degrades, diagnostic improvement is insufficient to establish task-level benefit. |
| F7 | If effects reverse by seed, characterize heterogeneity rather than forcing a common benefit claim. |
| F8 | If interaction varies strongly by seed, do not claim a stable interaction mechanism. |

## 9. Allowed conclusion types and forbidden conclusions/actions

A later report may, if directly supported by the frozen evidence, make bounded
descriptive conclusions such as a router being near-neutral or mixed relative to
A0, ownership being descriptively harmful, routing failing to rescue ownership,
an interaction being heterogeneous, a specific external class dominating
degradation, or the current factorial not supporting promotion of a component.
These are allowed conclusion types, not conclusions selected here.

The following are forbidden: causal mechanism proven; reason-specific supervision
proven superior; `explicit_local` ownership proven universally harmful;
statistical significance; population-level generalization; production readiness;
architecture promotion; replacing A0 solely from this N=3 evidence; E0 or A4
execution authorization; additional hyperparameter search; choosing a new
reason-loss weight; changing gradient semantics; retraining any cell; post-hoc
seed exclusion; and silently excluding unfavorable diagnostics.

## 10. Later interpretation-report requirements

After independent verification, dedicated freezing, push, and remote
authentication, the later interpretation report must include the exact authority
commit/blob and validated-evidence freeze commit/blob; per-seed A0/A1/A2/A3
metrics; all five primary factorial contrasts; macro-F1 and accuracy; class-F1
decomposition; structural diagnostic interpretation; sign consistency or
heterogeneity; interaction assessment; falsification-matrix assessment; bounded
scientific conclusion; explicit non-claims; and the exact next research
implication.

## 11. Next-phase boundary and non-actions

This authority may permit a later interpretation report to answer: “What do the
completed factorial results imply scientifically?”  It does not authorize the
research action that follows.  That later report may recommend authoring a
subsequent authority, but may not itself authorize implementation,
training/evaluation, E0, A4, a new seed, a parameter sweep, or promotion.

This task performs no implementation, training, evaluation, model/checkpoint
loading, dataset or sidecar modification, imported-artifact modification, Kaggle
execution, staging, commit, or push.  It does not change scripts, tests, datasets,
sidecars/provenance, the execution authority, the validated-evidence report,
imported run artifacts, checkpoints, controller files, or the Git index.
