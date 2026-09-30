# ContraMamba Gen5 Phase 3
## Training-Stressor Domain Amendment Candidate

### Status

AMENDMENT =
`GEN5_PHASE3_FROZEN_PP3_PP5_STRESSOR_DOMAIN_CORRECTION`

PARENT_PHASE3_DESIGN_COMMIT =
`ae52dbf98cce21226036ae38121edcdfa6f79d7b`

TRAINING_LABEL_AMENDMENT_COMMIT =
`14d2bd7db70fc6000b480221a17fa8475dfefbe5`

STATUS =
`CANDIDATE_FOR_FREEZE`

IMPLEMENTATION =
`NOT_AUTHORIZED`

TRAINING =
`NOT_AUTHORIZED`

SCIENTIFIC_EXECUTION =
`NOT_AUTHORIZED`

This amendment corrects the domain on which the inherited PP3/PP5
training-time intervention may be applied.

The correction is made before Phase 3 model execution.

No Phase 3 model response, contention value, causal endpoint, or p-value has
been observed.

---

## 1. Static incompatibility

The Phase 3 parent design states that PR/PC pressure should be applied during
every active training forward.

However, the inherited PP3/PP5 operational causal-role intervention was
prospectively defined and validated only on the four-cell XG1 branch family
used by the frozen causal program.

The frozen transport core specifies:

TARGET_PLUS_CELL =
`C2_NAME`

TARGET_MINUS_CELL =
`C0_SHAM`

REFERENCE_PLUS_CELL =
`C5_TITLE_NAME`

REFERENCE_MINUS_CELL =
`C1_TITLE`

ANCHOR_NAME =
`A_IDENTITY`

INTERVENTION_LAYER =
`17`

TARGET_OFFSET =
`2`

Therefore the frozen intervention coordinate is:

`A_IDENTITY_ANCHOR + 2`

at layer 17.

---

## 2. Unsupported extrapolation prohibited

The following training cells are not part of the frozen PP3/PP5 causal branch
domain:

`C3_ROLE`

`C4_PREDICATE`

`C6_EXPLICIT_DENIAL`

Applying PP3/PP5 intervention semantics to those cells would introduce a new
unvalidated intervention context.

Phase 3 must not silently make that extrapolation.

---

## 3. Corrected pressure-domain semantics

The seven-cell labeled training view remains unchanged:

- C0_SHAM
- C1_TITLE
- C2_NAME
- C3_ROLE
- C4_PREDICATE
- C5_TITLE_NAME
- C6_EXPLICIT_DENIAL

Task labels remain unchanged.

### P0

All seven cells use native forward semantics.

### PR

Apply the exact frozen PP3-neutralization intervention only on:

- `C0_SHAM`
- `C1_TITLE`
- `C2_NAME`
- `C5_TITLE_NAME`

For:

- `C3_ROLE`
- `C4_PREDICATE`
- `C6_EXPLICIT_DENIAL`

use native forward semantics.

### PC

Apply the exact frozen PP5 matched-control intervention only on:

- `C0_SHAM`
- `C1_TITLE`
- `C2_NAME`
- `C5_TITLE_NAME`

For:

- `C3_ROLE`
- `C4_PREDICATE`
- `C6_EXPLICIT_DENIAL`

use native forward semantics.

Thus PR and PC differ only in PP3 versus PP5 intervention identity over the
same frozen intervention-support domain.

---

## 4. Frozen stressor-domain set

STRESSOR_DOMAIN_CELLS =

`C0_SHAM,C1_TITLE,C2_NAME,C5_TITLE_NAME`

NON_STRESSOR_TRAINING_CELLS =

`C3_ROLE,C4_PREDICATE,C6_EXPLICIT_DENIAL`

STRESSOR_DOMAIN_ROWS_PER_PAIR =
`4`

TOTAL_TRAINING_ROWS_PER_PAIR =
`7`

No cell may migrate between these sets after Phase 3 responses are observed.

---

## 5. Exact target-coordinate contract

For every stressor-domain row:

ANCHOR =
`A_IDENTITY`

INTERVENTION_TOKEN =
`A_IDENTITY_ABSOLUTE_ANCHOR_TOKEN_INDEX + 2`

INTERVENTION_LAYER =
`17`

The static preparation stage must establish complete eligibility for all
stressor-domain rows under this exact coordinate rule.

No alternate anchor may be substituted.

No nearest-token fallback is allowed.

No target-offset search is allowed.

No intervention on a non-stressor-domain row is allowed.

---

## 6. Seven-cell tokenizer contract

All seven training cells must still:

- tokenize under the frozen active tokenizer;
- satisfy the frozen 128-token serialization envelope;
- preserve deterministic claim/evidence identity;
- avoid invalid or empty encoding.

However, only the four stressor-domain cells require the inherited PP3/PP5
target-anchor eligibility gate.

`C3_ROLE`, `C4_PREDICATE`, and `C6_EXPLICIT_DENIAL` do not require a
PP3/PP5 intervention coordinate because no training-time causal stressor is
applied to them.

---

## 7. Scientific interpretation

The Phase 3 manipulation is therefore not:

`GLOBAL_INTERNAL_PERTURBATION_OF_EVERY_TRAINING_ROW`

It is:

`FROZEN_CAUSAL_ROLE_STRESS_ON_THE_EXACT_PREVIOUSLY_VALIDATED_XG1_BRANCH_DOMAIN`

The remaining labeled rows continue to supply ordinary final-3way task
supervision.

This separation avoids extending a validated mechanistic intervention into
unvalidated row contexts merely for training convenience.

---

## 8. Phase 3A contention gate

The already frozen Phase 3A contention metrics and thresholds remain unchanged.

C0 is trained under:

- P0
- PR
- PC

for seeds:

- 6201
- 6202
- 6203

R22 contention is still assessed from the final unrestricted correction map.

No contention threshold changes.

No stressor-strength changes.

No response-conditioned cell selection.

---

## 9. Phase 3B interaction design

If and only if Phase 3A passes, the Phase 3B ownership comparison remains:

- C1 vs M1 under P0
- C1 vs M1 under PR

with the frozen ownership-by-contention interaction estimand.

No Phase 3B scientific endpoint changes under this amendment.

---

## 10. Static-preparation requirements

The next static stage must now freeze:

1. XG1 9001..9600 outcome-blind six-cell base;
2. prospective seven-cell labeled training view;
3. exact four-cell stressor-domain membership;
4. A_IDENTITY anchor eligibility on all four stressor-domain cells;
5. ordinary tokenizer/serialization validity on all seven cells;
6. seed-16384 pair split;
7. XG1 9601..9900 outcome-blind confirmatory cohort;
8. all physical and semantic hashes.

No model forward is authorized.

---

## 11. Superseded statements

This amendment supersedes only statements in the Phase 3 parent design and
training-label amendment that require PP3/PP5 stressor application on every
training row or require PP3 target-anchor qualification for all seven labeled
cells.

It does not supersede:

- seven-cell label semantics;
- training population;
- confirmatory population;
- optimizer envelope;
- contention thresholds;
- R22/C22 identity;
- Phase 3A stop rule;
- Phase 3B confirmatory interaction.

---

## 12. Final disposition

TRAINING_VIEW =
`SEVEN_CELL`

STRESSOR_DOMAIN =
`FOUR_FROZEN_XG1_CAUSAL_BRANCH_CELLS`

STRESSOR_DOMAIN_VALIDATED_CONTEXT_ONLY =
`YES`

NEW_CAUSAL_INTERVENTION_CONTEXT_INTRODUCED =
`NO`

PHASE3_RESPONSE_OBSERVED_BEFORE_CORRECTION =
`NO`

NEXT_STAGE =
`GEN5_PHASE3_XG1_9001_9900_STATIC_PREPARATION`

STATUS =
`CANDIDATE_FOR_FREEZE`
