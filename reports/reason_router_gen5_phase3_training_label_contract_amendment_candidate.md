# ContraMamba Gen5 Phase 3
## Training Label Contract Amendment Candidate

### Status

AMENDMENT =
`GEN5_PHASE3_TRAINING_LABEL_CONTRACT_CORRECTION`

PARENT_DESIGN_COMMIT =
`ae52dbf98cce21226036ae38121edcdfa6f79d7b`

PARENT_DESIGN =
`reports/reason_router_gen5_phase3_causal_role_contention_identification_static_design_candidate.md`

STATUS =
`CANDIDATE_FOR_FREEZE`

IMPLEMENTATION =
`NOT_AUTHORIZED`

TRAINING =
`NOT_AUTHORIZED`

MODEL_EXECUTION =
`NOT_AUTHORIZED`

CUDA =
`NOT_AUTHORIZED`

This amendment corrects a static design incompatibility discovered before
Phase 3 data preparation or scientific execution.

No Phase 3 model response, contention value, causal endpoint, or p-value has
been observed.

---

## 1. Defect

The frozen Phase 3 design specifies:

`FINAL_3WAY_CROSS_ENTROPY_ONLY`

on the fresh XG1 contention-training population:

`xg1_fact_9001..xg1_fact_9600`.

However, the frozen XG1 six-cell lineage is intentionally outcome blind.

Its row schema contains no `final_label`.

The XG1 generator explicitly forbids fields including:

- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`

Therefore the original Phase 3 training contract cannot be executed literally.

Silently assigning labels during implementation is forbidden.

---

## 2. Correction principle

The original outcome-blind XG1 six-cell artifacts remain unchanged.

Phase 3 static preparation will construct a separate deterministic:

`TRAINING_ONLY_LABELED_VIEW`

for `xg1_fact_9001..xg1_fact_9600`.

The training view is derived only from:

- frozen structured source facts;
- frozen XG1 cell identity;
- one prospectively defined explicit-denial cell.

No model output or scientific response participates in label construction.

The fresh Phase 3 confirmatory population:

`xg1_fact_9601..xg1_fact_9900`

remains the original unlabeled six-cell XG1 structure.

---

## 3. Existing six-cell identities

The inherited XG1 cells are:

1. `C0_SHAM`
2. `C1_TITLE`
3. `C2_NAME`
4. `C3_ROLE`
5. `C4_PREDICATE`
6. `C5_TITLE_NAME`

Their original claim/evidence construction is unchanged.

Their Phase 3 training labels are frozen as:

`C0_SHAM -> SUPPORT`

and:

`C1_TITLE -> NOT_ENTITLED`
`C2_NAME -> NOT_ENTITLED`
`C3_ROLE -> NOT_ENTITLED`
`C4_PREDICATE -> NOT_ENTITLED`
`C5_TITLE_NAME -> NOT_ENTITLED`

The predicate-substitution cell is NOT relabeled REFUTE.

A changed predicate is not prospectively assumed to be logical negation.

---

## 4. New explicit REFUTE cell

Phase 3 training only adds:

`C6_EXPLICIT_DENIAL`

For each structured source fact, retain the exact original:

- title;
- name;
- role;
- predicate;
- object;
- time;
- location.

The claim remains the exact native XG1 positive claim.

The evidence is deterministically rendered as:

`During {time}, records from {location} identify {title} {name} as {role}; the record explicitly denies that this person {predicate} {object}.`

This cell changes no entity, role, object, time, or location identity.

Its only intended task semantics are explicit denial of the positive claim.

Frozen label:

`C6_EXPLICIT_DENIAL -> REFUTE`

No alternate predicate is used for this cell.

No linguistic variant search is allowed.

---

## 5. Final-label mapping

Phase 3 inherits the historical final-label IDs exactly:

`REFUTE = 0`

`NOT_ENTITLED = 1`

`SUPPORT = 2`

Per training pair the frozen class composition is:

- REFUTE: `1`
- NOT_ENTITLED: `5`
- SUPPORT: `1`

No class reweighting is authorized in the first Phase 3 experiment.

No oversampling is authorized.

No label smoothing is authorized.

---

## 6. Revised training cardinalities

Contention-training source population:

`xg1_fact_9001..xg1_fact_9600`

SOURCE_PAIR_COUNT =
`600`

TRAINING_ROWS_PER_PAIR =
`7`

TOTAL_TRAINING_VIEW_ROWS =
`4200`

Pair-level split remains:

SPLIT_SEED =
`16384`

TRAIN_PAIR_COUNT =
`480`

DEV_PAIR_COUNT =
`120`

Therefore:

TRAIN_ROW_COUNT =
`3360`

DEV_ROW_COUNT =
`840`

Expected train-label counts:

- REFUTE: `480`
- NOT_ENTITLED: `2400`
- SUPPORT: `480`

Expected dev-label counts:

- REFUTE: `120`
- NOT_ENTITLED: `600`
- SUPPORT: `120`

The exact pair split must be frozen before model execution.

---

## 7. Static artifact separation

The Phase 3 training population must preserve both:

### A. Outcome-blind base artifact

Exact six-cell XG1:

`600 pairs x 6 = 3600 rows`

This artifact contains no final labels.

### B. Training-only labeled view

Derived seven-cell training artifact:

`600 pairs x 7 = 4200 rows`

This artifact may contain only the additional task-training fields required by
this amendment, including the frozen final label.

The labeled view must retain source pair identity and cell identity so every row
can be traced to its deterministic source fact.

The outcome-blind base artifact remains the canonical XG1 scientific lineage.

---

## 8. Tokenizer and target-anchor gate

All seven training cells must independently pass the frozen tokenizer and
target-anchor eligibility contract before any Phase 3 training is authorized.

In particular `C6_EXPLICIT_DENIAL` must satisfy the same target identity/name
coordinate requirements used by the inherited PP3 intervention semantics.

Static preparation must report separately:

- six-cell base eligibility;
- C6 explicit-denial eligibility;
- complete seven-cell pair eligibility.

If fewer than all 600 training pairs are eligible, Phase 3 blocks.

No pair dropping is allowed.

No shortened post-window rescue is allowed.

---

## 9. Scientific objective remains unchanged

After this correction, Phase 3 still uses:

`FINAL_3WAY_CROSS_ENTROPY_ONLY`

No R22 or causal endpoint enters the loss.

Forbidden training targets remain:

- R22 coefficients;
- C22 coefficients;
- Q;
- D_NEC22;
- D_SUF22;
- restoration integrity;
- PP3 response magnitude;
- contention score.

The training labels are defined entirely before any model execution.

---

## 10. Pressure conditions remain unchanged

The three frozen Phase 3 pressure conditions remain:

`P0`
`PR`
`PC`

where:

- P0 = no internal stressor;
- PR = frozen layer-17 PP3 causal-role stressor;
- PC = frozen layer-17 PP5 matched-control stressor.

Every pressure condition receives the exact same seven-cell labeled rows.

Therefore pressure condition is independent of task-label construction.

---

## 11. Phase 3A gate remains prospective

The previously frozen Phase 3A contention qualification remains unchanged
except for the corrected training-row cardinalities.

No contention threshold is changed.

No ownership endpoint is changed.

No seed is changed.

No stressor is changed.

No owner/control geometry is changed.

If PR fails the frozen contention gate, Phase 3B remains blocked.

---

## 12. Confirmatory population remains outcome blind

The prospective confirmatory population remains:

`xg1_fact_9601..xg1_fact_9900`

PAIR_COUNT =
`300`

ROWS_PER_PAIR =
`6`

ROW_COUNT =
`1800`

No training labels are added to this population.

It may not be used for training, dev selection, contention qualification, or
hyperparameter selection.

The ownership-contention interaction estimand and single confirmatory p-value
defined by the parent Phase 3 design remain unchanged.

---

## 13. Superseded parent-design fields

This amendment supersedes only parent-design statements asserting:

- `600 x 6 = 3600` as the actual optimization row count;
- `2880 / 720` as Phase 3 training/dev row counts;
- direct use of unlabeled six-cell XG1 rows as CE targets.

They become respectively:

- outcome-blind base: `3600` rows;
- labeled training view: `4200` rows;
- optimization train/dev: `3360 / 840`.

All other Phase 3 scientific semantics remain binding.

---

## 14. Immediate next stage

After this amendment is frozen:

`GEN5_PHASE3_XG1_9001_9900_STATIC_PREPARATION`

may proceed.

Static preparation must freeze:

1. 600-pair outcome-blind six-cell training base;
2. 600-pair seven-cell labeled training view;
3. seed-16384 exact pair split;
4. deterministic label counts;
5. 300-pair outcome-blind six-cell confirmatory cohort;
6. tokenizer/anchor eligibility;
7. all physical and semantic hashes.

It must execute:

- zero scientific model forwards;
- zero checkpoint loads;
- zero CUDA;
- zero training;
- zero backward;
- zero p-values.

---

## 15. Final amendment disposition

DESIGN_DEFECT =
`UNLABELED_XG1_CANNOT_DIRECTLY_SUPPORT_FINAL_3WAY_CE`

CORRECTION =
`SEPARATE_PROSPECTIVE_SEVEN_CELL_LABELED_TRAINING_VIEW`

ORIGINAL_XG1_MUTATED =
`NO`

CONFIRMATORY_COHORT_LABELS_ADDED =
`NO`

MODEL_RESPONSE_USED_FOR_LABELS =
`NO`

POSTHOC_SCIENTIFIC_RESPONSE_USED =
`NO`

STATUS =
`CANDIDATE_FOR_FREEZE`
