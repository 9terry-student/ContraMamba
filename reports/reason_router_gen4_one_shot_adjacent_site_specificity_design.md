# ContraMamba Gen4 Experiment 5 — One-Shot Adjacent-Site Specificity Design

## Status

`EXPERIMENT_5_DESIGN_ONLY`

This document freezes the scientific question and response boundary for the final
post-synthesis robustness/specificity experiment.

It does **not** authorize execution by itself.

No XG1 response may be inspected before the fresh structural cohort and the
implementation are frozen.

---

## 1. Scientific question

Primary objection:

> The Mamba-1.4B core effect may be an accidental consequence of the single chosen
> homologous layer triplet rather than a site-specific causal effect.

This experiment asks whether the already identified canonical Mamba-1.4B site carries
a stronger frozen core signal than exactly one predeclared adjacent homologous site
under matched measurement semantics.

This is a **one-shot specificity study**, not a layer sweep.

Experiment 5 cannot rescue any failure in Experiments 1–3.

---

## 2. Frozen model and prior causal object

Backbone:

`state-spaces/mamba-1.4b-hf`

Revision:

`6e46eae61c27280517feef46f536d16b91076f08`

Frozen arm:

`G3-GROUP-D-HALF`

Training seed:

`181`

Compact downstream checkpoint SHA256:

`915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a`

Source full checkpoint SHA256:

`2383cb212f7e87be1ac8920c7875602a73a1c6031212df17342457acb8b87555`

Frozen canonical geometry execution:

`g4k-mamba14b-geometry-xg2xg4-2gpu-c758d5e-retry1`

Frozen canonical geometry execution HEAD:

`c758d5e81b0ad18f9993846789b28efdad53e3e9`

Frozen canonical selected causal candidate:

`P5`

Frozen canonical response-blind control:

`P4`

Frozen finite-difference scale:

`epsilon = 0.025`

Frozen anchor semantics:

`A_IDENTITY`

Frozen target token offset:

`+2`

The selected causal candidate is **not reopened** in Experiment 5.

---

## 3. Canonical and adjacent sites

### 3.1 Canonical site

The frozen 48-layer homologous mapping is:

- source block: `33`;
- target residual layer: `34`;
- intervention layer: `35`.

Canonical triplet:

`(33, 34, 35)`

### 3.2 One-shot adjacent site

The adjacent triplet is frozen by an architecture/instrumentation-only rule:

> Shift the entire canonical triplet exactly one block in the downstream (`+1`)
> direction while preserving relative offsets `(-2,-1,0)`.

Adjacent triplet:

`(34, 35, 36)`

### 3.3 Why `+1` is frozen

The direction is selected without causal-response inspection.

The already frozen stagewise localization instrumentation validates and captures the
downstream segment beginning at `pre_block_35` and continuing through
`post_block_35`, `post_block_36`, ..., `post_block_47`.

Therefore the `+1` triplet remains inside the already exercised downstream
capture/replay region.

The alternative `-1` triplet `(32,33,34)` would require extending the validated
instrumentation upstream beyond that frozen stagewise capture boundary.

This static implementation-reuse asymmetry, not causal response, fixes the direction
to `+1`.

No second adjacent site is allowed if the `+1` site fails.

---

## 4. Fresh XG1 specificity cohort

Use exactly:

`xg1_fact_5101..xg1_fact_5400`

with:

`N = 300`

The preceding frozen shared behavioral cohort is:

`xg1_fact_4801..xg1_fact_5100`

and its structural manifest records zero pair, claim, and evidence overlap with prior
coverage `xg1_fact_001..xg1_fact_4800`.

Therefore Experiment 5 extends the deterministic XG1 sequence prospectively to the
next disjoint 300-pair block.

Before any scientific response execution, the new cohort must be materialized and
statically prove:

- exact pair order `5101..5400`;
- `300` source pairs;
- `1800` six-cell rows;
- zero pair overlap with `001..5100`;
- zero claim overlap with `001..5100`;
- zero evidence overlap with `001..5100`;
- labels absent from the structural source;
- response fields absent;
- tokenizer/anchor eligibility under the frozen Mamba-1.4B tokenizer;
- no response-based selection.

If any freshness gate fails, Experiment 5 is blocked.

Do not substitute another range after response inspection.

---

## 5. Geometry reconstruction and comparability

### 5.1 Canonical geometry

The canonical `(33,34,35)` geometry is reused from the frozen geometry artifact.

It is **not rerun**.

The canonical geometry used:

- XG2 `xg2_fact_301..xg2_fact_600`, `N=300`;
- XG4 `xg4_fact_301..xg4_fact_600`, `N=300`;
- five basis directions per family;
- five principal planes;
- canonical strong-coordinate dimension `829`.

### 5.2 Adjacent geometry

The adjacent `(34,35,36)` site receives one independent geometry reconstruction using
the **same already frozen XG2 and XG4 structural cohorts and the same response-blind
geometry procedure**.

No XG1 scientific response may be accessed during adjacent geometry preparation.

The adjacent strong-coordinate dimension is allowed to differ from `829`.
It is an observed architecture-local quantity and must be recorded, not tuned.

The adjacent geometry must satisfy the same structural gates:

- correct Mamba-1.4B layer count;
- exact adjacent triplet `(34,35,36)`;
- valid strong partition;
- nonzero finite geometry norms;
- five nondegenerate XG2 basis directions;
- five nondegenerate XG4 basis directions;
- five valid principal planes;
- finite positive `lambda_plus` values;
- orthonormality/eigenvector closure gates;
- no causal-response access;
- no control or plane selection from XG1 responses.

If the adjacent geometry fails these gates, Experiment 5 is **blocked**.
Do not switch to `-1`, another layer, another token, another epsilon, or another
population.

### 5.3 Adjacent causal candidate

The causal candidate rank is fixed before adjacent response execution:

`P5`

Experiment 5 does not perform adjacent-site causal discovery.

The same-numbered adjacent `P5` is a **rank-aligned site-local plane**, not a claim of
semantic identity with canonical `P5`.

### 5.4 Adjacent response-blind control

After adjacent geometry reconstruction, choose exactly one control with the same
frozen geometry-only rule used by the 1.4B discovery:

`c_adj = argmax_{k != P5} lambda_plus_adj,k`

Requirements:

- control selection uses geometry only;
- no XG1 response is available;
- the maximizer must be unique;
- no alternative control may be tried after responses.

Canonical control remains frozen `P4`.

---

## 6. Matched site measurement

On every fresh pair `i`, measure both sites using otherwise matched semantics.

For both sites:

- same Mamba-1.4B checkpoint;
- same tokenizer revision;
- same XG1 pair and six-cell structure;
- same `A_IDENTITY` anchor;
- same target offset `+2`;
- same `epsilon=0.025`;
- same XG2/XG4 direction count `K=5`;
- same central-difference construction;
- same `Q = E_XG2 - E_XG4` definition;
- same dominant-restored semantics;
- same matched response-blind control semantics;
- no task-head or behavioral endpoint is part of the primary specificity test.

Canonical pair-level core signal:

`D_CAN,i = Q_restored,canonical(P5,i) - Q_control,canonical(P4,i)`

Adjacent pair-level core signal:

`D_ADJ,i = Q_restored,adjacent(P5,i) - Q_control,adjacent(c_adj,i)`

Primary paired specificity contrast:

`S_i = D_CAN,i - D_ADJ,i`

Both `D_CAN,i` and `D_ADJ,i` are measured on the **same fresh pair i**.

Do not compare the fresh adjacent cohort against historical canonical response values.

---

## 7. Primary inference

Exactly one inferential p-value is allowed.

Primary test:

- endpoint: `S_i`;
- test: one-sample Student t-test on the paired differences;
- null: `E[S] <= 0`;
- alternative: `E[S] > 0`;
- tail: one-sided greater;
- alpha: `0.05`;
- `N=300`;
- p-value count: `1`;
- multiplicity correction: none because the inferential family contains one test.

This is equivalent to a paired Student t-test between fresh `D_CAN` and fresh
`D_ADJ`.

No alternative tail is allowed.

---

## 8. Specificity decision rule

Experiment 5 supports the site-specificity claim only if **both** gates hold:

1. fresh canonical sign gate:
   `mean(D_CAN) > 0`;
2. paired specificity gate:
   `mean(S) > 0` and one-sided Student `p < 0.05`.

The canonical sign gate adds **no additional p-value**.

Success label:

`MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_SUPPORTED`

Failure label:

`MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_NOT_ESTABLISHED`

If the paired test passes but `mean(D_CAN) <= 0`, specificity is not established.

If canonical is positive but the paired test fails, specificity is not established.

No second adjacent site may rescue failure.

---

## 9. Descriptive outputs

Report descriptively, without changing the primary decision:

- mean, SD, 95% t-CI, Cohen's `dz`, positive fraction for `D_CAN`;
- same quantities for `D_ADJ`;
- mean, SD, 95% t-CI, Cohen's `dz`, positive fraction for `S`;
- adjacent geometry strong dimension;
- adjacent `lambda_plus` by plane;
- frozen adjacent geometry-only control identity;
- canonical and adjacent Q components;
- finite/nonfinite counts;
- forward accounting;
- intervention and reconstruction audit maxima.

Do not add exploratory p-values.

---

## 10. Forward budget

Adjacent response-blind geometry preparation:

- XG2: `1200` model forwards;
- XG4: `1200` model forwards;
- adjacent geometry total: `2400`.

Fresh XG1 site measurement:

Existing 1.4B core protocol requires:

- 2 conditions per site;
- 10 directions per condition;
- 4 model forwards per direction;
- `80` model forwards per pair per site.

For `300` pairs:

- canonical: `24,000`;
- adjacent: `24,000`;
- paired site measurement total: `48,000`.

Total new scientific model forwards for Experiment 5:

`50,400`

Canonical frozen geometry rerun count:

`0`

No additional layer, token, epsilon, plane, control, or cohort execution is allowed.

---

## 11. Execution and analysis boundaries

### Static / CPU-only phases

Allowed before GPU execution:

- design freeze;
- fresh XG1 `5101..5400` materialization;
- freshness audit;
- implementation;
- tests;
- exact checkpoint/snapshot validation;
- forward-budget validation;
- output-schema validation.

### Adjacent geometry execution

GPU execution is allowed only after:

- this design is frozen;
- implementation is frozen;
- the run is pinned to an exact clean commit.

This phase may access only the frozen XG2/XG4 geometry cohorts.

It must not access XG1 specificity responses.

Freeze the adjacent geometry artifact before implementing or executing the XG1 paired
specificity run.

### Paired specificity execution

GPU execution is allowed only after:

- fresh XG1 `5101..5400` structural cohort is frozen;
- adjacent geometry artifact and geometry-only control are frozen;
- paired runner implementation is frozen;
- exact run command is pinned to a clean commit.

Raw execution must not perform primary inference.

Primary inference is CPU-only and runs once from frozen raw artifacts.

---

## 12. Prohibited rescue and search

The following are prohibited after this design is frozen:

- second adjacent site;
- switching from `+1` to `-1`;
- layer sweep;
- token sweep;
- epsilon sweep;
- P5 reselection from adjacent causal responses;
- alternative control selected from XG1 responses;
- another fresh range after inspecting `5101..5400`;
- row-subset rescue;
- alternative tail;
- additional primary endpoint;
- steering-based reinterpretation;
- rescue of Experiments 1–3.

---

## 13. Claim boundary

If supported, the allowed claim is narrow:

> On the prospectively frozen Mamba-1.4B XG1 `5101..5400` cohort, the canonical
> `(33,34,35)` homologous site carried a stronger positive rank-aligned P5 core signal
> than the single architecture-predeclared adjacent `+1` site `(34,35,36)` under the
> matched frozen measurement procedure.

This does **not** establish:

- uniqueness across all layers;
- a global layer optimum;
- absence of causal signal at every other layer;
- semantic identity of P5 across sites;
- architectural universality;
- improved downstream task behavior;
- useful steering;
- external natural-language transfer beyond already frozen results.

---

## 14. Immediate next subphase

After this design is frozen:

`EXPERIMENT_5_FRESH_COHORT_AND_ADJACENT_GEOMETRY_STATIC_IMPLEMENTATION`

The next work is CPU/static only.

It must:

1. materialize and freeze XG1 `5101..5400`;
2. implement adjacent `(34,35,36)` geometry preparation using the frozen XG2/XG4
   cohorts;
3. validate that no XG1 response path is reachable from geometry preparation;
4. validate exact `2400` adjacent-geometry forward budget;
5. stop before GPU execution.

No Kaggle run is authorized by this design freeze alone.
