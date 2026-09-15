# Gen4 × K Generator-Family Prevalence Transportability
## Validated Evidence Analysis Report Candidate

**Status:** STATIC SCIENTIFIC CLOSURE CANDIDATE  
**Evidence freeze commit:** `97ab2a7e6182645c6fb234928bf1e1e0aa0e8146`  
**Execution implementation HEAD:** `f18152e5c33071911591c682e8b3cac47f121118`  
**Branch:** `gen4-k-xg1-cross-generator-replication`  
**Scope:** frozen baseline prevalence evidence only; this document does not authorize any new execution.

---

## 1. Purpose

This report closes the validated-evidence interpretation of the **Gen4 × K Generator-Family Prevalence Transportability** stage using only the frozen XG2/XG4 full-baseline artifacts committed at `97ab2a7`.

The stage asks a narrow question:

> Does the already-frozen baseline discriminator produce enough LARGE and SMALL examples, within independent eligible synthetic generator families, to satisfy the inherited minimum group-size requirement for a later 300-pair response-replication design?

This is **not** an intervention-response experiment. It does not measure `R_ALIGN`, causal response, task-head behavior, logits, training behavior, or any downstream replication endpoint.

---

## 2. Frozen scientific contract

The discriminator was frozen before these generator-family baseline runs.

```text
alignment_shift_abs = abs(reference_C - target_C)

LARGE iff alignment_shift_abs >= 0.11228626366380845
SMALL iff alignment_shift_abs <  0.11228626366380845
```

Frozen geometry:

```text
target pair    = C2_NAME - C0_SHAM
reference pair = C5_TITLE_NAME - C1_TITLE
layers         = 15 -> 16 -> 17
```

The inherited minimum viable group size is:

```text
n_LARGE >= 30
n_SMALL >= 30
```

Each eligible family contains 300 frozen source pairs. The full-baseline scientific budget is four baseline cells per pair:

```text
300 pairs × 4 baseline cells = 1200 model forwards per eligible family
```

No family may be repaired, enriched, merged, or selectively resampled after observing outcomes. The threshold may not be re-estimated or swept.

---

## 3. Eligibility boundary

Exactly three generator families belong to this stage:

```text
XG2
XG3
XG4
```

### XG2

XG2 is structurally compatible with the frozen identity topology and passed tokenizer/anchor eligibility for all 300 pairs.

```text
PASS_300_OF_300
complete pairs = 300
```

### XG3

XG3 is structurally incompatible with the frozen contiguous identity topology:

```text
A_IDENTITY = title -> name
```

Its renderer places the relevant identity components in an incompatible topology. The frozen structural audit recorded 1200 anchor-relevant topology failures, and the tokenizer/anchor eligibility result was:

```text
BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY
complete pairs = 0
```

Therefore **no XG3 model forward is permitted or interpreted in this stage**. XG3 was not repaired, replaced, or rescued after observing this failure.

### XG4

XG4 is structurally compatible with the frozen identity topology and passed tokenizer/anchor eligibility for all 300 pairs.

```text
PASS_300_OF_300
complete pairs = 300
```

This eligibility boundary predates the XG2/XG4 full-baseline prevalence outcomes.

---

## 4. Execution and provenance boundary

The full-baseline runner was frozen at:

```text
f18152e5c33071911591c682e8b3cac47f121118
```

The runner executes one eligible family per invocation and reuses the already-validated baseline geometry path. Each family is limited to exactly 1200 scientific baseline forwards.

The preceding bounded CUDA equivalence evidence was frozen at:

```text
71f4e5106e36d36003e49e7424ab98d3df0dbe90
```

with gate execution HEAD:

```text
e646fc900d12b42621a6a58ee31eff255bb52bfd
```

Representative checkpoint identity:

```text
seed180 / G3-GROUP-D-HALF
SHA256 = 1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f
```

Both full-baseline summaries assert:

```text
baseline_only = true
alignment_intervention_executed = false
magnitude_intervention_executed = false
response_endpoint_computed = false
task_heads_executed = false
logits_read = false
training_executed = false
backward_executed = false
threshold_reestimated = false
xg3_model_forward_count = 0
```

The frozen evidence was committed together at:

```text
97ab2a7e6182645c6fb234928bf1e1e0aa0e8146
```

---

## 5. Validated family-level prevalence results

| Family | Pairs | Baseline forwards | n_LARGE | p_LARGE | n_SMALL | p_SMALL | Minimum group size | Family result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| XG2 | 300 | 1200 | 94 | 0.313333 | 206 | 0.686667 | 30 | **VIABLE** |
| XG4 | 300 | 1200 | 56 | 0.186667 | 244 | 0.813333 | 30 | **VIABLE** |

### XG2

Frozen artifact root:

```text
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r2/xg2/
```

Result:

```text
n_LARGE = 94
n_SMALL = 206
p_LARGE = 0.31333333333333335
p_SMALL = 0.6866666666666666
VIABLE = true
family_level_classification = VIABLE
```

The limiting group is LARGE, but it exceeds the inherited minimum by 64 examples.

Artifact-level hashes recorded by the frozen manifest:

```text
baseline_items.jsonl
SHA256 = d52df4ca14a406f9a81cceac6b81c2b31183a119130a20cea450c26152960f06

prevalence_summary.json
SHA256 = a84b2a71a52ce9edd67ce6de260bb1b945574369dad289f34f9cd0231564b888
```

### XG4

Frozen artifact root:

```text
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r1/xg4/
```

Result:

```text
n_LARGE = 56
n_SMALL = 244
p_LARGE = 0.18666666666666668
p_SMALL = 0.8133333333333334
VIABLE = true
family_level_classification = VIABLE
```

The limiting group is LARGE, but it exceeds the inherited minimum by 26 examples.

Artifact-level hashes recorded by the frozen manifest:

```text
baseline_items.jsonl
SHA256 = a764787bbb3e00f8511acb2a4251d54069d1f6b2eadc8fa178336eaba412b6ac

prevalence_summary.json
SHA256 = 378b7e658d6029cccf65455924ceddd30081b5ac59e6570cce9f3b07026e0fc9
```

---

## 6. Scientific interpretation

### 6.1 What the evidence establishes

The frozen discriminator produces **non-degenerate, independently viable LARGE/SMALL partitions in both eligible external generator families**.

For XG2, the observed LARGE prevalence is approximately 31.3%. For XG4, it is approximately 18.7%. Despite this descriptive difference, both families independently clear the same predeclared minimum of 30 examples in both groups without threshold adjustment, outcome-driven enrichment, pair replacement, or family merging.

Therefore the stage establishes the following family-level result:

> **The inherited prevalence criterion is feasible in XG2 and XG4 under the unchanged frozen discriminator and baseline geometry.**

This is stronger than merely observing some LARGE examples in each family: both families retain enough observations on both sides of the frozen threshold to support the inherited minimum-size requirement.

The lower LARGE prevalence in XG4 is scientifically relevant as a descriptive family difference, but it does not invalidate viability. XG4 remains 26 LARGE examples above the frozen minimum.

### 6.2 What the evidence does not establish

The evidence does **not** establish that the numeric prevalence itself is invariant across generator families. XG2 and XG4 have visibly different observed LARGE proportions.

It also does not establish:

- response transportability,
- causal intervention-response replication,
- `R_ALIGN` replication,
- task-head or logit behavior,
- training improvement,
- threshold optimality,
- three-family prevalence robustness,
- a pooled generator-family prevalence,
- validity of repaired or redesigned XG3 examples.

No hypothesis about those quantities was executed here.

---

## 7. Cross-family primary classification

The frozen summaries correctly retain:

```text
global_primary_classification = PREVALENCE_TRANSPORTABILITY_INCOMPLETE
```

That classification remains unchanged after the successful XG2 and XG4 runs.

The reason is structural rather than a failed XG2/XG4 prevalence result: XG3 cannot instantiate valid baseline geometry under the frozen topology/anchor contract and therefore contributes no valid prevalence sample.

Accordingly, the final stage interpretation has two levels that must not be conflated:

```text
XG2 family-level prevalence: VIABLE
XG4 family-level prevalence: VIABLE

3-family primary transportability status:
PREVALENCE_TRANSPORTABILITY_INCOMPLETE
```

It would be invalid to relabel the stage as a complete three-family `ROBUST`, `MIXED`, or `SYSTEMATICALLY_LOW` result, because the third prespecified family is structurally unobservable under the frozen contract.

---

## 8. Outcome-blindness and anti-repair checks

The result is interpretable because the critical selection and classification rules were frozen before the full-baseline outcomes.

The evidence set preserves the following constraints:

```text
threshold sweep/re-estimation: NO
pair replacement:             NO
LARGE enrichment:             NO
family merge:                 NO
XG3 repair/rescue:            NO
XG3 model forward:            NO
alignment intervention:       NO
magnitude intervention:       NO
response endpoint:            NO
task-head execution:          NO
logit readout:                NO
training:                     NO
backward:                     NO
```

This prevents the observed XG2/XG4 viability from being attributed to post-outcome cohort manipulation or threshold tuning.

---

## 9. Four-way status separation

### Code correctness

**PASS for the bounded scope used here.**

The full-baseline runner was separately implemented and validated after the bounded CUDA-equivalence path had been frozen.

### Execution success

**PASS for XG2 and XG4 full baseline.**

Each eligible family completed exactly 1200 scientific baseline forwards and produced the required prevalence artifacts.

### Artifact / provenance validity

**FROZEN in repository evidence commit `97ab2a7`.**

The family artifacts include item-level baseline records, summaries, manifests, and checksum files. The manifest hashes above bind the principal scientific files.

### Scientific conclusion

**PARTIAL / FAMILY-LEVEL POSITIVE, GLOBAL INCOMPLETE.**

XG2 and XG4 independently satisfy the frozen prevalence viability criterion. The prespecified three-family primary transportability question remains incomplete because XG3 is ineligible under the frozen topology/anchor contract.

These four statements are distinct. Successful execution does not by itself imply a broader scientific claim.

---

## 10. Final scientific conclusion

The Gen4 × K generator-family prevalence experiment provides a clean positive result for the two independently eligible external families:

```text
XG2: 94 LARGE / 206 SMALL -> VIABLE
XG4: 56 LARGE / 244 SMALL -> VIABLE
```

Thus, under the frozen discriminator

```text
alignment_shift_abs >= 0.11228626366380845
```

both XG2 and XG4 contain sufficient LARGE and SMALL observations to satisfy the inherited minimum group-size requirement of 30 **without any post-outcome adjustment**.

At the same time, the overall prespecified three-family transportability question is not complete. XG3 is structurally incompatible with the frozen identity/anchor topology, has zero complete eligible pairs, and correctly receives zero model forwards.

The appropriate scientific closure is therefore:

```text
FAMILY-LEVEL RESULT:
XG2 = VIABLE
XG4 = VIABLE

GLOBAL PRIMARY RESULT:
PREVALENCE_TRANSPORTABILITY_INCOMPLETE
```

This report closes only the **baseline prevalence** question. It neither performs nor authorizes intervention-response replication, training, backward execution, threshold modification, XG3 rescue, or any other downstream experiment.

---

## 11. Frozen evidence index

### Design / execution context

```text
Design freeze:
4b6f0c831ebc89f0e786e1eb526739c7c9c06413

Structural cohort freeze:
341de2398e1c06fffa73ee323502f52972c2d58e

Tokenizer/anchor implementation freeze:
8df2a5e2cb72f8f4eed33c4996dfba594a1f9687

Tokenizer/anchor artifact freeze:
0af44566eaadc324a6aaf5cec19f3972c8e371ef

Bounded CUDA-equivalence freeze:
71f4e5106e36d36003e49e7424ab98d3df0dbe90

Full-baseline runner:
f18152e5c33071911591c682e8b3cac47f121118

Validated evidence freeze:
97ab2a7e6182645c6fb234928bf1e1e0aa0e8146
```

### XG2 evidence

```text
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r2/xg2/baseline_items.jsonl
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r2/xg2/prevalence_summary.json
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r2/xg2/artifact_manifest.json
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r2/xg2/SHA256SUMS.txt
```

### XG4 evidence

```text
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r1/xg4/baseline_items.jsonl
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r1/xg4/prevalence_summary.json
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r1/xg4/artifact_manifest.json
reports/reason_router_gen4_generator_family_prevalence_full_baseline_f18152e_r1/xg4/SHA256SUMS.txt
```

---

## 12. Closure boundary

This candidate report is an **interpretation artifact**, not an execution authority.

It may be frozen only after ordinary repository review. Nothing in this document changes the already-frozen discriminator, cohort definitions, eligibility rules, runtime identity, threshold, or model checkpoint, and nothing in it authorizes additional model execution.
