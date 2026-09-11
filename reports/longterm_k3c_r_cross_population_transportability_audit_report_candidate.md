# K3C R Endpoint Cross-Population Transportability Audit Report Candidate

**Status:** READ-ONLY POST-HOC FAILURE-ANALYSIS / HYPOTHESIS-GENERATION REPORT.

**Authority boundary:** NOT execution authority. NOT a K3C amendment. NOT a K3C rescue.

This report closes the read-only analysis of why the preregistered K3C BASE replication gate failed on endpoint R.

No new model execution, recurrent-state read, intervention execution, training, probe fitting, or K3C rerun was performed during this audit.

## 1. Governing closed result

K3C closure commit:

`04b089e5424e1debe669b3ea809d7ab5c4efd111`

K3C closure report SHA256:

`9d5965853b9f30f6863a1b16c460d1478c760573fd057204556b9e625a207390`

K3C scientific verdict:

`INCONCLUSIVE_DUE_TO_BASELINE_REPLICATION_FAILURE`

The failed BASE endpoint was R.

K3C BASE R:

- positive blocks = 78;
- negative blocks = 68;
- zero blocks = 4;
- n_eff = 146;
- rank-biserial sign effect = `+0.0684931506849315`;
- Holm-adjusted p = `0.45648223496512574`;
- direction match = NO;
- directional contradiction = NO.

D, DISP, and P reproduced.

This audit does not alter that frozen result.

## 2. Frozen comparison population

The comparison is the previously validated K2R prospective replication population.

K2R candidate-pool SHA256:

`00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4`

K2R item-metrics SHA256:

`d43c8e739dda3b6cc018a14dc965c383190ed62fd20cfee84c51d55867965498`

K3C candidate-pool SHA256:

`9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e`

K3C item-metrics SHA256:

`c9c227bd3a7e208cce2d5ab0edc0c1eaf37ddcd1053a5f228ffb823ae4aab50e`

All analyses below reused frozen item/block quantities only.

## 3. Audit 1: construction-composition analysis

Audit script SHA256:

`aaf9ea39c2d5361098fd0c1bc19b48f3f692dddf0ad16788be281ea1282f3a63`

Inference status:

`POST_HOC_HYPOTHESIS_GENERATION_ONLY`

### 3.1 Overall R block effect

K2R:

- positive = 84;
- negative = 53;
- zero = 13;
- n_eff = 137;
- sign effect = `+0.22627737226277372`;
- median B_R = `+0.5207228781912931`.

K3C:

- positive = 78;
- negative = 68;
- zero = 4;
- n_eff = 146;
- sign effect = `+0.0684931506849315`;
- median B_R = `+0.11730368260602475`.

The positive R pair-specificity signal weakened substantially.

### 3.2 Correction-source composition

Both populations had exactly:

- `none = 150`;
- `polarity_flip = 150`.

Therefore correction-source marginal composition did not change.

### 3.3 Divergence-timing composition

Both populations had exactly:

Matched d-p:

- 2 = 150;
- 3 = 150.

Swapped d-p:

- 2 = 150;
- 3 = 150.

At reciprocal-block level, both populations also had the same matched and swapped d-pattern counts:

- `2+2 = 38`;
- `2+3 = 74`;
- `3+3 = 38`.

Therefore d-p marginal or block-pattern composition did not change.

### 3.4 Same d-pattern effect weakening

The most important mixed d-pattern stratum was:

`2+3`

K2R:

- n = 74;
- positive = 46;
- negative = 28;
- sign effect = `+0.24324324324324326`.

K3C:

- n = 74;
- positive = 38;
- negative = 36;
- sign effect = `+0.02702702702702703`.

Thus R weakened within an exactly equally represented frozen divergence-timing block stratum.

This rules out simple d-pattern frequency shift as a sufficient explanation.

## 4. Audit 2: exact prefix-pattern standardization

Audit script SHA256:

`c0ea1e3d78c48d543bf354faa53544a2157f57644b76b049bc938ea44de5439f`

Inference status:

`POST_HOC_HYPOTHESIS_GENERATION_ONLY`

The audit used exact reciprocal-block prefix endpoint patterns:

`(p_a, p_b)`

No optimized threshold, favorable binning, or endpoint selection was performed.

### 4.1 Common exact prefix-pattern support

Exact prefix patterns observed in both populations:

`25`

K2R blocks inside common exact-p support:

`116 / 150`

K3C blocks inside common exact-p support:

`114 / 150`

### 4.2 Common-support mean sign

K2R common-support mean sign:

`+0.10344827586206896`

K3C common-support mean sign:

`-0.05263157894736842`

Thus restricting both populations to exact prefix-pattern support did not restore K3C R.

### 4.3 K3C standardized to K2R prefix-pattern composition

K3C within-pattern effects reweighted to the K2R exact `(p_a,p_b)` composition:

`-0.04922469771607702`

This remained near the weak/negative K3C common-support value and did not approach the positive K2R common-support value.

Therefore exact prefix-pattern composition alone is insufficient to explain the R replication failure.

### 4.4 Reverse standardization

K2R within-pattern effects reweighted to the K3C exact prefix-pattern composition:

`+0.16546708651971812`

K2R remained positive under the K3C p-pattern mixture.

This further argues against prefix-pattern composition as the sole driver of the cross-population difference.

## 5. Exact joint matching boundary

The stronger post-hoc joint signature combined:

- exact p pattern;
- correction-source pattern;
- matched d pattern;
- swapped d pattern.

Common joint strata:

`37`

K2R common-support blocks:

`73 / 150`

K3C common-support blocks:

`54 / 150`

Within this restricted subset:

K2R mean sign:

`-0.0547945205479452`

K3C mean sign:

`+0.018518518518518517`

K3C joint effects reweighted to K2R joint composition:

`+0.0273972602739726`

K2R joint effects reweighted to K3C joint composition:

`+0.06922398589065255`

This matching becomes too sparse and selective to identify a clean within-stratum transportability mechanism.

Most importantly, the restricted common-support subset no longer carries the original positive K2R R effect.

Therefore further post-hoc exact matching is not scientifically identified as a route to the cause of the R failure.

## 6. Generator structure relevant to interpretation

The frozen controlled generator constructs generated facts deterministically from periodic lexical factor sequences.

Generated-factor cardinalities include:

- names: 12;
- titles: 6;
- roles: 6;
- predicate pairs: 7;
- times: 6;
- locations: 8.

Ignoring the unique numeric object suffix, the full generated lexical factor phase repeats every:

`LCM(12, 6, 7, 8) = 168`

generated templates.

K2R used global template indices:

`[300:600]`

K3C used:

`[600:900]`

The slice-start shift is:

`300 mod 168 = 132`

Each 300-item slice contains every one of the 168 lexical phase classes at least once, but the 132 phase classes receiving a second occurrence differ between the two slices.

The generated object text also contains the unique fact number, so exact token realizations are not periodic even when the nonnumeric factor phase repeats.

This means K2R and K3C are claim-disjoint and generated by the same deterministic mechanism, but they are not exact distributional duplicates in lexical/token realization.

## 7. Supported audit conclusions

The following conclusions are supported at hypothesis-generation level.

### 7.1 Simple correction-source composition is not the explanation

Both populations have the same 150/150 correction-source balance.

### 7.2 Simple divergence-timing composition is not the explanation

Both populations have the same item-level and block-level d-p composition.

R weakens substantially inside the equally represented `2+3` block pattern.

### 7.3 Exact prefix-length composition is not sufficient

Reweighting K3C to the K2R exact prefix-pattern mixture does not recover the K2R R signal.

### 7.4 The available artifacts do not identify one lexical cause

Exact high-dimensional matching loses too much support and removes the original K2R signal itself.

Therefore it is not justified to claim that a specific title, predicate, number length, token length, source class, or lexical factor caused the failure.

### 7.5 R is presently a non-transportable endpoint across these two prospective claim slices

The strongest defensible scientific characterization is:

`R_CROSS_CLAIM_TRANSPORTABILITY_NOT_ESTABLISHED`

This is stronger than saying only that K3C happened to miss significance.

The same R construction moved from a replicated K2R effect of approximately `+0.226` to a weak K3C effect of approximately `+0.068`, despite identical correction-source and d-pattern composition and despite prefix-pattern standardization failing to recover the earlier signal.

This does not prove that R is universally unstable.

It establishes that R has not shown the same cross-claim stability as D, DISP, and P in the currently validated controlled-generator evidence.

## 8. Relation to D, DISP, and P

K3C prospectively reproduced:

- D;
- DISP;
- P.

Therefore the current evidence supports a distinction:

- D/DISP/P have stronger cross-claim transportability within the tested controlled-generator regime;
- R has weaker and presently unestablished cross-claim transportability.

This distinction must not be converted into a post-hoc three-of-four K3C success rule.

K3C remains inconclusive exactly as frozen.

## 9. Mechanistic consequence

Because K3C required full BASE replication before mechanism promotion, the R instability prevents the K3C W-vs-H mechanism family from being promoted.

No mechanism conclusion may be rescued by deleting R after the outcome.

The correct mechanistic state remains:

- K3 coefficient-specialization hypothesis: contradicted;
- K3C write-injection / retained-carry hypothesis: inconclusive;
- no replacement mechanism established.

## 10. Stop condition for post-hoc analysis

The R failure analysis stops here.

Do not continue outcome-guided slicing by:

- individual predicate;
- title;
- name;
- location;
- number token;
- hand-selected p range;
- favorable generator phase;
- favorable reciprocal block class;
- alternate endpoint transformation.

Such analyses would create researcher degrees of freedom without a prospective identification strategy.

## 11. Prospective successor question

The next justified scientific question is observational, not mechanistic:

**Does R pair-specific speed geometry transport across a prospectively phase-balanced new generated population when the deterministic 168-phase lexical cycle is exactly balanced?**

This successor should be treated as a new transportability study, not a K3C rescue.

A scientifically natural population size is a multiple of the 168-phase generator cycle.

A two-cycle design:

`336 items = 168 reciprocal blocks`

would exactly balance the nonnumeric generated lexical phase twice while preserving a block count comparable to prior 150-block studies.

The exact future population recipe, tokenizer feasibility, stable-ID pairing, test family, and whether D/DISP/P are controls or reference endpoints must be prospectively frozen before any recurrent-state read.

No such execution is authorized by this report.

## 12. Authority state

`K3C_CLOSED = YES`

`K3C_RERUN_AUTHORIZED = NO`

`R_FAILURE_READ_ONLY_AUDIT_COMPLETE = YES`

`R_SIMPLE_SOURCE_COMPOSITION_EXPLANATION = NOT_SUPPORTED`

`R_SIMPLE_D_TIMING_COMPOSITION_EXPLANATION = NOT_SUPPORTED`

`R_EXACT_PREFIX_PATTERN_COMPOSITION_EXPLANATION = NOT_SUFFICIENT`

`R_SPECIFIC_LEXICAL_CAUSE_IDENTIFIED = NO`

`R_CROSS_CLAIM_TRANSPORTABILITY_ESTABLISHED = NO`

`R_CROSS_CLAIM_TRANSPORTABILITY_NOT_ESTABLISHED = YES`

`PROSPECTIVE_R_TRANSPORTABILITY_PREREGISTRATION_MOTIVATED = YES`

`PROSPECTIVE_R_TRANSPORTABILITY_EXECUTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

This report authorizes drafting a prospective phase-balanced R transportability preregistration only.

It does not authorize implementation or scientific execution.
