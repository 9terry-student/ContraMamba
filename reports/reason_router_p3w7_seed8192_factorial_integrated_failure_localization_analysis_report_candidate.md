# Seed8192 Factorial Integrated Failure-Localization Analysis Candidate

## Verdict and boundary

```text
VERDICT = PASS
ANALYSIS_MODE = INTEGRATED_READ_ONLY_STATIC_FAILURE_LOCALIZATION
TRAINING_EVALUATION_INFERENCE_CHECKPOINT_LOADING = NOT_PERFORMED
READY_FOR_INDEPENDENT_INTEGRATED_ANALYSIS_VERIFICATION = YES
```

This is a descriptive, matched-row analysis of only the 12 specified
`clean_dev_predictions.json` files. Checkpoints were neither loaded nor
deserialized; no model ran. Historical split174, historical seed180 r2, and
the seed180 A3 retry suffix were excluded. Frozen interpretation authority:
commit `0894a921bf7ed69151722e3ce2691eb49bf4f40f`, blob
`e4cf29f818566cad465d49703dba2e22978b1473`, file
`reports/reason_router_p3w7_seed8192_a1_a2_a3_factorial_scientific_interpretation_report_candidate.md`.
Frozen interpretation provenance: `d644523b11a9431f841ea1216004c5e1527a740a` /
`6463237fd7a1358496d25ca62d1458295a7ee50e`; frozen aggregate evidence:
`6dcef9520af2cb88691628a77b72f3fdd7042cd8` /
`17be9bead782f129de925148580b034d981079cd`.

All derived tables use `stable_id` and 720 rows per seed. Row source is the
selected prediction export family; `training_report.json`,
`training_report_predictions.jsonl`, and `run_provenance.json` were inspected
for source/provenance context. NE denotes `NOT_ENTITLED`.

## Alignment and reconciliation gate

| Seed | rows A0/A1/A2/A3 | unique IDs A0/A1/A2/A3 | common IDs | gold | inspected input identity | result |
|---:|---|---|---:|---|---|---|
|180|720/720/720/720|720/720/720/720|720|identical|claim, evidence, intervention, normalized intervention, pair/source IDs, and exported structural targets|PASS|
|181|720/720/720/720|720/720/720/720|720|identical|same|PASS|
|182|720/720/720/720|720/720/720/720|720|identical|same|PASS|

The 720 stable IDs are also exactly identical across seeds180/181/182, so
cross-seed recurrence is authorized. There were no duplicates, missing rows,
gold mismatches, or input/population mismatches. Recomputed macro-F1 contrasts
match the frozen values (180/181/182): A1-A0 `-.021638/+.017572/-.006813`;
A2-A0 `-.194338/-.105900/-.106088`; A3-A2
`+.008071/-.051785/-.114152`; A3-A1 `-.164629/-.175257/-.213427`; interaction
`+.029709/-.069357/-.107339`.

## T1. 12-cell aggregate summary

Correct rows, accuracy, macro-F1, and final-class F1 recomputed from the
frozen exports.

|seed|arm|correct|accuracy|macro-F1|NE F1|R F1|S F1|
|---:|---|---:|---:|---:|---:|---:|---:|
|180|A0|655|.909722|.804826|.942932|1.000000|.471545|
|180|A1|647|.898611|.783188|.935796|.994413|.419355|
|180|A2|543|.754167|.610488|.842718|.685714|.303030|
|180|A3|580|.805556|.618559|.910394|.615385|.329897|
|181|A0|647|.898611|.792734|.935455|1.000000|.442748|
|181|A1|655|.909722|.810307|.942731|1.000000|.488189|
|181|A2|579|.804167|.686835|.884507|.802260|.373737|
|181|A3|604|.838889|.635050|.932751|.666667|.305732|
|182|A0|649|.901389|.813349|.936550|1.000000|.503497|
|182|A1|655|.909722|.806536|.944688|.982857|.492063|
|182|A2|586|.813889|.707261|.885338|.863388|.373057|
|182|A3|560|.777778|.593109|.883888|.623188|.272251|

## T2. Per-arm final-class confusion matrices

Raw matrices have rows gold `NE/R/S` and columns prediction `NE/R/S`.

|seed|A0|A1|A2|A3|
|---:|---|---|---|---|
|180|`[[537,0,3],[0,89,0],[62,0,29]]`|`[[532,1,7],[0,89,0],[65,0,26]]`|`[[434,62,44],[0,84,5],[56,10,25]]`|`[[508,0,32],[10,40,39],[58,1,32]]`|
|181|`[[529,0,11],[0,89,0],[62,0,29]]`|`[[535,0,5],[0,89,0],[60,0,31]]`|`[[471,13,56],[4,71,14],[50,4,37]]`|`[[534,0,6],[7,46,36],[64,3,24]]`|
|182|`[[524,0,16],[0,89,0],[55,0,36]]`|`[[538,0,2],[1,86,2],[60,0,31]]`|`[[471,9,60],[4,79,6],[49,6,36]]`|`[[491,0,49],[21,43,25],[59,6,26]]`|

A2 creates large NE leakage to R/S and loses R/S. A3 has only 41/49/49 R
predictions versus A1's 90/89/86, foreshadowing the R collapse below.

## Matched contrast convention

`CC/CW/WC/WW` mean correct-to-correct, correct-to-wrong, wrong-to-correct,
wrong-to-wrong. `WC-CW` is the net correct-row change for every comparison.

## T3. C1 A0 -> A2, explicit_local under explicit_product

|seed|gold|CC|CW broken|WC repaired|WW|
|---:|---|---:|---:|---:|---:|
|180|NE|433|104|1|2|
|180|R|84|5|0|0|
|180|S|17|12|8|54|
|181|NE|471|58|0|11|
|181|R|71|18|0|0|
|181|S|25|4|12|50|
|182|NE|466|58|5|11|
|182|R|79|10|0|0|
|182|S|28|8|8|47|

Totals repaired/broken/net are 9/121/-112, 12/80/-68, 13/76/-63. A2 repairs
34 rows but breaks 277. Previously-correct R broken: 5/18/10; destinations
are R->S 5; R->NE 4 + R->S 14; R->NE 4 + R->S 6. Previously-correct S broken:
12/4/8; destinations S->NE 3 + S->R 9; S->R 4; S->NE 2 + S->R 6. The largest
loss is NE (104/58/58, mainly NE->R/S), so harm is broad, not a small isolated
transition; nevertheless R/S errors are predominantly cross-polarity.

## T4. C2 A1 -> A3, explicit_local under conditional_first_blocker

|seed|gold|CC|CW broken|WC repaired|WW|
|---:|---|---:|---:|---:|---:|
|180|NE|508|24|0|8|
|180|R|40|49|0|0|
|180|S|25|1|7|58|
|181|NE|533|2|1|4|
|181|R|46|43|0|0|
|181|S|24|7|0|60|
|182|NE|491|47|0|2|
|182|R|41|45|2|1|
|182|S|21|10|5|55|

Totals are 7/74/-67, 1/52/-51, and 7/102/-95. A3 breaks 49/43/45
A1-correct R rows and repairs 0/0/2: net R -49/-43/-43. Destinations are
R->NE 10 + R->S 39; R->NE 7 + R->S 36; R->NE 20 + R->S 25. Thus the dominant
A3 R failure is R->S (100/137 broken R) rather than solely R->NE (37/137).
S effects are smaller and unfavorable overall (break/repair 1/7, 7/0, 10/5);
NE effects differ by seed but never offset the R loss.

## T5. C3 A0 -> A1, router-only mixedness

|seed|gold|CC|CW|WC|WW|class net|
|---:|---|---:|---:|---:|---:|---:|
|180|NE|532|5|0|3|-5|
|180|R|89|0|0|0|0|
|180|S|26|3|0|62|-3|
|181|NE|527|2|8|3|+6|
|181|R|89|0|0|0|0|
|181|S|28|1|3|59|+2|
|182|NE|524|0|14|2|+14|
|182|R|86|3|0|0|-3|
|182|S|31|5|0|55|-5|

Total corrected/broken/net: 0/8/-8, 11/3/+8, 14/8/+6. A1 is mixed because it
changes few rows and the sparse corrections/breakages trade off: seed180 only
breaks; seed181 repairs NE/S; seed182 repairs NE but loses R/S.

## T6. C4 A2 -> A3, conditional router under explicit_local

|seed|gold|CC|CW|WC|WW|class net|
|---:|---|---:|---:|---:|---:|---:|
|180|NE|423|11|85|21|+74|
|180|R|40|44|0|5|-44|
|180|S|20|5|12|54|+7|
|181|NE|471|0|63|6|+63|
|181|R|46|25|0|18|-25|
|181|S|23|14|1|53|-13|
|182|NE|451|20|40|29|+20|
|182|R|43|36|0|10|-36|
|182|S|25|11|1|54|-10|

Totals corrected/broken/net are 97/60/+37, 64/39/+25, 41/67/-26. Seed180's
small macro-F1 positive is driven by 85 NE repairs (60 R->NE, 25 S->NE) plus
12 S repairs, despite 44 R breaks. Seed181 repeats NE repair but loses too
much R/S quality for macro-F1. Seed182 has fewer NE repairs and wider losses.

## T7. Exact REFUTE transitions

|contrast|seed180 broken / repaired|seed181 broken / repaired|seed182 broken / repaired|
|---|---|---|---|
|C1|R->S 5 / none|R->NE 4, R->S 14 / none|R->NE 4, R->S 6 / none|
|C2|R->NE 10, R->S 39 / none|R->NE 7, R->S 36 / none|R->NE 20, R->S 25 / S->R 2|
|C3|none / none|none / none|R->NE 1, R->S 2 / none|
|C4|R->NE 10, R->S 34 / none|R->NE 3, R->S 22 / none|R->NE 17, R->S 19 / none|

Gold R count is 89 each seed. C1 baseline/treatment-correct R is 89/84,
89/71, 89/79; retained R 84/71/79; newly acquired R 0/0/0. C2 is 89/40,
89/46, 86/43; retained 40/46/41; newly acquired 0/0/2. A3's stable R loss is
primarily lost R/S discrimination, though its internal polarity path is not
directly observable.

## T8. Exact SUPPORT transitions

|contrast|seed180 broken / repaired|seed181 broken / repaired|seed182 broken / repaired|
|---|---|---|---|
|C1|S->NE 3, S->R 9 / NE->S 8|S->R 4 / NE->S 12|S->NE 2, S->R 6 / NE->S 8|
|C2|S->R 1 / NE->S 7|S->NE 4, S->R 3 / none|S->NE 4, S->R 6 / NE->S 5|
|C3|S->NE 3 / none|S->NE 1 / NE->S 3|S->NE 5 / none|
|C4|S->NE 5 / NE->S 3, R->S 9|S->NE 14 / R->S 1|S->NE 6, S->R 5 / R->S 1|

C1 S correctness changes 29->25, 29->37, 36->36; C2 changes 26->32,
31->24, 31->26. SUPPORT exhibits a smaller recurring R/S-discrimination
failure pattern, chiefly S->R in C1 and mixed S->NE/S->R in C2, but the
current exported evidence does not establish it as an independently localized
mechanism. Authorization shifts are not sufficiently consistent to assign
SUPPORT to a distinct authorization mechanism.

## T9. Cross-seed recurrence of newly broken stable IDs

Each value is unique IDs broken in `3-of-3/2-of-3/1-of-3` seeds. Parenthetic
weighted totals equal the three seed-level broken rows.

|contrast|NE|REFUTE|SUPPORT|
|---|---:|---:|---:|
|C1|20/44/72 (220)|4/5/11 (33)|3/2/11 (24)|
|C2|1/13/44 (73)|24/23/19 (137)|1/2/11 (18)|
|C3|0/1/5 (7)|0/0/3 (3)|0/0/9 (9)|
|C4|0/2/27 (31)|9/25/28 (105)|3/3/15 (30)|

Required recurrent R/S stable-ID lists, ordered by recurrence then ID:

```text
C1 R 3/3: generated_fact_045__polarity_flip, generated_fact_165__none,
generated_fact_249__none, generated_fact_285__none
C1 R 2/3: generated_fact_152__paraphrase, generated_fact_157__none,
generated_fact_242__paraphrase, generated_fact_261__none, generated_fact_272__paraphrase
C1 S 3/3: clinic_expansion__paraphrase, generated_fact_056__paraphrase,
generated_fact_062__paraphrase
C1 S 2/3: generated_fact_034__paraphrase, generated_fact_139__paraphrase
C2 R 3/3: garden_award__polarity_flip, generated_fact_045__polarity_flip,
generated_fact_051__polarity_flip, generated_fact_073__polarity_flip,
generated_fact_085__polarity_flip, generated_fact_087__polarity_flip,
generated_fact_133__polarity_flip, generated_fact_152__paraphrase,
generated_fact_157__none, generated_fact_165__none, generated_fact_193__none,
generated_fact_205__none, generated_fact_225__none, generated_fact_225__paraphrase,
generated_fact_243__none, generated_fact_248__paraphrase, generated_fact_249__none,
generated_fact_258__paraphrase, generated_fact_261__none, generated_fact_272__paraphrase,
generated_fact_278__paraphrase, generated_fact_285__none, generated_fact_285__paraphrase,
railway_restoration__polarity_flip
C2 S 3/3: generated_fact_062__paraphrase
C2 S 2/3: clinic_expansion__paraphrase, generated_fact_056__paraphrase
C2 R 2/3: clinic_expansion__polarity_flip, forest_mapping__polarity_flip,
generated_fact_089__polarity_flip, generated_fact_136__polarity_flip,
generated_fact_152__none, generated_fact_157__paraphrase, generated_fact_166__none,
generated_fact_166__paraphrase, generated_fact_174__paraphrase,
generated_fact_179__none, generated_fact_195__none, generated_fact_205__paraphrase,
generated_fact_227__none, generated_fact_241__none, generated_fact_242__paraphrase,
generated_fact_248__none, generated_fact_249__paraphrase, generated_fact_257__none,
generated_fact_257__paraphrase, generated_fact_259__none, generated_fact_286__paraphrase,
jazz_archive__polarity_flip, satellite_launch__polarity_flip
C4 R 3/3: generated_fact_051__polarity_flip, generated_fact_085__polarity_flip,
generated_fact_133__polarity_flip, generated_fact_193__none, generated_fact_205__none,
generated_fact_225__none, generated_fact_225__paraphrase, generated_fact_258__paraphrase,
generated_fact_285__paraphrase
C4 R 2/3: clinic_expansion__polarity_flip, garden_award__polarity_flip,
generated_fact_073__polarity_flip, generated_fact_087__polarity_flip,
generated_fact_136__polarity_flip, generated_fact_152__none, generated_fact_157__paraphrase,
generated_fact_166__none, generated_fact_166__paraphrase, generated_fact_174__paraphrase,
generated_fact_179__none, generated_fact_195__none, generated_fact_205__paraphrase,
generated_fact_241__none, generated_fact_243__none, generated_fact_248__none,
generated_fact_248__paraphrase, generated_fact_249__paraphrase, generated_fact_257__paraphrase,
generated_fact_259__none, generated_fact_278__paraphrase, generated_fact_286__paraphrase,
jazz_archive__polarity_flip, railway_restoration__polarity_flip, satellite_launch__polarity_flip
C4 S 3/3: generated_fact_048__none, jazz_archive__none, museum_purchase__none
C4 S 2/3: generated_fact_136__none, generated_fact_139__none, generated_fact_286__polarity_flip
```

C3 has no recurrent (2/3 or 3/3) R/S break. C4 has recurrent R and S counts
shown in T9, confirming that its attempted rescue also has persistent R loss.
Failures are partially recurrent: C2 R is strongest (24 IDs in 3/3), while
many NE/S failures are one- or two-seed-specific.

## T10. Structural-state co-occurrence

Exports provide target/state, not row-level predicted local-head correctness:
F=`frame_compatible_label==0`, P=`predicate_covered_label==0`,
S=`sufficiency_label==0`, PolR=`polarity_label==REFUTE`. These are descriptive
co-occurrences, not causal local-head errors. Reason order is FRAME >
PREDICATE > SUFFICIENCY > AUTHORIZED.

|contrast seed group|n|F|P|S|PolR|primary reason F/P/S/A|
|---|---:|---:|---:|---:|---:|---|
|C1 180 broken|121|40 (33.1%)|23 (19.0%)|60 (49.6%)|5 (4.1%)|40/4/60/17|
|C1 180 remain-correct|534|317 (59.4%)|276 (51.7%)|120 (22.5%)|84 (15.7%)|317/56/60/101|
|C1 180 repaired|9|1|1|0|0|1/0/0/8|
|C1 181 broken|80|32 (40.0%)|32 (40.0%)|12 (15.0%)|18 (22.5%)|32/14/12/22|
|C1 181 remain-correct|567|318 (56.1%)|263 (46.4%)|168 (29.6%)|71 (12.5%)|318/45/108/96|
|C1 181 repaired|12|0|0|0|0|0/0/0/12|
|C1 182 broken|76|41 (53.9%)|32 (42.1%)|8 (10.5%)|10 (13.2%)|41/9/8/18|
|C1 182 remain-correct|573|308 (53.8%)|262 (45.7%)|172 (30.0%)|79 (13.8%)|308/46/112/107|
|C1 182 repaired|13|3|2|0|0|3/2/0/8|
|C2 180 broken|74|18 (24.3%)|17 (23.0%)|0|49 (66.2%)|18/6/0/50|
|C2 180 remain-correct|573|334 (58.3%)|278 (48.5%)|180 (31.4%)|40 (7.0%)|334/54/120/65|
|C2 180 repaired|7|0|0|0|0|0/0/0/7|
|C2 181 broken|52|2 (3.8%)|1 (1.9%)|0|43 (82.7%)|2/0/0/50|
|C2 181 remain-correct|603|353 (58.5%)|296 (49.1%)|180 (29.9%)|46 (7.6%)|353/60/120/70|
|C2 181 repaired|1|1|1|0|0|1/0/0/0|
|C2 182 broken|102|23 (22.5%)|35 (34.3%)|0|45 (44.1%)|23/24/0/55|
|C2 182 remain-correct|553|336 (60.8%)|264 (47.7%)|180 (32.5%)|41 (7.4%)|336/35/120/62|
|C2 182 repaired|7|0|0|0|2|0/0/0/7|

C1 has no stable single structural-state enrichment. C2 is clearer: broken
rows are PolR-heavy versus retained-correct (66.2/82.7/44.1% versus
7.0/7.6/7.4%), mostly AUTHORIZED targets, and have zero sufficiency-failure
targets. This weakens an explanation based only on gold earlier-reason state.
Exported q masses and A1/A3 reason probabilities are analyzed below; the
target table itself is still not a predicted-head correctness table.

## Deterministic semantic exemplars

Text is available; these are selected recurrence-descending then stable ID,
with brief paraphrases rather than long data quotes.

|category|stable ID|gold before->after|exported state / semantic pattern|
|---|---|---|---|
|C1 recurrent broken R|`generated_fact_045__polarity_flip` (3)|R R->S|AUTHORIZED, F/P/S pass, polarity R; polarity flip|
|C1 recurrent broken R|`generated_fact_165__none` (3)|R R->S|AUTHORIZED, F/P/S pass; ordinary fact|
|C1 recurrent broken S|`clinic_expansion__paraphrase` (3)|S S->R|AUTHORIZED, F/P/S pass; paraphrase|
|C1 recurrent broken S|`generated_fact_056__paraphrase` (3)|S S->R|AUTHORIZED, F/P/S pass; paraphrase|
|C2 recurrent broken R|`garden_award__polarity_flip` (3)|R R->S|AUTHORIZED, F/P/S pass; polarity flip|
|C2 recurrent broken R|`generated_fact_045__polarity_flip` (3)|R R->S|AUTHORIZED, F/P/S pass; polarity flip|
|C2 recurrent broken S|`generated_fact_062__paraphrase` (3)|S S->R|AUTHORIZED, F/P/S pass; paraphrase|
|C2 recurrent broken S|`clinic_expansion__paraphrase` (2)|S S->R|AUTHORIZED, F/P/S pass; paraphrase|
|A0->A1 corrected|`generated_fact_118__location_swap` (2)|NE S->NE|FRAME target; location swap|
|A0->A1 broken|`generated_fact_090__entity_swap` (2)|NE NE->S|FRAME target; entity swap|
|seed180 A2->A3 corrected|`clinic_expansion__evidence_truncation`|NE R->NE|SUFFICIENCY target; evidence truncation|
|seed180 A2->A3 corrected|`clinic_expansion__paraphrase`|S R->S|AUTHORIZED; paraphrase|

These examples reinforce a descriptive R/S exchange pattern on many
AUTHORIZED cases. They do not prove a hidden local-head or semantic cause.

## T11. Mechanistic-hypothesis ranking

|hypothesis|evidence and boundary|assessment|
|---|---|---|
|H1 explicit_local disrupts final-class discrimination|Both C1 and C2 harm every seed; broken REFUTE and SUPPORT groups have much larger adverse final R-S margin shifts. No calibration analysis was performed.|STRONGLY_SUPPORTED_DESCRIPTIVE_PATTERN|
|H2 explicit_local particularly damages R through polarity|Actual C2 polarity_probs_2 show substantially larger adverse REFUTE-SUPPORT movement on R->S than retained REFUTE in all seeds.|STRONGLY_SUPPORTED_DESCRIPTIVE_PATTERN|
|H3 damage is mainly earlier F/P/S routing|R->NE has qS/qA movement, but dominant C2 R->S has no consistent upstream q separation and C1 lacks reason vectors.|WEAK_OR_INCONCLUSIVE|
|H4 SUPPORT shows a smaller recurring discrimination pattern accompanying the REFUTE/SUPPORT boundary failure|SUPPORT losses recur and show opposite final discrimination movement; q_authorized is not consistently diagnostic. This is a descriptive recurring pattern only, not an independently localized mechanism.|MODERATELY_SUPPORTED_DESCRIPTIVE_PATTERN|
|H5 router-only is near-neutral by cancellation|A0->A1 has real internal q/final movement but sparse, seed-specific corrections and breakages that cancel.|STRONGLY_SUPPORTED_DESCRIPTIVE_PATTERN|
|H6 stable useful router subset hidden by average|There are NE/S repairs in seeds181/182, but no recurrent 2/3 or 3/3 REFUTE/SUPPORT correction.|WEAK_OR_INCONCLUSIVE|
|H7 A3 failure is mainly explicit_local|C1/C2 share broad final degradation; C2 adds direct polarity evidence, but one common edge is not localized.|MODERATELY_SUPPORTED_DESCRIPTIVE_PATTERN|
|H8 router x ownership interaction is stable|Frozen interaction signs are +/−/−; C4 composition differs materially.|CONTRADICTED_BY_CURRENT_EVIDENCE|

## WHAT THE FACTORIAL IS TELLING US

1. A0 remains the reference; A1 makes small, mixed, seed-dependent changes.
2. Current explicit_local ownership does not work in either router setting.
3. Ownership topology is most implicated descriptively, not causally proven.
4. REFUTE is most affected, especially C2 R->S; SUPPORT has a smaller
recurring R/S-discrimination failure pattern, not an independently localized
mechanism.
5. Failures are partially recurrent, especially C2 R (24 IDs in 3/3), not
either one immutable hard set or wholly seed-specific noise.
6. Conditional-first-blocker alone creates sparse corrections and breakages
that trade off by seed; it has no stable useful gain.
7. A3 cannot rescue A2 reproducibly because NE repairs are coupled to persistent
R loss with seed-varying macro-F1 consequences.
8. The strongest explanation is impaired external final-class discrimination
under explicit_local, concentrated in R/S separation with the conditional
router.
9. A stable beneficial interaction is contradicted; a pure earlier-target-state
F/P/S account is weakened but not causally ruled out.
10. A1/A3 provide direct internal polarity, reason, q, and final-decision
localization; A0/A2 lack only their polarity/reason vectors.

## T12. Next-design implications

|observation|bounded recommendation|not authorized|
|---|---|---|
|A0 is the stable reference|Preserve A0 as control.|Promotion or replacement|
|explicit_local harms C1 and C2|Deprioritize current topology; if revisited, redesign ownership to preserve downstream coordination.|Implementation or a loss weight|
|C2 R->S is recurrent|Use the already exported polarity, q/reason, and final decision states; no replay is needed for them.|New execution|
|A1 is mixed|Retain conditional-first-blocker only as an experimental concept.|Promotion or factorial run|
|interaction is unstable|Do not interpret seed180 rescue as a mechanism.|Training/scheduling|

The recommended next scientific direction is a bounded continuous-ownership
design question; T20 gives the exact non-implementation recommendation.

## Availability and final checks
## Availability and final checks

Used fields include stable ID, gold/predicted final label, text/input identity,
intervention identity, structural target/state, q masses, final
logits/probabilities, and every populated reason/polarity vector. A1/A3
reason_logits_4, reason_probs_4, polarity_logits_2, and polarity_probs_2 are
populated for all 2,160 rows each; A0/A2 have those vectors null. The
predicted_primary_reason field is partial, not complete. Row-level
predicted-head correctness and pairwise decision fields not represented by
these exports remain NOT_TESTABLE_FROM_CURRENT_EXPORTS. Co-occurrence is not
causation.
recurrence weighted totals reconcile. No checkpoint/model was loaded. This is
the only new file created by this task; pre-existing untracked imported-run
directories were preserved.

## T13. Field availability and coverage

Each arm/seed has 720 rows and each arm totals 2,160 rows. final_logits,
final_probs, q_frame, q_predicate, q_sufficiency, and q_authorized are
populated 2160/2160 in every A0/A1/A2/A3 arm. reason_logits_4,
reason_probs_4, polarity_logits_2, and polarity_probs_2 are populated
2160/2160 in A1/A3 and 0/2160 in A0/A2. predicted_primary_reason is partial:
A1 1791/2160 and A3 1752/2160, with A0/A2 0/2160.

Vector order has repository evidence: final is REFUTE, NOT_ENTITLED, SUPPORT
(P2_EXTERNAL_CLASS_ORDER, train_controlled_v6b_minimal.py:195-199);
polarity is REFUTE, SUPPORT (target construction 3902-3905 and negative/positive
logit stack 651-654); reason and q masses are FRAME, PREDICATE, SUFFICIENCY,
AUTHORIZED (195, 10461-10469, 10564-10570). Therefore no index is guessed.
C1 polarity/reason-vector analysis genuinely remains unavailable, but its q
and final analysis does not. No replay is needed for fields already present.

## T14. C2 REFUTE retained versus broken internal signals

Population is gold REFUTE, A1 prediction REFUTE. Counts retained/R->S/R->NE
are 40/39/10, 46/36/7, 41/25/20 for seeds180/181/182. Entries are
baseline mean -> A3 mean (matched delta mean; delta median [range]).

|seed group|polarity R-S|final probability R-S|final logit R-S|
|---|---|---|---|
|180 retained|.897->.310 (-.588;-.553[-.833,-.356])|.776->.205 (-.571;-.551[-.720,-.455])|2.924->.654 (-2.270;-2.242[-2.830,-1.648])|
|180 R->S|.849->-.423 (-1.272;-1.282[-1.701,-.469])|.698->-.280 (-.977;-1.004[-1.223,-.330])|2.735->-.990 (-3.725;-3.731[-5.101,-.984])|
|180 R->NE|.903->.452 (-.451;-.438[-.551,-.378])|.800->.192 (-.608;-.604[-.707,-.540])|2.979->.978 (-2.002;-1.995[-2.235,-1.796])|
|181 retained|.904->.508 (-.396;-.334[-.805,-.160])|.809->.391 (-.419;-.373[-.737,-.245])|2.992->1.181 (-1.812;-1.734[-2.887,-.992])|
|181 R->S|.862->-.578 (-1.440;-1.510[-1.827,-.712])|.718->-.381 (-1.099;-1.134[-1.338,-.523])|2.807->-1.570 (-4.377;-4.466[-6.205,-1.670])|
|181 R->NE|.897->.689 (-.207;-.211[-.222,-.185])|.806->.192 (-.614;-.683[-.740,-.429])|2.916->1.694 (-1.222;-1.258[-1.296,-1.018])|
|182 retained|.920->.414 (-.506;-.428[-.937,+.316])|.817->.319 (-.498;-.487[-.823,+.118])|3.394->.940 (-2.454;-2.478[-3.717,+.855])|
|182 R->S|.911->-.463 (-1.374;-1.369[-1.775,-.480])|.795->-.379 (-1.174;-1.163[-1.501,-.398])|3.330->-1.132 (-4.462;-4.317[-6.102,-.994])|
|182 R->NE|.920->.409 (-.510;-.475[-.913,-.261])|.814->.133 (-.682;-.698[-.833,-.510])|3.210->.904 (-2.306;-2.213[-3.265,-1.351])|

R->S is already polarity-associated: its polarity delta is more adverse than
retained in all seeds (-1.272 vs -.588, -1.440 vs -.396, -1.374 vs -.506).
The final decision amplifies rather than creates this movement.

For completeness, the corresponding group means for individual probabilities
are listed as baseline->A3. Columns are polarity P(REFUTE), P(SUPPORT), final
P(REFUTE), and P(SUPPORT), respectively.

|seed group|polarity R|polarity S|final R|final S|
|---|---|---|---|---|
|180 retained|.949->.655|.051->.345|.820->.439|.044->.235|
|180 R->S|.924->.288|.076->.712|.758->.198|.060->.478|
|180 R->NE|.952->.726|.048->.274|.843->.312|.043->.120|
|181 retained|.952->.754|.048->.246|.852->.579|.043->.188|
|181 R->S|.931->.211|.069->.789|.775->.150|.058->.531|
|181 R->NE|.948->.845|.052->.155|.852->.236|.046->.043|
|182 retained|.960->.707|.040->.293|.853->.551|.036->.232|
|182 R->S|.956->.269|.044->.731|.834->.216|.038->.595|
|182 R->NE|.960->.705|.040->.295|.850->.240|.036->.107|

|seed group|qF|qP|qS|qA|
|---|---|---|---|---|
|180 retained|.036->.110 (+.074)|.080->.122 (+.041)|.019->.093 (+.074)|.865->.675 (-.190)|
|180 R->S|.053->.188 (+.134)|.113->.123 (+.011)|.015->.013 (-.002)|.819->.676 (-.143)|
|180 R->NE|.019->.064 (+.045)|.076->.115 (+.040)|.019->.386 (+.367)|.886->.435 (-.451)|
|181 retained|.026->.089 (+.063)|.059->.108 (+.049)|.019->.036 (+.017)|.895->.767 (-.128)|
|181 R->S|.055->.175 (+.119)|.104->.140 (+.036)|.008->.005 (-.003)|.833->.681 (-.152)|
|181 R->NE|.011->.037 (+.025)|.030->.068 (+.038)|.060->.615 (+.555)|.899->.281 (-.619)|
|182 retained|.059->.016 (-.043)|.030->.082 (+.052)|.021->.118 (+.096)|.889->.784 (-.106)|
|182 R->S|.075->.028 (-.047)|.035->.110 (+.075)|.018->.051 (+.033)|.871->.811 (-.060)|
|182 R->NE|.019->.008 (-.011)|.019->.059 (+.040)|.076->.586 (+.510)|.886->.347 (-.539)|

The q_frame/q_predicate/q_sufficiency/q_authorized values and reason_probs_4
are related router/reason outputs but numerically distinct exported quantities.
They must be analyzed separately: no equality or substitutability is assumed,
and their distinct values must be preserved in tables and interpretation.

The following separately reports reason_probs_4 group means (baseline->A3) in
the same F/P/S/A order; it is not a relabeling of the q table above.

|seed group|reason F|reason P|reason S|reason A|
|---|---|---|---|---|
|180 retained|.036->.110|.080->.121|.019->.095|.865->.674|
|180 R->S|.053->.188|.113->.123|.015->.014|.819->.676|
|180 R->NE|.019->.064|.076->.115|.019->.389|.886->.433|
|181 retained|.026->.089|.059->.108|.019->.036|.895->.767|
|181 R->S|.055->.175|.104->.139|.008->.005|.833->.681|
|181 R->NE|.011->.036|.030->.067|.060->.618|.899->.279|
|182 retained|.059->.016|.030->.082|.022->.118|.889->.784|
|182 R->S|.075->.028|.035->.110|.018->.051|.872->.811|
|182 R->NE|.019->.008|.019->.058|.076->.587|.886->.347|

qA changes on retained rows too; its strong failure-specific separation is
R->NE, with qS +.367/+.555/+.510 and qA -.451/-.619/-.539. It is not a
consistent R->S separator. A3 predicted_primary_reason is SUFFICIENCY for all
R->NE rows (10/7/20) and missing for retained/R->S because they do not predict
NE; A1 is missing for these baseline-REFUTE rows. This preserves partial
coverage rather than treating it as a null signal.

## T15. C2 REFUTE destination comparison

R->S has negative post-A3 polarity and final margins in every seed. R->NE has
the qS/qA pattern above while its mean final R-S margin remains positive
(.192/.192/.133). The destinations are diagnostically different. No
significance test or causal claim is made.

## T16. Recurrent C2 REFUTE signature

For the 24 recurrent broken IDs versus 23 REFUTE rows retained in all three
seeds, polarity/final R-S delta is -0.974/-0.546, -1.043/-0.410,
-1.101/-0.442 (180/181/182); qA is -.233/-.186, -.263/-.115,
-.197/-.124. Recurrent qS is +.094/+.064, +.129/+.004, +.173/+.121.
Thus recurrent failures have a stable stronger polarity/final signature, but
not a uniform earlier-reason signature. It is descriptive, not causal.

## T17. C1 versus C2 signature comparison

C1 R->S final R-S deltas are -.921/-1.006/-.990 versus retained
-.648/-.366/-.540. C1 R->NE qS/qA is +.562/-.685 (181) and +.735/-.753
(182). This shares broad final degradation and R->NE authorization/sufficiency
movement with C2, but cannot establish C2 polarity in C1 because A0/A2 lack
polarity vectors. Highest supported localization is
LEVEL_2_AUTHORIZATION_ASSOCIATED for C1 and
LEVEL_1_POLARITY_ASSOCIATED for dominant C2 R->S; neither is causal.

## T18. SUPPORT internal-signal analysis

Broken baseline-correct SUPPORT destinations are C1 9R+3NE, 4R, 6R+2NE and
C2 1R, 3R+4NE, 6R+4NE. C1 broken final R-S deltas are
+.635/+1.079/+.924 versus retained +.420/+.069/+.307; C2 are
+.286/+.470/+.873 versus +.167/+.161/+.262. This is a smaller mirror-like
R/S discrimination failure. qA is inconsistent across retained/broken groups,
so SUPPORT exhibits a smaller recurring R/S-discrimination failure pattern
with a polarity-confusion component, not an independently localized mechanism
or a demonstrated downstream authorization effect.

## T19. Revised H1-H8

T11 is the revised H1-H8 table: it uses the required verdict vocabulary and
supersedes the prior unavailable-field assessments.

## T20. D1-D5 design decision

|direction|decision|
|---|---|
|D1 continuous/partial ownership|RECOMMENDED: broad C1/C2 degradation, no single q edge consistently separates dominant R->S|
|D2 edge-specific ownership|not yet justified: R->NE q pattern does not localize 100 R->S losses|
|D3 more static measurement|not recommended for signals already exported|
|D4 deprioritize ownership entirely|not supported by rejection of one binary intervention|
|D5 multi-stream semantic-state Mamba|out of scope: no G_I/backbone-state evidence|

The exactly one recommended next DESIGN direction is D1. Minimal concept only:
A0/reference, current explicit_local, and exactly one pre-specified
intermediate lambda in z_down = stopgrad(z) + lambda * (z - stopgrad(z)).
No numeric lambda is selected. Falsification: pre-register that the
intermediate must reduce three-seed baseline-correct REFUTE rows newly broken
versus current explicit_local without compensating deterioration versus A0 in
final discrimination and macro-F1; otherwise the hard 0-versus-1 boundary
explanation is falsified for this setting.

## Long-term research synthesis

The vision/hypothesis-map files are non-authority context. Structured Decision
remains inspectable here. The Gradient Ownership distinction is G_I forward
information access versus G_G backward modification authority. This factorial
tests an early binary downstream G_G intervention, not multi-stream owned
Mamba states, backbone-level G_I ownership, or semantic-state ownership.
Broad degradation without one stable edge fits Over-Isolation more than it
refutes ownership research: Generation 2 continuous ownership precedes
Generation 3 edge-specific ownership. Do not jump to multi-stream architecture.
Measure first. Explain second. Modify third.

```text
READY_FOR_INDEPENDENT_INTEGRATED_ANALYSIS_VERIFICATION = YES
```
