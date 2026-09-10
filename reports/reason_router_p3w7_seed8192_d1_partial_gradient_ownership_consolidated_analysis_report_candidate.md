# D1 partial-gradient ownership: consolidated validated-evidence analysis

```text
VERDICT = PASS
PHASE = D1_CONSOLIDATED_READ_ONLY_VALIDATED_EVIDENCE_ANALYSIS
HEADLINE = PARTIAL_OR_HETEROGENEOUS_SUPPORT_FOR_OVER_ISOLATION
TRAINING_EVALUATION_INFERENCE_CHECKPOINT_LOADING = NOT_PERFORMED
```

## 1. Overall verdict

The pre-specified midpoint, D1 (`partial`, lambda=0.5), is consistently much
better than hard `explicit_local` A2 (lambda=0) on macro-F1, accuracy, and
REFUTE preservation in all three seeds, without a class-collapse pathology.
It does **not** reach joint A0 (lambda=1) on macro-F1 or accuracy in any seed.
This is descriptive, three-seed support that the hard binary isolation was too
strong in this setting, but it is heterogeneous: D1 removes all A0-correct
REFUTE breakage in seeds 180/181 and reduces it 10->2 in seed 182, while its
NE/SUPPORT behavior remains materially short of A0. It is not a lambda-tuning
result, response curve, or causal localization of an individual G_G edge.

## 2. Evidence matrix and provenance/configuration reconciliation

The nine required `clean_dev_predictions.json`, `training_report.json`, and
`run_provenance.json` artifacts were read. No `.pt` file was opened. All nine
have seed 180/181/182, split seed 8192, `explicit_product`, and
`reason_loss_weight=0`. D1 has arm D1, `partial`, lambda=0.5, and commit
`1abe147484e2155b19d3c973175628d337807501` in every seed. A2 has arm A2 and
`explicit_local`; A0 has arm A0 and `joint`. Their imported endpoint commits
are respectively `3a76c6cd3f6bd8b011317f37938677822ce9191d` and
`55debe94f0d19d16a334395e8561901fed6b52fa`. This is the intended endpoint
semantics in the D1 authority. No provenance/configuration mismatch occurred.

## 3. Matched-row alignment gate

|seed|A2/D1/A0 rows|unique stable_id|same population|gold and relevant inputs|result|
|---:|---:|---:|---|---|---|
|180|720/720/720|720/720/720|identical|identical|PASS|
|181|720/720/720|720/720/720|identical|identical|PASS|
|182|720/720/720|720/720/720|identical|identical|PASS|

Relevant identity fields checked per stable ID were gold label, claim, evidence,
intervention and normalized intervention, intervention type, pair/source IDs,
frame/predicate/sufficiency targets, primary reason, and polarity label. The
720-ID population is also identical across all three seeds, so the frozen
integrated report's cross-seed recurrence convention is permissible.

## 4. Recomputed nine-cell metrics

All values below were recomputed from prediction exports; accuracy and macro-F1
match `training_report.json` to exported floating precision. Confusions use
gold rows/predicted columns in NE,R,S order.

|seed|arm|correct|accuracy|macro-F1|NE F1|R F1|S F1|prediction distribution NE/R/S|confusion|
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
|180|A2|543|.754167|.610488|.842718|.685714|.303030|490/156/74|[[434,62,44],[0,84,5],[56,10,25]]|
|180|D1|624|.866667|.778032|.912409|1.000000|.421687|556/89/75|[[500,0,40],[0,89,0],[56,0,35]]|
|180|A0|655|.909722|.804826|.942932|1.000000|.471545|599/89/32|[[537,0,3],[0,89,0],[62,0,29]]|
|181|A2|579|.804167|.686835|.884507|.802260|.373737|525/88/107|[[471,13,56],[4,71,14],[50,4,37]]|
|181|D1|631|.876389|.784193|.919457|1.000000|.433121|565/89/66|[[508,0,32],[0,89,0],[57,0,34]]|
|181|A0|647|.898611|.792734|.935455|1.000000|.442748|591/89/40|[[529,0,11],[0,89,0],[62,0,29]]|
|182|A2|586|.813889|.707261|.885338|.863388|.373057|524/94/102|[[471,9,60],[4,79,6],[49,6,36]]|
|182|D1|611|.848611|.774654|.900093|.988636|.435233|531/87/102|[[482,0,58],[0,87,2],[49,0,42]]|
|182|A0|649|.901389|.813349|.936550|1.000000|.503497|579/89/52|[[524,0,16],[0,89,0],[55,0,36]]|

## 5. Endpoint contrasts and ordered descriptive geometry

P1 is D1-A2; P2 is D1-A0. `between` is descriptive only.

|seed|metric|P1|P2|lambda=.5 location|
|---:|---|---:|---:|---|
|180|accuracy|+.112500|-.043056|between|
|180|macro-F1|+.167544|-.026794|between|
|180|NE/R/S F1|+.069690/+.314286/+.118656|-.030524/.000000/-.049858|between/tied A0/between|
|181|accuracy|+.072222|-.022222|between|
|181|macro-F1|+.097358|-.008542|between|
|181|NE/R/S F1|+.034950/+.197740/+.059384|-.015998/.000000/-.009627|between/tied A0/between|
|182|accuracy|+.034722|-.052778|between|
|182|macro-F1|+.067393|-.038695|between|
|182|NE/R/S F1|+.014755/+.125248/+.062176|-.036457/-.011364/-.068263|between|

The same 0->.5->1 order holds for A0-correct REFUTE breaks (5->0->0,
18->0->0, 10->2->0), R->S breaks (5->0->0, 14->0->0, 6->2->0), and R->NE
breaks (0->0->0, 4->0->0, 4->0->0). Thus all seeds improve from A2 to D1;
the degree and the remaining gap to A0 vary. This does not establish
continuous monotonicity.

## 6. Matched correctness transitions

CC/CW/WC/WW are correct->correct, correct->wrong, wrong->correct, wrong->wrong.
Broken/repaired/net below are CW/WC/WC-CW by gold NE,R,S.

|seed|contrast|CC/CW/WC/WW|NE broken/repaired/net|R|S|
|---:|---|---|---|---|---|
|180|A2->D1|514/29/110/67|24/90/+66|0/5/+5|5/15/+10|
|180|D1->A0|618/6/37/59|0/37/+37|0/0/0|6/0/-6|
|181|A2->D1|566/13/65/76|6/43/+37|0/18/+18|7/4/-3|
|181|D1->A0|624/7/23/66|2/23/+21|0/0/0|5/0/-5|
|182|A2->D1|573/13/38/96|10/21/+11|1/9/+8|2/8/+6|
|182|D1->A0|598/13/51/58|5/47/+42|0/2/+2|8/2/-6|

## 7. Frozen A0-correct recovery analysis

|seed|gold|A0-correct|broken A2|still broken D1|A2-broken recovered D1|new D1 break from A2/A0-correct|recovery fraction|
|---:|---|---:|---:|---:|---:|---:|---:|
|180|NE/R/S|537/89/29|104/5/12|37/0/0|90/5/12|23/0/0|.865/.1000/.1000|
|181|NE/R/S|529/89/29|58/18/4|23/0/0|41/18/4|6/0/0|.707/.1000/.1000|
|182|NE/R/S|524/89/36|58/10/8|47/2/2|20/9/6|9/1/0|.345/.900/.750|

For the central A0-correct REFUTE population, A2 destinations R->NE/R->S are
0/5, 4/14, and 4/6; D1 destinations are 0/0, 0/0, and 0/2. D1's exact
A2->D1 changes are therefore R->S -5/-14/-4 and R->NE 0/-4/-4.

## 8. Frozen recurrent C1 stable IDs

The historical population is unchanged: an occurrence is A0-correct then
A2-wrong, and only prior recurrent (2/3 or 3/3) REFUTE/SUPPORT IDs are listed.
All 3/3 REFUTE IDs recover in 3/3 occurrences:
`generated_fact_045__polarity_flip`, `generated_fact_165__none`,
`generated_fact_249__none`, `generated_fact_285__none` (each A2 R->S,
D1 R). The remaining recurrent REFUTE IDs are `generated_fact_152__paraphrase`
(2/2 recovered, A2 R->NE), `generated_fact_242__paraphrase` (2/2, R->NE),
`generated_fact_261__none` (2/2, R->S), `generated_fact_272__paraphrase`
(2/2, R->NE), and `generated_fact_157__none` (1/2; one remains R->S).

All recurrent SUPPORT occurrences recover: `clinic_expansion__paraphrase`,
`generated_fact_056__paraphrase`, and `generated_fact_062__paraphrase` are
3/3; `generated_fact_034__paraphrase` and `generated_fact_139__paraphrase`
are 2/2. D1 changes the A2 dominant R/S failure destination to the gold class
for every recovered occurrence; only the one `generated_fact_157__none`
occurrence remains R->S.

## 9. Class-collapse safeguard

No new collapse is present. D1 predicts REFUTE 89/89/87 times (A0: 89/89/89;
A2: 156/88/94), with REFUTE F1 1/1/.988636. SUPPORT predictions are nonzero
(75/66/102) and SUPPORT true positives are 35/34/42. Macro-F1 gains over A2
therefore are not a compensation for a new REFUTE or SUPPORT collapse.

## 10. Bounded scientific interpretation and non-claims

This is **PARTIAL_OR_HETEROGENEOUS_SUPPORT_FOR_OVER_ISOLATION**. The strongest
required axes move away from A2 in every seed: macro-F1 (+.167544, +.097358,
+.067393), accuracy (+.112500, +.072222, +.034722), REFUTE F1
(.685714->1, .802260->1, .863388->.988636), and A0-correct REFUTE breakage
(5->0, 18->0, 10->2), without collapse. The remaining macro/accuracy deficit
to A0 and seed-182 residual REFUTE/SUPPORT errors preclude a stronger claim.

It does not establish an optimal lambda, a response curve, a specific G_G-edge
cause, G_I/state ownership, native Mamba semantic-state ownership, D2/D5,
statistical significance, or production readiness.

## 11. Exactly one next research direction

Close D1: it answers the binary-over-isolation question descriptively. The one
bounded next direction is **independent static/mechanistic localization of the
remaining seed-182 D1 REFUTE/SUPPORT errors using already exported evidence**;
it is not another lambda run or a new execution decision.

## 12. Output identity and repository state

This report is the only requested tracked delta. Its byte count and SHA256 are
recorded after final write in the delivery summary. No source, tests, existing
reports, raw artifacts, checkpoints, data, training, evaluation, inference, or
checkpoint loading occurred in this analysis.
