# D1 seed182 residual static/mechanistic localization

```text
VERDICT = PASS
PHASE = D1_SEED182_RESIDUAL_STATIC_MECHANISTIC_LOCALIZATION
AUTHORITY = user instruction; frozen consolidated-evidence commit 76eb5b0926aecc8e4afc45a0df09dc83edd7e3d4
TRAINING_EVALUATION_INFERENCE_CHECKPOINT_LOADING = NOT_PERFORMED
```

## 1. Scope, sources, and alignment gate

This is a read-only analysis of the requested `clean_dev_predictions.json`
exports. `training_report.json` and `run_provenance.json` were used only to
reconcile configuration and exported-row counts. The prediction JSONL files
were not needed. No `.pt` path was opened, loaded, or deserialized.

The seed182 A2/D1/A0 exports each contain 720 rows and 720 unique stable IDs.
Their stable-ID sets are identical. For every ID, the following inspected
identity fields agree exactly across the three arms: `gold_label`, claim,
evidence, intervention and normalized intervention, intervention type,
pair/source IDs, frame/predicate/sufficiency targets, primary reason, and
polarity label. Thus the requested population is aligned rather than a
cross-export join artifact.

`run_provenance.json`/`training_report.json` reconcile the following: seed182
and split seed8192 for A2/D1/A0; 720 exported prediction rows; A2 arm
`explicit_product`/`explicit_local`; D1 arm `explicit_product`/`partial`,
lambda=.5; A0 arm `explicit_product`/`joint`; and `reason_loss_weight=0`.
The matched D1 controls seed180/181 also reconcile as split8192, D1,
`partial`, lambda=.5, 720 rows. No mismatch occurred.

Vector order used below is export order: final `(REFUTE, NOT_ENTITLED,
SUPPORT)` and q `(FRAME, PREDICATE, SUFFICIENCY, AUTHORIZED)`. `F/P/S/A` in
the gate column means `frame_logit/frame_prob/predicate_coverage_prob/
sufficiency_prob/entitlement_prob`; `PolM` is the native scalar
`polarity_margin`. A0/A2/D1 expose `reason_logits_4`, `reason_probs_4`,
`polarity_logits_2`, and `polarity_probs_2` as null for these rows; likewise
`predicted_primary_reason` is null. They are consequently not substituted or
invented below.

## 2. Exact pre-specified primary residual population

Definition fixed before inspection: seed182 rows with gold REFUTE or SUPPORT,
correct A0 prediction, and wrong D1 prediction. Recomputing this predicate
from the raw exports yields exactly four rows.

|stable ID|gold|A2|D1|A0|D1 seed180|D1 seed181|
|---|---|---|---|---|---|
|`generated_fact_157__none`|REFUTE|SUPPORT|SUPPORT|REFUTE|REFUTE|REFUTE|
|`generated_fact_166__paraphrase`|REFUTE|REFUTE|SUPPORT|REFUTE|REFUTE|REFUTE|
|`generated_fact_152__polarity_flip`|SUPPORT|NOT_ENTITLED|NOT_ENTITLED|SUPPORT|SUPPORT|NOT_ENTITLED|
|`generated_fact_285__polarity_flip`|SUPPORT|NOT_ENTITLED|NOT_ENTITLED|SUPPORT|NOT_ENTITLED|NOT_ENTITLED|

Therefore the frozen report's REFUTE statement is verified: the two exact
seed182 A0-correct REFUTE residual IDs are
`generated_fact_157__none` and `generated_fact_166__paraphrase`, and both are
REFUTE->SUPPORT under D1. The two separate SUPPORT residual IDs are
`generated_fact_152__polarity_flip` and `generated_fact_285__polarity_flip`,
both SUPPORT->NOT_ENTITLED.

All four have `primary_reason=AUTHORIZED`, empty secondary reasons,
F/P/S targets `1/1/1`, and matching polarity target (REFUTE for the first two,
SUPPORT for the latter two). Their pair/source IDs are respectively
`generated_fact_157`, `generated_fact_166`, `generated_fact_152`, and
`generated_fact_285`; intervention types are none, paraphrase, polarity_flip,
and polarity_flip. These are target/identity facts, not predicted local-head
correctness.

## 3. Row-level A2 -> D1 -> A0 diagnostics

Each cell gives `gate F/P/S/A/PolM; q; final logits; final probabilities`.
The latter two vectors are in final-class order stated above. This includes
the populated row-level gate, entitlement, q, polarity-margin, and final
decision observables that can discriminate the proposed levels.

|ID / arm|native populated diagnostics|
|---|---|
|152 / A2|`-1.1102/.2478/.3009/.9762/.0728/3.7253; .7522/.1733/.0018/.0728; .0154/.9580/.2866; .2050/.5262/.2689`|
|152 / D1|`-.4007/.4011/.4089/.9900/.1624/3.7400; .5989/.2371/.0016/.1624; .0333/.8635/.6406; .1950/.4472/.3579`|
|152 / A0|`-.3288/.4185/.5033/.9919/.2090/4.2149; .5815/.2079/.0017/.2090; .0196/.8211/.9003; .1773/.3951/.4276`|
|157 / A2|`.9789/.7269/.9596/.8186/.5710/.2459; .2731/.0294/.1265/.5710; .9657/.4537/1.1061; .3636/.2179/.4184`|
|157 / D1|`1.9380/.8741/.8901/.9910/.7711/.1222; .1259/.0960/.0070/.7711; 1.6311/.2482/1.7253; .4256/.1068/.4677`|
|157 / A0|`2.4888/.9234/.9385/.9926/.8601/-1.5252; .0766/.0568/.0065/.8601; 2.6549/.1616/1.3430; .7397/.0611/.1992`|
|166 / A2|`1.8795/.8676/.9688/.8291/.6969/-1.2365; .1324/.0271/.1436/.6969; 1.7344/.3262/.8727; .5999/.1467/.2534`|
|166 / D1|`2.5604/.9283/.9051/.9901/.8319/.3649; .0717/.0881/.0083/.8319; 1.6035/.1868/1.9070; .3850/.0934/.5216`|
|166 / A0|`4.0645/.9831/.9780/.9878/.9498/-3.7689; .0169/.0216/.0118/.9498; 3.9780/.0708/.3985; .9542/.0192/.0266`|
|285 / A2|`-.9071/.2876/.4575/.9890/.1301/3.7555; .7124/.1560/.0014/.1301; .0253/.9000/.5140; .1989/.4769/.3242`|
|285 / D1|`-.6631/.3400/.4426/.9913/.1492/3.8373; .6600/.1896/.0013/.1492; .0347/.8769/.6072; .1963/.4557/.3480`|
|285 / A0|`-.3044/.4245/.5434/.9933/.2291/4.3422; .5755/.1938/.0015/.2291; .0260/.8007/1.0208; .1702/.3694/.4604`|

### Three-point descriptive geometry

For SUPPORT 152 and 285, D1 raises entitlement/q-authorized and reduces the
NOT_ENTITLED final advantage, all toward A0, but leaves SUPPORT behind
NOT_ENTITLED: S-NE final-logit margins are `-.6714 -> -.2229 -> +.0792` and
`-.3859 -> -.2697 -> +.2201`. The q-predicate movement for 152 overshoots
A0 (`.1733 -> .2371 -> .2079`), and its polarity margin is essentially flat
from A2 to D1; 285's predicate probability dips before rising
(`.4575 -> .4426 -> .5434`). These are row-specific non-monotonic components,
not evidence of a response curve. The coherent retained failure is the
low-authorized / NOT_ENTITLED decision side.

For REFUTE 157, entitlement/q-authorized and most gate quantities move from
A2 toward A0 (`.5710 -> .7711 -> .8601` for both entitlement and qA), while
the final R-S logit margin remains on the SUPPORT side
(`-.1404 -> -.0942 -> +1.3119`) and native PolM remains near its A2 value
(`+.2459 -> +.1222 -> -1.5252`). Thus D1 repairs much of the entitlement-side
geometry without repairing the R/S discrimination observable.

For REFUTE 166, A2 is already correct, so this is a D1-introduced residual,
not a remaining A2 R/S error. Entitlement/qA move toward A0
(`.6969 -> .8319 -> .9498`), but D1 changes the R/S final margin from correct
to SUPPORT-side (`+.8617 -> -.3035 -> +3.5795`) and PolM from `-1.2365` to
`+.3649` before A0 reaches `-3.7689`. Predicate coverage and q-predicate are
also non-monotonic (`.9688 -> .9051 -> .9780`; `.0271 -> .0881 -> .0216`).
This is not an earlier-target failure: all three earlier structural targets
are pass/authorized targets in all arms; it is a downstream exported
polarity/final-boundary association.

## 4. Cross-seed controls and historical relationship

Both REFUTE residual IDs are correct in D1 seeds180 and181. Their seed180 /
seed181 / seed182 qA values remain relatively high (`.8463/.7772/.7711` for
157; `.8957/.8645/.8319` for 166), whereas PolM is REFUTE-side in both
successful controls and SUPPORT-side in seed182: 157 `-3.0482/-1.8292/+.1222`
and 166 `-1.9295/-2.3702/+.3649` (the space before `+.1222` is typographic
only). The final R-S margins make the same separation: 157
`+2.5796/+1.4217/-.0942`, 166 `+1.7282/+2.0491/-.3035`. This is a matched,
descriptive distinction between successful recovery and seed182 failure; it
does not identify the gradient edge that produced it.

SUPPORT 152 succeeds in seed180, where entitlement/qA is `.2138` and the
S-NE logit margin is exactly `0` (SUPPORT prediction at full precision), but
fails in seed181/182 with qA `.1395/.1624` and S-NE `-.3572/-.2229`.
SUPPORT 285 fails in all three D1 seeds (qA `.1768/.1606/.1492`; S-NE
`-.1641/-.2628/-.2697`), so it supplies no successful-D1 control.

Against the frozen C1 A0->A2 recurrent REFUTE/SUPPORT population,
`generated_fact_157__none` is a listed 2-of-3 recurrent REFUTE ID. The other
three are not members of the frozen recurrent C1 lists: 166 was A2-correct in
seed182, while 152 and 285 do not meet that frozen 2/3-or-3/3 SUPPORT
recurrence criterion. Historical recurrence is not redefined here.

## 5. Bounded localization ladder and aggregate interpretation

|stable ID|strongest supportable level|bounded rationale|
|---|---|---|
|`generated_fact_157__none`|`LEVEL_1_POLARITY_ASSOCIATED`|High/partly repaired entitlement geometry coexists with D1's SUPPORT-side PolM and R-S decision; controls retain REFUTE-side values.|
|`generated_fact_166__paraphrase`|`LEVEL_1_POLARITY_ASSOCIATED`|D1 introduces a SUPPORT-side PolM and R-S final reversal despite all authorized structural targets and increasing qA.|
|`generated_fact_152__polarity_flip`|`LEVEL_2_AUTHORIZATION_ASSOCIATED`|D1 retains a negative SUPPORT-vs-NE margin with qA/entitlement below A0; its scalar polarity margin does not show the analogous adverse reversal.|
|`generated_fact_285__polarity_flip`|`LEVEL_2_AUTHORIZATION_ASSOCIATED`|The same support-to-NE, low qA/entitlement, negative S-NE pattern persists in every D1 seed.|

The residual is therefore **heterogeneous row-specific residuals**: two
REFUTE->SUPPORT rows are a small continuation of the previously observed
R/S/polarity-associated pattern (one recurrent, one new at D1), while two
SUPPORT->NOT_ENTITLED rows are authorization-associated. It is not supported
as a single authorization-side explanation, nor as a single polarity-only
explanation.

## 6. Falsification discipline and decision

Directly observed: the aligned labels/transitions, targets, native scalar
gate/q/final/polarity-margin values, and matched-control values above.
Descriptive localization: the level assignments, based on co-movement and
separation in those exported quantities. Unidentifiable from these exports:
predicted reason-head state, polarity probability/logit vectors, a causal
gradient path or individual ownership edge, state ownership, and any native
Mamba mechanism. Association is not causation.

**Next-step decision: B.** Residual evidence is insufficient to justify a
narrowly specified D2 edge-specific ownership hypothesis. The four rows split
across R/S-boundary and support-to-NE patterns, and exported outputs do not
localize an edge or establish a common causal path. The next authorized step
should remain observational/static, not a D2 execution decision.

## 7. Validation and output identity

Validation actually performed: exact stable-ID/gold/input reconciliation;
raw-export recomputation of the four-row primary predicate and all reported
transitions; provenance/configuration reconciliation; and no checkpoint
access. No training, evaluation, inference, or checkpoint loading occurred.

Output path: `reports/reason_router_p3w7_seed8192_d1_seed182_residual_static_localization_report_candidate.md`.
Byte count and SHA256 are recorded in the delivery after the final write.
`git diff --check` and final status are also recorded in that delivery.
