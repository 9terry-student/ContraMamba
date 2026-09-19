# ContraMamba Gen4 — Experiment 3 Causal-Atlas-Guided Steering Design

## Status

This document preregisters the next post-synthesis experiment after closure of the pre-emission precursor program.

It is a design document only.

It does not authorize model execution by itself and creates no scientific model-response evidence.

## 1. Transition from the precursor program

The pre-emission precursor program is closed.

Original three-class finite-grammar Stage A:

- raw evidence freeze:
  `a58b1ebef0957a8437e0255fb4069d2fe9cf8449`;
- closure:
  `130aa76cf3bb0816505a45a18df8334445e6f0ce`;
- result:
  `PRIMARY_STAGE_A_TEMPORAL_PRECEDENCE_NOT_ESTIMABLE`.

Prospective forced-decisive Stage A:

- raw evidence freeze:
  `880eab834c442054642773935a88fa60a31287c3`;
- confirmatory analysis freeze:
  `7669bf998ac333af0a674b8db9691e34fd988143`;
- result:
  `FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`.

No Stage B or Stage C precursor experiment is licensed.

The project now resumes Experiment 3 from the frozen post-synthesis program:

`CAUSAL_ATLAS_GUIDED_STEERING`.

## 2. Scientific question

Primary question:

> Can the already-frozen Mamba-370M P3 causal geometry guide a fixed intervention that improves natural-language downstream decisions on a fresh AVeriTeC population without unacceptable damage to examples the native model already gets correct?

This is a practical control experiment.

It is not a precursor experiment, not a new direction-learning experiment, and not a hyperparameter sweep.

## 3. Why Mamba-370M natural-language steering is admissible

Frozen prior evidence:

### Synthetic downstream behavioral bridge

Mamba-370M:

- selected plane: `P3`;
- response-blind control: `P5`;
- intervention layer: `35`;
- shared bridge population N=300;
- Holm-supported behavioral bridge;
- primary mean `D_BEH = +0.0011206856990853946`;
- Holm p:
  `1.4957180487214262e-08`.

Mamba-1.4B did not support the corresponding behavioral bridge.

Therefore this steering experiment remains Mamba-370M only.

### Natural-language external transfer

On the frozen AVeriTeC dev-compatible population:

- Mamba-370M primary external transfer was supported;
- mean `D_EXT = +4.5550022e-05`;
- Holm p:
  `0.0113815`.

This supports using a natural-language steering target at 370M.

It does not establish benchmark accuracy improvement and does not justify tuning on the already-observed dev responses.

## 4. Fresh evaluation source

Use the same pinned upstream AVeriTeC repository:

- repository:
  `MichSchli/AVeriTeC`;
- commit:
  `7c62d1ec8df3fb560d6efe2b85fa191135636f81`.

Use a fresh source file not used by the completed dev transfer experiment:

`data/train.json`

Frozen upstream identity:

- git blob SHA1:
  `0f190e115cf2ee23416e8a539c8d6ac043d7cc83`;
- bytes:
  `10184813`.

Dataset-only static audit facts before any steering response is generated:

- source rows: 3068;
- `Supported`: 849;
- `Refuted`: 1742;
- `Not Enough Evidence`: 282;
- `Conflicting Evidence/Cherrypicking`: 195;
- three-class-compatible rows before freshness filtering: 2873.

These are source-data facts, not model-response observations.

## 5. Freshness and deduplication rule

The already-observed AVeriTeC dev source remains frozen as:

- `data/dev.json`;
- git blob SHA1:
  `40974243267f395dc583d805d10f043812419249`.

Define normalized claim text as:

1. Unicode text as stored in the pinned JSON;
2. strip leading/trailing whitespace;
3. collapse every maximal whitespace run to one ASCII space;
4. lowercase using the runtime language's deterministic Unicode lowercase operation.

Derive the fresh train cohort in this exact order:

1. exclude label `Conflicting Evidence/Cherrypicking`;
2. exclude a train row if its normalized claim exactly matches any normalized dev claim;
3. exclude a train row if its non-empty `original_claim_url` exactly matches any non-empty dev `original_claim_url`;
4. among remaining rows with duplicate normalized claim text, retain only the lowest zero-based train source index.

Dataset-only audit result under this rule:

- excluded incompatible label: 195;
- excluded dev-claim overlap: 6;
- excluded dev-URL overlap: 1;
- excluded later within-train duplicate claim: 66;
- final fresh unique compatible cohort: **2800**.

Frozen final source-label counts:

- `Refuted`: 1727;
- `Supported`: 806;
- `Not Enough Evidence`: 267.

Use all 2800 rows.

Do not subsample based on model responses, label-specific effects, margins, or prior steering results.

## 6. Input and anchor contract

Reuse the completed AVeriTeC natural-language external-transfer serialization contract unchanged:

- claim budget: 63 active tokens;
- boundary slot: 1 token;
- evidence budget: 64 active tokens;
- total active encoding contract: 128 positions;
- tokenizer bytes remain the frozen Mamba tokenizer;
- gold question-answer evidence serialization remains deterministic;
- textual `justification` is not used;
- fourth AVeriTeC label remains excluded;
- mapped class order remains:
  `REFUTE`, `NOT_ENTITLED`, `SUPPORT`.

Frozen anchor:

`A_CLAIM_EVIDENCE_BOUNDARY`

Frozen intervention offset:

`target_token = anchor + 2`

No prompt search, anchor search, token search, or retrieval change is allowed.

## 7. Frozen model and causal geometry

Model scale:

`Mamba-370M`

Frozen backbone:

- Hugging Face repository:
  `state-spaces/mamba-370m-hf`;
- revision:
  `589179554943157be31701edd8b4558889276674`.

Frozen downstream checkpoint SHA256:

`9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72`

Frozen causal object:

- selected plane: `P3`;
- response-blind matched control plane: `P5`;
- intervention block: `35`;
- no layer scan;
- no plane reselection.

## 8. Fixed mirror-steering intervention

Let `h` be the native strong-channel intervention-space vector at the frozen target coordinate.

Using the frozen P3 basis, compute the native P3 coefficients:

- `a`;
- `b`.

Define:

`C3(h) = a * P3_plus + b * P3_minus`

Using the same native P3 coefficients, define the matched P5 component:

`C5_match(h) = a * P5_plus + b * P5_minus`

Define the frozen causal contrast vector:

`delta(h) = C3(h) - C5_match(h)`

The already-established matched-control replacement is:

`h_control = h - delta(h)`

The new steering intervention is the exact mirror step beyond native:

`h_steer = h + delta(h)`

This is prospectively fixed.

There is:

- no free steering coefficient;
- no `lambda`;
- no epsilon search;
- no magnitude calibration;
- no sign search;
- no trigger threshold;
- no layer search;
- no response-dependent intervention strength.

The magnitude is determined entirely by the already-frozen per-example causal contrast.

## 9. Conditions

Run exactly three full-model conditions for every fresh cohort row:

1. `native`
2. `p3_mirror_steer`
3. `p5_matched_control`

Definitions:

- `native`: no intervention;
- `p3_mirror_steer`: `h + delta(h)`;
- `p5_matched_control`: `h - delta(h)`.

The third condition is the already-established response-blind causal control and provides a mechanistic directionality check.

No neutralized, restored, additional magnitude, alternative plane, or alternative layer condition is added.

Expected scientific forward budget:

`2800 rows × 3 conditions = 8400 full-model forwards`

The raw execution performs no statistical inference.

## 10. Prospective target and preservation populations

Define populations only from the `native` prediction on the fresh 2800-row cohort.

### Target population

Rows where:

`native_prediction != mapped_gold_label`

These are native errors.

### Preservation population

Rows where:

`native_prediction == mapped_gold_label`

These are native-correct examples.

This rule is frozen before any steering output is inspected.

Native correctness may define the evaluation strata, but it may not alter:

- steering vector;
- steering sign;
- intervention magnitude;
- layer;
- token;
- population membership in the source cohort.

## 11. Primary utility endpoint

For each row define whether the mirror steering changes correctness relative to native.

Let:

- `C` = count of native-incorrect rows that become correct under `p3_mirror_steer`;
- `D` = count of native-correct rows that become incorrect under `p3_mirror_steer`.

Primary inferential test:

one-sided exact McNemar/binomial test on the discordant pairs:

`H0: P(correction | discordant) <= 0.5`

versus

`H1: P(correction | discordant) > 0.5`

Equivalently, conditional on `C + D`, test whether corrections exceed damages.

Exactly one primary p-value is added.

Primary alpha:

`0.05`

No multiplicity correction is required because there is exactly one primary p-value.

If `C + D = 0`, the utility endpoint is not estimable and the experiment closes without a p-value.

## 12. Harm / preservation gate

A statistically positive correction-vs-damage result is not sufficient by itself.

The fixed preservation gate is:

`damage_rate_on_native_correct <= 0.05`

where:

`damage_rate_on_native_correct = D / N_native_correct`

The `0.05` bound is a prospectively fixed operational safety limit: no more than one in twenty examples already correct under native may be broken by steering.

This threshold is not calibrated from AVeriTeC responses and may not be changed after execution.

A second non-inferential utility gate is:

`C > D`

which guarantees positive net accuracy change on the fixed cohort.

## 13. Primary success rule

Experiment 3 steering is supported only if all of the following hold:

1. `C + D > 0`;
2. one-sided exact McNemar/binomial primary p `< 0.05`;
3. `C > D`;
4. `damage_rate_on_native_correct <= 0.05`.

If all four hold:

`AVERITEC_370M_FIXED_MIRROR_P3_STEERING_SUPPORTED`

Otherwise:

`AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED`

No rescue is permitted within this experiment.

## 14. Required descriptive diagnostics

These are descriptive only and add no p-values:

- native accuracy;
- mirror-steer accuracy;
- matched-control accuracy;
- correction count and rate on native errors;
- damage count and rate on native-correct rows;
- net accuracy change;
- mapped-label-specific correction and damage counts;
- correct-class margin change:
  `M_steer - M_native`;
- mean margin change on target rows;
- mean margin change on preservation rows;
- matched-control margin contrast:
  `M_native - M_control`;
- fraction of rows where the mirror-steer margin exceeds native;
- prediction transition table.

Do not promote a favorable subgroup diagnostic into a replacement primary claim.

## 15. Response-blind control interpretation

The matched P5 condition is not another steering candidate.

It is retained because the prior causal atlas used it as the response-blind contrast.

The preferred directional pattern is descriptively:

`M_steer > M_native > M_control`

but this pattern is not a separate inferential family and does not replace the primary correction-vs-damage test.

## 16. Anti-tuning boundary

After this design is frozen, do not modify within this experiment:

- source split;
- freshness rule;
- deduplication rule;
- serializer;
- token budget;
- anchor;
- target offset;
- P3;
- P5;
- intervention layer;
- mirror sign;
- mirror magnitude;
- target definition;
- preservation definition;
- alpha;
- harm threshold;
- primary test;
- success rule.

Specifically prohibited:

- trying `0.25`, `0.5`, `2.0`, or any other steering multiplier after seeing results;
- selecting only a favorable AVeriTeC label;
- excluding low-margin native-correct rows after seeing steering damage;
- changing the anchor or token position;
- replacing P5 with another control plane;
- using the failed precursor signal as a trigger;
- introducing a learned gate.

Any materially different steering rule is a new prospective experiment.

## 17. Claim boundary

A positive result would support only the bounded statement:

> On a fresh deduplicated AVeriTeC train-derived three-class gold-evidence cohort, the frozen Mamba-370M P3 causal atlas supported a pre-specified mirror intervention that produced more corrections than damages under the preregistered preservation gate.

It would not establish:

- general benchmark superiority;
- four-class AVeriTeC performance;
- retrieval competence;
- free-form hallucination prevention;
- a pre-emission precursor;
- cross-scale steering;
- universal P3 rank identity;
- optimal steering magnitude.

A negative result does not erase the already-frozen behavioral bridge or external causal transfer.

It means the specific fixed mirror extrapolation did not establish useful steering under the preregistered utility/harm rule.

## 18. Execution order

1. freeze this design;
2. implement a CPU/static train-source freshness and token-gate materializer;
3. validate and freeze the 2800-row fresh cohort before model response inspection;
4. implement the 370M three-condition raw steering runner;
5. validate and freeze implementation;
6. run one pinned 2-GPU raw execution;
7. collect/import/validate/freeze raw outputs;
8. only then execute the single primary exact McNemar/binomial inference and descriptive diagnostics;
9. freeze and interpret the result before Experiment 4.

No scientific model forward is authorized by this design document alone.
