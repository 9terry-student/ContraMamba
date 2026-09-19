# ContraMamba Gen4 — Causal-Atlas-Guided Steering Source-Eligibility Correction

## Status

`PROSPECTIVE_SOURCE_ELIGIBILITY_CORRECTION`

This document corrects one dataset-eligibility defect in the frozen causal-atlas-guided steering design before any steering model response, scientific model forward, steering inference, or p-value has been produced.

It does **not** reopen the steering rule, causal geometry, intervention site, intervention magnitude, outcome definition, harm gate, statistical test, or success criterion.

## 1. Frozen design being corrected

Frozen design:

`reports/reason_router_gen4_causal_atlas_guided_steering_design.md`

Frozen design commit:

`69256339cde93d77db44851782327a18bbdf965a`

The original design prospectively specified:

- pinned AVeriTeC repository `MichSchli/AVeriTeC`;
- upstream commit `7c62d1ec8df3fb560d6efe2b85fa191135636f81`;
- fresh source `data/train.json`;
- train git blob SHA1 `0f190e115cf2ee23416e8a539c8d6ac043d7cc83`;
- dev git blob SHA1 `40974243267f395dc583d805d10f043812419249`;
- three-class label compatibility;
- dev-claim and dev-URL exclusion;
- within-train normalized-claim deduplication;
- reuse of the frozen claim/evidence serialization and boundary token contract;
- no model-response-based filtering.

## 2. Dataset-only defect discovered before scientific execution

A pinned-source CPU/static audit of the exact prospective train cohort found two source-structure anomalies among the originally derived 2800 rows:

1. `train_index=438`, label `Not Enough Evidence`:
   - one question string is exactly empty;
   - claim is non-empty;
   - answers remain valid.

2. `train_index=1948`, label `Supported`:
   - claim is exactly empty.

No steering model response had been generated when these facts were discovered.

No scientific model forward had been executed.

No steering p-value had been calculated.

Therefore this correction is prospective with respect to the steering experiment.

## 3. Treatment of the empty-question row

`train_index=438` remains eligible.

Reason:

- its claim is non-empty;
- the pinned serializer deterministically represents the empty question together with its valid answer;
- the resulting evidence text remains non-empty;
- the downstream token gate can still validate the actual serialized sequence;
- excluding this row would introduce an unnecessary source-content criterion not required by the steering estimand.

The serializer is not changed.

The freshness rule is not changed for this row.

## 4. Treatment of the empty-claim row

`train_index=1948` is prospectively excluded from the steering cohort.

Reason:

The steering experiment is defined at a frozen `A_CLAIM_EVIDENCE_BOUNDARY` between a consumed claim and consumed evidence. An example with zero consumed claim tokens does not instantiate that claim/evidence boundary in the same sense as the rest of the cohort.

Allowing this row by redefining the degenerate input as effectively `EOS + evidence` would modify the frozen input/anchor semantics to preserve a nominal row count. That would be less rigorous than prospectively excluding a structurally ineligible source example before any steering response is observed.

This exclusion is based only on pinned source bytes and the already-frozen input contract.

It is not based on model behavior, margin, label-specific steering effects, or any response from the steering intervention.

## 5. Corrected source-eligibility rule

Define normalized claim text exactly as in the frozen design:

1. Unicode text as stored in the pinned JSON;
2. strip leading/trailing whitespace;
3. collapse every maximal whitespace run to one ASCII space;
4. lowercase using the runtime language's deterministic Unicode lowercase operation.

The corrected cohort derivation order is:

1. exclude label `Conflicting Evidence/Cherrypicking`;
2. exclude any row whose normalized claim is empty;
3. exclude a train row if its normalized claim exactly matches any normalized dev claim;
4. exclude a train row if its non-empty `original_claim_url` exactly matches any non-empty dev `original_claim_url`;
5. among remaining rows with duplicate normalized claim text, retain only the lowest zero-based train source index.

No other source filter is added.

## 6. Corrected deterministic cohort counts

Pinned train source rows:

`3068`

Pinned train label counts:

- `Refuted`: 1742;
- `Supported`: 849;
- `Not Enough Evidence`: 282;
- `Conflicting Evidence/Cherrypicking`: 195.

Three-class-compatible rows before freshness/eligibility filtering:

`2873`

Corrected exclusion counts:

- incompatible label: `195`;
- empty normalized claim: `1`;
- dev normalized-claim overlap: `6`;
- dev non-empty URL overlap: `1`;
- later within-train duplicate normalized claim: `66`.

Corrected final fresh eligible cohort:

`2799`

Corrected final source-label counts:

- `Refuted`: `1727`;
- `Supported`: `805`;
- `Not Enough Evidence`: `267`.

Use all 2799 eligible rows.

No response-adaptive subsampling is permitted.

## 7. Corrected forward budget

The frozen steering conditions remain exactly:

1. `native`;
2. `p3_mirror_steer`;
3. `p5_matched_control`.

Therefore the corrected expected scientific forward budget is:

`2799 rows × 3 conditions = 8397 full-model forwards`

This correction does not authorize those forwards.

## 8. Frozen elements that remain unchanged

The following remain exactly frozen:

- model scale: Mamba-370M;
- frozen backbone and downstream checkpoint;
- selected plane: P3;
- response-blind matched control plane: P5;
- intervention block: 35;
- claim budget: 63 active tokens;
- boundary slot: 1 token;
- evidence budget: 64 active tokens;
- total active encoding contract: 128 positions;
- deterministic gold question-answer evidence serializer;
- textual `justification` exclusion;
- mapped class order `REFUTE`, `NOT_ENTITLED`, `SUPPORT`;
- anchor `A_CLAIM_EVIDENCE_BOUNDARY`;
- target offset `anchor + 2`;
- fixed mirror contrast `delta(h) = C3(h) - C5_match(h)`;
- `h_steer = h + delta(h)`;
- `h_control = h - delta(h)`;
- no free steering coefficient;
- no magnitude, sign, layer, token, prompt, anchor, or retrieval search;
- target population definition;
- preservation population definition;
- one-sided exact McNemar/binomial primary test;
- exactly one primary p-value;
- alpha `0.05`;
- preservation gate `damage_rate_on_native_correct <= 0.05`;
- utility gate `C > D`;
- primary success labels.

## 9. Implementation consequence

The fresh-cohort materializer must be corrected so that:

- empty question strings remain permitted when the serialized evidence remains structurally valid;
- empty normalized claims are excluded prospectively before dev-overlap and within-train deduplication;
- the exact corrected final cohort is 2799 rows;
- final label counts are exactly `1727 / 805 / 267`;
- the token gate still requires a non-empty consumed claim for every included row;
- no model checkpoint is loaded;
- no model forward is executed;
- no CUDA is executed;
- no scientific inference is executed;
- no p-value is added.

The previously frozen 2800-row design count is superseded only by this source-eligibility correction.

## 10. Execution boundary

This document authorizes only correction of the CPU/static cohort materializer and its tests.

It does not authorize scientific steering execution.

After the corrected implementation is frozen, the 2799-row cohort must be materialized, validated, and frozen before the three-condition steering runner is implemented or executed.
