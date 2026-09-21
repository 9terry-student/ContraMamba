# ContraMamba Gen4 — Mamba-1.4B AVeriTeC Negative-Sign External Transfer Prospective Plan

Status: PROSPECTIVE_PLAN_ONLY
Execution authorized by this file: NO
Training authorized: NO

Frozen planning base commit:

`e1a3b1113ae5b57aa9b221053a08b11121148795`

## 1. Scientific purpose

This study asks whether the already-frozen Mamba-1.4B scale-local causal displacement
retains its negative downstream correct-class-margin sign when transported from the
synthetic XG1 domain to natural-language AVeriTeC gold-evidence inputs.

This is a new prospective one-scale external-transfer extension.

It does not reopen, rescue, or modify the completed Mamba-130M/Mamba-370M AVeriTeC
primary family.

The motivation is fixed before any Mamba-1.4B AVeriTeC model response is observed:

- frozen synthetic Mamba-1.4B behavioral bridge:
  `mean D_BEH = -0.002012885312239329`;
- frozen Mamba-1.4B local readout alignment:
  `mean Delta_L = -0.0011954475058862238`;
- both use selected plane `P5` against response-blind control `P4`.

The prospective external hypothesis therefore has a negative, not positive, direction.

## 2. Frozen Mamba-1.4B identity

Use exactly:

- model repository: `state-spaces/mamba-1.4b-hf`;
- Hugging Face revision:
  `6e46eae61c27280517feef46f536d16b91076f08`;
- compact checkpoint SHA256:
  `915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a`;
- selected scale-local plane: `P5`;
- response-blind control plane: `P4`;
- intervention layer: `35`;
- anchor offset: `+2`.

No checkpoint, plane, control, layer, or offset reselection is permitted.

## 3. Frozen prior evidence boundary

The completed synthetic behavioral bridge on
`xg1_fact_4801..xg1_fact_5100` reported:

- `N = 300`;
- mean `D_BEH = -0.002012885312239329`;
- fraction positive `= 0.37`;
- `t(299) = -6.823666290433015`;
- one-sided greater p-value was approximately `1`, so the historical positive bridge
  criterion was not supported.

The later frozen local readout analysis independently reported:

- mean `Delta_L_1.4B = -0.0011954475058862238`;
- positive fraction `= 0.35333333333333333`.

These historical outcomes fix the direction of the new external hypothesis only.
Their p-values are not part of the new AVeriTeC family.

## 4. Pinned AVeriTeC source and compatible population

Reuse the exact previously authenticated official AVeriTeC development source:

- upstream repository: `MichSchli/AVeriTeC`;
- upstream commit:
  `7c62d1ec8df3fb560d6efe2b85fa191135636f81`;
- source path: `data/dev.json`;
- Git blob SHA1:
  `40974243267f395dc583d805d10f043812419249`;
- source bytes: `1785475`;
- total examples: `500`.

Frozen source-label counts:

- `Refuted`: `305`;
- `Supported`: `122`;
- `Not Enough Evidence`: `35`;
- `Conflicting Evidence/Cherrypicking`: `38`.

Compatible three-way cohort:

- `Refuted -> REFUTE = 0`;
- `Not Enough Evidence -> NOT_ENTITLED = 1`;
- `Supported -> SUPPORT = 2`;
- `Conflicting Evidence/Cherrypicking` excluded.

Expected compatible population:

`N = 462`.

No label-balanced subsampling, retrieval, justification-field input, response-guided
filtering, or post-response cohort change is allowed.

## 5. Deterministic gold-evidence serialization

For each compatible example, preserve the existing AVeriTeC gold-evidence rule:

- use source question order;
- use source answer order;
- serialize every question-answer pair as:
  `Question: <question>\nAnswer: <answer>`;
- join blocks with two newlines;
- do not use the textual `justification` field;
- execute no retrieval and no question generation.

The active model input contract remains:

`claim[:63] + EOS + evidence[:64]`

with:

- max length `128`;
- claim budget `63`;
- evidence budget `64`;
- `add_special_tokens=False`.

## 6. Mandatory Mamba-1.4B tokenizer gate

The Mamba-1.4B tokenizer is not byte-identical to the tokenizer used by the completed
130M/370M AVeriTeC family.

Therefore the previously materialized 130M/370M tokenized cohort is not sufficient as
the 1.4B execution input.

Use exactly the Mamba-1.4B tokenizer snapshot:

- repository: `state-spaces/mamba-1.4b-hf`;
- revision:
  `6e46eae61c27280517feef46f536d16b91076f08`;
- `tokenizer.json` SHA256:
  `3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8`;
- `tokenizer_config.json` SHA256:
  `3ba257483d22a5a84aab5465aa427e59bdaeb55f09fb14349e2d571ff67e8020`;
- required tokenizers runtime: `0.22.2`.

The response-blind external anchor remains:

`A_CLAIM_EVIDENCE_BOUNDARY`

For consumed claim length `c`:

- separator index: `c`;
- intervention target: `c + 2`.

The token gate must pass all `462/462` compatible items and must verify that the
target is inside the attended serialized sequence.

The token gate:

- loads no model checkpoint;
- executes no model forward;
- executes no CUDA;
- performs no inference;
- emits no p-value.

If the gate does not pass `462/462`, scientific execution is blocked. No favorable
subset may be selected.

## 7. External intervention conditions

For every gated AVeriTeC example run exactly:

1. `native`;
2. `dominant_neutralized`;
3. `dominant_control`.

`dominant_control` is the coefficient-matched response-blind `P4` control corresponding
to the frozen `P5` selected component.

No exact-restoration condition is required because the native state already contains
the frozen selected component used by the external primary contrast.

No epsilon sweep, anchor search, plane search, layer search, or steering-strength
optimization is permitted.

## 8. Primary endpoint

For item `i`, define the final three-way correct-class logit margin under each
condition.

The primary external-transfer endpoint is:

`D_EXT_14B,i = M_native,i - M_control,i`.

This is the natural-language transported-coordinate analogue of the already-frozen
selected-versus-control downstream margin contrast.

## 9. Primary hypothesis

Exactly one new inferential test is permitted.

`H0: E[D_EXT_14B] >= 0`

`H1: E[D_EXT_14B] < 0`

Test:

- one-sided one-sample Student t-test;
- alternative: `less`;
- `N = 462` if and only if the token gate passes `462/462`;
- `df = 461`;
- alpha `0.05`;
- exactly one new primary p-value;
- no multiplicity correction because this extension contains one new test.

Support requires both:

1. `mean(D_EXT_14B) < 0`;
2. one-sided `p < 0.05`.

If either condition fails:

`MAMBA14B_AVERITEC_NEGATIVE_SIGN_TRANSFER_NOT_ESTABLISHED`.

If both pass:

`MAMBA14B_AVERITEC_NEGATIVE_SIGN_TRANSFER_SUPPORTED`.

Historical 130M/370M AVeriTeC p-values and historical synthetic p-values are not
included in this new family.

## 10. Raw-execution information boundary

The GPU raw runner must not read:

- the completed 130M/370M AVeriTeC item-level `D_EXT` values;
- the completed 130M/370M AVeriTeC p-values;
- any post-result AVeriTeC source-label decomposition;
- any 1.4B AVeriTeC response, because none exists prospectively.

It may read only the frozen source/cohort/tokenizer/model/checkpoint/geometry metadata
needed for execution.

Raw execution computes no p-value and no scientific conclusion.

## 11. Execution budget

If the token gate passes `462/462`:

- compatible items: `462`;
- conditions per item: `3`;
- full-model forwards:
  `462 x 3 = 1386`;
- backward passes: `0`;
- training steps: `0`;
- parameter updates: `0`.

A single T4 may be insufficient or inefficient for the exact 1.4B runtime. Execution
may use the available two T4 devices as deterministic item shards, but GPU assignment
must not alter the statistical sample or endpoint.

## 12. Raw artifact contract

Freeze exactly one raw run directory containing:

1. `external_transfer_rows.jsonl`;
2. `raw_external_transfer_summary.json`;
3. `artifact_manifest.json`;
4. `SHA256SUMS.txt`.

The raw summary must record:

- exact source/tokenizer/model/checkpoint identities;
- compatible count `462`;
- selected/control planes `P5/P4`;
- intervention layer `35`;
- anchor `A_CLAIM_EVIDENCE_BOUNDARY`;
- target offset `+2`;
- forward count `1386`;
- backward count `0`;
- training `false`;
- primary inference executed `false`;
- p-value count `0`;
- scientific conclusion `null`.

## 13. Static analysis after raw freeze

Only after raw evidence is frozen:

1. verify exact `462`-item coverage;
2. reconstruct `D_EXT_14B = M_native - M_control`;
3. execute the single prespecified one-sided `less` t-test;
4. apply the negative-mean sign gate;
5. freeze the result.

Descriptive outputs may include:

- mean, sample SD, median, quartiles, min/max;
- fraction negative / positive;
- prediction flips;
- `M_native - M_neutralized`;
- source-label breakdowns without subgroup p-values.

No second inferential test is permitted.

## 14. Cross-scale contextual boundary

After the 1.4B result is frozen, it may be placed descriptively beside the already
frozen 130M and 370M AVeriTeC results.

Allowed:

- per-scale mean `D_EXT`;
- sign fractions;
- checkpoint/plane/control identities;
- descriptive comparison with the synthetic/readout sign pattern.

Not allowed:

- a new three-scale trend p-value;
- a 130M/370M historical-family reopening;
- a pooled three-scale hypothesis test;
- a parameter-count threshold or zero-crossing estimate;
- response-dependent anchor, plane, or cohort changes.

## 15. Interpretation boundary

If the negative-sign primary test passes, the supported claim is limited to:

> At the frozen Mamba-1.4B checkpoint, the scale-local P5-versus-P4 causal
> displacement has a negative downstream correct-class-margin sign on the pinned
> AVeriTeC gold-evidence cohort under the response-blind claim/evidence boundary+2
> transport rule, consistent in sign with the separately frozen synthetic
> behavioral and local-readout results.

This does not establish:

- benchmark accuracy improvement;
- retrieval competence;
- universal negative transfer at 1.4B;
- a monotonic scaling law;
- a universal parameter threshold;
- complete causal mediation;
- semantic identity of the AVeriTeC boundary with the synthetic `A_IDENTITY` anchor.

## 16. Stop rule

Run the token gate once under the frozen 1.4B tokenizer.

If it passes, implement and freeze the raw runner before GPU execution.

After the first 1.4B AVeriTeC response is observed, do not alter:

- cohort;
- label mapping;
- tokenizer;
- serialization;
- anchor;
- offset;
- selected/control planes;
- intervention layer;
- checkpoint;
- endpoint;
- test direction;
- alpha

in response to the result.
