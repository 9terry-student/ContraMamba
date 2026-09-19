# ContraMamba Experiment 2 Scale-Family Correction
## AVeriTeC Gold-Evidence External Causal Transfer

### Status

`PRE_RESPONSE_SCIENTIFIC_DESIGN_CORRECTION`

Current repository anchor when this correction was made:

`de4a00efdf3008a66d6b3f24d46a273d4b74fb9f`

This correction is made before any Experiment 2 model checkpoint load, model forward,
CUDA execution, or external-transfer response inspection.

It supersedes only the **primary-scale choice** in:

`reports/reason_router_gen4_averitec_gold_evidence_external_transfer_static_feasibility_audit.md`

All other frozen Experiment 2 rules remain unchanged unless explicitly stated below.

---

## 1. Why the 370M-only choice is corrected

The previous static audit selected Mamba-370M as the first external-transfer scale
because the new Experiment 1 bridge supported 370M while Mamba-1.4B failed.

That correctly excludes Mamba-1.4B from the first external-transfer family, but it
unnecessarily omitted Mamba-130M.

Mamba-130M already has a completed prospective downstream behavioral bridge on a fresh
XG1 cohort.

Frozen 130M behavioral result:

- model: `state-spaces/mamba-130m-hf`;
- seed: `181`;
- arm: `G3-GROUP-D-HALF`;
- behavioral checkpoint SHA256:
  `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f`;
- selected plane: `P3`;
- matched response-blind control: `P5`;
- intervention layer: `17`;
- N: `300`;
- mean `D_BEH`: `+0.008332191656033197`;
- one-sided p-value: `2.454643852170648e-17`;
- result:
  `SEED181_BEHAVIORAL_RESTORATION_BRIDGE_SUPPORTED`.

Mamba-370M independently has a completed positive behavioral bridge:

- selected plane: `P3`;
- control plane: `P5`;
- intervention layer: `35`;
- N: `300`;
- mean `D_BEH`: `+0.0011206856990853946`;
- Holm-adjusted p-value in the 370M/1.4B family:
  `1.4957180487214262e-08`;
- supported: `true`.

Mamba-1.4B has a valid internal causal core result but its fresh downstream behavioral
bridge is not supported and has negative mean `D_BEH`.

Therefore the scientifically clean first natural-language transfer family is:

`MAMBA-130M + MAMBA-370M`

not 370M alone, and not 130M+370M+1.4B.

---

## 2. Corrected scientific question

The first AVeriTeC external causal-transfer study asks:

> Does the frozen behaviorally validated causal component transfer from synthetic XG1
> to natural-language AVeriTeC gold-evidence inputs at both Mamba-130M and Mamba-370M
> under one shared response-blind cohort and boundary-coordinate protocol?

This directly tests cross-scale external transfer across the two scales for which a
downstream behavioral bridge was already established.

Mamba-1.4B is excluded because including it would mix natural-language transfer with a
pre-existing failure of the synthetic downstream behavioral bridge.

---

## 3. Shared AVeriTeC cohort and tokenizer gate

The previously frozen AVeriTeC rules remain unchanged:

- pinned official AVeriTeC development bytes;
- exact source label inventory;
- compatible labels:
  `Refuted`, `Not Enough Evidence`, `Supported`;
- conflicting-evidence class excluded;
- expected compatible count: `462`;
- gold question-answer evidence only;
- no retrieval;
- no textual justification as model input;
- active serialization:
  `claim[:63] + EOS(0) + evidence[:64]`;
- response-blind anchor:
  `A_CLAIM_EVIDENCE_BOUNDARY`;
- target offset:
  `+2`;
- mandatory `PASS_462_OF_462` token gate.

The completed 130M behavioral bridge and the 370M behavioral bridge record the same
active tokenizer byte identities:

- `tokenizer.json` SHA256:
  `b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`;
- `tokenizer_config.json` SHA256:
  `9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`;
- `special_tokens_map.json` SHA256:
  `57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`;
- tokenizers runtime used by the completed active-encoding validation:
  `0.22.2`.

Therefore one exact-byte tokenizer/boundary gate can be shared prospectively by both
primary scales.

This shared tokenizer identity does not imply that the two backbones or their hidden
causal coordinates are identical.

---

## 4. Corrected primary family

Exactly two new primary external-transfer tests are planned:

1. Mamba-130M `D_EXT`;
2. Mamba-370M `D_EXT`.

For each scale:

`D_EXT,i = M_native,i - M_control,i`

where `M` is the final three-way correct-class logit margin for AVeriTeC item `i`.

Each scale uses:

- one-sided one-sample Student t-test;
- alternative: `greater`;
- N: `462` if and only if the shared token gate passes `462/462`.

The external primary family contains exactly:

`2 p-values`

Multiplicity correction:

`Holm`

Family alpha:

`0.05`

Historical behavioral p-values are not part of this new family.

Interpretation:

- both scales pass after Holm:
  `CROSS_SCALE_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_THROUGH_370M_SUPPORTED`;
- exactly one passes:
  `SCALE_SPECIFIC_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_ONLY`;
- neither passes:
  `AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_NOT_ESTABLISHED`.

No third scale or alternative anchor may be added as rescue.

---

## 5. Frozen scale-local causal objects

### Mamba-130M

- selected plane: `P3`;
- response-blind control: `P5`;
- intervention layer: `17`;
- same `+2` transported boundary offset;
- no reselection.

### Mamba-370M

- selected plane: `P3`;
- response-blind control: `P5`;
- intervention layer: `35`;
- same `+2` transported boundary offset;
- no reselection.

Plane-number equality is not interpreted as established cross-scale semantic identity.

These are scale-local frozen causal objects.

---

## 6. Planned execution budget after implementation freeze

Conditions per compatible item per scale:

1. native;
2. dominant neutralized;
3. matched dominant control.

No redundant exact-restoration forward is required for the external primary contrast.

If the shared token gate passes all 462 items:

- per scale:
  `462 × 3 = 1386 full-model forwards`;
- two scales:
  `2772 full-model forwards total`.

No training.

No backward pass.

No model response is authorized by this correction alone.

---

## 7. Current blocker is environmental, not scientific

The first local token-gate attempt stopped before cohort materialization because the
local Python runtime had:

`tokenizers==0.20.3`

while the frozen active-encoding contract requires:

`tokenizers==0.22.2`.

No model checkpoint was loaded.

No model forward occurred.

No external response was observed.

The gate must be rerun under `tokenizers==0.22.2`, preferably in an isolated temporary
environment rather than mutating the project's general Python environment.

---

## 8. Final corrected next step

`EXPERIMENT_2_SHARED_130M_370M_AVERITEC_TOKEN_GATE`

Requirements:

- exact pinned AVeriTeC source authentication;
- exact shared tokenizer byte authentication;
- tokenizers runtime `0.22.2`;
- deterministic 462-row compatible cohort;
- boundary+2 gate passes `462/462`;
- no checkpoint load;
- no model forward;
- no CUDA;
- no new p-value.

Only after this gate is frozen may the 130M+370M external runner and two-test Holm
analyzer be implemented.
