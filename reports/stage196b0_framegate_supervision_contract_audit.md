# Stage196-B0 FrameGate supervision-contract static audit

## 1. Files created

- `reports/stage196b0_framegate_supervision_contract_audit.md`
- `reports/stage196b0_framegate_supervision_contract_audit.json`

This is a read-only source/data-contract audit. No Python, analyzer, model forward, training, smoke run, threshold search, loss sweep, model patch, dataset mutation, commit, or push was performed.

## Decision

`STAGE196B0_FRAME_REPRESENTATION_FAILURE`

Recommended single Stage196-B1 intervention family:

`ADD_FRAME_STATE_POSITIVE_NEGATIVE_CONTRAST`

The taxonomy decision is operational: the current contract provides a frame-local target, direct unmasked BCE, trainable optimizer-owned FrameGate parameters, and an intact gradient path, while Stage196-A freezes recurrent FrameGate failures. It does **not** prove encoder-wide collapse, identify a unique latent subspace defect, authorize implementation, or supersede the earlier Stage179-A mixed/insufficient causal-localization closure.

## Scope and inspected evidence

Primary required evidence:

- `scripts/train_controlled_v6b_minimal.py`
- `scripts/train_controlled_v5.py`
- `scripts/build_controlled_v5.py`
- `data/controlled_v5_v3_without_time_swap.jsonl`
- `reports/stage196a_persistent_support_boundary_localization_spec.md`
- `scripts/analyze_stage196a_persistent_support_boundary.py`
- `src/contramamba/modeling_v6b_minimal.py`
- `src/contramamba/modeling_vnext_minimal.py`
- `src/contramamba/heads/frame_gate.py`
- `src/contramamba/heads/predicate_coverage.py`
- `src/contramamba/heads/sufficiency_gate.py`
- `src/contramamba/heads/entitlement_decision.py`

Focused existing specifications/reports:

- `reports/stage31c_coverage_entailment_head_report.md`
- `reports/stage176b_native_structural_separability_closure.md`
- `reports/stage177a_frame_head_hard_subset_closure.md`
- `reports/stage177b_frame_pairwise_feasibility_closure.md`
- `reports/stage177b_frame_pairwise_feasibility_closure.json`
- `reports/stage179a_mixed_insufficient_closure.md`
- `reports/stage183a_positive_preservation_design_20260715_131429/stage183a_positive_preservation_design_report.md`
- `reports/stage191b_deterministic_replay_implementation_spec.md`
- `reports/stage193a_tail3_fresh_seed_replication_spec.md`
- `reports/stage195a_tail3_parameter_swa_manifest_spec.md`
- `reports/stage195p0_tail3_parameter_swa_spec.md`

No broad unrelated history review was used.

## 2. Frame computation path

| Item | File and source anchor | Interpretation |
|---|---|---|
| FrameGate definition | `src/contramamba/heads/frame_gate.py:9-74`, `FrameGate` | Category-free pair-level frame head. |
| Input state | `frame_gate.py:37-47`, `FrameGate.forward` | Receives shared encoder `token_states` plus attention, claim, and evidence masks. |
| Claim/evidence separation | `frame_gate.py:44-49` | Masks are validated, then claim and evidence are pooled separately into `claim_state` and `evidence_state`. |
| Pairwise features | `frame_gate.py:50-59` | Concatenates claim state, evidence state, absolute difference, and elementwise product; this is explicitly pairwise. |
| `frame_logit` | `frame_gate.py:59-60` | `frame_classifier(pair_repr).squeeze(-1)`. |
| `frame_prob` | `frame_gate.py:67-73` | Exactly `torch.sigmoid(frame_logit)`. |
| Predicate dependency | `src/contramamba/heads/predicate_coverage.py:38-68` | PredicateCoverage separately pools claim/evidence predicate states and additionally consumes `frame_pair_repr` and `frame_prob`; the reserved claim/evidence frame states are currently deleted. |
| Sufficiency dependency | `src/contramamba/heads/sufficiency_gate.py:24-45` | Sufficiency consumes frame pair representation, predicate pair representation, frame probability, and predicate probability. |
| v6b forward wiring | `src/contramamba/modeling_v6b_minimal.py:299-327`, `ContraMambaV6BMinimal.forward` | Encoder states flow to FrameGate, then PredicateCoverage, then SufficiencyGate. No detach appears on this core path. |
| v6b entitlement/final composition | `modeling_v6b_minimal.py:430-439` and `src/contramamba/heads/entitlement_decision.py:36-76` | Default `explicit_product` computes `frame_prob * predicate_coverage_prob * sufficiency_prob`; it scales SUPPORT/REFUTE energy, while `1-entitlement` raises NOT_ENTITLED. |
| vNext wiring | `src/contramamba/modeling_vnext_minimal.py:281-347` | The same FrameGate/PredicateCoverage/Sufficiency chain is retained. `compositional_entitlement_prob` is the three-way product. Router modes may use product, min, learned×product, learned×frame×sufficiency, or omit frame in named ablations. Stage195 used v6b, not vNext. |
| Residual/router/gating status in Stage195 | `modeling_v6b_minimal.py:441-503`; Stage195 specs | v6b begins from product-composed base logits. Frozen Stage195 uses the temporal comparator but forbids temporal auxiliary adapters/channels, pair-contrastive auxiliary loss, and calibration routes. Comparator flags modulate final logits; they do not replace the native frame head or its BCE. |

### FrameGate parameters and trainability

From `FrameGate.__init__` (`frame_gate.py:21-34`), the native parameter prefixes are:

- `frame_gate.project.0.{weight,bias}`
- `frame_gate.project.2.{weight,bias}`
- `frame_gate.pair_projector.0.{weight,bias}`
- `frame_gate.pair_projector.3.{weight,bias}`
- `frame_gate.frame_classifier.{weight,bias}`
- optionally `frame_gate.token_diagnostic.{weight,bias}` when token diagnostics are enabled

No code freezes these head parameters. `scripts/train_controlled_v5.py:864-886` builds AdamW groups from every `requires_grad=True` non-encoder parameter, so the entire native FrameGate head is included. Stage195 additionally validates equality between unique trainable named parameters and optimizer-owned parameters (`reports/stage195p0_tail3_parameter_swa_spec.md`, “Exact parameter scope and optimizer ownership”).

The base encoder is controlled separately. The inherited parser default is `--freeze-encoder true` (`scripts/train_controlled_v5.py:1144`), and v6b construction sets Mamba parameters' `requires_grad` according to that flag (`scripts/train_controlled_v6b_minimal.py:2522-2574`). Thus, under a frozen encoder, encoder outputs still feed FrameGate, and gradients update FrameGate projection/pair/readout parameters, but they stop at frozen backbone parameters. This is not a detach of the FrameGate head.

### Gradient routes into FrameGate

1. Direct: `frame_loss -> frame_logit -> frame_classifier -> pair_projector -> project`.
2. Final-label indirect: final CE -> final logits -> entitlement product -> `frame_prob` -> FrameGate.
3. Downstream-channel indirect: PredicateCoverage and Sufficiency consume native frame tensors, so their direct losses also backpropagate through those dependencies.
4. Stage195 intervention arm only: compatible-positive hinge -> native `output["frame_logit"]` -> FrameGate.

There is no detach on routes 1-4. Detaches at `modeling_v6b_minimal.py:400-428` concern optional temporal adapter/channel inputs, not the native frame BCE path, and those auxiliary routes were forbidden/off in Stage195.

## 3. Actual frame target source

### Construction and transport

- Generator construction: `scripts/build_controlled_v5.py:353-377`, `_record`, writes an independent integer `frame_compatible_label` field.
- Family-specific assignments: `build_controlled_v5.py:388-455`, `_build_records`.
- On-disk raw field: every current controlled row carries `frame_compatible_label`; the first twelve rows of `data/controlled_v5_v3_without_time_swap.jsonl` demonstrate all current families and independent channel fields.
- Runtime tensorization: `scripts/train_controlled_v5.py:247-264`, `encode_label_tensors`, copies `record["frame_compatible_label"]` verbatim to `frame_compatible_labels`; no final-label, primary-failure, intervention, or heuristic reconstruction occurs.
- v6b loss consumption: `src/contramamba/modeling_v6b_minimal.py:505-550` and the active trainer path `scripts/train_controlled_v5.py:452-501` consume that tensor directly.

Classification of the actual source: **raw dataset field**, originally assigned by controlled-data construction. It is not an intervention-derived runtime flag, heuristic runtime target, final-label-derived target, primary-failure-derived target, or reconstructed target.

### Intervention target table

| intervention type | expected frame alignment | actual frame target | target source | masked? | notes |
|---|---:|---:|---|---:|---|
| `none` | 1 | 1 | raw `frame_compatible_label` | no | Ordinary SUPPORT and REFUTE originals retain the same referential frame. |
| `paraphrase` | 1 | 1 | raw field | no | Surface variation remains same-frame. |
| `polarity_flip` | 1 | 1 | raw field | no | Polarity changes while entity/event/role/time/location remain fixed. |
| `entity_swap` | 0 | 0 | raw field | no | Referential entity changes. |
| `event_swap` | 0 | 0 | raw field | no | Generator changes the controlled object/event referent. |
| `location_swap` | 0 | 0 | raw field | no | Frame-critical location changes. |
| `role_swap` | 0 | 0 | raw field | no | Role/context changes. |
| `title_name_swap` | 0 | 0 | raw field | no | Referential identity changes. |
| `irrelevant_evidence` | 0 | 0 | raw field | no | Evidence describes a different referential situation; predicate and sufficiency are also zero, but frame zero is independently appropriate. |
| `predicate_swap` | 1 | 1 | raw field | no | Predicate noncoverage is kept out of the frame-negative class. |
| `evidence_deletion` | 1 | 1 | raw field | no | Sufficiency is zero; frame remains positive by the frozen channel partition. |
| `evidence_truncation` | 1 | 1 | raw field | no | Sufficiency is zero; frame remains positive. |
| `time_swap` | 0 in generator | absent from main dataset | generator field; not tensorized in Stage195 main training | n/a | `build_controlled_v5.py:408-412` makes frame-critical time swap negative. `controlled_v5_v3_without_time_swap.jsonl` contains no such family; Stage195 contracts explicitly deny time-swap main data. |
| number/date/value mismatch | expected 1 when referential frame is preserved | `not_statically_determined` | no such current intervention family/row | n/a | The current 12-family JSONL has no `number_swap`, `date_swap`, or `value_swap`. No target is inferred. Polarity flip is the available same-frame value/polarity analogue and is positive. |

The current JSONL family set is exactly the twelve rows shown above excluding `time_swap` and grouping the six frame negatives explicitly; no value/number/date family was found by static text inspection.

## 4. Direct loss and gradient path

### Native loss contract

| Property | Finding | Exact evidence |
|---|---|---|
| Direct FrameGate loss | present | `modeling_v6b_minimal.py:515-518`; active trainer recomputation at `train_controlled_v5.py:465-468` |
| Type | BCE with logits | `F.binary_cross_entropy_with_logits(output["frame_logit"], inputs["frame_compatible_labels"])` |
| Target tensor | `frame_compatible_labels` | copied verbatim from raw field at `train_controlled_v5.py:252-254` |
| Mask | none | no indexing/mask in native frame BCE |
| Reduction | PyTorch default mean over all rows | no `reduction` argument |
| Class/instance weight | none | no `weight` or `pos_weight` argument |
| Native weight | implicit 1.0 | `train_controlled_v5.py:493` sums label + frame + predicate + sufficiency + polarity without scalar |
| Default CLI value | not applicable/hard-wired active | there is no native frame-BCE enable/weight CLI; labels are present for every main row |
| Sum location | `controlled_losses`, then trainer `total_loss` | `train_controlled_v5.py:493`; `train_controlled_v6b_minimal.py:17614-17661` |
| Backward | active | `train_controlled_v6b_minimal.py:18502-18514` |
| Metric/logging | present, but not a standalone exported weight knob | `train_controlled_v5.py:504-554` reports frame accuracy; `format_epoch` prints `losses['frame']` and frame accuracy; prediction/scalar exports carry native frame logit/prob. |

Structure verdict: **direct and frame-local supervision**. It is neither no-supervision nor indirect-only. There are additional indirect final/downstream gradients, and the Stage195 intervention arm adds a positive-only hinge, but those do not replace or contaminate the native BCE.

### Actual Stage195 activation

- All six Stage195 runs use `v6b_minimal`, Mamba, the controlled JSONL without time swap, and 20 epochs (`reports/stage195a_tail3_parameter_swa_manifest_spec.md:71-89`; `reports/stage195p0_tail3_parameter_swa_spec.md:47-70`).
- Native direct frame BCE is unconditional in the active `v5.controlled_losses` call for every run (`train_controlled_v6b_minimal.py:17611-17628`).
- Baseline arm: compatible-positive margin weight `0.0`; native BCE remains active.
- Intervention arm: native BCE remains active and adds the fixed `0.05 * mean(ReLU(0-frame_logit))` over the authoritative eligible positive mask (`train_controlled_v6b_minimal.py:2006-2044`, `17677-17718`). It changes gradient pressure, not the raw target.
- Stage177-C frame pairwise mode, pair-contrastive auxiliary data/loss, temporal auxiliary routes, and coverage-entailment auxiliary training are off/forbidden in the Stage195 envelope.

## 5. Positive/negative target audit

### Positive-frame correctness

- **none:** yes. Both base SUPPORT and base REFUTE are assigned frame target 1 (`build_controlled_v5.py:383-396`).
- **paraphrase:** yes. Lexical/syntactic rewording is assigned 1 (`build_controlled_v5.py:394-397`).
- **polarity_flip:** yes. Flipped SUPPORT/REFUTE final labels retain frame 1 (`build_controlled_v5.py:452-455`).
- **predicate noncoverage:** yes. `predicate_swap` has frame 1 and predicate 0 (`build_controlled_v5.py:432-436`).
- **insufficiency:** yes. `evidence_deletion` and `evidence_truncation` have frame 1 and sufficiency 0 (`build_controlled_v5.py:437-446`).
- **final NOT_ENTITLED:** no alias is present. Predicate-swap and insufficiency rows are NOT_ENTITLED while frame remains 1; frame-swap rows are NOT_ENTITLED with frame 0; ordinary REFUTE rows can have frame 1. This disproves final-label-derived frame targets.

### Negative-frame correctness

The six current negative families are entity, event/object, location, role, title/name, and irrelevant-evidence mismatch. They supply true referential-situation negatives. `time_swap` is a generator-level negative but is intentionally absent from main classification training.

No predicate-only, sufficiency-only, or polarity/value-only row is labeled frame-negative in the current contract.

## 6. Sampling and imbalance

- Existing audited topology is 300 pairs, with six compatible and six incompatible rows per pair; train has 1,440 compatible and 1,440 incompatible rows (`stage183a_positive_preservation_design_report.md`; `stage177b_frame_pairwise_feasibility_closure.md`). There is no native class-imbalance basis for positive reweighting.
- Native frame BCE uses the full differentiable clean-train `frame_logit` tensor and the full aligned frame-target tensor. `sample_indices` affects only final-label CE indexing (`train_controlled_v5.py:452-468`), not frame BCE.
- Positive FrameGate rows are not masked from native BCE.
- Native BCE has no class or instance weights.
- The final NOT_ENTITLED prior can send indirect CE gradients through the entitlement product, but it does not define or mask the direct frame target. Its relative gradient dominance is `not_statically_determined` without runtime gradient measurements.
- Stage195 arm status changes only an additional positive-margin gradient: baseline weight 0; intervention weight 0.05 on the integrity-sidecar eligible positives. It does not change any `frame_compatible_label`.
- Exact per-family gradient magnitudes, batchwise gradient conflicts, and the persistent-row membership of training families are `not_statically_determined` by source contract alone.

## 7. Channel-separation violations

| Candidate violation | Detected? | Evidence |
|---|---:|---|
| predicate swap -> frame target 0 | no | predicate swap is frame 1, predicate 0 |
| insufficiency -> frame target 0 | no | deletion/truncation are frame 1, sufficiency 0 |
| final NOT_ENTITLED -> frame target 0 | no | multiple NOT_ENTITLED counterexamples have frame 1 |
| polarity/value mismatch -> frame target 0 | no for represented polarity flip | polarity flip is frame 1; other value families absent |
| frame target aliases entitlement/final target | no | independent counterexamples in raw rows |
| frame target aliases predicate target | no | location/role frame 0 predicate 1; predicate swap frame 1 predicate 0 |
| frame loss uses final classifier tensor | no | native `frame_logit` is used, not `output["logits"]` |
| FrameGate excluded from optimizer | no | all trainable non-encoder head parameters enter AdamW; Stage195 ownership validation is exhaustive |
| detach blocks direct frame-loss gradient | no | no detach between native BCE and FrameGate |
| frame probability diagnostic-only | no | it participates in PredicateCoverage, Sufficiency, entitlement product, final logits, and direct BCE via its logit |

Detected contamination types: none.

One non-violation caveat: entity/event/title-name frame-negative rows also have predicate target 0, and irrelevant-evidence has all three local targets 0. These are multi-channel semantic cases, not evidence that predicate failure is being aliased into frame failure; the controlled construction independently sets the frame field, and predicate-only/sufficiency-only counterexamples preserve frame target 1.

## 8. Decision

Evaluation in the required order:

1. **Target contamination:** false.
2. **No direct supervision:** false; direct BCE is always active on main controlled rows.
3. **Positive underweighting or masking:** false under the static contract; positives are unmasked, the topology is 1:1, and native weight is 1.0. The intervention arm adds rather than removes positive pressure.
4. **Correct supervision / representation failure:** true under the required taxonomy. Target, loss, mask, optimizer ownership, and direct gradient path are correct and active, yet Stage196-A freezes recurrent FrameGate failures.

Selected decision: `STAGE196B0_FRAME_REPRESENTATION_FAILURE`.

Frozen Stage196-A evidence carried forward without recomputation: decision `STAGE196A_RECURRENT_LOCAL_CHANNEL_FAILURE`; 127 persistent rows; 72 multi-local and 55 frame-only; zero predicate-only, sufficiency-only, entitlement-aggregation, or final-composition failures; all-local-pass share 0.0; every persistent row includes FrameGate failure; recurrent positions baseline/intervention 22/19; universal 15/12; recurrent in both arms 19; intervention-only recurrent 0; ten positions persistent in all six runs; persistent row-run intervention types none 65, paraphrase 6, polarity_flip 56.

## 9. Recommended single Stage196-B1 intervention

`ADD_FRAME_STATE_POSITIVE_NEGATIVE_CONTRAST`

Rationale: an explicit frame-local BCE already exists and its partition is correct; restoring a mask/weight is unsupported because native positives are unmasked and balanced. The next taxonomy-consistent family is therefore a contrast on frame states/representations. This recommendation is a family selection only: it does not authorize implementation, choose pairs, define a margin/weight, add multiple losses, change the entitlement product, or select a threshold.

## 10. Unified diff summary

Exactly two new report artifacts are added. No existing source, data, specification, report, model, trainer, or configuration file is modified.

## 11. Static review performed

- Traced FrameGate inputs, state separation, pair features, logit/probability, product/router consumers, parameters, optimizer inclusion, and gradient paths.
- Traced generator assignment -> raw JSONL field -> tensorization -> direct BCE.
- Audited all twelve current intervention families plus generator-only `time_swap`; recorded absent value/number/date families as not statically determined.
- Compared v6b and current vNext construction/loss paths.
- Checked Stage195 baseline/intervention loss-envelope differences.
- Checked Stage196-A frozen localization definitions and supplied findings without running the analyzer.
- Reviewed the final diff/status only with read-only Git commands.

## 12. Validation not executed

Not run: Python, `py_compile`, any analyzer, JSON parser, model import/forward, training, smoke/mini/full run, Kaggle/Lightning command, threshold search, calibration, loss-weight sweep, checkpoint load, or external/OOD evaluation.

JSON validity is reviewed textually and with non-Python read-only inspection only; no executable JSON validation was authorized.

## 13. Remaining risks

- Static code establishes supervision availability and wiring, not realized gradient magnitudes or causal dominance among direct BCE, downstream local losses, and final CE.
- The encoder freeze state constrains learnable representation changes to the head-side projection/pair/readout unless a specific historical argv proves otherwise; the Stage195 specifications preserve the runtime envelope but do not print every argv token in the inspected repository files.
- Existing integrity work records known generator defects for some polarity-flip rows; the Stage195 intervention sidecar masks its added margin accordingly, but native BCE still trains on all raw rows. This does not expose a frame-target partition violation in the inspected contract, yet row-level semantic cleanliness beyond the frozen reports is not re-adjudicated here.
- Value/number/date mismatch semantics cannot be audited against actual targets because those intervention families are absent.
- `STAGE196B0_FRAME_REPRESENTATION_FAILURE` is the required taxonomy result, not proof of global representation collapse, a unique encoder defect, or external generalization.

