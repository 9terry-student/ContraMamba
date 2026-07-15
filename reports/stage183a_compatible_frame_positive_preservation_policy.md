# Stage183-A compatible-frame positive-preservation design audit

## Decision

`STAGE183A_CONTROLLED_TRAIN_INTEGRITY_MASK_REQUIRED_FIRST`

Authorized next route:

`STAGE184_CONTROLLED_TRAIN_INTEGRITY_MASK_SPEC`

Stage183-A is specification/report-only. No positive-preservation loss, checkpoint selector, or training pilot is authorized. The compatible-positive absolute-margin hinge is the best scientific match among the audited interventions, but it is only a contingent post-integrity candidate: the authoritative main JSONL does not carry enough row-level metadata to construct the required contamination-safe eligibility mask.

## Scope and evidence boundary

The audit is grounded in static inspection of:

- `scripts/train_controlled_v6b_minimal.py`
- `scripts/train_controlled_v5.py`
- `src/contramamba/modeling_v6b_minimal.py`
- `src/contramamba/heads/frame_gate.py`
- `scripts/stage175b_support_anchor.py`
- `scripts/stage177c_frame_pairwise.py`
- Stage174/175/177/182 closure artifacts
- `data/controlled_v5_v3_without_time_swap.jsonl`

No model, Torch operation, checkpoint, forward pass, training loop, smoke test, compile, threshold fit, calibration, or external evaluation is part of this decision.

## Stage182-B closure carried forward

The official Stage182-B decision is `STAGE182B_COMPATIBLE_POSITIVE_MARGIN_COLLAPSE_SIGNAL`. Thirteen of fourteen clean failures are compatible false negatives; one is an incompatible false positive. The candidate-minus-matched-control frame-logit median is `-0.555523656308651`, its fixed-seed bootstrap 95% CI is `[-0.7878567576408386, -0.3871966600418091]`, and the exact sign-test p is `0.0018310546875`.

Candidate and control centroid-correct rates are both `1/14`; neither a representation-mislocalization gate nor a readout-alignment gate passed. Bias-specific decomposition is unavailable. Representation movement has magnitude but no authorized direction. Thus the result motivates an absolute compatible-positive target, not a bias constraint, family rule, calibration claim, or polarity-causal remedy.

## Current native frame objective

For row `i`, let `z_i` be `output["frame_logit"]` and `y_i` be `frame_compatible_label`.

The trainer calls `scripts.train_controlled_v5.py::controlled_losses`, whose native frame term is

`L_frame = mean_i BCEWithLogits(z_i, y_i)`.

Equivalently,

`BCEWithLogits(z,y) = max(z,0) - z*y + log(1 + exp(-abs(z)))`.

Static facts:

- The formulation is `torch.nn.functional.binary_cross_entropy_with_logits`, not a separately instantiated module.
- There is no native-frame `pos_weight` or per-class weight.
- Reduction is the PyTorch default row mean over the complete train tensor, not a pair mean.
- `controlled_losses` sums final-label CE, frame BCE, predicate BCE, sufficiency BCE, and entitled-row polarity CE with implicit weight `1.0` each; the native frame weight is therefore `1.0`.
- Final-label CE may use sampling or CE class weights, but those do not reweight native frame BCE.
- `frame_classifier` is `nn.Linear(frame_size, 1)` with its default trainable bias.
- `frame_prob` is exactly `sigmoid(frame_logit)`.
- `FrameGate` forms `frame_pair_repr` from projected claim/evidence states, their absolute difference, and elementwise product.
- The final classifier consumes frame/predicate/sufficiency representations and probabilities, so final CE and native frame BCE can send different gradients into shared upstream representations. This is a structural conflict opportunity, not evidence that conflict caused Stage182-B.
- Polarity outputs are computed from the shared token states, while optional entitlement paths consume frame-derived signals; they do not redefine the native frame target.
- Default checkpoint selection maximizes clean-dev `final_macro_f1` (or optionally final accuracy). Native frame-positive recall and compatible-positive margin are not direct selection terms.

## Train label topology

The authoritative main dataset has 300 pair groups and 3,600 rows, with 12 rows per pair. The frozen Stage177 topology reports six compatible and six incompatible rows per pair. Under the Stage177 seed-174, 80/20 pair split this yields 240 train pairs / 2,880 train rows and 60 dev pairs / 720 dev rows; train labels are 1,440 compatible and 1,440 incompatible. There is no class imbalance in that frozen topology, so global positive reweighting has no imbalance-based justification.

The audit script recomputes these counts from row labels and reproduces the deterministic pair split. It never infers frame targets from family names.

## Shared gradient paths

`FrameGate.project` and `FrameGate.pair_projector` receive direct native-frame BCE gradients. The same frame representation and probability feed the final decision path, so final CE can also update these modules. The encoder/projected token states additionally support predicate, sufficiency, and polarity objectives. The frame classifier weight and bias receive direct frame-head gradients; final CE can reach them through `frame_prob` where the decision path consumes it.

This topology permits objective interaction but does not identify the sign or dominance of any interaction. Stage182-B's failed localization gates prohibit converting this static possibility into a causal claim.

## Contamination-safe eligibility boundary

A future compatible-positive preservation row must satisfy all of the following:

1. `frame_compatible_label == 1`.
2. Grammar validity is authoritatively resolved and true.
3. The intervention contract is authoritatively resolved and exact.
4. Polarity contamination is authoritatively resolved and absent.
5. Schema is resolved.
6. The canonical row is authoritatively resolved and valid.
7. The row is not `time_swap`.
8. The row is from the authoritative main controlled dataset, not an external dataset.
9. The row is not a Stage34/35 synthetic external-validation substitute.

The main JSONL supplies standard labels, IDs, texts, failure type, and intervention type. It does not supply authoritative row-level grammar validity, exact-contract status, polarity-contamination status, or canonical-valid status. Stage182-A proves that generator equality is not cleanliness and documents 22 deterministic contaminated review items (21 non-polarity polarity changes and one invalid do-support construction). Those review findings cannot safely be extrapolated into a complete train-row mask.

Therefore a full contamination-safe train eligibility mask is not deterministically constructible from current authoritative training metadata. Treating missing integrity fields as true is forbidden. Stage184 must specify authoritative provenance, complete coverage, fail-closed behavior, and train/dev isolation before a hinge implementation smoke can be authorized.

## Candidate comparison

### A — global positive class reweighting

`BCEWithLogits(pos_weight=alpha)` amplifies every compatible row. The frozen train topology is balanced, so imbalance does not motivate `alpha > 1`. It does not enforce an absolute margin and can globally increase false positives. A fixed nontrivial value would be arbitrary without tuning. Not selected.

### B — compatible-positive absolute-margin hinge

`L_pos = mean_{i in E} relu(m - z_i)`, where `E` is the clean-compatible eligibility set. This directly targets the diagnosed failure, leaves negative rows without a direct new term, needs neither teacher nor counterpart forward, and differs from Stage177 because it enforces an absolute positive floor rather than compatible-over-incompatible ordering. It also differs from Stage175 because it targets the native frame logit instead of a final SUPPORT margin relative to a detached canonical row.

This is the contingent preferred intervention. It is not spec-ready now because `E` is unavailable and Stage183-A has no evidence-based fixed nonzero `m` or weight. Default-off (`mode=off`, weight `0`) is required if a later implementation is authorized.

### C — detached teacher/reference preservation

A one-sided teacher hinge could preserve positive logits relative to a baseline, but it needs a teacher checkpoint or counterpart forward, adds provenance and compute costs, can preserve teacher errors, and resembles the reference-preservation family already explored by Stage175. It would reopen checkpoint/reference questions without evidence that an absolute teacher value is correct. Not selected.

### D — frame-classifier bias constraint

Stage182-B could not perform bias-specific decomposition and did not pass a readout gate. A bias constraint would behave like a global calibration/threshold intervention and risks negative-class degradation. Not scientifically justified.

### E — false-negative-aware checkpoint selection

This changes selection rather than the training objective and is operationally simple. However, clean-dev positive recall alone can trade away incompatible precision, and the Stage182 hard cohort must not become a selection set. A predeclared clean-dev metric could be designed later, but it is less direct than a train-time absolute positive target and still needs a leakage-safe selection contract. Not selected.

### F — family-conditioned weighting

The observed 14-row family mix is a selected hard subset, not a causal estimate. Family-specific weights risk shortcut learning and overfitting, particularly for `none` and `polarity_flip`. No polarity-causal evidence exists. Rejected.

## Gradient-level comparison for a compatible positive row

Let `sigma(z)` be the logistic sigmoid, `alpha` the positive weight, `m` the target margin, and `r` a detached reference logit with tolerance `tau`.

| Loss | Activation | `dL/dz` for one positive row | Near `z=0` | Strongly negative `z` | Already positive `z` | Absolute margin | Negative rows | Reference |
|---|---|---|---|---|---|---|---|---|
| Native BCE | every positive row | `sigma(z)-1` | `-0.5` | approaches `-1` | negative, approaches `0` | No | Yes, via `sigma(z)` | No |
| Positive reweighted BCE | every positive row | `alpha*(sigma(z)-1)` | `-alpha/2` | approaches `-alpha` | negative, approaches `0` | No | Native negative BCE remains; no positive multiplier | No |
| Positive absolute-margin hinge | eligible positive and `z < m` | `-1` (subgradient `0` chosen when inactive) | `-1` if `m>0` | `-1` | `0` once `z>=m` | Yes | No direct new term | No |
| Teacher/reference hinge | eligible positive and `z < r-tau` | `-1` | condition-dependent | `-1` if active | condition-dependent | Relative, not fixed absolute | No direct new term | Yes |

These are per-row derivatives before outer loss weights and eligible-row averaging.

## Redundancy audit

- Native BCE: already supplies positive gradients, but its gradient decays continuously and it does not guarantee `z >= m`. A hinge overlaps in gradient direction while adding a nonredundant absolute-floor target.
- Stage177 pairwise softplus: enforces compatible `>` incompatible within each pair, has no absolute margin, and changed representation/ranking without changing final predictions in its pilot. An absolute positive hinge targets a different failure mode.
- Stage175 support anchor: preserves a final SUPPORT margin relative to a detached same-pair canonical reference. It has a different head and target, requires a reference forward, and showed no clean benefit. Candidate C overlaps with this preservation family; Candidate B does not require a teacher/reference.

## Risk matrix summary

Ordinal ratings are evidence-based, not fitted scores.

| Candidate | Scientific fit | Redundancy | Contamination sensitivity | False-positive risk | Calibration-like risk | Complexity | Checkpoint dependency | Leakage risk | Cost | Interpretability | Fixed-value feasibility |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A reweighting | Medium | High | Medium | High | High | Low | Low | Low | Low | High | Low |
| B absolute hinge | High | Medium | High | Medium | Medium | Low | Low | Low | Low | High | Medium only after mask/default-off smoke |
| C teacher hinge | Medium | Medium/High | High | Medium | Medium | High | High | Medium | High | Medium | Low |
| D bias constraint | Low | Medium | Low | High | High | Medium | Low | Low | Low | Medium | Low |
| E checkpoint recall | Medium | Low | Medium | High | Medium | Medium | High | High | Low | High | Medium |
| F family weighting | Low | Medium | High | High | Medium | Medium | Low | Medium | Low | Medium | Low |

The machine-readable report expands this matrix across every required risk dimension.

## Stage184 gate and hyperparameter policy

The only authorized Stage184 route is `STAGE184_CONTROLLED_TRAIN_INTEGRITY_MASK_SPEC`.

That stage must define:

- authoritative row-level integrity fields or an immutable joined artifact;
- exact row identity and provenance;
- complete train coverage with no silent missing values;
- fail-closed semantics;
- explicit exclusion of `time_swap`, external data, and Stage34/35 substitutes;
- train-only construction independent of dev metrics and the Stage182 hard cohort;
- zero eligible rows producing an exact differentiable zero if a later loss is implemented.

No margin or nonzero weight is selected here. A later implementation, if authorized after the integrity gate, must be default off, weight `0` by default, use the current native `frame_logit`, use no counterpart or teacher, leave `output["logits"]` CE unchanged, average only over eligible rows, and include finite-gradient checks. Sweep, multi-seed training, full training, and pilot claims remain forbidden until separately authorized.

## Limitations

- Static source inspection establishes code topology, not runtime gradient dominance.
- Stage182-B is a 14-pair frozen-cohort signal, not a population causal estimate.
- No bias-specific mechanism or polarity-conditioned mechanism is established.
- The current main JSONL cannot prove grammar or semantic intervention integrity for every train row.
- No fixed positive margin or nonzero loss weight is evidence-justified by this audit.
