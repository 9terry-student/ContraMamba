# ContraMamba Gen4 Study C — Intervention Manifold-Deviation Prospective Capture and Audit Design

Status: PROSPECTIVE_DESIGN_HASH_FREEZE
Execution authorized by this file: NO
Training authorized: NO
Current Study-A outcome required to define this design: NO
Current Study-B outcome required to define this design: NO

## 1. Role and independence

Study C is a reviewer-defense / intervention-faithfulness audit.

It does not reopen or rescue any causal endpoint.

Study C outcomes may not change:
- Study A site, transport basis, population, endpoint, or test;
- Study B scale-local planes, controls, population, endpoint, or test;
- any frozen intervention magnitude or causal selection.

Study A/B outcomes may affect only later synthesis language, not this design.

## 2. Static-reuse audit result

Existing frozen Experiment-1 behavioral rows preserve:
- row identity;
- scale;
- selected/control plane identity;
- native selected coefficients;
- selected-component norm;
- correction norm;
- applied-correction residual diagnostics;
- downstream behavioral outputs.

They do not preserve the full native content-half intervention activation,
the native strong-coordinate activation norm, or native/intervened activation
vectors required for the frozen manifold metrics.

Therefore the following cannot be recovered compatibly from existing frozen rows:
- R_rel denominator ||h_native|| in full content-half coordinates;
- R_rel denominator ||h_native,strong|| in strong coordinates;
- d_native nearest-native distance;
- d_int nearest-native distance after intervention;
- R_NN.

The static-reuse branch therefore closes as:

STATIC_REUSE_INSUFFICIENT_FOR_PRIMARY_MANIFOLD_METRICS

No missing quantity may be reconstructed from a different representation or from
downstream logits.

A bounded new state capture is required.

## 3. Audit scope

The primary Study-C audit targets the main cross-scale Experiment-1 behavioral
intervention because it is the principal frozen behavioral causal bridge and already
provides matched 370M/1.4B intervention semantics.

Population:
- xg1_fact_4801..xg1_fact_5100
- N = 300 matched source pairs

Rows:
- C0_SHAM
- C2_NAME

Scales:
- Mamba-370M
- Mamba-1.4B

Frozen scale-local objects:

Mamba-370M:
- selected P3
- response-blind control P5

Mamba-1.4B:
- selected P5
- response-blind control P4

For both:
- source block 33
- target residual layer 34
- intervention layer 35
- anchor A_IDENTITY
- target offset +2
- mixer.in_proj content-half intervention semantics
- frozen strong mask

No new population, plane, token, layer, coefficient, or intervention magnitude is
selected from Study-C observations.

## 4. Conditions audited

Primary descriptive intervention condition:
- dominant_control

Secondary descriptive intervention condition:
- dominant_neutralized

Sanity condition:
- dominant_restored

Native is the baseline reference.

For each native row with strong-coordinate activation h:

selected component:
C_sel(h) = a u_sel,+ + b u_sel,-

where:
a = h^T u_sel,+
b = h^T u_sel,-

dominant_neutralized correction:
delta_neutralized = -C_sel(h)

coefficient-matched control component:
C_ctrl(h) = a u_ctrl,+ + b u_ctrl,-

dominant_control correction:
delta_control = -C_sel(h) + C_ctrl(h)

dominant_restored correction:
delta_restored = 0

These are the exact frozen Experiment-1 correction semantics.
No correction is fitted from Study-C data.

## 5. Capture can piggyback on Study B

Study B already requires the exact native block-35 target-token content-half activation
for each of the same:
- two scales;
- 300 source pairs;
- two cells.

Therefore Study C capture SHOULD be emitted as a separate artifact stream during the
same Study-B native forward/backward execution.

This does not change Study-B scientific computation:
- no extra model forward;
- no extra backward;
- no Study-C metric is used by Study B;
- no Study-C summary or conclusion is inspected during Study-B execution.

Study-B scientific budget remains:
- 1200 native full-model forwards;
- 1200 local backward evaluations;
- zero training steps;
- zero parameter updates.

Study-C piggyback adds serialization and deterministic local correction construction
only.

If implementation coupling would compromise either artifact boundary, Study C must
instead use a separate capture run with the same 1200 native forwards and zero
training. Scientific definitions remain unchanged.

## 6. Raw capture object

For every scale x pair x cell row, capture at the exact intervention coordinate:

1. h_native_full
   - full content-half activation before any correction
   - stored in the actual model activation dtype/bytes used at the hook boundary

2. h_native_strong
   - exact gather of h_native_full by the frozen scale-local strong mask

3. delta_neutralized_applied
   - correction scattered into full content-half coordinates
   - cast exactly as it would be applied by the frozen Experiment-1 hook

4. delta_control_applied
   - correction scattered into full content-half coordinates
   - cast exactly as it would be applied by the frozen Experiment-1 hook

5. delta_restored_applied
   - exact zero vector, represented algebraically rather than redundantly serialized

Also record:
- source_pair_id;
- contrast_cell_id;
- scale;
- checkpoint SHA256;
- selected/control plane identity;
- absolute anchor and target-token index;
- strong-mask identity/count;
- activation dtype;
- correction-construction residual diagnostics;
- boundary flags showing no response-guided selection and no parameter update.

Full vectors should be stored in a compact array container rather than expanded JSON.

## 7. Raw Study-C capture artifacts

The piggyback capture is a separate sibling artifact set from Study B.

Freeze exactly:

1. manifold_capture_states.npz
2. manifold_capture_index.jsonl
3. artifact_manifest.json
4. SHA256SUMS.txt

The artifact directory must be distinct from Study-B readout-alignment artifacts.

The manifest pins:
- execution HEAD;
- both checkpoint hashes;
- population;
- cell order;
- scale order;
- frozen selected/control identities;
- intervention coordinate;
- vector dtype/shape;
- exact row count;
- zero additional model-forward count attributable to piggyback capture;
- no Study-C scientific conclusion computed during raw execution.

## 8. Frozen displacement metric

For each row and audited condition c:

h_int,c = h_native + delta_c

Full content-half displacement:

R_rel_full,c =
    ||delta_c,full||_2
    /
    max(||h_native,full||_2, 1e-12)

Strong-coordinate displacement:

R_rel_strong,c =
    ||delta_c,strong||_2
    /
    max(||h_native,strong||_2, 1e-12)

Both metrics are primary descriptive outputs.

No threshold is prospectively defined for "safe", "unsafe", "on-manifold", or
"off-manifold".

dominant_restored must satisfy R_rel = 0 up to exact representation semantics and is
used only as a sanity check.

## 9. Frozen nearest-native reference set

Nearest-native comparisons are performed separately within each:

scale x contrast_cell_id

stratum.

Thus each row has a native reference set of the other 299 source pairs with:
- same model scale;
- same C0_SHAM or C2_NAME semantic cell;
- same intervention coordinate.

The same row is always excluded.

This prevents scale or cell identity from trivially determining nearest-neighbor
distance.

No label/correctness/sign-based filtering is allowed.

## 10. Frozen nearest-native metrics

For row q in a fixed scale x cell stratum:

d_native_full(q) =
    min_{r != q} ||h_native_full(q) - h_native_full(r)||_2

d_int_full,c(q) =
    min_{r != q} ||h_int_full,c(q) - h_native_full(r)||_2

R_NN_full,c(q) =
    d_int_full,c(q) / max(d_native_full(q), 1e-12)

Analogously in frozen strong coordinates:

d_native_strong(q)
d_int_strong,c(q)
R_NN_strong,c(q)

The reference set is native-only.
Intervened states are never added to H_native.

## 11. Descriptive summaries

For each scale x cell x audited condition, report for:
- R_rel_full
- R_rel_strong
- R_NN_full
- R_NN_strong

the:
- mean;
- population SD;
- min;
- q25;
- median;
- q75;
- max.

Also report:
- fraction R_NN > 1;
- fraction R_NN > 2;
- nearest-neighbor identity change rate between native and intervened query;
- zero/near-zero native norm counts;
- nonfinite counts;
- restored-condition sanity failures, if any.

The R_NN > 1 and > 2 fractions are descriptive landmarks only, not inferential or
safety thresholds.

## 12. Cross-scale comparison boundary

Mamba-370M and Mamba-1.4B summaries may be placed side by side.

No primary p-value is added.

No statement that one scale is "more faithful", "safer", or "more on-manifold" is
authorized solely from these geometric metrics.

If a numerical cross-scale difference is reported, it is descriptive.

## 13. No Mahalanobis / covariance rescue

No covariance-regularized, Mahalanobis-like, PCA-selected, learned-density, or
representation-classifier metric is primary.

No dimensionality-reduction or regularization search is allowed after observing
Study-C values.

If future work adds such a metric, its estimator and hyperparameters require a new
prospective design before outcomes are inspected.

## 14. Computation phase boundary

Raw capture phase:
- may access frozen structural rows, checkpoints, geometry, strong masks and planes;
- may not read Study-A response outcomes;
- may not read Study-B readout-alignment results;
- computes no Study-C manifold summary.

Static CPU audit phase begins only after:
- Study-B raw scientific artifacts are frozen;
- Study-C raw capture artifacts are frozen;
- provenance/hash validation passes.

The CPU audit then computes the frozen Study-C metrics from the raw capture only.

## 15. Interpretation boundary

Study C may support statements such as:
- intervention displacement is small/large relative to native activation scale under
  the explicitly defined R_rel metric;
- intervened states lie nearer/farther from the frozen local native neighborhood under
  the explicitly defined R_NN metric;
- these descriptive quantities differ across frozen conditions, cells or scales.

Study C may not, from geometry alone, claim:
- that an intervention is causally invalid;
- that hidden pathways were activated or not activated;
- that a representation is literally on/off the model's manifold;
- that a scale is globally more mechanistically faithful;
- that Study A or B should be retuned.

## 16. No-rescue rules

After any Study-C state or metric is observed, do not:
- change native reference strata;
- remove rows by prediction correctness, margin, norm, distance or sign;
- change distance metric;
- change coordinate set;
- add tuned normalization;
- change intervention condition;
- select a different population;
- add a learned density estimator;
- retune Study A or B.

If the raw capture is technically invalid, fix only the technical capture while
preserving this design.
