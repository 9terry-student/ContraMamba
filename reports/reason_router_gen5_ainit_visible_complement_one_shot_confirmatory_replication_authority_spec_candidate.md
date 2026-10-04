# Gen5 A-init Visible-vs-Complement One-Shot Confirmatory Replication Authority

SOURCE_CAUSAL_EVIDENCE_FREEZE_COMMIT=5694f962855bd2ab4f4035feb15cf1f4bfb3f784
SOURCE_PROJECTION_CLUSTERING_EVIDENCE_FREEZE_COMMIT=a82ca54f331274c7037368f2c458bcd5dedbeae9
SOURCE_FORWARD_JACOBIAN_RECOVERY_EVIDENCE_FREEZE_COMMIT=a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e

STATUS=READY_FOR_ONE_SHOT_CONFIRMATORY_REPLICATION

TRAINING_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
PARAMETER_GRADIENT_UPDATE_ALLOWED=NO
ANALYSIS_AUTOGRAD_ALLOWED=NO
BACKWARD_METHOD_ALLOWED=NO
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=YES_ONE_SHOT_THIS_AUTHORITY_ONLY
CUDA_EVALUATION_ALLOWED=YES_FROZEN_CONFIRMATORY_9601_9900_ONLY

## Purpose

The frozen dev evidence established a strong layer-22 causal decomposition:
large A-init-specific hidden-state differences are dominated by a component
that is effectively null / very low gain downstream, while a much smaller
task-visible component carries essentially all measurable endpoint logit and
margin difference.

The remaining high-value scientific question is whether that decomposition
replicates on a population that was not used to construct the task-sensitive
basis or to select the primary compact dimension.

This authority consumes the previously unopened xg1_fact_9601..xg1_fact_9900
population exactly once for this preregistered confirmatory question.

No result-dependent redefinition of the basis, k values, pair classes,
intervention, or pass/fail thresholds is allowed after confirmatory execution.

## Why the previous causal intervention remains valid

The source dev intervention is not invalidated by this confirmatory stage.

It directly tested finite layer-22 visible-only and complement-only endpoint
interventions and established causal downstream effects under the frozen dev
contract.

The confirmatory stage addresses a different limitation: whether the
dev-derived decomposition generalizes to previously unseen examples.

The earlier proposal to localize an upstream network-layer precursor is not
part of this authority because the Gen5 parent Mamba is frozen and the only
trainable A-init-dependent tensors are the layer-22 correction A_theta and
B_theta parameters. An apparent first emergence at layer 22 would therefore
be structurally tautological rather than a new scientific result.

## Frozen confirmatory population

Use only:

`data/reason_router_gen5_phase3_xg1_ownership_interaction_assay_v1/`

Population identity:

- pair range: `xg1_fact_9601..xg1_fact_9900`
- source pair count: `300`
- rows per pair: `6`
- total rows: `1800`
- labels present: false
- generator family: `xg1_independent_structured_records_v1`

Frozen source identities from the previously frozen Phase3 static preparation:

- bundle checksums SHA256:
  `c77154aa8aaff0818ccb3825b43f7f6f39fd924b78099c9c54c086203f92a8aa`
- structural manifest SHA256:
  `d52918fc6c6090e2cccca6ffaeabb63dae9cd7cf0389430c568d16e881aa9920`
- structured source facts SHA256:
  `c96cc4d084766ae1a2ea12f9fdcf6ec82d3146db759eabe9c02a28443bdcfb01`
- synthetic six-cell rows SHA256:
  `2e8ca5241b24f7844346cb15e414a476db3b51bd1ecbb9154211e23252e2775c`
- tokenizer anchor manifest SHA256:
  `82e6a18736701f17d3fda3e8c79c9abe5e947782bc7a33c99bd734c76a900c32`
- tokenizer eligibility summary SHA256:
  `31c29d3bad8df823bb8543b20d12fcef9ecf64d640a142ea70b8049d6f695b37`

The frozen static preparation records zero pair, claim, and evidence overlap
between this assay population and the prior/training populations.

Because confirmatory labels are intentionally absent, this audit must not
invent labels and must not report CE, accuracy, label-conditioned statistics,
or p-values.

## Frozen model and representation inputs

Use exactly the same:

- nine frozen A-init x training-RNG correction checkpoints;
- parent checkpoint;
- active tokenizer snapshot;
- recovered task-sensitive basis;
- functional implementation;
- layer-22 intervention boundary;
- centered three-class logit readout;
- two-margin readout;
- A-init and training-RNG pair definitions

as the frozen dev causal intervention evidence.

No checkpoint retraining or model selection is authorized.

## Frozen pair classes

Primary causal pair class:

- same training RNG;
- different A-init;
- exactly 9 unordered checkpoint pairs.

Control class:

- same A-init;
- different training RNG;
- exactly 9 unordered checkpoint pairs.

Different-A / different-RNG pairs are not required for the primary
confirmatory conclusion and should not be used to rescue a failed primary
gate.

## Frozen intervention

For each aligned confirmatory row and each real endpoint pair `(i,j)` at the
layer-22 wrapper output:

`d = h_j - h_i`

For the already frozen recovered basis `V_k`:

`P_k = V_k V_k^T`

`d_visible = P_k d`

`d_complement = d - d_visible`

`h_visible = h_i + d_visible`

`h_complement = h_i + d_complement`

Run only the frozen downstream computation from each real or hybrid state.

No basis fitting, PCA fitting, Jacobian recomputation, or representation
learning is authorized on confirmatory rows.

## Prospectively frozen k values

Use exactly:

`k = {1,2,4,8,16,32,64,128,256}`

Primary confirmatory reference:

`k = 8`

Robustness band required for strong replication:

`k = {4,8,16}`

No outcome-conditioned k selection is allowed.

## Required label-free metrics

For the primary and control pair classes, report:

1. full layer-22 squared-distance energy;
2. top-k projected squared-distance energy;
3. complement squared-distance energy;
4. `Q_k = D_projected / D_full`;
5. same-RNG/different-A full-distance divided by same-A/different-RNG
   full-distance;
6. centered-logit endpoint effect energy `E_full`;
7. centered-logit visible-only effect energy `E_visible`;
8. centered-logit complement-only effect energy `E_complement`;
9. centered-logit nonlinear interaction energy `E_interaction`;
10. corresponding ratios `R_visible`, `R_complement`, `R_interaction`;
11. the same effect energies and ratios for the two-margin vector;
12. endpoint prediction disagreement;
13. visible-only prediction disagreement versus source and target;
14. complement-only prediction disagreement versus source and target;
15. reverse-corner affine-state symmetry diagnostics.

Do not compute CE because the confirmatory population has no frozen labels.

## Preregistered strong-replication gates

All gates below are fixed before confirmatory execution.

### Gate A: A-init ambient separation persists

For the confirmatory population:

`D_full_same_RNG_diff_A / D_full_same_A_diff_RNG >= 20`

This is deliberately much weaker than the frozen dev ratio and only tests
whether A-init remains the dominant source of ambient endpoint separation.

### Gate B: most k=8 A-init hidden residual energy remains outside visible space

For primary same-RNG/different-A pairs:

`Q_8 <= 0.20`

Equivalently at least `80%` of hidden residual squared energy must remain in the
k=8 complement.

### Gate C: k=8 visible component carries the endpoint functional effect

For both centered logits and two-margin readout:

`0.70 <= R_visible(k=8) <= 1.30`

### Gate D: k=8 complement remains low-effect

For both centered logits and two-margin readout:

`R_complement(k=8) <= 0.02`

### Gate E: k=8 nonlinear interaction remains small

For both centered logits and two-margin readout:

`R_interaction(k=8) <= 0.02`

### Gate F: effect concentration

For both centered logits and two-margin readout:

`R_visible(k=8) / max(R_complement(k=8), 1e-12) >= 50`

### Gate G: stable compact-subspace replication

For every `k` in `{4,8,16}` and for both centered logits and two margins:

- `0.60 <= R_visible <= 1.40`
- `R_complement <= 0.05`
- `R_interaction <= 0.05`

This prevents a single favorable k from defining the conclusion.

## Outcome classification

### STRONG_REPLICATION

Allowed only if Gates A through G all pass.

Authorized conclusion:

`THE_DEV_DERIVED_LAYER22_A_INIT_VISIBLE_COMPLEMENT_DECOMPOSITION_REPLICATES_ON_THE_PREVIOUSLY_UNSEEN_9601_9900_POPULATION`

A bounded interpretation may state that A-init selects substantially different
layer-22 representatives, most of whose residual energy is downstream
effectively null / very low gain under the tested endpoint-chord interventions,
while the smaller frozen visible component carries the measurable functional
difference on unseen examples.

### QUALIFIED_REPLICATION

Use if the central visible-versus-complement ordering persists but one or more
strong-replication gates fail without a direct causal reversal.

Examples include moderate degradation of effect concentration, a larger but
still subdominant complement effect, or increased nonlinear interaction.

Do not report strong replication.

### CONFIRMATORY_FAILURE

Use if any of the following occurs:

- A-init ambient separation no longer dominates the RNG control;
- complement-only effect becomes comparable to the full endpoint effect;
- visible-only effect fails to account for a substantial fraction of endpoint
  functional difference;
- nonlinear interaction becomes a dominant term;
- frozen source identities or data identities fail authentication.

A confirmatory failure must be reported as such and must not be repaired by
refitting the basis, choosing another k, excluding unfavorable rows, or
reopening the same confirmatory population under a revised threshold.

## Prediction disagreement

Prediction disagreement is required as a descriptive diagnostic but is not a
standalone strong-replication gate.

The source dev endpoints had zero prediction disagreement. Previously unseen
rows may be closer to decision boundaries, so a small number of class flips
must not override the preregistered continuous-logit and margin causal gates.

Any prediction-disagreement result must nevertheless be reported exactly.

## Runtime and provenance constraints

- frozen 1800 confirmatory rows only;
- no dev rows may be mixed into confirmatory metrics;
- no training;
- no optimizer;
- no parameter update;
- no autograd;
- no backward;
- no checkpoint mutation;
- no new scientific seed;
- no basis recomputation;
- no outcome-conditioned row filtering;
- no outcome-conditioned k selection;
- both Tesla T4 devices must be used for scientific execution;
- deterministic two-worker sharding is required;
- each endpoint pair must be evaluated entirely on exactly one GPU;
- no tensor, hidden-state, logit, or metric reduction may cross GPUs during a pair evaluation;
- aggregate pair-level sufficient statistics only after per-pair computation is complete;
- GPU assignment must be fixed before execution and independent of outcomes;
- no DDP gradient synchronization or distributed training is permitted;
- exact runtime, tokenizer, model snapshot, parent checkpoint, CUDA kernel, and
  nine correction-checkpoint identities must be authenticated before
  scientific output is accepted.

## Two-GPU execution contract

The validated Kaggle environment contains exactly two Tesla T4 devices and both
must be used to reduce wall-clock time without changing scientific semantics.

Use exactly two independent inference workers:

- worker 0: `CUDA_VISIBLE_DEVICES=0`
- worker 1: `CUDA_VISIBLE_DEVICES=1`

Deterministic pair assignment:

- worker 0 evaluates the 9 primary same-training-RNG / different-A-init pairs;
- worker 1 evaluates the 9 control same-A-init / different-training-RNG pairs.

Each worker must:

- load the same frozen parent checkpoint and authenticated artifacts;
- process all confirmatory rows required for its assigned pair class;
- evaluate every frozen `k` value for each assigned pair;
- keep every individual endpoint/hybrid forward pass on its assigned GPU;
- emit pair-level sufficient statistics only;
- perform no scientific communication with the other GPU during inference.

After both workers complete successfully, a CPU-only coordinator must merge
their pair-level sufficient statistics in the frozen canonical pair order and
compute the preregistered aggregate metrics and gates.

The coordinator must authenticate that:

- exactly 9 primary and 9 control pairs were completed;
- every frozen pair appears exactly once;
- no pair was split across GPUs;
- no row was omitted or duplicated;
- both GPUs report the same frozen model/data/artifact identities;
- both workers completed without fallback to CPU;
- no outcome-dependent reassignment occurred.

This two-GPU arrangement is a runtime acceleration only. It must not alter the
scientific estimator, pair definitions, basis, k values, intervention states,
or preregistered thresholds.

## Required artifacts

Write only under:

`reports/reason_router_gen5_ainit_visible_complement_confirmatory_runs/<run-name>/`

Required files:

- `ainit_visible_complement_confirmatory_summary.json`
- `visible_complement_confirmatory_metrics.pt`
- `run_provenance.json`

Do not persist full hidden-state dumps.

## One-shot boundary

This authority consumes `xg1_fact_9601..xg1_fact_9900` for this scientific
hypothesis.

After the first scientifically valid confirmatory execution is imported:

- do not rerun the same population to tune thresholds;
- do not refit the task-sensitive basis using confirmatory rows;
- do not choose a new primary k based on confirmatory results;
- do not silently reclassify the population as development data.

A runtime or implementation failure that occurs before scientific metrics are
produced may be repaired under the same frozen scientific contract, with the
failed execution retained as provenance.

## Scientific boundary

A strong replication would establish out-of-development-population replication
within the same synthetic generator family and frozen task contract.

It would not establish:

- Transformer inferiority or Mamba superiority;
- an exact mathematical gauge symmetry;
- a universal null manifold;
- arbitrary-perturbation invariance;
- natural-language-domain generalization outside this generator family;
- an optimization-trajectory mechanism.

Architecture comparison, if later pursued, requires a separately controlled
Transformer or other architecture experiment and must not be inferred from
this confirmatory run.
