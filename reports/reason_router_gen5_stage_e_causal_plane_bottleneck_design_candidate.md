# Gen5 Stage E — Causal-Plane Bottleneck Training Design

## Status

PHASE=PROSPECTIVE_SCIENTIFIC_DESIGN_ONLY

IMPLEMENTATION_ALLOWED=NO

TRAINING_ALLOWED=NO

EVALUATION_ALLOWED=NO

This document freezes the prospective Stage E scientific comparison before
implementation or observation of Stage E outcomes.

## Prior validated milestone

Stage D evidence freeze:

`3e0a985abaf99fc44c3ec808636ce1410d58c370`

The validated Stage A–D sequence supports the bounded mechanism:

`NATIVE_CAUSAL_IMPORTANCE_DOES_NOT_IMPLY_OPTIMIZATION_PRIVILEGE`

Relevant established evidence:

- R22 is locally causally important in the frozen native layer-22 computation.
- Learned rank-2 correction planes occupy directions almost orthogonal to R22.
- Removing the learned correction's R22 component preserves essentially all
  realized learned task benefit.
- The initial task-loss-driven correction gradient already bypasses R22.
- R22 and learned-B downstream image planes show partial anisotropic functional
  convergence without effective-rank collapse.

Stage E must not reinterpret those results as complete downstream equivalence.

## Scientific question

Does R22 contain task-usable correction capacity that free optimization ignores
because it can select a different output-plane orientation?

Equivalently:

If the rank-2 correction is denied the ability to choose an arbitrary
24576-dimensional output plane, can optimization succeed when its output plane
is fixed to the native-causal R22 plane, and does that success differ from a
matched frozen orthogonal C22 control plane?

## Experimental intervention

Retain the rank-2 correction form but fix its output plane.

For a frozen orthonormal basis Q with shape [24576, 2]:

`B_eff = Q M`

where:

- `M` has shape `[2, 2]`;
- `A` has shape `[2, 768]`;
- correction operator is `Q M A`;
- `Q` is frozen;
- `M` and `A` are trainable;
- `M` is initialized exactly to zero;
- A initialization is seed-matched across the two new arms.

This permits any rank-at-most-2 linear correction whose output lies in the
fixed plane Q.

The two Stage E arms are:

`E-R22`
- `Q = R22`

`E-C22`
- `Q = C22`

R22 and C22 must use the already frozen exact basis identities:

R22 SHA256:

`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22 SHA256:

`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

No new plane fitting is allowed.

## Why C22 is the matched control

C22 has the same rank and the same state-write ambient space as R22 and is
frozen orthogonal to R22.

Therefore E-R22 versus E-C22 compares two fixed rank-2 output planes under the
same parameterization, optimizer, data, seeds, objective, and training budget.

The unrestricted free-B condition is a reference baseline, not a
parameter-count-matched treatment arm.

## Existing unrestricted baseline

Do not retrain the unrestricted P0 arm.

Reuse the frozen Phase3A P0 results and Stage B dev evaluation.

Frozen source execution:

`d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

Frozen Stage B evidence:

`b49c1339231a1c5dbb7f870ddc81159109b08c1e`

P0 frozen dev ZERO CE:

`1.3409655094146729`

Seed 6201:

- unrestricted FULL CE = `0.8424794673919678`
- unrestricted gain vs ZERO = `0.4984860420227051`

Seed 6202:

- unrestricted FULL CE = `0.8389552235603333`
- unrestricted gain vs ZERO = `0.5020102858543396`

Seed 6203:

- unrestricted FULL CE = `0.841503918170929`
- unrestricted gain vs ZERO = `0.4994615912437439`

Mean unrestricted P0 gain:

`0.4999859730402629`

These values are frozen reference quantities and must not be recomputed from a
new unrestricted training run.

## Matrix

New Stage E training matrix:

- seed6201 / E-R22
- seed6201 / E-C22
- seed6202 / E-R22
- seed6202 / E-C22
- seed6203 / E-R22
- seed6203 / E-C22

Exactly six new cells.

No PR or PC Stage E cells are planned.

Reason:

Stage A–D showed no material pressure-specific change sufficient to justify a
new three-pressure training matrix.

## Frozen training contract

Reuse the Phase3A training population, dev population, tokenizer/runtime
identity, parent checkpoint, final 3-way CE objective, and optimizer contract.

Required training settings:

- seeds: 6201, 6202, 6203
- pressure: P0 only
- optimizer: AdamW
- learning rate: 0.001
- weight decay: 0.0001
- optimizer steps: 20
- gradient clip norm: 5.0
- no scheduler
- frozen parent model
- only Stage E correction parameters trainable
- no hyperparameter tuning
- no early stopping
- no checkpoint selection
- final fixed step only

Parent checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

The confirmatory population `xg1_fact_9601..9900` remains forbidden.

## Primary quantities

For each new cell compute on the frozen Phase3A dev population:

`gain_Q = CE_ZERO - CE_Q`

where Q is R22 or C22.

For each seed compute seed-matched unrestricted recovery:

`recovery_Q = gain_Q / gain_FREE_P0`

and the matched-plane contrast:

`delta_RC = recovery_R22 - recovery_C22`

Report all seedwise values and their descriptive mean.

Do not use scientific p-values.

Do not set a post-hoc promotion threshold after observing Stage E results.

## Secondary diagnostics

Record:

- train objective at step 0 and final fixed step;
- final dev CE;
- final dev accuracy;
- optimizer-step count;
- gradient norms;
- exact A and M tensor hashes;
- effective correction operator rank;
- maximum residual outside the authorized fixed output plane;
- parent parameter identity before and after training.

The fixed-plane residual must be validated directly from the realized
correction operator.

## Prospective interpretation map

### Pattern A — R22-specific constrained recoverability

If E-R22 consistently recovers more of the unrestricted P0 task gain than
E-C22, this supports the interpretation that R22 contains task-usable capacity
but unrestricted optimization preferentially selects an alternate plane.

This would strengthen the causal explanation of optimization non-privilege.

### Pattern B — broad fixed-plane substitutability

If E-R22 and E-C22 both recover substantial and similar unrestricted task gain,
the evidence favors broad downstream functional substitutability rather than
R22-specific task utility.

### Pattern C — fixed-plane insufficiency

If both E-R22 and E-C22 recover little unrestricted task gain, the arbitrary
orientation learned by the unrestricted rank-2 correction is important.

This would be consistent with native-causal importance being distinct from the
task objective and would strengthen the motivation for H6.

### Pattern D — C22 exceeds R22

If the matched C22 bottleneck consistently outperforms the R22 bottleneck,
there is no evidence that native-causal R22 enjoys special task-objective
utility under the constrained correction.

This would sharpen rather than weaken the distinction between native causal
importance and optimization privilege.

## Falsification and scope

Stage E must not claim that R22 is generally optimal or generally useless.

The experiment is limited to:

- the frozen Mamba-130M lineage;
- layer 22;
- the frozen R22 and C22 planes;
- the frozen Gen5 correction mechanism;
- the frozen Phase3A train/dev populations;
- the frozen 20-step optimization contract.

No layer sweep, plane sweep, rank sweep, learning-rate sweep, optimizer sweep,
seed expansion, pressure expansion, token search, or confirmatory-data access
is authorized by this design.

## Next boundary

After this design is frozen, the next phase is bounded Stage E implementation.

A separate explicit execution authority is still required before any new
training or evaluation is run.
