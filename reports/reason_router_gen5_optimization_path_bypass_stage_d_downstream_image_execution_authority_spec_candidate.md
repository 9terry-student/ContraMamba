# Gen5 Optimization-Path Bypass Stage D
## Downstream Image Equivalence Execution Authority

SCIENTIFIC_EXECUTION_ALLOWED=YES_FORWARD_ONLY_STAGE_D_DOWNSTREAM_IMAGE_MATRIX

IMPLEMENTATION_FREEZE_COMMIT=fde945d6266649bfeea9de7f2b9b37b226cae94f

TRAINING_ALLOWED=NO

BACKWARD_ALLOWED=NO

OPTIMIZER_ALLOWED=NO

CONFIRMATORY_9601_9900_ALLOWED=NO

BASELINE_TRAJECTORY=PHASE3A_ZERO_CORRECTION_DEV_EVAL

POPULATION=FROZEN_PHASE3A_DEV_STRESSOR_DOMAIN_480_ROWS

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP

## Objective

Measure whether the previously causally validated native R22 write subspace
and the learned final correction-write subspace have similar downstream image
geometry when probed from the same frozen zero-correction Phase3A dev-eval
baseline.

The primary comparison is:

`J Q_R22`

versus:

`J Q_Bfinal`

where `Q_R22` is the frozen orthonormal R22 basis and `Q_Bfinal` is an
orthonormal basis for each frozen Phase3A cell's final learned B column space.

This stage measures downstream subspace image geometry only.

It does not train, update, optimize, select, or refit any object.

## Frozen ancestry

Stage C validated evidence freeze:

`bc226f5a19f42d26c916fe7b339b26c41a8fdfe8`

Stage B validated evidence freeze:

`b49c1339231a1c5dbb7f870ddc81159109b08c1e`

Source Phase3A execution:

`d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

Stage D implementation freeze:

`fde945d6266649bfeea9de7f2b9b37b226cae94f`

Frozen parent checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Frozen R22 SHA256:

`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

Frozen C22 SHA256:

`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

No checkpoint search, basis refit, plane search, layer search, token search,
pressure search, rank search, or response-guided filtering is authorized.

## Matrix

Exactly nine frozen Phase3A cells:

- seed6201 / P0
- seed6201 / PR
- seed6201 / PC
- seed6202 / P0
- seed6202 / PR
- seed6202 / PC
- seed6203 / P0
- seed6203 / PR
- seed6203 / PC

Each cell uses exactly the frozen Phase3A dev stressor-domain subset:

`480 rows`

The other 360 non-stressor dev rows are outside this Stage D image probe.

The confirmatory population `xg1_fact_9601..9900` must not be loaded.

## Baseline trajectory

All directional probes are evaluated around:

`PHASE3A_ZERO_CORRECTION_DEV_EVAL`

The learned final B matrix supplies only the two-dimensional comparison
subspace `span(B_final)`.

The final learned correction itself is not activated as the baseline
perturbation.

The correction B parameter remains exactly zero throughout probing.

Therefore this stage compares downstream local images of two input write
subspaces around one common frozen dev-eval operating point.

## Direction bases

R22 directions:

- use the exact frozen rank-2 orthonormal R22 bytes.

Learned directions:

- load each frozen Phase3A cell's final B;
- require effective rank exactly two;
- construct an orthonormal basis for `span(B_final)` by SVD;
- do not interpret the original factorization gauge as scientifically
  meaningful.

All R22/Bfinal comparisons must be basis-invariant.

## Probe radii

Primary symmetric finite-difference radius:

`0.025`

Scale-audit radius:

`0.05`

The scale audit is restricted to the frozen first:

`32 rows per cell`

The 0.05 radius is a diagnostic linearity/scale check only.

It must not replace, tune, or select the primary radius.

## Target surface

The controlled perturbation is injected at the layer-22 native state-write
surface represented in the same 24576-dimensional WRITE22 coordinate system
used by R22.

The target token coordinates come only from the frozen Phase3A stressor target
manifest.

No target coordinate may be selected from observed Stage D responses.

A causal audit must verify that the induced final-hidden response is zero,
within implementation tolerance, before the target coordinate.

## Primary downstream surfaces

For each row compare the downstream image subspaces on:

1. final Mamba hidden sequence;
2. 384-dimensional task representation formed from:
   - frame pair representation;
   - predicate pair representation;
   - sufficiency representation;
3. five decision primitives:
   - frame probability;
   - predicate coverage probability;
   - sufficiency probability;
   - positive energy;
   - negative energy.

For each surface:

- compute effective rank before interpretation;
- compute basis-invariant principal cosines/angles;
- compute subspace affinity;
- mark rank-2 interpretation invalid if either image rank is below two.

Centered 3-way logits may be recorded descriptively, but their low output
dimension must not be used as the primary rank-2 functional-equivalence
verdict.

## Scale audit

For the frozen first 32 stressor-domain rows of each cell, compare derivative
estimates from radius 0.025 and radius 0.05.

Record:

- relative Frobenius difference;
- Frobenius cosine.

This audit diagnoses local finite-difference stability.

It does not authorize radius selection or exclusion of rows.

## GPU topology

Use exactly two Tesla T4 GPUs.

Topology:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

GPU0 queue:

1. seed6201 / P0
2. seed6201 / PC
3. seed6202 / PR
4. seed6203 / P0
5. seed6203 / PC

GPU1 queue:

1. seed6201 / PR
2. seed6202 / P0
3. seed6202 / PC
4. seed6203 / PR

Each cell loads a fresh model/runtime state.

No model parameters, correction parameters, optimizer state, gradients, or
scientific state are shared between workers.

## Forward budget

Primary probing requires:

- 4 basis directions per cell;
- positive and negative probe for each direction;
- 30 streams for 480 rows at stream size 16.

Primary forward budget:

`240 forwards per cell`

Scale audit adds:

`16 forwards per cell`

Total:

`256 forwards per cell`

Across nine cells:

`2304 scientific model forwards`

No baseline task-evaluation forward is added to this budget.

## Scientific firewall

Forbidden:

- backward
- optimizer construction
- optimizer step
- training
- parameter update
- gradient clipping
- new checkpoint creation
- new basis fitting
- radius tuning
- layer search
- target-token search
- seed expansion
- pressure expansion
- stressor search
- row dropping
- response-guided filtering
- confirmatory 9601..9900 access
- scientific p-values
- scientific conclusion during execution

## Interpretation boundary

Stage D must keep two scientific objects separate.

### A. Realized perturbation equivalence

Already measured statically from frozen Stage B outputs.

This concerns the actual learned perturbation and its R22 / R22-orthogonal
components.

### B. Subspace downstream-image equivalence

This execution measures whether R22 and `span(B_final)` themselves map into
similar downstream functional image subspaces around the common zero-correction
baseline.

Evidence for A does not imply B.

Evidence for B does not imply equality of actual perturbation magnitudes or
coefficients.

The two may only be synthesized after Stage D artifacts are imported and
validated.

## Stop conditions

Stop without scientific interpretation if any of the following occurs:

- implementation drift
- authority drift
- parent checkpoint mismatch
- R22/C22 identity mismatch
- Phase3A final-correction identity mismatch
- dev encoding mismatch
- stressor-domain row count other than 480
- target-coordinate mismatch
- final B effective rank below two
- non-finite derivative
- pre-target causal-response violation
- runtime/kernel mismatch
- GPU topology mismatch
- any backward execution
- optimizer construction
- optimizer step
- training execution
- parent parameter mutation
- nonzero baseline correction B
- confirmatory population access

A successful run establishes Stage D measurement evidence only.
