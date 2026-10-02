# Gen5 Cross-Seed Functional Geometry and Initialization Anchoring Mechanism Report Candidate

## Status

VALIDATED_STATIC_MECHANISM_ANALYSIS_CANDIDATE

## Evidence boundary

Repository HEAD:

`579b10783979c38d9561b0b81c1fa9dd3fe270e7`

Frozen SCALEMATCH evidence commit:

`579b10783979c38d9561b0b81c1fa9dd3fe270e7`

Source unrestricted Phase3A execution commit:

`d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

Seeds:

- 6201
- 6202
- 6203

Pressure:

`P0`

All analyses in this report were read-only CPU linear algebra over already frozen checkpoints.

No CUDA, model construction, model forward, backward, optimizer construction, training, task evaluation, confirmatory 9601–9900 access, or scientific p-values were used.

## Static analyses

### 1. Cross-seed subspace meta-audit

Analyzer:

`analyze_gen5_cross_seed_subspace_meta_audit_579b107.py`

SHA256:

`dd086a20b1545fec3dbcaf5caffaa06b51720f1656ff1d894db23bdb93897d22`

Terminal result:

`GEN5_CROSS_SEED_SUBSPACE_META_AUDIT_PASS`

#### Output/write subspace `Q_B`

Pairwise principal cosines:

- 6201–6202: `[0.9212861666, 0.0489090504]`
- 6201–6203: `[0.9057647613, 0.4492294787]`
- 6202–6203: `[0.9352755300, 0.3474620622]`

Mean leave-one-seed-out held-out energy captured by the other two seeds' 4D span:

`0.572740648609`

Mean held-out residual:

`0.427259351391`

The first principal direction is strongly shared across seeds, while the second direction is substantially more seed-specific.

#### Input/read subspace `row(A)`

Pairwise principal cosines:

- 6201–6202: `[0.5409203535, 0.0284949129]`
- 6201–6203: `[0.5255198344, 0.0005105742]`
- 6202–6203: `[0.5424678123, 0.0273358303]`

Mean leave-one-seed-out held-out energy captured by the other two seeds' 4D span:

`0.188418995271`

Mean held-out residual:

`0.811581004729`

Thus the input/read side is much more seed-specific than the output/write side.

### 2. Dominant functional-axis cross-seed audit

Analyzer:

`analyze_gen5_dominant_functional_axis_cross_seed_579b107.py`

SHA256:

`c767ace017628f7af6e0a91f50b65d795ec7bd9dc87559169ed44837a342f381`

Terminal result:

`GEN5_DOMINANT_FUNCTIONAL_AXIS_CROSS_SEED_AUDIT_PASS`

#### Within-seed source versus SCALEMATCH

Source unrestricted operator top-rank-1 energy fractions:

- seed6201: `0.920064383108`
- seed6202: `0.946928306568`
- seed6203: `0.894321580215`

SCALEMATCH operator top-rank-1 energy fractions:

- seed6201: `0.999999971040`
- seed6202: `0.999958186742`
- seed6203: `0.999999960282`

Source versus SCALEMATCH dominant rank-1 operator cosine:

- seed6201: `0.924559222492`
- seed6202: `0.956792097409`
- seed6203: `0.951135857044`

Source versus SCALEMATCH dominant right/input singular-vector cosine:

- seed6201: `0.999982864073`
- seed6202: `0.999952059129`
- seed6203: `0.999997028242`

Therefore the ~99 percent SCALEMATCH task-gain recovery is associated with near-exact preservation of the same-seed dominant input/read direction and strong recovery of the same-seed dominant rank-1 functional operator.

#### Cross-seed dominant-axis geometry

Unrestricted source dominant axes:

- mean left/output cosine: `0.912596022790`
- mean right/input cosine: `0.535762257703`
- mean dominant rank-1 operator cosine: `0.488989070757`

SCALEMATCH dominant axes:

- mean left/output cosine: `0.832712402936`
- mean right/input cosine: `0.535659385956`
- mean dominant rank-1 operator cosine: `0.446076206883`

Thus the dominant output/write direction is comparatively shared across seeds, while the dominant input/read direction remains substantially seed-specific.

### 3. Read-side initialization-anchoring audit

Analyzer:

`analyze_gen5_read_side_initialization_anchoring_579b107.py`

SHA256:

`338eb4182f005edacea757ac0565de7f9d1e9b2dc7b2e606e8985b738b313ff0`

Terminal result:

`GEN5_READ_SIDE_INITIALIZATION_ANCHORING_AUDIT_PASS`

Frozen initialization rule reconstructed exactly:

`CPU float32 kaiming_uniform_(shape=(2,768), a=sqrt(5), generator=manual_seed(seed))`

Pairwise initial `row(A_init)` affinities:

- 6201–6202: `0.000870931893283`
- 6201–6203: `0.000699951660147`
- 6202–6203: `0.00112546176542`

These seed-matched initial planes are essentially mutually unrelated at the descriptive high-dimensional scale.

Final `row(A_final)` versus same-seed initialization:

- seed6201 affinity: `0.731693185850`
- seed6202 affinity: `0.726283861038`
- seed6203 affinity: `0.721162602821`

Mean:

`0.726379883236`

Final `row(A_final)` versus other-seed initializations:

Mean affinity:

`0.001362400813`

For reference, rank-2 random-subspace affinity in 768 dimensions is:

`0.002604166667`

Thus the final read-side plane remains strongly anchored to its own seed-matched initialization and has essentially no corresponding affinity to the other seeds' initial planes.

#### Dominant right/input direction versus A initialization

Source unrestricted dominant right direction captured by own seed `row(A_init)`:

Mean:

`0.474968690695`

Source dominant right direction captured by other-seed initializations:

Mean:

`0.001767005125`

SCALEMATCH dominant right direction captured by own seed `row(A_init)`:

Mean:

`0.474684885622`

SCALEMATCH dominant right direction captured by other-seed initializations:

Mean:

`0.001761910583`

The same-seed source and SCALEMATCH dominant right directions are themselves almost identical, with cosine greater than `0.99995` in every seed.

## Combined interpretation

The combined static evidence supports the following bounded mechanism picture.

### 1. The unrestricted rank-2 solution is not a universal shared subspace

Both the output/write and especially the input/read subspaces vary across seeds.

A single tiny universal rank-2 meta-subspace is not supported by these three seeds.

### 2. Cross-seed variation is strongly asymmetric

The output/write side contains a strongly shared first direction across seeds.

The input/read side is much more seed-specific.

This asymmetry persists at the dominant functional-axis level.

### 3. Near-full SCALEMATCH task recovery is associated with dominant-mode recovery, not arbitrary operator substitution

SCALEMATCH does not reconstruct the full source rank-2 operator.

However, it recovers the same-seed dominant rank-1 functional axis with high cosine and preserves the same-seed dominant right/input direction almost exactly.

Therefore the strongest interpretation is not that arbitrary operators are interchangeable.

Instead:

`NEAR_FULL_TASK_RECOVERY_IS_COMPATIBLE_WITH_PRESERVING_THE_DOMINANT_FUNCTIONAL_MODE_WHILE_DISCARDING_MOST_OF_THE_SECOND_SOURCE_MODE`

### 4. The seed-specific read-side geometry is strongly initialization-anchored

The final read-side plane retains approximately `0.726` affinity to its own deterministic random initialization but only approximately `0.00136` affinity to other-seed initializations.

The dominant task-effective right/input direction also retains approximately `0.475` energy inside the own seed's initial 2D read plane while having approximately `0.00177` capture in other-seed initial planes.

This establishes strong geometric path dependence on the seed-matched A initialization.

It does not, by static analysis alone, prove that initialization is the sole causal source of final read-side variation.

## Supported bounded conclusions

`GEN5_CROSS_SEED_OUTPUT_WRITE_GEOMETRY_CONTAINS_A_STRONGLY_SHARED_DOMINANT_DIRECTION_BUT_NOT_A_FULLY_SHARED_RANK2_PLANE`

`GEN5_CROSS_SEED_INPUT_READ_GEOMETRY_IS_SUBSTANTIALLY_MORE_SEED_SPECIFIC_THAN_OUTPUT_WRITE_GEOMETRY`

`GEN5_SCALEMATCH_NEAR_FULL_TASK_RECOVERY_TRACKS_SAME_SEED_DOMINANT_RANK1_FUNCTIONAL_AXIS_RECOVERY`

`GEN5_FINAL_READ_SIDE_GEOMETRY_IS_STRONGLY_ANCHORED_TO_SEED_MATCHED_A_INITIALIZATION`

`GEN5_DOMINANT_INPUT_DIRECTION_RETAINS_STRONG_SAME_SEED_INITIALIZATION_MEMORY`

## What is not established

The current evidence does not establish:

- that A initialization is the unique causal source of read-side seed dependence;
- that fixing A initialization would make final operators seed-invariant;
- that the shared output/write direction is universal across tasks, datasets, layers, models, or seeds beyond 6201–6203;
- that the second source mode is universally task-irrelevant;
- a universal intrinsic dimension;
- a general optimization theorem;
- a LoRA or parameter-efficient fine-tuning mechanism;
- a confirmatory statistical claim.

## Next discriminating experiment

The next execution, if authorized, should isolate initialization causally rather than adding architectural complexity.

A minimal causal design is an `A-init seed × training RNG seed` intervention under the existing frozen Phase3A P0 contract.

The experiment should separate:

- the seed controlling `A_theta` initialization;
- the seed controlling all remaining training RNG.

The primary mechanistic question is:

> Does the final read-side geometry and dominant right/input functional direction follow the A-initialization seed more strongly than the remaining training RNG seed?

The experiment should keep fixed:

- parent model/checkpoint;
- data and row order;
- P0 pressure;
- rank 2;
- objective;
- optimizer;
- learning rate;
- weight decay;
- gradient clipping;
- 20-step horizon;
- zero initialization of B;
- evaluation domain.

No architecture change, rank sweep, learning-rate sweep, horizon sweep, new objective, or confirmatory population should be introduced before this causal test.

This report itself authorizes no implementation, training, evaluation, CUDA execution, or Kaggle execution.
