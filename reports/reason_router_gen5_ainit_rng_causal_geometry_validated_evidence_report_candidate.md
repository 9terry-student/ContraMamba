# Gen5 A-Initialization × Training-RNG Causal Geometry Validated Evidence Report Candidate

## Status

VALIDATED_CAUSAL_GEOMETRY_EVIDENCE_CANDIDATE

## Evidence identity

Execution commit:

`87f82551c721f953f710cd5dc102aca23161c4e7`

Corrected implementation freeze:

`06a245f24e4474e2b039b954abd7b6cbf436e822`

Recovery implementation authority:

`3f5d267508d3efaf123b93563f1563c20db62fe1`

Historical failed execution authority:

`54a0bffa075ac7c2f25147ff53fb971412b99749`

Run:

`gen5-ainit-rng-causal-six-offdiag-87f8255-r1`

Imported run root:

`reports/reason_router_gen5_ainit_rng_causal_intervention_runs/gen5-ainit-rng-causal-six-offdiag-87f8255-r1/`

Execution/artifact status:

- corrected CUDA preflight: PASS;
- six off-diagonal matrix: PASS;
- new off-diagonal cells: `6`;
- reused frozen diagonal cells: `3`;
- GPU topology: `TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`;
- optimizer steps per new cell: `20`;
- total new optimizer steps: `120`;
- confirmatory 9601–9900 loaded: `false`;
- scientific p-value count: `0`;
- collector: PASS;
- import: PASS;
- imported files validated: `22`.

The failed preflight under the earlier cross-version raw-byte hash assumption is historical provenance only and is not part of the successful scientific matrix.

## Scientific question

The preceding static mechanism analysis established strong seed-matched A-initialization anchoring, but could not distinguish whether the apparent read-side seed dependence was caused by A initialization itself or by other stochastic training effects coupled to the same seed.

The present experiment factorized the two sources:

`a_init_seed ∈ {6201, 6202, 6203}`

`training_rng_seed ∈ {6201, 6202, 6203}`

The six off-diagonal cells were executed prospectively and the three diagonal controls were reused from the frozen Phase3A P0 runs.

The primary question was:

> Does final read-side geometry and the dominant right/input functional direction track A-initialization identity more strongly than the remaining training RNG?

## Frozen contract

All nine cells share:

- arm: `G5-C0`;
- pressure: `P0`;
- train/dev rows: `3360/840`;
- split seed: `16384`;
- rank: `2`;
- `A_theta.weight` shape: `[2,768]`;
- `B_theta.weight` shape: `[24576,2]`;
- B initialization: exact zero;
- objective: final 3-way cross entropy only;
- optimizer: AdamW;
- learning rate: `0.001`;
- weight decay: `0.0001`;
- gradient clip norm: `5.0`;
- fixed horizon: `20` optimizer steps;
- no scheduler;
- no early stopping;
- frozen Phase3A dev evaluation;
- no confirmatory 9601–9900 population.

A initialization used the frozen mathematical rule:

`CPU float32 kaiming_uniform_(shape=(2,768), a=sqrt(5), generator=manual_seed(A_INIT_SEED))`

The corrected runtime authentication used exact same-process reconstruction and did not treat cross-PyTorch raw tensor hashes as universal identities.

## Execution and provenance validation

The successful Kaggle runtime used:

- Python `3.12.13`;
- PyTorch `2.10.0+cu128`;
- CUDA `12.8`;
- two Tesla T4 GPUs;
- Transformers `5.0.0`;
- Tokenizers `0.22.2`;
- kernels `0.10.2`.

The matrix completed with:

`GEN5_AINIT_RNG_CAUSAL_SIX_OFFDIAGONAL_MATRIX_PASS`

Run provenance reported:

- execution commit `87f82551c721f953f710cd5dc102aca23161c4e7`;
- implementation freeze `06a245f24e4474e2b039b954abd7b6cbf436e822`;
- new cell count `6`;
- reused diagonal cell count `3`;
- optimizer step count `120`;
- training executed `true`;
- task evaluation executed `true`;
- status `PASS`.

The collector and local importer validated all 22 run artifacts before scientific interpretation.

## Static causal-geometry analysis

Post-import analysis was read-only CPU tensor/linear algebra over the imported off-diagonal checkpoints and frozen diagonal checkpoints.

No CUDA, model construction, forward pass, backward pass, optimizer construction, training, or new task evaluation occurred in this analysis.

The local A-init raw-byte hashes were treated only as runtime-specific diagnostic metadata.

Terminal result:

`GEN5_AINIT_RNG_CAUSAL_GEOMETRY_STATIC_ANALYSIS_PASS`

## Endpoint 1: final row(A) follows A initialization

Across all nine cells, final `row(A)` had a large affinity advantage for its matched A-initialization plane.

Mean matched-initialization row-space affinity advantage:

`0.725006440782`

Range:

`[0.719684425782, 0.730177454452]`

The grouped factorial contrast was decisive.

Mean final `row(A)` affinity for pairs with the same A-init but different training RNG:

`0.999963824427`

Range:

`[0.999926054326, 0.999981861193]`

Mean final `row(A)` affinity for pairs with the same training RNG but different A-init:

`0.144047932169`

Range:

`[0.137493800558, 0.147922527498]`

Difference:

`0.855915892257`

Thus changing the remaining training RNG while preserving A initialization barely changes the final read-side plane, whereas changing A initialization under a fixed training RNG produces a large geometric change.

## Endpoint 2: dominant right/input functional direction follows A initialization

Across all nine cells, the dominant right/input direction of the learned correction operator also retained a large matched A-initialization advantage.

Mean matched-initialization dominant-right capture advantage:

`0.473334650113`

Range:

`[0.462067641609, 0.487928227297]`

Mean dominant-right absolute cosine for pairs with the same A-init but different training RNG:

`0.999984480062`

Range:

`[0.999974740072, 0.999989833897]`

Mean dominant-right absolute cosine for pairs with the same training RNG but different A-init:

`0.535662952798`

Range:

`[0.524116408526, 0.542863787959]`

Difference:

`0.464321527264`

Therefore the A-initialization intervention does not merely preserve a raw parameter-plane identity. It also strongly determines the dominant input/read direction of the final learned correction operator under the tested Phase3A P0 training contract.

## Cellwise stability under training-RNG changes

For A-init seed 6201, all three training-RNG realizations gave pairwise final row-space affinities above `0.99992` and dominant-right cosines above `0.99997`.

For A-init seed 6202, all three training-RNG realizations gave pairwise final row-space affinities above `0.99998` and dominant-right cosines above `0.99998`.

For A-init seed 6203, all three training-RNG realizations gave pairwise final row-space affinities above `0.99996` and dominant-right cosines above `0.99998`.

This near-identity is observed across all three A-init levels rather than being driven by one seed.

## Secondary dev guardrail

All six newly executed off-diagonal cells had dev accuracy:

`0.714285714286`

Mean off-diagonal dev cross entropy:

`0.841002384822`

Range:

`[0.838918626308, 0.842584609985]`

Within each A-init group, changing training RNG produced only small off-diagonal dev-CE variation:

- A6201: mean `0.842542707920`, range width `0.000083804130`;
- A6202: mean `0.838922411203`, range width `0.000007569790`;
- A6203: mean `0.841542035341`, range width `0.000040471554`.

These dev results are a secondary guardrail, not the primary causal endpoint.

## Combined interpretation

The experiment resolves the principal ambiguity left by the earlier static initialization-anchoring result.

The earlier observation that final read-side geometry remained close to its own seed-matched initialization could have been explained by other stochastic training effects sharing the same seed.

The present full 3×3 factorization breaks that coupling.

When A initialization is held fixed and the remaining training RNG is changed, both the final `row(A)` plane and the dominant right/input functional direction remain almost unchanged.

When the training RNG is held fixed and A initialization is changed, those same endpoints separate strongly by A-init identity.

Under the exact tested Phase3A P0 contract, the evidence therefore supports a causal role for A initialization in selecting the final read-side solution family.

## Supported bounded conclusions

`GEN5_READ_SIDE_GEOMETRY_CAUSALLY_TRACKS_A_INITIALIZATION_UNDER_FIXED_PHASE3A_P0_TRAINING`

`GEN5_DOMINANT_INPUT_FUNCTIONAL_DIRECTION_CAUSALLY_TRACKS_A_INITIALIZATION_UNDER_FIXED_PHASE3A_P0_TRAINING`

`GEN5_REMAINING_TRAINING_RNG_HAS_ONLY_MINOR_EFFECT_ON_FINAL_READ_SIDE_GEOMETRY_RELATIVE_TO_A_INITIALIZATION_UNDER_THE_TESTED_CONTRACT`

`GEN5_STATIC_INITIALIZATION_ANCHORING_WAS_NOT_EXPLAINED_BY_SHARED_TRAINING_RNG_SEED_ALONE`

## What is not established

The current evidence does not establish:

- that A initialization is the sole causal determinant of the full learned operator;
- that the output/write side is fully determined by A initialization;
- that training RNG has literally zero effect;
- that A initialization alone determines task performance;
- that the same causal dominance holds outside seeds 6201–6203;
- that the same result holds outside P0, this dataset, layer, model, optimizer, learning rate, or 20-step horizon;
- that the observed effect generalizes to other architectures;
- a universal intrinsic dimension;
- a general optimization theorem;
- a LoRA or parameter-efficient fine-tuning mechanism;
- a confirmatory statistical claim.

The correct scope is the frozen Phase3A P0 training contract and the tested 3×3 seed factorization.

## Next scientific action

Do not start another training sweep.

The next discriminating stage should test whether the A-init-selected geometric identity is also expressed in model behavior.

Use the already frozen 3×3 checkpoints and the same frozen Phase3A dev set.

Perform a bounded no-training functional fingerprint evaluation that compares:

1. dev-logit similarity for same-A-init / different-training-RNG pairs;
2. dev-logit similarity for same-training-RNG / different-A-init pairs;
3. prediction agreement;
4. per-example margin-vector similarity;
5. classwise logit residual structure;
6. grouped descriptive A-init versus training-RNG contrasts.

The primary question is:

> Does functional output behavior cluster by A-initialization identity in the same way as final read-side geometry?

This stage should not modify checkpoints, train, sweep hyperparameters, access confirmatory 9601–9900 data, or introduce a new architecture.

If functional behavior follows A-init identity, the causal chain would extend from:

`A initialization -> final read-side geometry`

to the stronger bounded mechanism:

`A initialization -> final read-side geometry -> reproducible functional behavior`

If functional behavior does not follow A-init identity despite geometric separation, the result would instead imply substantial downstream functional equivalence across distinct A-selected read-side solutions.
