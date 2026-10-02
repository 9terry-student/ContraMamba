# Gen5 Stage E Learned-B-Plane A-Initialization Diagnostic
# Execution Authority

## Authority identities

DESIGN_IMPLEMENTATION_AUTHORITY_COMMIT=c24bc199b6b8fb94002c0563535ff7b5284794b2

IMPLEMENTATION_FREEZE_COMMIT=69f92f54f6a143940b7687c5e4b8e1931c5fae8d

BFREE_EVIDENCE_FREEZE_COMMIT=a17006164d6fc73de64cb138b12b6a7975750b4e

SCIENTIFIC_EXECUTION_ALLOWED=YES_STAGE_E_BFREE_AINIT_THREE_CELL_DIAGNOSTIC

CUDA_PREFLIGHT_ALLOWED=YES

TRAINING_ALLOWED=YES_EXACT_STAGE_E_BFREE_AINIT_THREE_CELL

EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV

BACKWARD_ALLOWED=YES_TRAINING_ONLY

OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS

CONFIRMATORY_9601_9900_ALLOWED=NO

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP

## Scientific purpose

This execution tests exactly one remaining Stage E ambiguity.

The validated E-BFREE control recovered:

`0.07142508872336398`

of the unrestricted seed-matched P0 gain on average.

This was substantially larger than E-R22 and E-C22, but still only
approximately 7.14 percent of unrestricted recovery.

The present diagnostic asks:

If the learned-B output span is unchanged, but A is initialized from the same
seed's unrestricted final A_theta.weight, does the exact same 20-step QMA
optimization contract recover materially more task benefit?

Only A initialization differs from E-BFREE.

This is a diagnostic intervention, not an independent confirmatory treatment.

## Exact implementation

Execution must use the implementation frozen at:

`69f92f54f6a143940b7687c5e4b8e1931c5fae8d`

Exact implementation files:

1. `src/contramamba/gen5_stage_e_learned_b_plane_ainit_control.py`
2. `scripts/train_reason_router_gen5_stage_e_learned_b_plane_ainit_control.py`
3. `tests/test_reason_router_gen5_stage_e_learned_b_plane_ainit_control.py`

No modification of these files is authorized during execution.

The frozen E-BFREE implementation and prior Stage E implementation must remain
unchanged.

## Exact matrix

Single arm:

`E-BFREE-AINIT`

Exactly three cells:

- seed6201 / E-BFREE-AINIT
- seed6202 / E-BFREE-AINIT
- seed6203 / E-BFREE-AINIT

Pressure:

`P0`

No other scientific cell is authorized.

## Exact source identities

Only the existing unrestricted Phase3A P0 final corrections may supply
A_free and B_free.

### seed6201

Source checkpoint SHA256:

`157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf`

A_free SHA256:

`23796c2fbdaeee6e76d58bf613b85fdf52d25a47c86d98bc9376422478e79469`

B_free SHA256:

`16d51fc2a579a61def0620542ea1a9c78ef13c2dcf35645b87a01626c83403ff`

Frozen execution Q SHA256:

`1cfc7e1b55b68b0b71404f75c4920788331c9fcccb14ed25b40a316973721705`

### seed6202

Source checkpoint SHA256:

`1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214`

A_free SHA256:

`46472cf48fd5973fc2635c99ebc3d0b77798141c359ae5798b1ec9019e1f24f9`

B_free SHA256:

`7a9165f0db7baacdbeccc930ae3d50c5910948931366385007d88bd27f3ffab4`

Frozen execution Q SHA256:

`44b81288f73f605bc12fbab90a51cdb87f421f5cbd6fec7b648dd6621d77ff57`

### seed6203

Source checkpoint SHA256:

`c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770`

A_free SHA256:

`5f47d66270a2ee2900741395e54dd274d4cb12e8359cbc223ea9fa1be07911c8`

B_free SHA256:

`f5d4b05bb91dcee119d7e9b1308ff60ca8a914f882854b503973977cc114b862`

Frozen execution Q SHA256:

`28dd4b583b8019688ca656cc063924e75bd4b2fee9c35fa6680b5570d0953330`

Each source checkpoint must authenticate:

- schema `GEN5_PHASE3A_FINAL_CORRECTION_V1`;
- matching seed;
- arm `G5-C0`;
- pressure `P0`;
- exact parent checkpoint identity;
- exact source file SHA256;
- exact A_free SHA256;
- exact B_free SHA256;
- A_free shape `[2,768]`;
- B_free shape `[24576,2]`;
- A row rank exactly 2;
- B rank exactly 2.

## QR execution identity

The Q derivation is identical to the already validated E-BFREE derivation:

- float64 thin QR;
- positive-diagonal sign canonicalization;
- no SVD;
- no random rotation;
- no basis search.

Exact Q tensor bytes depend on CPU linear-algebra execution state.

Therefore every execution-relevant static verification, CUDA preflight,
single-cell runtime, and matrix runtime must inherit:

`OMP_NUM_THREADS=1`

`MKL_NUM_THREADS=1`

`OPENBLAS_NUM_THREADS=1`

`NUMEXPR_NUM_THREADS=1`

The three frozen Q SHA256 values above are the execution identities previously
established by fresh single-thread Kaggle processes.

Local-development Q hashes are not execution identities.

Any mismatch in Kaggle is a blocker.

## Parameterization

For each seed:

`correction(x) = Q_B M A x`

where:

- Q_B `[24576,2]`, frozen;
- M `[2,2]`, trainable;
- A `[2,768]`, trainable;
- no bias.

Initialization:

- `A = A_free` from the authenticated same-seed unrestricted source;
- `M = 0` exactly.

Exactly two trainable tensors:

- `A_theta.weight`
- `M_theta.weight`

Expected trainable parameter count:

`1540`

## Step-zero firewall

At initialization:

`M = 0`

therefore:

`Q_B M A_free x = 0`

for every x.

Required before scientific execution:

- M exact-zero initialization;
- zero nonzero entries in M;
- exact-zero deterministic correction output;
- A_theta exactly equals authenticated A_free.

No unrestricted final correction amplitude is installed at step zero.

## Exact representability gate

The implementation has statically established that the parameterization can
represent the unrestricted final operator.

For:

`B_free = Q_B R_B`

the exact target parameterization is:

`M_exact = R_B`

`A = A_free`

so:

`Q_B R_B A_free = B_free A_free`

The implementation must continue to validate the factorized operator residual
without materializing a `[24576,768]` matrix.

Required:

`representability_relative <= 1e-12`

The completed local static verification observed:

- seed6201: `9.80749728823e-16`
- seed6202: `6.83592574192e-16`
- seed6203: `6.12125159968e-16`

These local floating-point values are diagnostic, not exact execution hashes.

The tolerance is authoritative.

## Parent checkpoint

Exact parent SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Parent parameters must remain frozen.

Parent fingerprint must be identical before and after every cell.

## Data contract

Use exactly the frozen Phase3A/Stage E data contract.

Training:

- 3360 rows
- 480 source pairs

Dev:

- 840 rows
- 120 source pairs

Split seed:

`16384`

Pressure:

`P0`

Frozen train-order SHA256:

`0be453c8d5d78397e1387f1ec3aac7cc5e83983a9ec0b49107f5b4770d38a71b`

Frozen dev-order SHA256:

`b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25`

Frozen train-encoding SHA256:

`d845a53923db045ed58ad2514e2dbf397b4869c856700db06390f86c6c646f10`

Frozen dev-encoding SHA256:

`e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51`

Confirmatory population `9601..9900` is forbidden.

## Optimization contract

Exactly:

- AdamW
- learning rate `0.001`
- weight decay `0.0001`
- exactly 20 optimizer steps per cell
- gradient clip norm `5.0`
- no scheduler
- final three-way cross entropy only
- no early stopping
- no checkpoint selection
- final fixed step only

Total scientific optimizer steps:

`60`

A and M remain trainable.

Do not freeze A during execution.

Because M is zero at initialization, task-loss gradient to A is zero at the
initial point, but AdamW decoupled weight decay may update A after the first
optimizer step.

This is part of the authorized matched optimizer contract.

## Frozen comparison values

Do not rerun E-BFREE.

Use these validated seed-matched recovery values:

- seed6201: `0.0690419274517625`
- seed6202: `0.0737218360466545`
- seed6203: `0.0715115026716749`

Mean E-BFREE recovery:

`0.07142508872336398`

Frozen unrestricted P0 gains:

- seed6201: `0.4984860420227051`
- seed6202: `0.5020102858543396`
- seed6203: `0.4994615912437439`

## Primary quantities

For each seed:

`gain_AINIT = CE_ZERO - CE_AINIT`

`recovery_AINIT = gain_AINIT / gain_FREE_P0`

Primary descriptive comparison:

`delta_AINIT_MINUS_BFREE = recovery_AINIT - recovery_BFREE`

Report all seedwise values and descriptive mean.

No scientific p-value.

No post-hoc success threshold.

## CUDA preflight

Exactly one bounded CUDA preflight is authorized before the matrix.

Use:

`seed6201 / E-BFREE-AINIT`

The preflight may:

- authenticate runtime/package/kernel identities;
- authenticate exact source/A/B/Q identities;
- instantiate the frozen parent;
- install the AINIT correction;
- verify A_free initialization;
- verify M zero initialization;
- perform scientific-model forward plumbing;
- perform backward plumbing;
- verify finite A/M gradients;
- verify no parent gradients;
- verify fixed-plane geometry.

The preflight must have:

- optimizer constructed: false;
- optimizer step count: 0;
- training executed: false;
- task evaluation executed: false;
- confirmatory data loaded: false;
- scientific conclusion: null.

Preflight is runtime validation only.

It is not scientific evidence and is not collected/imported if it passes.

## GPU topology

Full matrix must use exactly:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

Required:

- exactly two Tesla T4 GPUs;
- capability `(7,5)`;
- worker0 seeds `(6201,6203)`;
- worker1 seed `(6202)`;
- fresh parent model per cell;
- no DDP;
- no shared model;
- no shared optimizer.

## Runtime surface

Execution must preserve the previously authenticated Stage E runtime surface:

- Python `3.12.13`
- NumPy `2.0.2`
- PyTorch `2.10.0+cu128`
- CUDA runtime `12.8`
- Transformers `5.0.0`
- tokenizers `0.22.2`
- kernels `0.10.2`

Model revision:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Model config SHA256:

`784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`

Tokenizer identities:

- tokenizer.json:
  `b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`
- tokenizer_config.json:
  `9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`
- special_tokens_map.json:
  `57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

Kernel build variant:

`torch210-cxx11-cu128-x86_64-linux`

Mamba kernel scientific revision:

`c8ffc584c147878a6eb978ae0e8db4d116c93a8c`

Mamba transport revision:

`a80a7604874b108585feb87096a0c86df2a1e5e3`

Mamba binary SHA256:

`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

causal-conv1d scientific revision:

`f2651e776f66069cdcf842840db637583def1223`

causal-conv1d transport revision:

`02ab414d848bbee389d801f87b24fa536de60273`

causal-conv1d binary SHA256:

`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

Runtime/provisioning mismatch is a blocker.

## Required execution artifacts

Each cell must preserve at minimum:

- seed;
- arm;
- source checkpoint SHA;
- A_free SHA;
- B_free SHA;
- Q SHA;
- train/dev identities;
- training losses;
- step0 loss;
- post-step20 matched-RNG loss;
- gradient norms;
- optimizer-step count;
- dev CE;
- dev accuracy;
- gain;
- unrestricted seed-matched gain;
- recovery;
- frozen BFREE recovery;
- delta AINIT-minus-BFREE;
- fixed-plane geometry;
- final A hash;
- final M hash;
- parent fingerprint before/after;
- execution/runtime provenance.

Execution artifacts must retain:

`scientific_conclusion = null`

## Prohibited expansion

Not authorized:

- E-BFREE rerun;
- E-R22 rerun;
- E-C22 rerun;
- unrestricted rerun;
- random-plane control;
- new seed;
- pressure expansion;
- layer sweep;
- rank sweep;
- learning-rate sweep;
- weight-decay sweep;
- optimizer sweep;
- step-count sweep;
- freezing A;
- nonzero M initialization;
- final-B amplitude initialization;
- QR/SVD alternatives;
- task-data changes;
- tokenizer changes;
- model changes;
- confirmatory evaluation;
- scientific p-values;
- implementation edits during execution.

If an implementation defect is exposed, stop.

Do not patch the frozen implementation in place.

## Interpretation boundary

Keep separate:

1. code/runtime correctness;
2. execution success;
3. artifact/provenance validity;
4. scientific interpretation.

Scientific interpretation is allowed only after successful collect/import and
artifact validation.

## Prospective interpretation

If E-BFREE-AINIT is consistently and materially above the frozen E-BFREE
recovery across matched seeds, then relearning the A/read-side geometry is
implicated as a major contributor to the remaining fixed-plane optimization
bottleneck.

If E-BFREE-AINIT remains near E-BFREE, then A restart initialization is not a
sufficient explanation, and the remaining ambiguity shifts toward QMA
scale/gauge conditioning or short-horizon optimization dynamics.

If outcomes are materially heterogeneous across seeds, preserve that
heterogeneity.

Neither a high nor low AINIT result establishes a general claim outside this
frozen optimization contract.
