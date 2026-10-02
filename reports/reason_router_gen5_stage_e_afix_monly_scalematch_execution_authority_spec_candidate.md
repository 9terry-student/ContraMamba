# Gen5 Stage E AFIX-MONLY SCALEMATCH Diagnostic
# Execution Authority

## Authority identities

DESIGN_IMPLEMENTATION_AUTHORITY_COMMIT=1ea8d156c86a7f8ec6b9ad638632e46b54b0de38

IMPLEMENTATION_FREEZE_COMMIT=498a938e9321b7eb47fb8c222f0c25d0cdb1675e

STATIC_SCALE_MODE_FREEZE_COMMIT=8abab776801eb106ddd97c39883a99f83257b833

AFIX_MONLY_EVIDENCE_FREEZE_COMMIT=d3d0f86fca9111ab19944f020c1efbb3d6b37d0a

SCIENTIFIC_EXECUTION_ALLOWED=YES_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_THREE_CELL_DIAGNOSTIC

CUDA_PREFLIGHT_ALLOWED=YES

TRAINING_ALLOWED=YES_EXACT_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_THREE_CELL

EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV

BACKWARD_ALLOWED=YES_TRAINING_ONLY

OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS_M_ONLY_SCALEMATCH_LR

CONFIRMATORY_9601_9900_ALLOWED=NO

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP

## Scientific purpose

Validated AFIX-MONLY execution with:

- exact learned-B output plane Q_B;
- exact seed-matched unrestricted A_free, frozen;
- zero-initialized 2x2 M;
- only four trainable M parameters;
- AdamW learning rate 0.001;
- exactly 20 optimizer steps;

recovered only:

`0.0629884423403136`

of the frozen unrestricted P0 gain on average.

Subsequent frozen static analysis established:

- mean observed `||M||/||R_B|| = 0.0122257826760901`;
- mean nominal 20-step M-budget / `||R_B|| = 0.0122572018249573`;
- observed `||M||` matched the nominal 4-coordinate Adam movement scale
  `0.04` to approximately 99.7 percent;
- unrestricted B has 49152 trainable coordinates;
- the corresponding nominal unrestricted-B 20-step Frobenius movement scale
  is `4.43405006737633`;
- mean nominal unrestricted-B budget / `||R_B|| = 1.35872616444493`;
- AFIX-MONLY M was numerically almost rank 1;
- mean `M_sv2/M_sv1 = 1.51302665397091e-05`;
- mean source `R_sv2/R_sv1 = 0.419079628207686`;
- mean first source-mode acquisition ratio was
  `0.0123036749792163`;
- mean second source-mode acquisition ratio was
  `-0.000139581831174714`.

The previous absolute unrestricted-B versus QMA recovery gap is therefore
confounded by parameterization-dependent Adam step scale.

The present experiment removes exactly that confound.

It changes exactly one scientific quantity relative to validated AFIX-MONLY:

`M learning rate`.

No learning-rate sweep is authorized.

## Exact implementation

Execution must use the implementation frozen at:

`498a938e9321b7eb47fb8c222f0c25d0cdb1675e`

Exact implementation files:

1. `src/contramamba/gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py`
2. `scripts/train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py`
3. `tests/test_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py`

No modification of these files is authorized during execution.

The underlying frozen AFIX-MONLY implementation and all prior Stage E,
Phase3A, Phase2, data, tokenizer, model, and evidence files must remain
unchanged.

## Exact matrix

Single arm:

`E-BFREE-AFIX-MONLY-SCALEMATCH`

Exactly three cells:

- seed6201 / E-BFREE-AFIX-MONLY-SCALEMATCH
- seed6202 / E-BFREE-AFIX-MONLY-SCALEMATCH
- seed6203 / E-BFREE-AFIX-MONLY-SCALEMATCH

Pressure:

`P0`

No other scientific cell is authorized.

No comparison arm may be rerun.

## Exact scale-match derivation

Original unrestricted-B learning rate:

`0.001`

Unrestricted-B trainable coordinate count:

`24576 * 2 = 49152`

M trainable coordinate count:

`2 * 2 = 4`

Dimension-normalized scale factor:

`sqrt(49152 / 4) = sqrt(12288)`

Exact frozen scale factor:

`110.85125168440814`

Exact authorized M learning rate:

`0.001 * sqrt(49152 / 4)`

which equals:

`0.11085125168440814`

This learning rate is determined only from:

- the frozen original unrestricted-B learning rate;
- unrestricted-B parameterization dimension;
- M parameterization dimension.

It must not be changed based on:

- R_B norm;
- AFIX-MONLY recovery;
- dev loss;
- runtime loss trajectory;
- any task outcome.

No second learning rate is authorized.

## Exact source identities

Only the same unrestricted Phase3A P0 final corrections used by E-BFREE,
E-BFREE-AINIT, and AFIX-MONLY may supply A_free and B_free.

### seed6201

Source checkpoint SHA256:

`157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf`

A_free SHA256:

`23796c2fbdaeee6e76d58bf613b85fdf52d25a47c86d98bc9376422478e79469`

B_free SHA256:

`16d51fc2a579a61def0620542ea1a9c78ef13c2dcf35645b87a01626c83403ff`

Frozen single-thread Kaggle Q SHA256:

`1cfc7e1b55b68b0b71404f75c4920788331c9fcccb14ed25b40a316973721705`

Frozen single-thread Kaggle R SHA256:

`3638d26696668af13c7337500c6c114265cded1e2b693987c2c95f1e5af26198`

### seed6202

Source checkpoint SHA256:

`1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214`

A_free SHA256:

`46472cf48fd5973fc2635c99ebc3d0b77798141c359ae5798b1ec9019e1f24f9`

B_free SHA256:

`7a9165f0db7baacdbeccc930ae3d50c5910948931366385007d88bd27f3ffab4`

Frozen single-thread Kaggle Q SHA256:

`44b81288f73f605bc12fbab90a51cdb87f421f5cbd6fec7b648dd6621d77ff57`

Frozen single-thread Kaggle R SHA256:

`26664c2555eebb3f536217b9d2950a600015cf630a656ede305150ac8c768adf`

### seed6203

Source checkpoint SHA256:

`c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770`

A_free SHA256:

`5f47d66270a2ee2900741395e54dd274d4cb12e8359cbc223ea9fa1be07911c8`

B_free SHA256:

`f5d4b05bb91dcee119d7e9b1308ff60ca8a914f882854b503973977cc114b862`

Frozen single-thread Kaggle Q SHA256:

`28dd4b583b8019688ca656cc063924e75bd4b2fee9c35fa6680b5570d0953330`

Frozen single-thread Kaggle R SHA256:

`af8c582b0fb5636df944d77af6182c8097dc0cd49cbe8ec721906efc43202c8a`

Each source checkpoint must authenticate:

- schema `GEN5_PHASE3A_FINAL_CORRECTION_V1`;
- matching seed;
- arm `G5-C0`;
- pressure `P0`;
- exact parent checkpoint identity;
- exact source-file SHA256;
- exact A_free SHA256;
- exact B_free SHA256;
- A_free shape `[2,768]`;
- B_free shape `[24576,2]`;
- A row rank exactly 2;
- B rank exactly 2.

## QR execution identity

The Q/R derivation remains identical to the validated BFREE, AINIT, and
AFIX-MONLY Kaggle execution:

- float64 thin QR;
- positive-diagonal sign canonicalization;
- no SVD;
- no random rotation;
- no basis search.

Every execution-relevant static/provisioning verification, CUDA preflight,
single-cell runtime, and matrix runtime must inherit:

`OMP_NUM_THREADS=1`

`MKL_NUM_THREADS=1`

`OPENBLAS_NUM_THREADS=1`

`NUMEXPR_NUM_THREADS=1`

The frozen Kaggle Q/R SHA256 values above are the expected execution
identities.

Local-development Q/R byte hashes are not execution identities.

Any Kaggle Q/R mismatch is a blocker.

## Parameterization

For each seed:

`correction(x) = Q_B M A_free x`

where:

- Q_B `[24576,2]`, frozen;
- A `[2,768]`, copied exactly from authenticated same-seed A_free and frozen;
- M `[2,2]`, trainable;
- no bias.

Initialization:

- `A = A_free` exactly;
- `M = 0` exactly.

Exactly one trainable tensor:

`M_theta.weight`

Expected trainable parameter count:

`4`

Required throughout:

- `A_theta.weight.requires_grad = False`;
- `M_theta.weight.requires_grad = True`;
- optimizer parameter list contains only `M_theta.weight`;
- no parent parameter is trainable;
- A SHA256 remains exactly the authenticated source A SHA256;
- A gradient remains `None`.

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
- A exactly equals authenticated A_free;
- A is frozen;
- M is the only trainable tensor.

No unrestricted correction amplitude is installed at step zero.

## Exact representability gate

For:

`B_free = Q_B R_B`

the exact unrestricted source correction is representable at:

`M_exact = R_B`

with frozen:

`A = A_free`.

Thus:

`Q_B R_B A_free = B_free A_free`.

Require factorized float64 representability:

`representability_relative <= 1e-12`

without materializing the full `[24576,768]` operator.

This is a capacity gate only and does not imply optimization success.

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

- AdamW;
- optimizer parameterization `M_theta.weight` only;
- learning rate `0.11085125168440814`;
- weight decay `0.0001`;
- exactly 20 optimizer steps per cell;
- gradient clip norm `5.0`;
- no scheduler;
- final three-way cross entropy only;
- no early stopping;
- no checkpoint selection;
- final fixed step only.

Total scientific optimizer steps:

`60`

No other optimization quantity may change.

## Frozen comparison values

Do not rerun AFIX-MONLY.

Use validated AFIX-MONLY recovery:

- seed6201: `0.063109275770733`
- seed6202: `0.0610331932890345`
- seed6203: `0.0648228579611734`

Mean:

`0.0629884423403136`

Secondary frozen references:

E-BFREE-AINIT recovery:

- seed6201: `0.135931570756102`
- seed6202: `0.124898380318522`
- seed6203: `0.139898741881301`

Mean:

`0.133576230985308`

E-BFREE mean recovery:

`0.07142508872336398`

Frozen unrestricted P0 gains:

- seed6201: `0.4984860420227051`
- seed6202: `0.5020102858543396`
- seed6203: `0.4994615912437439`

## Primary quantities

For each seed:

`gain_SCALEMATCH = CE_ZERO - CE_SCALEMATCH`

`recovery_SCALEMATCH = gain_SCALEMATCH / gain_FREE_P0`

Primary descriptive comparison:

`delta_SCALEMATCH_MINUS_AFIX_MONLY = recovery_SCALEMATCH - recovery_AFIX_MONLY`

Report all seedwise values and descriptive mean.

Also preserve final M and fixed-plane geometry so that post-import static
analysis can measure:

- `||M||/||R_B||`;
- `||M-R_B||/||R_B||`;
- singular values;
- second-mode acquisition;
- effective-operator fidelity.

No scientific p-value.

No post-hoc success threshold.

## CUDA preflight

Exactly one bounded CUDA preflight is authorized before the matrix.

Use:

`seed6201 / E-BFREE-AFIX-MONLY-SCALEMATCH`

The preflight may:

- authenticate runtime/package/kernel identities;
- authenticate exact source/A/B/Q/R identities;
- instantiate the frozen parent;
- install the SCALEMATCH correction;
- verify A_free exact initialization and frozen state;
- verify M exact-zero initialization;
- perform model forward plumbing;
- perform backward plumbing;
- verify `A.grad is None`;
- verify finite M gradient;
- verify no non-M gradients;
- verify A SHA unchanged;
- verify fixed-plane geometry.

The preflight must have:

- optimizer constructed: false;
- optimizer step count: 0;
- training executed: false;
- task evaluation executed: false;
- confirmatory data loaded: false;
- scientific conclusion: null.

Preflight is runtime validation only.

It is not scientific evidence and must not be collected/imported if it passes.

If this authorized preflight fails for any reason, stop. A retry requires a
new explicit recovery amendment.

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

Execution must preserve:

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
- R SHA;
- train/dev identities;
- A frozen status;
- A source SHA;
- final A SHA;
- A gradient-present count;
- optimizer parameterization `M_THETA_WEIGHT_ONLY`;
- learning rate;
- scale-match factor;
- unrestricted-B coordinate-count reference;
- M coordinate count;
- training losses;
- step0 loss;
- post-step20 matched-RNG loss;
- M gradient norms;
- optimizer-step count;
- dev CE;
- dev accuracy;
- gain;
- unrestricted seed-matched gain;
- recovery;
- frozen AFIX-MONLY recovery;
- delta SCALEMATCH-minus-AFIX-MONLY;
- fixed-plane geometry;
- final M hash;
- parent fingerprint before/after;
- execution/runtime provenance.

Required invariants:

- final A SHA equals source A SHA;
- `A_grad_present_count = 0`;
- trainable tensor count = 1;
- trainable parameter count = 4;
- learning rate equals `0.11085125168440814`;
- scale-match factor equals `110.85125168440814`.

Execution artifacts must retain:

`scientific_conclusion = null`

## Prohibited expansion

Not authorized:

- AFIX-MONLY rerun;
- AINIT rerun;
- BFREE rerun;
- R22 rerun;
- C22 rerun;
- unrestricted rerun;
- random-plane control;
- new seed;
- pressure expansion;
- layer sweep;
- rank sweep;
- additional learning-rate value;
- learning-rate sweep;
- weight-decay sweep;
- optimizer sweep;
- step-count/horizon sweep;
- A training;
- nonzero M initialization;
- R_B initialization of M;
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
independent artifact validation.

## Prospective interpretation

If SCALEMATCH substantially increases recovery relative to frozen AFIX-MONLY
and simultaneously moves `||M||/||R_B||` toward order 1 with emergence of the
second singular mode, parameterization-dependent optimizer scale is supported
as a major cause of the previous absolute fixed-plane recovery failure.

If SCALEMATCH restores M norm but the second singular mode remains absent,
the scale confound is real but a separate anisotropic mode-conditioning
bottleneck remains.

If SCALEMATCH fails to improve recovery despite corrected movement scale, the
explanation shifts toward path-dependent/core-conditioning effects rather than
simple step budget.

No result from these three descriptive seeds establishes a universal claim
outside this frozen Stage E contract.
