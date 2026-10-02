# Gen5 Stage E Learned-B-Plane A-Initialization Diagnostic Validated Evidence Report

## Status

VALIDATED_EVIDENCE

This report records the validated E-BFREE-AINIT diagnostic result for Gen5 Stage E.

It is an evidence interpretation report, not a new execution authority.

## Execution identity

Execution commit:

`51bfc4d1d2e1d3fae970e7c09a75cf3aa72cf488`

Implementation freeze:

`69f92f54f6a143940b7687c5e4b8e1931c5fae8d`

Run:

`gen5-stagee-bfree-ainit-three-cell-51bfc4d-r1`

Imported handoff ZIP SHA256:

`ecbded9ef0545cc7f9a623d8eabbcc850610bf7637b42fb6c91f4133a7fe5f3e`

Pinned command SHA256:

`59102352924cfb63ea7e1166390f506efb42e71bcfc3263c1d0999d620d30b9f`

Run log SHA256:

`298718e7adfbc413a015aafd759938eb81feaea4a9e1d2b5bcde726ec478f46d`

Run meta SHA256:

`0fa28236e5628cb8680088fbbd128772de640ae667ede64bb17b79c61dd73485`

The collector reported 11 files and EXIT_CODE=0.

`cm import` validated and copied all 11 files.

An independent local artifact/provenance validator then passed all 11 imported
files, including report/provenance/checkpoint SHA linkages, tensor hashes,
runtime/data identities, optimizer-step counts, metric recomputation, and
confirmatory-set firewalls.

## Frozen execution contract

Arm:

`E-BFREE-AINIT`

Pressure:

`P0`

Seeds:

- 6201
- 6202
- 6203

GPU topology:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

Worker allocation:

- GPU worker 0: seeds 6201 and 6203
- GPU worker 1: seed 6202

Each cell used exactly 20 AdamW optimizer steps.

Total optimizer steps:

`60`

Trainable parameter count per cell:

`1540`

Initialization:

- Q_B fixed to the same seed-matched unrestricted Phase3A P0 final-B span used
  by E-BFREE;
- A initialized exactly from the same seed's unrestricted Phase3A P0 final
  `A_theta.weight`;
- M initialized exactly to zero;
- A and M trainable.

Because M was exactly zero, the initial correction output was exactly zero.

The parameterization statically represented the unrestricted final operator to
numerical precision for all three seeds.

Confirmatory IDs 9601--9900 were not loaded.

No scientific p-values were computed.

## Provenance and execution validation

The run passed:

- exact execution HEAD authentication;
- exact implementation-freeze authentication;
- exact execution-authority authentication;
- exact parent checkpoint SHA256 authentication;
- exact model/tokenizer snapshot authentication;
- exact frozen CUDA-kernel binary authentication;
- exact seed-specific source checkpoint authentication;
- exact seed-specific A_free, B_free, and Q identities;
- single-thread QR execution identity authentication;
- exact-zero M initialization;
- exact-zero step-0 correction output;
- factorized exact-operator representability check;
- two Tesla T4 topology authentication;
- three-cell matrix execution;
- collector validation;
- local import validation;
- independent 11-file artifact/provenance validation.

All three cells reported:

- effective output rank = 2;
- effective operator rank = 2;
- parent parameter fingerprint unchanged;
- fixed-plane residual within the frozen tolerance;
- exactly 20 optimizer steps;
- task evaluation executed;
- confirmatory set not loaded.

Therefore code/runtime execution success and artifact/provenance validity are
established for this matrix.

## Primary result

Frozen E-BFREE recovery:

- seed6201: `0.0690419274517625`
- seed6202: `0.0737218360466545`
- seed6203: `0.0715115026716749`
- mean: `0.07142508872336398`

Validated E-BFREE-AINIT result:

| Seed | Dev CE | Dev accuracy | Gain vs ZERO | Recovery vs unrestricted P0 | Delta vs E-BFREE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6201 | 1.27320551872253 | 0.277380952380952 | 0.0677599906921387 | 0.135931570756102 | 0.0668896433043397 |
| 6202 | 1.27826523780823 | 0.277380952380952 | 0.0627002716064453 | 0.124898380318522 | 0.0511765442718676 |
| 6203 | 1.27109146118164 | 0.278571428571429 | 0.0698740482330322 | 0.139898741881301 | 0.0683872392096258 |

Mean E-BFREE-AINIT recovery:

`0.133576230985308`

Mean delta versus E-BFREE:

`0.0621511422619444`

Ratio of mean E-BFREE-AINIT recovery to mean E-BFREE recovery:

approximately `1.87016x`.

Thus supplying the unrestricted seed-matched A initialization increased
short-horizon recovery in the same direction for all three seeds.

However, E-BFREE-AINIT still recovered only approximately `13.36%` of the
unrestricted seed-matched P0 gain.

Approximately `86.64%` of unrestricted gain remained unrecovered.

## Interpretation

The E-BFREE-AINIT diagnostic resolves an important part of the ambiguity left
by E-BFREE.

Under the same learned-B output plane, QMA parameterization, optimizer,
20-step horizon, P0 data, and matched seed contract, replacing the random
restart A initialization with the seed-matched unrestricted final A increased
recovery from approximately 7.14% to approximately 13.36%.

The increase was directionally consistent across all three seeds.

Therefore relearning the input/read-side A geometry from a restart
initialization was a real contributor to the E-BFREE short-horizon optimization
bottleneck.

The supported result is:

`READ_SIDE_A_REACQUISITION_MATERIALLY_CONTRIBUTES_TO_THE_FIXED_PLANE_SHORT_HORIZON_OPTIMIZATION_BOTTLENECK`

However, the result does not establish that A reacquisition is the dominant or
sufficient explanation.

Even with A initialized at the unrestricted seed-matched solution, the fixed
learned-B-plane QMA run recovered only approximately 13.36% of unrestricted
gain.

Therefore a large residual optimization-accessibility gap remains.

The result rules out the simplest explanation that the low E-BFREE recovery was
primarily an artifact of having to rediscover A from a random restart.

The remaining ambiguity is now narrower:

1. whether A drifts away from its useful source geometry during joint A/M
   optimization;
2. whether learning the 2x2 core M from zero is poorly scaled or conditioned
   relative to the source target R_B;
3. whether the 20-step matched optimization horizon is intrinsically too short
   for the QMA gauge even when the correct input/read-side geometry is supplied.

No inferential claim is made from the three-seed descriptive matrix.

## Stage E status

Stage E remains scientifically open, but the next step should not be another
GPU sweep or a broader intervention matrix.

The highest-value next action is a read-only static decomposition of the
already-frozen E-BFREE and E-BFREE-AINIT checkpoints against the unrestricted
source factorization.

For each seed, the static analysis should compare:

- final A versus source A_free;
- final M versus source QR target R_B;
- A row-space affinity / principal geometry;
- M Frobenius norm ratio and relative error to R_B;
- vectorized M cosine to R_B;
- effective low-rank operator error between Q M A and B_free A_free using a
  factorized computation that does not materialize the full 24576x768
  operator;
- E-BFREE versus E-BFREE-AINIT on the same quantities.

This analysis requires no new training, no CUDA execution, no confirmatory
data, and no new scientific p-value.

Only if that static decomposition remains ambiguous should another bounded
diagnostic be authorized.

A possible later diagnostic, if required, is to hold A fixed at A_free and
train only the 2x2 M core from zero under the same 20-step contract. That
experiment is not authorized by this report.

## Gen5 status

The broader Gen5 synthesis remains supported:

`NATIVE_CAUSAL_IMPORTANCE_DOES_NOT_IMPLY_OPTIMIZATION_PRIVILEGE`

The Stage E evidence now additionally shows that learned output-plane
orientation and read-side initialization both affect short-horizon optimizer
accessibility, while neither alone explains the unrestricted solution.

Gen5 should not yet be declared experimentally closed.

This report authorizes no new training or evaluation.
