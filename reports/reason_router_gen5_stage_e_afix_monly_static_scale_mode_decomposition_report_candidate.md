# Gen5 Stage E AFIX-MONLY Static Core/Mode Decomposition Report Candidate

## Status

VALIDATED_STATIC_ANALYSIS_CANDIDATE

This report records read-only static analysis performed after the validated
E-BFREE-AFIX-MONLY evidence freeze.

It is not an execution authority and authorizes no CUDA, training, evaluation,
backward pass, optimizer construction, optimizer step, or confirmatory-data
access.

## Evidence identity

AFIX-MONLY validated evidence freeze commit:

`d3d0f86fca9111ab19944f020c1efbb3d6b37d0a`

Validated execution run:

`gen5-stagee-bfree-afix-monly-three-cell-094fecb-r1`

Seeds:

`6201,6202,6203`

Pressure:

`P0`

AFIX-MONLY trainable parameterization:

- `A = A_free`, frozen;
- `Q = Q_B`, frozen;
- `M in R^(2x2)`, zero initialized;
- only `M_theta.weight` trainable;
- exactly 4 trainable parameters;
- AdamW learning rate `0.001`;
- exactly 20 optimizer steps per seed.

No CUDA or task evaluation was executed by the static analyses.

Confirmatory IDs `9601..9900` were not loaded.

No scientific p-values were computed.

## QR identity policy

The unrestricted source tensor `B_free` is the exact cross-environment
identity anchor.

For each seed, the source `B_free` byte identity was authenticated against its
frozen SHA256. A positive-diagonal thin float64 QR decomposition

`B_free = Q_B R_B`

was then recomputed locally.

The local QR reconstruction relative error was approximately `1e-14`, below
the frozen `1e-12` numerical tolerance.

Cross-environment Q/R byte equality is not required because LAPACK/BLAS
backends can produce last-bit differences while preserving the same numerical
canonical QR decomposition. The Kaggle execution Q/R SHA256 values remain
recorded in the execution artifacts as provenance identities.

## Why AFIX-MONLY removes the previous gauge ambiguity

In AFIX-MONLY both `Q_B` and `A_free` are frozen.

The learned operator is

`Q_B M A_free`.

The unrestricted source operator is

`B_free A_free = Q_B R_B A_free`.

Therefore the exact source-equivalent core target is unambiguous:

`M_star = R_B`.

Unlike trainable-A BFREE/AINIT analyses, no 2x2 gauge alignment is needed to
define the target core.

## Static core decomposition

### seed6201

- recovery: `0.063109275770733`
- `||M-R||_F / ||R||_F`: `0.989388772239442`
- cosine(`M`,`R`): `0.853965156666195`
- `||M||_F / ||R||_F`: `0.0124506699023663`
- best scalar coefficient alpha in `M ~= alpha R`: `0.0106324382737733`
- residual after best scalar fit, relative to `||R||_F`: `0.00647845949065753`
- effective operator relative error: `0.988357931088204`
- effective operator cosine: `0.894973151476381`
- effective operator norm ratio: `0.0130273811432382`
- singular values of `R_B`: `[2.9391785434223046, 1.281299264235051]`
- singular values of `M_final`: `[0.03992085205912446, 2.515176167465647e-07]`
- condition number of `R_B`: `2.29390480855152`
- condition number of `M_final`: `158719.904297398`
- step0 loss: `1.26252424716949`
- matched-RNG post-step20 loss: `1.23746192455292`
- step0 gradient norm: `0.338748335838318`
- step19 gradient norm: `0.32720810174942`

### seed6202

- recovery: `0.0610331932890345`
- `||M-R||_F / ||R||_F`: `0.989807032614847`
- cosine(`M`,`R`): `0.850290657619025`
- `||M||_F / ||R||_F`: `0.0120113698825317`
- best scalar coefficient alpha: `0.0102131555963233`
- scalar-aligned relative residual: `0.00632174495058771`
- effective operator relative error: `0.989254491823089`
- effective operator cosine: `0.87557834440805`
- effective operator norm ratio: `0.012292825713047`
- singular values of `R_B`: `[3.1549086835969056, 1.0344348571691084]`
- singular values of `M_final`: `[0.03987974551066894, 1.2284126155197166e-06]`
- condition number of `R_B`: `3.04988628499121`
- condition number of `M_final`: `32464.4545381819`
- step0 loss: `1.25624358654022`
- matched-RNG post-step20 loss: `1.23191499710083`
- step0 gradient norm: `0.343288242816925`
- step19 gradient norm: `0.332473516464233`

### seed6203

- recovery: `0.0648228579611734`
- `||M-R||_F / ||R||_F`: `0.98964931646738`
- cosine(`M`,`R`): `0.849075756331557`
- `||M||_F / ||R||_F`: `0.0122153082433721`
- best scalar coefficient alpha: `0.0103717220855643`
- scalar-aligned relative residual: `0.006452994379388`
- effective operator relative error: `0.988475900631696`
- effective operator cosine: `0.89590199153537`
- effective operator norm ratio: `0.0128816154586429`
- singular values of `R_B`: `[2.9285651666907, 1.445012411374165]`
- singular values of `M_final`: `[0.03989108011586855, 3.305962282854942e-07]`
- condition number of `R_B`: `2.02667128921455`
- condition number of `M_final`: `120664.050896006`
- step0 loss: `1.26229310035706`
- matched-RNG post-step20 loss: `1.23652720451355`
- step0 gradient norm: `0.342848867177963`
- step19 gradient norm: `0.327956885099411`

### Means

- mean recovery: `0.0629884423403136`
- mean `M` relative error: `0.989615040440556`
- mean cosine(`M`,`R`): `0.851110523538926`
- mean `||M||/||R||`: `0.0122257826760901`
- mean best scalar alpha: `0.0104057719852203`
- mean scalar-aligned residual relative to `||R||`: `0.00641773294021108`
- mean effective operator relative error: `0.988696107847663`
- mean effective operator cosine: `0.888817829139934`
- mean effective operator norm ratio: `0.0127339407716427`

Frozen comparison values:

- BFREE mean M-only relative error at exact `A_free`: `0.988579644897`
- AINIT mean M-only relative error at exact `A_free`: `0.988275293096`

## Source-mode decomposition

Under Adam-like near-coordinate-normalized updates, a diagnostic nominal
Frobenius movement scale after T steps is

`T * learning_rate * sqrt(number_of_coordinates)`.

This quantity is a diagnostic scale, not a hard optimizer bound.

For AFIX-MONLY:

- parameter coordinates: `4`
- learning rate: `0.001`
- steps: `20`
- nominal 20-step Frobenius scale: `0.04`

For unrestricted B:

- parameter coordinates: `24576 * 2 = 49152`
- learning rate: `0.001`
- steps: `20`
- nominal 20-step Frobenius scale: `4.43405006737633`

Observed AFIX-MONLY final M norms:

- seed6201: `0.0399208520599168`
- seed6202: `0.0398797455295883`
- seed6203: `0.0398910801172385`

Mean observed `||M|| / 0.04`:

`0.997430647556196`

Thus the observed M norm almost exactly saturates this diagnostic 20-step
coordinate-normalized movement scale.

The exact source target `R_B` norms are:

- seed6201: `3.20632161746811`
- seed6202: `3.32016630239535`
- seed6203: `3.2656629961739`

Mean observed `||M||/||R||`:

`0.0122257826760901`

Mean nominal M budget / `||R||`:

`0.0122572018249573`

Mean nominal unrestricted-B budget / `||R||`:

`1.35872616444493`

The observed AFIX-MONLY scale ratio therefore closely matches the scale
predicted by its parameter-coordinate count, learning rate, and horizon.

## Singular-mode acquisition

The target `R_B` is well-conditioned rank 2.

Mean target singular-value ratio:

`R_sv2 / R_sv1 = 0.419079628207686`

The learned AFIX-MONLY M is numerically almost rank 1.

Mean learned singular-value ratio:

`M_sv2 / M_sv1 = 1.51302665397091e-05`

In the singular basis of `R_B`, the mean acquisition ratios were:

- first target mode: `0.0123036749792163`
- second target mode: `-0.000139581831174714`

Mean off-diagonal core magnitude relative to `||R_B||`:

`0.00457244025630827`

The top right singular vector of M aligned almost exactly with the first right
singular vector of R_B in every seed:

- seed6201: `0.999966351660398`
- seed6202: `0.999790842889793`
- seed6203: `0.999997577790527`

Therefore the learned M does not wander arbitrarily in core space. It
selectively acquires the leading source-core mode while leaving the second
source-core mode nearly absent.

## Interpretation

The previous absolute comparison between unrestricted B optimization and the
fixed-plane QMA controls contained an optimizer-scale confound.

Using the same AdamW learning rate and the same 20-step horizon did not provide
comparable effective parameter-space movement budgets to:

- unrestricted `B in R^(24576x2)`, with 49152 coordinates; and
- fixed-plane `M in R^(2x2)`, with 4 coordinates.

Under the observed near-coordinate-normalized Adam behavior, the nominal
Frobenius movement scale differs by

`sqrt(49152 / 4) = sqrt(12288) ~= 110.85125168440814`.

The AFIX-MONLY final M norm quantitatively matches the small-parameter nominal
movement scale, while the exact target R_B norm is about eighty times larger.

The residual failure is not explained by representation capacity:

`Q_B R_B A_free = B_free A_free`

up to numerical QR reconstruction error. The fixed learned-B plane plus
`A_free` can exactly represent the unrestricted rank-2 source operator.

The residual failure is also not a simple arbitrary-direction failure.
M develops substantial directional alignment with R_B, and its top right
singular direction is almost exactly the leading R_B singular direction.

The strongest current mechanism is therefore:

1. parameterization-dependent Adam step scale makes the 4-parameter M control
   severely under-scaled relative to unrestricted B under the nominally
   "matched" learning-rate/horizon contract;
2. within that restricted movement budget, optimization is strongly
   anisotropic and almost exclusively acquires the leading R_B singular mode;
3. the second target mode remains essentially absent after 20 steps;
4. A trainability in AINIT can provide useful co-adaptation, but it does not
   remove the underlying M-core scale/mode-acquisition limitation.

This means the earlier statement that low absolute BFREE recovery by itself
showed that learned-B plane orientation could not be a major cause must be
qualified. The absolute unrestricted-vs-QMA recovery gap is confounded by the
parameterization-dependent optimizer movement budget.

The following within-QMA comparisons remain valid:

- learned-B plane versus R22/C22 fixed-plane controls;
- BFREE versus AINIT;
- AINIT versus AFIX-MONLY.

## Supported conclusions

`GEN5_STAGE_E_ABSOLUTE_FREE_VS_QMA_RECOVERY_GAP_IS_CONFOUNDED_BY_PARAMETERIZATION_DEPENDENT_ADAM_STEP_SCALE`

`GEN5_STAGE_E_AFIX_MONLY_CORE_ACQUISITION_IS_SCALE_LIMITED_AND_STRONGLY_FIRST_MODE_DOMINATED`

`GEN5_STAGE_E_FIXED_LEARNED_B_PLANE_HAS_EXACT_SOURCE_OPERATOR_REPRESENTATION_CAPACITY_WHEN_Q_B_AND_A_FREE_ARE_PROVIDED`

These conclusions are restricted to the frozen Stage E P0, rank-2, 20-step,
AdamW contract.

## Next discriminating control

Do not run a learning-rate sweep.

Run exactly one dimension-normalized step-budget control:

`E-BFREE-AFIX-MONLY-SCALEMATCH`

Keep fixed:

- same seeds `6201,6202,6203`;
- same P0 train/dev split;
- same parent checkpoint;
- same exact seed-matched `Q_B`;
- same exact seed-matched `A_free`, frozen;
- M exact zero initialization;
- only M trainable;
- AdamW;
- weight decay `0.0001`;
- gradient clipping `5.0`;
- exactly 20 optimizer steps;
- same final-only evaluation;
- no confirmatory IDs.

Change exactly one quantity:

`learning_rate_M = 0.001 * sqrt(49152/4)`

which is

`0.11085125168440814`.

This value is determined from parameterization dimensions and the original
unrestricted-B learning rate. It is not fitted to the observed target R_B norm
or to task outcomes.

Primary questions:

1. Does scale matching substantially increase recovery relative to AFIX-MONLY?
2. Does `||M||/||R_B||` approach order 1 rather than approximately 0.012?
3. Does the second singular mode emerge, reducing the extreme condition number?
4. Does M approach R_B in relative error and effective-operator error?

Interpretation:

- large recovery plus rank-2/core recovery would establish optimizer-scale
  mismatch as a major cause of the previous absolute fixed-plane failure;
- norm recovery without second-mode recovery would preserve a separate
  anisotropic mode-conditioning bottleneck;
- failure despite matched scale would shift the explanation toward
  path-dependent/core-conditioning effects rather than simple step budget.

This report authorizes no implementation or execution.
