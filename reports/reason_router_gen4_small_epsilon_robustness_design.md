# ContraMamba Gen4 Experiment 4 — Small-Epsilon Robustness Design

## Status

`EXPERIMENT_4_DESIGN_ONLY`

This document freezes the exact small-epsilon robustness question and execution boundary.
It does not authorize interpretation of unobserved smaller-epsilon responses before raw
artifacts are frozen.

No training, backward pass, task-head evaluation, AVeriTeC steering rescue, layer search,
plane search, token search, epsilon optimization, or post-hoc epsilon addition is allowed.

## 1. Scientific purpose

Experiment 4 addresses only the numerical-robustness objection:

> The frozen five-plane internal-Q geometry may be an artifact of the single
> finite-difference scale epsilon = 0.025.

This is a robustness study of the previously frozen synthetic internal-Q geometry.
It is not a rescue of Experiment 3 steering.

Experiment 3 remains frozen as:

`AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED`

with zero prediction discordances (`C = D = 0`).

## 2. Frozen reference evidence

Research-program order is frozen by:

`reports/reason_router_gen4_post_synthesis_research_program.md`

The existing finite-epsilon reference run is:

`g4k-finite-epsilon-five-plane-decomposition-xg1-2401-2700-e27dbbd-retry1`

Reference execution HEAD:

`e27dbbd45635da4beb03691f5a266a7d72824424`

Reference population:

- family: `XG1`
- pair range: `xg1_fact_2401` through `xg1_fact_2700`
- N: `300`

Reference epsilon:

`0.025`

Reference raw artifacts:

- `finite_epsilon_principal_decomposition_items.jsonl`
  - SHA256:
    `8db506d872ff81e72b08d44f9ff0af907cb1a65086c0071e3e365a75bd166e17`
  - Git blob:
    `edd46066805082734f624a54736df8a6b82544e5`
- `finite_epsilon_principal_decomposition_summary.json`
  - SHA256:
    `7f9e01dd28ffb2d2a286b464aa4701c639e68f265e11de9eef67ca1e3c1e448e`
  - Git blob:
    `996cc03c15d7cc8a04109325e35d6b0512e00ea2`
- `artifact_manifest.json`
  - SHA256:
    `bcf6facd7d7a506b2dcbd1da4ce6eca67749eaa8e0e014bfb0ffac0467fa1d10`
  - Git blob:
    `4967aca0c5634d136867e500548c757489053b6b`
- `SHA256SUMS.txt`
  - Git blob:
    `9b19f40ac9843ccc74a72118e3ce4683bb79bea9`

Reference representative checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

The same frozen native-Q0 evidence is reused. No new original-basis scientific model
forward is required.

## 3. Exact epsilon set

The epsilon set is frozen by algebraic halving of the existing value:

1. `0.025` — existing frozen reference; do not rerun.
2. `0.0125` — new observation.
3. `0.00625` — new observation.

No fourth epsilon may be added after smaller-epsilon responses are observed.

This is not an epsilon sweep and no epsilon is selected for downstream optimization.

## 4. Frozen geometry and population

Keep unchanged:

- N = `300`;
- XG1 pairs `2401..2700`;
- planes `P1..P5`;
- principal direction order:
  `P1_plus, P1_minus, P2_plus, P2_minus, P3_plus, P3_minus, P4_plus,
  P4_minus, P5_plus, P5_minus`;
- positive contrast eigenvalues and frozen principal vectors;
- representative checkpoint;
- native Q0 source and item order;
- serialization, anchor, state measurement, runtime timing, and Q definition;
- two-GPU 150/150 pair sharding.

No fresh pair selection is performed.

## 5. Finite-difference measurement

For every pair, epsilon, and principal direction d:

`J_epsilon(d) = [F(+epsilon d) - F(-epsilon d)] / (2 epsilon)`

For plane k with positive contrast eigenvalue lambda_k:

`C_epsilon(k) = lambda_k * [J_epsilon(k,+)^2 - J_epsilon(k,-)^2] / 5`

and:

`Q_principal_epsilon = sum_k C_epsilon(k)`

`R_epsilon = Q0 - Q_principal_epsilon`

These are the same definitions as the frozen epsilon=0.025 decomposition, with only
epsilon changed.

## 6. New forward budget

The existing epsilon=0.025 run is reused and contributes zero new forwards.

Each new epsilon requires:

- 10 principal directions;
- 2 signs;
- 2 full model forwards per signed probe;
- 40 scientific model forwards per pair.

Therefore:

- `0.0125`: `300 * 40 = 12000` new scientific forwards;
- `0.00625`: `300 * 40 = 12000` new scientific forwards;
- Experiment 4 new total: `24000` scientific forwards.

With the frozen 150/150 two-GPU shard split:

- GPU0: `12000` forwards total across both new epsilons;
- GPU1: `12000` forwards total across both new epsilons.

No baseline/native-Q0 rerun is permitted.

## 7. Reference spectral profile

At epsilon `0.025`, the frozen mean finite-epsilon plane contributions are:

- P1: `+2.8557904254582005e-08`
- P2: `+3.4693191890669348e-08`
- P3: `+7.2858080938013261e-08`
- P4: `+2.8744010983894693e-09`
- P5: `+4.8767819587928708e-08`

Thus the frozen reference finite-epsilon **spectral dominant candidate** is:

`P3`

and the frozen reference P3-minus-P5 mean contribution contrast is:

`+2.4090261350084553e-08`

Important: finite-epsilon spectral contribution magnitude is descriptive and is not a
causal ranking. The phrase "spectral dominant candidate" in Experiment 4 must not be
promoted into a new causal-plane discovery claim.

## 8. Primary qualitative robustness rule

Experiment 4 is descriptive and adds no p-values.

Define the mean contribution vector at epsilon e:

`m(e) = [mean C_e(P1), ..., mean C_e(P5)]`

For each new epsilon independently define its spectral dominant candidate as the unique
argmax of the five entries of `m(e)`.

The frozen **core qualitative robustness rule** is satisfied only if, at both
`0.0125` and `0.00625`:

1. the unique spectral dominant candidate is `P3`; and
2. `mean C_e(P3) - mean C_e(P5) > 0`.

If both conditions hold at both new epsilons, the bounded result label is:

`SMALL_EPSILON_P3_SPECTRAL_DOMINANCE_PRESERVED`

Otherwise:

`SMALL_EPSILON_P3_SPECTRAL_DOMINANCE_NOT_PRESERVED`

No numerical similarity cutoff is part of this result rule.

## 9. Required descriptive diagnostics

The analysis must report, with zero inferential p-values:

### 9.1 Mean signed plane profile

For each epsilon:

- mean contribution for P1..P5;
- sign of each mean contribution;
- full rank ordering by mean contribution;
- P3-minus-P5 mean contribution contrast.

### 9.2 Normalized profile similarity

For each epsilon define:

`v(e) = m(e) / ||m(e)||_2`

Report:

- cosine similarity `cos[v(0.025), v(0.0125)]`;
- cosine similarity `cos[v(0.025), v(0.00625)]`;
- cosine similarity `cos[v(0.0125), v(0.00625)]`.

No cosine threshold is allowed.

### 9.3 Finite-difference magnitude convergence

Across the fixed 300 pairs × 10 directions, report:

- RMS magnitude of `J_epsilon`;
- RMS difference `J_0.0125 - J_0.025`;
- RMS difference `J_0.00625 - J_0.0125`;
- ratio of the latter RMS difference to the former when the denominator is nonzero;
- mean and median absolute inter-scale J difference.

These are descriptive convergence diagnostics only.

### 9.4 Reconstruction stability

At every epsilon report:

- mean Q0;
- mean Q_principal;
- mean residual;
- mean absolute residual;
- residual RMSE;
- normalized RMSE / RMS(Q0);
- normalized MAE / mean(|Q0|);
- Pearson(Q0, Q_principal);
- sign agreement;
- absolute relative residual q50/q90/q95/q99.

### 9.5 Numerical degeneracy diagnostics

At every epsilon report:

- nonfinite J count;
- exact-zero J count;
- min / q05 / median / q95 / max of absolute J;
- min / q05 / median / q95 / max of absolute central-difference numerator
  `|F_plus - F_minus|`.

No post-hoc threshold or epsilon exclusion may be introduced from these diagnostics.

## 10. Analysis boundary

Experiment 4 has:

- inferential test count: `0`;
- p-value count: `0`;
- multiplicity correction count: `0`;
- model-selection count: `0`;
- epsilon-selection count: `0`.

The analysis may describe profile stability, convergence, reconstruction fidelity, and
degeneracy only.

It must not claim:

- causal additivity;
- plane independence;
- exact finite-epsilon identity;
- optimal epsilon;
- improved steering;
- AVeriTeC utility;
- a new causal rank discovery.

## 11. No-rescue rule

After this design is frozen:

- do not add an epsilon;
- do not remove an epsilon because of an unfavorable result;
- do not change the pair population;
- do not change P3/P5 identities;
- do not substitute another layer/site;
- do not alter the Q endpoint;
- do not use a smaller epsilon as a steering multiplier;
- do not create a p-value family.

A materially different experiment requires a new prospective design.

## 12. Execution order

1. freeze this design;
2. implement one two-epsilon raw runner reusing the existing epsilon=0.025 artifact;
3. CPU/static tests and implementation freeze;
4. one pinned T4x2 Kaggle raw execution for epsilons `0.0125` and `0.00625`;
5. collect/import/validate/freeze raw artifacts;
6. run one CPU/static descriptive robustness analysis joining the frozen reference with
   both new epsilons;
7. freeze the Experiment 4 result;
8. only then proceed to Experiment 5 one-shot adjacent-site specificity.
