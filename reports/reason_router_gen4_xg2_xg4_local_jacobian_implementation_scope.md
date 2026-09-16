# Gen4-K XG2/XG4 Unit-Direction Local Jacobian — Implementation Scope

Parent HEAD: `29e0cc6330b5e5270b6057a02c1b86fdeb45ca96`

Frozen inputs:
- Phase-1 plans: `reports/reason_router_gen4_xg2_xg4_fresh_response_restartable_phase1_d217bad_r3/`
- Phase-2 responses remain historical context only and must not select/exclude primary items.

This study is prospective only for previously unobserved local-Jacobian outcomes; it is not an independent replication study.

## Population and direction
Use all 300 frozen pairs per family, XG2 and XG4, in exact order 301..600.

For pair i:
`d_i = alignment_delta_h[i]`
`u_i = d_i / ||d_i||_2`

Require finite non-zero norm. No direction optimization, sign search, PCA projection, tail selection, subgroup selection, or channel/layer/offset/checkpoint search.

## Fixed local radii
Exactly:
- epsilon_1 = 0.025
- epsilon_2 = 0.05

The existing runtime applies `delta_h` as +0.5*delta_h on the plus branch and -0.5*delta_h on the minus branch. Therefore branch-wise radius epsilon uses:
`delta_h = 2*epsilon*u_i`
and the reverse probe uses:
`delta_h = -2*epsilon*u_i`.

## Response
For each pair and epsilon:

`F(+epsilon) = PE(tp,+epsilon*u_i) - PE(tm,-epsilon*u_i)`
`F(-epsilon) = PE(tp,-epsilon*u_i) - PE(tm,+epsilon*u_i)`

No new baseline forward is allowed. Use frozen Phase-1:
`F(0) = delta_baseline`

`J_epsilon = (F(+epsilon)-F(-epsilon))/(2*epsilon)`
`K_epsilon = (F(+epsilon)+F(-epsilon)-2*F(0))/(epsilon^2)`

All values must be finite.

## Forward budget
4 forwards/pair/epsilon, 8 forwards/pair total:
- 2400 new scientific forwards/family
- 4800 across XG2+XG4
- 0 new baseline forwards

No training, backward, task heads, or logits.

## Primary confirmatory rule
Primary endpoint: `J_0.025`.

For each family independently:
`H_local: mean(J_0.025) < 0`

Use one-sample Student t-test, one-sided less-than-zero. Multiplicity family is exactly XG2 and XG4; Holm correction, alpha=0.05.

`epsilon=0.05` is descriptive scale-consistency only: mean/median J, Pearson/Spearman between J_0.025 and J_0.05, sign agreement, and K summaries. No additional p-value.

Assign `LOCAL_DIRECTIONAL_SENSITIVITY_REPLICATED_ACROSS_XG2_XG4` only if:
1. XG2 mean J_0.025 < 0 and Holm rejects;
2. XG4 mean J_0.025 < 0 and Holm rejects;
3. XG2 mean J_0.05 < 0;
4. XG4 mean J_0.05 < 0.

Otherwise assign `LOCAL_DIRECTIONAL_SENSITIVITY_REPLICATION_NOT_ESTABLISHED`.

This does not revise the already frozen full-intervention confirmatory result.

## Implementation authorization
Authorized now:
- one scientific runner;
- one dedicated test file;
- static validation only.

Runner must fail closed on wrong branch/HEAD, dirty worktree, Phase-1 drift, pair-order drift, zero/non-finite plan norm, any baseline model forward, budget mismatch, intervention-audit mismatch, output collision, or non-finite output.

Not authorized yet:
- Kaggle/scientific execution;
- result interpretation;
- epsilon sweep beyond 0.025 and 0.05;
- threshold optimization or subgroup rescue;
- training/backward/task heads/logits.

Execution requires a separate minimal execution freeze after implementation validation.
