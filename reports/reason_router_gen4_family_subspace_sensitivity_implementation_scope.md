# Gen4-K XG2/XG4 Family-Specific Subspace Sensitivity — Implementation Scope

Parent HEAD: `6e5db0a69e05417938bd05038cf9dba1a39846bf`

Frozen inputs:
- Phase-1 direction plans:
  `reports/reason_router_gen4_xg2_xg4_fresh_response_restartable_phase1_d217bad_r3/`
- XG2 `alignment_delta_h.pt` SHA256:
  `b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c`
- XG4 `alignment_delta_h.pt` SHA256:
  `792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f`
- exact population: 300 pairs/family, pair ids 301..600, exact frozen order.

The motivating local-Jacobian result is already frozen at parent HEAD. It may motivate the new hypothesis class but must not be used to construct, select, sign, rotate, exclude, or weight any subspace direction.

This study is prospective only for the previously unobserved subspace-sensitivity outcomes. It is not an independent replication study.

## Outcome-free family subspaces

For family `f` and pair `i`:

`d_fi = alignment_delta_h_f[i]`

`u_fi = d_fi / ||d_fi||_2`

Require every norm finite and strictly positive.

Construct the uncentered sign-invariant second moment:

`M_f = (1/300) * sum_i u_fi u_fi^T`

Use float64 CPU linear algebra and take the five eigenvectors associated with the five largest eigenvalues of `M_f`.

Define the resulting orthonormal basis as:

`U_f = [v_f1, ..., v_f5]`

with subspace dimension fixed exactly at `k = 5`.

The choice `k=5` is frozen now from the outcome-free Phase-1 geometry diagnostic, where top-5 energy was approximately 0.7600 for XG2 and 0.7770 for XG4. That diagnostic used no Phase-2 response, no local-Jacobian J, no regime labels, and no p-values.

No alternative k, PCA dimension sweep, response-guided rotation, sign search, subgroup selection, tail selection, layer/offset/channel/checkpoint search, or posthoc basis replacement is allowed.

Eigenvector sign is arbitrary and scientifically irrelevant. The primary endpoint below is invariant to sign and to orthonormal basis rotation within a fixed five-dimensional subspace.

## Fixed local probe

Use exactly:

`epsilon = 0.025`

For any unit basis direction `v`, the existing intervention runtime applies `delta_h` as `+0.5*delta_h` on the plus branch and `-0.5*delta_h` on the minus branch.

Therefore:

forward probe: `delta_h = +2*epsilon*v`

reverse probe: `delta_h = -2*epsilon*v`

For pair `i`:

`F_i(+epsilon; v) = PE(tp, +epsilon*v) - PE(tm, -epsilon*v)`

`F_i(-epsilon; v) = PE(tp, -epsilon*v) - PE(tm, +epsilon*v)`

`J_i(v) = [F_i(+epsilon; v) - F_i(-epsilon; v)] / (2*epsilon)`

All values must be finite.

No baseline forward is required or allowed.

## Own-versus-cross subspace endpoint

For an XG2 pair:
- own basis = `U_xg2`
- cross basis = `U_xg4`

For an XG4 pair:
- own basis = `U_xg4`
- cross basis = `U_xg2`

For pair `i` in family `f`:

`E_own_i = (1/5) * sum_{j=1..5} J_i(v_own,j)^2`

`E_cross_i = (1/5) * sum_{j=1..5} J_i(v_cross,j)^2`

Primary paired endpoint:

`D_i = E_own_i - E_cross_i`

This measures whether local first-order response sensitivity is more concentrated in the family-local frozen subspace than in the other family's frozen subspace.

The endpoint uses squared directional derivatives and therefore does not assign a causal sign to the subspace.

## Forward budget

Each basis direction requires four forwards per pair:
- two branches for `F(+epsilon; v)`
- two branches for `F(-epsilon; v)`

There are 10 directions per pair: five own plus five cross.

Therefore:
- 40 scientific forwards/pair
- 12,000 scientific forwards/family
- 24,000 scientific forwards across XG2+XG4
- 0 new baseline forwards

No training, backward, task heads, or logits.

## Primary confirmatory rule

For each family independently:

`H_subspace: mean(D) > 0`

Use a one-sample Student t-test on the 300 paired `D_i` values, one-sided greater-than-zero.

Multiplicity family is exactly:
- XG2
- XG4

Apply Holm correction at alpha = 0.05.

Assign:

`FAMILY_SPECIFIC_SUBSPACE_SENSITIVITY_REPLICATED`

only if:
1. XG2 mean `D > 0` and Holm rejects;
2. XG4 mean `D > 0` and Holm rejects.

Otherwise assign:

`FAMILY_SPECIFIC_SUBSPACE_SENSITIVITY_NOT_ESTABLISHED`

Allowed descriptive outputs without extra p-values:
- mean/median/SD of `E_own`, `E_cross`, and `D`;
- per-family mean absolute J for own and cross bases;
- finite-value and forward-budget audits.

No additional hypothesis test, subgroup analysis, tail analysis, basis-wise significance test, or alternative multiplicity family is allowed.

## Scientific boundary

A positive result would support family-specific concentration of local first-order sensitivity in the frozen five-dimensional Phase-1 geometry.

It would not establish:
- that every direction in the subspace is causal;
- a common signed adverse direction;
- transport to XG3 or another generator;
- independence from this research program's prior diagnostics;
- superiority of k=5 over other dimensions.

A negative result would not establish absence of all generator-specific geometry.

XG3 is not part of this experiment. Its frozen tokenizer/anchor eligibility has zero complete eligible pairs under the current anchor semantics, so changing XG3 anchor semantics is outside scope.

## Implementation authorization

Authorized now:
- one scientific runner;
- one dedicated test file;
- deterministic reconstruction and validation of the two frozen k=5 subspaces;
- static validation only.

The runner must fail closed on:
- wrong branch or expected HEAD;
- dirty worktree;
- Phase-1 plan SHA drift;
- wrong tensor shape/dtype/order;
- zero or non-finite plan norm;
- subspace dimension other than 5;
- non-orthonormal basis beyond a strict numerical tolerance;
- output collision;
- non-finite response;
- baseline model forward count other than 0;
- scientific forward-budget mismatch;
- intervention-audit mismatch.

Not authorized yet:
- Kaggle/scientific execution;
- primary t-tests, Holm correction, or scientific interpretation during the observation run;
- dimension sweep;
- epsilon sweep;
- response-guided basis adaptation;
- subgroup/tail rescue;
- XG3 anchor redesign;
- training/backward/task heads/logits.

Execution requires a separate minimal execution freeze after implementation validation.
