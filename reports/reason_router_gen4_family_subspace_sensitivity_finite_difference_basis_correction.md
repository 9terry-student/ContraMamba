# Gen4-K Family Subspace Sensitivity — Finite-Difference Basis Correction

Parent scope commit: `4f36483080b6f63b9c9a3dd031cd97a88995e0f3`

This correction changes one mathematical claim in the frozen implementation scope before implementation or scientific execution.

## Corrected statement

The primary energy

`E = (1/5) * sum_j J_epsilon(v_j)^2`

is exactly invariant to eigenvector sign flips because the symmetric probe satisfies
`J_epsilon(-v) = -J_epsilon(v)`.

It is **not claimed to be exactly invariant to arbitrary orthonormal basis rotations inside the same five-dimensional subspace**, because `J_epsilon` is a finite-difference quantity at fixed `epsilon = 0.025`, not an exact infinitesimal directional derivative.

Therefore the experiment uses the **ordered top-5 eigenbasis itself as the frozen probe basis**.

For each family:
1. construct the float64 CPU uncentered second moment `M_f`;
2. use `torch.linalg.eigh(M_f)`;
3. sort eigenpairs by eigenvalue descending;
4. take exactly the first five eigenvectors in that order;
5. permit only per-eigenvector sign ambiguity, which cannot change the squared-J endpoint;
6. do not rotate, optimize, replace, or response-adapt the basis.

The runner must validate:
- finite symmetric `M_f`;
- five finite selected eigenvalues;
- strict positive eigengaps between each adjacent selected eigenvalue and between eigenvalues 5 and 6, using a fixed numerical tolerance defined in implementation;
- top-5 orthonormality within strict float64 tolerance.

If these conditions fail, execution must fail closed rather than choose another basis.

## Unchanged design

Everything else in
`reports/reason_router_gen4_family_subspace_sensitivity_implementation_scope.md`
remains unchanged, including:

- frozen Phase-1 inputs;
- exact 300 pairs/family;
- `k = 5`;
- `epsilon = 0.025`;
- own five directions plus cross five directions;
- `E_own`, `E_cross`, and `D = E_own - E_cross`;
- 40 scientific forwards/pair;
- 12,000 scientific forwards/family;
- 24,000 total scientific forwards;
- zero baseline forwards;
- family-wise one-sided Student t-test `mean(D) > 0`;
- Holm correction across exactly XG2 and XG4;
- no scientific inference during the observation run;
- no dimension/epsilon/basis/subgroup/tail search;
- no XG3 anchor redesign;
- no training, backward, task heads, or logits.

No model execution, outcome inspection, or inferential test was performed to make this correction.
