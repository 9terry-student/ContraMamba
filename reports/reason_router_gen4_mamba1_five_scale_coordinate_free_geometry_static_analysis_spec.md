# Mamba-1 Five-Scale Coordinate-Free Native Geometry Static Analysis Specification

## Status

`PRE_OUTCOME_STATIC_ANALYSIS_SPEC`

This specification is frozen before inspecting any cross-scale CKA or RSM
result.

The purpose is to test whether the native geometry already used by the
five-scale mechanism analysis is preserved across scale up to changes of
feature coordinates.

This is a static analysis of already-frozen artifacts.

No model execution, tokenizer execution, training, evaluation, forward pass,
backward pass, task-response access, or new representation collection is
authorized.

---

## 1. Scientific question

The existing study shows scale-dependent changes in native geometry summaries
and in the scale-local causal realization.

Those observations alone do not determine whether the underlying sample
geometry could remain similar after a rotation or other orthogonal change of
feature coordinates.

The confirmatory descriptive question here is:

> For the same frozen stimulus population, how similar is the response-blind
> alignment geometry across Mamba-1 scales under a coordinate-insensitive
> comparison?

The analysis does not assume that either similarity or dissimilarity will be
observed.

---

## 2. Scales

The fixed scale order is:

1. 130M
2. 370M
3. 790M
4. 1.4B
5. 2.8B

130M is retained with explicit historical-provenance qualification.

The 370M--2.8B artifacts belong to the later homogeneous geometry-preparation
series.

---

## 3. Frozen geometry object

For each scale and generator family, the analysis uses the already-frozen
per-pair alignment vectors:

`alignment_delta_h`

No representation is recomputed.

The frozen matrices have one row per source pair:

- 130M: 300 x 395
- 370M: 300 x 650
- 790M: 300 x 975
- 1.4B: 300 x 829
- 2.8B: 300 x 1003

The differing feature dimensions are expected and are not reconciled by
padding, projection, truncation, interpolation, learned alignment, or any
other fitted transformation.

---

## 4. Frozen stimulus populations

Two families are analyzed independently:

- XG2: `xg2_fact_301` through `xg2_fact_600`
- XG4: `xg4_fact_301` through `xg4_fact_600`

Each family contains exactly N=300 source pairs.

The analysis must verify exact pair identity and exact ordering from the
corresponding frozen item artifacts before any statistic is calculated.

XG2 and XG4 are never pooled.

---

## 5. Response-blind boundary

The consumed geometry artifacts must retain the frozen contract:

- no XG1 response access;
- no task-response observation;
- no task-head use for geometry construction;
- no backward pass;
- no response-guided plane or control selection.

This analysis consumes frozen vectors only and therefore performs zero
scientific model forwards.

---

## 6. Primary representation

Let the frozen alignment matrix for scale s be

    D_s in R^(N x d_s).

For every row i, define the L2-normalized vector

    U_s[i,:] = D_s[i,:] / ||D_s[i,:]||_2.

Rows with zero or non-finite norm are a hard failure.

Row normalization is primary because the frozen principal-geometry
construction itself forms its second moment from normalized alignment vectors.

No alternative normalization may replace this definition after results are
observed.

---

## 7. Primary statistic: centered linear CKA

For each scale s define the sample Gram matrix

    K_s = U_s U_s^T.

Let

    H = I_N - (1/N) 11^T

and

    Kc_s = H K_s H.

For every unordered pair of scales (s,t), compute

    CKA(s,t)
      = <Kc_s, Kc_t>_F
        / ( ||Kc_s||_F ||Kc_t||_F ).

This is computed separately for XG2 and XG4.

All ten unordered scale pairs must be reported for each family.

No pair may be omitted on the basis of its result.

No threshold for "same geometry" or "different geometry" is specified.

The statistic is descriptive.

No p-value or significance test is authorized.

---

## 8. Secondary statistic: cosine-RSM similarity

Because rows of U_s are unit normalized,

    G_s = U_s U_s^T

is the pairwise cosine-similarity matrix among the same N stimulus pairs.

For each unordered scale pair, extract the strict upper triangles of G_s and
G_t in identical row order and report their Pearson correlation.

This secondary statistic is descriptive and must not supersede CKA as the
primary coordinate-insensitive comparison.

No Spearman alternative, permutation test, subset analysis, or result-driven
metric replacement is authorized in this stage.

---

## 9. Required comparisons

For each of XG2 and XG4 report the complete symmetric 5 x 5 CKA matrix and the
complete symmetric 5 x 5 cosine-RSM Pearson matrix.

The report must also serialize the ten unique off-diagonal scale pairs.

The scale order is fixed as:

    130M, 370M, 790M, 1.4B, 2.8B

No monotonic trend test or scaling-law fit is authorized.

---

## 10. Interpretation rules fixed before outcome

The analysis distinguishes two possible scientific outcomes without selecting
between them in advance.

If coordinate-insensitive similarity remains strong across scales, the paper
must not claim global geometric non-invariance. The interpretation should
instead distinguish relatively conserved sample geometry from reorganization
of the scale-local principal/causal realization.

If coordinate-insensitive similarity is substantially heterogeneous across
scales, that provides additional descriptive evidence that reorganization is
not merely a relabeling or orthogonal rotation of feature coordinates.

No numerical cutoff for either interpretation is frozen.

The exact matrices and pairwise values must be shown so that readers can judge
the pattern directly.

In either case the analysis does not establish:

- a continuous scaling law;
- a size threshold;
- semantic homology of equal plane numbers;
- arbitrary-linear non-equivalence;
- architectural universality beyond the sampled Mamba-1 checkpoints.

Linear CKA is invariant to orthogonal feature rotations and isotropic scaling;
it is not treated as invariant to arbitrary invertible transformations.

---

## 11. Provenance requirements

Before calculation, record for every consumed artifact:

- repository-relative path;
- SHA256;
- tensor shape;
- dtype;
- source-pair first/last ID;
- source-pair count.

The analysis must fail closed on:

- missing artifact;
- hash/read failure;
- unexpected tensor shape;
- non-finite value;
- zero row norm;
- pair identity mismatch;
- pair ordering mismatch;
- N != 300.

The 130M outputs must remain explicitly labeled as historical-provenance
geometry.

---

## 12. Authorized outputs

A later implementation may create only static-analysis code and outputs needed
for this specification, including:

- one CPU-only analysis script;
- one JSON result;
- one pair-level CSV;
- one Markdown analysis report;
- checksums/provenance manifest if required.

No manuscript or figure modification is authorized by this specification.

Paper changes require interpretation of the frozen static result first.

---

## 13. Execution boundary

Training/Evaluation allowed: NO

Model execution allowed: NO

Tokenizer execution allowed: NO

Forward/backward allowed: NO

GPU required: NO

Static CPU tensor loading and deterministic numerical calculation: YES

Commit/Push: manual only after review.

---

## 14. Stop conditions

Stop without producing a scientific conclusion if:

- any scale lacks its frozen matrix;
- XG2 or XG4 item identity is not common across all five scales;
- the stored object is found not to have the same alignment-vector semantics
  across the five scales;
- any artifact/provenance check fails.

Otherwise proceed only after this specification is frozen in repository
history.
