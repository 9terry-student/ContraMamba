# Five-Scale Coordinate-Free Native Geometry Static Analysis

## Status

`DESCRIPTIVE_STATIC_RESULT`

Execution commit: `9f077f16daccbec8038c396f65a5f381d9d5c6c8`

No model/tokenizer execution, training, evaluation, forward pass, backward pass, or new representation collection occurred.

130M is historical-provenance geometry; 370M--2.8B are the later homogeneous geometry-preparation series.

## XG2

### Centered linear CKA

| | 130M | 370M | 790M | 1.4B | 2.8B |
|---|---:|---:|---:|---:|---:|
| 130M | 1.000000 | 0.742546 | 0.435747 | 0.342510 | 0.555414 |
| 370M | 0.742546 | 1.000000 | 0.564059 | 0.504483 | 0.649344 |
| 790M | 0.435747 | 0.564059 | 1.000000 | 0.663678 | 0.639963 |
| 1.4B | 0.342510 | 0.504483 | 0.663678 | 1.000000 | 0.532185 |
| 2.8B | 0.555414 | 0.649344 | 0.639963 | 0.532185 | 1.000000 |

### Cosine-RSM Pearson correlation

| | 130M | 370M | 790M | 1.4B | 2.8B |
|---|---:|---:|---:|---:|---:|
| 130M | 1.000000 | 0.716312 | 0.398941 | 0.295730 | 0.522254 |
| 370M | 0.716312 | 1.000000 | 0.541258 | 0.474489 | 0.634246 |
| 790M | 0.398941 | 0.541258 | 1.000000 | 0.640851 | 0.620152 |
| 1.4B | 0.295730 | 0.474489 | 0.640851 | 1.000000 | 0.504037 |
| 2.8B | 0.522254 | 0.634246 | 0.620152 | 0.504037 | 1.000000 |

## XG4

### Centered linear CKA

| | 130M | 370M | 790M | 1.4B | 2.8B |
|---|---:|---:|---:|---:|---:|
| 130M | 1.000000 | 0.542295 | 0.451862 | 0.481304 | 0.544277 |
| 370M | 0.542295 | 1.000000 | 0.558569 | 0.436834 | 0.590983 |
| 790M | 0.451862 | 0.558569 | 1.000000 | 0.444332 | 0.392717 |
| 1.4B | 0.481304 | 0.436834 | 0.444332 | 1.000000 | 0.455698 |
| 2.8B | 0.544277 | 0.590983 | 0.392717 | 0.455698 | 1.000000 |

### Cosine-RSM Pearson correlation

| | 130M | 370M | 790M | 1.4B | 2.8B |
|---|---:|---:|---:|---:|---:|
| 130M | 1.000000 | 0.509857 | 0.431574 | 0.452640 | 0.511619 |
| 370M | 0.509857 | 1.000000 | 0.533989 | 0.403381 | 0.545762 |
| 790M | 0.431574 | 0.533989 | 1.000000 | 0.412623 | 0.337620 |
| 1.4B | 0.452640 | 0.403381 | 0.412623 | 1.000000 | 0.382808 |
| 2.8B | 0.511619 | 0.545762 | 0.337620 | 0.382808 | 1.000000 |

## Interpretation boundary

These matrices are descriptive. No p-value, permutation test, threshold, monotonic trend test, or scaling-law fit was performed.

Linear CKA addresses similarity up to orthogonal feature rotations and isotropic scaling; it does not establish equivalence or non-equivalence under arbitrary invertible transformations.

Scientific interpretation is intentionally deferred until this static result and its provenance are reviewed.
