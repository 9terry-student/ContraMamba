# Gen4-K XG2/XG4 Family-Specific Subspace Sensitivity — Execution Freeze

Implementation HEAD:

`2ad38ed4bba303fdd3ec32d8dd6b6fa8bf0b1093`

Frozen design inputs:

- implementation scope: `4f36483080b6f63b9c9a3dd031cd97a88995e0f3`
- finite-difference basis correction: `594cb45bdd8740b2dfd63c4780f766f1d0b375bc`
- Phase-1 artifact freeze: `f4f5aef2025a66e8b7e7ed6d077523eec46eb6f0`

## Authorized execution

This freeze permits exactly one scientific observation execution for each frozen family:

- XG2 pairs 301..600
- XG4 pairs 301..600

The implementation must reconstruct both family bases from the frozen Phase-1 `alignment_delta_h.pt` artifacts before model setup.

Fixed geometry and probe:

- subspace dimension: `k = 5`
- ordered eigenbasis: five largest eigenvalues of the float64 CPU uncentered second moment
- fixed eigenvector sign canonicalization only
- epsilon: `0.025`
- own basis: five directions
- cross-family basis: five directions
- symmetric forward/reverse probe for every direction

No basis, sign, dimension, epsilon, pair, layer, channel, offset, endpoint, or checkpoint adaptation is authorized.

## Scientific forward budget

Exactly:

- 4 forwards per basis direction
- 10 basis directions per pair
- 40 scientific forwards per pair
- 12,000 scientific forwards per family
- 24,000 scientific forwards across XG2 and XG4
- 0 new baseline model forwards

No training, backward, task heads, or logits.

## Authorized observation outputs

The execution may persist only runner-defined observation/provenance artifacts, including:

- per-direction `F_plus`
- per-direction `F_minus`
- per-direction finite-difference `J`
- per-direction `J_squared`
- per-pair `E_own`
- per-pair `E_cross`
- per-pair `D = E_own - E_cross`
- intervention audits
- basis reconstruction diagnostics
- forward-budget/provenance fields
- manifest, checksum, and observation summary

The observation summary must retain:

- `primary_inference_executed = false`
- `holm_correction_executed = false`
- `scientific_conclusion = null`

## Explicitly prohibited during execution

- Student t-tests
- p-values
- Holm correction
- final family-subspace scientific conclusion
- subgroup or tail analysis
- response-guided basis changes
- dimension or epsilon sweeps
- alternative eigenbasis rotation
- XG3 anchor redesign
- new baseline forwards
- training or backward
- task heads or logits
- any additional mechanistic execution outside the fixed XG2/XG4 observation

## Post-execution boundary

A successful run establishes only execution success for the fixed observation protocol.

After both family artifacts are collected/imported:

1. validate execution identity, hashes, budgets, manifests, and frozen geometry;
2. freeze the imported observation artifacts;
3. only then run the pre-specified read-only confirmatory analysis:
   - family-wise one-sample Student t-test of `mean(D) > 0`;
   - Holm correction across exactly XG2 and XG4;
   - `FAMILY_SPECIFIC_SUBSPACE_SENSITIVITY_REPLICATED` only if both corrected tests reject with positive family means;
   - otherwise `FAMILY_SPECIFIC_SUBSPACE_SENSITIVITY_NOT_ESTABLISHED`.

No scientific conclusion is authorized before artifact validation and freeze.
