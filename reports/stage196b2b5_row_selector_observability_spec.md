# Stage196-B2-B5 Row-Selector Observability Specification

## Scope

Stage196-B2-B5 is an artifact-only, frozen-Mamba analysis. It asks whether existing mechanistically meaningful state admits an exact deterministic primitive-coalition selector. It does not fit a classifier, learn or search a threshold, optimize a score, train a model, load a model or checkpoint, or authorize promotion.

The scientific authority is the successful Stage196-B2-B4 result at analyzer runtime commit `8664fef0527a461ea8e8644bbf04770f526d4569`. The P0 runtime authority is `fa16787efa84bb15d832b6d9382fafd77016c4e2`.

## Explicit inputs

The analyzer requires exactly these CLI options:

```text
--repo-root
--stage196b2b4-analysis-json
--stage196b2b2-analysis-json
--stage196b2b3p0-run-root
--stage196b2b3p0-runtime-git-commit
--current-git-commit
--output-dir
```

All paths are explicit. Source discovery by timestamp, mtime, recursive guessing, or substring fallback is forbidden. B2-B4 and B2-B2 companions are resolved only beside the supplied exact analysis JSON and only by their exact expected names. P0 uses the six exact run directory names. The mixed-purpose `trajectory/` directory validates only `stage196b2p0_epoch_channels_001.jsonl` through `_020.jsonl`; unrelated files are ignored and malformed namespace-like files fail closed.

## Source contracts

B2-B4 must have the exact nine-file closure, the required decision and next-stage values, no blocking reasons, all contract gates passed, 20,480 primitive coalition rows, 20,480 primitive Möbius rows, 1,024 primitive tail summaries, 20,480 residual coalition rows, 20,480 residual Möbius rows, 16 identities, 640 directional states, and 32 coalitions in each lattice. Primitive and residual Möbius reconstruction error must be at most `1e-6`; empty and full controls must pass.

B2-B2 must have its exact nine-file closure, the required seed-specific multipath decision and next stage, exactly 155 passed gates, 16 identities, 320 paired epoch rows, and tail epochs 18, 19, and 20. Seed184 retains five recovery and six harm rows; seed185 retains two recovery and three harm rows. Seed183 is contrast-only and cannot affect selector decisions.

P0 must have exactly six run directories, 86,400 composer rows, 86,400 trajectory rows, 120 composer sidecars, 120 trajectory sidecars, exact paired identity closure, manifest hash closure, and runtime commit agreement.

## Selector target

The primary direction is only `JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR`. The reverse direction is excluded from the primary target and may only support symmetry diagnostics. The target comprises 16 tail-level identities and its temporal diagnostic comprises 16 identities by 20 epochs.

Primitive order is `FRAME`, `PREDICATE`, `SUFFICIENCY`, `POSITIVE_ENERGY`, `NEGATIVE_ENERGY`; masks are the complete ordered five-bit lattice `00000` through `11111`.

For a recovery identity, its acceptable set is every coalition whose counterfactual predictions exactly reproduce donor predictions at epochs 18, 19, and 20. For a harm identity, its acceptable set is every coalition whose counterfactual predictions exactly preserve recipient predictions at those epochs. The empty coalition is accepted only when it satisfies the corresponding exact objective. Inclusion-minimal masks are reported without selecting an arbitrary winner.

A signature is feasible exactly when the set intersection of acceptable coalitions for every row with that signature is nonempty. Rates, average scores, rankings, majority votes, partial coverage, and learned thresholds cannot establish feasibility.

## Feature separation and leakage prohibition

Recipient-local features use only the joint recipient state. Paired-delta features use exact frame-local-only donor minus joint recipient state and are always labeled `diagnostic_only = true` and `deployment_authorized = false`. Each semantic feature records exact exported source fields and formula. A required semantic feature with absent or null source fields is marked unavailable; no diagnostic or outcome-derived substitute is allowed.

Natural thresholds are limited to zero with tolerance `1e-12`, and probability halfspace 0.5 with tolerance `1e-12`. Prediction sequences preserve exact epoch order. Entitlement bottleneck ties are preserved as sorted sets. Mismatch and branch activity use raw exported active flags.

Selector signatures prohibit seed, training seed, stable row ID, ID, source row ID, dev position, transition role, path class, subtype, B2-B4 minimal-coalition labels, donor-tail-reproduced outcomes, recipient-tail-preserved outcomes, counterfactual prediction, and counterfactual margin. These values may serve only as keys or stratified reporting metadata.

## Exact subset and transfer audits

If `N` recipient-local semantic features are available, every nonempty subset, exactly `2^N - 1`, is enumerated. The same exhaustive lattice is independently enumerated for available paired-delta features. Every subset records exact signatures, action intersections, full row coverage, signature count, maximum signature size, mixed recovery/harm structure, cross-seed structure, and feasibility. All inclusion-minimal feasible subsets are reported.

Seed184 and seed185 receive independent zero-conflict subset audits without using seed as a feature. The pooled audit is separate.

For each pooled feasible recipient subset, and for each paired diagnostic subset, transfer is evaluated from seed184 to seed185 and from seed185 to seed184. The mapping for a source signature is the exact acceptable-action intersection over source rows. An unseen target signature is `UNSEEN`. A seen row passes only when source intersection and target acceptable set intersect. Full transfer requires zero unseen rows and zero incompatible seen rows in both directions; no partial threshold applies.

## Epoch stability

For every tail-acceptable row/coalition pair, the analyzer audits all 20 epochs using the row-specific recovery or harm objective. It reports satisfying epochs, first and last satisfying epoch, tail-three stability, all-20 stability, and state-transition count. This is diagnostic and does not redefine the tail-level selector target.

## Ordered decisions

Contract failure yields `STAGE196B2B5_BLOCKED_CONTRACT_FAILURE` and `STAGE196B2B5_REPAIR`.

Otherwise the analyzer applies, in order: cross-seed recipient selector localized; recipient selector observable in-sample only; exact seed-specific recipient selectors for both primary seeds; paired-delta selector only; current observability insufficient. The corresponding next stages are exactly those frozen in the Stage196-B2-B5 authority. No decision authorizes training or promotion.

## Outputs and atomicity

The output directory must be absent. Exactly nine required files are rendered to a fresh sibling staging directory, flushed and fsynced, checked for exact closure, and atomically renamed to the requested output directory. Existing output is never overwritten. Contract `required` and `observed` cells are JSON-encoded from typed values, so booleans remain JSON booleans rather than quoted strings.

## Scientific limits

The analysis cannot support formal causal mediation, external or OOD validity, unfrozen-Mamba validity, training improvement, promotion, a universal selector from an in-sample-only partition, deployability from paired deltas, validity from outcome-derived fields, partial-coverage success, average-accuracy success, learned-threshold success, row identity as a selector, or path class/subtype as an inference-time selector.
