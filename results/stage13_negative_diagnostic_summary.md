# Stage13 Negative Diagnostic Summary

## 1. Purpose

Stage13 tested whether the accepted dummy-backbone ContraMamba-v6A residual-composer gains transfer to Mamba-backed runs and to the Stage10A `number_swap` OOD-lite probe. The goal was diagnostic rather than confirmatory: identify whether the v6A residual composer remains a valid current Mamba-backed improvement, or whether its controlled gains are specific to the dummy controlled setting.

This summary should be read as a negative diagnostic result. It does not establish OOD validation success.

## 2. Reproduction status

After patching the Stage13 runner, the dummy controlled pattern was reproduced:

- v5 clean macro-F1 was around 0.850.
- v6A clean macro-F1 was around 0.888.
- v6A fixed some `number_swap` false-entitled errors in the dummy setting.

This supports the earlier conclusion that v6A residual composition can improve the controlled dummy-backbone setting.

However, the Mamba-backed run did not reproduce the accepted clean advantage:

- v5 clean macro-F1 was around 0.766.
- v6A clean macro-F1 was around 0.693.
- v6A increased `number_swap` false-entitlement relative to v5 in the 50epoch/3seed Mamba run.

Therefore, Stage13 did not validate v6A as the current Mamba-backed improvement.

## 3. Main Mamba failure

The Mamba-backed v6A failure appears to be a residual-composer over-correction problem:

- Raw/full composer correction norms were large relative to product logits.
- Clean-dev corrections consistently suppressed `NOT_ENTITLED`.
- Clean-dev corrections boosted `SUPPORT`.
- `SUPPORT` carryover appeared under Mamba features.
- On the `number_swap` probe, this produced false-entitled behavior rather than robust rejection.

The `number_swap` set should be treated as an OOD-lite diagnostic probe, not as a full OOD validation benchmark.

## 4. Diagnostic ablations

### Fixed correction scale

Fixed residual scaling showed that correction magnitude matters:

- Scale 1.0 reproduced the failure, with `number_swap` false-entitled around 11 in the seed1/20epoch run.
- Scale 0.5 reduced false-entitled to about 4.
- Scale 0.25 and 0.1 removed false-entitled but collapsed OOD predictions toward `NOT_ENTITLED`.

Conclusion: reducing correction magnitude mitigates the failure, but fixed scaling is heuristic and can create OOD collapse.

### Correction L2

Correction L2 regularization was tested as a softer alternative to fixed scaling:

- Weak L2 values up to 0.01 barely affected correction ratio.
- Weak L2 values up to 0.01 barely affected false-entitled behavior.

Conclusion: the tested L2 range was too weak to solve the residual over-correction problem in its current form.

### Learnable global alpha

The learnable global residual scale did not find an initialization-independent solution:

- Init 0.01 led to final alpha around 0.010.
- Init 0.1 led to final alpha around 0.101.
- Init 0.5 led to final alpha around 0.505.
- Init 1.0 led to final alpha around 1.010.

Conclusion: alpha stayed near initialization rather than learning a robust correction magnitude.

### Product-gated correction

The product-conditioned gate was intended to learn when to trust the composer correction:

- Hidden=16, detach=True produced clean macro-F1 around 0.604 and `number_swap` false-entitled around 4.
- Clean gate mean was around 0.575.
- `number_swap` gate mean was around 0.566.
- detach=False gave nearly the same behavior.
- Hidden dimensions 64 and 128 increased gate magnitude or variance but did not improve results.
- Hidden 128 increased false-entitled to about 5.

Conclusion: the product gate behaved more like a learned global scale than a sample-conditioned trust policy.

### Trust-supervised gate

The trust-supervised gate tried to directly teach the gate when to suppress correction:

- gate_trust_loss_weight values 0.01, 0.03, 0.1, and 0.3 yielded almost identical behavior.
- Clean macro-F1 stayed around 0.603-0.604.
- `number_swap` false-entitled stayed around 4.
- Clean gate mean stayed around 0.58.
- `number_swap` gate mean stayed around 0.57.
- The `product_correct_margin` target often pushed the gate toward opening broadly rather than selectively suppressing risky correction.

Conclusion: this auxiliary trust signal did not produce a useful sample-conditioned correction policy.

## 5. Interpretation

Stage13 suggests that the v6A residual composer is not simply under-regularized. Under Mamba features, the composer correction appears to learn a broad label-shifting behavior that suppresses `NOT_ENTITLED` and boosts entitled labels, especially `SUPPORT`.

The tested mitigation strategies did not produce a satisfying Mamba-backed model:

- Fixed scaling reduced the problem but introduced heuristic sensitivity and OOD collapse.
- L2 regularization was too weak in the tested range.
- Global learnable alpha stayed near initialization.
- Product-conditioned gating did not become sample-conditioned.
- Trust supervision did not create selective correction behavior.

The failure mode is therefore not resolved by simple residual magnitude control in Stage13A.

## 6. Implication for paper framing

Stage13A should be closed as a negative diagnostic.

The paper should not claim OOD validation success from Stage13. It should not present `number_swap` as proof of real OOD robustness. It should describe `number_swap` as an OOD-lite probe used to reveal residual-composer failure under Mamba features.

The accepted Mamba model should not be v6A unless future evidence changes this conclusion. v6A residual composition may remain useful as a controlled/dummy finding, but it is not currently supported as a Mamba-backed improvement.

The safest framing is:

- v6A improves controlled dummy-backbone behavior.
- v6A does not currently transfer cleanly to Mamba-backed runs.
- Stage13 provides negative evidence and failure analysis, not validation.

## 7. Recommended next action

Do not promote v6A residual composer as the accepted Mamba-backed architecture.

Recommended next steps:

1. Close Stage13A as a negative diagnostic.
2. Preserve the Stage13 diagnostic scripts and logs as failure-analysis evidence.
3. If continuing, design a new diagnostic stage around a different Mamba-backed correction mechanism rather than adding more scalar/gate patches to v6A.
4. Treat any future Mamba-backed candidate as unvalidated until it first reproduces the clean controlled advantage before OOD-lite probe claims are considered.

## Notes on evidence location

In this local checkout, only Stage13 smoke files are present under `results/`. The fuller 20epoch/50epoch Mamba diagnostic numbers summarized here appear to come from external run logs or result files not currently checked into this workspace.
