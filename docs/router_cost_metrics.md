# Router Cost-of-Conservatism Metrics

## Why post-routing zero violations are not independent evidence

ContraMamba-CAR uses a conservative router: a classifier `SUPPORT` or `REFUTE` proposal is retained only when the auditor gates pass. If a candidate fails the audit, the router changes the output to `NOT_ENTITLED`. A correctly implemented conservative router should therefore have zero gate violations among the entitled outputs it retains.

This post-routing value is useful as an implementation invariant, but it is not an independent empirical achievement. The router enforces it by construction. Reporting it without the pre-routing candidate failures can hide how often the rule intervenes and what correct entitled predictions are lost.

We consequently refer to the existing quantity as the **post-routing retained-output gate violation rate**. It must be interpreted together with the metrics below.

## Pre-routing audit and downgrade metrics

- `classifier_entitled_count`: raw classifier `SUPPORT` or `REFUTE` predictions.
- `pre_router_candidate_gate_fail_count` and `pre_router_candidate_gate_fail_rate`: classifier-entitled candidates that fail the relevant auditor gates before routing.
- `routed_entitled_count`: entitled labels remaining after routing.
- `downgraded_count` and `downgrade_rate_among_classifier_entitled`: classifier-entitled candidates changed to `NOT_ENTITLED`.
- `downgraded_gold_support_count`: correct classifier `SUPPORT` candidates lost to downgrading.
- `support_recall_pre_router`, `support_recall_post_router`, and `support_recall_drop`: the principal measure of lost `SUPPORT` coverage.
- `support_precision_pre_router`, `support_precision_post_router`, and `support_precision_gain`: whether conservative filtering improves the purity of retained `SUPPORT` outputs.
- `false_support_removed_count` and `false_refute_removed_count`: incorrect entitled predictions removed by routing.
- `retained_violation_rate`: post-routing violations among retained entitled outputs; expected to be zero for conservative routers.

## Interpreting the cost

The main empirical tradeoff is consistency versus `SUPPORT` recall. A larger downgrade rate may remove false entitled predictions and improve precision, but it may also downgrade correct support decisions. Threshold sweeps should therefore report pre-router gate failure, downgrade rate, recall loss, precision gain, and removed false predictions alongside final-label metrics.

This framing revises the interpretation of CAR in four ways:

1. CAR **enforces** evidence-entitlement consistency for retained outputs.
2. Zero retained violations document rule compliance, not an unconstrained model property.
3. Conservative downgrading is the cost of that enforcement.
4. The important empirical question is whether removed false entitled outputs justify the resulting `SUPPORT` recall loss.

## Recommended paper wording

> The router enforces zero retained-output gate violations by construction; we therefore report downgrade rates and SUPPORT recall loss as the empirical cost of entitlement-conservative routing.

## Stage 9A evaluation workflow

Use `scripts/sweep_router_thresholds.py --include-router-cost` to evaluate thresholds 0.3 through 0.7 for the conservative balanced, conservative strict, and dual-auditor routers. The per-seed CSV is a wide table with one row per threshold and system. Aggregate the three seed files with `scripts/write_stage9a_router_cost_aggregate.py` to obtain mean +/- sample standard deviation reports.

No Stage 9A numerical claims should be added to the paper until the prediction files have been evaluated and the generated aggregate has been inspected.
