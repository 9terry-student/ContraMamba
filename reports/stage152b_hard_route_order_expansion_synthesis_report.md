# Stage152-B Hard Route/Order Expansion Synthesis

## Summary decision

`STAGE152B_ROUTE_ORDER_HARD_EXPANSION_CONSERVATIVE_PASS_WITH_BOUNDARY_MISSES`

Stage152-A passed as a conservative hard route/order expansion. `route_order_reversal_v2` catches direct, alias, lowercase, and via explicit route reversal while preserving required safety controls. Final integration remains blocked.

## Why Stage152-A was needed

Stage150-B showed that `route_order_reversal_v2` retained targeted route/order behavior and removed the Stage148-A `Side to Side` self-loop/title false positive, but the broader scan had no natural true-positive route/order gains. Stage151-A therefore registered the analyzer as a conservative shadow diagnostic and recommended a hard route/order expansion.

Stage152-A created that expansion to test explicit reversals, route aliases, lowercase route mentions, via-route phrasing, title/self-loop protections, one-sided routes, return trips, and boundary cases that are intentionally outside v2 scope.

## Aggregate result

| Metric | Value |
| --- | ---: |
| n_rows | 99 |
| n_changed_total | 30 |
| delta_false_support | -30 |
| delta_false_ne | 0 |
| feature_false_support_tp | 30 |
| feature_correct_support_fp | 0 |
| feature_false_support_fn | 12 |
| feature_precision | 1.0 |
| feature_recall | 0.7142857142857143 |
| macro_f1_delta | 0.2906962895098488 |
| support_rows_with_route_reversal | 30 |

The analyzer decision was `STAGE150_ROUTE_ORDER_SHADOW_V2_CANDIDATE_PASS`.

## Required trigger checks

| Check | n_total | delta_false_support | Pass |
| --- | ---: | ---: | --- |
| direct_route_reversal_ne | 12 | -12 | true |
| alias_route_reversal_ne | 8 | -8 | true |
| lowercase_route_reversal_ne | 4 | -4 | true |
| via_route_reversal_ne | 6 | -6 | true |

v2 catches direct, alias, lowercase, and via explicit route reversal.

## Required safety checks

| Check | n_total | n_changed_total | delta_false_ne | Pass |
| --- | ---: | ---: | ---: | --- |
| same_route_support | 12 | 0 | 0 | true |
| alias_same_route_support | 8 | 0 | 0 | true |
| nondirectional_between_support | 6 | 0 | 0 | true |
| directional_vs_between_support | 6 | 0 | 0 | true |
| org_endpoint_reversal_support | 6 | 0 | 0 | true |
| title_side_to_side_support | 3 | 0 | 0 | true |
| quoted_title_to_phrase_support | 4 | 0 | 0 | true |
| one_sided_route_support | 4 | 0 | 0 | true |
| non_support_prediction_safety | 4 | 0 | 0 | true |
| return_trip_same_route_support | 4 | 0 | 0 | true |

v2 is safe on same-route, between, organization endpoint, title/self-loop, quoted title, one-sided route, return trip, and non-SUPPORT controls. The title/self-loop guard remained active with 3 rows containing rejected self-loop candidates and 7 rows containing title-like rejected candidates.

## Expected boundary misses

| Boundary case | n_total | false_support_after | Expected miss allowed |
| --- | ---: | ---: | --- |
| implicit_depart_arrive_reversal_ne_expected_miss | 4 | 4 | true |
| origin_destination_reversal_ne_expected_miss | 4 | 4 | true |
| multi_hop_order_swap_ne_expected_miss | 4 | 4 | true |

The remaining 12 false supports are expected boundary misses, not failures. They define the conservative scope of the current analyzer.

## Conservative scope of `route_order_reversal_v2`

The current scope is explicit directional route reversal. It includes conservative deterministic route extraction, alias canonicalization, self-loop rejection, title/context guarding, and organization endpoint guarding.

Out of scope:

- Implicit depart/arrive reversal.
- Origin/destination start-end reversal.
- Multi-hop order swap.
- Context-dependent itinerary semantics.

`route_order_reversal_v2` remains a conservative shadow diagnostic. Final integration remains blocked.

## Policy input safety

The policy uses claim text, evidence text, original prediction label, deterministic route rules, deterministic alias rules, deterministic organization-like rules, deterministic self-loop guards, and deterministic title/context guards.

It does not use gold labels for policy decisions, intervention type, slot mismatch targets, diagnostic family metadata, file path heuristics, or row id heuristics.

## Remaining limitations

- Implicit depart/arrive reversals remain missed.
- Origin/destination start-end reversals remain missed.
- Multi-hop order swaps remain missed.
- Complex route language and context-dependent itinerary semantics remain outside current scope.
- Analyzer remains shadow-only and diagnostic-only.

## Stage153 recommendation

Default recommendation: freeze `route_order_reversal_v2` as a conservative shadow diagnostic and move back to broader ContraMamba architecture/evaluation work unless implicit route semantics become a priority.

Possible extensions:

- Depart/arrive semantic analyzer.
- Origin/destination start-end analyzer.
- Multi-hop itinerary order analyzer.

Avoid automatic final prediction override, training loss integration, checkpoint selection using route/order outputs, and expanding route semantics without negative controls.
