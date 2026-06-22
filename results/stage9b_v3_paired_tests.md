# Stage 9B Paired Tests

## McNemar tests

| seed | scope | intervention | n | classifier only correct | router only correct | statistic | p_value | accuracy_delta |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | global | all | 780 | 0 | 1 | 0.0000 | 1.0000 | 0.0013 |
| 1 | intervention | entity_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | event_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | evidence_deletion | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | evidence_truncation | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | irrelevant_evidence | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | location_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | none | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | paraphrase | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | polarity_flip | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | predicate_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | role_swap | 60 | 0 | 1 | 0.0000 | 1.0000 | 0.0167 |
| 1 | intervention | time_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 1 | intervention | title_name_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | global | all | 780 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | entity_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | event_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | evidence_deletion | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | evidence_truncation | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | irrelevant_evidence | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | location_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | none | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | paraphrase | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | polarity_flip | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | predicate_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | role_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | time_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 2 | intervention | title_name_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | global | all | 780 | 0 | 3 | 1.3333 | 0.2500 | 0.0038 |
| 3 | intervention | entity_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | intervention | event_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | intervention | evidence_deletion | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | intervention | evidence_truncation | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | intervention | irrelevant_evidence | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | intervention | location_swap | 60 | 0 | 1 | 0.0000 | 1.0000 | 0.0167 |
| 3 | intervention | none | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | intervention | paraphrase | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | intervention | polarity_flip | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | intervention | predicate_swap | 60 | 0 | 1 | 0.0000 | 1.0000 | 0.0167 |
| 3 | intervention | role_swap | 60 | 0 | 1 | 0.0000 | 1.0000 | 0.0167 |
| 3 | intervention | time_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |
| 3 | intervention | title_name_swap | 60 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 |

## Pair-ID bootstrap

| seed | metric | estimate | 95% CI | samples | unit |
|---:|---|---:|---:|---:|---|
| 1 | accuracy_delta | 0.0013 | [0.0000, 0.0038] | 1000 | pair_id |
| 1 | macro_f1_delta | 0.0014 | [0.0000, 0.0042] | 1000 | pair_id |
| 1 | support_precision_gain | 0.0042 | [0.0000, 0.0133] | 1000 | pair_id |
| 1 | support_recall_drop | 0.0000 | [0.0000, 0.0000] | 1000 | pair_id |
| 1 | downgrade_rate | 0.0043 | [0.0000, 0.0129] | 1000 | pair_id |
| 1 | pre_router_candidate_gate_fail_rate | 0.0043 | [0.0000, 0.0129] | 1000 | pair_id |
| 2 | accuracy_delta | 0.0000 | [0.0000, 0.0000] | 1000 | pair_id |
| 2 | macro_f1_delta | 0.0000 | [0.0000, 0.0000] | 1000 | pair_id |
| 2 | support_precision_gain | 0.0000 | [0.0000, 0.0000] | 1000 | pair_id |
| 2 | support_recall_drop | 0.0000 | [0.0000, 0.0000] | 1000 | pair_id |
| 2 | downgrade_rate | 0.0000 | [0.0000, 0.0000] | 1000 | pair_id |
| 2 | pre_router_candidate_gate_fail_rate | 0.0000 | [0.0000, 0.0000] | 1000 | pair_id |
| 3 | accuracy_delta | 0.0038 | [0.0000, 0.0090] | 1000 | pair_id |
| 3 | macro_f1_delta | 0.0049 | [0.0000, 0.0110] | 1000 | pair_id |
| 3 | support_precision_gain | 0.0087 | [0.0000, 0.0198] | 1000 | pair_id |
| 3 | support_recall_drop | 0.0000 | [0.0000, 0.0000] | 1000 | pair_id |
| 3 | downgrade_rate | 0.0128 | [0.0000, 0.0283] | 1000 | pair_id |
| 3 | pre_router_candidate_gate_fail_rate | 0.0128 | [0.0000, 0.0283] | 1000 | pair_id |
