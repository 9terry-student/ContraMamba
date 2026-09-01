# P3-W7 A0 Validated-Evidence Analysis Report

Status: READY.

THIS IS NOT THE PLANNED 3-SEED A0 AGGREGATE.

This report materializes the authorized static A0 validated-evidence analysis for the explicit primary membership only:

- seed181 REPLACEMENT_R1
- seed182

Primary N = 2. Seed180 is excluded from all calculations in this report.

## Authority And Scope

Current HEAD is `759124743a9441a4c1811912770c9389fe7432f6`, which is the frozen P3-W7-A0 validated-evidence analysis authority commit. The authority artifact present at that HEAD is `reports/reason_router_p3w7_a0_validated_evidence_analysis_authority_spec_candidate.md`.

Initial git status before report creation:

```text
## p3w7-a0-validated-evidence-analysis
```

Analysis phase: authorized static A0 validated-evidence analysis.

Training/evaluation/model/checkpoint execution: not authorized and not performed. This report uses static JSON parsing and arithmetic over already validated stored metrics only.

## Immutable Evidence Sources

| Seed member | Result commit | Source artifact | Use |
|---|---:|---|---|
| seed181 REPLACEMENT_R1 | `fb4f0e2c2a8382a642f1272b66f29552adaecb0e` | `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/training_report.json` | Selected metric evidence and diagnostics |
| seed181 REPLACEMENT_R1 | `fb4f0e2c2a8382a642f1272b66f29552adaecb0e` | `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/run_provenance.json` | Selected checkpoint identity and runtime provenance |
| seed182 | `82739bdfc8eee184de10ed8f55434f203a6d59a5` | `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/training_report.json` | Selected metric evidence and diagnostics |
| seed182 | `82739bdfc8eee184de10ed8f55434f203a6d59a5` | `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/run_provenance.json` | Selected checkpoint identity and runtime provenance |

No mutable working-copy result files were used as evidence for metrics or checkpoint identity.

## Envelope Verification

Both primary members match the required scientific envelope:

| Required property | seed181 REPLACEMENT_R1 JSON key path | seed181 REPLACEMENT_R1 value | seed182 JSON key path | seed182 value |
|---|---|---:|---|---:|
| architecture | `training_report.json:architecture` / `configuration.architecture` | `v6b_minimal` | `training_report.json:architecture` / `configuration.architecture` | `v6b_minimal` |
| backbone | `configuration.backbone` | `mamba` | `configuration.backbone` | `mamba` |
| model | `configuration.model_name` | `state-spaces/mamba-130m-hf` | `configuration.model_name` | `state-spaces/mamba-130m-hf` |
| split seed | `split_seed_contract.resolved_split_seed` / `configuration.configured_split_seed` | `174` | `split_seed_contract.resolved_split_seed` / `configuration.configured_split_seed` | `174` |
| training seed | `split_seed_contract.training_seed` / `configuration.training_seed` / `configuration.seed` | `181` | `split_seed_contract.training_seed` / `configuration.training_seed` / `configuration.seed` | `182` |
| arm | `configuration.reason_router_p2.contract.arm` | `A0` | `configuration.reason_router_p2.contract.arm` | `A0` |
| router mode | `configuration.reason_router_p2.contract.router_mode` | `explicit_product` | `configuration.reason_router_p2.contract.router_mode` | `explicit_product` |
| gradient ownership | `configuration.reason_router_p2.contract.gradient_ownership_mode` | `joint` | `configuration.reason_router_p2.contract.gradient_ownership_mode` | `joint` |
| effective reason loss | `configuration.reason_router_p2.contract.reason_loss_weight` | `0.0` | `configuration.reason_router_p2.contract.reason_loss_weight` | `0.0` |
| epochs | `run_provenance.json:finalization.completed_epochs`; command `--epochs 20` | `20` completed | `run_provenance.json:finalization.completed_epochs`; command `--epochs 20` | `20` completed |
| selection metric | `active_selection_rules.standard_clean_dev.metric`; `run_provenance.json:finalization.selected_checkpoint.selection_source`; command `--select-metric final_macro_f1` | `final_macro_f1` | `active_selection_rules.standard_clean_dev.metric`; `run_provenance.json:finalization.selected_checkpoint.selection_source`; command `--select-metric final_macro_f1` | `final_macro_f1` |
| class weighting | `configuration.class_weighting` | `none` | `configuration.class_weighting` | `none` |

Neutralized auxiliary flags/objectives are also consistent: `configuration.weighted_label_loss=false`, `configuration.use_intervention_loss=false`, `configuration.compatible_positive_margin.enabled=false`, `configuration.compatible_positive_margin.weight=0.0`, `configuration.stage174c_clean_pairwise.enabled=false`, `configuration.stage174c_clean_pairwise.weight=0.0`, `configuration.stage175b_support_anchor.enabled=false`, `configuration.stage175b_support_anchor.weight=0.0`, `configuration.stage177c_frame_pairwise.enabled=false`, `configuration.stage177c_frame_pairwise.weight=0.0`, all bridge row counts are zero, and external data/metrics are not used for training, calibration, checkpoint selection, threshold selection, or dev.

## Observed Per-Seed Measurements

All values below are observed per-seed measurements from immutable `training_report.json`, except selected checkpoint SHA256 and size, which come from immutable `run_provenance.json:finalization.selected_checkpoint`.

| Metric | JSON key path | seed181 REPLACEMENT_R1 | seed182 |
|---|---|---:|---:|
| best/selected epoch | `best_epoch`; cross-check `run_provenance.json:finalization.selected_checkpoint.selected_epoch` | 18 | 16 |
| best dev macro F1 | `best_dev_macro_f1`; selected value `best_dev_metrics.final_macro_f1` | 0.7842949573612369 | 0.7987512386137077 |
| best dev accuracy / selected final accuracy | `best_dev_acc`; selected value `best_dev_metrics.final_accuracy` | 0.8902778029441833 | 0.9069444537162781 |
| frame accuracy | `best_dev_metrics.frame_accuracy` | 0.8291667103767395 | 0.8291667103767395 |
| predicate accuracy | `best_dev_metrics.predicate_accuracy` | 0.7472222447395325 | 0.7486111521720886 |
| sufficiency accuracy | `best_dev_metrics.sufficiency_accuracy` | 1.0 | 1.0 |
| polarity accuracy entitled | `best_dev_metrics.polarity_accuracy_entitled` | 1.0 | 1.0 |
| selected checkpoint SHA256 | `run_provenance.json:finalization.selected_checkpoint.sha256` | `3dbd7c32cc2d60b2de13da3a72cff05eaa080520f7cac076225c5a55870721ca` | `212873153bc6cecf107e79a4ea86385033c7944a9af222d4984192b232803946` |
| selected checkpoint size bytes | `run_provenance.json:finalization.selected_checkpoint.size_bytes` | 518269943 | 518269815 |

### Per-Label Metrics

| Label | Metric | JSON key path | seed181 REPLACEMENT_R1 | seed182 |
|---|---|---|---:|---:|
| REFUTE | precision | `best_dev_metrics.per_label.REFUTE.precision` | 1.0 | 1.0 |
| REFUTE | recall | `best_dev_metrics.per_label.REFUTE.recall` | 1.0 | 1.0 |
| REFUTE | F1 | `best_dev_metrics.per_label.REFUTE.f1` | 1.0 | 1.0 |
| NOT_ENTITLED | precision | `best_dev_metrics.per_label.NOT_ENTITLED.precision` | 0.8967297762478486 | 0.8974789915966387 |
| NOT_ENTITLED | recall | `best_dev_metrics.per_label.NOT_ENTITLED.recall` | 0.9648148148148148 | 0.9888888888888889 |
| NOT_ENTITLED | F1 | `best_dev_metrics.per_label.NOT_ENTITLED.f1` | 0.9295272078501338 | 0.9409691629955947 |
| SUPPORT | precision | `best_dev_metrics.per_label.SUPPORT.precision` | 0.6041666666666666 | 0.8235294117647058 |
| SUPPORT | recall | `best_dev_metrics.per_label.SUPPORT.recall` | 0.3258426966292135 | 0.3146067415730337 |
| SUPPORT | F1 | `best_dev_metrics.per_label.SUPPORT.f1` | 0.4233576642335766 | 0.45528455284552843 |

## Descriptive Aggregate Metrics

N = 2 for every row in this aggregate table. Sample SD uses denominator N-1.

| Metric | seed181 REPLACEMENT_R1 | seed182 | mean | sample SD | min | max |
|---|---:|---:|---:|---:|---:|---:|
| best dev macro F1 | 0.7842949573612369 | 0.7987512386137077 | 0.7915230979874723 | 0.010222134504362046 | 0.7842949573612369 | 0.7987512386137077 |
| best dev accuracy / selected final accuracy | 0.8902778029441833 | 0.9069444537162781 | 0.8986111283302307 | 0.011785101780616189 | 0.8902778029441833 | 0.9069444537162781 |
| frame accuracy | 0.8291667103767395 | 0.8291667103767395 | 0.8291667103767395 | 0 | 0.8291667103767395 | 0.8291667103767395 |
| predicate accuracy | 0.7472222447395325 | 0.7486111521720886 | 0.7479166984558105 | 0.0009821058640008527 | 0.7472222447395325 | 0.7486111521720886 |
| sufficiency accuracy | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 |
| polarity accuracy entitled | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 |
| REFUTE precision | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 |
| REFUTE recall | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 |
| REFUTE F1 | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 |
| NOT_ENTITLED precision | 0.8967297762478486 | 0.8974789915966387 | 0.8971043839222437 | 0.0005297752536985475 | 0.8967297762478486 | 0.8974789915966387 |
| NOT_ENTITLED recall | 0.9648148148148148 | 0.9888888888888889 | 0.9768518518518519 | 0.017022941028565077 | 0.9648148148148148 | 0.9888888888888889 |
| NOT_ENTITLED F1 | 0.9295272078501338 | 0.9409691629955947 | 0.9352481854228643 | 0.008090684073387769 | 0.9295272078501338 | 0.9409691629955947 |
| SUPPORT precision | 0.6041666666666666 | 0.8235294117647058 | 0.7138480392156863 | 0.15511288459851963 | 0.6041666666666666 | 0.8235294117647058 |
| SUPPORT recall | 0.3258426966292135 | 0.3146067415730337 | 0.3202247191011236 | 0.007945020013332015 | 0.3146067415730337 | 0.3258426966292135 |
| SUPPORT F1 | 0.4233576642335766 | 0.45528455284552843 | 0.43932110853955253 | 0.022575719439698685 | 0.4233576642335766 | 0.45528455284552843 |

## Pairwise And Intervention Diagnostics

These diagnostics characterize the selected A0 baseline/control behavior only. They are not evidence for any A1/A2/A3 comparison.

Selected-epoch prediction distributions from `best_dev_metrics.prediction_distribution`:

| Seed member | REFUTE | NOT_ENTITLED | SUPPORT |
|---|---:|---:|---:|
| seed181 REPLACEMENT_R1 | 91 | 581 | 48 |
| seed182 | 91 | 595 | 34 |

Per-intervention selected-epoch distributions from `best_dev_interventions.<intervention>.prediction_distribution` show that both runs strongly route failure variants to `NOT_ENTITLED`, while SUPPORT remains the weak positive class. For both seeds, `entity_swap`, `evidence_deletion`, `evidence_truncation`, `irrelevant_evidence`, and `location_swap` are all 60/60 `NOT_ENTITLED`. The main observed differences are small SUPPORT leakage on several mutation types: seed181 has SUPPORT predictions on `event_swap` 3/60, `predicate_swap` 6/60, and `role_swap` 6/60; seed182 has SUPPORT predictions on `event_swap` 1/60, `predicate_swap` 2/60, and `role_swap` 3/60. Both seeds predict `none` as 31 REFUTE and 29 NOT_ENTITLED, and both predict `polarity_flip` as 29 REFUTE and 31 NOT_ENTITLED.

N = 2 for every row in this diagnostic aggregate table. Sample SD uses denominator N-1.

| Pairwise diagnostic | JSON key path | seed181 REPLACEMENT_R1 | seed182 | mean | sample SD | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| deletion sufficiency drop mean | `best_dev_pairwise_checks.deletion_sufficiency_drop.mean` | 0.9720582872629165 | 0.9593147665262223 | 0.9656865268945694 | 0.009011029929107905 | 0.9593147665262223 | 0.9720582872629165 |
| deletion sufficiency lower pass rate | `best_dev_pairwise_checks.deletion_sufficiency_lower.pass_rate` | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 |
| entity frame drop mean | `best_dev_pairwise_checks.entity_frame_drop.mean` | 0.42026272987325985 | 0.44037266162534555 | 0.4303176957493027 | 0.014219869111098465 | 0.42026272987325985 | 0.44037266162534555 |
| entity frame lower pass rate | `best_dev_pairwise_checks.entity_frame_lower.pass_rate` | 0.8 | 0.8333333333333334 | 0.8166666666666667 | 0.02357022603955158 | 0.8 | 0.8333333333333334 |
| event frame drop mean | `best_dev_pairwise_checks.event_frame_drop.mean` | 0.4503513153642416 | 0.4577472724020481 | 0.45404929388314486 | 0.005229731374797339 | 0.4503513153642416 | 0.4577472724020481 |
| event frame lower pass rate | `best_dev_pairwise_checks.event_frame_lower.pass_rate` | 0.9 | 0.9166666666666666 | 0.9083333333333333 | 0.01178511301977575 | 0.9 | 0.9166666666666666 |
| flip entitlement delta mean | `best_dev_pairwise_checks.flip_entitlement_delta.mean` | 0.8294672081867854 | 0.8133105178674062 | 0.8213888630270958 | 0.011424505286364132 | 0.8133105178674062 | 0.8294672081867854 |
| paraphrase gate delta mean | `best_dev_pairwise_checks.paraphrase_gate_delta.mean` | 0.3459458231925964 | 0.25810310890277227 | 0.30202446604768435 | 0.0621141789521671 | 0.25810310890277227 | 0.3459458231925964 |
| paraphrase preserved pass rate | `best_dev_pairwise_checks.paraphrase_preserved.pass_rate` | 0.5 | 0.5166666666666667 | 0.5083333333333333 | 0.011785113019775828 | 0.5 | 0.5166666666666667 |
| polarity flip preserved and reversed pass rate | `best_dev_pairwise_checks.polarity_flip_preserved_and_reversed.pass_rate` | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 0.0 |
| predicate coverage drop mean | `best_dev_pairwise_checks.predicate_coverage_drop.mean` | 0.29530251373847327 | 0.36091701984405516 | 0.32810976679126425 | 0.04639646221146308 | 0.29530251373847327 | 0.36091701984405516 |
| predicate disentangled pass rate | `best_dev_pairwise_checks.predicate_disentangled.pass_rate` | 0.13333333333333333 | 0.06666666666666667 | 0.1 | 0.04714045207910317 | 0.06666666666666667 | 0.13333333333333333 |
| predicate frame delta mean | `best_dev_pairwise_checks.predicate_frame_delta.mean` | 0.37835597010950245 | 0.4122427805016438 | 0.39529937530557313 | 0.02396159342106593 | 0.37835597010950245 | 0.4122427805016438 |
| truncation sufficiency drop mean | `best_dev_pairwise_checks.truncation_sufficiency_drop.mean` | 0.9699175328016281 | 0.9521554778019587 | 0.9610365053017934 | 0.012559669538074596 | 0.9521554778019587 | 0.9699175328016281 |
| truncation sufficiency lower pass rate | `best_dev_pairwise_checks.truncation_sufficiency_lower.pass_rate` | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 |

Pass/fail booleans from `best_dev_pairwise_checks.*.passed` are identical across both seeds for the named checks: deletion/truncation sufficiency lower pass, while entity/event frame lower, paraphrase preserved, polarity flip preserved-and-reversed, and predicate disentangled do not pass.

## Provenance And Limitations

Seed181 REPLACEMENT_R1 is not the original seed181 run. The original seed181 attempt is consumed and inadmissible under the current authorization. REPLACEMENT_R1 is the separately authorized admissible replacement and is the only seed181 member included here.

Seed180 is excluded from this primary analysis. Caveated recovery evidence exists for seed180, but seed180 is not admitted under the current explicit analysis authorization. No seed180 metric value enters any aggregate in this report.

This is a descriptive N=2 analysis of admissible A0 baseline/control results. It is underpowered for final multi-seed claims and is explicitly not the planned 3-seed A0 aggregate.

## Claims Established

The admissible N=2 A0 baseline/control evidence establishes:

- In the two authorized runs, selected clean-dev macro F1 is 0.7842949573612369 for seed181 REPLACEMENT_R1 and 0.7987512386137077 for seed182, with descriptive N=2 mean 0.7915230979874723.
- In the two authorized runs, selected final accuracy is 0.8902778029441833 for seed181 REPLACEMENT_R1 and 0.9069444537162781 for seed182, with descriptive N=2 mean 0.8986111283302307.
- Both selected A0 checkpoints show perfect REFUTE precision/recall/F1 and perfect selected sufficiency and entitled-polarity accuracy under the reported selected clean-dev metrics.
- SUPPORT remains the weakest reported label in the selected clean-dev metrics, with N=2 mean SUPPORT recall 0.3202247191011236 and mean SUPPORT F1 0.43932110853955253.
- Disabled pairwise/intervention objectives remained disabled; the pairwise diagnostics are observational checks over the A0 baseline/control outputs.

## Claims Not Established

This report does not establish:

- effectiveness of P2;
- superiority of conditional first-blocker routing;
- effectiveness of reason-specific CE;
- effectiveness of explicit-local gradient ownership;
- any A1, A2, or A3 result;
- any planned 3-seed A0 aggregate result;
- any claim using seed180 as an admitted primary analysis member.

## Validation Notes

All aggregate metrics in this report were recomputed from the two immutable source values shown in the tables above. Sample SD uses denominator N-1. No original seed181 artifact and no seed180 metric was used in these calculations.

Candidate report SHA256 and byte size are reported after materialization, because embedding the final file hash inside this same file would make the artifact self-referential.

Final validation commands and final git status are reported in the accompanying completion note.
