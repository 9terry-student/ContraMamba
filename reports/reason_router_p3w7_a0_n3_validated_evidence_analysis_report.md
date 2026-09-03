# P3-W7 A0 N=3 Validated-Evidence Analysis Report

## 1. Status

Status: READY.

This report materializes the authorized static P3-W7 A0 N=3 validated-evidence descriptive analysis. It is the planned A0 baseline replicate set with primary N = 3.

Training, evaluation, inference, model loading, checkpoint loading, Kaggle execution, recovery execution, merge, cherry-pick, staging, commit, and push were not authorized and were not performed.

## 2. Authority And Scope

Frozen N=3 validated-evidence analysis authority: `052d38bae4f795b0fef5ae802f35151a730a8126`.

Historical N=2 validated-evidence analysis used as structural reference only: `34a5df306a00a28492b08f666bf3f7ac06c26944`.

Result evidence commits:

- seed180 normal recovered A0 reference: `b32d73dfa49b6b9dfabf3093802904323cf679cd`
- seed181 REPLACEMENT_R1: `fb4f0e2c2a8382a642f1272b66f29552adaecb0e`
- seed182: `82739bdfc8eee184de10ed8f55434f203a6d59a5`

This is report-only static analysis over immutable commit-addressed result artifacts. The historical N=2 report remains immutable and is not rewritten, reinterpreted, or described as though it included seed180.

## 3. Exact N=3 Membership

| Member | Result commit | Namespace | Admission |
|---|---|---|---|
| seed180 | `b32d73dfa49b6b9dfabf3093802904323cf679cd` | `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0` | `PRIMARY_ADMISSIBLE_AS_NORMAL_RECOVERED_A0_REFERENCE_FOR_THIS_AUTHORIZED_N3_ANALYSIS` |
| seed181 REPLACEMENT_R1 | `fb4f0e2c2a8382a642f1272b66f29552adaecb0e` | `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0` | `PRIMARY_ADMISSIBLE_AS_SEED181_REPLACEMENT_R1` |
| seed182 | `82739bdfc8eee184de10ed8f55434f203a6d59a5` | `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0` | `PRIMARY_ADMISSIBLE` |

Primary N = 3 exactly. Original consumed seed181 is not included and remains INADMISSIBLE.

## 4. Immutable Evidence Sources

All evidence was read directly from immutable Git commits using commit-addressed reads. No result commit was merged, cherry-picked, checked out over the worktree, copied, imported, or materialized into the current tree.

| Member | Artifact | Observed size | Observed SHA256 | Verdict |
|---|---|---:|---|---|
| seed180 | `training_report.json` | 306114 | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` | MATCH |
| seed180 | `clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` | MATCH |
| seed180 | `training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` | MATCH |
| seed180 | `run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` | MATCH |
| seed180 | `A0_REFERENCE_AUDIT.json` | 3167 | `37db2a99f55346f0b91bd8f5a1e8b3b9134922b4c51e87babab29d6c73e52d7b` | MATCH |
| seed181 REPLACEMENT_R1 | `training_report.json` | 305374 | `6eddb2f101bd513b91befe1b6edefcd078cc61e5626db6d13876d9b85b198ff3` | MATCH |
| seed181 REPLACEMENT_R1 | `clean_dev_predictions.json` | 4840808 | `893534cd27d8df7bf5eb6a7fa888b17b4ee803bb1d2a8233d40c15f8cf62ae12` | MATCH |
| seed181 REPLACEMENT_R1 | `training_report_predictions.jsonl` | 3936706 | `b186af00684279095a2257ef46f826be281a92bb2da9b0b9ee8f157f5bdbc13c` | MATCH |
| seed181 REPLACEMENT_R1 | `run_provenance.json` | 68927 | `11885151854401107b82b05c49da009bf47d9f465d348bc13965d60facad4e10` | MATCH |
| seed182 | `training_report.json` | 305735 | `319e13bbda07363a334d0b6615b2c4074dfcf5d30d0c43e1f0735f269c2b5e3e` | MATCH |
| seed182 | `clean_dev_predictions.json` | 4842166 | `80205044dceed9b2131cd3caf06524f7869b1651577504dca1503605d0471036` | MATCH |
| seed182 | `training_report_predictions.jsonl` | 3938064 | `95aa57b9f14ff7119b19f0ec8e412bf5b4494ae325e73d4c0ef0df5b24e050e5` | MATCH |
| seed182 | `run_provenance.json` | 68388 | `8357efe9ff609b8a99f580aa47ecbe4d018b5a0d24668755369e6e65a6cab421` | MATCH |

Immutable evidence identity verdict: PASS.

## 5. Seed180 Recovery/Provenance Caveat

Seed180 `A0_REFERENCE_AUDIT.json` was verified at `b32d73dfa49b6b9dfabf3093802904323cf679cd`.

| Audit field | Observed value |
|---|---|
| `status` | `PASS` |
| `recovery_reference_status` | `RECOVERY_REFERENCE_AUDIT_PASS` |
| `standard_cm_wrapper_provenance` | `INCOMPLETE` |
| `provenance_disposition` | `RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE` |
| `source_execution_commit` | `2737c3c6116ae3766b469801f990e2c45ba9a55e` |
| `dataset_sha256_expected` / observed | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` / same |
| `sidecar_semantic_sha256_expected` / observed | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` / same |

Seed180 caveat verdict: PASS, with caveat preserved. Historical standard-CM wrapper provenance remains INCOMPLETE. This report does not normalize, upgrade, complete, repair, or weaken that caveat.

## 6. Shared A0 Envelope Verification

All three commit-addressed members match the required A0 envelope.

| Property | seed180 | seed181 REPLACEMENT_R1 | seed182 |
|---|---|---|---|
| architecture | `v6b_minimal` | `v6b_minimal` | `v6b_minimal` |
| backbone | `mamba` | `mamba` | `mamba` |
| model | `state-spaces/mamba-130m-hf` | `state-spaces/mamba-130m-hf` | `state-spaces/mamba-130m-hf` |
| encoder | frozen | frozen | frozen |
| split seed | 174 | 174 | 174 |
| training seed | 180 | 181 | 182 |
| arm | `A0` | `A0` | `A0` |
| router mode | `explicit_product` | `explicit_product` | `explicit_product` |
| gradient ownership | `joint` | `joint` | `joint` |
| effective reason loss | 0.0 | 0.0 | 0.0 |
| epochs completed | 20 | 20 | 20 |
| max length | 128 | 128 | 128 |
| dev ratio | 0.2 | 0.2 | 0.2 |
| learning rate | 0.001 | 0.001 | 0.001 |
| selection metric | `final_macro_f1` | `final_macro_f1` | `final_macro_f1` |
| class weighting | `none` | `none` | `none` |
| dataset SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` | same | same |
| controlled sidecar semantic SHA256 | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` | same | same |

Auxiliary/ranking objectives required neutral under formal A0 were verified neutral across all three: `weighted_label_loss=false`, `use_intervention_loss=false`, compatible positive margin disabled with weight 0.0, Stage174C clean pairwise disabled with weight 0.0, Stage175B support anchor disabled with weight 0.0, Stage177C frame pairwise disabled with weight 0.0, combined bridge row count 0, no external data used for training, and no external metrics used for threshold tuning.

No A0-reference prediction was consumed during any A0 execution: `reason_router_a0_reference_predictions = null` in all three commit-addressed `run_provenance.json` parsed argument records.

Shared-envelope verdict: PASS.

## 7. Observed Per-Seed Measurements

All values below come from immutable `training_report.json`, except selected checkpoint identity and provenance selected epoch, which come from immutable `run_provenance.json` or seed180 audit/provenance binding only. No checkpoint was opened, loaded, or deserialized.

| Metric | seed180 | seed181 REPLACEMENT_R1 | seed182 |
|---|---:|---:|---:|
| best/selected epoch | 20 | 18 | 16 |
| best dev macro F1 / selected final macro F1 | 0.756831509304464 | 0.7842949573612369 | 0.7987512386137077 |
| best dev accuracy / selected final accuracy | 0.8541666865348816 | 0.8902778029441833 | 0.9069444537162781 |
| frame accuracy | 0.8097222447395325 | 0.8291667103767395 | 0.8291667103767395 |
| predicate accuracy | 0.7388889193534851 | 0.7472222447395325 | 0.7486111521720886 |
| sufficiency accuracy | 1.0 | 1.0 | 1.0 |
| polarity accuracy entitled | 0.9944444894790649 | 1.0 | 1.0 |

Selected checkpoint identity:

| Member | SHA256 | Size bytes | Provenance selected epoch |
|---|---|---:|---:|
| seed180 | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` | 518269815 | 20 |
| seed181 REPLACEMENT_R1 | `3dbd7c32cc2d60b2de13da3a72cff05eaa080520f7cac076225c5a55870721ca` | 518269943 | 18 |
| seed182 | `212873153bc6cecf107e79a4ea86385033c7944a9af222d4984192b232803946` | 518269815 | 16 |

## 8. Per-Label Metrics

| Label | Metric | seed180 | seed181 REPLACEMENT_R1 | seed182 |
|---|---|---:|---:|---:|
| REFUTE | precision | 1.0 | 1.0 | 1.0 |
| REFUTE | recall | 0.989010989010989 | 1.0 | 1.0 |
| REFUTE | F1 | 0.994475138121547 | 1.0 | 1.0 |
| NOT_ENTITLED | precision | 0.894927536231884 | 0.8967297762478486 | 0.8974789915966387 |
| NOT_ENTITLED | recall | 0.9148148148148149 | 0.9648148148148148 | 0.9888888888888889 |
| NOT_ENTITLED | F1 | 0.9047619047619049 | 0.9295272078501338 | 0.9409691629955947 |
| SUPPORT | precision | 0.3974358974358974 | 0.6041666666666666 | 0.8235294117647058 |
| SUPPORT | recall | 0.34831460674157305 | 0.3258426966292135 | 0.3146067415730337 |
| SUPPORT | F1 | 0.3712574850299401 | 0.4233576642335766 | 0.45528455284552843 |

## 9. N=3 Descriptive Aggregate Metrics

For every row, N = 3 exactly. Mean is arithmetic mean. Sample SD uses denominator N - 1. Full stored numeric precision was used for arithmetic.

| Metric | mean | sample SD | min | max |
|---|---:|---:|---:|---:|
| best/selected epoch | 18 | 2.0 | 16 | 20 |
| final_macro_f1 | 0.7799592350931361 | 0.02129353873900824 | 0.756831509304464 | 0.7987512386137077 |
| final_accuracy | 0.8837963143984476 | 0.026979259580088556 | 0.8541666865348816 | 0.9069444537162781 |
| frame_accuracy | 0.8226852218310038 | 0.011226267469889908 | 0.8097222447395325 | 0.8291667103767395 |
| predicate_accuracy | 0.7449074387550354 | 0.005258250450880065 | 0.7388889193534851 | 0.7486111521720886 |
| sufficiency_accuracy | 1.0 | 0.0 | 1.0 | 1.0 |
| polarity_accuracy_entitled | 0.9981481631596884 | 0.0032074754947476543 | 0.9944444894790649 | 1.0 |
| REFUTE precision | 1.0 | 0.0 | 1.0 | 1.0 |
| REFUTE recall | 0.9963369963369964 | 0.006344508452633227 | 0.989010989010989 | 1.0 |
| REFUTE F1 | 0.998158379373849 | 0.0031897804927603552 | 0.994475138121547 | 1.0 |
| NOT_ENTITLED precision | 0.8963787680254571 | 0.0013114443178128582 | 0.894927536231884 | 0.8974789915966387 |
| NOT_ENTITLED recall | 0.9561728395061728 | 0.03778564430321701 | 0.9148148148148149 | 0.9888888888888889 |
| NOT_ENTITLED F1 | 0.9250860918692111 | 0.018507674639653693 | 0.9047619047619049 | 0.9409691629955947 |
| SUPPORT precision | 0.6083773252890899 | 0.21307796219411 | 0.3974358974358974 | 0.8235294117647058 |
| SUPPORT recall | 0.32958801498127344 | 0.017163204850021884 | 0.3146067415730337 | 0.34831460674157305 |
| SUPPORT F1 | 0.4166332340363484 | 0.042415215460131946 | 0.3712574850299401 | 0.45528455284552843 |

## 10. Prediction, Intervention, And Pairwise Diagnostics

These diagnostics describe A0 baseline behavior only. They are not evidence of A1, A2, A3, factorial, router/supervision/gradient-ownership causal effects, or promotion readiness.

Selected-epoch prediction distributions:

| Member | REFUTE | NOT_ENTITLED | SUPPORT |
|---|---:|---:|---:|
| seed180 | 90 | 552 | 78 |
| seed181 REPLACEMENT_R1 | 91 | 581 | 48 |
| seed182 | 91 | 595 | 34 |

Selected-epoch prediction distribution N=3 count aggregates:

| Class | mean | sample SD | min | max |
|---|---:|---:|---:|---:|
| REFUTE | 90.66666666666667 | 0.5773502691896257 | 90 | 91 |
| NOT_ENTITLED | 576 | 21.93171219946131 | 552 | 595 |
| SUPPORT | 53.333333333333336 | 22.479620400116488 | 34 | 78 |

Pairwise scalar diagnostic aggregates:

| Diagnostic | mean | sample SD | min | max |
|---|---:|---:|---:|---:|
| deletion_sufficiency_drop.mean | 0.9695519834756852 | 0.009242542611917549 | 0.9593147665262223 | 0.9772828966379166 |
| deletion_sufficiency_lower.pass_rate | 1.0 | 0.0 | 1.0 | 1.0 |
| entity_frame_drop.mean | 0.40299052459498247 | 0.04838827532224022 | 0.348336182286342 | 0.44037266162534555 |
| entity_frame_lower.pass_rate | 0.8111111111111111 | 0.01924500897298752 | 0.8 | 0.8333333333333334 |
| event_frame_drop.mean | 0.4362730123930507 | 0.03101070131376361 | 0.40072044941286245 | 0.4577472724020481 |
| event_frame_lower.pass_rate | 0.9166666666666666 | 0.016666666666666663 | 0.9 | 0.9333333333333333 |
| flip_entitlement_delta.mean | 0.8069856388701333 | 0.026222476215380663 | 0.7781791905562083 | 0.8294672081867854 |
| paraphrase_gate_delta.mean | 0.2889961649974187 | 0.04937909981176223 | 0.25810310890277227 | 0.3459458231925964 |
| paraphrase_preserved.pass_rate | 0.5055555555555555 | 0.009622504486493792 | 0.5 | 0.5166666666666667 |
| polarity_flip_preserved_and_reversed.pass_rate | 0.0 | 0.0 | 0.0 | 0.0 |
| predicate_coverage_drop.mean | 0.3141546923253271 | 0.04074982408452101 | 0.28624454339345295 | 0.36091701984405516 |
| predicate_disentangled.pass_rate | 0.1111111111111111 | 0.03849001794597505 | 0.06666666666666667 | 0.13333333333333333 |
| predicate_frame_delta.mean | 0.37495624224344887 | 0.039097418942791 | 0.3342699761192004 | 0.4122427805016438 |
| pairwise_active_group_count | 60 | 0.0 | 60 | 60 |
| pairwise_groups_considered | 60 | 0.0 | 60 | 60 |
| pairwise_groups_skipped_missing_none | 0 | 0.0 | 0 | 0 |
| pairwise_groups_skipped_missing_variant | 0 | 0.0 | 0 | 0 |
| truncation_sufficiency_drop.mean | 0.964757795797454 | 0.010973452433582635 | 0.9521554778019587 | 0.9722003767887751 |
| truncation_sufficiency_lower.pass_rate | 1.0 | 0.0 | 1.0 | 1.0 |

Pairwise boolean diagnostics were identical across all three where present: deletion and truncation sufficiency lower checks passed; entity/event frame lower, paraphrase preserved, polarity flip preserved-and-reversed, and predicate disentangled did not pass. The pairwise guard was enabled in all three reports.

Per-intervention prediction distributions:

| Intervention | seed180 | seed181 REPLACEMENT_R1 | seed182 |
|---|---|---|---|
| entity_swap | NOT_ENTITLED 56, SUPPORT 4 | NOT_ENTITLED 60 | NOT_ENTITLED 60 |
| event_swap | NOT_ENTITLED 56, SUPPORT 4 | NOT_ENTITLED 57, SUPPORT 3 | NOT_ENTITLED 59, SUPPORT 1 |
| evidence_deletion | NOT_ENTITLED 60 | NOT_ENTITLED 60 | NOT_ENTITLED 60 |
| evidence_truncation | NOT_ENTITLED 60 | NOT_ENTITLED 60 | NOT_ENTITLED 60 |
| irrelevant_evidence | NOT_ENTITLED 60 | NOT_ENTITLED 60 | NOT_ENTITLED 60 |
| location_swap | NOT_ENTITLED 57, SUPPORT 3 | NOT_ENTITLED 60 | NOT_ENTITLED 60 |
| none | NOT_ENTITLED 27, REFUTE 31, SUPPORT 2 | NOT_ENTITLED 29, REFUTE 31 | NOT_ENTITLED 29, REFUTE 31 |
| paraphrase | REFUTE 30, SUPPORT 30 | REFUTE 31, SUPPORT 29 | NOT_ENTITLED 1, REFUTE 31, SUPPORT 28 |
| polarity_flip | NOT_ENTITLED 31, REFUTE 29 | NOT_ENTITLED 31, REFUTE 29 | NOT_ENTITLED 31, REFUTE 29 |
| predicate_swap | NOT_ENTITLED 47, SUPPORT 13 | NOT_ENTITLED 54, SUPPORT 6 | NOT_ENTITLED 58, SUPPORT 2 |
| role_swap | NOT_ENTITLED 49, SUPPORT 11 | NOT_ENTITLED 54, SUPPORT 6 | NOT_ENTITLED 57, SUPPORT 3 |
| title_name_swap | NOT_ENTITLED 49, SUPPORT 11 | NOT_ENTITLED 56, SUPPORT 4 | NOT_ENTITLED 60 |

Per-intervention scalar diagnostics were structurally comparable for `entitlement_prob`, `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, and `polarity_margin` across all three. N=3 aggregate ranges are summarized here:

| Intervention | entitlement mean / SD | frame mean / SD | predicate mean / SD | sufficiency mean / SD | polarity margin mean / SD |
|---|---:|---:|---:|---:|---:|
| entity_swap | 0.07368574601908524 / 0.04844759975628505 | 0.21082883576552072 / 0.10438748838154871 | 0.3142970601717631 / 0.07217116390381163 | 0.9915206034978231 / 0.0019843377623400783 | 3.5835746924082437 / 0.3352592689965864 |
| event_swap | 0.05963659596939882 / 0.035934118239270924 | 0.1775463546315829 / 0.08677496221933219 | 0.29430626332759857 / 0.06293080825449736 | 0.9913074374198914 / 0.0020278244513686912 | 3.5633044242858887 / 0.3438565386932179 |
| evidence_deletion | 0.013035771436989307 / 0.008085344331906146 | 0.9852515459060669 / 0.0029790127351245112 | 0.9780364632606506 / 0.00274557965987962 | 0.013555861078202724 / 0.008485399145471185 | -1.440810203552246 / 0.9325538889829865 |
| evidence_truncation | 0.017063189297914505 / 0.009427997916610599 | 0.9751108686129252 / 0.011034349407865289 | 0.9627147316932678 / 0.02018495183809929 | 0.01835004674891631 / 0.010726923061023462 | -1.8914196491241455 / 0.7750537240576149 |
| irrelevant_evidence | 1.2720918675768189e-05 / 6.2214128275754415e-06 | 0.03494619702299436 / 0.013109990726204147 | 0.022650060554345448 / 0.006504846425281957 | 0.01574562241633733 / 0.003404826238136678 | -0.34768465409676236 / 0.26562649682898476 |
| location_swap | 0.08273117368419965 / 0.04598815460953044 | 0.22710716227690378 / 0.10113527091915912 | 0.3407457172870636 / 0.053588255723265234 | 0.9917035897572836 / 0.001922407739210004 | 3.5859321753184 / 0.3308873077018749 |
| none | 0.5010108153025309 / 0.033633697116074446 | 0.613819420337677 / 0.05614147334906029 | 0.6523842414220175 / 0.02979339172839075 | 0.9831078847249349 / 0.0033394479552877245 | -0.17828130597869554 / 0.159882888846772 |
| paraphrase | 0.7552985747655233 / 0.06280113942192823 | 0.8979917764663696 / 0.06526619396416586 | 0.8438074787457784 / 0.03655453594297691 | 0.9773650964101156 / 0.010130712012573113 | -0.10927513303856055 / 0.08058962786191677 |
| polarity_flip | 0.4632914662361145 / 0.03329446965863662 | 0.5694367488225301 / 0.05476042625735614 | 0.6269057194391886 / 0.033279520427736985 | 0.9832931756973267 / 0.002948351390405156 | 0.15012963426609835 / 0.1759115427817461 |
| predicate_swap | 0.11500445380806923 / 0.05865326816065715 | 0.3066217104593913 / 0.11442644797669357 | 0.33822953701019287 / 0.07053849823547285 | 0.991782526175181 / 0.001887848242360352 | 3.5934173266092935 / 0.3275004904893391 |
| role_swap | 0.10641913985212643 / 0.04963229203385894 | 0.2524271408716838 / 0.0990491018314169 | 0.35360581676165265 / 0.05714505574409892 | 0.991746743520101 / 0.0018993244982122386 | 3.5866506099700928 / 0.32256907117340466 |
| title_name_swap | 0.08806744900842507 / 0.06420433510910235 | 0.22832385698954263 / 0.12180967613337058 | 0.31406691670417786 / 0.09434040686558347 | 0.9909758567810059 / 0.0021547516479379267 | 3.524137814839681 / 0.3230569675643135 |

## 11. N3_AGGREGATE_NOT_COMPARABLE Items

No required scalar metric listed in Sections 7 through 10 silently reduced to N < 3.

`N3_AGGREGATE_NOT_COMPARABLE`: pairwise `*.passed` fields and `stage45b3_pairwise_check_guard_enabled` are boolean verdict fields rather than continuous scalar diagnostics. They are reported as identical per-field pass/fail states instead of mean/sample-SD aggregates.

`N3_AGGREGATE_NOT_COMPARABLE`: nested empty dictionary `pairwise_missing_variant_counts` has no scalar child values; all three members report no missing variants via the comparable scalar field `pairwise_groups_skipped_missing_variant = 0`.

## 12. Evidence/Admissibility Conclusions

The planned A0 baseline replicate set is complete at N=3.

The exact membership is seed180, seed181 REPLACEMENT_R1, and seed182. Seed180 is admitted for this authorized N=3 analysis with its recovery/provenance caveat preserved. Seed181 is a separately authorized replacement execution and is not recovery of the consumed original seed181 attempt. Seed182 remains the original validated result.

Descriptive A0 baseline central tendency and dispersion can be reported for structurally comparable metrics across all three members.

## 13. Scientific Interpretation Boundary

This report establishes descriptive A0 baseline behavior only. It does not establish P2 mechanism superiority, router/supervision/gradient-ownership causal effects, statistical significance, or release/promotion readiness.

SUPPORT remains the weakest label in this A0 descriptive set by mean F1 and mean recall. NOT_ENTITLED is the dominant selected prediction class. REFUTE remains near-perfect by the reported selected clean-dev metrics. These are descriptive A0 observations only.

## 14. Factorial/A1-A3 Boundary

This report does not authorize, perform, or conclude any factorial analysis. It does not decide A1, A2, or A3 release. It does not infer A1/A2/A3 effects from A0 diagnostics.

After this N=3 report is independently verified and frozen, factorial authority may be reassessed separately by the controller.

## Validation Note

Final report SHA256, byte size, newline counts, `git diff --check`, git status, and git diff state are intentionally reported outside this file to avoid self-referential file-hash content.
