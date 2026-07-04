# Stage107 - ContraMamba-vNext Architecture Scope Lock

## Decision

`STAGE107_VNEXT_ARCHITECTURE_SCOPE_LOCKED`

## Summary

| stage    | decision                                 | current_primary                                               | stage107_type                       | promoted_candidate   | main_conclusion                                                                                                                                                                                       | vnext_target                                                                                 | core_bottleneck                                                                                | stage108_next                                                | do_not_do_next                                                                                                                                                                                         |
|:---------|:-----------------------------------------|:--------------------------------------------------------------|:------------------------------------|:---------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage107 | STAGE107_VNEXT_ARCHITECTURE_SCOPE_LOCKED | Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery | report_only_architecture_scope_lock |                      | Do not proceed to broad implementation. Stage107 locks ContraMamba-vNext as an entitlement-first state-space epistemic judgment controller and defers implementation to Stage108+ after static audit. | ContraMamba-vNext as a state-space epistemic judgment controller, not a flat 3-way verifier. | Final entitlement/routing cannot safely recover latent SUPPORT signal without external tuning. | static repo audit for v7/vNext salvage before any code patch | ["Do not add more synthetic bridge data.", "Do not tune thresholds on VitaminC.", "Do not build RAG or hallucination controller in one stage.", "Do not patch architecture before static repo audit."] |

## Closed evidence from Stage71-106

| evidence                                | summary                                                                                                                                                                        | source_stages               | implication_for_vnext                                                                    |
|:----------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------|:-----------------------------------------------------------------------------------------|
| Bridge append limitation                | Synthetic bridge append branches can improve REFUTE recall or external macro-F1, but repeatedly suppress SUPPORT entitlement.                                                  | Stage88/92/95/97/99/100/106 | Do not continue bridge-append search as the main mechanism.                              |
| External threshold signal               | External-label threshold sweep found recoverable latent SUPPORT signal in Stage92C probabilities.                                                                              | Stage102/103                | Representation contains useful signal, but external-tuned thresholds are not promotable. |
| Clean-only threshold underdetermination | Clean-dev already has perfect SUPPORT/REFUTE recall, so clean-only threshold selection becomes underdetermined. No strict nontrivial clean-safe threshold candidate was found. | Stage104/104B               | Final routing must be structurally redesigned; threshold-only transfer is insufficient.  |
| Portable clean delta weak/negative      | Clean-derived portable delta slightly improves Stage92C but remains below Stage73 on SUPPORT recall.                                                                           | Stage105                    | Do not promote threshold branch; use as design motivation only.                          |

## vNext bottlenecks

| bottleneck                            | description                                                                                                                      | required_fix                                                                  |
|:--------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
| Final entitlement/routing instability | The model can carry SUPPORT signal in probabilities, but final routing fails to safely recover it without external-label tuning. | Entitlement-first hierarchical routing rather than flat 3-way final softmax.  |
| NOT_ENTITLED collapse                 | NOT_ENTITLED currently acts as one class covering frame mismatch, predicate absence, insufficiency, ambiguity, and novelty.      | Decompose not-entitled reasons into explicit intermediate states.             |
| Bridge intervention non-locality      | Small synthetic data interventions shift global SUPPORT/REFUTE/NE balance rather than selectively fixing errors.                 | Mechanism-level constraints and routing structure before further data append. |
| Clean-dev saturation                  | Clean-dev is too easy for SUPPORT/REFUTE and cannot select robust thresholds.                                                    | Use clean-dev preservation as a gate, not as the sole design signal.          |

## Stage107 scope

| category     | item                                                                                         |
|:-------------|:---------------------------------------------------------------------------------------------|
| in_scope     | Summarize Stage71-106 failure evidence into vNext requirements.                              |
| in_scope     | Lock vNext architectural identity and output schema direction.                               |
| in_scope     | Define module order: Frame → Predicate → Sufficiency/Entitlement → Polarity → Final routing. |
| in_scope     | Define what is explicitly out of scope for the next implementation patch.                    |
| in_scope     | Define Stage108-112 sequence before any coding or training.                                  |
| out_of_scope | No model implementation.                                                                     |
| out_of_scope | No runner patch.                                                                             |
| out_of_scope | No training.                                                                                 |
| out_of_scope | No clean-dev run.                                                                            |
| out_of_scope | No external/VitaminC evaluation.                                                             |
| out_of_scope | No RAG integration.                                                                          |
| out_of_scope | No hallucination benchmark expansion.                                                        |
| out_of_scope | No ask-back/retrieve-more action implementation.                                             |
| out_of_scope | No synthetic bridge generation.                                                              |
| out_of_scope | No threshold tuning.                                                                         |

## vNext module order

|   order | module                          |
|--------:|:--------------------------------|
|       1 | Frame state first               |
|       2 | Predicate coverage second       |
|       3 | Sufficiency / entitlement third |
|       4 | Polarity only after entitlement |
|       5 | Final routing last              |

## Loss direction

| loss_component                      | status                              | purpose                                                                                       |
|:------------------------------------|:------------------------------------|:----------------------------------------------------------------------------------------------|
| final_label_ce                      | keep but demote from sole objective | Still needed for 3-way compatibility.                                                         |
| entitlement_consistency             | required                            | Prevent SUPPORT/REFUTE when frame/predicate/sufficiency states are invalid.                   |
| polarity_conditioned_on_entitlement | required                            | Polarity should matter only after entitlement is established.                                 |
| not_entitled_reason_supervision     | future or synthetic-controlled      | Split NE into frame/predicate/sufficiency/ambiguity/novelty reasons.                          |
| routing_margin                      | candidate                           | Separate entitled SUPPORT/REFUTE from non-entitled states without global SUPPORT suppression. |

## Promotion criteria direction

| criterion                      | required   | description                                                                |
|:-------------------------------|:-----------|:---------------------------------------------------------------------------|
| clean_dev_preservation         | True       | Must preserve clean-dev before any external diagnostic.                    |
| support_recall_floor           | True       | No top-line promotion if SUPPORT recall collapses.                         |
| refute_recall_floor            | True       | REFUTE gain cannot come solely by over-suppressing SUPPORT.                |
| false_NE_control               | True       | False NOT_ENTITLED on entitled examples must be tracked separately.        |
| false_entitlement_control      | True       | Avoid over-entitling true NOT_ENTITLED examples.                           |
| external_diagnostic_non_tuning | True       | External labels can diagnose but not tune or select deployable thresholds. |

## Next stage plan

| stage    | name                                   | type                         | goal                                                                                                                        | forbidden                                              |
|:---------|:---------------------------------------|:-----------------------------|:----------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------|
| Stage108 | static repo audit for v7/vNext salvage | static audit only            | Inspect existing v7_hierarchical / modeling files / runner flags / output export paths. Decide salvage vs new vNext file.   | training, smoke run, external eval, dataset generation |
| Stage109 | minimal vNext architecture patch       | code patch                   | Implement or salvage minimal entitlement-first logit composition. No new benchmark and no run commands inside agent prompt. | full training, external eval, bridge append            |
| Stage110 | plumbing validation                    | static/local validation only | Check imports/help/config compatibility after Stage109 patch.                                                               | training and evaluation                                |
| Stage111 | clean-dev preservation run             | Kaggle run by user           | Run main clean train/dev only after plumbing is valid.                                                                      | external diagnostic until clean gate passes            |
| Stage112 | external diagnostic                    | diagnostic only              | Apply external/VitaminC diagnostic only if Stage111 clean preservation passes.                                              | external tuning or threshold selection for promotion   |

## Checks

| check                                       | pass   |
|:--------------------------------------------|:-------|
| stage106_closed                             | True   |
| stage71_primary_preserved                   | True   |
| stage107_report_only                        | True   |
| no_training                                 | True   |
| no_data_generation                          | True   |
| no_external_eval                            | True   |
| no_code_patch                               | True   |
| vnext_scope_locked                          | True   |
| stage108_static_audit_required_before_patch | True   |
