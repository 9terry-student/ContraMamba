# Frozen Mamba-1 main-paper figures

Scientific authority: `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`.

Run from repository root (Python with reportlab; Poppler pdftoppm on PATH):

```powershell
python paper/iclr2027/scripts/build_figures.py
python -B paper/iclr2027/scripts/check_figure_values.py --rebuild
git diff --check
```

The optional --rebuild check regenerates the artifacts and requires identical bytes for all eight outputs, the manifest, and this README.
Use a Python environment with reportlab, pypdf, and Pillow installed.
PDFs are 7 inches wide, entirely vector, using embedded Arial fonts. PNGs are rendered from the PDFs at 400 dpi.
The builder reads exact Git blobs at the pinned endpoint and rejects worktree source drift (CRLF conversion permitted).
It fails on missing paths/fields, identity mismatches, changed signs, or missing renderer. No run globbing or fallback.
PDF timestamps/IDs are deterministic. Reproducible byte hashes require the same ReportLab/Poppler versions.

## Scientific boundaries and source decisions

- Fig. 1 is a conceptual overview of the supplied narrative; arrows between scales denote study order, not parameter-count causation.
- Fig. 1 specifies a task-margin consequence and a shared frozen substrate within each scale, not one literal substrate across model sizes.
- Fig. 2 reads historical transport, necessity, and sufficiency means directly from uniquely anchored Markdown bullets: the raw JSON summaries contain no aggregate means. Specificity uses the frozen inference JSON. No means or correlations are recomputed.
- Fig. 2 Panel A uses equal-size, equal-color evidence cards with full frozen decimal means and cohort sizes. Distinct estimands/holdouts are not encoded as comparable magnitudes. Panels B/C are unchanged.
- Fig. 2 uses all 300 pair-level rows in frozen order; Delta L remains the owned readout used for the frozen correlation. Behavioral restoration is a task-margin endpoint, not an accuracy gain.
- Fig. 3 uses historical D_SUF at 130M, D_DOM at 370M, and D_CORE at 790M/1.4B/2.8B; these are labeled separately. The failed historical 370M joint residual criterion is preserved.
- Fig. 3 Panel A states the common selected-restored minus coefficient-matched-control contrast and retains historical D_SUF/D_DOM/D_CORE labels and separate confirmation protocols. Frozen support statuses are source-bound; no statistical decision is recomputed. Raw means remain in this provenance inventory only.
- Fig. 3 Panels B/C add the frozen historical 130M mu_k2 and lambda1 with open markers. The later homogeneous response-blind bridge covers 370M-2.8B only. No 130M spectral summary scalar or D_CORE value is reconstructed.
- Fig. 4 uses exact machine-readable comparator means, checked against the five-scale Delta-L synthesis (rounding tolerance 1e-12). Task and vanilla retain separate raw y-scales and zero lines.
- 130M vanilla LM is a later post-primary completeness extension. Original prospective control: 370M, 790M, 1.4B, 2.8B.
- No smoothing, fitted curve, normalization, new statistic, p-value, model loading, tokenizer, training, evaluation, forward or backward pass is used.
- Every numerical data value, including scatter coordinates and pair identities, has a source locator in figure_data_manifest.json. Axis ticks, display sizes, and rounding are presentation choices.
- Figures use embedded Arial/ASCII text; rendered output was visually inspected.

## Exact extracted values

### Figure 2

| Endpoint | Frozen mean |
|---|---:|
| C_PP3 | 7.120276192878565e-08 |
| D_SPEC | 2.3040673691981465e-08 |
| D_NEC | 4.7414371121837106e-08 |
| D_SUF | 4.078872356598753e-08 |
| D_BEH | 0.008332191656033197 |

- Pearson: `0.617548086733788`
- Spearman: `0.8324696941077123`
- Sign agreement: `0.8233333333333334`

### Figure 3

| Scale | Selected/control | Causal mean | mu_k2 | lambda1 |
|---|---|---:|---:|---:|
| 130M | P3/P5 | 4.078872356598753e-08 | 0.027899337798707836 | 0.8706181418918275 |
| 370M | P3/P5 | 3.974290010502882e-08 | 0.024623525099917637 | 0.9671797709152048 |
| 790M | P2/P5 | 7.193077507846526e-08 | 0.02384271108497352 | 0.8929611457472746 |
| 1.4B | P5/P4 | 9.282848764823318e-09 | 0.014838786182259655 | 0.8890412240357709 |
| 2.8B | P3/P5 | 4.1159870535991766e-08 | 0.00985710670421181 | 0.8932258269422545 |

### Figure 4

| Scale | Task forward-equivalent | Vanilla TASK_MATCHED | Pearson | Spearman | Sign agreement |
|---|---:|---:|---:|---:|---:|
| 130M | 0.002240732073064253 | -0.05641684638644803 | 0.3207983196366968 | 0.20438093756597295 | 0.5566666666666666 |
| 370M | 0.0010179127037708347 | -0.0019753191100731616 | -0.45931009371803017 | 0.03460349559439549 | 0.6033333333333334 |
| 790M | 0.004553422899712989 | -0.01596436428883387 | 0.18618979990253204 | -0.03793553261702908 | 0.44333333333333336 |
| 1.4B | -0.0023908950117724477 | 0.007833182803823639 | -0.36640482568810084 | -0.18272247469416325 | 0.45666666666666667 |
| 2.8B | -0.0001483499070046979 | 0.016448402252375247 | 0.13304896234651936 | 0.09660996233291481 | 0.5733333333333334 |

## Exact source paths by figure

### fig1

- `reports/reason_router_gen4_pp3_transport_specificity_necessity_restoration_sufficiency_mechanism_synthesis.md`
- `reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_synthesis.md`
- `reports/reason_router_gen4_mamba1_five_scale_delta_l_descriptive_synthesis.md`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.md`
- `reports/reason_router_gen4_mamba130m_vanilla_lm_completeness_analysis_v1/mamba130m_completeness_analysis.json`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.json`

### fig2

- `reports/reason_router_gen4_pp3_transport_specificity_necessity_restoration_sufficiency_mechanism_synthesis.md`
- `reports/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_primary_inference_report_candidate.json`
- `reports/reason_router_gen4_seed181_behavioral_restoration_bridge_analysis_retry2.json`
- `reports/reason_router_gen4_mamba130m_readout_behavior_pair_merge_v1/descriptive_analysis.json`
- `reports/reason_router_gen4_mamba130m_readout_behavior_pair_merge_v1/pair_level_merge.jsonl`

### fig3

- `reports/reason_router_gen4_pp3_transport_specificity_necessity_restoration_sufficiency_mechanism_synthesis.md`
- `reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_synthesis.md`
- `reports/reason_router_gen4_mamba130m_vanilla_lm_completeness_analysis_v1/mamba130m_completeness_analysis.json`
- `reports/reason_router_gen4_mamba1_native_geometry_functional_coupling_static_bridge.md`
- `scripts/reason_router_gen4_k_directional_alignment_transport_core.py`
- `reports/reason_router_gen4_pp3_excluded_residual_static_analysis_7a6c30f.json`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.json`
- `reports/reason_router_gen4_mamba370m_confirmation_runs/g4k-mamba370-confirmation-xg1-3301-3600-2gpu-ac69e80/confirmation_inference.json`
- `reports/reason_router_gen4_mamba370m_geometry_preparation_runs/g4k-mamba370-geometry-xg2xg4-2gpu-d8e71ad-retry2/geometry_summary.json`
- `reports/reason_router_gen4_mamba790m_confirmation_runs/g4k-mamba790m-confirmation-xg1-7201-7500-p2-p5-2gpu-d3117c1-fresh/confirmation_inference.json`
- `reports/reason_router_gen4_mamba790m_geometry_preparation_runs/g4k-mamba790m-geometry-xg2xg4-2gpu-774983b-retry2/geometry_summary.json`
- `reports/reason_router_gen4_mamba14b_confirmation_runs/g4k-mamba14b-confirmation-xg1-4201-4500-2gpu-15304c1/confirmation_inference.json`
- `reports/reason_router_gen4_mamba14b_geometry_preparation_runs/g4k-mamba14b-geometry-xg2xg4-2gpu-c758d5e-retry1/geometry_summary.json`
- `reports/reason_router_gen4_mamba28b_confirmation_runs/g4k-mamba28b-confirmation-xg1-6301-6600-p3-p5-2gpu-50a63a3/confirmation_inference.json`
- `reports/reason_router_gen4_mamba28b_geometry_preparation_runs/g4k-mamba28b-geometry-xg2xg4-2gpu-7744d94/geometry_summary.json`

### fig4

- `reports/reason_router_gen4_mamba1_five_scale_delta_l_descriptive_synthesis.md`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.md`
- `reports/reason_router_gen4_mamba130m_vanilla_lm_completeness_analysis_v1/mamba130m_completeness_analysis.json`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.json`
