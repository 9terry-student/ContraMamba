# Frozen Mamba-1 main-paper figures

Base scientific authority: `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`.
Coordinate-free geometry paper snapshot: `2e65dd887ba4159d4024d85b8996306cfadec023`.
Objective pair-resampling source: `reports/reason_router_gen4_mamba1_iclr_pair_resampling_robustness_v1/objective_bootstrap.csv` @ `8423ef9d1325c007ecd31849a668131197e2214d`.

Run from repository root:

```powershell
python paper/iclr2027/scripts/build_figures.py
python -B paper/iclr2027/scripts/check_figure_values.py --rebuild
```

The builder reads only pinned Git blobs. No model, tokenizer, training, evaluation, forward/backward pass, or new statistical test is run.

## Main-figure design

- Fig. 1: conceptual separation of causal role, native geometry, and objective-conditioned use.
- Fig. 2: reader-facing 130M causal chain; endpoint notation and exact inferential statistics remain in Appendix A.3.
- Fig. 3: scale-local causal recurrence plus frozen coordinate-insensitive XG2/XG4 centered-linear CKA matrices.
- Fig. 3 no longer uses kernel mean-square or leading-plane eigenvalue as the main evidence for non-invariance; those scalar native-geometry measurements remain supporting evidence in the appendix.
- Cosine-RSM Pearson matrices are a frozen secondary coordinate-insensitive check and are reported in Appendix A.2.
- Fig. 4: objective-conditioned point means and frozen percentile 95% pair-resampling intervals; prospective chronology is kept in the appendix.

## Frozen coordinate-free geometry

### XG2 centered linear CKA

| | 130M | 370M | 790M | 1.4B | 2.8B |
|---|---:|---:|---:|---:|---:|
| 130M | 1.000000 | 0.742546 | 0.435747 | 0.342510 | 0.555414 |
| 370M | 0.742546 | 1.000000 | 0.564059 | 0.504483 | 0.649344 |
| 790M | 0.435747 | 0.564059 | 1.000000 | 0.663678 | 0.639963 |
| 1.4B | 0.342510 | 0.504483 | 0.663678 | 1.000000 | 0.532185 |
| 2.8B | 0.555414 | 0.649344 | 0.639963 | 0.532185 | 1.000000 |

### XG4 centered linear CKA

| | 130M | 370M | 790M | 1.4B | 2.8B |
|---|---:|---:|---:|---:|---:|
| 130M | 1.000000 | 0.542295 | 0.451862 | 0.481304 | 0.544277 |
| 370M | 0.542295 | 1.000000 | 0.558569 | 0.436834 | 0.590983 |
| 790M | 0.451862 | 0.558569 | 1.000000 | 0.444332 | 0.392717 |
| 1.4B | 0.481304 | 0.436834 | 0.444332 | 1.000000 | 0.455698 |
| 2.8B | 0.544277 | 0.590983 | 0.392717 | 0.455698 | 1.000000 |

### Secondary cosine-RSM Pearson

| Family | Scale pair | Pearson |
|---|---|---:|
| XG2 | 130M--370M | 0.716312 |
| XG2 | 130M--790M | 0.398941 |
| XG2 | 130M--1.4B | 0.295730 |
| XG2 | 130M--2.8B | 0.522254 |
| XG2 | 370M--790M | 0.541258 |
| XG2 | 370M--1.4B | 0.474489 |
| XG2 | 370M--2.8B | 0.634246 |
| XG2 | 790M--1.4B | 0.640851 |
| XG2 | 790M--2.8B | 0.620152 |
| XG2 | 1.4B--2.8B | 0.504037 |
| XG4 | 130M--370M | 0.509857 |
| XG4 | 130M--790M | 0.431574 |
| XG4 | 130M--1.4B | 0.452640 |
| XG4 | 130M--2.8B | 0.511619 |
| XG4 | 370M--790M | 0.533989 |
| XG4 | 370M--1.4B | 0.403381 |
| XG4 | 370M--2.8B | 0.545762 |
| XG4 | 790M--1.4B | 0.412623 |
| XG4 | 790M--2.8B | 0.337620 |
| XG4 | 1.4B--2.8B | 0.382808 |

## Supporting historical scalar geometry

| Scale | mu_k2 | lambda1 |
|---|---:|---:|
| 130M | 0.027899337798707836 | 0.8706181418918275 |
| 370M | 0.024623525099917637 | 0.9671797709152048 |
| 790M | 0.02384271108497352 | 0.8929611457472746 |
| 1.4B | 0.014838786182259655 | 0.8890412240357709 |
| 2.8B | 0.00985710670421181 | 0.8932258269422545 |

## Figure 4 objective means

| Scale | Task forward-equivalent | Vanilla next-token |
|---|---:|---:|
| 130M | 0.002240732073064253 | -0.05641684638644803 |
| 370M | 0.0010179127037708347 | -0.0019753191100731616 |
| 790M | 0.004553422899712989 | -0.01596436428883387 |
| 1.4B | -0.0023908950117724477 | 0.007833182803823639 |
| 2.8B | -0.0001483499070046979 | 0.016448402252375247 |

## Exact source paths by figure

### fig1

- `reports/reason_router_gen4_pp3_transport_specificity_necessity_restoration_sufficiency_mechanism_synthesis.md` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_synthesis.md` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba1_five_scale_delta_l_descriptive_synthesis.md` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.md` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba130m_vanilla_lm_completeness_analysis_v1/mamba130m_completeness_analysis.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`

### fig2

- `reports/reason_router_gen4_pp3_transport_specificity_necessity_restoration_sufficiency_mechanism_synthesis.md` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_primary_inference_report_candidate.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_seed181_behavioral_restoration_bridge_analysis_retry2.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba130m_readout_behavior_pair_merge_v1/descriptive_analysis.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba130m_readout_behavior_pair_merge_v1/pair_level_merge.jsonl` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`

### fig3

- `reports/reason_router_gen4_pp3_transport_specificity_necessity_restoration_sufficiency_mechanism_synthesis.md` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_synthesis.md` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba130m_vanilla_lm_completeness_analysis_v1/mamba130m_completeness_analysis.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba370m_confirmation_runs/g4k-mamba370-confirmation-xg1-3301-3600-2gpu-ac69e80/confirmation_inference.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba790m_confirmation_runs/g4k-mamba790m-confirmation-xg1-7201-7500-p2-p5-2gpu-d3117c1-fresh/confirmation_inference.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba14b_confirmation_runs/g4k-mamba14b-confirmation-xg1-4201-4500-2gpu-15304c1/confirmation_inference.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba28b_confirmation_runs/g4k-mamba28b-confirmation-xg1-6301-6600-p3-p5-2gpu-50a63a3/confirmation_inference.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `paper/iclr2027/frozen_sources/coordinate_free_geometry_v1/coordinate_free_geometry_result.json` @ `2e65dd887ba4159d4024d85b8996306cfadec023`

### fig4

- `reports/reason_router_gen4_mamba1_five_scale_delta_l_descriptive_synthesis.md` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.md` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba130m_vanilla_lm_completeness_analysis_v1/mamba130m_completeness_analysis.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.json` @ `3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69`
- `reports/reason_router_gen4_mamba1_iclr_pair_resampling_robustness_v1/objective_bootstrap.csv` @ `8423ef9d1325c007ecd31849a668131197e2214d`

## Final publication renderer

Figures 1--4 use the independently validated final vector redesign. Scientific values remain sourced by `figure_data_source.py`; production SVG/PDF/PNG bytes are byte-identical to the validated candidate after filename mapping.

The validated render used Matplotlib 3.11.2. The immediate promotion intentionally did not re-render under Python 3.13 because the existing isolated NumPy wheel was built for CPython 3.12. This ABI issue does not alter the promoted figure bytes or frozen scientific data.
