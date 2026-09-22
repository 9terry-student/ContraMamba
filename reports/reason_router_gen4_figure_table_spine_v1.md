# ContraMamba Gen4 — Main Figure and Table Spine v1

## Status

- Status: STATIC FIGURE/TABLE SPECIFICATION
- Evidence HEAD: a55c7e935f2539c49a43edf538e00a7a83951226
- Scientific execution: CLOSED
- New endpoints/tests/p-values: NONE

## Figure 1 — Central question and causal reconstruction

- Show the scale-local causal reconstruction/intervention pipeline.
- Show Mamba-130M, 370M, and 1.4B.
- Dominant ranks are checkpoint-local: 130M P3, 370M P3, 1.4B P5.
- Central schematic: causal-role recurrence does not guarantee geometric invariance or downstream functional invariance.

## Figure 2 — Prospective validation of the 130M causal component

- External-generator PP3 transport.
- PP3 versus response-blind max-separation PP5 specificity.
- Matched-control necessity.
- Restoration sufficiency relative to PP5 replacement.
- Finite-epsilon robustness at 0.025, 0.0125, and 0.00625.
- Do not describe PP5 as a random-direction baseline.

## Figure 3 — CORE-STABLE / RESIDUAL-PLASTIC

- 370M dominant/control: P3/P5.
- 1.4B dominant/control: P5/P4.
- Show checkpoint-local residual coefficient-mass and signed-effect profiles.
- Residual cross-checkpoint comparison is descriptive; no cross-scale p-value.
- Do not imply semantic identity of same-numbered planes.

## Figure 4 — Cross-block geometric reorientation

- Canonical 1.4B site versus prospectively fixed adjacent +1 site.
- Show transported rank-two geometry and large principal angles.
- Show D_CAN, D_ADJ, and D_TRANSPORT together.
- Primary message: transport shifts the adjacent response toward canonical but does not restore a positive canonical-like response.
- Do not report an explained or mediated percentage.

## Figure 5 — Matched downstream readout reversal

- Three-checkpoint readout means are context only; do not fit a scaling curve.
- Inferential centerpiece: matched 370M versus 1.4B N=300 comparison.
- Owned means: +5.09e-4 versus -1.20e-3.
- Paired mean difference: +1.70e-3; paired SD 2.74e-3.
- t(299)=10.77; one-sided p=2.22e-23.
- Distinguish Delta_L_owned from Delta_L_forward = 2 * Delta_L_owned.
- Add direct behavioral bridge and stagewise localization as supporting panels.
- Do not imply inference over model seeds.

## Main Table 1 — External transfer and control boundary

- AVeriTeC 130M: D_EXT +2.31e-4, dz +0.120, Holm p 0.0104.
- AVeriTeC 370M: D_EXT +4.56e-5, dz +0.106, Holm p 0.0114.
- AVeriTeC 1.4B: D_EXT -7.85e-5, dz -0.067, p 0.0757; negative transfer not established.
- Fixed-mirror 370M steering: 0 corrections, 0 damages, 0 net accuracy change; useful steering not established.
- State explicitly that AVeriTeC uses gold evidence and does not evaluate retrieval or benchmark superiority.

## Main-paper rules

1. Use manuscript precision rather than raw full-precision floats.
2. Preserve matched versus unmatched cohort distinctions.
3. No fitted model-size scaling curve or threshold.
4. No cross-seed generalization.
5. Null and failed gates remain visible.
6. Captions distinguish inferential from descriptive evidence.
7. Factor-2 diagnostics, provenance, precursor nulls, and detailed residual tables move to appendix.

FIGURE_TABLE_SPINE_V1_READY = YES
