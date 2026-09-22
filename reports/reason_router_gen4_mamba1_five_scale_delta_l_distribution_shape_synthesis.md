# ContraMamba Gen4 Mamba-1 Five-Scale Delta-L Distribution-Shape Synthesis

## Status

`STATIC_FIVE_SCALE_DISTRIBUTION_SHAPE_SYNTHESIS`

This artifact is a CPU-only static synthesis of already frozen evidence.

It performs no model forward, backward pass, training, new hypothesis test, new p-value, threshold estimation, monotonicity test, rescue, re-selection, or row filtering.

All reported values use `Delta_L_owned`. For the frozen `G3-GROUP-D-HALF` graph, `Delta_L_forward_equivalent = 2 * Delta_L_owned`; therefore signs and sign fractions are invariant.

## Canonical frozen sources

- `reports/reason_router_gen4_mamba130m_readout_alignment_analysis_v1/readout_alignment_analysis.json`: `6ae8866d4c39cd1ffceb490c77ba9f8f207a14bce26059d96dcf845fa1946184`
- `reports/reason_router_gen4_mamba370m14b_readout_alignment_analysis_v1/readout_alignment_analysis.json`: `c5568b73e85d4e2f9b1ce52cc923533ca80435ea96d1e7484b248c907e99265b`
- `reports/reason_router_gen4_mamba790m_readout_alignment_analysis_v1/readout_alignment_analysis.json`: `9a73deb8f17ea62c19e53e5728c8baa0b5dbdd4fd063133b3313acf92ee5e929`
- `reports/reason_router_gen4_mamba28b_readout_alignment_analysis_v1/readout_alignment_analysis.json`: `555e3f23da20a29deff04b2e7fac1f279d14a61c80709c13affe28574c86f24f`
- `reports/reason_router_gen4_mamba370m14b_readout_alignment_raw_runs/g4k-gen4-readout-alignment-manifold-4ab7d55-r2-2t4/readout_alignment_items.jsonl`: `08a5ce10349260171c253199b4f3fbf02642edfda004b2bb4afff9e0ce072efa`

Source hashes above are verified from canonical Git blob bytes, not working-tree line-ending representations.

## Uniform pair-level distribution table

| Scale | Mean | Median | SD sample | Positive fraction | Q25 | Q75 | Min | Max | Mean sign | Median sign | Majority sign | Mean-majority mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 130M | 0.0011203660365321265 | 0.00026329389872246066 | 0.0062092171152280931 | 0.53666666666666663 | -0.0021958507353393216 | 0.0050189753523138762 | -0.011805185836611012 | 0.019247481027104857 | positive | positive | positive | false |
| 370M | 0.00050895635188541737 | 0.00011899476035213567 | 0.0016151749318805684 | 0.57666666666666666 | -0.00049074736007765707 | 0.00087152585597231164 | -0.0025008454091293326 | 0.005693937906959835 | positive | positive | positive | false |
| 790M | 0.0022767114498564943 | 0.00033527816508260237 | 0.005038501765613824 | 0.73666666666666669 | -2.481178610837958e-05 | 0.0026201407080446671 | -0.0022082947217932215 | 0.030217224034164408 | positive | positive | positive | false |
| 1.4B | -0.0011954475058862238 | -0.00045655454773016832 | 0.0025553833538824733 | 0.35333333333333333 | -0.0022409890267693607 | 0.00043847752197409632 | -0.012622477281337433 | 0.0028153091383682342 | negative | negative | negative | false |
| 2.8B | -7.4174953502348948e-05 | 5.4122475344360534e-05 | 0.0014912247559287139 | 0.55333333333333334 | -0.000151244318050565 | 0.00055468526412105756 | -0.0045173449133746281 | 0.0031233639821747813 | negative | positive | positive | true |

## Cell-level decomposition

| Scale | C0_SHAM mean | C0 positive fraction | C2_NAME mean | C2 positive fraction | Opposite cell-mean signs |
|---|---:|---:|---:|---:|---|
| 130M | -0.0031965390157730894 | 0.29999999999999999 | 0.0054372710888373416 | 0.79000000000000004 | true |
| 370M | -0.0012992047752502967 | 0.28333333333333333 | 0.0023171174790211314 | 0.70333333333333337 | true |
| 790M | 0.0018714933597737589 | 0.5 | 0.00268192953993923 | 0.72999999999999998 | false |
| 1.4B | -0.0016611404633400152 | 0.40666666666666668 | -0.00072975454843243249 | 0.37 | false |
| 2.8B | -0.00058783190159716664 | 0.42999999999999999 | 0.00043948199459246863 | 0.61333333333333329 | true |

## Distributional interpretation

The frozen sampled-grid mean-sign sequence remains:

`130M:+ -> 370M:+ -> 790M:+ -> 1.4B:- -> 2.8B:-`

The two negative-mean scales are not distributionally equivalent.

- Mamba-1.4B has negative mean, negative median, and a negative majority.
- Mamba-2.8B has negative mean but positive median and a positive majority.

At 2.8B, mean sign therefore disagrees with both median sign and majority sign. The negative mean is descriptively associated with magnitude/tail asymmetry rather than a majority-negative pair population.

Scales whose C0_SHAM and C2_NAME means have opposite signs:

`130M, 370M, 2.8B`

Cell antagonism is descriptive only. Selected/control plane identities differ across scales, so these values must not be interpreted as a common literal plane intervention across all five backbones.

## Scientific boundary

This synthesis does not establish:

- a continuous model-size threshold;
- a monotonic Delta-L scaling law;
- a homogeneous post-1B negative regime;
- a five-scale inferential p-value;
- causal equivalence of scale-specific planes;
- any result for D_CORE.

The 790M-to-1.4B interval remains the first adjacent sampled-scale mean-sign bracket only.

## Provenance

Five-scale mean-sign synthesis freeze:

`8c39408cee6a862c067c4da85ba7ec7aa34de1df`

`FIVE_SCALE_DELTA_L_DISTRIBUTION_SHAPE_SYNTHESIS = FROZEN_CANDIDATE`
