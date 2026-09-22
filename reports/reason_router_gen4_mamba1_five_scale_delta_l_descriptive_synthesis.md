# ContraMamba Gen4 Mamba-1 Five-Scale Delta-L Descriptive Synthesis

## Status

`STATIC_FIVE_SCALE_DESCRIPTIVE_SYNTHESIS`

No model execution, training, backward pass, new p-value, threshold estimation,
monotonicity test, rescue, re-selection, or response-guided filtering is
performed by this synthesis.

## Endpoint semantics

All five entries are frozen scale-specific selected-minus-control local
directional readouts under `G3-GROUP-D-HALF`.

For this ownership graph:

`Delta_L_forward_equivalent = 2 * Delta_L_owned`.

The x2 conversion is deterministic and preserves sign, rank, positive fraction,
and historical t statistics/p-values.

Selected/control planes differ by scale and must not be interpreted as one
literal common plane across all backbones.

## Frozen five-scale map

| Scale | Selected / control | Mean Delta_L_owned | Mean forward-equivalent | Positive fraction | Evidence status |
|---|---|---:|---:|---:|---|
| Mamba-130M | P3 / P5 | +0.001120366036532127 | +0.002240732073064253 | 0.5366666666666666 | historical positive supported |
| Mamba-370M | P3 / P5 | +0.0005089563518854174 | +0.001017912703770835 | 0.5766666666666667 | historical positive sign gate |
| Mamba-790M | P2 / P5 | +0.0022767114498564943 | +0.004553422899712989 | 0.7366666666666667 | descriptive only |
| Mamba-1.4B | P5 / P4 | -0.001195447505886224 | -0.002390895011772448 | 0.35333333333333333 | historical negative sign gate |
| Mamba-2.8B | P3 / P5 | -0.00007417495350234895 | -0.0001483499070046979 | 0.5533333333333333 | descriptive only |

Observed sampled-grid mean-sign sequence:

`130M:+ -> 370M:+ -> 790M:+ -> 1.4B:- -> 2.8B:-`

The first adjacent sampled-scale sign change is therefore between 790M and
1.4B. This is a sampled-grid bracket only. No continuous parameter-count
threshold is estimated.

## Non-monotonicity

The frozen mean magnitudes are not monotonic with parameter count.

In particular:

- 370M has a smaller positive mean than 130M;
- 790M rises to the largest positive mean in this five-scale set;
- 1.4B changes sign;
- 2.8B remains negative in mean but rebounds strongly toward zero.

Therefore this evidence does not justify a simple monotonic scaling-law account
of Delta-L magnitude.

No formal monotonicity test is added post hoc.

## 2.8B distributional qualification

Mamba-2.8B has:

- mean Delta_L_owned = -0.00007417495350234895;
- median Delta_L_owned = +0.000054122475344360534;
- positive fraction = 0.5533333333333333;
- negative fraction = 0.44666666666666666.

Thus the negative mean is not a majority-negative phenomenon.

By cell:

- C0_SHAM mean Delta_L_owned = -0.0005878319015971666;
- C0_SHAM positive fraction = 0.43;
- C2_NAME mean Delta_L_owned = +0.00043948199459246863;
- C2_NAME positive fraction = 0.6133333333333333.

The frozen descriptive evidence therefore indicates substantial cell
antagonism and magnitude/tail asymmetry at 2.8B.

This differs from Mamba-1.4B, where the overall positive fraction is only
0.35333333333333333.

## Interpretation boundary

The five-scale evidence supports the descriptive statement that the frozen
scale-specific Delta-L mean is positive at 130M, 370M, and 790M and negative at
1.4B and 2.8B.

It does not establish:

- an exact scaling threshold;
- a monotonic Delta-L scaling law;
- a single homogeneous negative regime above 1B;
- a new five-scale inferential p-value;
- causal equivalence between the different selected/control planes;
- any conclusion based on D_CORE, which is a distinct endpoint.

## Provenance

Historical three-scale ownership correction:

`reports/reason_router_gen4_three_scale_readout_owned_forward_correction.md`

Mamba-790M descriptive analysis freeze:

`ff1e6c0530fa737ec7322320299ce67bcc87034f`

Mamba-2.8B descriptive analysis freeze:

`40e1e1b151d9700f6f7b24d5a982e99e3385f66c`

`FIVE_SCALE_DELTA_L_DESCRIPTIVE_SYNTHESIS = FROZEN_CANDIDATE`
