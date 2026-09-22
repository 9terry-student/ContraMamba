# ContraMamba Gen4 Mamba-1 Five-Scale Delta-L Signed-Mass Synthesis

## Status

`STATIC_FIVE_SCALE_SIGNED_MASS_SYNTHESIS`

This artifact performs no model execution, backward pass, training, new hypothesis test, new p-value, threshold estimation, rescue, re-selection, or row filtering.

For pair-level `Delta_L_owned = x`, define:

- `M_pos = sum(x for x > 0)`;
- `M_neg = sum(abs(x) for x < 0)`.

Then exactly:

`mean(x) = M_pos / N - M_neg / N`.

This decomposition introduces no arbitrary tail cutoff.

## Canonical frozen sources

- `reports/reason_router_gen4_mamba130m_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl`: `8012e07d2a53b90d9d271ca4e4aa93af26f78c2fdfeaf0d149e38ff602c232b4`
- `reports/reason_router_gen4_mamba370m14b_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl`: `f566f704b11d6b9574ac381f8584c44f3be191183ddb0e65eb3cb2e315f62779`
- `reports/reason_router_gen4_mamba790m_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl`: `8c9b2b5d2d523dafce0fed837a983f1cdf1f21194b2f0c0da8f99e3a0b31d99d`
- `reports/reason_router_gen4_mamba28b_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl`: `1aeb28e34ae0f9ee3816b4c85dd8dc07ee7924515451b8b7e77a802c118e00eb`

## Five-scale signed-mass decomposition

| Scale | N+ | N- | Positive mass | Negative abs mass | Positive contribution to mean | Negative contribution to mean | Mean positive pair | Mean abs negative pair | Neg/pos mass ratio | Conditional magnitude ratio | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 130M | 161 | 139 | 0.87078683222032172 | 0.53467702126068373 | 0.0029026227740677393 | -0.0017822567375356125 | 0.0054086138647224955 | 0.0038465972752567174 | 0.61401596978375372 | 0.71119835349053484 | 0.0011203660365321265 |
| 370M | 173 | 127 | 0.24252421229091312 | 0.08983730672528796 | 0.0008084140409697104 | -0.0002994576890842932 | 0.0014018740594850469 | 0.0007073803679156532 | 0.37042613550487957 | 0.50459623182948166 | 0.00050895635188541726 |
| 790M | 221 | 79 | 0.71641369143436395 | 0.033400256477415603 | 0.0023880456381145465 | -0.00011133418825805201 | 0.0032416909114677101 | 0.00042278805667614685 | 0.046621465888715012 | 0.13042207546083567 | 0.0022767114498564943 |
| 1.4B | 106 | 194 | 0.094245512321980515 | 0.45287976408784764 | 0.00031415170773993505 | -0.0015095992136261588 | 0.00088910860681113693 | 0.002334431773648699 | 4.805319138598648 | 2.6255867458322508 | -0.0011954475058862236 |
| 2.8B | 166 | 134 | 0.13919988113236156 | 0.16145236718306624 | 0.00046399960377453854 | -0.00053817455727688749 | 0.00083855350079735879 | 0.0012048684118139271 | 1.1598599500925246 | 1.4368414307116351 | -7.4174953502348948e-05 |

## Main distributional distinction

Mamba-1.4B is negative by both frequency and magnitude:

- 194 negative pairs versus 106 positive pairs;
- total negative absolute mass is approximately 4.8053 times positive mass;
- the average absolute negative pair is approximately 2.6256 times the average positive pair.

Mamba-2.8B is different:

- 166 positive pairs versus 134 negative pairs;
- nevertheless total negative absolute mass is approximately 1.1599 times positive mass;
- the average absolute negative pair is approximately 1.4368 times the average positive pair.

Therefore the 2.8B negative mean is a magnitude-dominance phenomenon despite a positive pair majority, whereas 1.4B combines negative frequency dominance with negative magnitude dominance.

These are distinct negative-mean morphologies.

## Scientific boundary

This synthesis is descriptive arithmetic only.

It does not establish:

- a new inferential result;
- a model-size threshold;
- a monotonic scaling law;
- a homogeneous post-1B regime;
- causal equivalence of the scale-specific selected/control planes;
- any D_CORE conclusion.

Previous distribution-shape synthesis freeze:

`3544544a05698e10a323246a6cecae9a997d0bc9`

`FIVE_SCALE_DELTA_L_SIGNED_MASS_SYNTHESIS = FROZEN_CANDIDATE`
