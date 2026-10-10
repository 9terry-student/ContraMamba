# Native Recurrent Constructive-Interference Serialization-Region Pair Localization — Validated Evidence Analysis Report Candidate

Status: `VALIDATED_EVIDENCE_ANALYSIS_READY_FOR_FREEZE`

Execution commit:

`032e68ef85cf18917589d9d5efbc52857688f062`

Validated run:

`gen5-recurrent-serialization-region-localization-032e68e-r1`

Artifact root:

`reports/native_recurrent_constructive_interference_serialization_region_pair_localization_runs/gen5-recurrent-serialization-region-localization-032e68e-r1`

## 1. Scope

This report interprets the validated recurrence-only serialization-region pair localization audit.

It does not introduce training, checkpoint fitting, projector refitting, downstream transport, confirmatory-seed loading, VitaminC execution, token-string inspection, new semantic labels, threshold tuning, or a new causal intervention.

The scientific question is:

> Within the already validated short-to-mid recurrent pair-gap regime, which frozen serialization-region pair classes carry the PRIMARY_A-specific loss of complement-side constructive coherence relative to matched CONTROL_R?

The frozen serialization is:

`claim[:63] + EOS(0) + evidence[:64]`

with PAD also using token id 0 but distinguished from EOS by frozen attention/region masks.

The only authorized ordered region-pair classes are:

1. `CLAIM->CLAIM`
2. `CLAIM->EOS`
3. `CLAIM->EVIDENCE`
4. `EOS->EVIDENCE`
5. `EVIDENCE->EVIDENCE`

For component `c`, the additive quantity remains:

`H_c(r->s, B) = sum h_c(tau, sigma)`

over strict ordered source-token pairs in region class `r->s` and gap window `B`.

These `H` terms are additive diagnostics. They are not additive attributions of `L_interference`.

## 2. Provenance and execution validity

The imported run passed the complete frozen validation chain.

Execution/runtime:

- execution HEAD: `032e68ef85cf18917589d9d5efbc52857688f062`;
- Python: `3.12.13`;
- NumPy: `2.0.2`;
- PyTorch: `2.10.0+cu128`;
- Transformers: `5.0.0`;
- tokenizers: `0.22.2`;
- kernels: `0.10.2`;
- exact CUDA runtime: `12.8`;
- two independent Tesla T4 workers, no DDP;
- `BATCH_ROWS=32`;
- worker 0 rows: `0:416`;
- worker 1 rows: `416:840`;
- 840 dev rows;
- 60,094 valid tokens;
- 36 ordered orientations;
- 243 source-gradient forwards;
- 486 base pair-gap group decompositions;
- 486 serialization-region group decompositions;
- recurrence-only;
- downstream stage transport calls: 0;
- final head transport calls: 0;
- training executed: false;
- optimizer constructed: false;
- backward method called: false;
- parameter gradients accumulated: false;
- checkpoint mutation: false;
- confirmatory seeds 9601/9900 loaded: false;
- VitaminC loaded: false.

CUDA batch-32 execution gate:

- both T4 devices passed fast-path numerical brute-force checks;
- both T4 devices passed large-span dyadic fallback checks;
- both T4 devices passed max-batch region stress at state width 24,576;
- peak allocated memory: approximately `7.879 GiB`;
- peak reserved memory: approximately `8.170 GiB`;
- free memory after stress: approximately `6.264 GiB`;
- scientific model forward count during the gate: 0.

Final scientific execution:

- worker 0 result SHA256:
  `12269683210b9c985af4cd35d7bf9f3e83ac5667a4fb4a432e1dd4e564337aa9`;
- worker 1 result SHA256:
  `afee2d250dd082d1eb07e34d5e6a5696f01932e33bbcc0c254d41fd33b8067a4`;
- final artifact validation: PASS;
- validated pair-gap replay: PASS;
- serialization-region reconstruction: PASS.

Final imported artifact SHA256 identities:

- `serialization_region_pair_summary.json`:
  `24006b8c650582348bb6b9754b4568f30c5bf079329eb1a02f7e0f1b536d78fa`;
- `orientation_serialization_region_pair_metrics.jsonl`:
  `a2eae4c48bbbcb3d677f44fc8189f344201d3e7b3991a4509906959d13d4eb13`;
- `shard_manifest.json`:
  `d0781d2c20d654e570b656d5b32feb5494078a01465aecbe6f5bdfb954c673da`;
- `run_provenance.json`:
  `fdaa9016fd003a16cc66ce0df85736add658a95d6ee8c6f04d0fc0d592c903a8`.

Run wrapper identities:

- command SHA256:
  `dc0623b33a3a86cd63e8f99602a6dc60ccb3afc4e00c5d4952b5d5dd032264ea`;
- run log SHA256:
  `90d6c499de1be35d984d38387b81363224eab5d4a85e682817ffd8848dbf4f8d`;
- run meta SHA256:
  `421d17183f727abcad56309900e7481a1380c5205663eaa7a1f34a8ca8dce919`;
- imported ZIP SHA256:
  `2a3047257237766a668908fb0bf3cd867aa03762f9d472877324edf40515d92f`.

## 3. Aggregate endpoint replay

Source-matched PRIMARY_A minus CONTROL_R across the nine frozen source cells:

- `Δ log Q_visible`:
  median `-0.005391115513626588`,
  negative `6/9`,
  positive `3/9`;
- `Δ log Q_complement`:
  median `-1.0435337630442745`,
  negative `9/9`,
  positive `0/9`;
- `Δ L_interference`:
  median `-1.037444014279346`,
  negative `9/9`,
  positive `0/9`.

Therefore the previously validated aggregate result is preserved: the factor-specific interference shift is overwhelmingly complement-side, while the aggregate visible-side shift remains near zero and mixed in sign.

## 4. Primary-window serialization localization

For the primary `gap1_32` window, source-matched complement median `ΔH` is:

| Region pair | Median ΔH | Negative source cells | Pair opportunities | Median Δmean_h |
|---|---:|---:|---:|---:|
| CLAIM->CLAIM | -2.694797538624105 | 9/9 | 493,920 | -5.455939299125578e-06 |
| EVIDENCE->EVIDENCE | -2.259347735545224 | 9/9 | 515,168 | -4.385652322242887e-06 |
| CLAIM->EVIDENCE | -0.9143700607629331 | 9/9 | 416,640 | -2.1946286020615716e-06 |
| EOS->EVIDENCE | -0.061346027495691305 | 9/9 | 26,880 | -2.2822182848099445e-06 |
| CLAIM->EOS | -0.04231726473837188 | 9/9 | 26,880 | -1.574303003659668e-06 |

All five complement region classes are negative in all `9/9` matched source cells.

The dominant additive deficits are therefore `CLAIM->CLAIM` and `EVIDENCE->EVIDENCE`, with `CLAIM->EVIDENCE` materially smaller but still substantial.

The pair-opportunity-normalized descriptive density control preserves the main ordering:

1. `CLAIM->CLAIM`
2. `EVIDENCE->EVIDENCE`
3. `EOS->EVIDENCE` / `CLAIM->EVIDENCE`
4. `CLAIM->EOS`

Thus the large within-CLAIM and within-EVIDENCE additive masses are not explained solely by those classes having more available token pairs.

The density normalization is descriptive only. It does not define a new causal estimand or replace additive `H`.

## 5. Distance-dependent role transition

The complement source-matched median `ΔH` by fixed window is:

| Region pair | gap1-8 | gap9-16 | gap17-32 | gap33-64 | gap65-127 |
|---|---:|---:|---:|---:|---:|
| CLAIM->CLAIM | -1.5673129741284637 | -0.7293546026445727 | -0.39812996185106914 | -0.004180120425924403 | 0 |
| CLAIM->EOS | -0.018098162931406467 | -0.011836707080046721 | -0.013808469524892687 | -0.0016130664455086263 | 0 |
| CLAIM->EVIDENCE | -0.11929705013035115 | -0.24474960963590894 | -0.5503234009966731 | -0.4331998418695331 | -0.003744008000905463 |
| EOS->EVIDENCE | -0.028309538184925813 | -0.016173993914393184 | -0.016862495396372315 | -0.0009378482188063669 | approximately `-1.22e-10` |
| EVIDENCE->EVIDENCE | -1.3979689381131446 | -0.5872674890536741 | -0.27817852862281367 | -0.0034104279924029486 | approximately `-5.24e-09` |

This reveals a distance-dependent structural transition.

### gap1-8

The deficit is overwhelmingly within-segment:

- `CLAIM->CLAIM`: `-1.5673`;
- `EVIDENCE->EVIDENCE`: `-1.3980`;
- `CLAIM->EVIDENCE`: only `-0.1193`.

### gap9-16

Within-segment terms remain dominant:

- `CLAIM->CLAIM`: `-0.7294`;
- `EVIDENCE->EVIDENCE`: `-0.5873`;
- `CLAIM->EVIDENCE`: `-0.2447`.

### gap17-32

The ordering changes:

- `CLAIM->EVIDENCE`: `-0.5503`;
- `CLAIM->CLAIM`: `-0.3981`;
- `EVIDENCE->EVIDENCE`: `-0.2782`.

Thus cross-region claim-to-evidence coherence becomes the largest individual region deficit in the upper half of the primary window.

### gap33-64

The previously observed small long-range tail is almost entirely localized to `CLAIM->EVIDENCE`:

- `CLAIM->EVIDENCE`: `-0.4332`;
- `CLAIM->CLAIM`: `-0.00418`;
- `EVIDENCE->EVIDENCE`: `-0.00341`;
- EOS-boundary terms are smaller still.

### gap65-127

The aggregate effect is negligible, consistent with the prior pair-gap result.

The only non-negligible value at this scale is `CLAIM->EVIDENCE = -0.003744`, which is itself tiny relative to the primary-window deficits.

## 6. EOS-boundary hypothesis

The result does not support an EOS-boundary-primary explanation.

Within `gap1_32`:

- `CLAIM->EOS`: median `ΔH = -0.0423`;
- `EOS->EVIDENCE`: median `ΔH = -0.0613`.

Both are sign-consistently negative in `9/9` cells, but their additive magnitude is small compared with:

- `CLAIM->CLAIM = -2.6948`;
- `EVIDENCE->EVIDENCE = -2.2593`;
- `CLAIM->EVIDENCE = -0.9144`.

Therefore EOS participates in the broad deficit but is not the principal localization locus.

## 7. Visible-side control structure

Visible-side `gap1_32` source-matched medians are:

- `CLAIM->CLAIM`:
  `+1.854903540402871`, positive `9/9`;
- `CLAIM->EOS`:
  `+0.007411033461686395`, negative `3/9`, positive `6/9`;
- `CLAIM->EVIDENCE`:
  `+0.04123763883050158`, negative `3/9`, positive `6/9`;
- `EOS->EVIDENCE`:
  `-0.2260137606454944`, negative `9/9`;
- `EVIDENCE->EVIDENCE`:
  `-1.9405986483265778`, negative `9/9`.

The visible control is therefore not region-wise null.

Instead, its near-zero aggregate `Δ log Q_visible` reflects substantial cancellation between:

- a strong positive `CLAIM->CLAIM` shift;
- a strong negative `EVIDENCE->EVIDENCE` shift;
- a smaller negative `EOS->EVIDENCE` shift;
- near-zero or mixed cross-boundary terms.

This is qualitatively distinct from the complement component, where all five region classes are negative in all `9/9` source-matched cells.

The complement result should therefore be described as a broad region-wise contraction of constructive coherence, not as a generic recurrence-wide loss that appears identically in the visible component.

## 8. Scientific conclusion

The coarse serialization-region analysis supports the following result:

> The PRIMARY_A-specific complement-side constructive-coherence deficit is not localized to the EOS boundary and is not carried by a single serialization region. At short separations, the deficit is dominated jointly by within-CLAIM and within-EVIDENCE recurrence. As separation increases through the short-to-mid regime, CLAIM->EVIDENCE coherence becomes increasingly important and becomes the largest individual region contribution over gaps 17-32. The residual 33-64 tail is localized almost entirely to CLAIM->EVIDENCE, while gaps 65+ remain negligible.

A compact description is:

**short-range within-segment contraction with a mid-range claim-to-evidence cross-region tail.**

This refines the prior pair-gap conclusion without contradicting it.

The prior result established a broad, distance-decaying complement-side coherence deficit concentrated in gaps 1-32 with a small 33-64 tail.

The present result shows that:

- the shortest part of that deficit is principally within serialized claim and evidence regions;
- the upper short-to-mid range increasingly reflects claim-to-evidence interaction;
- the remaining 33-64 tail is specifically cross-region rather than within-region;
- EOS is not the main boundary carrying the effect.

## 9. Falsification outcome

The analysis falsifies or narrows several stronger localization hypotheses.

Not supported:

- a primarily EOS-mediated deficit;
- a single serialization-region explanation;
- an explanation based only on pair-opportunity count;
- a claim that the entire short-to-mid deficit is cross-region;
- a claim that the entire deficit is within-region.

Supported:

- a distributed complement-side deficit across all authorized serialization pair classes;
- dominant within-CLAIM and within-EVIDENCE loss at the shortest distances;
- a distance-dependent transition toward CLAIM->EVIDENCE loss;
- a cross-region CLAIM->EVIDENCE explanation for the small 33-64 tail.

## 10. Claim boundary

This evidence supports statements about frozen serialization-region roles within the seed180 frozen Phase3A dev/source-cell population.

It does not establish:

- semantic identity of individual claim or evidence tokens;
- TITLE, NAME, ROLE, PREDICATE, OBJECT, or other newly invented semantic labels;
- lexical, syntactic, or entity-level causality;
- a unique responsible token pair;
- a unique native state coordinate;
- universality beyond the frozen population;
- causal mediation by EOS;
- percentage attribution of nonlinear `L_interference` to region pairs;
- a new training rule or architecture.

The region-pair `H` quantities are additive diagnostics for recurrence interference. They must not be reinterpreted as Shapley values or additive decompositions of `L_interference`.

No scientific p-values or post-hoc mechanism thresholds are introduced.

## 11. Recommended next scientific question

The coarse serialization-role question is resolved sufficiently to freeze this evidence before any finer analysis.

If a finer follow-up is pursued, it should not target EOS.

The next scientifically motivated question is:

> Which pre-existing frozen generator spans account for the short-range within-CLAIM / within-EVIDENCE contraction and the mid-range CLAIM->EVIDENCE cross-region deficit?

Any such follow-up must be separately designed and frozen before execution.

It should preserve:

- the existing recurrence-only quantity;
- the validated `gap1-8`, `gap9-16`, `gap17-32`, and `gap33-64` windows;
- visible-side controls;
- pair-opportunity descriptive normalization;
- the prohibition on token-string inspection and post-hoc semantic fishing;
- the prohibition on new semantic labels unless they correspond to already-existing frozen generator spans.

No new training, confirmatory-seed execution, or downstream intervention is authorized by this report.
