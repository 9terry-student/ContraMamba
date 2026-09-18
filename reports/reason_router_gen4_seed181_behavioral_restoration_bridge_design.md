# Gen4 seed181 prospective behavioral restoration bridge

## Status

Prospective confirmatory design freeze.

No behavioral model output has been inspected under this design.

This document authorizes no scientific execution by itself.

## Scientific question

Does the response-blind P3 homolog that independently replicated the internal
seed181 restoration effect also improve the downstream task decision relative
to the matched P5 geometric control?

## Checkpoint

Exactly one checkpoint is primary:

- seed: `181`
- arm: `G3-GROUP-D-HALF`
- selected epoch: `19`
- SHA256:
  `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f`

Seed180 is not part of this behavioral confirmatory experiment.

Failure on seed181 must not trigger a seed180 rescue experiment.

## Geometry

Geometry is inherited unchanged from the completed seed181 replication.

- homolog plane: `P3`
- matched control plane: `P5`

No response-based plane reselection is permitted.

No new geometry discovery is permitted.

## Fresh population

Use exactly:

`xg1_fact_2701..xg1_fact_3000`

with:

`N = 300`

This population must be deterministically generated under the existing XG1
independent structured-record semantics and verified disjoint from all prior
XG1 populations through `xg1_fact_001..2700`.

No behavioral response may be inspected during materialization or structural
validation.

## Rows

Exactly two rows per source pair are behavioral targets:

- `C0_SHAM`
- `C2_NAME`

No other six-cell row participates in the confirmatory endpoint.

### Frozen labels

For this prospective XG1 population:

- `C0_SHAM -> SUPPORT -> class id 2`
- `C2_NAME -> NOT_ENTITLED -> class id 1`

Rationale:

`C0_SHAM` renders the same positive structured fact as claim and evidence.

`C2_NAME` preserves the claim but substitutes only the evidence-side name with
the pre-generated alternate name, matching the established entity/frame
mismatch semantics whose final class is `NOT_ENTITLED`.

These labels are fixed before any behavioral inference.

## Intervention site

Reuse the frozen Gen4-K intervention site and token anchoring.

- intervention layer: `17`
- semantic target: existing `A_IDENTITY` / target-coordinate construction
- geometry: frozen seed181 P3/P5 vectors

No layer search, token search, or intervention-site sweep is permitted.

## Conditions

Exactly four conditions are evaluated for each behavioral row.

### C0 — native

No geometric correction.

### CN — P3 neutralized

Remove the native P3 component.

### CR — exact P3 restoration

Use the frozen exact-native P3 restoration construction.

### CC — matched P5 replacement

Replace the removed P3 component with the matched P5 control using the same
coefficient magnitude construction inherited from the restoration experiment.

No additional intervention condition is primary or confirmatory.

## Downstream behavioral endpoint

The intervention must occur during a full model forward.

The changed Mamba hidden computation must continue through the existing
ContraMamba downstream heads to the frozen final three-way logits.

Class order:

`REFUTE, NOT_ENTITLED, SUPPORT`

For row `r` with frozen correct class `y`, define correct-class logit margin:

`m_r = z_y - max_{c != y} z_c`

For source pair `i`, average the two prespecified behavioral rows:

`M_i = (m_i,C0_SHAM + m_i,C2_NAME) / 2`

This pair average is the unit used by the primary test.

## Primary contrast

For each source pair:

`D_BEH,i = M_R,i - M_C,i`

where:

- `M_R` is the exact P3-restoration pair margin;
- `M_C` is the matched P5-control pair margin.

Primary hypothesis:

`H0: E[D_BEH] <= 0`

versus:

`H1: E[D_BEH] > 0`

## Confirmatory test

Exactly one confirmatory inferential test is permitted.

- test: one-sided one-sample Student t-test
- sample size: `N = 300`
- degrees of freedom: `299`
- alpha: `0.05`
- direction: positive

Behavioral bridge support requires both:

1. `mean(D_BEH) > 0`
2. one-sided `p < 0.05`

No multiplicity correction is needed because there is exactly one primary
inferential endpoint.

No endpoint switching is permitted after inspection.

## Secondary / descriptive quantities

The following are secondary or descriptive only:

- behavioral necessity:
  `M_native - M_neutralized`
- per-condition accuracy
- prediction-flip rates
- per-row margins
- C0_SHAM-only effects
- C2_NAME-only effects
- relationship between internal-state effect magnitude and behavioral margin
  effect

These quantities do not promote or rescue the primary result.

No secondary p-value may replace the primary decision rule.

## Two-GPU execution plan

Scientific semantics are independent of GPU assignment.

The population is deterministically partitioned into two disjoint contiguous
shards:

### shard 0

- physical device: `cuda:0`
- pairs: `xg1_fact_2701..2850`
- pairs: `150`
- forwards per pair: `8`
- expected scientific full-model forwards: `1200`

### shard 1

- physical device: `cuda:1`
- pairs: `xg1_fact_2851..3000`
- pairs: `150`
- forwards per pair: `8`
- expected scientific full-model forwards: `1200`

Total:

`2400` scientific full-model forwards.

Execution must fail closed unless at least two CUDA devices are visible.

Each shard must independently record:

- execution commit;
- checkpoint SHA256;
- model/tokenizer identity;
- device identity;
- exact pair range;
- exact forward count;
- geometry identity;
- condition identity;
- output checksums.

## Merge requirements

Primary analysis may begin only after both shards pass.

The merge must verify:

- exactly two shards;
- same execution commit;
- same checkpoint SHA;
- same model/tokenizer identity;
- same frozen design identity;
- same P3/P5 geometry identity;
- shard 0 exact range `2701..2850`;
- shard 1 exact range `2851..3000`;
- zero pair overlap;
- exact union `2701..3000`;
- 150 pairs per shard;
- 1200 forwards per shard;
- 2400 total forwards;
- exactly four conditions per row;
- exactly two rows per pair;
- no missing or duplicate pair/row/condition tuple.

GPU/shard is an execution partition only.

The statistical sample remains the merged 300 source pairs.

## Interpretation boundary

If the primary test passes, the supported claim is limited to:

The independently replicated seed181 P3 homolog has a downstream behavioral
effect under the prespecified fresh-holdout matched-restoration test, with
exact P3 restoration producing greater correct-class decision margin than the
matched P5 replacement.

If the primary test does not pass, the behavioral bridge is not established.

No seed180 rescue, alternative plane, alternative layer, alternative token,
alternative margin, alternative row subset, or alternative significance test
may be introduced in response to failure.

## Frozen execution scale

`300 pairs x 2 rows x 4 conditions = 2400 full-model forwards`

distributed as:

`1200 on cuda:0 + 1200 on cuda:1`
