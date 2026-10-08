# Native Recurrent Constructive-Interference Serialization-Region Pair Localization — Scientific Design Candidate

Status: `SCIENTIFIC_DESIGN_READY_FOR_FREEZE`

Base validated evidence commit:

`9f60dacc98240eb9a087974e0996e5763f4d2b9a`

Source validated run:

`gen5-recurrent-pair-gap-localization-57ea0b5-r2`

Source validated artifact root:

`reports/native_recurrent_constructive_interference_pair_gap_localization_runs/gen5-recurrent-pair-gap-localization-57ea0b5-r2`

## 1. Scientific target

The validated pair-gap analysis established that the PRIMARY_A-specific loss of
complement-side constructive interference is not localized to one narrow token
gap.

Instead, the deficit:

- is strongest at the shortest separations;
- is distributed across short-to-mid gaps;
- accumulates approximately half of its signed total by `d <= 8`;
- accumulates approximately three quarters by `d <= 16`;
- accumulates over 93% by `d <= 32`;
- is essentially saturated by `d <= 64`;
- has negligible aggregate mass beyond 64 tokens.

The next question is therefore not:

> Which single gap causes the deficit?

The next bounded question is:

> Within the already-validated short-to-mid temporal regime, is the
> complement-side constructive-coherence deficit primarily carried by
> interactions within the claim, across the claim/evidence boundary, or within
> the evidence?

This stage deliberately stops at frozen serialization regions. It does not
introduce lexical, syntactic, semantic, or manually annotated token classes.

## 2. Why serialization regions are the next admissible level

The frozen Phase-3A P0 dev population already has an authenticated active
serialization:

`claim[:63] + EOS(0) + evidence[:64]`

with:

- 840 dev rows;
- 60,094 valid tokens;
- max sequence length 128;
- one explicit EOS boundary between claim and evidence;
- frozen dev row order;
- frozen active encoding.

These regions existed before the pair-gap result and are therefore not selected
post hoc from the observed interference profile.

The current stage MUST NOT inspect token strings and then invent categories from
the scientific output.

In particular, this design does not yet split evidence into TITLE, NAME, ROLE,
PREDICATE, OBJECT, or any other finer semantic field. Such a refinement would
require a later, separately frozen design after the coarse serialization-region
result is known.

## 3. Frozen population and recurrence boundary

Reuse exactly the validated population and computational boundary:

- population: `FROZEN_PHASE3A_P0_DEV`;
- dev rows: 840;
- valid tokens: 60,094;
- same 36 ordered factor orientations;
- factor seeds: 6201, 6202, 6203;
- same source-matched PRIMARY_A and CONTROL_R comparison;
- same raw-write visible/complement projector;
- same `raw_write -> recurrent_state` recurrence only;
- no downstream readout;
- no gate transport;
- no out-projection transport;
- no final head transport;
- no projector refit;
- no checkpoint mutation;
- no training;
- no confirmatory 9601/9900 population;
- no VitaminC population.

The validated pair-gap artifact and its identities are immutable references, not
inputs that may be silently recomputed with changed semantics.

## 4. Exact serialization-region token map

Every valid serialized token receives exactly one frozen coarse role.

Let token index `i` be in the active serialized sequence.

Define:

- `CLAIM`:
  every retained claim token;
- `EOS`:
  the single inserted EOS token between retained claim and retained evidence;
- `EVIDENCE`:
  every retained evidence token;
- `PAD`:
  padding only, never scientifically active and never included in pair sums.

The role assignment MUST be derived from the exact active encoding metadata used
by the frozen Phase-3A dev runtime.

No re-tokenization with a different tokenizer, tokenizer version, truncation
policy, or special-token policy is allowed.

The exact active serialization contract is:

- claim budget: 63;
- evidence budget: 64;
- EOS token id: 0;
- effective pad token id: 0;
- add special tokens: false;
- serialized length <= 128.

Although EOS and PAD share token id 0, they MUST be distinguished by serialized
position and attention mask. Token id alone is not a valid role classifier.

## 5. Exact pair contribution

For component `c` in `{VISIBLE, COMPLEMENT}`, retain the validated recurrence
definition.

For source-token pair `tau < sigma`, define its exact signed recurrent
cross-token contribution:

`I_c(tau, sigma) =
  2 * sum_(t >= sigma)
      <z_(t,tau)^c, z_(t,sigma)^c>`.

Then:

`C_c = sum_(tau < sigma) I_c(tau, sigma)`.

Normalize by the same propagated self-energy used in the validated pair-gap
audit:

`h_c(tau, sigma) = I_c(tau, sigma) / S_c`.

Therefore:

`Q_c = 1 + sum_(tau < sigma) h_c(tau, sigma)`.

The implementation may use any algebraically equivalent stable GPU reduction,
but the mathematical quantity is fixed by this identity.

## 6. Exact region-pair decomposition

Let `R(i)` be the frozen serialization role of token `i`.

For ordered region pair `(r,s)` and gap set `B`, define:

`H_c(r -> s, B) =
  sum_(tau < sigma,
       R(tau)=r,
       R(sigma)=s,
       sigma-tau in B)
       h_c(tau, sigma)`.

Because serialization order is fixed, the scientifically possible non-padding
region-pair classes are:

1. `CLAIM -> CLAIM`
2. `CLAIM -> EOS`
3. `CLAIM -> EVIDENCE`
4. `EOS -> EVIDENCE`
5. `EVIDENCE -> EVIDENCE`

There is no valid `EVIDENCE -> CLAIM`, `EVIDENCE -> EOS`, or `EOS -> CLAIM`
class under the frozen serialization order.

Every valid strict token pair must belong to exactly one of the five classes.

Therefore, for every component and gap set:

`sum_(region pairs) H_c(region pair, B) = sum_(d in B) H_c(d)`.

This exact reconstruction is mandatory.

## 7. Frozen temporal windows

The previous validated evidence fixes the temporal windows for this stage.

Primary short-to-mid regime:

- `W1 = gaps 1..8`
- `W2 = gaps 9..16`
- `W3 = gaps 17..32`

Tail controls:

- `W4 = gaps 33..64`
- `W5 = gaps 65..127`

The primary scientific interpretation is restricted to `d <= 32`, i.e.
`W1 + W2 + W3`.

`W4` and `W5` are retained as controls so the new role decomposition can replay
the validated distance-decay result.

These windows must not be altered after inspecting region-pair output.

## 8. Primary scientific quantities

For each of the 36 ordered orientations and both components, report:

- `H_c(region_pair, W1)`;
- `H_c(region_pair, W2)`;
- `H_c(region_pair, W3)`;
- `H_c(region_pair, W4)`;
- `H_c(region_pair, W5)`;
- `H_c(region_pair, d<=32)`;
- full-range `H_c(region_pair, all gaps)`;
- strict-pair opportunity count for every region pair/window;
- exact reconstruction residual against the validated `H_c(d)` profile.

The signed additive `H` quantities are primary.

For descriptive density control only, also report:

`mean_h_c(region_pair, B) =
 H_c(region_pair, B) / N_pairs(region_pair, B)`

when the pair count is nonzero.

`mean_h` is not a replacement for the additive total. It exists only to expose
whether a large class total is primarily explained by having more admissible
token pairs.

Do not take absolute values before aggregation.

## 9. Source-matched factor comparison

Retain the exact validated source matching.

For each of the nine source cells:

1. mean its two PRIMARY_A orientations;
2. mean its two matched CONTROL_R orientations;
3. compute PRIMARY_A minus CONTROL_R.

For every region pair and temporal window report:

`Delta H_visible(region_pair, B)`

and:

`Delta H_complement(region_pair, B)`.

Also replay:

- `Delta log Q_visible`;
- `Delta log Q_complement`;
- `Delta L_interference`;
- the validated full `Delta H_visible(d)` profile;
- the validated full `Delta H_complement(d)` profile.

No new source matching or population filtering is permitted.

## 10. Primary interpretation question

The primary question is:

> Which frozen serialization-region pair carries the validated complement-side
> deficit inside `d <= 32`?

The intended distinctions are structural:

- `CLAIM -> CLAIM` dominant:
  the deficit is primarily internal to claim-side recurrence;
- `CLAIM -> EVIDENCE` dominant:
  the deficit is primarily associated with cross-boundary recurrent coherence;
- `EVIDENCE -> EVIDENCE` dominant:
  the deficit is primarily internal to evidence-side recurrent coherence;
- substantial EOS-involving mass:
  boundary-token behavior requires explicit caution before any semantic
  interpretation;
- no clear region separation:
  serialization-region localization is not supported, and the deficit remains
  primarily distance-structured.

No numeric dominance threshold is introduced.

Report the continuous values and source-matched sign patterns.

## 11. Mandatory visible-side control

The validated aggregate result showed:

- median source-matched `Delta log Q_visible` near zero;
- mixed source-matched sign;
- extensive mixed-sign full-gap structure.

Therefore the visible component is a mandatory negative/control comparison.

A proposed region-specific complement interpretation is weakened if the same
region-pair pattern of comparable scale appears on the visible side.

Do not report complement-only region localization without the matched visible
table.

## 12. Pair-count confounding boundary

Region classes have unequal numbers of admissible token pairs.

Therefore:

- additive `Delta H` answers where the total validated deficit resides;
- pair counts describe class opportunity;
- `mean_h` describes signed average contribution per admissible pair.

A large additive region total MUST NOT be described as stronger per-pair
coherence loss unless its pair-normalized diagnostic supports that statement.

Conversely, pair-normalized values MUST NOT be used to replace the exact
additive reconstruction of the validated aggregate mechanism.

## 13. Non-additivity boundary

Region-pair terms are additive for:

`C_c / S_c`.

They are not additive components of:

`L_interference =
 log(Q_complement / Q_visible)`.

Therefore this stage forbids:

- percentage attribution of `L_interference` to serialization regions;
- Shapley attribution;
- arbitrary region ordering;
- log-ratio decomposition by naive subtraction;
- post-hoc rescaling to make region totals sum to a log statistic.

`L_interference` remains an authenticated aggregate endpoint only.

## 14. Computational requirement

The implementation must extend the validated pair-gap recurrence arithmetic,
not introduce a second scientific mechanism.

Required:

- reuse the same raw-write source and target tensors;
- reuse the same detached two-margin projector;
- reuse the same native recurrence coefficients;
- reuse the same backward survival factor where algebraically valid;
- assign frozen serialization-role masks before pair reduction;
- reduce exact signed strict-pair terms by region pair and gap;
- no additional model forward solely for role accounting.

A naive Python or CPU loop over all token pairs is not an admissible scientific
execution path.

GPU vectorization/FFT/dyadic reduction is allowed if it reproduces exact
small-tensor brute force within the already-frozen numerical tolerances.

The implementation should preserve the existing two independent single-GPU
worker topology and disjoint example shards unless a static resource check
proves that a narrower equivalent topology is required.

## 15. Mandatory implementation validation

Before scientific execution, tests must establish:

1. exact CLAIM/EOS/EVIDENCE role assignment on synthetic serialized examples;
2. EOS distinguished from PAD despite shared token id 0;
3. truncated claim/evidence budgets map to the correct absolute indices;
4. padding contributes no scientific pair;
5. every strict valid token pair belongs to exactly one region-pair class;
6. no duplicate region-pair accounting;
7. region-pair sums reconstruct the exact gap-resolved `C(d)` and `H(d)`;
8. window sums reconstruct `d<=8`, `d<=16`, `d<=32`, `d<=64`, and full totals;
9. mixed positive/negative pair-interference synthetic case;
10. visible and complement paths use identical role masks;
11. source-matched orientation coverage remains exactly 36;
12. dev population remains exactly 840 rows / 60,094 valid tokens;
13. frozen dev order and active encoding hashes replay exactly;
14. no downstream stage or final-head transport;
15. no training, optimizer, backward, or checkpoint mutation;
16. no confirmatory 9601/9900 loading;
17. validated `57ea0b5` pair-gap aggregate metrics replay within the existing
    frozen tolerances;
18. tolerance widening after scientific output is forbidden.

## 16. Falsification and narrowing logic

The working hypothesis is:

> The short-to-mid-range complement coherence deficit has a coarse structural
> locus in the frozen claim/EOS/evidence serialization.

This hypothesis is not supported if:

- exact region-pair reconstruction fails;
- the validated pair-gap profile does not replay;
- complement source-matched region deltas do not show a coherent structural
  distinction;
- the apparent largest class is explained only by pair opportunity while
  pair-normalized behavior is otherwise undifferentiated;
- the visible side shows a comparable region-specific pattern;
- the deficit is spread across claim-claim, claim-evidence, and
  evidence-evidence without a stable source-matched distinction.

A negative result is scientifically valid and should terminate this branch
before finer semantic token classes are introduced.

## 17. Semantic-fishing guard

This design explicitly forbids inspecting token strings from high-magnitude
pairs and inventing explanatory labels afterward.

No TITLE/NAME/ROLE/PREDICATE/OBJECT decomposition is authorized here.

A finer field-level analysis may be considered only after this coarse
serialization-region analysis is validated and frozen, and only under a new
prospective design using generator-defined spans that existed before the new
field-level output is inspected.

## 18. Authorization boundary

This document authorizes only a bounded implementation of the exact
serialization-region pair decomposition after the design itself is frozen.

During design freeze:

- Training: `NO`
- Evaluation: `NO`
- Kaggle scientific execution: `NO`
- Confirmatory seeds 9601/9900: `NO`
- VitaminC: `NO`
- Downstream transport: `NO`

After a later implementation is locally validated and frozen, scientific
execution still requires the normal fresh execution gate at that exact
implementation commit.

No run is authorized by this design document alone.
