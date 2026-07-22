# Stage196-B2-B6P0 Selector Safety-State Observability Specification

## Scope

Stage196-B2-B6P0 is an artifact-only analysis of whether exact, categorical,
pre-intervention recipient state can distinguish rows on which one of the three
frozen Stage196-B2-B6 selectors must be allowed, must be blocked, or may safely
abstain. It performs no training, classifier fitting, threshold learning,
aggregate-score optimization, model loading, checkpoint loading, promotion, or
architecture integration.

The three selector candidates are fixed independently:

```text
00100000000000
01000000000000
10000000000000
```

Their masks, semantic member names, and assigned primitive action sets are
policy context. They are never members of the safety-feature subset lattice.

## Required invocation

The analyzer requires exactly these explicit arguments:

```text
--repo-root
--stage196b2b6-analysis-json
--stage196b2b5-analysis-json
--stage196b2b4-analysis-json
--stage196b2b3p0-run-root
--stage196b2b3p0-runtime-git-commit
--current-git-commit
--output-dir
```

All paths must be absolute and under the explicit repository root. Source
selection never uses timestamp discovery, modification time, recursive
guessing, sibling fallback, or partial filename matching. Companion artifacts
are admitted only by their exact expected names in the supplied analysis JSON's
directory.

## Scientific authority and source closure

The analyzer fails closed unless Stage196-B2-B6 has the decision
`STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE`, recommends
`STAGE196B2B6P0_SELECTOR_SAFETY_STATE_OBSERVABILITY`, has no blocking reasons,
has its exact nine-file closure, has every contract gate pass, was produced at
analyzer commit `d21457a3b514a304994c799357c725df3edbcc18`, retains exactly the
three candidates above, passes the exact primary contract for all three, and
fails nondiscovery safety on both seed184 and seed185 for all three.

Stage196-B2-B5 must retain decision
`STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED`, its prescribed next
stage, empty blocking reasons, exact nine-file closure, and a fully passing
contract. Its feature dictionary is the semantic authority for existing
recipient-local and paired-delta features.

Stage196-B2-B4 must retain decision
`STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION`, its prescribed next stage,
empty blocking reasons, exact nine-file closure, and a fully passing contract.
The primitive order is frozen as `FRAME`, `PREDICATE`, `SUFFICIENCY`,
`POSITIVE_ENERGY`, `NEGATIVE_ENERGY`; all 32 action masks and all exact row
action sets are independently closed against the B2-B5 authority.

The P0 source must contain exactly the six named run directories. Each composer
namespace contains its exact manifest and twenty exact composer sidecars. The
mixed-purpose trajectory directory is filtered only by the exact
`stage196b2p0_epoch_channels_NNN.jsonl` namespace; unrelated files are ignored,
while malformed namespace-like files fail closed. Runtime commit
`fa16787efa84bb15d832b6d9382fafd77016c4e2`, 120 composer sidecars, 120
trajectory sidecars, 86,400 composer rows, and 86,400 trajectory rows are
required.

## Population semantics

Seed-conditioned primary treatment cases use `(seed, stable_row_id)` and total
16: seed184 has five recovery and six harm cases; seed185 has two recovery and
three harm cases.

Discovery data identity uses `(id, source_row_id, dev_position)`. Seed184 has 11
identities, seed185 has five, their intersection is three, and their union is
13. Each seed's clean-dev partition is 13 discovery-union identities, 707
nondiscovery identities, and all 720 identities. Seed183 uses the same 13/707/720
partition only as contrast and never participates in gate discovery, subset
selection, cross-seed authorization, or decision-rule promotion.

## Primitive reconstruction

For each candidate and clean-dev row, the analyzer independently rebuilds the
B2-B5 selector signature, obtains the frozen B2-B6 action set, installs only the
selected primitive fields from the exact paired frame-local-only composer state,
and recomputes entitlement, decision-head logits, residual branch magnitude,
final logits, and prediction. It never copies entitlement probability,
decision-head logits, residual deltas, final logits, prediction, or final margin.

Native composer quantities and B2-B4 tail counterfactual predictions must agree
within `1e-6`, with exact prediction equality. B2-B6 clean-dev row counts,
coverage, prediction changes, correct-to-incorrect, incorrect-to-correct,
stable-correct preservation, and both seed safety results are independently
reproduced.

## Safety targets

One target record is built per candidate, seed, and clean-dev identity.

- `MUST_ALLOW` requires a seed-conditioned primary row for which `00000` fails
  the exact B2-B4 row objective and every assigned selector action passes it.
- `MUST_BLOCK` requires a nondiscovery row whose joint prediction is correct and
  for which any allowed action makes the prediction incorrect.
- `OPTIONAL` covers every other row, including already-satisfied primary
  objectives, stable rows, beneficial transitions, and unseen-signature
  abstentions.

Target and outcome metadata are used only for exact feasibility evaluation.
They never enter a safety signature.

## Observability families

The single-state family uses only epoch 20 joint recipient state. Its categorical
features are recipient prediction; final and head margin signs; head/final sign
conflict; frame, predicate, sufficiency, and entitlement halfspaces;
entitlement bottleneck with sorted-set ties; polarity energy order; predicate
and temporal mismatch; and temporal adapter and channel activity. Margin and
delta zero use tolerance `1e-12`; halfspace equality uses `0.5` with tolerance
`1e-12`. No raw continuous value is a signature feature. This is the only
family marked potentially integration-authorized.

The tail-trajectory family uses the exact available Stage196-B2-B5 recipient
feature semantics over epochs 18, 19, and 20. It is diagnostic-only and never
described as inference-time deployable.

The paired-delta family uses epoch-20 frame-local-only donor minus joint
recipient signs for frame, predicate, sufficiency, positive energy, negative
energy, entitlement, head margin, and final margin, plus exact predicate and
temporal mismatch change categories. It is diagnostic-only and never supports
an integration-authorized claim.

Gold label, correctness, transition outcomes, target, transition role, path
class, subtype, seed, row/data identity, coalition labels, counterfactual or
donor outcomes, candidate mask, assigned action mask, and action cardinality are
prohibited safety features.

## Exact subset feasibility and transfer

For every candidate and family, all `2^N - 1` nonempty subsets of available
semantic features are enumerated. Constrained rows are exactly `MUST_ALLOW` and
`MUST_BLOCK`. A subset is feasible only if no exact signature contains both
labels. Allow-only signatures map to `ALLOW`; block-only signatures map to
`BLOCK`; optional-only and unseen signatures default to `BLOCK`. There is no
majority vote, accuracy/rate threshold, fitted threshold, learned binning, or
score optimization. Every inclusion-minimal feasible subset is retained.

Every feasible subset is transferred seed184-to-seed185 and
seed185-to-seed184 using only constrained source-seed rows. An unseen target
`MUST_ALLOW` fails; an unseen target `MUST_BLOCK` passes because unseen defaults
to block. Full transfer requires zero unseen or incorrectly blocked
`MUST_ALLOW` rows and zero incorrectly allowed `MUST_BLOCK` rows in both
directions.

## Conservative policy and seed183 contrast

For every feasible gate, `ALLOW` applies the fixed candidate action set and all
other states use `00000`. Set-valued actions are evaluated conservatively: any
harmful allowed action counts as harm. Audits cover all 16 primary cases,
seed184 nondiscovery 707, seed185 nondiscovery 707, and seed183 contrast all 720.
Primary objective failures and prediction-change, correct-to-incorrect,
incorrect-to-correct, and stable-correct-preservation metrics are reported.

Seed183 reports seen/unseen signatures, allowed/blocked rows, transition counts,
and stable-correct preservation. Its outcomes never alter a mapping or subset.
Audit completion is a contract property distinct from gate success.

## Ordered decisions

The analyzer applies the specified order: contract failure; cross-seed
single-state safety gate localized; pooled single-state gate in-sample only;
seed-specific single-state gates; tail-trajectory gate only; paired-delta gate
only; current safety observability insufficient. Only the first scientific
success rule may recommend the safety-gated counterfactual audit, and even that
does not authorize integration, training, or promotion.

## Outputs and atomicity

Exactly nine files are staged in a new temporary sibling directory, flushed and
fsynced, checked for exact closure, and atomically renamed to the requested
output directory. An existing output directory is never overwritten:

```text
stage196b2b6p0_analysis.json
stage196b2b6p0_report.md
stage196b2b6p0_safety_feature_dictionary.csv
stage196b2b6p0_row_safety_targets.csv
stage196b2b6p0_single_state_signature_rows.csv
stage196b2b6p0_single_state_gate_summary.csv
stage196b2b6p0_diagnostic_gate_summary.csv
stage196b2b6p0_gated_policy_audit.csv
stage196b2b6p0_contract.csv
```

Contract observations are structured JSON; booleans remain booleans in the
analysis and are rendered as lowercase JSON booleans in CSV cells. Contract
failure still emits the exact nine-file blocked artifact with decision
`STAGE196B2B6P0_BLOCKED_CONTRACT_FAILURE` and next stage
`STAGE196B2B6P0_REPAIR`.

## Scientific limits

The stage cannot establish formal causal mediation, external/OOD or
unfrozen-Mamba validity, training improvement, promotion, tail-trajectory or
paired-delta deployability, safety from pooled separation or partial transfer,
or safety evidence from optional activation or unseen blocking. Gold, seed, and
identity are never inference-time features, and no arbitrary threshold is
presented as a mechanistic gate.
