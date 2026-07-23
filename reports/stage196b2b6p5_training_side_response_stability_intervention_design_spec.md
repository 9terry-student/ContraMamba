# Stage196-B2-B6P5 Training-Side Response-Stability Intervention Design

## Scope

Stage196-B2-B6P5 is a design and static source-feasibility stage. It consumes
the frozen P4 response-topology artifacts, localizes the instability, audits
exact pairwise candidate gaps, and defines the smallest independently testable
training-side intervention families.

It does not implement an intervention, train, load a model or checkpoint,
change model behavior, or evaluate a safety target. It does not use gold
correctness, recovery, harm, `MUST_ALLOW`, `MUST_BLOCK`, or `OPTIONAL`.

## Command line

The analyzer requires:

```text
--repo-root
--stage196b2b6p4-analysis-json
--stage196b2b6p3-analysis-json
--stage196b2b6p2-analysis-json
--current-git-commit
--output-dir
```

Every path must be explicit and inside the repository. P4 companion artifacts
are resolved only from the supplied P4 analysis directory. The analyzer uses
no timestamp glob discovery.

## Frozen P4 authority

The supplied P4 directory must contain exactly the frozen eleven-file closure.
P5 loads the analysis, tail response rows, topology audit, relative-order
audit, endpoint audit, state dictionary, leakage boundary, and contract at
minimum. It requires:

```text
decision = STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE
recommended_next_stage =
STAGE196B2B6P5_TRAINING_SIDE_RESPONSE_STABILITY_INTERVENTION_DESIGN
blocking_reasons = []
failed contracts = 0
```

P5 validates all frozen counts, including 45,360 topology rows, 15,120
relative-order rows, 15,120 endpoint rows, 26,575 non-monotonic trajectories,
16,630 sign reversals, the within-tail and cross-seed disagreement counts, and
zero equal cross-seed endpoints. It also requires exact P2 epoch-20
reproduction on all nine disagreement categories.

The exact seeds are 183, 184, and 185. Candidate masks are:

```text
00100000000000
01000000000000
10000000000000
```

P4 must expose its exact seven-coordinate signed-response set. P5 intervention
design is restricted to:

```text
delta_support_minus_not_entitled
delta_support_minus_refute
delta_refute_minus_not_entitled
delta_top1_runner_up_margin
```

## Instability localization

The localization output partitions evidence by:

```text
seed
candidate mask
row-specific candidate action key
signed response coordinate
instability family
```

The instability-family value is an exact combination of within-tail topology,
candidate-relative order, and cross-seed topology flags. This creates a
partition rather than duplicating populations across several family rows.
Every row reports trajectory, non-monotonic, sign-reversal, zero-crossing,
winner-change, transition-change, within-tail order-disagreement, cross-seed
order-disagreement, cross-seed sign-sequence-disagreement, and cross-seed
monotonic-direction-disagreement counts.

No pooled-coordinate result is promoted as one mechanism.

## Exact pairwise-gap audit

For every seed, data row, tail epoch, primary coordinate, and unordered pair
among the three exact candidate masks, P5 reconstructs:

```text
gap(a,b) = response(a) - response(b)
```

The expected population is:

```text
3 seeds x 720 rows x 3 epochs x 4 coordinates x 3 candidate pairs
= 77,760 rows
```

Each output row includes the exact gap and absolute gap, exact-tie flag,
epoch-18/19/20 gaps, the three-epoch sign sequence, a strict pairwise-order
reversal flag, and cross-seed pairwise-order disagreement.

The analysis JSON reports exact distributions and linearly interpolated
quantiles at probabilities 0, .01, .05, .10, .25, .50, .75, .90, .95, .99,
and 1. It selects no magnitude threshold. It never labels a numerical gap
small or large. Mechanism language is limited to exact tie-contact reversal
counts versus strict positive-to-negative reversal counts.

## Static training-source feasibility

The exact training entry is:

```text
scripts/train_controlled_v6b_minimal.py
main::<locals>.run_training_v6b
```

The native final scores are differentiable tensors produced by:

```text
src/contramamba/modeling_v6b_minimal.py
ContraMambaV6BMinimal.forward
output["logits"]
```

Native signed class margins retain autograd connectivity. The top1-runner-up
coordinate is piecewise because its selected classes can change.

P4 counterfactuals are not a training graph. They are Python recompositions of
exported, detached diagnostic values from separately trained `joint` and
`frame_local_only` arms. The row-specific candidate key is a five-bit
primitive-substitution action selected through a fourteen-bit candidate
feature mask. No live training operator currently applies that exact action to
differentiable composer state.

Accordingly:

```text
native final margins:
TRAINING_GRADIENT_PATH_AVAILABLE

P4 recomposition:
DIAGNOSTIC_RECOMPOSITION_ONLY

exact live candidate geometry:
MINIMAL_GRADIENT_INSTRUMENTATION_REQUIRED

counterpart-arm state:
FULL_COUNTERFACTUAL_FORWARD_REQUIRED
```

Once live joint and counterpart primitive tensors exist, all three candidate
compositions can be vectorized in one training step. They do not inherently
require three backbone forwards. The minimum estimated active-step increment
is one counterpart-arm full forward plus negligible
`O(batch x 3 candidates x 4 coordinates)` composer arithmetic. A separate
teacher adds one no-gradient full forward and an EMA parameter update.

The intervention loss must enter `total_loss` before `loss_for_backward`.
`optimizer.step()` and the AMP `GradScaler.step()` occur after backward and
gradient clipping.

No EMA or frozen-anchor response teacher currently exists. Although EMA state
can be added independently, P5 does not authorize an arbitrary teacher while
the exact live candidate geometry and counterpart-arm provenance are absent.

## Family A: signed response-direction consistency

For row \(i\), candidate \(a\), and primary coordinate \(k\), let
\(r^S_{iak}\) be the student counterfactual-minus-native signed-margin response.
Let:

\[
t_{iak} = \operatorname{sgn}(r^T_{iak})
\]

where the teacher response is stop-gradient. Exact teacher ties
\(r^T_{iak}=0\) are excluded. The independently normalized loss is:

\[
L_A =
\operatorname{mean}_{(i,a,k):r^T_{iak}\ne0}
\operatorname{softplus}(-t_{iak}r^S_{iak})
\]

If no eligible target exists, the loss is exactly zero. The loss matches
topology, not raw scores or score magnitude. It uses no classifier labels,
safety targets, seed feature, row-identity feature, or global threshold.

The conceptual teacher preference is a stop-gradient EMA teacher because it
tracks the training trajectory without choosing a literal pre-tail checkpoint.
That preference is not implementation authorization: the teacher remains
`UNAVAILABLE_WITHOUT_ADDITIONAL_INSTRUMENTATION` at P5.

The initial coefficient is exactly `0.1`, with one flag:

```text
--use-response-direction-consistency
```

## Family B: candidate-relative order consistency

For the same row and coordinate, center the three candidate responses:

\[
c^S_{iak} =
r^S_{iak} - \frac{1}{3}\sum_b r^S_{ibk}
\]

For each unordered candidate pair \(a<b\), define the stop-gradient teacher
pair sign:

\[
q_{iabk} = \operatorname{sgn}(c^T_{iak}-c^T_{ibk})
\]

Exact teacher ties are excluded. The topology-only loss is:

\[
L_B =
\operatorname{mean}_{(i,k,a<b):q_{iabk}\ne0}
\operatorname{softplus}
\left[-q_{iabk}(c^S_{iak}-c^S_{ibk})\right]
\]

This loss imposes no fixed universal action ordering, performs no
candidate-mask lexical tie-break, matches no absolute score, and has no
seed-specific target.

The initial coefficient is exactly `0.1`, with one flag:

```text
--use-candidate-order-consistency
```

## Separation rule

Family A addresses the sign and monotonic topology of each individual action
response over training. Family B addresses within-row relative geometry among
the exact three actions. They are different mechanisms and must not be
accumulated opaquely.

The first causal experiment contains exactly:

```text
baseline
direction-consistency only
candidate-order-consistency only
```

A combined variant is permitted only after both independent interventions
have causal results.

## Activation scope

P4 evidence is tail-local, not whole-training evidence. The current training
loop exposes the total epoch count deterministically. Therefore the design
uses the semantic rule:

```text
last N epochs, N = 3
```

This is not a literal check for epochs 18, 19, and 20 and remains deterministic
for another configured training length. A trigger based on classification-loss
stabilization is not selected because the source contains no precommitted,
deterministic stabilization detector. Throughout-training activation would
extend beyond the observed mechanism without evidence.

## Data boundary

Main classification data remains exactly:

```text
data/controlled_v5_v3_without_time_swap.jsonl
```

`time_swap` cannot enter main classifier-label training. No external or OOD
data defines teacher signs, pair order, activation, coefficients, selection,
or any other stability target.

## Minimal precommitted trial

The three variants share seeds, data, optimizer, training length, and
evaluation. Each intervention has one hypothesis, exact inputs, one gradient
path, last-three-epochs activation, eligible-term mean normalization, one
initial coefficient, and one independent ablation flag. There is no
coefficient sweep.

Evaluation includes:

```text
clean-dev primary metrics
P2 exact composer reproduction
P4 topology metrics
support recall
false entitlement
false not-entitled
polarity errors
```

Failure is any precommitted clean-dev primary regression, adverse change in
the named error metrics, loss of P2 reproduction, or failure to improve the
family's targeted P4 metrics across the three fixed seeds. Nonfinite gradients,
failure to ignore exact teacher ties, or a changed data boundary also fail the
trial. Rollback disables the one family flag and restores the baseline path.

Topology stability alone does not guarantee classifier safety.

## Decision hierarchy

The analyzer evaluates all specified scientific decisions rather than
hardcoding the result. Under the statically observed source:

```text
decision =
STAGE196B2B6P5_GRADIENT_PATH_INSTRUMENTATION_REQUIRED

recommended_next_stage =
STAGE196B2B6P6_MINIMAL_GRADIENT_PATH_INSTRUMENTATION
```

This is a negative implementation-readiness conclusion, not a blocker. A
contract failure alone produces:

```text
decision =
STAGE196B2B6P5_BLOCKED_CONTRACT_FAILURE

recommended_next_stage =
STAGE196B2B6P5_REPAIR_CONTRACT
```

## Exact outputs and write behavior

The analyzer writes exactly nine files:

```text
stage196b2b6p5_analysis.json
stage196b2b6p5_report.md
stage196b2b6p5_source_feasibility_audit.csv
stage196b2b6p5_instability_localization.csv
stage196b2b6p5_pairwise_gap_audit.csv
stage196b2b6p5_intervention_designs.csv
stage196b2b6p5_trial_manifest.json
stage196b2b6p5_decision_gate.csv
stage196b2b6p5_contract.csv
```

Files are created in a staging directory, each staged file is atomically
renamed, and the complete staging directory is atomically renamed into place.
An existing output directory is never overwritten.

The analyzer returns zero exactly when `blocking_reasons == []`. It returns two
on a contract failure or when the requested output directory already exists.
A negative scientific design conclusion is not a blocker.

## Contract

The contract validates current commit identity; P4 decision, exact artifact,
and zero-failure closure; P2 reproduction; all three frozen populations; exact
seeds, candidate masks, and coordinates; localization closure; 77,760-row
pairwise population closure; source-gradient feasibility; intervention
independence; absence of a combined first-stage variant; prohibited-target
nondependency; `time_swap` exclusion; decision reachability; and the exact
nine-file declaration.
