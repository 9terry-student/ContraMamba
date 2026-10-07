# Gen5 M8a Latent Alignment / Task-Relevant Quotient Validated Evidence Report Candidate

## Status

`VALIDATED_M8A_CASE_C_EVIDENCE_CANDIDATE`

Bounded scientific classification:

`NONSEPARABLE_PATH_DEPENDENT_REPRESENTATIVES_WITH_FUNCTIONAL_EQUIVALENCE`

Gen5 A-init mechanism-line status:

`CLOSE_AINIT_AS_PRIMARY_EXPLANATORY_AXIS_AFTER_THIS_EVIDENCE_FREEZE`

This report does **not** claim that the downstream functional-reconvergence
mechanism has been fully explained.

The unresolved mechanism is explicitly retained as:

`UNRESOLVED_NATIVE_DOWNSTREAM_FUNCTIONAL_RECONVERGENCE_MECHANISM`

## Evidence identity

M8a implementation freeze / static execution HEAD:

`b0853fd5fef64dafffc55ad3e7b6a5067bf54b4d`

M8a authority commit:

`e1a091e313a605868dd790968d2fa80ca0bec038`

Source M7b evidence head:

`286435b586688889f513ea77190ed65a0ba3a2b1`

Run name:

`gen5-m8a-latent-alignment-b0853fd-r1`

Artifact root:

`reports/reason_router_gen5_m8a_latent_alignment_quotient_runs/gen5-m8a-latent-alignment-b0853fd-r1`

Frozen source identities:

- Phase-A trajectory SHA256:
  `0f7cd4248faa92223597e0816597b59e426f08829dadd366f9603f56a8de809e`
- M7 factor-swap logits SHA256:
  `d5e53f6841a968d9f397294680ef7938e1821858d42e920f3a95d199b1e58262`
- M7 factor-swap summary SHA256:
  `421af32a25b2d3bcf75a208cddb090010ae50fb5c0ff46804e67d677056a9cf8`
- task-state Gram SHA256:
  `649cd0208e40a1e0c60d3b4955eaf74131752a9ddf405da502c008fd673edaac`

Generated artifact identities:

- `m8a_latent_alignment_quotient_summary.json`
  - bytes: `2738`
  - SHA256: `55d458539edd24376e13ba3c8cbe0ab255d0cfd225533a61f35cd1d0ce3d10f1`
- `m8a_ordered_pair_metrics.jsonl`
  - bytes: `16389`
  - SHA256: `6cd82d165af857304535d21a85bca520ffa40f08c1e6b8e5cd3d617a961e385f`
- `run_provenance.json`
  - bytes: `1022`
  - SHA256: `e6f78fa4e2a56c2a64e3e81a105872ce8c8a9cff14575eb53ed06c7c826795df`
- `artifact_manifest.json`
  - bytes: `448`
  - SHA256: `9d8975e61b2a0b6c3174b8cd368467dfb0494873bc683977ed07544df11fcf9a`

## Execution and provenance validity

Pre-execution static verification passed against the exact frozen repository
state.

Authenticated frozen inputs:

- trajectory schema:
  `GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_TRAJECTORY_V2`
- M7 schema:
  `GEN5_M7_FACTOR_SWAP_SUMMARY_V1`
- task-state Gram schema:
  `GEN5_TASK_REACHABLE_LAYER22_STATE_GRAM_V1`
- task-state Gram tensor path:
  `gram`
- valid-token count:
  `60094`
- unique A0 representatives:
  `3`
- B1 cells:
  `9`
- ordered off-diagonal A-init pairs:
  `6`

The static scientific audit completed successfully.

Execution boundary:

- model forward count: `0`
- CUDA executed: `false`
- autograd executed: `false`
- backward executed: `false`
- optimizer constructed: `false`
- training executed: `false`
- checkpoint mutation: `false`
- confirmatory 9601-9900 population loaded: `false`

The local artifact validation then authenticated:

1. exact four-file output tree;
2. manifest file sizes and SHA256 identities;
3. provenance execution HEAD;
4. implementation-freeze identity;
5. authority commit;
6. frozen source SHA256 chain;
7. summary and pair-metrics SHA256 links;
8. anti-circularity flag.

Result:

`M8A_LOCAL_ARTIFACT_VALIDATION_PASS`

## Anti-circularity validity

The coordinate transport was fit using only:

`A_rec`
`A_don`

M7 outputs were not used to fit the transport:

`m7_outputs_used_for_transport_fitting = false`

M7 factor-swap outputs and M7b interaction strengths were used only as frozen
descriptive validation targets.

Therefore failure or success of the transport is not an output-fitted
explanation of the M7 interaction.

## Scientific question

M7 and M7b established that the t1 finite intervention is not additive.

The dominant structure was:

`recipient A0 + donor A/B1-history + recipient-A × donor-A/B1-history interaction`

M8a asked whether this interaction can be reduced to a deterministic latent
coordinate mismatch.

For each ordered pair `(recipient r, donor d)`, it fit:

`A_r ~= T_(d->r) A_d`

using a source-only 2x2 GL(2) transport and then represented donor B1 history in
recipient coordinates using:

`B_d^(r) = B_d T_(d->r)^+`

The test then compared raw and aligned A/operator discrepancies in both ambient
and frozen task-reachable metrics.

## Result 1 — the A0 row spaces are almost orthogonal across A-init seeds

Across the six ordered off-diagonal A-init pairs, the two-dimensional A0 row
spaces are extremely weakly aligned.

Mean-squared row-space affinity range:

`0.000699951753` to `0.001125461960`

Principal-angle range:

`87.396852°` to `89.985776°`

Thus the learned A0 representatives do not differ primarily by a 2x2 basis
change inside one common two-dimensional ambient row subspace.

This point is structurally important: left multiplication by an invertible
2x2 matrix can change coordinates within `row(A_d)` but cannot rotate that row
space into a different ambient two-dimensional subspace.

Therefore a large row-space mismatch is intrinsically outside the explanatory
capacity of a pure 2x2 coordinate reparameterization.

## Result 2 — GL(2) alignment fails in ambient A geometry

Across the six ordered pairs:

Mean raw ambient A residual:

`1.404047422076`

Mean aligned ambient A residual:

`1.412941181808`

Mean aligned/raw suppression ratio:

`1.006337308776`

Range of aligned/raw ambient A ratios:

`1.004573080274` to `1.008791993927`

Every ordered pair became slightly worse after the best source-only GL(2)
transport under this normalized residual.

Therefore:

`A_ambient_reduced_all_ordered_pairs = false`

This rejects the primary requirement for
`APPROXIMATE_COORDINATE_COMPATIBILITY`.

## Result 3 — donor-B1 transport also fails to recover matched ambient operators

Mean raw ambient operator residual:

`1.405249036307`

Mean aligned ambient operator residual:

`1.413941504884`

Mean aligned/raw ambient operator ratio:

`1.006194006920`

Range:

`1.001994697023` to `1.010882704054`

Again, every ordered pair failed to improve.

Therefore:

`operator_ambient_reduced_all_ordered_pairs = false`

The recipient-A × donor-history interaction is not explained by simply
expressing donor B1 in a recipient coordinate basis.

## Result 4 — task-state weighting does not reveal a consistent hidden coordinate law

For A alignment:

Mean raw task-weighted residual:

`1.425723865802`

Mean aligned task-weighted residual:

`1.412808952209`

Mean aligned/raw task ratio:

`0.996422250199`

But the ordered-pair ratio range was broad:

`0.939819365418` to `1.104342400728`

Some directions improved modestly while others worsened.

For operator alignment:

Mean raw task-restricted operator residual:

`1.425082146228`

Mean aligned task-restricted operator residual:

`1.413024078564`

Mean aligned/raw task ratio:

`0.999061515753`

Ordered-pair ratio range:

`0.926131607939` to `1.162168066648`

Thus task weighting does not produce a stable ordered-pair transport law.

In particular:

`A_task_reduced_all_ordered_pairs = false`

`operator_task_reduced_all_ordered_pairs = false`

`A_aligned_task_below_aligned_ambient_all_ordered_pairs = false`

`operator_aligned_task_below_aligned_ambient_all_ordered_pairs = false`

Therefore the preregistered support condition for
`TASK_RESTRICTED_QUOTIENT_EQUIVALENCE` is not met.

## Result 5 — some fitted transports are numerically fragile rather than explanatory

All six fitted 2x2 transports were technically full-rank.

Condition-number range:

`3.270132739` to `154.519493391`

Two ordered directions had condition numbers approximately:

`147.12`
and
`154.52`

Full rank therefore does not imply that the fitted transport provides a useful
or stable coordinate-equivalence law.

The important result is not merely that the matrices are invertible, but that
their use does not reduce the relevant A/operator mismatches.

## Result 6 — the frozen task-state distribution remains strongly anisotropic

The authenticated layer-22 task-state Gram has participation-ratio effective
dimension:

`3.585322797137`

Cumulative Gram energy:

- top 1: `0.514807209755`
- top 2: `0.616320258854`
- top 4: `0.672563012120`
- top 8: `0.736281281097`
- top 16: `0.801582397795`
- top 32: `0.862357183017`
- top 64: `0.923497942254`
- top 128: `0.970657505734`

Therefore failure of task-weighted alignment did not occur because the metric
was an approximately isotropic full-dimensional identity weighting.

It occurred despite a highly anisotropic frozen task-state distribution.

## Relation to M7b interaction

For the six ordered pairs, frozen M7b recipient×donor interaction strengths
range from:

`0.002797168855`
to
`0.017994820761`

The descriptive Spearman coefficients were:

- interaction vs aligned ambient operator residual:
  `-0.142857142857`
- interaction vs aligned task operator residual:
  `-0.657142857143`
- M7 ordered affinity vs aligned task operator residual:
  `0.2`

There are only six ordered pairs.

These values are descriptive only and support no population-level inference.

Most importantly, they do not rescue a coordinate-law explanation after the
primary ambient and task-weighted transport criteria fail.

## Primary scientific decision

The preregistered structural predicates were:

`case_A_structural_support = false`

`case_B_structural_support = false`

Therefore M8a rejects both:

`APPROXIMATE_COORDINATE_COMPATIBILITY`

and

`TASK_RESTRICTED_QUOTIENT_EQUIVALENCE`

as explanations of the M7b A-recipient × A-donor/B1-history compatibility
structure.

Given the already-frozen evidence that:

1. A-init causally dominates trained endpoint geometry;
2. A-init-dependent geometry appears from the earliest optimization dynamics;
3. B1 history is learned conditionally on the A0-selected coordinate system;
4. finite A0/B1 factor swaps produce a real nonadditive compatibility term;
5. different-A endpoint residuals are not explained by generic flat
   downstream-null variation;
6. final task behavior nevertheless reconverges strongly;

the bounded M8 classification is:

`NONSEPARABLE_PATH_DEPENDENT_REPRESENTATIVES_WITH_FUNCTIONAL_EQUIVALENCE`

## What this Case C classification means

The result supports:

`A0_SELECTION`
`-> A0_CONDITIONED_EARLY_HISTORY_FORMATION`
`-> NONSEPARABLE_A0_B1_COMPATIBILITY`
`-> PERSISTENT_INTERNAL_REPRESENTATIVE_DIFFERENCE`
`-> DOWNSTREAM_FUNCTIONAL_RECONVERGENCE`

The internal solutions reached from different A-init conditions are not
adequately described as the same latent state written in different GL(2)
coordinates.

The result therefore rejects a simple gauge/basis interpretation of the M7b
compatibility term.

## What this Case C classification does NOT mean

Case C does **not** establish:

- a complete dynamical explanation of why the downstream network reconverges;
- the unique nonlinear mechanism that suppresses decision-visible differences;
- a global equivalence manifold;
- formal gauge symmetry;
- universal Mamba representational non-identifiability;
- semantic ownership of native states;
- the earliest native precursor of correct versus incorrect decisions;
- behavior outside the frozen contract.

In particular, the unresolved question is:

> What native downstream dynamical mechanism maps strongly different,
> path-dependent internal representatives to nearly the same task decision?

That question is not answered by further A-init coordinate fitting.

## Why M8b is not warranted

M8b was reserved for the case where M8a discovered a viable coordinate
transport law whose finite aligned swap required causal confirmation.

M8a instead found:

- no ambient A rescue;
- no ambient operator rescue;
- no consistent task-weighted rescue;
- near-orthogonal A row spaces;
- no stable ordered transport explanation.

Therefore an aligned finite swap would test a mechanism that M8a did not
support.

`M8B_NOT_AUTHORIZED_AND_NOT_SCIENTIFICALLY_WARRANTED`

## Gen5 A-init mechanism-line closure

The A-init line has now answered its principal causal questions:

1. A-init is a dominant causal selector of internal endpoint geometry.
2. The effect is born early rather than appearing only at convergence.
3. Early learned B/history becomes A0-conditioned.
4. A0 and B1/history are not independently interchangeable.
5. Their nonadditivity is not reducible to a simple 2x2 latent coordinate
   mismatch.
6. The different internal paths can nevertheless converge to nearly equivalent
   task behavior.

The remaining central problem is no longer principally:

`WHAT_DOES_A_INIT_DO?`

It is:

`HOW_DO_NATIVE_MAMBA_DYNAMICS_COMPRESS_OR_RECONVERGE_PATH_DEPENDENT_INTERNAL_STATES_INTO_TASK_DECISIONS?`

Therefore further A-init-specific experiments are not the default next action.

A-init may remain useful later as an intervention/control family if a native
reconvergence mechanism is discovered.

## Next scientific stage

Return to the native-state program.

Primary next question:

`NATIVE_Q1_PRECURSOR_COHORT`

Compare prefix-only native state kinematics between:

- confident-correct decisions;
- confident-wrong decisions.

The first goal is not architecture modification.

The first goal is to identify whether a native temporal/kinematic precursor
separates these cohorts before the final decision.

If a robust precursor exists, the next step should be a causal evidence-order
perturbation targeting that precursor.

If the precursor is weak or semantically entangled, only then should the
longer-term semantic state-ownership M0-M4 architecture family become the next
candidate program.

## Final bounded conclusion

The Gen5 A-init mechanism line is closed at:

`NONSEPARABLE_PATH_DEPENDENT_REPRESENTATIVES_WITH_FUNCTIONAL_EQUIVALENCE`

with the explicit unresolved boundary:

`UNRESOLVED_NATIVE_DOWNSTREAM_FUNCTIONAL_RECONVERGENCE_MECHANISM`

This is a closure of A-init as the primary explanatory axis.

It is not a claim that the full native Mamba decision mechanism has been
solved.
