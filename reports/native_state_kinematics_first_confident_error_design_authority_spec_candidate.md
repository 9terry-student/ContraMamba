# Native State Kinematics First Confident-Error Experiment
# Scientific Design Authority Specification

STATUS = CANDIDATE

PHASE = NATIVE_STATE_KINEMATICS_FIRST_CONFIRMATORY_DESIGN

PARENT_HYPOTHESIS =
12c86088f68482870dd53cbcf6c363499b248f81

PARENT_HYPOTHESIS_FILE =
docs/CONTRAMAMBA_NATIVE_STATE_KINEMATICS_HYPOTHESIS.md

O0C_VALIDATED_NATIVE_STATE_RESULT =
ff2fb076f6e66a34a632515bb8502d8b1c90ad7f

NEW_SCIENTIFIC_EVIDENCE_CREATED_BY_THIS_SPEC = NO

IMPLEMENTATION_ALLOWED = NO

TRAINING_ALLOWED = NO

EVALUATION_EXECUTION_ALLOWED = NO

KAGGLE_ALLOWED = NO

CHECKPOINT_INFERENCE_ALLOWED = NO

PROMOTION_ALLOWED = NO


## 1. Purpose

This specification freezes the scientific design of the first proper
ContraMamba Native State Kinematics experiment.

It does not authorize implementation or execution.

The experiment is deliberately narrower than the full long-term hypothesis.

The first question is:

> Does Mamba's native recurrent-state trajectory contain a reproducible
> prefix-only kinematic difference between confidence-matched correct and
> wrong decisive factual commitments?

The target is:

CONFIDENT-CORRECT
vs
CONFIDENT-WRONG

not:

generic correct vs wrong

and not:

hallucination vs non-hallucination.


## 2. Relationship to prior O0c evidence

O0c established:

BROAD_NATIVE_PRECURSOR_NOT_SUPPORTED

and:

TERMINAL_LOCALIZED_RECURRENT_STATE_SEPARATION_OBSERVED

That result must not be rewritten.

The first Native State Kinematics experiment addresses a different object:

full local trajectory motion around a prespecified evidence event

rather than sparse anchor-state separation alone.

Existing O0c artifacts may be used only for methodological feasibility such
as:

- tensor-shape verification;
- state-index verification;
- velocity implementation checks;
- turning implementation checks;
- numerical-stability checks;
- serialization checks.

O0c may not supply the confirmatory confident-correct versus confident-wrong
effect estimate for this experiment.


## 3. Primary scientific hypothesis

Let the native recurrent state after token t at layer l be:

s_t^(l)

and define:

v_t^(l) = s_t^(l) - s_(t-1)^(l)

speed:

nu_t^(l) = ||v_t^(l)||_2

and directional turning:

kappa_t^(l) =
1 - cos(v_t^(l), v_(t-1)^(l)).

For an interval [a,b], define path efficiency:

E_[a:b]^(l) =
||s_b^(l) - s_a^(l)||_2
/
sum_{t=a+1}^b ||v_t^(l)||_2

when the denominator is nonzero.

The primary hypothesis is intentionally direction-agnostic:

At least one preregistered native-state kinematic endpoint differs between
confidence-matched correct and wrong decisive commitments before the final
decision.

The experiment does not assume in advance that wrong trajectories are:

- faster;
- slower;
- more unstable;
- more stable;
- more curved;
- more persistent.

All primary tests are two-sided.


## 4. Native-state geometry lock

The first experiment uses the simplest native geometry.

STATE_REPRESENTATION =
vectorized native selective-SSM recurrent state

DISTANCE_GEOMETRY =
ordinary Euclidean geometry within one fixed layer

CROSS_LAYER_RAW_NORM_COMPARISON =
PROHIBITED

WHITENING =
NO

MAHALANOBIS_GEOMETRY =
NO

LEARNED_TRAJECTORY_EMBEDDING =
NO

NONLINEAR_TRAJECTORY_DETECTOR =
NO

The first experiment must not be rescued by changing geometry after viewing
the confirmatory outcome.

Coordinate-robustness analyses may be authorized later only if the simple
geometry produces evidence worth testing for robustness.


## 5. Primary layer rule

Layer selection must be independent of the outcome.

For a model with L recurrent layers indexed:

0, ..., L-1

define the primary layer:

L_PRIMARY =
floor((L - 1) / 2)

That is the architecture midpoint.

The first scientific verdict is determined at L_PRIMARY only.

Two prespecified robustness layers may later be reported:

L_Q1 =
floor((L - 1) / 4)

L_Q3 =
floor(3 * (L - 1) / 4)

The Q1 and Q3 layers:

- are secondary;
- cannot rescue a negative primary-layer result;
- cannot replace the primary layer after outcome inspection.

No best-layer scan is permitted.


## 6. Primary population

The primary population contains examples whose final predicted class is a
decisive factual commitment:

SUPPORT
or
REFUTE

Primary NOT_ENTITLED predictions are excluded from the first confirmatory
population.

This restriction is methodological.

It does not assert that NOT_ENTITLED dynamics are scientifically unimportant.

For each example:

CORRECT =
final predicted label equals gold label

WRONG =
final predicted label differs from gold label

The native backbone, tokenizer, task head, and inference procedure must be
frozen before cohort construction.


## 7. Confidence definition

Final confidence must be defined using only the frozen final decision output.

No native-state trajectory quantity may participate in the definition of
confidence.

The exact confidence statistic and confident threshold must be frozen by a
later execution-preparation authority before native-state outcome analysis.

Permitted examples include:

- predicted-class probability;
- a fixed final logit margin.

The confident threshold may be selected using a separate calibration split.

It may not be selected by maximizing native-state separation.

CONFIDENCE_THRESHOLD_TUNING_ON_CONFIRMATORY_STATE_OUTCOMES =
PROHIBITED


## 8. Confidence and confound control

The primary comparison must control final confidence sufficiently that a
trajectory difference cannot be reduced to:

wrong cases were merely less confident.

Matching or deterministic balancing must be performed without looking at
native-state trajectory outcomes.

At minimum account for:

- final predicted class;
- final confidence;
- input token length;
- decisive-evidence position;
- evidence/template or intervention family where available.

Predicted SUPPORT and predicted REFUTE remain explicit strata.

Gold label is retained and reported, but exact matching simultaneously on both
gold and predicted label is not required when mathematically incompatible with
correct-versus-wrong status.

The exact matching algorithm must be frozen before scientific execution.

If adequate balance cannot be achieved, the confirmatory experiment must stop.

The matching rule must not be weakened after state outcomes are inspected.


## 9. Decisive evidence time

Each confirmatory example must have one prespecified decisive-evidence token
index:

tau_e

tau_e must come from:

- an existing validated annotation; or
- a deterministic annotation rule frozen before native-state analysis.

It must not be selected from the observed native-state trajectory.

Examples without a defensible tau_e are not eligible for the primary
confirmatory population.

The first experiment uses event-relative model time.


## 10. Prefix requirement

Let T denote the final input token whose resulting representation is available
to the final task decision.

Primary examples must satisfy:

tau_e + 4 <= T - 1

so that the primary kinematic window ends before the terminal decision state.

The primary event-relative interval is:

[tau_e, tau_e + 4]

The primary analysis therefore cannot use:

- terminal native state;
- terminal classifier confidence beyond cohort definition;
- future tokens after tau_e + 4;
- any post-decision information.

This is the formal prefix-only lock.


## 11. Primary observable family

The first experiment has exactly three primary kinematic endpoints.

No fourth primary endpoint may be added after outcome inspection.

### P1. Post-evidence mean speed

POST4_SPEED =
mean(
    nu_(tau_e+1),
    nu_(tau_e+2),
    nu_(tau_e+3),
    nu_(tau_e+4)
)

### P2. Post-evidence mean directional turning

POST4_TURNING =
mean(
    kappa_(tau_e+1),
    kappa_(tau_e+2),
    kappa_(tau_e+3),
    kappa_(tau_e+4)
)

### P3. Post-evidence path efficiency

POST4_PATH_EFFICIENCY =
||s_(tau_e+4) - s_(tau_e)||_2
/
sum_{t=tau_e+1}^{tau_e+4}
||v_t||_2

when the denominator is nonzero.

These three endpoints cover:

- motion magnitude;
- directional change/persistence;
- cumulative directedness versus circuitous motion.

They are computed at L_PRIMARY.


## 12. Secondary diagnostics

The following may be prespecified secondary diagnostics:

- mean acceleration magnitude over the same POST4 window;
- pre-evidence speed;
- pre-evidence turning;
- pre-evidence path efficiency;
- Q1 robustness-layer versions;
- Q3 robustness-layer versions.

Secondary diagnostics cannot convert a failed primary family into a supported
primary hypothesis.

The following remain outside the first experiment:

- jerk;
- spectral trajectory statistics;
- learned trajectory embeddings;
- nonlinear detectors;
- manifold learning;
- decision-space Jacobian projections;
- learned semantic-state probes;
- evidence-response latency thresholds;
- commitment-before-entitlement timing.

These require later authority.


## 13. Statistical family

The confirmatory statistical family contains exactly:

P1 POST4_SPEED
P2 POST4_TURNING
P3 POST4_PATH_EFFICIENCY

at L_PRIMARY.

Tests must be:

- two-sided;
- confidence-controlled;
- based on a cohort frozen before state-outcome inspection.

The planned default test is a matched-pair permutation test when deterministic
one-to-one matching is feasible.

If the final cohort design is not one-to-one matched, the execution authority
must freeze the alternative statistical model before native-state outcome
inspection.

Family-wise multiplicity across P1-P3 must be controlled.

Default correction:

HOLM

Default family-wise alpha:

0.05

No token-wise significance scan is part of the primary analysis.


## 14. Effect reporting

Statistical significance alone is insufficient.

For every primary endpoint report:

- group sample counts;
- matched-pair or stratum structure;
- center and dispersion by outcome class;
- effect direction;
- effect magnitude;
- uncertainty interval;
- adjusted p-value;
- confidence-balance diagnostics;
- length-balance diagnostics;
- evidence-position balance diagnostics.

The effect magnitude definition must be frozen with the final statistical
design.

No threshold for a scientifically meaningful effect may be chosen after
viewing confirmatory outcomes.


## 15. Reproducibility and seed semantics

A deterministic frozen native Mamba backbone produces the same native state
for the same exact input.

Therefore downstream training seeds must not be counted as independent
native-state replications when they do not alter:

- backbone parameters;
- tokenization;
- input;
- inference computation.

The experiment must explicitly distinguish:

MODEL-SEED REPLICATION

from:

DUPLICATED ANALYSIS OF IDENTICAL NATIVE STATES

If multiple genuinely distinct frozen Mamba model seeds/checkpoints are not
available, the experiment must not claim cross-model-seed replication.

Reproducibility may instead be established through:

- held-out examples;
- prespecified predicted-class strata;
- later independent model replication.

This limitation must be reported explicitly.


## 16. Primary support criterion

NATIVE_KINEMATICS_PRECURSOR = SUPPORTED

requires all of the following:

1. cohort/provenance admission passes;
2. confidence and major confound balance passes;
3. primary prefix window is valid for every admitted example;
4. at least one of P1-P3 survives the frozen family-wise correction;
5. the effect is not produced only by terminal-state information;
6. the effect is not the result of post-hoc layer or token selection;
7. the supported endpoint has a nontrivial prespecified effect magnitude;
8. the effect direction is not grossly contradictory across the prespecified
   SUPPORT and REFUTE predicted-class strata when both strata satisfy their
   frozen minimum sample requirement.

A positive result supports a bounded native-state kinematic precursor.

It does not establish a causal mechanism.


## 17. Reformulation criterion

NATIVE_KINEMATICS_PRECURSOR = REFORMULATE

is appropriate when, for example:

- signal is confined to one predicted-class stratum;
- confidence balance is inadequate;
- effect exists only in secondary layers;
- effect exists only at terminal state;
- simple Euclidean geometry is numerically unstable;
- one endpoint appears suggestive but misses the frozen confirmatory criterion;
- adequate confident-wrong sample size cannot be obtained;
- effect is dataset/template specific.

A reformulation must receive new authority.

It must not silently alter this design.


## 18. Not-supported criterion

NATIVE_KINEMATICS_PRECURSOR = NOT_SUPPORTED

is appropriate if an adequately powered, properly admitted, confidence-balanced
confirmatory experiment finds:

no P1-P3 endpoint with reproducible nontrivial early separation
after frozen multiplicity correction.

A negative result must not be rescued by:

- scanning additional layers;
- scanning arbitrary tokens;
- adding dozens of metrics;
- training a nonlinear detector;
- changing confidence threshold;
- changing the confident-error cohort;
- changing geometry post hoc;
- redefining terminal separation as an early precursor.


## 19. Power and sample-size gate

This design does not yet authorize execution because prospective sample-size
requirements are not frozen.

Before execution authority, a non-outcome-leaking cohort feasibility stage must
freeze:

- exact dataset identity;
- exact model/checkpoint identity;
- exact final-confidence statistic;
- confident threshold;
- exact tau_e annotation source/rule;
- correct/wrong counts by predicted class;
- achievable matching balance;
- minimum analyzable sample size;
- prospective power or smallest-effect-of-interest rule.

Native-state trajectory outcomes must not be used to tune these quantities.


## 20. Required provenance for eventual execution

Any later scientific execution must bind:

- full source commit SHA;
- exact model/checkpoint SHA256;
- tokenizer identity;
- dataset manifest SHA256;
- cohort manifest SHA256;
- tau_e annotation manifest SHA256;
- confidence-rule identity;
- matching-rule identity;
- exact layer indices derived from the frozen layer rule;
- exact metric implementation identity;
- exact statistical-analysis identity;
- output artifact SHA256.

Branch name alone is never sufficient provenance.


## 21. Immediate next stage after this design freezes

If this design is frozen, the next permissible research action is:

COHORT_AND_MEASUREMENT_FEASIBILITY_AUDIT

That audit must be read-only with respect to model training.

Its purpose is only to establish whether the required confident-error cohort,
evidence-time annotation, state tensor availability, and prospective sample
size exist.

It must not test the P1-P3 confident-error scientific outcomes.

In particular, the feasibility audit may not report:

correct-versus-wrong speed separation
correct-versus-wrong turning separation
correct-versus-wrong path-efficiency separation

because doing so would leak the confirmatory outcome into design preparation.


## 22. Explicit exclusions

This design does not establish or authorize:

- a hallucination detector;
- semantic recurrent states;
- architecture modification;
- gradient intervention;
- evidence-order perturbation;
- causal state intervention;
- decision-space factorization;
- Authorization x Signed Polarity as established ontology;
- optimal layer;
- optimal metric;
- optimal confidence threshold;
- parameter ownership;
- training;
- evaluation execution;
- Kaggle execution.

Evidence-order perturbation remains a potential later causal test.

It is not part of the first confirmatory experiment.


## 23. Stop conditions

Stop before implementation or execution if any of the following remains
unresolved:

- exact native-state tensor semantics;
- exact frozen model/checkpoint;
- exact dataset;
- confidence definition;
- tau_e annotation;
- cohort balance;
- prospective sample size/power;
- provenance binding;
- implementation authority;
- execution authority.

No blocker may be bypassed by weakening the scientific question.


## 24. Current verdict

DESIGN_VERDICT =
READY_FOR_AUTHORITY_REVIEW

FIRST_TARGET =
CONFIDENT_CORRECT_VS_CONFIDENT_WRONG_DECISIVE_COMMITMENT

PRIMARY_LAYER =
ARCHITECTURE_MIDPOINT

PRIMARY_TIME_WINDOW =
TAU_E_THROUGH_TAU_E_PLUS_4_PREFIX_ONLY

PRIMARY_ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

PRIMARY_ENDPOINT_COUNT =
3

O0C_ROLE =
METHODOLOGICAL_FEASIBILITY_ONLY

NEW_EXECUTION_AUTHORITY =
NO

NEXT_STAGE_IF_FROZEN =
COHORT_AND_MEASUREMENT_FEASIBILITY_AUDIT

END_OF_NATIVE_STATE_KINEMATICS_FIRST_CONFIDENT_ERROR_DESIGN_AUTHORITY_SPEC
