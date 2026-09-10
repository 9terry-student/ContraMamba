# Seed8192 A0/A1/A2/A3 Factorial Scientific Interpretation Candidate

## 1. Status and authority

```text
INTERPRETATION_AUTHORITY_COMMIT = d644523b11a9431f841ea1216004c5e1527a740a
INTERPRETATION_AUTHORITY_BLOB = 6463237fd7a1358496d25ca62d1458295a7ee50e
VALIDATED_EVIDENCE_COMMIT = 6dcef9520af2cb88691628a77b72f3fdd7042cd8
VALIDATED_EVIDENCE_BLOB = 17be9bead782f129de925148580b034d981079cd
UPSTREAM_EXECUTION_AUTHORITY_COMMIT = 3a76c6cd3f6bd8b011317f37938677822ce9191d
TRAINING_EVALUATION_AUTHORIZED = NO
IMPLEMENTATION_AUTHORIZED = NO
NEW_EXECUTION_AUTHORIZED = NO
```

Status: candidate report in the static factorial scientific-interpretation
phase.  The frozen validated-evidence report is the sole numerical authority;
this report makes the bounded N=3 descriptive conclusion permitted by the
frozen interpretation authority.  It performs no training, evaluation,
checkpoint loading, inference, Kaggle action, implementation, staging, commit,
or push.

## 2. Scientific question and completed design

The completed matched 2x2 factorial used seeds 180/181/182 and `split_seed`
8192:

| Cell | Router | Gradient ownership | Reason-loss weight |
|---|---|---|---:|
| A0 | `explicit_product` | `joint` | 0 |
| A1 | `conditional_first_blocker` | `joint` | 0.6273209029272248 |
| A2 | `explicit_product` | `explicit_local` | 0 |
| A3 | `conditional_first_blocker` | `explicit_local` | 0.6273209029272248 |

The complete frozen design contract is:

```text
A0: router = explicit_product; gradient_ownership = joint; reason_loss_weight = 0
A1: router = conditional_first_blocker; gradient_ownership = joint; reason_loss_weight = 0.6273209029272248
A2: router = explicit_product; gradient_ownership = explicit_local; reason_loss_weight = 0
A3: router = conditional_first_blocker; gradient_ownership = explicit_local; reason_loss_weight = 0.6273209029272248

REASON_ORDER = FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED
SECONDARY_REASONS = DIAGNOSTIC_ONLY
FINAL_3WAY_CE = ROUTER_ONLY
ENCODER = FROZEN

frame_downstream_gradient_mode = joint
gradient_ownership_mode = the factorial ownership factor
```

`frame_downstream_gradient_mode = joint` is a separate global setting from
`gradient_ownership_mode` and must not be interpreted as the factorial
ownership factor.

Its scientific objective is to distinguish the conditional-first-blocker router
effect, the explicit-local ownership effect, whether conditional routing rescues
explicit-local ownership, the router x ownership interaction, and the
class/structural sources of change.  Primary outcome is macro-F1; accuracy is a
descriptive safeguard, not a replacement outcome.

## 3. Exact frozen cell metrics

The frozen validated per-seed cell table is reproduced below, one cell per
line for readability.

| Seed | Cell | Epoch | Accuracy | Macro-F1 | NOT_ENTITLED F1 | REFUTE F1 | SUPPORT F1 | Prediction distribution (NE/R/S) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
|180|A0|20|.909722|.804826|.942932|1.000000|.471545|599/89/32|
|180|A1|18|.898611|.783188|.935796|.994413|.419355|597/90/33|
|180|A2|16|.754167|.610488|.842718|.685714|.303030|490/156/74|
|180|A3|20|.805556|.618559|.910394|.615385|.329897|576/41/103|
|181|A0|17|.898611|.792734|.935455|1.000000|.442748|591/89/40|
|181|A1|19|.909722|.810307|.942731|1.000000|.488189|595/89/36|
|181|A2|20|.804167|.686835|.884507|.802260|.373737|525/88/107|
|181|A3|20|.838889|.635050|.932751|.666667|.305732|605/49/66|
|182|A0|20|.901389|.813349|.936550|1.000000|.503497|579/89/52|
|182|A1|19|.909722|.806536|.944688|.982857|.492063|599/86/35|
|182|A2|19|.813889|.707261|.885338|.863388|.373057|524/94/102|
|182|A3|20|.777778|.593109|.883888|.623188|.272251|571/49/100|

Structural accuracies (frame/predicate/sufficiency/entitled polarity) are:

| Seed | A0 | A1 | A2 | A3 |
|---:|---|---|---|---|
|180|.831944/.750000/1.000000/1.000000|.806944/.584722/1.000000/1.000000|.770833/.759722/.916667/.916667|.759722/.583333/.912500/.777778|
|181|.825000/.745833/1.000000/1.000000|.822222/.587500/.994444/1.000000|.823611/.726389/.984722/.900000|.802778/.583333/.911111/.783333|
|182|.838889/.784722/1.000000/1.000000|.833333/.626389/.998611/.988889|.823611/.768056/.944444/.933333|.830556/.583333/.893056/.816667|

Relevant pairwise diagnostics have 60 active groups in every cell.  Across all
cells, deletion and truncation sufficiency-lower pass rates are 1.000000;
entity-frame-lower is .783333--.950000 and event-frame-lower is
.900000--.950000, but neither passes its aggregate check.  Paraphrase-preserved
is .033333--.483333, polarity-flip-preserved-and-reversed is 0--.100000, and
predicate-disentangled is .033333--.300000; none passes its aggregate check.
Thus these diagnostic families do not supply a hidden favorable exception to the
final-task pattern.

## 4. Q1 — A1 minus A0: router effect under joint ownership

**Verdict: MIXED_OR_SEED_DEPENDENT.**  Macro-F1 deltas for seeds 180/181/182
are -.021638, +.017572, and -.006813 (mean -.003626; range .039210).
Accuracy deltas are -.011111, +.011111, and +.008333 (mean +.002778).
The signs reverse for macro-F1, accuracy, NOT_ENTITLED F1 (-.007136,
+.007276, +.008138), REFUTE F1 (-.005587, .000000, -.017143), and SUPPORT F1
(-.052190, +.045441, -.011433).  Structural behavior also has no coherent
task-level improvement: frame is slightly lower in all three cells, predicate
is lower, sufficiency remains near ceiling, and polarity is nearly unchanged.

Conditional-first-blocker alone therefore does **not** give reproducible
practical improvement over A0.  F1 is SUPPORTED: the observed mean-scale
macro-F1 change is near baseline while its signs differ by seed.

## 5. Q2 — A2 minus A0: explicit-local ownership under explicit-product

**Verdict: DESCRIPTIVELY_HARMFUL.**  Macro-F1 is lower in every seed:
-.194338, -.105900, -.106088 (mean -.135442; range .088438).  Accuracy is
also lower in every seed: -.155556, -.094444, -.087500 (mean -.112500).
Every class degrades: NOT_ENTITLED F1 -.100214/-.050948/-.051212; REFUTE F1
-.314286/-.197740/-.136612; SUPPORT F1 -.168514/-.069011/-.130440.

The prediction distribution moves away from A0's 579--599 NOT_ENTITLED,
89 REFUTE, and 32--52 SUPPORT counts toward fewer NOT_ENTITLED (490/525/524)
and substantially more SUPPORT (74/107/102), while REFUTE is especially
unstable in seed180 (156).  Structural accuracy does not rescue this result:
predicate is similar or higher in A2, but frame/sufficiency/polarity are lower
than A0 and final macro-F1 falls in all seeds.  F2, F5, and F6 are SUPPORTED.

## 6. Q3 — A3 minus A2: router effect under explicit-local ownership

**Verdict: MIXED_OR_SEED_DEPENDENT. Conditional routing does not show a
reproducible rescue of explicit-local ownership across the three frozen seeds.**
Macro-F1 deltas are +.008071, -.051785, -.114152 (mean -.052622; range
.122223); accuracy deltas are +.051389, +.034722, -.036111 (mean +.016667).
NOT_ENTITLED F1 changes +.067676/+.048244/-.001450, REFUTE F1
-.070330/-.135593/-.240200, and SUPPORT F1 +.026867/-.068005/-.100806.

Seed180 shows a small positive A3-A2 macro-F1 effect (+.008071), while seeds181
and 182 show negative effects (-.051785 and -.114152). Therefore conditional
routing does not establish a reproducible rescue of explicit-local ownership
across the three frozen seeds; the rule is MIXED, not uniformly supported.
REFUTE F1 falls in all three seeds, and A3 also retains low predicate,
sufficiency, and polarity accuracies. F4 is MIXED; F7 is SUPPORTED because
macro-F1 reverses by seed.

## 7. Q4 — A3 minus A1: explicit-local ownership under conditional-first-blocker

**Verdict: DESCRIPTIVELY_HARMFUL.**  Macro-F1 deltas are
-.164629/-.175257/-.213427 (mean -.184438; range .048798), and accuracy deltas
are -.093056/-.070833/-.131944 (mean -.098611): all three signs are negative.
NOT_ENTITLED F1 declines -.025402/-.009980/-.060800; REFUTE F1 collapses
-.379029/-.333333/-.359669; SUPPORT F1 declines -.089458/-.182456/-.219812.
The A3 distributions have only 41/49/49 REFUTE predictions versus A1's
90/89/86, alongside higher SUPPORT counts in seeds180 and 182.

Frame, predicate, sufficiency, and polarity are all lower for A3 than A1 in
every seed.  F3, F5, and F6 are SUPPORTED: explicit-local is harmful with the
conditional router, class collapse is obscured if one relies on accuracy alone,
and no structural improvement reverses the final-task degradation.

## 8. Q5 — factorial interaction

For `I = A3 - A2 - A1 + A0`, each seed is reported separately.

| Measure | Seed180 | Seed181 | Seed182 | Descriptive assessment |
|---|---:|---:|---:|---|
| Macro-F1 | +.029709 | -.069357 | -.107339 | heterogeneous |
| Accuracy | +.062500 | +.023611 | -.044444 | heterogeneous |
| NOT_ENTITLED F1 | +.074812 | +.040968 | -.009588 | heterogeneous |
| REFUTE F1 | -.064743 | -.135593 | -.223057 | antagonistic, with magnitude heterogeneity |
| SUPPORT F1 | +.079056 | -.113446 | -.089373 | heterogeneous |

The macro-F1 interaction mean is -.048996 (population SD .057772; range
.137048), so the factorial is **heterogeneous**, not approximately additive,
synergistic, or stably antagonistic as a whole.  F8 is SUPPORTED: interaction
sign and magnitude differ materially across the three frozen seeds.

## 9. Q6 — failure localization

Final-task behavior localizes the main degradation to explicit-local ownership,
not to a uniform change in a single structural diagnostic.  Relative to A0,
A2 loses all three external classes, with the largest class loss in REFUTE
(-.314286/-.197740/-.136612) and substantial SUPPORT loss.  Relative to A1,
A3 has an even more consistent REFUTE collapse (-.379029/-.333333/-.359669),
accompanied by SUPPORT and NOT_ENTITLED declines.  It is therefore most clearly
associated with **REFUTE instability/collapse**, accompanied by SUPPORT loss;
it is not supported as primarily an overproduction-of-NOT_ENTITLED pattern.

Prediction counts agree with that localization.  A2 under explicit-product
often shifts mass from NOT_ENTITLED toward SUPPORT and, for seed180, REFUTE.
A3 under conditional routing has markedly fewer REFUTE predictions than A1 in
every seed and unstable SUPPORT counts.  These shifts are descriptive evidence
about final decisions, not a threshold diagnosis.

Internal structural diagnostics tell a separate, non-overriding story.  A1's
predicate accuracy is lower than A0 despite near-neutral final macro-F1.  A2's
predicate accuracy can remain near A0 while frame, sufficiency, and polarity
decline and final macro-F1 is much lower.  A3 is lower than A1 on frame,
predicate, sufficiency, and polarity in all seeds.  Pairwise sufficiency-lower
rates remain high, but paraphrase, polarity-flip, and predicate-disentanglement
families are weak throughout.  Therefore structural diagnostic improvement, if
any isolated value is considered, does not override the degraded macro-F1.

## 10. Cross-seed synthesis

| Main contrast | Macro-F1 signs, seed180/181/182 | Mean-scale macro-F1 behavior | Accuracy signs | Synthesis |
|---|---|---:|---|---|
| A1-A0 | - / + / - | -.003626; range .039210 | - / + / + | mixed; sign reversal |
| A2-A0 | - / - / - | -.135442; range .088438 | - / - / - | directionally consistent across the three frozen seeds |
| A3-A2 | + / - / - | -.052622; range .122223 | + / + / - | mixed; sign reversal |
| A3-A1 | - / - / - | -.184438; range .048798 | - / - / - | directionally consistent across the three frozen seeds |

All means, ranges, and the interaction SD/range above are frozen/derived
descriptive summaries only; N=3 does not support an inferential claim.

## 11. Falsification-matrix assessment

| ID | Assessment | Evidence |
|---|---|---|
| F1 A1 approximately A0 | SUPPORTED | Macro-F1 mean -.003626 with -/+/− signs; no reproducible practical router-only gain. |
| F2 A2 below A0 | SUPPORTED | Macro-F1 and accuracy are lower in all three seeds. |
| F3 A3 below A1 | SUPPORTED | Macro-F1 and accuracy are lower in all three seeds. |
| F4 A3 not above A2 | MIXED | Seed180 is +.008071, while seed181 is -.051785 and seed182 is -.114152; conditional routing does not establish a reproducible rescue across the three frozen seeds. |
| F5 class collapse vs aggregate accuracy | SUPPORTED | Large REFUTE and SUPPORT losses accompany the ownership contrasts. |
| F6 structural gains vs task degradation | SUPPORTED | Isolated predicate/pairwise behavior does not offset final macro-F1 loss. |
| F7 seed reversals | SUPPORTED | Q1 and Q3 macro-F1 signs reverse by seed. |
| F8 interaction heterogeneity | SUPPORTED | Interaction signs vary for macro-F1, accuracy, NOT_ENTITLED, and SUPPORT. |

## 12. Required bounded scientific conclusion

What is supported: (A) conditional-first-blocker alone is
MIXED_OR_SEED_DEPENDENT and near-neutral on the mean scale relative to A0, so
it does not establish reproducible practical improvement; (B) explicit-local
ownership is DESCRIPTIVELY_HARMFUL under explicit-product; (C) conditional
routing does not establish a reproducible rescue of explicit-local ownership
across the three frozen seeds; (D) the
factorial interaction is heterogeneous; and (E) this N=3 factorial does not
support promotion of any tested changed configuration.

What is not supported: a beneficial router-only effect, beneficial
explicit-local ownership, a stable rescue by conditional routing, or a stable
factorial interaction.  This is a valid descriptive negative result within the
frozen three-seed boundary.

## 13. Promotion boundary

This N=3 factorial **does not support promotion** of
`conditional_first_blocker`, `explicit_local` ownership, or the A3 combined
configuration.  It neither replaces A0 nor authorizes replacement,
implementation, or execution.

## 14. Explicit non-claims

This report does **not** establish statistical significance, population
generalization, a causal mechanism, universal superiority or harm, production
readiness, architecture promotion authorization, an optimal reason-loss weight,
new gradient semantics, E0/A4 authorization, or new seed/arm execution.  It
does not claim that a result is proven, generalizes, production ready, or
execution authorized.

## 15. Seed180 A3 provenance caveat

Imported seed180 A3 scientific evidence is valid.  The `-retry1` naming is
controller-history context; repo evidence verification of that suffix is
NOT_ESTABLISHED; and the suffix is not a scientific premise.  This caveat does
not create scientific uncertainty about the validated A3 numerical cell.

## 16. Next research implication

The narrow scientific implication is to retain A0 as the current reference,
deprioritize explicit-local ownership as currently formulated, and investigate
SUPPORT/REFUTE failure localization and the near-neutral/mixed router behavior
only through a later, separately authored authority.  That later authority may
also consider revisiting the gradient-ownership formulation before any further
factorial work.  This report does not select a hyperparameter, schedule a run,
or authorize implementation.
