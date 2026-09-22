# ContraMamba Gen4 Discussion and Limitations Manuscript Draft v1

## Status

- Status: STATIC MANUSCRIPT DISCUSSION DRAFT
- Evidence HEAD: `e22257df78f2a6ce0ea3d531103a0f00d53a947d`
- Scientific execution: CLOSED
- New scientific claims: NONE
- New model execution: NONE
- New statistical tests: NONE
- New p-values: NONE

This document interprets the frozen Results without expanding their scope.

---

## 4. Discussion

Our results separate three properties that can otherwise be conflated in
cross-model mechanistic analysis: recurrence of a causal role, preservation of
its geometric realization, and preservation of its downstream functional
alignment. Across the tested Mamba checkpoints, these properties do not move
together.

The scale-local dominant causal role recurs under the same reconstruction and
matched-control program, but the coordinates that realize that role do not
remain fixed. The selected plane rank changes, the residual coefficient mass
redistributes, signed residual effects reorganize, and generator-family
coupling changes. We therefore interpret the cross-checkpoint result as
`CORE-STABLE / RESIDUAL-PLASTIC`: recurrence is a property of the causal role
under the frozen procedure, not evidence for a checkpoint-invariant principal
plane.

The downstream readout result provides a second separation. In the matched
Mamba-370M-versus-Mamba-1.4B population, the independently frozen scale-local
causal displacements have opposite mean alignment with the downstream
correct-class task margin. Thus recurrence of the internal causal role does
not guarantee recurrence of its downstream functional orientation. The
Mamba-130M positive readout supplies a separately frozen contextual scale
point, but it is not part of the matched 370M-versus-1.4B inferential sample
and is not used to fit a scaling law.

These observations refine what should be meant by mechanistic recurrence.
Finding an analogous causal role at another checkpoint is evidence about the
organization of causal computation, but it does not by itself establish that
the same coordinate system, intervention direction, or downstream task effect
has been preserved.

### 4.1 Geometric reorganization is functionally relevant but incomplete

The Mamba-1.4B cross-block transport experiment helps connect geometric
reorganization to causal function. The canonical P5 plane is strongly
reoriented relative to the independently reconstructed adjacent plane, and
transporting the canonical geometry into the adjacent site moves the causal
response in the canonical direction. This establishes that the geometric
change is not merely a representational relabeling.

At the same time, transport does not recover the canonical positive response.
The transported mean remains negative and fails the pre-specified positive
restoration gate. Geometric reorientation therefore contributes causally to
the local functional difference without fully accounting for it.

This distinction is important. A significant relative transport effect is not
equivalent to complete mediation, and ratios between canonical, adjacent, and
transported means do not define an explained percentage. The remaining
functional discrepancy is left unresolved rather than assigned post hoc to an
additional mechanism.

### 4.2 Downstream orientation is not fixed by internal causal recurrence

The matched readout experiment directly tests whether recurrence of the
internal role preserves its downstream orientation. The answer is negative
within the tested checkpoints. Mamba-370M has positive mean local alignment,
whereas Mamba-1.4B has negative mean local alignment on the same 300 source
pairs.

The historical stored readout values are gradient-ownership-weighted
quantities. Under the frozen `G3-GROUP-D-HALF` graph, the corresponding
numerical forward directional derivative is exactly

`Delta_L_forward = 2 * Delta_L_owned`.

This correction changes the magnitude units but not signs, rankings, paired
t statistics, or p-values. The substantive result is therefore the
cross-checkpoint orientation difference, not the factor-of-two bookkeeping
itself.

The direct behavioral intervention results are consistent with a similarly
bounded conclusion. Positive downstream behavioral coupling is established at
370M, whereas the same pre-specified positive bridge is not established at
1.4B. We therefore do not interpret the checkpoint sequence as a monotonic
behavioral scaling trend.

### 4.3 Implications for transferring mechanistic interventions

A practical implication is that intervention directions should be revalidated
after changing checkpoint or model scale. An apparently analogous causal role
may recur even when its geometric realization and downstream readout
orientation have changed.

This matters for workflows that transfer editing, steering, or mechanistic
correspondences from one checkpoint to another. The present results do not
show that such transfer generally fails. They show that recurrence of an
internal causal role is insufficient evidence, by itself, for assuming
coordinate-level or downstream-functional transfer.

The appropriate operational lesson is therefore revalidation rather than
non-transferability: checkpoint changes can preserve causal role while altering
the geometry and task coupling relevant to a particular intervention.

### 4.4 Mechanistic validity and control utility are distinct

The natural-language and steering results provide a useful boundary on the
interpretation of causal mechanisms. The frozen intervention transfers at the
correct-class-margin level to AVeriTeC gold-evidence inputs at 130M and 370M,
showing that the synthetic mechanism is not confined entirely to the XG1
generator.

However, the preregistered fixed-mirror Mamba-370M steering intervention
produces neither corrections nor damages at the prediction level. Mechanistic
validity, measurable margin influence, and useful discrete control are
therefore empirically distinct outcomes in this study.

This null result should not be generalized into a claim that Mamba is
unsteerable. Only one frozen steering transformation was tested. Conversely,
the existence of a causally validated direction should not be taken as
evidence that a fixed application of that direction will improve predictions.

---

## 5. Limitations

### 5.1 Checkpoint scope and model-seed generalization

The principal cross-checkpoint conclusions concern three fixed pretrained
Mamba checkpoints. The matched 370M-versus-1.4B inferential test estimates
variation across matched items for those two checkpoints; it does not estimate
variation across independently trained model seeds.

Accordingly, the results do not establish a population-level scaling law, a
parameter-count phase transition, a zero-crossing threshold, or a universal
trajectory with increasing model size.

### 5.2 Architecture scope

All primary evidence is obtained in the tested Mamba checkpoints. The study
does not establish architecture-independent universality and does not show
that the same recurrence/geometry/readout separation holds in Transformers or
other state-space architectures.

The paper therefore treats the result as an empirical property of the tested
Mamba program rather than a theorem about neural mechanisms in general.

### 5.3 Geometry is procedure-defined and checkpoint-local

The principal planes are independently reconstructed within each checkpoint.
Their integer ranks are local labels rather than semantic identifiers shared
across scales.

The observed causal recurrence is consequently defined with respect to the
frozen reconstruction, intervention, and matched-control procedure. Other
valid decompositions could expose complementary structure. The present study
does not claim that its selected planes are the unique causal coordinates of
the model.

### 5.4 Residual reorganization is descriptive across checkpoints

The residual signed-effect and coefficient-mass comparisons are intentionally
descriptive. No cross-checkpoint residual significance test was prospectively
frozen, and none is added after observing the results.

The `RESIDUAL-PLASTIC` label therefore summarizes the observed reorganization
across the tested checkpoints; it is not an omnibus inferential claim that
every aspect of the residual must differ between every pair of model scales.

### 5.5 Cross-block transport does not identify a complete mediator

The transported canonical geometry significantly shifts the adjacent response
toward the canonical direction, but positive restoration is not achieved.
The experiment therefore supports causal contribution rather than complete
explanation.

The remaining difference may reflect additional downstream geometry, nonlinear
state dependence, other local subspaces, or mechanisms outside the measured
intervention surface. The current evidence does not distinguish among these
possibilities.

### 5.6 Site specificity is bounded

The 1.4B site-specificity experiment compares the canonical site with exactly
one prospectively fixed adjacent `+1` site. It establishes that the canonical
site is stronger under that matched comparison.

It does not identify a global optimum over layers and does not rule out other
sites with similar or stronger causal effects.

### 5.7 Finite-difference scope

The local susceptibility program uses a primary finite-difference scale of
`epsilon=0.025`, with smaller finite perturbations used only for descriptive
robustness. Stability across the tested finite values does not establish an
exact infinitesimal limit.

### 5.8 External-validity scope

The AVeriTeC experiment uses the compatible three-class development subset and
annotated gold evidence. Retrieval is deliberately removed from the causal
transfer question.

The result therefore does not measure end-to-end fact-verification performance,
retrieval quality, benchmark superiority, or robustness to retrieved evidence
noise. The 1.4B negative external-transfer extension also does not pass its
pre-specified significance gate, so negative natural-language transfer at
1.4B is not established.

### 5.9 Steering scope

The failed steering result concerns one preregistered fixed-mirror
intervention rule at Mamba-370M. It does not evaluate adaptive steering,
optimization-based control, alternative intervention magnitudes, or different
control objectives.

No additional steering transform is introduced to rescue the null result.

### 5.10 Statistical and multiplicity boundaries

Inferential tests are attached only to endpoints prospectively designated for
inference. Descriptive analyses such as residual decomposition, cross-block
geometry summaries, and stagewise localization are not promoted to
confirmatory claims by adding post-hoc p-values.

This constraint limits the number of formal cross-analysis comparisons that
can be made, but preserves the distinction between prospectively tested
hypotheses and descriptive mechanism characterization.

---

## 6. Discussion synthesis

Across the tested Mamba checkpoints, a causal role can recur without
preserving either a fixed geometric realization or a guaranteed downstream
readout orientation. Cross-block transport shows that geometric reorientation
is causally relevant but incomplete, while the matched readout reversal shows
that downstream functional alignment can change even when the internal role
recurs.

The resulting picture is not one of mechanistic instability in every sense.
Rather, different levels of description have different invariants: causal role
can be comparatively stable while geometry and downstream coupling remain
plastic. This separation motivates treating role recurrence, coordinate
correspondence, and intervention transfer as distinct empirical questions.

For mechanistic practice, the immediate implication is bounded but concrete:
revalidate intervention geometry and downstream alignment after checkpoint or
scale changes instead of assuming transfer from causal-role recurrence alone.

---

## Evidence mapping for manuscript verification

This discussion is constrained by:

- `reports/reason_router_gen4_final_paper_claim_disposition.md`
- `reports/reason_router_gen4_paper_novelty_positioning_matrix.md`
- `reports/reason_router_gen4_introduction_central_question_freeze.md`
- `reports/reason_router_gen4_results_integrated_manuscript_v1.md`
- `reports/reason_router_gen4_methods_manuscript_draft_v1.md`

The evidence-mapping block is an internal drafting aid and should not appear
in the submitted manuscript.

`DISCUSSION_LIMITATIONS_MANUSCRIPT_DRAFT_V1_READY = YES`
