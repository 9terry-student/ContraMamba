# Stage180-A hard-frame manual taxonomy codebook

## Authority and scope

This codebook applies only to the fixed Stage176 hard-39 clean-dev cohort and its matched correct controls. It guides review; it does not change labels.

The authoritative native target is the row field `frame_compatible_label`. Its operational semantics come from the controlled-data schema and construction rules in `scripts/build_controlled_v5.py`, especially `_record` and `_build_records`. That source keeps frame compatibility separate from predicate coverage, sufficiency, polarity, and the final three-way label:

- canonical `none`, `paraphrase`, `predicate_swap`, `evidence_deletion`, `evidence_truncation`, and `polarity_flip` rows are constructed with frame label 1;
- `entity_swap`, `event_swap`, `time_swap`, `location_swap`, `role_swap`, `title_name_swap`, and `irrelevant_evidence` are constructed with frame label 0;
- predicate mismatch, polarity reversal, or missing detail therefore does not by itself make the frame incompatible; and
- intervention names describe construction provenance and must never be used as an annotation shortcut.

For an independent semantic judgment, treat claim and evidence as frame-compatible when they concern the same identifiable event frame: the relevant participant/entity, event object or identity, time, location, role/name identity, and referential structure align sufficiently for the evidence to address the claimed event. Judge this relation directly from the text. Then assess predicate coverage, polarity, and evidence sufficiency separately during diagnostic review.

Packet construction must hard-block if this source is missing, if the data lacks `frame_compatible_label`, or if another supplied authoritative source conflicts with these construction rules. Do not invent a replacement definition.

## Prohibited inference

- Do not infer an answer from the intervention-family name.
- Do not infer frame compatibility from the final SUPPORT/REFUTE/NOT_ENTITLED label.
- Do not revise Pass 1 after seeing a model prediction or native label.
- Do not equate model error with annotation error.
- Do not automatically modify data from one reviewer's judgment.
- Do not treat a near duplicate as a label conflict without semantic adjudication.

## Two-pass protocol

### Pass 1: blinded semantic review

Review only the randomized instance ID, claim, evidence, minimal segmentation supplied in the packet, this codebook, and free-text fields. The packet withholds hard/control identity, native and final labels, every model prediction/score/margin, Stage176 cohort, intervention family, pair identity, representation/centroid diagnostics, and source run.

Choose one independent frame judgment:

- `compatible`: the texts address the same event frame, even if predicate, polarity, or sufficiency differs;
- `incompatible`: a material frame slot or referent identifies a different event frame;
- `ambiguous`: the text permits more than one reasonable frame reading;
- `insufficient_context`: the provided text does not permit the relation to be judged.

Record an integer confidence from 1 (very uncertain) to 5 (very confident) and a non-empty rationale grounded only in the visible text.

### Pass 2: unblinded diagnostic review

Begin only after Pass 1 has been frozen. The unblinded packet may reveal the native frame label, intervention, the pair's canonical `none` row, Stage176 cohort, native frame output, final prediction/gold label, frame-head projection, representation movement, Stage179 centroid result, and matched-control metadata. These fields diagnose failure locus and data design; they do not overwrite Pass 1.

## Annotation axes

### Gold-frame assessment

`gold_consistent`, `gold_questionable`, `gold_likely_incorrect`, `cannot_determine`

### Intervention validity

`clean_single_axis_edit`, `valid_but_multi_axis_edit`, `weak_or_ineffective_edit`, `unnatural_or_broken_text`, `canonical_control`, `cannot_determine`

Use `canonical_control` only for an unedited canonical control identified in Pass 2. A matched control with another intervention is evaluated under the ordinary validity values.

### Primary semantic phenomenon

`entity_identity`, `event_identity`, `location_scope`, `role_relation`, `title_name_identity`, `predicate_scope`, `polarity`, `temporal_scope`, `referent_resolution`, `evidence_sufficiency_interaction`, `world_knowledge_dependency`, `lexical_or_surface_artifact`, `other`, `cannot_determine`

Select the main phenomenon needed to understand the case, not merely the intervention name.

### Diagnostic failure locus

`input_representation_insensitivity`, `head_direction_or_readout`, `downstream_final_boundary`, `data_or_intervention_design`, `label_semantics_or_annotation`, `genuinely_hard_semantic_case`, `mixed`, `cannot_determine`

This is a reviewer taxonomy, not causal proof. Use `mixed` when more than one locus is materially supported and no single locus dominates.

### Recommended data action

`keep_unchanged`, `manual_adjudication`, `revise_frame_label`, `rewrite_claim_or_evidence`, `redesign_intervention`, `add_minimal_counterpart`, `exclude_from_training`, `retain_as_diagnostic_only`, `cannot_determine`

Recommendations enter a review queue only. No option authorizes automatic relabeling, exclusion, or dataset editing.

### Confidence and rationale

Pass 2 confidence is an integer 1–5. The rationale is required, should cite the decisive text or diagnostic evidence, and should distinguish observation from causal inference.

## Edge cases

- If evidence mentions the same participant but no event, distinguish frame addressability from evidence sufficiency; use ambiguity or insufficient context only when the event frame itself cannot be resolved.
- If a title or role changes while the named person remains the same, decide whether the change identifies another participant/event or only changes a descriptive predicate.
- If pronouns or descriptions have multiple plausible antecedents, use `referent_resolution` and avoid forced binary judgment.
- If an edit changes several slots, judge the resulting frame independently in Pass 1 and mark `valid_but_multi_axis_edit` in Pass 2.
- Surface awkwardness is not automatically frame incompatibility; record it as intervention validity or `lexical_or_surface_artifact`.

## Adjudication boundary

Disagreement, low confidence, questionable gold, and likely label error all require explicit human adjudication. Reviewer counts and majority agreement are descriptive only. The analyzer must never emit or apply an automatic replacement label.
