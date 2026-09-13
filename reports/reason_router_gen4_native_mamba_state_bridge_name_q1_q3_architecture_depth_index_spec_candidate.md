# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Architecture-Defined Depth Index Specification
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_ARCHITECTURE_DEPTH_INDEX_SPECIFICATION

THIS_DOCUMENT_CREATES_NEW_SCIENTIFIC_EVIDENCE =
NO

IMPLEMENTATION_ALLOWED =
NO

STATISTICAL_TESTING_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_LOADING_ALLOWED =
NO

TOKENIZER_EXECUTION_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_ALLOWED =
NO

## 1. Frozen lineage

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

PHASE_F_VALIDATED_STATISTICAL_RESULT_FREEZE =
ab3428e7be08af26fa1fdafd1483a34e48fbcf8c

POST_PHASE_F_MECHANISTIC_INTERPRETATION =
029182257c3f9d89d41c0f37dd508eb3d5f405cd

DEPTH_SELECTIVITY_INTERPRETATION_CORRECTION =
2e076cbd8e9633c3ab7abb222a05e409366539a7

PHASE_D_CORRECTED_IMPLEMENTATION =
c7ae7fac4c64e9bd64adcae819a3da3dd46f17f7

FROZEN_MEASUREMENT_IMPLEMENTATION_PATH =
scripts/reason_router_gen4_native_mamba_state_measurement.py

FROZEN_MEASUREMENT_IMPLEMENTATION_SHA256 =
7729424f03058b86b4f120dc0e6da573d6c996b0877858f2d6d38aa94dac268c

## 2. Purpose

The frozen mechanistic bridge specification permits Q1 and Q3 recurrent
layers only under a separate secondary authority.

It does not itself define their exact layer indices.

This document freezes that missing architecture-only indexing rule before any
Q1/Q3 native-state outcome is computed or inspected.

OUTCOME_DEPENDENT_LAYER_SELECTION =
PROHIBITED

BEST_LAYER_SELECTION =
PROHIBITED

## 3. Frozen architecture coordinate

The validated measurement implementation binds:

NATIVE_MAMBA_LAYER_COUNT =
24

LAYER_INDEX_DOMAIN =
0_THROUGH_23

PRIMARY_LAYER_INDEX =
11

The primary layer is already frozen by the original rule:

L_PRIMARY =
floor((L - 1) / 2)

For L = 24:

L_PRIMARY =
floor(23 / 2)
=
11

This document does not alter the primary layer.

## 4. Secondary quartile rule

The secondary layers are defined from the same zero-based architecture index
interval:

0 ... L - 1

Define:

L_Q1 =
floor((L - 1) / 4)

L_Q3 =
floor(3 * (L - 1) / 4)

This rule uses only architecture depth.

It does not use:

- Phase F effect sizes;
- Phase F p-values;
- R6 effect sizes;
- native-state trajectories;
- visualization;
- downstream prediction;
- best-layer search;
- any future Q1/Q3 result.

SECONDARY_LAYER_RULE =
ARCHITECTURE_INDEX_INTERVAL_QUARTILES

## 5. Exact Gen4 secondary indices

For:

L =
24

therefore:

L_Q1 =
floor(23 / 4)
=
5

L_Q3 =
floor(69 / 4)
=
17

Q1_LAYER_INDEX =
5

Q3_LAYER_INDEX =
17

PRIMARY_LAYER_INDEX =
11

The three prespecified depth locations are therefore:

Q1 =
5

MIDPOINT =
11

Q3 =
17

## 6. Scope of the rule

These indices are frozen for the planned NAME secondary-layer localization
only.

They do not authorize:

- extraction at layers 5 or 17;
- statistical testing;
- model execution;
- new endpoint construction;
- all-layer analysis;
- layer interpolation;
- nearest-significant-layer substitution.

If layer 5 or layer 17 later fails an implementation or runtime provenance
gate, that failure is a blocker.

It may not be repaired by choosing another layer based on observed outcomes.

FALLBACK_LAYER_SELECTION =
PROHIBITED

## 7. Preserved scientific question

NEXT_PRIMARY_FOLLOWUP_HYPOTHESIS =
NAME_SECONDARY_LAYER_LOCALIZATION

STRUCTURAL_ESTIMAND =
DELTA_NAME

CELL_CONTRAST =
C2_NAME_MINUS_C0_SHAM

SEMANTIC_ANCHOR =
A_NAME

PRESERVED_ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

SECONDARY_LAYERS =
5
17

PLANNED_HYPOTHESIS_COUNT =
6

The six future hypotheses are:

Q1 / POST4_SPEED / DELTA_NAME
Q1 / POST4_TURNING / DELTA_NAME
Q1 / POST4_PATH_EFFICIENCY / DELTA_NAME
Q3 / POST4_SPEED / DELTA_NAME
Q3 / POST4_TURNING / DELTA_NAME
Q3 / POST4_PATH_EFFICIENCY / DELTA_NAME

No statistical decision is made by this document.

## 8. Cross-layer interpretation boundary

This rule defines locations.

It does not define or test a between-layer contrast.

CROSS_LAYER_DIFFERENCE_ESTIMAND =
NOT_DEFINED

CROSS_LAYER_DIFFERENCE_TEST =
NOT_AUTHORIZED

DEPTH_SELECTIVITY =
NOT_ESTABLISHED

A future supported result at layer 5 or 17 may establish only the bounded
secondary-layer localization claim permitted by correction commit:

2e076cbd8e9633c3ab7abb222a05e409366539a7

It may not by itself establish a significant difference versus midpoint layer
11.

## 9. Multiplicity boundary

The planned secondary scientific family contains exactly:

2 layers
x
3 endpoints
x
1 structural estimand

equals:

SECONDARY_FAMILY_HYPOTHESIS_COUNT =
6

The later statistical specification must control multiplicity over all six
planned secondary hypotheses together unless a different procedure is frozen
before Q1/Q3 outcome inspection.

PHASE_F_PRIMARY_FAMILY =
UNCHANGED_AND_CLOSED

The secondary family must not replace, revise, or merge away any Phase F
decision.

## 10. Execution boundary

Q1_Q3_NATIVE_STATE_EXTRACTION =
NOT_AUTHORIZED

Q1_Q3_STATISTICAL_EXECUTION =
NOT_AUTHORIZED

Q1_Q3_MODEL_EXECUTION =
NOT_AUTHORIZED

KAGGLE_EXECUTION =
NOT_AUTHORIZED

GPU_EXECUTION =
NOT_AUTHORIZED

NEXT_EXECUTION =
NOT_AUTHORIZED

NEXT_PHASE =
NAME_Q1_Q3_SECONDARY_LAYER_LOCALIZATION_SCIENTIFIC_SPECIFICATION
