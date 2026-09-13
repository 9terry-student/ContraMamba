# ContraMamba Gen4 Native Mamba State Bridge
# Phase F Output-Hash Self-Reference Correction
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_F_OUTPUT_HASH_SELF_REFERENCE_CORRECTION

## Affected authorities

PHASE_F_STATISTICAL_SPECIFICATION_COMMIT =
830f7ea697ce24388dddd181c9aa301ec2b442fc

PHASE_F_IMPLEMENTATION_AUTHORITY_COMMIT =
d249de57dc8dac69f49f7110c6f1a07532f939cb

## Defect

DEFECT_CLASS =
SELF_REFERENTIAL_OUTPUT_MANIFEST_HASH_CONTRACT

The Phase F statistical specification requires the statistical-analysis
manifest to bind all output SHA256 values.

If that phrase includes the manifest's own SHA256, the contract is
self-referential: changing the manifest to record its own digest changes the
digest being recorded.

A deterministic exact-byte artifact cannot satisfy that interpretation.

## Corrected output-hash contract

OUTPUT_FILE_COUNT =
4

OUTPUT_FILES =
phase_f_pair_level_contrasts.csv
phase_f_primary_confirmatory_results.csv
phase_f_statistical_analysis_manifest.json
phase_f_statistical_analysis_report_candidate.md

MANIFEST_PEER_ARTIFACT_SHA256_BINDING =
REQUIRED

The manifest must record exact SHA256 values for these three peer artifacts:

phase_f_pair_level_contrasts.csv
phase_f_primary_confirmatory_results.csv
phase_f_statistical_analysis_report_candidate.md

MANIFEST_SELF_SHA256_FIELD =
FORBIDDEN

MANIFEST_SELF_SHA256_AUTHORITY =
FOLLOWUP_PHASE_F_ARTIFACT_FREEZE

The manifest's own exact SHA256 must be established externally after its
canonical bytes have been written, by the later Phase F artifact-freeze
authority/commit.

## Deterministic write order

1. construct deterministic pair-level CSV bytes;
2. construct deterministic primary-results CSV bytes;
3. construct deterministic report bytes;
4. compute SHA256 of those three exact byte strings;
5. construct canonical JSON manifest containing those three peer SHA256 values;
6. write exactly the four frozen output files.

No output file may contain a claimed hash of its own exact bytes.

## Scientific scope

SCIENTIFIC_QUESTION_CHANGE =
NONE

PRIMARY_HYPOTHESIS_COUNT_CHANGE =
NONE

ESTIMAND_CHANGE =
NONE

ENDPOINT_CHANGE =
NONE

STATISTICAL_TEST_CHANGE =
NONE

MULTIPLICITY_CHANGE =
NONE

DECISION_RULE_CHANGE =
NONE

INPUT_ARTIFACT_CHANGE =
NONE

CANONICAL_STATISTICAL_EXECUTION =
NOT_AUTHORIZED

This correction affects deterministic output provenance only.

The remaining Phase F statistical specification and implementation authority
remain in force.

SCIENTIFIC_CONCLUSION =
NONE

NEXT_PHASE =
PHASE_F_STATISTICAL_ANALYSIS_IMPLEMENTATION_AND_SYNTHETIC_VALIDATION
