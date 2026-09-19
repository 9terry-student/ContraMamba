# ContraMamba Gen4 — Pre-emission Stage A All-Abstain Closure

## Status

This document is a static scientific closure of the frozen Stage A raw generation artifact.

It is not an execution authority, does not authorize new generation, does not add statistical tests, and does not reopen the frozen finite-grammar Stage A protocol.

## Frozen evidence identity

- Repository branch: `gen4-mamba370m-core-replication`
- Raw evidence freeze commit: `a58b1ebef0957a8437e0255fb4069d2fe9cf8449`
- Execution commit: `db24de69ed737fab4f484d2f6e273c8286988d89`
- Pre-emission design freeze commit: `8879e80db019384eef37e63e1095e9f79366d60f`
- Causal-LM identity / grammar gate freeze commit: `f3f38b3a96cd2436ffbbe9de0c229a78378abf61`
- Run name: `g4k-preemission-stagea-raw-370m-db24de6-gpu2`
- Frozen raw artifact root:
  `reports/reason_router_gen4_pre_emission_stage_a_generation_runs/g4k-preemission-stagea-raw-370m-db24de6-gpu2/`

Frozen artifact identities:

- `SHA256SUMS.txt`
  - git blob: `248f39e1b5d419ffbb3115f288986c1f66480db0`
- `execution_summary.json`
  - git blob: `6ffa9bc36d028a6fe66364b05b926fb7807c5afb`
  - SHA256: `cd319ff1c8a1eefabb4aef49738a382ea314a75788cca375e7d6dac5ef51c003`
- `stage_a_generation_rows.jsonl`
  - git blob: `a1720bedd62bc2a5e41491c581179dc61dd42848`
  - SHA256: `0398560b39660d2af76d1d508d0e193f4c1f9cd3b4d01bda3be9ee73554055fb`

Partition identity:

- deterministic response-blind split;
- calibration: 231;
- confirmatory: 231;
- partition SHA256:
  `871fb5c1e2c62f247c284ceae409ef9ec19acc75f63ffc832829c2664db46311`.

## Frozen Stage A protocol relevant to closure

The frozen design defines the primary Stage A comparison as:

- unsupported decisive commitments
  versus
- supported decisive commitments

at relative pre-emission offsets:

- `t*-4`
- `t*-3`
- `t*-2`
- `t*-1`

using the already-frozen block-35 P3 signal.

The raw implementation additionally froze the intended confirmatory family as:

- partition: `confirmatory`;
- signal: `p3_component_l2`;
- test: two-sided Welch t-test;
- multiplicity: Holm;
- planned p-value count: 4;
- alpha: 0.05.

The raw execution correctly did **not** run that family.

## Observed raw generation outcome

The valid Stage A execution completed with 462 rows and 6,468 scientific model forwards.

Generated commitment counts:

- `NOT_ENTITLED`: 462
- `REFUTE`: 0
- `SUPPORT`: 0

Outcome counts:

- `nondecisive_abstention_error`: 427
- `nondecisive_correct_abstention`: 35
- `unsupported_decisive`: 0
- `supported_decisive`: 0

Therefore every generated sequence was non-decisive under the frozen primary event definition.

The raw artifact records:

- `primary_inference_executed = false`;
- `p_value_count_added = 0`;
- `training_executed = false`;
- `backward_executed = false`;
- `selection_reopened = false`;
- `layer_scan_executed = false`;
- `rescue_performed = false`;
- `stage_b_executed = false`;
- `stage_c_executed = false`;
- `scientific_conclusion = null`.

## Stage A closure

The frozen Stage A primary comparison requires both unsupported decisive commitments and supported decisive commitments.

The execution produced neither group.

Therefore the preregistered Stage A inferential family is **not estimable**. No Welch test and no Holm correction are defined for an empty primary comparison, and no p-value should be manufactured.

The correct bounded closure labels are:

`FINITE_GRAMMAR_GREEDY_DECODING_ALL_ABSTAINED`

`PRIMARY_STAGE_A_TEMPORAL_PRECEDENCE_NOT_ESTIMABLE`

This is not equivalent to:

`PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`

The latter would require an estimable Stage A comparison that failed to support temporal precedence. That comparison never existed in this execution.

## Scientific interpretation

The experiment established a decoding-regime fact, not a precursor-mechanism result:

> Under the frozen finite commitment grammar and frozen greedy constrained decoding protocol, the Mamba-370M model selected `NOT_ENTITLED` for all 462 AVeriTeC-compatible examples.

Because decisive commitment incidence was zero, the execution cannot answer whether frozen block-35 P3 differs before unsupported versus supported decisive commitments.

Accordingly:

- Stage A does not support `PRE_EMISSION_TEMPORAL_PRECEDENCE_OBSERVED`;
- Stage A also does not support a negative temporal-precedence claim;
- Stage B is not licensed by this Stage A result;
- Stage C is not licensed;
- no bounded precursor claim is licensed from this run;
- no broad hallucination-precursor claim is licensed.

The all-abstain outcome must not be treated as evidence that P3 lacks prospective information. It only blocks the frozen primary comparison under this decoding regime.

## Anti-rescue boundary

Generation responses from this protocol have now been observed and frozen.

Therefore the following changes cannot be made and then represented as a continuation of this same Stage A protocol:

- changing the commitment surfaces because they yielded all abstentions;
- removing or weakening the `NOT_ENTITLED` alternative because it dominated;
- selecting a subset of items based on emitted commitment;
- choosing a temperature, beam rule, logit bias, or decoding threshold based on these outcomes;
- choosing new layers, coordinates, offsets, or P3 variants based on these outcomes;
- running the planned Stage A p-values on a post-hoc redefined decisive subset.

Any such investigation must be a separately preregistered prospective experiment with a new execution identity.

## Next authorized research action

Do not proceed to Stage B or Stage C.

The next research step is a new **prospective decisive-generation protocol design**, separated from the closed Stage A run. Its decoding rule must be specified before any outputs from that new protocol are inspected, while preserving:

- the frozen Mamba-370M backbone identity;
- the frozen block-35 P3 coordinate;
- block-47 as a native/readout propagation site only;
- the frozen AVeriTeC-compatible population identity;
- outcome-blind partition discipline;
- no newly trained hallucination classifier;
- no layer sweep or P3 reselection.

The existing all-abstain execution remains permanent evidence about the original finite-grammar greedy protocol and must not be overwritten or reinterpreted as a failed precursor test.
