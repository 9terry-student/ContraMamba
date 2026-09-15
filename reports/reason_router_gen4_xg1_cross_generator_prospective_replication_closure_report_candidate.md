# Gen4 × K XG1 Cross-Generator Prospective Replication Closure Report Candidate

## Status

**CLOSURE REPORT CANDIDATE — NO NEW SCIENTIFIC EXECUTION AUTHORITY**

This report closes the frozen XG1 cross-generator prospective replication study using only already-frozen design, execution, and imported artifacts.

It does not authorize threshold modification or re-estimation, XG1 population repair or enlargement, additional XG1 intervention responses, additional confirmatory or exploratory hypothesis tests, renderer redesign, a real-world external bridge, or a new generator-family study.

Any later scientific stage requires separate authority.

## Frozen authority and evidence

Prospective design:

- `reports/reason_router_gen4_xg1_cross_generator_prospective_replication_design.md`
- design freeze commit: `a31d2bc5ab4b939f52e969c89f3783feb9c3b233`

Structural XG1 freeze:

- structural freeze commit: `d9029801fd47636c155b1c846c433fc561424c8f`

Tokenizer / anchor eligibility freeze:

- eligibility freeze commit: `ee7c2c10a0cbb4930b78eb0047ec0603b63e0d41`

Bounded CPU-slow ↔ CUDA-fast equivalence freeze:

- gate freeze commit: `f41aa2abec3fa372699913c387082131b0447051`
- gate execution head: `6afb6191d1ef8c0aa39d4c5497a9250fda192110`
- frozen equivalence artifact SHA256:
  `865124fd1804c2198d58ae01f3846319770fb3eda6482363b4036ba35f721877`

Full XG1 scientific execution implementation:

- execution head:
  `e55148401264b046d6f2effd9a0e3857c29b1dad`

Imported full-run artifact freeze:

- result freeze commit:
  `189f3e6a060859b05d78898b6129844111378aa2`
- artifact directory:
  `reports/reason_router_gen4_xg1_fast_cuda_full_e551484_r6/`
- summary SHA256:
  `979c6aba213f9445d466c60bb888374c88484c17b65dce865f8595660f22e523`
- regime freeze SHA256:
  `4fb6993c10a7465e5ed63ba8f57b497f890606789f0550198dafd4e3e121db2d`
- regime manifest SHA256:
  `3fab70afe5c7fc719eb36b6d8e30c90e107b1349eef2c4fb680b5bb27b7d74e4`

## Frozen prospective rule

The carried-forward discriminator was:

`alignment_shift_abs = abs(reference_C - target_C)`

The carried-forward threshold was:

`T = 0.11228626366380845`

Classification was frozen as:

- LARGE iff `alignment_shift_abs >= T`
- SMALL iff `alignment_shift_abs < T`

The prospective design required, after all 300 baseline regimes had been frozen and before any alignment response was observed:

- `n_LARGE >= 30`
- `n_SMALL >= 30`

Otherwise the study was required to stop with:

`XG1_PROSPECTIVE_REGIME_TEST = BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE`

The design explicitly prohibited changing the threshold or pair population to repair group size.

## Execution validity

The successful XG1 run was:

`gen4-k-xg1-full-e551484-r6`

The imported run provenance established:

- exact execution head:
  `e55148401264b046d6f2effd9a0e3857c29b1dad`
- command SHA256:
  `9d45ec065ba522f78e1345d1bcef1e3e893cdf493ffec8e47dae24a372fcd94b`
- exit code: `0`
- imported files validated: `5`
- training executed: `false`
- backward executed: `false`
- task heads executed: `false`
- logits read: `false`
- response-dependent exclusion: `false`
- threshold re-estimated: `false`

The successful execution completed the baseline regime-freeze phase and passed the artifact checksum and summary-contract validations.

## Frozen XG1 result

The baseline-only scientific phase executed exactly:

`1200` model forwards.

The complete 300-pair regime freeze yielded:

- `n_LARGE = 19`
- `n_SMALL = 281`
- minimum required group size: `30`
- group-size gate pass: `false`

At the regime freeze:

- alignment model forward count: `0`
- response fields observed at freeze: `false`

The frozen result is:

`PASS_XG1_BASELINE_PHASE_BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE`

and:

`XG1_PROSPECTIVE_REGIME_TEST = BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE`

The scientific conclusion field is intentionally:

`null`

## Scientific interpretation

This XG1 study did not reach the confirmatory intervention-response phase.

The fixed threshold transferred to the independently constructed XG1 generator family in the sense that all 300 pairs could be classified prospectively, but only 19 of 300 pairs entered the pre-specified LARGE regime. This was below the frozen minimum of 30 required for the confirmatory LARGE-vs-SMALL response analysis.

Therefore:

1. the XG1 cross-generator adverse-regime replication claim was not tested to completion;
2. H1-XG1 was not tested;
3. H2-XG1 was not tested;
4. no Holm family was evaluated;
5. the 600 alignment-intervention forwards were not executed;
6. no `R_ALIGN` response was used to repair, redefine, or enrich the LARGE group.

The valid scientific conclusion is therefore not that the adverse response effect failed to replicate.

The valid conclusion is:

> Under the frozen original threshold, the independently generated XG1 population contained too few LARGE-regime pairs to satisfy the prospectively fixed minimum group-size requirement, so the pre-specified cross-generator intervention-response replication test was blocked before any XG1 alignment response was observed.

This is informative about transport of the original discriminator's regime prevalence across generator families, but it does not by itself establish the cause of the prevalence shift and does not establish the intervention response inside the XG1 LARGE regime.

## What this result does not establish

This result does not establish:

- absence of an adverse alignment-intervention response in XG1;
- successful cross-generator replication;
- failed cross-generator replication in the confirmatory H1/H2 sense;
- behavioral mediation;
- hallucination causation;
- general Mamba instability;
- real-world external validity;
- validity outside the matched slot-manipulation setting;
- a new optimal threshold;
- that the threshold should be lowered;
- that the XG1 population should be enlarged or repaired.

`CROSS_GENERATOR_ADVERSE_REGIME_NOT_ESTABLISHED` is also not the appropriate frozen outcome string here, because that decision path requires the fixed group-size gate to pass and the confirmatory response tests to execute.

## XG1 closure decision

The XG1 cross-generator prospective replication study is closed with a valid, provenance-checked, prospectively specified group-size block.

No rescue analysis is authorized inside XG1.

In particular, the following are outside the closed study:

- threshold scanning;
- threshold re-estimation;
- pair replacement;
- selective addition of LARGE-like pairs;
- larger XG1 reruns intended to force `n_LARGE >= 30`;
- alternate renderer comparison;
- post hoc subgroup testing;
- intervention-response execution on the 19 observed LARGE pairs.

The completed same-generator-family prospective result remains frozen and unchanged. XG1 does not overturn that result.

## Next-stage boundary

The frozen XG1 design authorizes a real-world external bridge only after a positive completed XG1 replication. That condition was not met.

Therefore a real-world external bridge is not authorized by this result.

The scientifically motivated next question is instead whether the prevalence and distribution of the already-frozen discriminator are stable across independently constructed generator families. That would be a distinct generator-family prevalence-transportability study, not an XG1 rescue.

Such a study would require a new prospective design and separate execution authority before any new generator construction, tokenizer/model execution, or statistical analysis begins.

## Closure

**XG1 STATUS: CLOSED / FROZEN**

Frozen closure statement:

> The independently generated XG1 population passed structural, tokenizer/anchor, backend-equivalence, runtime, and provenance gates, but the frozen pre-intervention discriminator produced only 19 LARGE pairs out of 300. Because the prospectively fixed minimum LARGE group size was 30, the study stopped after 1200 baseline forwards and before any alignment-intervention response. The XG1 confirmatory cross-generator replication test therefore remained unexecuted, with no threshold or population rescue permitted.
