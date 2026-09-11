# K3T Phase-Balanced R Transportability Closure Report Candidate

**Stage:** `K3T — Phase-Balanced R Transportability`

**Status:** validated scientific closure candidate.

K3T scientific execution completed exactly once under the frozen one-run authority.

Independent artifact/provenance/statistical reconstruction passed exactly.

The final scientific verdict is:

`R_PHASE_BALANCED_TRANSPORTABILITY_NOT_ESTABLISHED`

K3T does not support the preregistered positive R phase-balanced transportability signal.

K3T does not significantly contradict that signal in the opposite direction.

The broader scientific state remains:

`R_CROSS_CLAIM_TRANSPORTABILITY_NOT_ESTABLISHED`

## 1. Frozen authority and implementation

Preregistration commit:

`58cbcd316c7714ddc8c041c2a2ec4376e79a4bd0`

Preregistration SHA256:

`57b5bd2375fbacb7ef5e22260f3f1ecf71dba839f383f7726c03544e33a45cc5`

Implementation commit:

`1a2ed06ad22ec7c293acd827d11646c92390cf0d`

Runner SHA256:

`acb9390ac8fafc64edcac07562e84c4d9de99df844b41ad8e82652640e7527a3`

Test SHA256:

`144870ce18ae6e57d12a179a88bab853b7a1d1d88ae658d818786ee0a7f85d88`

One-run execution-authority commit:

`8c96b476da283195060b0827bc901a03a239f169`

Authority SHA256:

`e3b73da89a8b471bfbce551e3bf33ae38c9b8363d93102386f7ded2015b2dc5b`

The one-run execution authority is consumed.

No K3T rerun is authorized.

## 2. Scientific execution

Scientific run directory:

`C:\Users\Home1\Desktop\ContraMamba-K3T-Runs\k3t-r-transportability-8c96b476da28`

Scientific runtime:

- device = CPU;
- recurrence = `MambaMixer.slow_forward`;
- Transformers = `5.12.1`;
- native state timing = `post_consumption_s_t`;
- primary layer = 23;
- post-prefix window = `W=8`.

Scientific population:

- global template slice = `[900:1236]`;
- exact pair IDs = `generated_fact_901` through `generated_fact_1236`;
- item count = 336;
- reciprocal blocks = 168;
- nonnumeric lexical generator phase period = 168;
- exact occurrences per phase = 2;
- correction-source counts = `{"none":168,"polarity_flip":168}`;
- every phase contains exactly one `none` item and one `polarity_flip` item.

The population remained claim-disjoint from K2W, K2R/K3, and K3C on pair ID, exact claim SHA, and exact claim text.

## 3. Frozen artifact identities

Archived scientific artifacts have the following exact SHA256 identities.

`generated_source.jsonl`

`fb699bcc99e00b8c437fd49c204345933d61c69215af853746bc4639545c7400`

`candidate_pool.jsonl`

`d95d245e358ff497ea50b95e4f54d1192be64d09fec2c06538fc15f75b09ef70`

`reciprocal_mapping.json`

physical SHA256:

`6f05c7acc406cd73fcfc38d81e24be10482200467f95b7616d59c34b24aa2312`

canonical JSON SHA256 excluding the final LF:

`1c458054540ac0d38857cb3f55595e0b286428225ef11ed22059981dfd1e28ad`

`item_metrics.jsonl`

`e01b6b0cb22acf791188affb732fc13378c59f60a10a7b9eb07396766414c23b`

`block_metrics.jsonl`

`164eca4d99f88586209b22970d29f628adcdaa98b83fc74ae5beb22175464aad`

`primary_stats.json`

`4a314bc00dd68f2e4b70c06a8bdbf0b5d79879bd2b16a4ebeb4a523c5e0e70a2`

`integrity.json`

`e6cd582a78c9f6a369b270a96f862ef54809b86518f5fbd9c82c7bc90e3ed50d`

`report.md`

`2241422efa799c25dbb43fb38e5ac17d3697641cbf8da4279f72efc567a85c4a`

`manifest.json`

`0c312bb864cdf18f3da537a3e408a7b889e3a8fce33fde00661f79a4c8879770`

`SHA256SUMS.txt`

`b21f134523a430574bec1a1ad41ce7aaf9587ec38b69a98859d6fb71781b87a7`

## 4. Independent artifact validation

Independent validation was read-only and did not import the scientific runner.

Independent validator SHA256:

`a36d75d1016738da767e05af005faed7f735a05b82a0ed1d1b0fb514d96a0d74`

It established:

`EXECUTION_SUCCESS=YES`

`ARTIFACT_PROVENANCE_VALID=YES`

`SCIENTIFIC_INTEGRITY=PASS_EXACT`

The validator independently verified:

- all 10 physical artifact SHA256 values;
- the `SHA256SUMS.txt` file by independent reconstruction;
- authority/preregistration/implementation Git topology;
- exact frozen file identities;
- candidate population size and pair-ID set;
- exact two-occurrence phase balance;
- exact per-phase correction-source balance;
- reciprocal pairing;
- candidate-to-item alignment;
- item-to-block alignment;
- matched and swapped d-p distributions;
- common-prefix state-hash evidence;
- manifest provenance bindings;
- scientific runtime and encoder identities;
- primary statistics.

## 5. Independent R reconstruction

The independent validator did not trust `primary_stats.json` as the source of the final inference.

For every scientific item it reconstructed:

`Delta_R_matched = R(matched_corr) - R(matched_ctrl)`

`Delta_R_swapped = R(swapped_corr) - R(swapped_ctrl)`

and then:

`X_R = |Delta_R_matched| - |Delta_R_swapped|`

from the serialized branch-level `R_mean_speed` values.

For each reciprocal block:

`B_R = (X_R(a) + X_R(b)) / 2`

was reconstructed independently.

All 168 archived B_R values matched the independent reconstruction exactly within the frozen floating-point comparison contract.

## 6. Final primary statistics

Confirmatory endpoint:

`R`

Inferential unit:

reciprocal block.

Valid blocks:

`168`

Positive blocks:

`81`

Negative blocks:

`72`

Zero blocks:

`15`

Effective nonzero blocks:

`153`

Promotion floor:

PASS.

Two-sided exact sign-test p-value:

`0.517921030799045`

Family size:

`m=1`

Adjusted p-value:

`0.517921030799045`

Rank-biserial sign effect:

`+0.058823529411764705`

The positive direction was weak and not statistically established.

There was no statistically significant opposite-direction contradiction.

Final K3T verdict:

`R_PHASE_BALANCED_TRANSPORTABILITY_NOT_ESTABLISHED`

## 7. Relation to K2R and K3C

The relevant prospective sequence is now:

K2R:

positive R replication on its claim-disjoint 300-item population.

K3C BASE:

R not established on a second claim-disjoint 300-item population.

K3T:

R not established on a third claim-disjoint population specifically constructed to balance the deterministic 168-class nonnumeric lexical generator phase exactly.

Therefore the earlier K3C R failure is not rescued by exact prospective balancing of that 168-class generator phase.

The K3T result strengthens the conclusion that simple deterministic lexical-phase composition is not sufficient to explain the K2R-to-K3C R instability.

It does not prove that lexical or token realization is irrelevant.

It does not identify a unique cause of the instability.

It does establish that positive R pair-specific speed geometry has not demonstrated stable cross-claim transportability across the currently tested controlled-generator populations.

## 8. Scientific interpretation boundary

The strongest supported current statement is:

`R_CROSS_CLAIM_TRANSPORTABILITY_NOT_ESTABLISHED`

Do not replace this with:

`R_CROSS_CLAIM_TRANSPORTABILITY_CONTRADICTED`

because K3T did not produce significant negative-direction evidence.

Do not replace this with:

`R_CROSS_CLAIM_TRANSPORTABILITY_ESTABLISHED`

because K3C and K3T did not replicate the positive R effect.

Do not reinterpret K3C as a successful three-of-four result.

D, DISP, and P were not K3T confirmatory endpoints.

K3T therefore does not update their inferential status.

## 9. Mechanistic consequence

K3T was observational and R-only.

It does not establish a W/G/H recurrence mechanism.

The frozen mechanism states remain:

- K3 coefficient-specialization hypothesis: contradicted;
- K3C write-injection / retained-carry successor: inconclusive due to BASE replication failure;
- no replacement causal mechanism established.

K3T does not reopen K3 or K3C mechanism promotion.

## 10. Rerun and rescue boundary

`K3T_EXECUTION_AUTHORITY_CONSUMED=YES`

`K3T_RERUN_AUTHORIZED=NO`

No alternate:

- population slice;
- generator phase subset;
- correction-source subset;
- prefix-length subset;
- d-p subset;
- layer;
- window;
- R transformation;
- one-sided test;
- block filter

may be used to rescue K3T.

Any future R study must be a separately motivated and prospectively preregistered successor.

## 11. K4 boundary

K3T is not K4 evidence.

K3T does not establish decision-space linkage, authorization causality, or final-decision mechanism.

Therefore:

`K4_EXECUTION_AUTHORIZED=NO`

## 12. Closure state

`K3T_VALID_EXECUTION=YES`

`K3T_ARTIFACT_PROVENANCE_VALID=YES`

`K3T_SCIENTIFIC_INTEGRITY=PASS_EXACT`

`K3T_PHASE_BALANCED_R_REPLICATED=NO`

`K3T_PHASE_BALANCED_R_CONTRADICTED=NO`

`K3T_SCIENTIFIC_VERDICT=R_PHASE_BALANCED_TRANSPORTABILITY_NOT_ESTABLISHED`

`R_CROSS_CLAIM_TRANSPORTABILITY_ESTABLISHED=NO`

`R_CROSS_CLAIM_TRANSPORTABILITY_NOT_ESTABLISHED=YES`

`K3T_EXECUTION_AUTHORITY_CONSUMED=YES`

`K3T_RERUN_AUTHORIZED=NO`

`K3C_RERUN_AUTHORIZED=NO`

`K4_EXECUTION_AUTHORIZED=NO`

K3T is closed after this report and the exact scientific artifact archive are committed.

No successor scientific execution is authorized by this closure.
