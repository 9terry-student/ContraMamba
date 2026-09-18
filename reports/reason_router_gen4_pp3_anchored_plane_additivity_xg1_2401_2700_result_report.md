# Gen4 Stage 3 — PP3-Anchored Plane Additivity Result

## Scope

This report freezes the descriptive scientific interpretation of the
PP3-anchored plane-additivity screen on XG1 `xg1_fact_2401..2700`.

Execution commit:

`3628a6fe70241324ab0c27e594e162618dead6c5`

Run:

`g4k-pp3-anchored-plane-additivity-xg1-2401-2700-3628a6f-retry3`

Population:

- family: XG1
- pair range: `xg1_fact_2401..2700`
- N: 300
- epsilon: 0.025

The frozen Stage 2 Q0 values were reused. No new baseline forwards were
executed.

Scientific model forwards in this run: 108000.

This screen did not execute a full factorial, matched-control
interactions, primary inference, multiplicity correction, training,
backward passes, or task-head evaluation.

## Provenance

Imported raw artifact SHA256 identities:

- items:
  `bc04b53d4c6d30cc5214bd249e3c5a9fa6f688010b93b9ef65f8b4e9f662c3cb`
- summary:
  `829284a1e28818b2697df579f154e3ad1d9a4e34776b8acc795acc085e61c69e`
- manifest:
  `9034991a426ec5110c656d93e9948827b1b84ee830fe3b1107abd5d72a11a013`
- checksum file:
  `2ec5a5390b115d22e779ff1e71b66ba89d3950748d9d20d75c54c43664130a96`

Handoff ZIP SHA256:

`38f39b515bb8340e3843679ba000267e9d0e5cd0f93d783fd692749664228a18`

The run, collection, and local import all completed with validated
commit, command, and artifact provenance.

## Definitions

For each PP3 partner plane k in P1, P2, P4, P5:

- `A3 = Q0 - Q3`
- `Ak = Q0 - Qk`
- `A3k = Q0 - Q3k`
- `I3k = A3k - A3 - Ak`

`I3k = 0` is exact pairwise additivity for the measured endpoint.
Positive and negative values are retained descriptively; no
post-hoc equivalence margin or interaction threshold is introduced.

## Primary descriptive results

| Partner | mean I3k | mean abs(I3k) | mean abs(I) / mean abs(additive) | dominant sign | joint-vs-additive r |
|---|---:|---:|---:|---:|---:|
| P1 | -4.3801e-09 | 1.3442e-08 | 0.1170 | 64.0% negative | 0.9108 |
| P2 | -9.6825e-09 | 1.3158e-08 | 0.1153 | 81.7% negative | 0.9530 |
| P4 | -3.1069e-09 | 9.3532e-09 | 0.1058 | 58.7% negative | 0.8957 |
| P5 | +4.6219e-09 | 1.2163e-08 | 0.1231 | 67.0% positive | 0.9252 |

The central interaction magnitude is smaller than the additive signal,
but it is not negligible relative to that signal.

The four joint effects remain strongly associated with their additive
predictions, with Pearson correlations from approximately 0.896 to
0.953.

## Tail localization

Interaction magnitude is heavy-tailed.

The largest 10% of absolute interactions account for the following
fractions of total squared interaction magnitude:

- P1: 0.9425
- P2: 0.8798
- P4: 0.9810
- P5: 0.9238

However, the signed 10%-trimmed means remain:

- P1: -4.1082e-09
- P2: -7.0959e-09
- P4: -1.0452e-09
- P5: +4.6717e-09

Therefore the observed partner-specific signed structure is not
explained solely by a small number of extreme items, although the
interaction energy is strongly concentrated in the tails.

## Cross-partner structure

Pairwise correlations among the four interaction vectors range from
approximately 0.466 to 0.648.

Top-30 absolute-interaction overlaps range from 8 to 15 items.

This indicates a nontrivial shared item-level component, but the four
partner interactions are not reducible to a single common interaction
vector.

The per-item four-partner interaction/additive L2 ratio has:

- median: 0.0939
- 75th percentile: 0.1709
- 90th percentile: 0.3349
- 95th percentile: 0.4326

Two items exceed an interaction/additive L2 ratio of 1:

- `xg1_fact_2643`: 1.1774
- `xg1_fact_2683`: 1.0876

These are descriptive tail observations and are not promoted to a new
threshold or exclusion rule.

## Interpretation

The PP3-anchored plane effects exhibit a strong approximate-additive
backbone: joint effects track additive predictions closely across the
population.

The data do not support exact additivity.

They also do not support introducing a retrospective equivalence claim:
interaction magnitudes are structured rather than pure numerical noise,
their signs are partner-dependent, and the interaction distribution is
heavy-tailed.

P2 shows the clearest negative interaction tendency, while P5 shows the
clearest positive interaction tendency. P1 and P4 are weaker in signed
population bias but still show nonzero structured interaction.

The most conservative Stage 3 scientific interpretation is therefore:

`PP3_ANCHORED_PLANE_EFFECTS_HAVE_A_HIGH_FIDELITY_APPROXIMATE_ADDITIVE_BACKBONE_WITH_NONNEGLIGIBLE_PARTNER_SPECIFIC_HEAVY_TAILED_INTERACTIONS_ON_XG1_2401_2700`

This statement is descriptive. No p-value, equivalence margin,
multiplicity correction, or primary inferential decision is attached
to it.

## Boundary and next research direction

Stage 3 is complete at the descriptive level.

The next scientific step should be a limited interaction-localization
follow-up aimed at distinguishing shared item-level susceptibility from
partner-specific interaction mechanisms.

A 32-cell/full-factorial expansion is not justified at this point.
Checkpoint replication is also deferred until the interaction structure
is better localized.

## Repository storage note

The canonical imported raw item artifact is
`pp3_anchored_plane_additivity_items.jsonl`.

Its exact original SHA256 is:

`bc04b53d4c6d30cc5214bd249e3c5a9fa6f688010b93b9ef65f8b4e9f662c3cb`

The original JSONL is larger than GitHub's single-file hard limit, so
the repository stores an exact-byte lossless deterministic gzip archive:

`pp3_anchored_plane_additivity_items.jsonl.gz`

Compressed archive:

- bytes: 5188752
- SHA256: `cc7425d2c2223ff2901029c9a93e75ac01b69fffeeb6e8d6d16bce63d80e9d37`

The gzip archive was round-trip validated by decompression before commit;
the decompressed bytes reproduce the canonical raw-item SHA256 exactly.

The execution-produced `artifact_manifest.json` and `SHA256SUMS.txt`
remain unchanged. They continue to describe and authenticate the
canonical uncompressed execution artifact. The compression is only a
repository transport/storage representation and does not modify the
scientific evidence.
