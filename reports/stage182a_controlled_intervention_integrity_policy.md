# Stage182-A controlled-intervention integrity policy

## Purpose and authority

Stage182-A is a deterministic, read-only construction audit. The original
controlled JSONL is the record of what was evaluated; `scripts/build_controlled_v5.py`
is the authoritative generator schema. Stage176/179 identify the 39 hard rows,
the Stage180 hidden key is authoritative for review-instance identity and the
39 one-to-one hard/control links, and `row_id` is the only cross-artifact item
identity. Repeats never increase item counts.

Stage180 subjective judgments and the Stage181 20/19 split are retained only as
provenance. They are not inputs to an integrity verdict.

## Deterministic checks

For every unique hard and control item, the analyzer checks:

1. dataset identity, required fields, uniqueness, and exact reproduction by the
   generator for the dataset's pair count;
2. packet-to-hidden-key-to-dataset identity and the canonical-`none` claim and
   evidence copied into the Pass 2 packet;
3. exact label/schema values specified by the generator;
4. changed structured axes relative to the same-pair canonical `none` item;
5. whether the changed axes are exactly those allowed for the declared
   intervention;
6. whether a non-polarity intervention also changes surface polarity;
7. whether a generated negative construction uses `did not` followed by the
   generator's already-inflected predicate; and
8. the same construction checks independently for the matched control.

The allowed semantic axes are fixed by generator intervention:

| intervention | allowed changed axes |
|---|---|
| `none` | none |
| `paraphrase` | realization only |
| `entity_swap` | name |
| `event_swap` | object |
| `location_swap` | location |
| `role_swap` | role |
| `title_name_swap` | title and name |
| `predicate_swap` | predicate |
| `polarity_flip` | polarity |
| deletion, truncation, irrelevant evidence | declared content operation |

`time_swap` is recognized by the schema but is outside the frozen clean dataset.
An exact match to buggy generator output establishes provenance, not cleanliness;
therefore grammar and multi-axis checks remain separate from generator equality.

## Item and pair verdicts

- `CLEAN_SINGLE_AXIS_CONSTRUCTION`: exact generator reproduction, valid packet
  anchor, no auxiliary-morphology defect, no polarity leak, and no unintended
  structured-axis change.
- `CONTAMINATED_CONSTRUCTION`: one or more deterministic construction defects.
- `UNAUDITABLE_CONSTRUCTION`: required identity, canonical anchor, or generator
  reconstruction is missing or contradictory. This blocks completion rather
  than being silently treated as contamination.

A hard row enters the Stage182-A clean model-failure candidate set only when the
hard row and its linked control are both `CLEAN_SINGLE_AXIS_CONSTRUCTION`, the
hard row is present in the frozen Stage176 and Stage179 hard-39 artifacts, and
the link is one-to-one. This is a clean *candidate* set: it isolates observed
model failure from detected construction contamination but does not prove a
mechanistic cause or validate the Stage180 Pass 2 taxonomy.

## Output and decision contract

The analyzer writes a JSON report, Markdown report, unique-item audit CSV,
matched-pair audit CSV, clean-model-failure-candidate CSV, and contamination
CSV. All inputs and the generator source are SHA-256 recorded.

Successful exhaustive validation emits
`STAGE182A_CONTROLLED_INTERVENTION_INTEGRITY_AND_CLEAN_FAILURE_SET_AUDIT_COMPLETE`.
A missing/duplicate identity, incompatible topology, packet-anchor mismatch,
unreproducible dataset schema, or non-exhaustive join emits
`STAGE182A_CONTROLLED_INTERVENTION_INTEGRITY_AUDIT_BLOCKED` and no scientific
conclusion.

No verdict authorizes dataset edits, annotation edits, relabeling, filtering,
training-subset construction, model import, checkpoint loading, forward passes,
training, fitting, calibration, threshold search, or external evaluation.

