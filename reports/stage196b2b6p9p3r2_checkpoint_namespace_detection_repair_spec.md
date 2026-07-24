# Stage196-B2-B6P9-P3-R2 Checkpoint Namespace Detection Repair Spec

## Authority

Stage196-B2-B6P9-P3-R2 repairs only the static checkpoint observer namespace
detector used by the P9-P3 runtime analyzer. It does not modify training,
evaluation, checkpoint writing, teacher lifecycle timing, observer metrics,
losses, optimizer behavior, manifests, orchestration, or runtime outputs.

## Raw-Substring False Positive

The previous analyzer implementation concatenated checkpoint pickle bytes and
counted raw occurrences of `teacher_observer_state`. That byte-substring method
conflated the logical top-level checkpoint namespace with unrelated serialized
text atoms, including schema-version strings and metadata/configuration strings
that merely contain the same substring.

As a result, a checkpoint with one logical `teacher_observer_state` dictionary
could be classified as `MULTIPLE` because the substring appeared more than once
inside the serialized pickle payload.

## Grounded Checkpoint Structure

The control checkpoint top-level structure is:

- `schema_version`
- `model_state_dict`
- `metadata`

It has no top-level `teacher_observer_state` key.

Each enabled checkpoint top-level structure is:

- `schema_version`
- `model_state_dict`
- `metadata`
- `teacher_observer_state`

The enabled observer state is one dictionary namespace. Metadata fields such as
`metadata.training_args.teacher_observer_*` are configuration metadata, not
additional serialized observer-state roots.

## Exact Pickle Atom Semantics

The analyzer remains static and must not call `torch.load`, `pickle.load`, load a
model, materialize checkpoint tensors, run a forward pass, train, or evaluate.

For every relevant checkpoint pickle entry (`.pkl` and `data.pkl`), the analyzer
reads that pickle stream independently and iterates it with `pickletools.genops`.
It inspects only string-producing opcodes and counts an atom only when its decoded
value is exactly `teacher_observer_state`.

Supported string opcodes include:

- `UNICODE`
- `BINUNICODE`
- `SHORT_BINUNICODE`
- `BINUNICODE8`
- `STRING`
- `BINSTRING`
- `SHORT_BINSTRING`

Strings that merely contain the substring, such as
`stage196b2b6p9p2_teacher_observer_state_v1`,
`teacher_observer_state_missing`, or `metadata.teacher_observer_state`, do not
count.

## Presence Semantics

The public classifications are preserved:

- missing checkpoint file: `MISSING`, count `null`
- malformed or unsupported static pickle inspection: `UNAVAILABLE`, count `null`
- exact key count `0`: `ABSENT`
- exact key count `1`: `ONE`
- exact key count greater than `1`: `MULTIPLE`

`UNAVAILABLE` is not treated as `ABSENT`.

## Evidence Column

The existing run-completion output adds one evidence column:

`checkpoint_observer_state_exact_key_count`

Expected values on the existing seven completed runs are:

- `control_off_none = 0`
- all six enabled runs `= 1`

The analyzer also includes checkpoint namespace evidence in the analysis JSON and
mentions the static exact-atom detector in the report. No additional analyzer
output file is added; the ten-output closure is preserved.

## Contract Semantics

`control_checkpoint_no_observer_state` means the control checkpoint has exact key
count `0` and presence `ABSENT`.

`enabled_checkpoint_one_observer_state` means every enabled checkpoint has exact
key count `1` and presence `ONE`.

The contracts do not accept `MULTIPLE` and do not infer namespace count from
checkpoint size or tensor-storage counts.

## Rerun Requirements

No training rerun is required. The existing seven completed run artifacts can be
reanalyzed statically with the repaired analyzer. The expected downstream result
is that the control checkpoint is classified as `ABSENT` with exact count `0`,
each enabled checkpoint is classified as `ONE` with exact count `1`, and the
runtime decision follows from the repaired evidence rather than being forced.
