# O0b Token-Aligned Native Hidden-State Observer Implementation Authority Specification Candidate

Status: **PASS-READY CANDIDATE / STATIC IMPLEMENTATION-AUTHORITY SPECIFICATION ONLY**

This file freezes a future O0b observer implementation contract. It does not authorize tokenizer execution, model loading, model weights, hidden-state forwards, training, evaluation, Kaggle, checkpoint mutation, commit, or push. Independent verification and a later authority freeze are required first.

## 1. Authority and scientific boundary

Authority precedence is: current controller instruction; O0b scientific-design authority `df461469cb087f7f5db1e41a2b08e65ea517ad8a`; O0b full-sequence offset boundary-recovery authority `2ed4439e511f7534186cbd5df9110e45fdc1d66c`; repaired matched-control implementation freeze `7ce4e0cd05d87118c29526a53ab5178dc722db27`; O0a artifacts as technical precedent only; and repository `AGENTS.md` except where superseded. URP/reason-router work is unrelated and protected.

Exact question: **After controlling full serialized token count, claim bytes, terminal position, surface wording controls, and pair-relative divergence coordinates, do frozen native Mamba hidden-state proxies retain an early response specifically associated with insufficient evidence?**

Measure Hugging Face `MambaModel` layer hidden states, called **native pretrained Mamba hidden-state proxies**. They are explicitly not selective SSM recurrent state, `cache_params` state, selective-scan matrices, direct A/B/C/Delta dynamics, generation, an emitted unsupported commitment, hallucination detection, a classifier, a learned probe, or an architecture modification. Preserve: **measure first; explain second; modify third**.

## 2. Exact frozen input bindings

| Item | Exact binding |
|---|---|
| Dataset | `data/longterm_o0b_matched_controls_v1.jsonl` |
| Dataset SHA256 | `75a675bee49cb26eb0935d364f0f5d090922dd01576dfc23294961b28394aec2` |
| Validation artifact | `reports/longterm_o0b_matched_controls_v1_validation.json` |
| Validation artifact SHA256 | `e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff` |
| Validator | `scripts/validate_longterm_o0b_matched_controls.py` |
| Validator SHA256 | `e4b488c8f7a16a7004b27f0bd47e712785b9f9f9fe40def81cd9836e7d25ff67` |
| Validation test | `tests/test_validate_longterm_o0b_matched_controls.py` |
| Validation-test SHA256 | `558f1f718d9c0024d18b46a5da91cf89a6a98a55b91e840615782243c13205e0` |

The repaired input implementation freeze commit (`7ce4e0cd05d87118c29526a53ab5178dc722db27`) is distinct from the validation artifact production `repository_head` (`2ed4439e511f7534186cbd5df9110e45fdc1d66c`). Never regenerate the canonical validation artifact at observer execution.

## 3. Required artifact checks before model loading

Parse the frozen validation artifact and require exactly:

~~~text
scientific_design_authority_commit = df461469cb087f7f5db1e41a2b08e65ea517ad8a
implementation_authority_commit = 31e6d7882586e312f783cb2fd69718eb1ee7e452
boundary_recovery_authority_commit = 2ed4439e511f7534186cbd5df9110e45fdc1d66c
repository_head = 2ed4439e511f7534186cbd5df9110e45fdc1d66c
tokenized_text_coordinate_domain = printable_ascii_u0020_u007e
overall = PASS
~~~

Require the exact dataset path/SHA and exactly pair IDs `o0b_pair_001`, `o0b_pair_002`, `o0b_pair_003`, each with exactly `reference_sufficient`, `paraphrase_sufficient`, `insufficient_matched`, and `surface_null_matched`. Any discrepancy fails closed before model loading.

## 4. Pre-model-load ordering and token revalidation

The exact order is: (1) read dataset bytes; (2) verify dataset SHA; (3) parse; (4) verify the exact three-pair/four-condition structure; (5) apply the printable-ASCII/nonempty/strip-stable source guard; (6) read validation-artifact bytes; (7) verify its SHA; (8) validate provenance; (9) load exact tokenizer only; (10) tokenize every complete serialized sequence; (11) require exact artifact agreement; (12) only then load `MambaModel`.

Serialization is exactly `Claim: <claim>\nEvidence: <evidence>`. Tokenizer: ID and revision `state-spaces/mamba-130m-hf` and `5708daa364c50b880e7bd92eab456e0d34492ee9`; `use_fast=True`; `trust_remote_code=False`; `tokenizer.is_fast is True`; `add_special_tokens=False`; `return_offsets_mapping=True`. Do not separately tokenize the scaffold or regenerate/update the canonical artifact.

For all 12 members require exact full token IDs, count, offsets, `evidence_char_start`, `evidence_start_index`, evidence-start offset start/end, `boundary_crossing`, and terminal index. For every comparison require exact `first_divergent_token_index` and all six anchors.

## 5. Frozen pair-relative coordinates

The observer must consume and reconfirm these from the artifact, never choose new anchors:

| Pair | Count | Evidence start | Terminal | Comparison | Divergence | Anchors: pre-1, divergence, post+1, post+2, post+4, terminal |
|---|---:|---:|---:|---|---:|---|
| o0b_pair_001 | 45 | 17 | 44 | reference/paraphrase | 18 | 17,18,19,20,22,44 |
| o0b_pair_001 | 45 | 17 | 44 | reference/insufficient | 25 | 24,25,26,27,29,44 |
| o0b_pair_001 | 45 | 17 | 44 | reference/surface-null | 17 | 16,17,18,19,21,44 |
| o0b_pair_002 | 36 | 14 | 35 | reference/paraphrase | 14 | 13,14,15,16,18,35 |
| o0b_pair_002 | 36 | 14 | 35 | reference/insufficient | 24 | 23,24,25,26,28,35 |
| o0b_pair_002 | 36 | 14 | 35 | reference/surface-null | 14 | 13,14,15,16,18,35 |
| o0b_pair_003 | 36 | 16 | 35 | reference/paraphrase | 21 | 20,21,22,23,25,35 |
| o0b_pair_003 | 36 | 16 | 35 | reference/insufficient | 16 | 15,16,17,18,20,35 |
| o0b_pair_003 | 36 | 16 | 35 | reference/surface-null | 16 | 15,16,17,18,20,35 |

Each comparison owns its pair schedule. Reference vectors may be reused after one forward.

## 6. Runtime, full-sequence forward, layer exposure

Freeze model/tokenizer ID and revision above, `device=cpu`, `dtype=float32`. Use `model.eval()`, frozen parameters, and `torch.inference_mode()`. Forward each complete member exactly once, one sequence per forward, no padding, with `output_hidden_states=True`, `return_dict=True`, `use_cache=False`. No prefix forwards, truncation, generation, CUDA, float16, bfloat16, optimizer, backward, compile, train, parameter mutation, checkpoint writing, cache inspection, `cache_params`, ContraMamba, or reason-router components.

Require nonempty `outputs.hidden_states` and `outputs.last_hidden_state`. Preserve hidden-state order; append last state only when it is not shape/value identical to the final hidden state. Require one descriptor layout for all 12 forwards. Index 0 role is `embedding_or_initial_hidden_state`; final role is `output_hidden_state`; intervening roles are `intermediate_hidden_state`. These are native proxies, not recurrent state. Extract only required anchor vectors and provenance metadata. Position `i` is the ordinary full-sequence causal output after consuming through `i`.

## 7. Pre-divergence invariant

For every pair, comparison, and layer at `anchor_pre_minus_1`, input IDs through that position must match and the vectors must satisfy:

~~~python
np.allclose(reference, comparison, rtol=0.0, atol=1e-6)
~~~

Violation is execution/implementation invalidity, not signal: fail closed, do not average away, and do not interpret. Nonzero `rtol` is forbidden.

## 8. Metrics and descriptive summaries

Require finite vectors and norm greater than zero. Normalize deterministically using float64 accumulation from stored float32 values:

~~~text
unit(h) = h / sqrt(sum(h_j*h_j))
D_l2 = sqrt(sum((unit64(member)-unit64(reference))**2))
D_cos = 1 - sum(unit64(member)*unit64(reference))
~~~

Require finite metrics and `D_l2**2 = 2 * D_cos` within `1e-12`. These coordinates are algebraically redundant and not independent evidence.

Record A = insufficient vs reference, B = paraphrase vs reference, and C = surface-null vs reference for every pair/layer/frozen anchor; primary early anchors are divergence, post+1, post+2, post+4, plus terminal and pre-1. No learned aggregate, weighted score, threshold, best-layer/anchor score, or trajectory-summary score.

Summaries may expose every raw A/B/C value and `A_gt_B = A > B`, `A_gt_C = A > C`. Allowed aggregates: predeclared mean/median by layer and anchor and counts of pair IDs with A>B or A>C. No significance test, confidence interval, population inference, hard PASS threshold, or favorable layer/anchor selection.

## 9. Required future artifacts

~~~text
manifest.json
anchor_observations.jsonl
anchor_hidden_states.npz
paired_distances.jsonl
summary.json
report.md
SHA256SUMS.txt
~~~

Manifest records exact provenance/input/runtime/model/tokenizer/layer/schema identities and no timestamp, UUID, hostname, or absolute local path in canonical scientific identity. Observations contain per pair/condition/layer/anchor metadata and NPZ indices. NPZ contains float32 vectors. Paired distances contain one row per pair/comparison/layer/anchor, L2, cosine, pre-divergence flag, and anchor provenance. Summary/report are deterministic descriptive renderings. Checksums cover every other file exactly in deterministic order.

## 10. Deterministic NPZ strategy

The inspected O0a precedent uses `numpy.savez_compressed`, and its tests do not enforce byte identity. It is therefore insufficient.

Freeze strategy B: the future observer must implement an explicit deterministic ZIP/NPY writer. Generate each `.npy` with fixed NumPy format/version, little-endian dtype, C order, and canonical shape/content. Use a frozen fixed member list in lexicographic order, including `schema_version.npy`, `anchor_hidden_states.npy`, and fixed manifest metadata arrays. Use `ZIP_STORED`, DOS timestamp `1980-01-01 00:00:00`, fixed permissions/external attributes, fixed UTF-8 policy, no comments/extra fields, and no duplicate names. Verify exact member set and byte-identical reproduction from identical payloads. Future tests must write twice and compare bytes while checking order, timestamps, dtype, and metadata. If APIs cannot meet this, fail closed; never fall back to `numpy.savez` or `numpy.savez_compressed`.

## 11. Provenance identity separation

- **Input implementation identity:** repaired freeze `7ce4e0cd05d87118c29526a53ab5178dc722db27`; artifact `implementation_authority_commit` remains `31e6d7882586e312f783cb2fd69718eb1ee7e452`.
- **Observer authority identity:** future full SHA freezing this document; this candidate must not predict it.
- **Observer implementation identity:** separate future full SHA freezing exactly the two implementation files and exact observer-script SHA256.
- **Scientific execution identity:** later authority's exact observer commit/SHA, model/runtime, command, run name, output directory, collection/import provenance, and artifact set.

A later authority binds separately the O0b design, boundary, repaired-input, observer-authority, and observer-implementation commits, observer SHA, input SHAs, immutable revisions, package versions, CPU/float32, command, run name, and required artifacts.

## 12. Future implementation whitelist

After independent verification and freeze, authorize exactly:

1. `scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`
2. `tests/test_observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`

No dataset, validator, validation artifact, O0a material, O0b authority, historical root patch, protected temporary directory, or URP/reason-router file may be edited. Tests may use fake/synthetic model/tokenizer objects. A bounded exact-revision tokenizer-only test may be separately authorized; model-weight loading is not.

## 13. Future safety-test matrix

Tests must cover: wrong dataset/artifact SHA; every wrong provenance commit and production head; wrong pair/condition set; zero model-loader calls on pre-load failure; invalid ASCII source; tokenizer ID/revision, slow tokenizer, special-token, remote-code, token-ID, offset, boundary, divergence, and anchor drift; non-CPU/non-float32; eval/frozen model; optimizer/backward/generate/train/compile and prohibited imports via AST/runtime guards; missing/empty hidden states, missing last state, inconsistent layout, malformed shape, nonfinite vector; exact pre-divergence pass/fail and rtol; zero norm/nonfinite metrics, known L2, cosine identity, wrong pairing; canonical JSON/JSONL; deterministic NPZ bytes; exact artifact set, checksum completeness, and output collision.

## 14. Future validation and execution boundary

Later implementation validation may run only:

~~~text
python -m pytest -q tests/test_observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py
python -m py_compile scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py tests/test_observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py
git diff --check -- scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py tests/test_observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py
~~~

These must not load weights. Even after implementation PASS/freeze, **no scientific run is authorized**. A separate O0b execution authority must freeze implementation commit/SHA, runtime/model, command, output directory, run name, collection/import provenance, and interpretation boundary. No Kaggle command is authorized.

## 15. Static validation performed for this candidate

Only these commands are authorized for this task:

~~~text
git diff --check -- reports/longterm_o0b_token_aligned_native_hidden_state_observer_implementation_authority_spec_candidate.md
git status --short
~~~

Required explicit result boundary:

~~~text
NO EXISTING FILE MODIFIED
NO TOKENIZER EXECUTION
NO MODEL LOADING
NO MODEL WEIGHTS
NO HIDDEN-STATE FORWARD
NO TRAINING
NO EVALUATION
NO KAGGLE
NO COMMIT
NO PUSH
~~~

Final verdict: **PASS_READY_FOR_INDEPENDENT_OBSERVER_IMPLEMENTATION_AUTHORITY_VERIFICATION**

## 16. Precision-repair authority (superseding exactness clauses)

This section is normative and supersedes any less-specific or conflicting
serialization, coordinate, provenance, metric, NPZ, checksum, or publication
wording above. It does not change the scientific question, frozen inputs,
coordinates, three-pair/four-condition design, full-sequence single-forward
semantics, pre-divergence tolerance, protected state, no-model-execution
boundary, or two-file future implementation whitelist.

### 16.1 Model-loader security

The authorized future loader is semantically equivalent to:

~~~python
MambaModel.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=torch.float32,
    trust_remote_code=False,
)
~~~

`MODEL_REVISION` is an exact immutable revision. Loading is CPU-only and
float32-only. No fallback may omit or override `trust_remote_code=False`.
Future tests must inspect the actual authorized loader kwargs and fail when
`trust_remote_code` is true or unspecified, when the immutable revision is
omitted or incorrect, or when device/dtype is not CPU/float32.

### 16.2 Comparison and anchor ownership

The exact comparison and relative-anchor orders are:

~~~python
COMPARISON_ORDER = (
    "insufficient_matched",
    "paraphrase_sufficient",
    "surface_null_matched",
)
ANCHOR_ORDER = (
    "anchor_pre_minus_1",
    "anchor_divergence",
    "anchor_post_plus_1",
    "anchor_post_plus_2",
    "anchor_post_plus_4",
    "anchor_terminal",
)
~~~

The comparisons mean A = `insufficient_matched`, B =
`paraphrase_sufficient`, and C = `surface_null_matched`. Every logical
anchor measurement has the complete identity
`(pair_id, comparison_id, anchor_name, absolute_token_index, layer_index)`.
Vector provenance additionally contains `vector_role` (`reference` or
`member`) and condition identity.

The exact physical hidden-vector reuse key is
`(pair_id, condition, absolute_token_index, layer_index)`. Reuse is permitted
only when the complete key is identical; matching anchor names, layer indices,
or pair membership alone never permits reuse. For `o0b_pair_001`, the
comparison-owned divergence coordinates are B = 18, A = 25, and C = 17.
Therefore `anchor_divergence` is not one shared absolute coordinate. At every
relative anchor, A, B, and C summaries use their own artifact-owned absolute
anchor: this is divergence-aligned, not same-absolute-position, comparison.

### 16.3 Layer provenance

Every vector and distance record persists `layer_index`, `layer_role`, and
`state_source`. Layer 0 is `embedding_or_initial_hidden_state`; the final
exposed state is `output_hidden_state`; every other state is
`intermediate_hidden_state`. `state_source` is the exact runtime descriptor:
`hidden_states[0]`, `hidden_states[1]`, ..., or `last_hidden_state`.
Layer descriptors must be semantically byte-identical across all 12 forwards.

### 16.4 Exact metric paths

For every source 1-D hidden-state vector, execute exactly:

~~~python
member64 = np.asarray(member_vector, dtype=np.float64)
reference64 = np.asarray(reference_vector, dtype=np.float64)
~~~

Require both vectors to be rank 1, nonempty, and finite. Compute
`member_norm = float(np.linalg.norm(member64))` and
`reference_norm = float(np.linalg.norm(reference64))`; both must be finite and
strictly positive. Zero norm fails closed. Add no epsilon and perform no
clipping. Then compute the float64 arrays
`member_unit = member64 / member_norm` and
`reference_unit = reference64 / reference_norm`, followed by
`delta64 = member_unit - reference_unit`,
`d_l2_64 = np.linalg.norm(delta64)`, and
`normalized_l2_distance = float(d_l2_64)`. `d_l2_64` must be finite; no
float32 accumulation is allowed.

Using those same unit arrays, compute
`dot64 = np.dot(member_unit, reference_unit)`,
`d_cos64 = 1.0 - dot64`, and
`cosine_distance = float(d_cos64)`. It must be finite. Compute exactly
`redundancy_error = abs(normalized_l2_distance * normalized_l2_distance -
2.0 * cosine_distance)` and require `redundancy_error <= 1e-12` with no
relative tolerance. Emit `cosine_redundancy_error = float(redundancy_error)`.
Cosine is audit-only and is not independent evidence.

### 16.5 Exact anchor-observation records

There is one JSONL record per logical vector use, with exactly this key set and
no additional keys:

~~~text
schema_version, pair_id, comparison_id, reference_condition,
member_condition, vector_role, condition, anchor_name,
absolute_token_index, layer_index, layer_role, state_source, vector_index
~~~

`reference_condition` is exactly `reference_sufficient`; member condition and
comparison ID are from `COMPARISON_ORDER`; `condition` agrees with
`vector_role`; and `vector_index` addresses one NPZ matrix row. Logical rows
are ordered by pair IDs 001, 002, 003, then `COMPARISON_ORDER`, then
`ANCHOR_ORDER`, then ascending layer, then vector role reference/member.
Unavailable anchors emit no record.

### 16.6 Exact vector matrix and paired-distance records

Walk observation rows in that order. On first occurrence of a physical key,
assign the next zero-based `vector_index`; later identical keys reuse it and no
other reuse is allowed. NPZ payload is exactly one array named `vectors`, with
shape `(number_of_unique_physical_vectors, hidden_size)`, rank 2, nonempty first
dimension, constant hidden size, little-endian float32 dtype, and C-contiguous
layout. No metadata array is stored in the NPZ.

There is one paired-distance record per pair, comparison, available relative
anchor, and layer, with exactly this key set:

~~~text
schema_version, pair_id, comparison_id, reference_condition, member_condition,
anchor_name, absolute_token_index, reference_absolute_token_index,
member_absolute_token_index, layer_index, layer_role, state_source,
reference_vector_index, member_vector_index, normalized_l2_distance,
cosine_distance, cosine_redundancy_error, pre_divergence_integrity_status
~~~

All three absolute-token fields are equal for each comparison-owned anchor.
`pre_divergence_integrity_status` is `PASS` at `anchor_pre_minus_1` and
`NOT_APPLICABLE` everywhere else. A failed pre-divergence record is never a
valid artifact. Distance rows use pair, comparison, anchor, layer order.

### 16.7 Null anchors and summary contract

If a validation artifact anchor is null, emit no vector index, observation row,
or distance row; never substitute a nearest or terminal anchor. Aggregates use
only pairs where that exact comparison-owned anchor exists and must expose
contributing IDs and exact denominators. For A/B/C, persist
`available_pair_ids`, `available_pair_count`, `mean`, and `median`; count zero
means mean and median are JSON null. A>B and A>C persist comparable IDs,
denominator, and count; zero denominator means count 0 and rate, if emitted,
JSON null. No denominator may change silently.

`summary.json` has exactly top-level keys
`schema_version`, `comparison_order`, `anchor_order`, `integrity`, and
`aggregates`. `integrity` has exactly `pre_divergence_all_pass`, `pair_ids`,
and `layer_descriptors`. `aggregates` is an ordered JSON array. Each record
has exactly this key set:

~~~text
layer_index, layer_role, state_source, anchor_name,
a_available_pair_ids, a_available_pair_count, a_mean, a_median,
b_available_pair_ids, b_available_pair_count, b_mean, b_median,
c_available_pair_ids, c_available_pair_count, c_mean, c_median,
a_gt_b_comparable_pair_ids, a_gt_b_denominator, a_gt_b_count,
a_gt_c_comparable_pair_ids, a_gt_c_denominator, a_gt_c_count
~~~

Records are anchor order then ascending layer. No significance, p-values,
confidence intervals, thresholds, or selected-best fields are emitted.

### 16.8 Canonical bytes

Manifest and summary bytes are exactly:

~~~python
json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
           separators=(",", ": "), allow_nan=False) + "\n"
~~~

Encode UTF-8, without BOM, with LF only and exactly one final LF. Every JSONL
record is exactly `json.dumps(record, ensure_ascii=False, sort_keys=True,
separators=(",", ":"), allow_nan=False) + "\n"`, UTF-8 without BOM, no blank
lines, and one LF after the final record. Ordering is the frozen observation
or distance ordering above.

`report.md` has exactly these sections, in order: `# O0b Token-Aligned Native
Hidden-State Proxy Screening`, `## Provenance`, `## Integrity Checks`,
`## Pair-Level Distances`, `## Descriptive Summaries`, `## Scientific Boundary`.
It is generated only from validated manifest, distances, and summary; contains
no time, UUID, host, user, absolute path, or environment prose. Tables follow
distance/aggregate order. Finite numbers use `format(value, ".17g")`, null is
`NA`, booleans are `true`/`false`, and only frozen templates are allowed.
Encode UTF-8 without BOM, LF only, exactly one final LF.

### 16.9 Deterministic NPZ and ZIP

Construct the source matrix with `np.asarray(..., dtype=np.dtype("<f4"))`,
then `np.ascontiguousarray`, rank 2. Generate `vectors.npy` entirely in
memory using `io.BytesIO()` and exactly
`np.lib.format.write_array(buffer, vectors_array, version=(1, 0),
allow_pickle=False)`. NPY format is exactly 1.0; no automatic version or
pickle is allowed. The scientific execution authority must freeze the exact
Python and NumPy versions, and tests must compare two same-environment writes
byte-for-byte.

Create `anchor_hidden_states.npz` explicitly with construction semantics
equivalent to:

~~~python
zipfile.ZipFile(
    archive_buffer,
    mode="w",
    compression=zipfile.ZIP_STORED,
    allowZip64=False,
    compresslevel=None,
)
~~~

Freeze exactly `compression=zipfile.ZIP_STORED`, `compresslevel=None`, and
`allowZip64=False`. No DEFLATE or other compression, numeric compression
level, omitted `compresslevel`, or ambient/default compression-level policy is
permitted. Because `ZIP_STORED` is mandatory, no compressed payload semantics
may enter scientific artifact identity. Its `ZipInfo`
must set `date_time=(1980,1,1,0,0,0)`, `compress_type=ZIP_STORED`,
`create_system=3`, `create_version=20`, `extract_version=20`,
`external_attr=0o100644 << 16`, `internal_attr=0`, `extra=b""`,
`comment=b""`, and `flag_bits=0`; archive comment is `b""`, no duplicates,
filesystem path writes, mtime, or ambient permissions. Write the in-memory
NPY bytes with `writestr`. ZIP64 requirement fails closed. Future tests MUST
inspect the archive-construction kwargs and the resulting member metadata,
including `ZipInfo.compress_type == zipfile.ZIP_STORED`; they MUST fail if
archive construction relies on an unspecified/default compression level. Tests
inspect every field and exact bytes across two writes.

### 16.10 Checksums, manifest, publication, and failure

`SHA256SUMS.txt` covers exactly the six files
`anchor_hidden_states.npz`, `anchor_observations.jsonl`, `manifest.json`,
`paired_distances.jsonl`, `report.md`, `summary.json`, sorted by ASCII filename.
Each line is lowercase 64-hex SHA256, two ASCII spaces, filename, and `\n`;
ASCII, LF-only, no blank lines, exactly one final LF. It excludes itself.

The manifest must carry at least this exact required key set:

~~~text
schema_version, experiment_name,
scientific_design_authority_commit, boundary_recovery_authority_commit,
input_implementation_freeze_commit, observer_implementation_authority_commit,
observer_implementation_commit, observer_script_sha256, dataset_path,
dataset_sha256, validation_artifact_path, validation_artifact_sha256,
validation_artifact_repository_head, model_id, model_revision,
model_trust_remote_code, tokenizer_id, tokenizer_revision,
tokenizer_trust_remote_code, tokenizer_use_fast, add_special_tokens, device,
dtype, python_version, numpy_version, torch_version, transformers_version,
serialization_template, comparison_order, anchor_order, layer_descriptors,
pre_divergence_rtol, pre_divergence_atol, cosine_redundancy_atol,
exact_command, run_name, required_artifacts, execution_status
~~~

Freeze
`model_trust_remote_code=false`, `tokenizer_trust_remote_code=false`,
`tokenizer_use_fast=true`, `add_special_tokens=false`, `device="cpu"`,
`dtype="float32"`, `pre_divergence_rtol=0.0`,
`pre_divergence_atol=1e-6`, and `cosine_redundancy_atol=1e-12`.
Published output may have `execution_status` only `COMPLETE`; staging output
does not have valid COMPLETE semantics.

Before any write, require both `output_dir` and
`staging_dir = output_dir.with_name(output_dir.name + ".tmp")` not to exist.
Write all seven files only in staging, in this order: manifest, observations,
NPZ, distances, summary, report, then checksums. Before publication require
the exact seven-file set, all six matching checksums, valid canonical schemas,
pre-divergence PASS, and finite/valid records. Rename staging to output only
after validation on the same filesystem; output must still not exist. Any
failure before rename means no valid scientific output directory exists.
Never delete or reuse leftover staging. Rename failure fails closed.

### 16.11 Required 27 Precision-Repair Adversarial Obligations

The future implementation test suite MUST substantively cover ALL 27 numbered
obligations below. These are 27 mandatory test obligations. They may be
implemented as individual tests or explicit parametrized cases, but no item
may be merged away, omitted, or treated as implied coverage.

1. Model loader with `trust_remote_code=True` fails.
2. Model loader with missing or incorrect immutable model revision fails.
3. Pair001 comparison-specific divergence ownership is exact: paraphrase =
   18, insufficient = 25, surface-null = 17.
4. Two logical measurements with the same `anchor_name` but different
   `absolute_token_index` MUST NOT share a physical hidden vector.
5. Reference-vector reuse succeeds only when the entire physical vector key is
   identical: `(pair_id, condition, absolute_token_index, layer_index)`.
6. Every anchor-observation logical identity contains: `pair_id`,
   `comparison_id`, `anchor_name`, `absolute_token_index`, `layer_index`.
7. Every persisted vector/distance record preserves exact: `layer_index`,
   `layer_role`, `state_source`.
8. Metric calculation demonstrably follows: source hidden vector ->
   `np.asarray(..., dtype=np.float64)` -> float64 norm -> float64 unit vectors
   -> float64 subtraction/L2.
9. Emitted normalized-L2, cosine-distance, and redundancy-error values are
   Python `float` values.
10. Zero-norm vector fails closed and no epsilon/clipping path is used.
11. Cosine redundancy error strictly greater than `1e-12` fails.
12. Frozen unavailable/null anchor emits no observation row, no distance row,
    no vector index, and no substitute anchor.
13. Summary aggregates expose exact contributing IDs and exact denominators,
    including zero-denominator behavior.
14. Canonical JSON generation performed twice from identical controlled payload
    produces byte-identical output and identical SHA256.
15. Canonical JSONL generation performed twice verifies both exact byte identity
    and exact frozen row ordering.
16. `report.md` generation performed twice from identical controlled structures
    produces exact byte identity.
17. `vectors.npy` uses NumPy NPY format version exactly `(1, 0)`.
18. `anchor_hidden_states.npz` contains exactly one member: `vectors.npy`.
19. The resulting `ZipInfo` for `vectors.npy` is inspected and all frozen
    fields match exactly: filename, `date_time`, `compress_type`,
    `create_system`, `create_version`, `extract_version`, `external_attr`,
    `internal_attr`, `extra`, `comment`, `flag_bits`.
20. ZIP64 is disabled and a case requiring ZIP64 fails closed rather than
    silently changing archive format.
21. Two deterministic NPZ writes from identical controlled vectors under the
    same frozen environment are byte-identical and SHA256-identical.
22. `SHA256SUMS.txt` verifies exact six-file membership, lexicographic order,
    lowercase 64-hex digests, exactly two ASCII spaces, LF-only encoding, and
    exactly one final LF.
23. Pre-existing final `output_dir` causes fail-closed behavior before artifact
    writing.
24. Pre-existing `.tmp` staging directory causes fail-closed behavior and is
    neither deleted nor reused.
25. A partial/incomplete staging artifact set cannot obtain valid
    `execution_status="COMPLETE"` and cannot be treated as publishable output.
26. `SHA256SUMS.txt` is created only after the other six required artifacts are
    complete and hashed.
27. Final staging-directory rename/publication occurs only after exact
    seven-file-set validation, schema validation, checksum validation,
    finite-record validation, and all pre-divergence integrity checks pass.

For deterministic ZIP construction, future tests MUST assert:

~~~text
compression == zipfile.ZIP_STORED
compresslevel is None
allowZip64 is False
~~~

They must fail if archive construction relies on an unspecified/default
compression level.

Final repaired-candidate verdict: **LONGTERM_O0B_OBSERVER_AUTHORITY_FINAL_TWO_BLOCKER_REPAIR_PASS_READY_FOR_REVERIFICATION**
