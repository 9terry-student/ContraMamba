# ContraMamba Gen4 Low-Displacement Tokenizer-Anchor Eligibility Prospective Plan

## 0. Status

This file is a prospective static eligibility plan, not a new authority document.

It does not authorize scientific model execution by itself.

It introduces:

- no model forward;
- no checkpoint load;
- no GPU requirement;
- no behavioral endpoint;
- no p-value;
- no scientific conclusion.

Plan creation base:

`37e8176edceb9da9a976df4520904e31e1134ff5`

Frozen structural input:

`data/reason_router_gen4_mamba370m14b_low_displacement_xg1_v1`

with:

- `xg1_fact_5401..xg1_fact_5700`
- 300 source pairs
- 1800 six-cell structural rows
- target cells `C0_SHAM`, `C2_NAME`
- no response/model/tokenizer fields observed during materialization.

## 1. Purpose

Before implementing or executing the low-displacement GPU runner, verify that
the exact fresh target rows are valid under the exact tokenizer identities that
will be used by each model scale.

The question is purely technical:

> For every fresh C0/C2 target row, does the exact scale-local tokenizer map the
> frozen `A_IDENTITY` event into the active serialized sequence with the required
> post+4 prefix window available?

This gate does not inspect model responses or scientific outcomes.

## 2. Scale-local tokenizer identities

The two scales must be checked independently.

### Mamba-370M

- repository: `state-spaces/mamba-370m-hf`
- exact revision:
  `589179554943157be31701edd8b4558889276674`
- tokenizer runtime:
  `tokenizers==0.22.2`
- frozen historical tokenizer file SHA256:
  - `tokenizer.json`:
    `b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`
  - `tokenizer_config.json`:
    `9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`
  - historical behavioral provenance also records
    `special_tokens_map.json`:
    `57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

Implementation must defer to the frozen
`reason_router_gen4_mamba370m_geometry_prepare_fast_cuda.load_tokenizer`
contract for the exact accepted file set and IDs rather than reimplementing a
different tokenizer loader.

### Mamba-1.4B

- repository: `state-spaces/mamba-1.4b-hf`
- exact revision:
  `6e46eae61c27280517feef46f536d16b91076f08`
- tokenizer runtime:
  `tokenizers==0.22.2`
- frozen historical tokenizer file SHA256:
  - `tokenizer.json`:
    `3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8`
  - `tokenizer_config.json`:
    `3ba257483d22a5a84aab5465aa427e59bdaeb55f09fb14349e2d571ff67e8020`

Implementation must defer to the frozen
`reason_router_gen4_mamba14b_geometry_prepare_fast_cuda.load_tokenizer`
contract for the exact accepted file set and IDs.

The two tokenizer file identities are not assumed equal.

## 3. Exact snapshots

Eligibility is evaluated against exact snapshot directories whose final path
component equals the frozen revision.

Expected execution-time snapshot locations in the established Kaggle workflow:

### 370M

`/kaggle/working/contramamba_external_snapshots/589179554943157be31701edd8b4558889276674`

### 1.4B

`/kaggle/working/contramamba_external_snapshots/6e46eae61c27280517feef46f536d16b91076f08`

The implementation must accept explicit snapshot arguments. It must not silently
fall back to another cached revision.

If the exact snapshots are unavailable locally, this eligibility step may run on
Kaggle with GPU OFF. It remains a CPU/static tokenizer operation.

## 4. Frozen structural population

Read only:

- `structured_source_facts.jsonl`
- `synthetic_reason_router_six_cell.jsonl`
- `structural_manifest.json`
- `SHA256SUMS.txt`

from:

`data/reason_router_gen4_mamba370m14b_low_displacement_xg1_v1`

Validate the frozen structural checksums before tokenization.

Required exact contract:

- first pair: `xg1_fact_5401`
- last pair: `xg1_fact_5700`
- source pair count: `300`
- row count: `1800`
- rows per pair: `6`
- generator family:
  `xg1_independent_structured_records_v1`
- target cells:
  - `C0_SHAM`
  - `C2_NAME`
- response fields absent;
- endpoint values absent;
- model execution false;
- CUDA execution false.

Do not regenerate or replace the frozen cohort for scientific convenience.

## 5. Target rows

The low-displacement behavioral runner will use only:

- `C0_SHAM`
- `C2_NAME`

For each scale:

- target source pairs: `300`
- target cells per pair: `2`
- required target rows: `600`

The eligibility gate may read the full six-cell structural artifact for
validation, but the primary eligibility verdict is defined on exactly these 600
target rows.

No row filtering is allowed.

## 6. Active serialization contract

Use the exact active encoding contract already used by the 370M/1.4B behavioral
bridge:

- add special tokens: `False`
- maximum serialized length: `128`
- claim budget: `63`
- explicit EOS separator
- evidence budget: `64`

The serialized coordinate system must match the production
`six_cell_tier2_inference_adapter` / scale-local geometry tokenizer contract.

Do not introduce a different padding, truncation, or special-token policy.

## 7. Anchor definition

For every target row, use the existing XG1 structural anchor logic.

Required event:

`A_IDENTITY`

The structural event is derived from the frozen generator-declared title/name
span, not substring search.

For `C0_SHAM` and `C2_NAME`, also compute `A_NAME` as a structural sanity event.

Require:

`absolute_index(A_IDENTITY) == absolute_index(A_NAME)`

for every target row.

This identity is a technical sanity check and is not a scientific endpoint.

## 8. Prefix eligibility rule

For every target row and scale, require:

`a + 4 <= terminal_index - 1`

where `a` is the absolute serialized `A_IDENTITY` token index.

No shortened post window is allowed.

Technical exclusion codes may be emitted for diagnostics, including:

- span not mapped;
- incomplete span mapping;
- anchor removed by evidence truncation;
- post+4 prefix ineligible.

Any exclusion on any target row blocks that scale.

Do not drop the row.

## 9. Per-scale PASS rule

For each scale independently:

- source pairs = 300
- target rows = 600
- `A_IDENTITY` eligible rows = 600
- `A_NAME` eligible rows = 600
- identity/name absolute-index mismatch count = 0
- exclusion count = 0

Only then:

`PASS_600_OF_600`

Otherwise:

`BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY`

## 10. Cross-scale PASS rule

The combined eligibility gate passes only if:

- Mamba-370M = `PASS_600_OF_600`
- Mamba-1.4B = `PASS_600_OF_600`

Combined result:

`PASS_MAMBA370M14B_LOW_DISPLACEMENT_TOKENIZER_ANCHOR_ELIGIBILITY`

Any scale failure blocks low-displacement GPU implementation/execution until the
technical incompatibility is understood.

A failure does not authorize:

- row deletion;
- replacement cohort;
- token sweep;
- shorter post window;
- alternate tokenizer;
- alternate revision.

## 11. Output artifact

Create exactly:

`reports/reason_router_gen4_mamba370m14b_low_displacement_tokenizer_anchor_eligibility_v1/`

with:

1. `mamba370m_anchor_manifest.jsonl`
2. `mamba370m_eligibility_summary.json`
3. `mamba14b_anchor_manifest.jsonl`
4. `mamba14b_eligibility_summary.json`
5. `cross_scale_summary.json`
6. `artifact_manifest.json`
7. `SHA256SUMS.txt`

## 12. Required summary provenance

Each scale summary must record:

- scale;
- exact HF repository;
- exact HF revision;
- tokenizer runtime version;
- tokenizer file SHA256 map returned by the frozen scale-local loader;
- vocabulary size;
- structural source SHA256;
- structural row SHA256;
- source pair count;
- target row count;
- eligible identity count;
- eligible name count;
- exclusion counts;
- identity/name mismatch count;
- model forward count = `0`;
- checkpoint load count = `0`;
- GPU used = `false`;
- scientific outcomes observed = `false`.

The cross-scale summary must record both scale verdicts and the combined verdict.

## 13. Information boundary

This eligibility process must not read:

- historical behavioral means;
- historical behavioral p-values;
- Study-B readout values;
- Study-B pair-level merge values;
- Study-C displacement values;
- Study-A response values.

It may read only structural rows, tokenizer files, and frozen tokenizer/encoding
implementation code required to establish exact token coordinates.

## 14. Execution boundary

This phase is static tokenization only.

Forbidden:

- model instantiation;
- checkpoint loading;
- CUDA tensor creation;
- model forward;
- backward;
- training;
- behavioral margin computation;
- native-state capture;
- intervention construction;
- p-values.

GPU must remain OFF if this is run in Kaggle.

## 15. Stop rule

Execute the gate once on the frozen fresh cohort and exact tokenizer revisions.

Freeze the outcome regardless of PASS or BLOCKED.

Do not change:

- cohort;
- target cells;
- anchor;
- post-window rule;
- tokenizer revision;
- truncation budgets;
- special-token policy

after observing eligibility.

If and only if the combined gate passes, proceed to low-displacement runner
implementation.
