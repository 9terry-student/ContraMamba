# Stage34-A Held-Out Structured Coverage Probe Report

## Purpose

Stage34-A creates a diagnostic-only held-out structured coverage probe. It audits whether the Stage33-F structured coverage owner generalizes beyond Stage31 lexical pairs and templates.

The held-out probe must not be used for training, calibration, threshold selection, loss computation, or checkpoint selection.

## Stage33-F Context

Stage33-F F2 improved Stage31 external shadow behavior substantially, including whole/part recovery. Because the Stage33 whole/part v2 lexicon includes many Stage31 probe pairs, Stage31 success alone may reflect symbolic lexical memorization rather than general coverage generalization.

## Generated Probe

Output:

- `data/stage34a_heldout_coverage_probe.jsonl`

Default size:

- 400 rows
- 20 groups
- 20 rows per group

Required schema:

- `id`
- `pair_id`
- `claim`
- `evidence`
- `final_label`
- `label`
- `gold_label`
- `group`
- `stage34_family`
- `stage34_relation`
- `stage34_expected_route`
- `stage34_is_heldout`

## Groups

- `heldout_all_to_some_support`
- `heldout_some_to_all_not_entitled`
- `heldout_none_to_some_refute`
- `heldout_some_to_none_refute`
- `heldout_only_to_base_support`
- `heldout_also_to_only_not_entitled`
- `heldout_specific_to_general_support`
- `heldout_general_to_specific_not_entitled`
- `heldout_whole_to_part_support`
- `heldout_part_to_whole_not_entitled`
- `heldout_collection_to_member_support`
- `heldout_member_to_collection_not_entitled`
- `heldout_region_to_subregion_support`
- `heldout_subregion_to_region_not_entitled`
- `heldout_category_to_subcategory_support`
- `heldout_subcategory_to_category_not_entitled`
- `heldout_role_to_specialized_role_support`
- `heldout_specialized_role_to_role_not_entitled`
- `heldout_material_to_variant_support`
- `heldout_variant_to_material_not_entitled`

## Held-Out Lexical Policy

The builder avoids Stage31 expanded whole/part pairs and instead uses held-out synthetic pairs such as:

- `animals -> dogs`
- `instruments -> violins`
- `documents -> invoices`
- `machines -> turbines`
- `rooms -> kitchens`
- `courses -> laboratory courses`
- `devices -> routers`
- `roads -> bike lanes`
- `policies -> privacy policies`
- `messages -> urgent messages`
- `accounts -> administrator accounts`
- `files -> encrypted files`
- `buildings -> libraries`
- `regions -> coastal districts`
- `teams -> youth teams`
- `products -> refurbished products`
- `tickets -> priority tickets`
- `materials -> recycled materials`
- `medicines -> antibiotics`
- `facilities -> clinics`

## Evaluator Metrics

`scripts/evaluate_stage34_heldout_coverage.py` reports:

- current metrics
- shadow metrics
- delta metrics
- group metrics
- Stage33 route and reason counts
- safety counters
- support recovery counters
- whole/part-family diagnostics
- lexical generalization diagnostics
- direct-support diagnostics

## Safety Decisions

Decision labels:

- `STAGE34A_HELDOUT_GENERALIZATION_PROMISING`
- `STAGE34A_HELDOUT_SYMBOLIC_MEMORIZATION_RISK`
- `STAGE34A_HELDOUT_UNSAFE`
- `STAGE34A_HELDOUT_DIAGNOSTIC_ONLY`

## Limitations

- The probe is synthetic and template-based.
- Held-out lexical pairs reduce but do not eliminate template familiarity risk.
- The evaluator depends on prediction exports containing Stage32/Stage33 shadow fields.
