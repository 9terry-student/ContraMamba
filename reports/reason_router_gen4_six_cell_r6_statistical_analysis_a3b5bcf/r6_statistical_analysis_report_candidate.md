# ContraMamba Gen4 R6 Statistical Analysis Report - Candidate

## Provenance

- Statistical specification: `4dc5bacd10a254b5ecd339ac1fe78bad9def5c47`
- R5 artifact freeze: `a3b5bcf2ded8dc0e86e859bbba12b5601a2fdea0`
- R6 implementation authority: `5ae6d7cfd11f9d2b64617a145c7c248becd97aad`
- Input SHA256: `e3157cb5e4e878e4fbe99914a689e57648568d18b4ae71162e3ba1278093204e`
- Input rows: `32400`
- Source pairs: `300`
- Evaluators: `18`

## Confirmatory family

| Estimand | Mean | t | raw p | Holm p | Reject | d_z |
|---|---:|---:|---:|---:|:---:|---:|
| delta_title | -0.004618828506295189 | -1.1792923205852774 | 0.23921930738175834 | 0.23921930738175834 | False | -0.06808647387431682 |
| delta_name | -0.019670098963779983 | -11.208148391039456 | 1.4165100168122468e-24 | 8.49906010087348e-24 | True | -0.64710274906839 |
| delta_role | 0.010517086642794311 | 2.325121966578214 | 0.020734632886988656 | 0.04146926577397731 | True | 0.13424097933026438 |
| delta_predicate | 0.04420943002363115 | 8.046324793457737 | 2.023468937347122e-14 | 1.0117344686735611e-13 | True | 0.4645547785489984 |
| interaction_title_name | 0.00919857732080682 | 4.334050709614281 | 2.0021118820517502e-05 | 8.008447528207001e-05 | True | 0.2502265343877293 |
| title_minus_name | 0.015051270457484793 | 3.854623238554979 | 0.00014191065076442054 | 0.00042573195229326166 | True | 0.2225467764404304 |

## Scope limitation

These tests concern only the frozen 300 source-pair population under the fixed prespecified 18-evaluator population.

They do not establish arbitrary-model generalization, native-Mamba state causality, training benefit, or task-performance improvement.
