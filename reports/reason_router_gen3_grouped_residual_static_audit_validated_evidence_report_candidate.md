# Generation-3 Grouped Residual Static-Audit Validated Evidence

VERDICT = PASS_CANDIDATE

PHASE = GEN3_GROUPED_RESIDUAL_STATIC_AUDIT

STATIC_AUDIT_AUTHORITY = fc787fcdcbbb41eafa4f0e508a6b551b069efe9f

PARENT_GROUPED_VALIDATED_EVIDENCE = c97fd33dd8aa9c116f45071ee4545093ebba8f1c

GROUPED_SOURCE_COMMIT = 3e0e9a435068c552abf20f3a74e0c3eccca344a3

CORRECTED_STATIC_AUDIT_SHA256 = 100a3ed558d148a628bf10158a0c383d0a40c3990f128502c4e6e9a89a61b30a

HISTORICAL_SOURCE_ADMISSION = PASS_6_OF_6
GROUPED_IMPORT_AUDIT_BINDING = PASS_18_OF_18
GROUPED_PARENT_ANALYSIS_BINDING = PASS_18_OF_18

D1_SPECIFIC_RESIDUAL_GEOMETRY = 0_OF_4
OVERALL_RESIDUAL_VERDICT = HETEROGENEOUS_RESIDUAL
NEW_HIGHER_ORDER_EXECUTION_JUSTIFIED_BY_THIS_AUDIT = NO

## 1. Frozen residual population

- clinic_expansion__predicate_swap
- generated_fact_045__role_swap
- generated_fact_193__predicate_swap
- generated_fact_258__title_name_swap

No stable ID outside this frozen four-ID population is promoted by this audit.

## 2. Per-ID bounded classifications

| Stable ID | Frozen class | Reproducible proper subset | D1 seeds |
|---|---|---|---|
| `clinic_expansion__predicate_swap` | `DISTRIBUTED_NEAR_THRESHOLD` | — | 180, 182 |
| `generated_fact_045__role_swap` | `PROPER_SUBSET_NEAR_THRESHOLD` | `U+D` | 180, 181 |
| `generated_fact_193__predicate_swap` | `PROPER_SUBSET_NEAR_THRESHOLD` | `U+D` | 180, 182 |
| `generated_fact_258__title_name_swap` | `DISTRIBUTED_NEAR_THRESHOLD` | — | 180, 181, 182 |

### Interpretation

`generated_fact_193__predicate_swap` is the strongest `PROPER_SUBSET_NEAR_THRESHOLD` case: U+D is the closest negative-side proper subset, the closest proper subset to the D1 final margin, and the most supportward proper subset in both historical D1 seeds.

`generated_fact_045__role_swap` is also `PROPER_SUBSET_NEAR_THRESHOLD`: U+D is the closest negative-side proper subset in both relevant seeds and reproduces the authorization-side and final-margin direction in both seeds, although another subset crosses the final boundary in seed181.

`clinic_expansion__predicate_swap` and `generated_fact_258__title_name_swap` are `DISTRIBUTED_NEAR_THRESHOLD`: several proper subsets move toward the historical D1 authorization/final-boundary geometry, but the dominant proper subset changes across seeds.

Therefore the four residual stable IDs do not support one common bounded geometry class.

OVERALL_RESIDUAL_VERDICT = HETEROGENEOUS_RESIDUAL

## 3. Seed-resolved boundary rankings

### `clinic_expansion__predicate_swap`

| Seed | Closest negative proper subset | Closest proper subset to D1 margin | Most supportward proper subset |
|---:|---|---|---|
| 180 | QD | QD | QD |
| 182 | UD | UQ | UQ |

### `generated_fact_045__role_swap`

| Seed | Closest negative proper subset | Closest proper subset to D1 margin | Most supportward proper subset |
|---:|---|---|---|
| 180 | UD | UD | UD |
| 181 | UD | QD | QD |

### `generated_fact_193__predicate_swap`

| Seed | Closest negative proper subset | Closest proper subset to D1 margin | Most supportward proper subset |
|---:|---|---|---|
| 180 | UD | UD | UD |
| 182 | UD | UD | UD |

### `generated_fact_258__title_name_swap`

| Seed | Closest negative proper subset | Closest proper subset to D1 margin | Most supportward proper subset |
|---:|---|---|---|
| 180 | UD | QD | QD |
| 181 | D | D | D |
| 182 | U | U | U |

## 4. Seed-resolved row-level geometry

All deltas below are relative to matched canonical A0. `D1 ratio` is the prespecified descriptive `(margin_X - margin_A0) / (margin_D1 - margin_A0)`.

### `clinic_expansion__predicate_swap`

#### seed180

| Condition | Prediction | qAuth | ΔqAuth | Entitlement | ΔEnt | S-NE margin | Δmargin | D1 ratio | Cross up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | NOT_ENTITLED | 0.136744 | — | 0.136744 | — | -0.420830 | — | — | NO |
| D1 | SUPPORT | 0.305242 | 0.168497 | 0.305242 | 0.168498 | 0.433553 | 0.854383 | 1.000000 | YES |
| U | NOT_ENTITLED | 0.118859 | -0.017885 | 0.118859 | -0.017885 | -0.521536 | -0.100706 | -0.117870 | NO |
| Q | NOT_ENTITLED | 0.188685 | 0.051941 | 0.188685 | 0.051941 | -0.214980 | 0.205850 | 0.240934 | NO |
| D | NOT_ENTITLED | 0.129066 | -0.007679 | 0.129066 | -0.007678 | -0.457660 | -0.036830 | -0.043107 | NO |
| UQ | NOT_ENTITLED | 0.167511 | 0.030767 | 0.167511 | 0.030767 | -0.327975 | 0.092855 | 0.108681 | NO |
| UD | NOT_ENTITLED | 0.139547 | 0.002803 | 0.139547 | 0.002803 | -0.404145 | 0.016685 | 0.019529 | NO |
| QD | NOT_ENTITLED | 0.189217 | 0.052473 | 0.189217 | 0.052473 | -0.153768 | 0.267062 | 0.312579 | NO |

Auxiliary geometry:

| Condition | frame | predicate | sufficiency | qF | qP | qS | polarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.411749 | 0.334928 | 0.991573 | 0.588251 | 0.273843 | 0.001162 | 3.301329 |
| D1 | 0.563110 | 0.544366 | 0.995771 | 0.436890 | 0.256572 | 0.001296 | 3.625044 |
| U | 0.402876 | 0.299704 | 0.984390 | 0.597124 | 0.282133 | 0.001885 | 3.100201 |
| Q | 0.616912 | 0.307805 | 0.993662 | 0.383088 | 0.427024 | 0.001203 | 3.153473 |
| D | 0.452179 | 0.287752 | 0.991932 | 0.547821 | 0.322064 | 0.001050 | 3.224263 |
| UQ | 0.395336 | 0.428624 | 0.988555 | 0.604664 | 0.225885 | 0.001939 | 3.026409 |
| UD | 0.463116 | 0.304615 | 0.989188 | 0.536884 | 0.322044 | 0.001525 | 3.292397 |
| QD | 0.466214 | 0.407720 | 0.995437 | 0.533786 | 0.276129 | 0.000867 | 3.394049 |

#### seed182

| Condition | Prediction | qAuth | ΔqAuth | Entitlement | ΔEnt | S-NE margin | Δmargin | D1 ratio | Cross up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | NOT_ENTITLED | 0.089292 | — | 0.089292 | — | -0.560489 | — | — | NO |
| D1 | SUPPORT | 0.239896 | 0.150604 | 0.239896 | 0.150604 | 0.191558 | 0.752047 | 1.000000 | YES |
| U | NOT_ENTITLED | 0.153477 | 0.064185 | 0.153477 | 0.064185 | -0.232742 | 0.327747 | 0.435807 | NO |
| Q | NOT_ENTITLED | 0.137399 | 0.048107 | 0.137399 | 0.048107 | -0.336970 | 0.223519 | 0.297214 | NO |
| D | NOT_ENTITLED | 0.139954 | 0.050662 | 0.139954 | 0.050662 | -0.281817 | 0.278672 | 0.370551 | NO |
| UQ | SUPPORT | 0.196779 | 0.107487 | 0.196779 | 0.107487 | 0.009033 | 0.569522 | 0.757296 | YES |
| UD | NOT_ENTITLED | 0.155688 | 0.066396 | 0.155688 | 0.066396 | -0.207054 | 0.353435 | 0.469964 | NO |
| QD | NOT_ENTITLED | 0.127012 | 0.037720 | 0.127012 | 0.037720 | -0.339963 | 0.220526 | 0.293234 | NO |

Auxiliary geometry:

| Condition | frame | predicate | sufficiency | qF | qP | qS | polarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.423068 | 0.213299 | 0.989495 | 0.576932 | 0.332828 | 0.000948 | 4.179676 |
| D1 | 0.621210 | 0.390955 | 0.987777 | 0.378790 | 0.378345 | 0.002969 | 3.820296 |
| U | 0.473712 | 0.328458 | 0.986389 | 0.526288 | 0.318118 | 0.002118 | 3.908430 |
| Q | 0.499948 | 0.279957 | 0.981672 | 0.500052 | 0.359984 | 0.002565 | 3.607061 |
| D | 0.494205 | 0.286221 | 0.989409 | 0.505795 | 0.352753 | 0.001498 | 4.233362 |
| UQ | 0.491605 | 0.404045 | 0.990682 | 0.508395 | 0.292974 | 0.001851 | 4.043728 |
| UD | 0.416007 | 0.378354 | 0.989133 | 0.583993 | 0.258609 | 0.001710 | 4.108412 |
| QD | 0.526328 | 0.243291 | 0.991887 | 0.473672 | 0.398277 | 0.001039 | 4.291503 |

### `generated_fact_045__role_swap`

#### seed180

| Condition | Prediction | qAuth | ΔqAuth | Entitlement | ΔEnt | S-NE margin | Δmargin | D1 ratio | Cross up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | NOT_ENTITLED | 0.092816 | — | 0.092816 | — | -0.605039 | — | — | NO |
| D1 | SUPPORT | 0.210463 | 0.117647 | 0.210463 | 0.117647 | 0.016444 | 0.621483 | 1.000000 | YES |
| U | NOT_ENTITLED | 0.118448 | 0.025632 | 0.118448 | 0.025632 | -0.499735 | 0.105304 | 0.169440 | NO |
| Q | NOT_ENTITLED | 0.132583 | 0.039767 | 0.132583 | 0.039767 | -0.446337 | 0.158702 | 0.255360 | NO |
| D | NOT_ENTITLED | 0.131237 | 0.038421 | 0.131237 | 0.038421 | -0.437082 | 0.167957 | 0.270252 | NO |
| UQ | NOT_ENTITLED | 0.137260 | 0.044444 | 0.137260 | 0.044444 | -0.433119 | 0.171920 | 0.276629 | NO |
| UD | NOT_ENTITLED | 0.179095 | 0.086279 | 0.179095 | 0.086279 | -0.194728 | 0.410311 | 0.660213 | NO |
| QD | NOT_ENTITLED | 0.151463 | 0.058647 | 0.151463 | 0.058647 | -0.324704 | 0.280335 | 0.451074 | NO |

Auxiliary geometry:

| Condition | frame | predicate | sufficiency | qF | qP | qS | polarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.238212 | 0.391987 | 0.994000 | 0.761788 | 0.144836 | 0.000560 | 3.406980 |
| D1 | 0.293544 | 0.719538 | 0.996434 | 0.706456 | 0.082328 | 0.000753 | 3.729752 |
| U | 0.281967 | 0.425009 | 0.988399 | 0.718033 | 0.162128 | 0.001390 | 3.264295 |
| Q | 0.380147 | 0.350844 | 0.994082 | 0.619853 | 0.246775 | 0.000789 | 3.204305 |
| D | 0.346775 | 0.380872 | 0.993641 | 0.653225 | 0.214698 | 0.000840 | 3.273359 |
| UQ | 0.269710 | 0.514198 | 0.989727 | 0.730290 | 0.131026 | 0.001425 | 3.141472 |
| UD | 0.363305 | 0.496972 | 0.991925 | 0.636695 | 0.182752 | 0.001458 | 3.423411 |
| QD | 0.322712 | 0.471394 | 0.995651 | 0.677288 | 0.170587 | 0.000662 | 3.390555 |

#### seed181

| Condition | Prediction | qAuth | ΔqAuth | Entitlement | ΔEnt | S-NE margin | Δmargin | D1 ratio | Cross up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | NOT_ENTITLED | 0.167902 | — | 0.167902 | — | -0.242339 | — | — | NO |
| D1 | SUPPORT | 0.251421 | 0.083519 | 0.251421 | 0.083519 | 0.193913 | 0.436252 | 1.000000 | YES |
| U | NOT_ENTITLED | 0.203186 | 0.035283 | 0.203186 | 0.035284 | -0.054519 | 0.187820 | 0.430531 | NO |
| Q | NOT_ENTITLED | 0.185465 | 0.017563 | 0.185465 | 0.017563 | -0.165130 | 0.077209 | 0.176983 | NO |
| D | SUPPORT | 0.232907 | 0.065005 | 0.232907 | 0.065005 | 0.117890 | 0.360229 | 0.825736 | YES |
| UQ | SUPPORT | 0.236260 | 0.068358 | 0.236260 | 0.068358 | 0.100554 | 0.342893 | 0.785998 | YES |
| UD | NOT_ENTITLED | 0.207764 | 0.039862 | 0.207764 | 0.039862 | -0.034145 | 0.208194 | 0.477233 | NO |
| QD | SUPPORT | 0.240414 | 0.072512 | 0.240414 | 0.072512 | 0.132920 | 0.375259 | 0.860189 | YES |

Auxiliary geometry:

| Condition | frame | predicate | sufficiency | qF | qP | qS | polarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.394638 | 0.428775 | 0.992268 | 0.605362 | 0.225427 | 0.001308 | 3.400406 |
| D1 | 0.539916 | 0.468407 | 0.994151 | 0.460084 | 0.287016 | 0.001479 | 3.553733 |
| U | 0.492115 | 0.415642 | 0.993361 | 0.507885 | 0.287571 | 0.001358 | 3.597867 |
| Q | 0.369168 | 0.504710 | 0.995398 | 0.630832 | 0.182846 | 0.000857 | 3.472430 |
| D | 0.435739 | 0.537216 | 0.994966 | 0.564261 | 0.201653 | 0.001178 | 3.627428 |
| UQ | 0.531331 | 0.447183 | 0.994351 | 0.468669 | 0.293728 | 0.001342 | 3.545369 |
| UD | 0.455450 | 0.459695 | 0.992340 | 0.544550 | 0.246082 | 0.001604 | 3.504192 |
| QD | 0.447050 | 0.540017 | 0.995857 | 0.552950 | 0.205635 | 0.001000 | 3.588219 |

### `generated_fact_193__predicate_swap`

#### seed180

| Condition | Prediction | qAuth | ΔqAuth | Entitlement | ΔEnt | S-NE margin | Δmargin | D1 ratio | Cross up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | NOT_ENTITLED | 0.094596 | — | 0.094596 | — | -0.587609 | — | — | NO |
| D1 | SUPPORT | 0.253921 | 0.159325 | 0.253921 | 0.159325 | 0.239291 | 0.826900 | 1.000000 | YES |
| U | NOT_ENTITLED | 0.123104 | 0.028508 | 0.123104 | 0.028508 | -0.470742 | 0.116867 | 0.141331 | NO |
| Q | NOT_ENTITLED | 0.132848 | 0.038252 | 0.132848 | 0.038252 | -0.430152 | 0.157457 | 0.190418 | NO |
| D | NOT_ENTITLED | 0.115284 | 0.020689 | 0.115284 | 0.020688 | -0.500063 | 0.087546 | 0.105873 | NO |
| UQ | NOT_ENTITLED | 0.145818 | 0.051222 | 0.145818 | 0.051222 | -0.384077 | 0.203532 | 0.246139 | NO |
| UD | NOT_ENTITLED | 0.160140 | 0.065545 | 0.160140 | 0.065544 | -0.273622 | 0.313987 | 0.379716 | NO |
| QD | NOT_ENTITLED | 0.151210 | 0.056615 | 0.151210 | 0.056614 | -0.314103 | 0.273506 | 0.330761 | NO |

Auxiliary geometry:

| Condition | frame | predicate | sufficiency | qF | qP | qS | polarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.288136 | 0.330413 | 0.993612 | 0.711864 | 0.192932 | 0.000608 | 3.474980 |
| D1 | 0.510816 | 0.499125 | 0.995921 | 0.489184 | 0.255855 | 0.001040 | 3.712620 |
| U | 0.365171 | 0.342530 | 0.984186 | 0.634829 | 0.240089 | 0.001978 | 3.312888 |
| Q | 0.462123 | 0.289289 | 0.993723 | 0.537877 | 0.328436 | 0.000839 | 3.293102 |
| D | 0.387348 | 0.299614 | 0.993359 | 0.612652 | 0.271293 | 0.000771 | 3.312315 |
| UQ | 0.339578 | 0.435511 | 0.985988 | 0.660422 | 0.191688 | 0.002072 | 3.209854 |
| UD | 0.458312 | 0.352884 | 0.990164 | 0.541688 | 0.296581 | 0.001591 | 3.445613 |
| QD | 0.399762 | 0.379857 | 0.995769 | 0.600238 | 0.247910 | 0.000643 | 3.415393 |

#### seed182

| Condition | Prediction | qAuth | ΔqAuth | Entitlement | ΔEnt | S-NE margin | Δmargin | D1 ratio | Cross up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | NOT_ENTITLED | 0.052494 | — | 0.052494 | — | -0.755395 | — | — | NO |
| D1 | SUPPORT | 0.213369 | 0.160875 | 0.213369 | 0.160875 | 0.054748 | 0.810143 | 1.000000 | YES |
| U | NOT_ENTITLED | 0.143231 | 0.090737 | 0.143231 | 0.090737 | -0.292549 | 0.462846 | 0.571314 | NO |
| Q | NOT_ENTITLED | 0.086641 | 0.034147 | 0.086641 | 0.034147 | -0.603407 | 0.151988 | 0.187606 | NO |
| D | NOT_ENTITLED | 0.106929 | 0.054434 | 0.106929 | 0.054435 | -0.454899 | 0.300496 | 0.370917 | NO |
| UQ | NOT_ENTITLED | 0.174871 | 0.122376 | 0.174871 | 0.122377 | -0.119330 | 0.636065 | 0.785127 | NO |
| UD | NOT_ENTITLED | 0.174192 | 0.121697 | 0.174192 | 0.121698 | -0.104717 | 0.650678 | 0.803164 | NO |
| QD | NOT_ENTITLED | 0.089688 | 0.037194 | 0.089688 | 0.037194 | -0.542173 | 0.213222 | 0.263191 | NO |

Auxiliary geometry:

| Condition | frame | predicate | sufficiency | qF | qP | qS | polarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.374322 | 0.141471 | 0.991291 | 0.625678 | 0.321366 | 0.000461 | 4.167233 |
| D1 | 0.578123 | 0.372501 | 0.990793 | 0.421877 | 0.362772 | 0.001983 | 3.786702 |
| U | 0.443667 | 0.326196 | 0.989692 | 0.556333 | 0.298945 | 0.001492 | 3.795226 |
| Q | 0.364028 | 0.242223 | 0.982591 | 0.635972 | 0.275852 | 0.001535 | 3.513680 |
| D | 0.486190 | 0.221761 | 0.991753 | 0.513810 | 0.378372 | 0.000889 | 4.255454 |
| UQ | 0.493433 | 0.357211 | 0.992118 | 0.506567 | 0.317174 | 0.001389 | 3.948439 |
| UD | 0.450241 | 0.390128 | 0.991687 | 0.549759 | 0.274590 | 0.001460 | 4.081322 |
| QD | 0.511463 | 0.176687 | 0.992469 | 0.488537 | 0.421094 | 0.000681 | 4.294351 |

### `generated_fact_258__title_name_swap`

#### seed180

| Condition | Prediction | qAuth | ΔqAuth | Entitlement | ΔEnt | S-NE margin | Δmargin | D1 ratio | Cross up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | NOT_ENTITLED | 0.137839 | — | 0.137839 | — | -0.399580 | — | — | NO |
| D1 | SUPPORT | 0.261241 | 0.123402 | 0.261241 | 0.123402 | 0.256192 | 0.655772 | 1.000000 | YES |
| U | NOT_ENTITLED | 0.152123 | 0.014284 | 0.152123 | 0.014284 | -0.357326 | 0.042254 | 0.064434 | NO |
| Q | NOT_ENTITLED | 0.210342 | 0.072503 | 0.210342 | 0.072503 | -0.104801 | 0.294779 | 0.449514 | NO |
| D | NOT_ENTITLED | 0.167273 | 0.029434 | 0.167273 | 0.029434 | -0.274927 | 0.124653 | 0.190086 | NO |
| UQ | NOT_ENTITLED | 0.195127 | 0.057288 | 0.195127 | 0.057288 | -0.182269 | 0.217311 | 0.331382 | NO |
| UD | NOT_ENTITLED | 0.212124 | 0.074285 | 0.212124 | 0.074285 | -0.039162 | 0.360418 | 0.549609 | NO |
| QD | SUPPORT | 0.231381 | 0.093542 | 0.231381 | 0.093542 | 0.046426 | 0.446006 | 0.680124 | YES |

Auxiliary geometry:

| Condition | frame | predicate | sufficiency | qF | qP | qS | polarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.341391 | 0.406427 | 0.993432 | 0.658609 | 0.202641 | 0.000911 | 3.370714 |
| D1 | 0.495318 | 0.529992 | 0.995147 | 0.504682 | 0.232804 | 0.001274 | 3.629419 |
| U | 0.359072 | 0.431292 | 0.982293 | 0.640928 | 0.204207 | 0.002742 | 3.208057 |
| Q | 0.593123 | 0.357106 | 0.993081 | 0.406877 | 0.381315 | 0.001466 | 3.189562 |
| D | 0.438649 | 0.383961 | 0.993164 | 0.561351 | 0.270225 | 0.001151 | 3.227759 |
| UQ | 0.357892 | 0.554974 | 0.982411 | 0.642108 | 0.159271 | 0.003494 | 3.129529 |
| UD | 0.458436 | 0.467699 | 0.989339 | 0.541564 | 0.244026 | 0.002286 | 3.379544 |
| QD | 0.455904 | 0.509860 | 0.995414 | 0.544096 | 0.223457 | 0.001066 | 3.303116 |

#### seed181

| Condition | Prediction | qAuth | ΔqAuth | Entitlement | ΔEnt | S-NE margin | Δmargin | D1 ratio | Cross up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | NOT_ENTITLED | 0.148711 | — | 0.148711 | — | -0.332980 | — | — | NO |
| D1 | SUPPORT | 0.210851 | 0.062140 | 0.210851 | 0.062140 | 0.007188 | 0.340168 | 1.000000 | YES |
| U | NOT_ENTITLED | 0.173901 | 0.025190 | 0.173901 | 0.025190 | -0.195895 | 0.137085 | 0.402992 | NO |
| Q | NOT_ENTITLED | 0.134780 | -0.013931 | 0.134780 | -0.013931 | -0.407018 | -0.074038 | -0.217651 | NO |
| D | NOT_ENTITLED | 0.209114 | 0.060403 | 0.209114 | 0.060403 | -0.017026 | 0.315954 | 0.928818 | NO |
| UQ | NOT_ENTITLED | 0.187774 | 0.039063 | 0.187774 | 0.039063 | -0.113536 | 0.219444 | 0.645105 | NO |
| UD | NOT_ENTITLED | 0.175150 | 0.026439 | 0.175150 | 0.026439 | -0.175973 | 0.157007 | 0.461557 | NO |
| QD | NOT_ENTITLED | 0.164806 | 0.016095 | 0.164806 | 0.016095 | -0.241251 | 0.091729 | 0.269658 | NO |

Auxiliary geometry:

| Condition | frame | predicate | sufficiency | qF | qP | qS | polarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.402371 | 0.373272 | 0.990127 | 0.597629 | 0.252177 | 0.001483 | 3.380530 |
| D1 | 0.486041 | 0.437502 | 0.991569 | 0.513959 | 0.273397 | 0.001793 | 3.541543 |
| U | 0.464389 | 0.377890 | 0.990959 | 0.535611 | 0.288901 | 0.001587 | 3.600060 |
| Q | 0.327454 | 0.413827 | 0.994617 | 0.672546 | 0.191945 | 0.000729 | 3.423462 |
| D | 0.424053 | 0.496164 | 0.993891 | 0.575947 | 0.213653 | 0.001285 | 3.534857 |
| UQ | 0.466531 | 0.405756 | 0.991950 | 0.533469 | 0.277233 | 0.001524 | 3.625232 |
| UD | 0.420692 | 0.420919 | 0.989116 | 0.579308 | 0.243615 | 0.001927 | 3.579543 |
| QD | 0.370689 | 0.446692 | 0.995303 | 0.629311 | 0.205105 | 0.000778 | 3.528772 |

#### seed182

| Condition | Prediction | qAuth | ΔqAuth | Entitlement | ΔEnt | S-NE margin | Δmargin | D1 ratio | Cross up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | NOT_ENTITLED | 0.045855 | — | 0.045855 | — | -0.788668 | — | — | NO |
| D1 | SUPPORT | 0.251335 | 0.205480 | 0.251335 | 0.205480 | 0.191841 | 0.980509 | 1.000000 | YES |
| U | NOT_ENTITLED | 0.168072 | 0.122216 | 0.168072 | 0.122217 | -0.178750 | 0.609918 | 0.622042 | NO |
| Q | NOT_ENTITLED | 0.108745 | 0.062890 | 0.108745 | 0.062890 | -0.487579 | 0.301089 | 0.307074 | NO |
| D | NOT_ENTITLED | 0.083009 | 0.037154 | 0.083009 | 0.037154 | -0.583957 | 0.204711 | 0.208780 | NO |
| UQ | NOT_ENTITLED | 0.144513 | 0.098657 | 0.144513 | 0.098658 | -0.294007 | 0.494661 | 0.504494 | NO |
| UD | NOT_ENTITLED | 0.145784 | 0.099929 | 0.145784 | 0.099929 | -0.262806 | 0.525862 | 0.536315 | NO |
| QD | NOT_ENTITLED | 0.050583 | 0.004727 | 0.050583 | 0.004728 | -0.759725 | 0.028943 | 0.029518 | NO |

Auxiliary geometry:

| Condition | frame | predicate | sufficiency | qF | qP | qS | polarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.262552 | 0.176130 | 0.991606 | 0.737448 | 0.216309 | 0.000388 | 4.199889 |
| D1 | 0.549429 | 0.461815 | 0.990542 | 0.450571 | 0.295694 | 0.002400 | 3.536595 |
| U | 0.444505 | 0.382227 | 0.989229 | 0.555495 | 0.274603 | 0.001830 | 3.751544 |
| Q | 0.358538 | 0.308664 | 0.982627 | 0.641462 | 0.247870 | 0.001923 | 3.497730 |
| D | 0.357324 | 0.234241 | 0.991748 | 0.642676 | 0.273624 | 0.000691 | 4.253641 |
| UQ | 0.357531 | 0.407501 | 0.991890 | 0.642469 | 0.211837 | 0.001182 | 3.860686 |
| UD | 0.362465 | 0.405771 | 0.991204 | 0.637535 | 0.215387 | 0.001294 | 4.056767 |
| QD | 0.305133 | 0.167019 | 0.992538 | 0.694867 | 0.254170 | 0.000380 | 4.227004 |

## 5. Cross-seed D1-approach ratios

### `clinic_expansion__predicate_swap`

| Arm | Seed-resolved D1-approach ratios |
|---|---|
| U | seed180=-0.117870, seed182=0.435807 |
| Q | seed180=0.240934, seed182=0.297214 |
| D | seed180=-0.043107, seed182=0.370551 |
| UQ | seed180=0.108681, seed182=0.757296 |
| UD | seed180=0.019529, seed182=0.469964 |
| QD | seed180=0.312579, seed182=0.293234 |

### `generated_fact_045__role_swap`

| Arm | Seed-resolved D1-approach ratios |
|---|---|
| U | seed180=0.169440, seed181=0.430531 |
| Q | seed180=0.255360, seed181=0.176983 |
| D | seed180=0.270252, seed181=0.825736 |
| UQ | seed180=0.276629, seed181=0.785998 |
| UD | seed180=0.660213, seed181=0.477233 |
| QD | seed180=0.451074, seed181=0.860189 |

### `generated_fact_193__predicate_swap`

| Arm | Seed-resolved D1-approach ratios |
|---|---|
| U | seed180=0.141331, seed182=0.571314 |
| Q | seed180=0.190418, seed182=0.187606 |
| D | seed180=0.105873, seed182=0.370917 |
| UQ | seed180=0.246139, seed182=0.785127 |
| UD | seed180=0.379716, seed182=0.803164 |
| QD | seed180=0.330761, seed182=0.263191 |

### `generated_fact_258__title_name_swap`

| Arm | Seed-resolved D1-approach ratios |
|---|---|
| U | seed180=0.064434, seed181=0.402992, seed182=0.622042 |
| Q | seed180=0.449514, seed181=-0.217651, seed182=0.307074 |
| D | seed180=0.190086, seed181=0.928818, seed182=0.208780 |
| UQ | seed180=0.331382, seed181=0.645105, seed182=0.504494 |
| UD | seed180=0.549609, seed181=0.461557, seed182=0.536315 |
| QD | seed180=0.680124, seed181=0.269658, seed182=0.029518 |

## 6. Scientific interpretation

The previously unrecovered four-ID residual is not supported as a single D1-specific or all-group-only geometry.

Two stable IDs show reproducible U+D near-threshold structure. Two stable IDs instead show distributed near-threshold structure whose dominant proper subset changes across frozen training seeds.

Accordingly, failure of these IDs to satisfy the earlier recurrent proper-subset error criterion must not be interpreted as evidence that proper subsets have no relevant authorization/final-boundary effect.

The static evidence is consistent with heterogeneous threshold proximity under already-tested proper subsets. It does not establish the causal origin of those threshold positions.

D1_SPECIFIC_RESIDUAL_GEOMETRY = 0_OF_4
OVERALL_RESIDUAL_VERDICT = HETEROGENEOUS_RESIDUAL

## 7. Execution decision

This audit does not scientifically justify an arbitrary three-way or higher-order grouped search, a lambda sweep, additional seeds, or a new Generation-3 combinatorial execution.

The remaining four-ID residual does not constitute evidence for one unresolved D1-specific global mechanism. Further combinatorial execution would therefore require a new independently motivated scientific hypothesis rather than continuation by residual chasing.

NEW_HIGHER_ORDER_EXECUTION_JUSTIFIED_BY_THIS_AUDIT = NO
KAGGLE_REQUIRED = NO

## 8. Claim boundary

This audit does not establish:

- causal necessity;
- causal sufficiency;
- unique edge or group causation;
- parameter ownership;
- gradient orthogonality;
- a native Mamba recurrent-state mechanism;
- polarity irrelevance;
- an optimal lambda;
- an untested higher-order interaction;
- production readiness.

In particular, `PROPER_SUBSET_NEAR_THRESHOLD` and `DISTRIBUTED_NEAR_THRESHOLD` are descriptive geometry classes, not mechanistic causal labels.

## 9. Final verdict

STATIC_AUDIT_SOURCE_AUTHENTICATION = PASS
STATIC_AUDIT_DERIVED_RANKING_CORRECTION = PASS
STATIC_AUDIT_CLASSIFICATION = PASS

PROPER_SUBSET_NEAR_THRESHOLD = 2_OF_4
DISTRIBUTED_NEAR_THRESHOLD = 2_OF_4
D1_SPECIFIC_RESIDUAL_GEOMETRY = 0_OF_4

OVERALL_RESIDUAL_VERDICT = HETEROGENEOUS_RESIDUAL

NEW_HIGHER_ORDER_EXECUTION_JUSTIFIED_BY_THIS_AUDIT = NO

No execution authority is granted by this report.

END_OF_GEN3_GROUPED_RESIDUAL_STATIC_AUDIT_VALIDATED_EVIDENCE_REPORT
