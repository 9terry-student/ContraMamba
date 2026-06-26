import json
from pathlib import Path

Path("results").mkdir(exist_ok=True)

summary = {
    "stage": "Stage26 old-head real-Mamba validation",
    "date": "2026-06-26",
    "final_decision": {
        "final_v6b_old_head_config": "frame_preserve_w03",
        "boundary_w03": "valid secondary signal",
        "pair_contrastive_support_safe": "component signal KEEP; final config DROP / NOT ROBUST",
        "next_stage": "Stage27-H2A v7-H1 explicit entitlement decision signal"
    },
    "frame_preserve_w03_3seed": {
        "macro": "0.9785 ± 0.0159",
        "acc": "0.9856 ± 0.0112",
        "SUP_p": "0.9293 ± 0.0850",
        "SUP_r": "0.9667 ± 0.0484",
        "NE_r": "0.9864 ± 0.0172",
        "REF_r": "1.0000 ± 0.0000",
        "loc_SUP_total": 8,
        "role_SUP_total": 6,
        "entity_SUP_total": 2,
        "event_SUP_total": 4,
        "predicate_SUP_total": 1,
        "title_SUP_total": 1,
        "bad_SUP_total": 22,
        "missing_SUP_total": 0,
        "decision": "final v6B old-head winner"
    },
    "frame_preserve_pair_w01_supportsafe_sample300_3seed": {
        "macro": "0.9778 ± 0.0142",
        "acc": "0.9852 ± 0.0095",
        "SUP_p": "0.9116 ± 0.0490",
        "SUP_r": "0.9778 ± 0.0294",
        "NE_r": "0.9840 ± 0.0091",
        "REF_r": "1.0000 ± 0.0000",
        "loc_SUP_total": 16,
        "role_SUP_total": 5,
        "entity_SUP_total": 1,
        "event_SUP_total": 3,
        "predicate_SUP_total": 0,
        "title_SUP_total": 1,
        "bad_SUP_total": 26,
        "missing_SUP_total": 0,
        "decision": "component signal KEEP; final config DROP / NOT ROBUST"
    },
    "head_to_head_delta_pair_minus_baseline": {
        "macro": -0.0007,
        "acc": -0.0005,
        "SUP_p": -0.0177,
        "SUP_r": 0.0111,
        "NE_r": -0.0025,
        "REF_r": 0.0,
        "loc_SUP": 8,
        "role_SUP": -1,
        "entity_SUP": -1,
        "event_SUP": -1,
        "predicate_SUP": -1,
        "title_SUP": 0,
        "bad_SUP": 4,
        "missing_SUP": 0
    },
    "interpretation": [
        "Old Stage22 diagnostic heads are real-Mamba-valid in v6b_minimal.",
        "frame+preserve_w03 is the most robust final v6B old-head configuration.",
        "support_safe pair contrastive learns a nonzero component signal but is not robust as a directly mixed final training loss.",
        "The main regression from pair contrastive is increased location-swap false SUPPORT.",
        "Pair contrastive should be retained as a component-level H2/H3 candidate, not as the current final config."
    ]
}

json_path = Path("results/stage26_oldhead_real_mamba_summary.json")
md_path = Path("results/stage26_oldhead_real_mamba_summary.md")

json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

md = """# Stage26 Old-Head Real-Mamba Validation

Date: 2026-06-26

## Final decision

- **Final v6B old-head config:** `frame+preserve_w03`
- **Boundary:** valid secondary signal
- **Pair contrastive support-safe:** component signal **KEEP**, final config **DROP / NOT ROBUST**
- **Next stage:** Stage27-H2A, v7-H1 explicit entitlement decision signal

## frame+preserve_w03, 3 seeds

- Macro-F1: **0.9785 ± 0.0159**
- Accuracy: **0.9856 ± 0.0112**
- SUPPORT precision: **0.9293 ± 0.0850**
- SUPPORT recall: **0.9667 ± 0.0484**
- NOT_ENTITLED recall: **0.9864 ± 0.0172**
- REFUTE recall: **1.0000 ± 0.0000**
- Location false SUPPORT total: **8**
- Role false SUPPORT total: **6**
- Bad SUPPORT total: **22**
- Missing-evidence false SUPPORT total: **0**

Decision: **final v6B old-head winner**.

## frame+preserve+pair_w01_supportsafe_sample300, 3 seeds

- Macro-F1: **0.9778 ± 0.0142**
- Accuracy: **0.9852 ± 0.0095**
- SUPPORT precision: **0.9116 ± 0.0490**
- SUPPORT recall: **0.9778 ± 0.0294**
- NOT_ENTITLED recall: **0.9840 ± 0.0091**
- REFUTE recall: **1.0000 ± 0.0000**
- Location false SUPPORT total: **16**
- Role false SUPPORT total: **5**
- Bad SUPPORT total: **26**
- Missing-evidence false SUPPORT total: **0**

Decision: **component signal KEEP; final config DROP / NOT ROBUST**.

## Head-to-head delta: pair minus baseline

- Macro-F1: **-0.0007**
- Accuracy: **-0.0005**
- SUPPORT precision: **-0.0177**
- SUPPORT recall: **+0.0111**
- NOT_ENTITLED recall: **-0.0025**
- Location false SUPPORT: **+8**
- Bad SUPPORT: **+4**
- Missing-evidence false SUPPORT: **0**

## Interpretation

Old Stage22 diagnostic heads are real-Mamba-valid in `v6b_minimal`.

The most robust old-head configuration is `frame+preserve_w03`. Pair-contrastive frame supervision learns a nonzero ranking signal and improves some seeds, but direct mixing into final v6B training is not robust because it increases location-swap false SUPPORT. Keep pair contrastive as a component-level H2/H3 hypothesis, not as the current final v6B configuration.

## Next step

Stage27-H2A: add explicit compositional entitlement decision signal selection to v7-H1.
"""

md_path.write_text(md, encoding="utf-8")

print("wrote", json_path)
print("wrote", md_path)
