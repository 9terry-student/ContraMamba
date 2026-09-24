from __future__ import annotations
import hashlib, json, sys, xml.etree.ElementTree as ET
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
OUT = PAPER / "figures"
sys.path.insert(0, str(HERE))
import figure_data_source as frozen
NS = "{http://www.w3.org/2000/svg}"
STEMS = [
    "fig1_study_overview",
    "fig2_130m_causal_foundation",
    "fig3_five_scale_recurrence_reorganization",
    "fig4_objective_conditioned_functionalization",
]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def get(root, gid):
    n = root.find(f".//*[@id='{gid}']")
    assert n is not None, gid
    return n

def main():
    sources, data = frozen.load_data()
    manifest = json.loads((OUT / "figure_data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["sources"] == sources.sources
    assert manifest["extracted_values"] == sources.values
    assert manifest["plot_data"] == data
    for k in ["new_statistics", "model_execution", "tokenizer_execution", "training_execution", "evaluation_execution", "gpu_required"]:
        assert manifest[k] is False
    for name, expected in manifest["outputs"].items():
        p = OUT / name
        assert p.is_file() and p.stat().st_size > 1000, p
        assert digest(p) == expected, p
    roots = [ET.parse(OUT / (s + ".svg")).getroot() for s in STEMS]
    marks = [e for e in roots[1].iter() if (e.get("id") or "").startswith("scatter-pair-")]
    assert len(marks) == 300
    by_id = {e.get("id"): e for e in marks}
    for i, row in enumerate(data["pairs"]):
        e = by_id[f"scatter-pair-{i:03}"]
        assert float(e.get("data-x")) == row["Delta_L"]
        assert float(e.get("data-y")) == row["D_BEH"]
    ranks = ["P3/P5", "P3/P5", "P2/P5", "P5/P4", "P3/P5"]
    for i, rank in enumerate(ranks):
        txt = "".join(get(roots[2], f"ranks-{i}").itertext()).replace(" ", "")
        assert rank in txt
    count = 0
    for family in ["xg2", "xg4"]:
        matrix = data["coordinate_free_geometry"][family]["cka_matrix"]
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                e = get(roots[2], f"{family}-cell-{i}-{j}")
                assert float(e.get("data-value")) == value
                assert float(e.get("data-norm-min")) == 0.3
                assert float(e.get("data-norm-max")) == 1.0
                count += 1
    assert count == 50
    means = endpoints = 0
    for key in ["contra", "vanilla"]:
        for i, scale in enumerate(data["scale_order"]):
            e = get(roots[3], f"{key}-bar-{i}")
            x = data["objective_intervals"][scale][key]
            assert float(e.get("data-mean")) == x["point_mean"]
            assert float(e.get("data-ci-low")) == x["ci_low"]
            assert float(e.get("data-ci-high")) == x["ci_high"]
            means += 1; endpoints += 2
    assert means == 10 and endpoints == 20
    v = manifest["validation"]
    assert v["scatter_mark_count"] == 300
    assert v["figure3_printed_cka_cells_exact"] == 50
    assert v["monotonic_colormap"] == "PASS"
    assert v["figure4_means_exact"] == 10
    assert v["figure4_ci_endpoints_exact"] == 20
    assert v["figure4_association_cells_exact"] == 15
    assert v["figure4_table_rule_intersections"] == 0
    assert min(a["minimum_effective_font_pt"] for a in manifest["render_audits"].values()) >= 8.5
    print("RESULT=PASS_FINAL_FIGURE_STATIC_CHECKS")
    print("FIG2_SCATTER_COUNT=300")
    print("FIG3_RANKS=PASS")
    print("FIG3_CKA=50/50")
    print("FIG3_COLORMAP=PASS")
    print("FIG4_MEANS=10/10")
    print("FIG4_CI_ENDPOINTS=20/20")
    print("FIG4_ASSOCIATIONS=15/15")
    print("FIG4_TABLE_RULE_INTERSECTIONS=0")

if __name__ == "__main__": main()
