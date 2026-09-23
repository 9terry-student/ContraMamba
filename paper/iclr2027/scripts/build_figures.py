"""Render frozen Mamba-1 evidence; no models, estimators, or statistical tests.

Run from the repository root. Requires reportlab and PyMuPDF.
All source bytes are read from pinned Git commits, never an untracked run.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/iclr2027/figures"
BASE_SCIENCE_HEAD = "3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69"
PAPER_EVIDENCE_HEAD = "2e65dd887ba4159d4024d85b8996306cfadec023"
CF_ROOT = "paper/iclr2027/frozen_sources/coordinate_free_geometry_v1"
CF_RESULT = CF_ROOT + "/coordinate_free_geometry_result.json"
CF_PROVENANCE = CF_ROOT + "/provenance_manifest.json"
CF_SNAPSHOT = CF_ROOT + "/snapshot_manifest.json"
SCALES = ["130M", "370M", "790M", "1.4B", "2.8B"]
PREFIX = "reports/reason_router_gen4_"
SYNTH = PREFIX + "pp3_transport_specificity_necessity_restoration_sufficiency_mechanism_synthesis.md"
CORE_SYNTH = PREFIX + "core_stable_residual_plastic_cross_scale_synthesis.md"
GEO_SYNTH = PREFIX + "mamba1_native_geometry_functional_coupling_static_bridge.md"
GEO_130_KERNEL = "scripts/reason_router_gen4_k_directional_alignment_transport_core.py"
GEO_130_PLANES = PREFIX + "pp3_excluded_residual_static_analysis_7a6c30f.json"
DELTA_SYNTH = PREFIX + "mamba1_five_scale_delta_l_descriptive_synthesis.md"
LM_MD = PREFIX + "mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.md"
LM = PREFIX + "mamba1_vanilla_lm_functional_control_analysis_v1/functional_control_analysis.json"
EXT = PREFIX + "mamba130m_vanilla_lm_completeness_analysis_v1/mamba130m_completeness_analysis.json"
BEH = PREFIX + "seed181_behavioral_restoration_bridge_analysis_retry2.json"
PAIR = PREFIX + "mamba130m_readout_behavior_pair_merge_v1/"
SPEC = PREFIX + "pp3_pp5_fresh_xg1_specificity_primary_inference_report_candidate.json"
CONF = {
    "370M": PREFIX + "mamba370m_confirmation_runs/g4k-mamba370-confirmation-xg1-3301-3600-2gpu-ac69e80/confirmation_inference.json",
    "790M": PREFIX + "mamba790m_confirmation_runs/g4k-mamba790m-confirmation-xg1-7201-7500-p2-p5-2gpu-d3117c1-fresh/confirmation_inference.json",
    "1.4B": PREFIX + "mamba14b_confirmation_runs/g4k-mamba14b-confirmation-xg1-4201-4500-2gpu-15304c1/confirmation_inference.json",
    "2.8B": PREFIX + "mamba28b_confirmation_runs/g4k-mamba28b-confirmation-xg1-6301-6600-p3-p5-2gpu-50a63a3/confirmation_inference.json",
}
GEO = {
    "370M": PREFIX + "mamba370m_geometry_preparation_runs/g4k-mamba370-geometry-xg2xg4-2gpu-d8e71ad-retry2/geometry_summary.json",
    "790M": PREFIX + "mamba790m_geometry_preparation_runs/g4k-mamba790m-geometry-xg2xg4-2gpu-774983b-retry2/geometry_summary.json",
    "1.4B": PREFIX + "mamba14b_geometry_preparation_runs/g4k-mamba14b-geometry-xg2xg4-2gpu-c758d5e-retry1/geometry_summary.json",
    "2.8B": PREFIX + "mamba28b_geometry_preparation_runs/g4k-mamba28b-geometry-xg2xg4-2gpu-7744d94/geometry_summary.json",
}
STEMS = ["fig1_study_overview", "fig2_130m_causal_foundation",
         "fig3_five_scale_recurrence_reorganization", "fig4_objective_conditioned_functionalization"]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


class Sources:
    def __init__(self):
        self.sources, self.values, self.cache = {}, {}, {}

    def read_at(self, authority, path, figure):
        cache_key = (authority, path)
        if cache_key not in self.cache:
            raw = git("show", f"{authority}:{path}")
            self.cache[cache_key] = raw.decode("utf-8")
            if path in self.sources:
                if self.sources[path]["authority_commit"] != authority:
                    raise RuntimeError(f"Source reused under different authority: {path}")
            else:
                self.sources[path] = {
                    "authority_commit": authority,
                    "git_blob": git("rev-parse", f"{authority}:{path}").decode().strip(),
                    "frozen_bytes_sha256": hashlib.sha256(raw).hexdigest(),
                    "figures": [],
                }
        if figure not in self.sources[path]["figures"]:
            self.sources[path]["figures"].append(figure)
        return self.cache[cache_key]

    def read(self, path, figure):
        return self.read_at(BASE_SCIENCE_HEAD, path, figure)

    def record(self, key, value, path, locator, figure):
        if key in self.values:
            raise ValueError(f"Duplicate data key: {key}")
        self.values[key] = {
            "value": value,
            "source": path,
            "locator": locator,
            "figure": figure,
            "transformation": "none; display rounding only",
        }
        return value

    def field(self, key, path, pointer, figure):
        value = json.loads(self.read(path, figure))
        for part in pointer.split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
        return self.record(key, value, path, "/" + pointer, figure)

    def mean_md(self, key, section, endpoint, figure):
        text = self.read(SYNTH, figure)
        block = text.split(f"### {section}. ")[1].split("\n### ")[0]
        matches = re.findall(r"^- mean " + re.escape(endpoint) + r": `([^`]+)`", block, re.M)
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous frozen Markdown mean: {key}")
        value = self.record(
            key,
            float(matches[0]),
            SYNTH,
            f"section {section}; exact bullet 'mean {endpoint}'",
            figure,
        )
        self.values[key]["source_decimal_lexeme"] = matches[0]
        return value

def load_data():
    s = Sources()
    for fig, paths in {"fig1": [SYNTH, CORE_SYNTH, DELTA_SYNTH, LM_MD, EXT],
                       "fig2": [SYNTH], "fig3": [SYNTH, CORE_SYNTH],
                       "fig4": [DELTA_SYNTH, LM_MD]}.items():
        for path in paths:
            s.read(path, fig)
    d = {"scale_order": SCALES, "planes": {}, "core": {}, "geometry": {}, "objectives": {}, "coordinate_free_geometry": {}}
    d["chain"] = [
        s.mean_md("transport.mean", 3, "C_PP3", "fig2"),
        s.field("specificity.mean", SPEC, "descriptive/D_SPEC/mean", "fig2"),
        s.mean_md("necessity.mean", 5, "D_NEC", "fig2"),
        s.mean_md("sufficiency.mean", 6, "D_SUF", "fig2"),
    ]
    # Preserve the full frozen decimal text in equal-size evidence cards.
    d["chain_display_means"] = [
        s.values[key].get("source_decimal_lexeme", repr(s.values[key]["value"]))
        for key in ["transport.mean", "specificity.mean", "necessity.mean", "sufficiency.mean"]
    ]
    chain_expected = [
        "PP3_PROJECTOR_CONTRAST_TRANSPORT_SUPPORTED_ON_XG1_EXTERNAL_GENERATOR",
        "PP3_SPECIFICITY_OVER_MAX_SEPARATION_PP5_SUPPORTED_ON_FRESH_XG1_HOLDOUT",
        "PP3_NECESSITY_OVER_MATCHED_PP5_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT",
        "PP3_RESTORATION_SUFFICIENCY_OVER_MATCHED_PP5_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT",
    ]
    d["chain_supported"] = []
    for section, expected in zip([3, 4, 5, 6], chain_expected):
        block = s.read(SYNTH, "fig2").split(
            f"### {section}. "
        )[1].split("\n### ")[0]
        conclusions = re.findall(
            r"Frozen conclusion:\s*`([^`]+)`",
            block,
        )
        if len(conclusions) != 1:
            raise ValueError(
                f"Missing/ambiguous frozen conclusion in section {section}"
            )
        d["chain_supported"].append(
            s.record(
                f"chain.{section}.supported",
                conclusions[0] == expected,
                SYNTH,
                f"section {section}; Frozen conclusion",
                "fig2",
            )
        )
    restoration_section = s.read(SYNTH, "fig3").split("### 6. ")[1].split("\n### ")[0]
    conclusions = re.findall(r"Frozen conclusion:\s*`([^`]+)`", restoration_section)
    if len(conclusions) != 1:
        raise ValueError("Missing/ambiguous historical 130M restoration conclusion")
    historical_status = s.record("130M.causal_result", conclusions[0], SYNTH,
                                 "section 6; Frozen conclusion", "fig3")
    d["causal_supported"] = {"130M": historical_status ==
        "PP3_RESTORATION_SUFFICIENCY_OVER_MATCHED_PP5_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT"}
    kernel_source = s.read(GEO_130_KERNEL, "supporting")
    kernel_matches = re.findall(r"^EXPECTED_LAYER17_MU_K2 = ([0-9.]+)$", kernel_source, re.M)
    if len(kernel_matches) != 1:
        raise ValueError("Missing/ambiguous frozen 130M kernel mean")
    d["geometry"]["130M"] = {
        "mu_k2": s.record("130M.mu_k2", float(kernel_matches[0]), GEO_130_KERNEL,
                          "EXPECTED_LAYER17_MU_K2", "supporting"),
        "lambda1": s.field("130M.lambda1", GEO_130_PLANES,
                           "geometry/positive_eigenvalues/0", "supporting"),
    }
    for flag in ["backward_executed", "scientific_model_forward_count", "task_heads_executed"]:
        if s.field("130M.geometry." + flag, GEO_130_PLANES,
                   "boundaries/" + flag, "supporting") not in (False, 0):
            raise ValueError("Historical 130M static geometry boundary mismatch")
    # N is extracted separately for each prospective cohort, not inferred from rows.
    d["chain_n"] = []
    for section in [3, 4, 5, 6]:
        block = s.read(SYNTH, "fig2").split(f"### {section}. ")[1].split("\n### ")[0]
        matches = re.findall(r"^- N: `(\d+)`", block, re.M)
        if len(matches) != 1:
            raise ValueError("Missing/ambiguous PP3 cohort N")
        d["chain_n"].append(s.record(f"chain.{section}.n", int(matches[0]), SYNTH,
                                     f"section {section}; N", "fig2"))
    d["behavior"] = s.field("behavior.mean", BEH, "mean_D_BEH", "fig2")
    d["behavior_n"] = s.field("behavior.n", BEH, "N", "fig2")
    d["metrics"] = {}
    for name, pointer in {"Pearson": "pearson_Delta_L_vs_D_BEH", "Spearman": "spearman_Delta_L_vs_D_BEH",
                          "Sign agreement": "sign_agreement/fraction"}.items():
        d["metrics"][name] = s.field("behavior." + name, PAIR + "descriptive_analysis.json",
                                       "descriptive/" + pointer, "fig2")
    rows = [json.loads(line) for line in s.read(PAIR + "pair_level_merge.jsonl", "fig2").splitlines()]
    d["pairs"] = []
    for i, row in enumerate(rows):
        d["pairs"].append(s.record(f"pair.{i}", {k: row[k] for k in ["source_pair_id", "Delta_L", "D_BEH"]},
                                  PAIR + "pair_level_merge.jsonl", f"JSONL line {i + 1}; named fields", "fig2"))
    for scale in SCALES:
        if scale == "130M":
            path, base = EXT, "mamba130m/"
            cp = base + "contra_comparison/"
            lp = base + "task_matched_pair/mean"
        else:
            path, base = LM, "scales/" + scale + "/"
            cp = base + "task_matched/contra_comparison/"
            lp = base + "task_matched/pair_endpoint/mean"
        d["planes"][scale] = s.field(scale + ".planes", path, base + "task_matched_planes", "fig4")
        for fig in ["fig1", "fig3"]:
            s.read(path, fig)
        o = {"contra": s.field(scale + ".contra", path, cp + "contra_forward_equivalent/mean", "fig4"),
             "vanilla": s.field(scale + ".vanilla", path, lp, "fig4")}
        for name in ["pearson", "spearman", "pair_sign_agreement_fraction"]:
            o[name] = s.field(scale + "." + name, path, cp + name, "fig4")
        d["objectives"][scale] = o
        # Cross-check Markdown synthesis, retaining full JSON precision for plotting.
        table_rows = [line.split("|") for line in s.read(DELTA_SYNTH, "fig4").splitlines()
                      if line.startswith("| Mamba-" + scale + " |")]
        if len(table_rows) != 1 or not math.isclose(float(table_rows[0][4]), o["contra"], rel_tol=1e-12):
            raise ValueError(f"Contra synthesis mismatch at {scale}")
        if scale != "130M":
            p = CONF[scale]
            root = "H_DOM" if scale == "370M" else "primary_test"
            d["core"][scale] = s.field(scale + ".core.mean", p, root + "/mean", "fig3")
            status_pointer = "H_DOM/reject_h0" if scale == "370M" else "core_supported"
            d["causal_supported"][scale] = s.field(scale + ".causal_supported", p, status_pointer, "fig3")
            for field, expected in zip(["selected_dominant_candidate", "response_blind_control_plane"], d["planes"][scale]):
                if s.field(scale + "." + field, p, field, "fig3") != expected:
                    raise ValueError("Selected/control identity mismatch")
            p = GEO[scale]
            for flag in ["backward_executed", "causal_response_observed", "plane_selection_performed", "training_executed"]:
                if s.field(scale + ".geometry." + flag, p, flag, "supporting") is not False:
                    raise ValueError("Response-blind geometry contract mismatch")
            d["geometry"][scale] = {
                "mu_k2": s.field(scale + ".mu_k2", p, "strong_mask/mu_k2", "supporting"),
                "lambda1": s.field(scale + ".lambda1", p, "lambda_plus_by_plane/0", "supporting"),
            }
            # Bind exact raw artifacts to the rows of the frozen native bridge.
            native_section = s.read(GEO_SYNTH, "supporting").split("## 2. Frozen native Mamba-side scale values")[1].split("## 3.")[0]
            rows_md = [line.split("|") for line in native_section.splitlines()
                       if line.startswith("| " + scale + " |")]
            if len(rows_md) != 1:
                raise ValueError("Ambiguous native geometry synthesis row")
            for column, key in [(2, "mu_k2"), (4, "lambda1")]:
                if not math.isclose(float(rows_md[0][column]), d["geometry"][scale][key], rel_tol=1e-12):
                    raise ValueError("Native geometry artifact does not match synthesis")
    # Frozen coordinate-free geometry is imported into the paper tree as
    # exact committed Git blobs from the separately frozen science result.
    cf_snapshot = json.loads(
        s.read_at(PAPER_EVIDENCE_HEAD, CF_SNAPSHOT, "supporting")
    )
    cf_result = json.loads(
        s.read_at(PAPER_EVIDENCE_HEAD, CF_RESULT, "fig3")
    )
    cf_prov = json.loads(
        s.read_at(PAPER_EVIDENCE_HEAD, CF_PROVENANCE, "supporting")
    )

    if cf_snapshot["scientific_result_commit"] != (
        "593199d8e849572fc77b1b0bc7475d3d19468d12"
    ):
        raise ValueError("Coordinate-free science commit mismatch")
    if cf_snapshot["analysis_execution_head"] != cf_result["execution_head"]:
        raise ValueError("Coordinate-free execution-head mismatch")
    for filename, source_path in [
        ("coordinate_free_geometry_result.json", CF_RESULT),
        ("provenance_manifest.json", CF_PROVENANCE),
    ]:
        expected_sha = cf_snapshot["sources"][filename]["committed_bytes_sha256"]
        actual_sha = s.sources[source_path]["frozen_bytes_sha256"]
        if actual_sha != expected_sha:
            raise ValueError(
                f"Coordinate-free snapshot SHA mismatch: {filename}"
            )
    if cf_result["scale_order"] != SCALES or cf_prov["scale_order"] != SCALES:
        raise ValueError("Coordinate-free scale order mismatch")
    if cf_prov["N"] != 300 or len(cf_prov["artifacts"]) != 10:
        raise ValueError("Coordinate-free provenance population mismatch")
    if any(row["source_pair_count"] != 300 for row in cf_prov["artifacts"]):
        raise ValueError("Coordinate-free source-pair count mismatch")
    if cf_result["scientific_model_forward_count"] != 0:
        raise ValueError("Coordinate-free result unexpectedly used model forward")
    for key in [
        "training_executed",
        "evaluation_executed",
        "tokenizer_executed",
        "backward_executed",
    ]:
        if cf_result[key] is not False:
            raise ValueError(f"Coordinate-free execution boundary violated: {key}")

    for family in ["xg2", "xg4"]:
        block = cf_result["families"][family]
        if block["scale_order"] != SCALES or len(block["pairs"]) != 10:
            raise ValueError(f"Coordinate-free family contract mismatch: {family}")
        d["coordinate_free_geometry"][family] = {
            "cka_matrix": s.record(
                f"{family}.cka_matrix",
                block["cka_matrix"],
                CF_RESULT,
                f"/families/{family}/cka_matrix",
                "fig3",
            ),
            "rsm_matrix": s.record(
                f"{family}.rsm_matrix",
                block["cosine_rsm_pearson_matrix"],
                CF_RESULT,
                f"/families/{family}/cosine_rsm_pearson_matrix",
                "fig3",
            ),
        }

    d["chronology"] = s.field("chronology", EXT, "five_scale_descriptive_completeness/prospective_status", "fig4")
    validate_values(d)
    return s, d


def validate_values(d):
    if d["scale_order"] != SCALES:
        raise ValueError("Scale order mismatch")
    if list(d["causal_supported"]) != SCALES or any(
        v is not True for v in d["causal_supported"].values()
    ):
        raise ValueError("Frozen causal support status mismatch")
    if d["chain_supported"] != [True, True, True, True]:
        raise ValueError("Frozen 130M causal-chain support mismatch")
    if list(d["planes"].values()) != [
        ["P3", "P5"],
        ["P3", "P5"],
        ["P2", "P5"],
        ["P5", "P4"],
        ["P3", "P5"],
    ]:
        raise ValueError("Plane map mismatch")

    for objective, expected in [
        ("contra", [1, 1, 1, -1, -1]),
        ("vanilla", [-1, -1, -1, 1, 1]),
    ]:
        actual = [
            1 if d["objectives"][x][objective] > 0
            else -1 if d["objectives"][x][objective] < 0
            else 0
            for x in SCALES
        ]
        if actual != expected:
            raise ValueError("Objective sign vector mismatch")

    if (
        len(d["pairs"]) != d["behavior_n"]
        or len({r["source_pair_id"] for r in d["pairs"]}) != len(d["pairs"])
    ):
        raise ValueError("Incomplete or duplicate pair rows")
    expected_ids = [f"xg1_fact_{i}" for i in range(2701, 3001)]
    if [r["source_pair_id"] for r in d["pairs"]] != expected_ids:
        raise ValueError("Pair population/order mismatch; filtering forbidden")

    if "130M" in d["core"] or list(d["geometry"]) != SCALES:
        raise ValueError("Historical 130M geometry and causal status must remain distinct")
    if d["geometry"]["130M"] != {
        "mu_k2": 0.027899337798707836,
        "lambda1": 0.8706181418918275,
    }:
        raise ValueError("Frozen historical 130M geometry mismatch")

    for family in ["xg2", "xg4"]:
        block = d["coordinate_free_geometry"][family]
        for name, low in [("cka_matrix", 0.0), ("rsm_matrix", -1.0)]:
            matrix = block[name]
            if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
                raise ValueError(f"{family} {name} shape mismatch")
            for i in range(5):
                if abs(matrix[i][i] - 1.0) > 1e-10:
                    raise ValueError(f"{family} {name} diagonal mismatch")
                for j in range(5):
                    value = matrix[i][j]
                    if not math.isfinite(value):
                        raise ValueError(f"{family} {name} nonfinite")
                    if value < low - 1e-10 or value > 1.0 + 1e-10:
                        raise ValueError(f"{family} {name} range violation")
                    if abs(value - matrix[j][i]) > 1e-12:
                        raise ValueError(f"{family} {name} asymmetry")

# Compact vector drawing primitives. Page size is the intended two-column width.
INK, BLUE, ORANGE, PALE, GRAY = "#202832", "#0072B2", "#B65A00", "#EEF3F6", "#66727C"
W = 504


class Figure:
    def __init__(self, stem, height):
        from reportlab.pdfgen import canvas
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        font_dir = Path('C:/Windows/Fonts')
        for name, filename in [('PaperSans', 'arial.ttf'), ('PaperSans-Bold', 'arialbd.ttf')]:
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
        self.color = HexColor
        self.h = height
        self.c = canvas.Canvas(str(OUT / (stem + ".pdf")), pagesize=(W, height),
                               invariant=1, pageCompression=1, initialFontName="PaperSans")
        self.c.setTitle(stem.replace("_", " "))
        self.c.setAuthor("")

    def text(self, x, y, text, size=9, bold=False, color=INK, align="left"):
        self.c.setFillColor(self.color(color))
        self.c.setFont("PaperSans-Bold" if bold else "PaperSans", size)
        method = {"left": self.c.drawString, "center": self.c.drawCentredString, "right": self.c.drawRightString}[align]
        method(x, self.h - y, str(text))

    def line(self, x1, y1, x2, y2, color=GRAY, width=.6):
        self.c.setStrokeColor(self.color(color)); self.c.setLineWidth(width)
        self.c.line(x1, self.h-y1, x2, self.h-y2)

    def rect(self, x, y, w, h, fill=PALE, stroke=None):
        self.c.setFillColor(self.color(fill))
        self.c.setStrokeColor(self.color(stroke or fill))
        self.c.rect(x, self.h-y-h, w, h, fill=1, stroke=bool(stroke))

    def dot(self, x, y, color=BLUE, radius=2.6, square=False):
        self.c.setFillColor(self.color(color)); self.c.setStrokeColor(self.color(color))
        if square:
            self.c.rect(x-radius, self.h-y-radius, 2*radius, 2*radius, fill=1, stroke=0)
        else:
            self.c.circle(x, self.h-y, radius, fill=1, stroke=0)

    def open_dot(self, x, y, color=BLUE, radius=3.4):
        self.c.setFillColor(self.color("#FFFFFF"))
        self.c.setStrokeColor(self.color(color))
        self.c.setLineWidth(1.4)
        self.c.circle(x, self.h-y, radius, fill=1, stroke=1)

    def arrow(self, x1, y1, x2, y2):
        self.line(x1,y1,x2,y2)
        a = math.atan2(y2-y1,x2-x1)
        for offset in [-.45,.45]:
            self.line(x2,y2,x2-4*math.cos(a+offset),y2-4*math.sin(a+offset))

    def panel(self, letter, title, y, x=15):
        self.text(x,y,letter,11,True,BLUE); self.text(x+17,y,title,10,True)

    def save(self):
        self.c.showPage(); self.c.save()


def chart(f, x, y, w, h, values, labels, low, high, ticks, fmt, color=BLUE, bars=False, historical_index=None):
    """Categorical raw-value chart, no fitted/connecting lines or normalization."""
    def yp(v): return y+h-(v-low)/(high-low)*h
    for tick in ticks:
        yy = yp(tick)
        f.line(x,yy,x+w,yy,INK if tick == 0 else "#DAE0E5",1 if tick == 0 else .4)
        f.text(x-5,yy+3,fmt(tick),8,align="right")
    for i,(value,label) in enumerate(zip(values,labels)):
        xx = x+w*(i+.5)/len(values)
        f.text(xx,y+h+15,label,9,align="center")
        if value is None:
            f.text(xx,y+h*.47,"n/a",8,color=GRAY,align="center")
            continue
        yy = yp(value)
        if bars:
            f.rect(xx-12,min(yy,yp(0)),24,max(abs(yy-yp(0)),.7),fill=color)
        elif i == historical_index:
            f.open_dot(xx, yy, color)
        else:
            f.dot(xx,yy,color,square=color==ORANGE)
    return yp

def _blend_hex(a, b, t):
    a = a.lstrip("#")
    b = b.lstrip("#")
    av = tuple(int(a[i:i+2], 16) for i in (0, 2, 4))
    bv = tuple(int(b[i:i+2], 16) for i in (0, 2, 4))
    out = tuple(round(x + (y - x) * t) for x, y in zip(av, bv))
    return "#" + "".join(f"{v:02X}" for v in out)


def heatmap(f, x, y, w, h, matrix, labels):
    rows = len(labels)
    cw = w / rows
    ch = h / rows
    for j, label in enumerate(labels):
        f.text(x + (j + .5) * cw, y - 7, label, 7.2, True, align="center")
    for i, label in enumerate(labels):
        f.text(x - 7, y + (i + .5) * ch + 2, label, 7.2, True, align="right")
        for j in range(rows):
            value = float(matrix[i][j])
            t = max(0.0, min(1.0, value))
            fill = _blend_hex("#F5F8FA", BLUE, t)
            f.rect(x + j * cw, y + i * ch, cw - .8, ch - .8, fill=fill)
            text_color = "#FFFFFF" if t >= .58 else INK
            f.text(
                x + (j + .5) * cw,
                y + (i + .5) * ch + 2.5,
                f"{value:.2f}",
                6.8,
                True,
                color=text_color,
                align="center",
            )


def figure1(d):
    f=Figure(STEMS[0],326)
    f.panel("A","Deep causal establishment at Mamba-130M",19)
    boxes=[("Controlled", "semantic contrasts"),("Local native-state", "localization"),("Geometry", "specificity"),
           ("Matched", "necessity"),("Restoration", "sufficiency"),("Task-margin", "consequence")]
    for i,lines in enumerate(boxes):
        x=15+i*81
        f.rect(x,32,73,45)
        for j,t in enumerate(lines): f.text(x+36.5,49+j*12,t,7.6,align="center")
        if i<5:f.arrow(x+74,54,x+80,54)
    f.text(15,94,"Native susceptibility mechanism",8.5,color=GRAY)
    f.text(489,94,"Task-functional bridge",8.5,color=GRAY,align="right")
    f.line(15,106,489,106,color="#DAE0E5")
    f.panel("B","Five-scale extension: causal role recurs; geometry reorganizes",125)
    for i,scale in enumerate(SCALES):
        x=15+i*97
        f.rect(x,138,86,46)
        f.text(x+43,155,scale,10,True,align="center")
        f.text(x+43,172," / ".join(d["planes"][scale]),9,align="center")
        if i<4:f.arrow(x+87,160,x+95,160)
    f.text(15,199,"Selected / control are scale-local ranks; equal rank numbers do not imply semantic homology.",8)
    f.line(15,211,489,211,color="#DAE0E5")
    f.panel("C","Objective-conditioned functional readout",230)
    f.rect(15,245,143,47)
    f.text(86.5,258,"Within each scale:",9,True,align="center")
    f.text(86.5,272,"same frozen native",9,align="center")
    f.text(86.5,286,"Mamba substrate",9,align="center")
    f.arrow(159,261,188,252);f.arrow(159,276,188,287)
    f.rect(190,239,299,27,fill="#E6F0F7")
    f.text(339.5,257,"Structured downstream task objective",8.5,align="center")
    f.rect(190,275,299,27,fill="#FAEEE2")
    f.text(339.5,293,"Vanilla Mamba: pretrained next-token LM objective",8.5,align="center")
    f.text(252,319,"Causal role   |   Native geometry   |   Objective-conditioned readout",9,True,align="center")
    f.save()


def figure2(d):
    f = Figure(STEMS[1], 360)
    f.panel("A", "Prospective evidence chain at 130M", 18)
    f.text(
        15,
        34,
        "Four independent holdouts test distinct causal requirements; exact statistics are in Appendix A.3.",
        8.3,
        color=GRAY,
    )
    labels = [
        ("Transport",),
        ("Specificity",),
        ("Necessity",),
        ("Restoration", "sufficiency"),
    ]
    for i, (label, n, supported) in enumerate(
        zip(labels, d["chain_n"], d["chain_supported"])
    ):
        x = 15 + 122 * i
        f.rect(x, 49, 108, 87)
        for j, line in enumerate(label):
            f.text(x + 54, 70 + j * 12, line, 9.5, True, align="center")
        f.text(
            x + 54,
            105,
            "Supported" if supported else "Not supported",
            9.2,
            True,
            BLUE,
            align="center",
        )
        f.text(x + 54, 124, f"N = {n}", 8.5, align="center")
        if i < 3:
            f.arrow(x + 110, 92, x + 120, 92)

    f.panel("B", "Behavioral restoration", 171)
    f.text(15, 192, "Restored selected component minus matched control", 8.5)
    f.text(15, 216, "Mean task-margin shift", 9)
    f.text(15, 238, f"{d['behavior']:+.5f}", 15, True, BLUE)
    f.text(15, 260, f"N = {d['behavior_n']} paired items", 8.5)
    f.text(15, 280, "Continuous margin consequence; not an accuracy-improvement claim.", 8, color=GRAY)

    f.panel("C", "Local readout tracks behavioral effect", 171, x=231)
    x, y, w, h = 275, 208, 205, 94
    xmin, xmax = -.014, .022
    ymin, ymax = -.03, .08
    xp = lambda v: x + (v - xmin) / (xmax - xmin) * w
    yp = lambda v: y + h - (v - ymin) / (ymax - ymin) * h
    for v in [-.01, 0, .01, .02]:
        f.text(xp(v), y + h + 12, f"{v:g}", 7.5, align="center")
    for v in [0, .04, .08]:
        f.text(x - 5, yp(v) + 2, f"{v:g}", 7.5, align="right")
    f.line(x, yp(0), x + w, yp(0), width=.6)
    f.line(xp(0), y, xp(0), y + h, width=.6)
    for row in d["pairs"]:
        f.dot(xp(row["Delta_L"]), yp(row["D_BEH"]), radius=1.15)
    f.text(x + w, y - 5, "Behavioral margin shift", 7.7, align="right")
    f.text(x + w / 2, y + h + 26, "Local task readout", 8.5, align="center")
    f.text(
        245,
        190,
        f"Pearson {d['metrics']['Pearson']:.3f}   "
        f"Spearman {d['metrics']['Spearman']:.3f}",
        8.5,
    )
    f.text(
        245,
        202,
        f"Sign agreement {d['metrics']['Sign agreement']:.3f}",
        8.5,
    )
    f.text(
        15,
        349,
        "All 300 pairs retained; associations are frozen descriptive results.",
        8,
        color=GRAY,
    )
    f.save()

def figure3(d):
    f = Figure(STEMS[2], 390)
    f.panel("A", "The causal role recurs across all five Mamba-1 scales", 18)
    f.text(
        15,
        36,
        "Selected/control ranks are local identities; equal rank numbers do not imply semantic homology.",
        8.2,
        color=GRAY,
    )

    xs = [170, 246, 322, 398, 474]
    f.rect(15, 47, 474, 24)
    for x, scale in zip(xs, SCALES):
        f.text(x, 64, scale, 9.5, True, align="center")

    rows = [
        ("Selected / control", [" / ".join(d["planes"][s]) for s in SCALES]),
        (
            "Causal role",
            ["Supported" if d["causal_supported"][s] else "Not supported" for s in SCALES],
        ),
    ]
    for i, (label, vals) in enumerate(rows):
        yy = 91 + i * 22
        f.text(20, yy, label, 8.5, True)
        for x, val in zip(xs, vals):
            f.text(x, yy, val, 8.2, align="center")
        f.line(15, yy + 7, 489, yy + 7, color="#DAE0E5")

    f.text(
        15,
        141,
        "Common scale-local contrast: selected restoration minus coefficient-matched control.",
        8.2,
    )
    f.text(
        15,
        153,
        "Historical endpoint labels and inferential-family details are retained in Appendix A.4.",
        8,
        color=GRAY,
    )

    f.panel("B", "XG2 centered linear CKA", 180)
    f.panel("C", "XG4 centered linear CKA", 180, x=260)

    heatmap(
        f,
        57,
        207,
        176,
        145,
        d["coordinate_free_geometry"]["xg2"]["cka_matrix"],
        SCALES,
    )
    heatmap(
        f,
        302,
        207,
        176,
        145,
        d["coordinate_free_geometry"]["xg4"]["cka_matrix"],
        SCALES,
    )

    f.text(
        15,
        374,
        "Same 300 response-blind pairs per family. CKA = 1 denotes identical centered sample geometry;",
        7.8,
    )
    f.text(
        15,
        386,
        "off-diagonal values show partial, not invariant, cross-scale geometry. Cosine-RSM: Appendix A.2.",
        7.8,
        color=GRAY,
    )
    f.save()

def figure4(d):
    f=Figure(STEMS[3],398)
    f.panel("A","Downstream task objective",19)
    f.panel("B","Vanilla next-token LM objective",19,x=260)
    f.text(51,37,"Mean Delta L (forward-equivalent)",8)
    f.text(303,37,"Mean TASK_MATCHED readout",8)
    chart(f,51,53,183,126,[d["objectives"][s]["contra"] for s in SCALES],SCALES,
          -.003,.005,[-.002,0,.002,.004],lambda v:f"{v:.3f}",bars=True)
    chart(f,303,53,183,126,[d["objectives"][s]["vanilla"] for s in SCALES],SCALES,
          -.065,.025,[-.06,-.04,-.02,0,.02],lambda v:f"{v:.2f}",color=ORANGE,bars=True)
    for start,key in [(51,"contra"),(303,"vanilla")]:
        for i,s in enumerate(SCALES):
            value=d["objectives"][s][key]
            f.text(start+183*(i+.5)/5,208,"+" if value>0 else "-",12,True,align="center")
    f.text(15,225,"Separate y-scales preserve raw units. Bars are frozen means; signs are shown explicitly.",8,color=GRAY)
    f.panel("C","Opposite aggregate signs do not imply rowwise inversion",247)
    xs=[165,238,311,384,457]
    f.rect(15,259,474,23)
    for x,s in zip(xs,SCALES):f.text(x,275,s,9,True,align="center")
    for i,(name,key) in enumerate([("Pearson", "pearson"),("Spearman", "spearman"),("Pair sign agreement", "pair_sign_agreement_fraction")]):
        yy=298+20*i;f.text(20,yy,name,9)
        for x,s in zip(xs,SCALES):
            v=d["objectives"][s][key]
            f.text(x,yy,f"{v:+.4f}" if key!="pair_sign_agreement_fraction" else f"{v:.4f}",9,align="center")
        f.line(15,yy+6,489,yy+6,color="#DAE0E5")
    f.text(
        15,
        375,
        "All five scales are shown descriptively; the 130M LM cell is the later completeness extension (Appendix A.6).",
        8.2,
        color=GRAY,
    )
    f.save()


def write_readme(s, d):
    lines = [
        "# Frozen Mamba-1 main-paper figures",
        "",
        f"Base scientific authority: `{BASE_SCIENCE_HEAD}`.",
        f"Coordinate-free geometry paper snapshot: `{PAPER_EVIDENCE_HEAD}`.",
        "",
        "Run from repository root:",
        "",
        "```powershell",
        "python paper/iclr2027/scripts/build_figures.py",
        "python -B paper/iclr2027/scripts/check_figure_values.py --rebuild",
        "```",
        "",
        "The builder reads only pinned Git blobs. No model, tokenizer, training, evaluation, forward/backward pass, or new statistical test is run.",
        "",
        "## Main-figure design",
        "",
        "- Fig. 1: conceptual separation of causal role, native geometry, and objective-conditioned use.",
        "- Fig. 2: reader-facing 130M causal chain; endpoint notation and exact inferential statistics remain in Appendix A.3.",
        "- Fig. 3: scale-local causal recurrence plus frozen coordinate-insensitive XG2/XG4 centered-linear CKA matrices.",
        "- Fig. 3 no longer uses kernel mean-square or leading-plane eigenvalue as the main evidence for non-invariance; those scalar native-geometry measurements remain supporting evidence in the appendix.",
        "- Cosine-RSM Pearson matrices are a frozen secondary coordinate-insensitive check and are reported in Appendix A.2.",
        "- Fig. 4: objective-conditioned readouts; prospective chronology is kept in the appendix rather than repeated in the visual narrative.",
        "",
        "## Frozen coordinate-free geometry",
        "",
    ]

    for family in ["xg2", "xg4"]:
        lines += [f"### {family.upper()} centered linear CKA", ""]
        lines.append("| | " + " | ".join(SCALES) + " |")
        lines.append("|---|" + "|".join(["---:" for _ in SCALES]) + "|")
        for scale, row in zip(
            SCALES,
            d["coordinate_free_geometry"][family]["cka_matrix"],
        ):
            lines.append(
                "| " + scale + " | "
                + " | ".join(f"{float(v):.6f}" for v in row)
                + " |"
            )
        lines.append("")

    lines += [
        "### Secondary cosine-RSM Pearson",
        "",
        "| Family | Scale pair | Pearson |",
        "|---|---|---:|",
    ]
    for family in ["xg2", "xg4"]:
        matrix = d["coordinate_free_geometry"][family]["rsm_matrix"]
        for i in range(len(SCALES)):
            for j in range(i + 1, len(SCALES)):
                lines.append(
                    f"| {family.upper()} | {SCALES[i]}--{SCALES[j]} | "
                    f"{matrix[i][j]:.6f} |"
                )

    lines += [
        "",
        "## Supporting historical scalar geometry",
        "",
        "| Scale | mu_k2 | lambda1 |",
        "|---|---:|---:|",
    ]
    for scale in SCALES:
        geo = d["geometry"][scale]
        lines.append(
            f"| {scale} | {geo['mu_k2']!r} | {geo['lambda1']!r} |"
        )

    lines += [
        "",
        "## Figure 4 objective means",
        "",
        "| Scale | Task forward-equivalent | Vanilla next-token |",
        "|---|---:|---:|",
    ]
    for scale, row in d["objectives"].items():
        lines.append(
            f"| {scale} | {row['contra']!r} | {row['vanilla']!r} |"
        )

    lines += ["", "## Exact source paths by figure", ""]
    for fig in ["fig1", "fig2", "fig3", "fig4"]:
        lines += [f"### {fig}", ""]
        lines += [
            f"- `{path}` @ `{meta['authority_commit']}`"
            for path, meta in s.sources.items()
            if fig in meta["figures"]
        ]
        lines.append("")

    (OUT / "README.md").write_text(
        "\n".join(lines).rstrip("\n") + "\n",
        encoding="utf-8",
        newline="\n",
    )

def main():
    import pymupdf

    s, d = load_data()
    OUT.mkdir(parents=True, exist_ok=True)

    for function in [figure1, figure2, figure3, figure4]:
        function(d)

    render_scale = 400.0 / 72.0
    render_matrix = pymupdf.Matrix(render_scale, render_scale)

    for stem in STEMS:
        pdf_path = OUT / (stem + ".pdf")
        png_path = OUT / (stem + ".png")

        with pymupdf.open(pdf_path) as doc:
            if len(doc) != 1:
                raise RuntimeError(
                    f"Expected one-page figure PDF: {pdf_path}"
                )
            pix = doc[0].get_pixmap(
                matrix=render_matrix,
                alpha=False,
            )
            pix.save(png_path)

    import reportlab

    manifest = {
        "schema_version": 2,
        "scientific_head": BASE_SCIENCE_HEAD,
        "scientific_authorities": {
            "base_evidence": BASE_SCIENCE_HEAD,
            "coordinate_free_geometry_snapshot": PAPER_EVIDENCE_HEAD,
        },
        "scale_order": SCALES,
        "sources": s.sources,
        "extracted_values": s.values,
        "plot_data": d,
        "tools": {
            "reportlab": reportlab.Version,
            "pymupdf": pymupdf.__version__,
            "png_dpi": 400,
        },
        "outputs": {
            stem + ext: hashlib.sha256((OUT / (stem + ext)).read_bytes()).hexdigest()
            for stem in STEMS
            for ext in [".pdf", ".png"]
        },
        "new_statistics": False,
        "model_execution": False,
        "notes": [
            "Figure 3 uses the frozen coordinate-free geometry result; no CKA or RSM value is recomputed by the paper builder.",
            "Historical mu_k2/lambda1 measurements remain supporting appendix evidence rather than the main non-invariance visualization.",
            "D_SUF, D_DOM, and D_CORE retain their historical inferential families in the appendix.",
            d["chronology"],
        ],
    }

    (OUT / "figure_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_readme(s, d)

    print("PASS: built 4 vector PDFs + 4 PNGs (400 dpi)")
    print("PASS: pinned base evidence + pinned coordinate-free paper snapshot")
    print("PASS: Figure 3 uses frozen XG2/XG4 centered-linear CKA matrices")
    print("PASS: no model execution and no new scientific statistics")

if __name__ == "__main__":
    main()
