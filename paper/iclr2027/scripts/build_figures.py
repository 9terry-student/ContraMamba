"""Render frozen Mamba-1 evidence; no models, estimators, or statistical tests.

Run from the repository root. Requires reportlab and Poppler's pdftoppm.
All source bytes are read from the pinned Git commit, never an untracked run.
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
HEAD = "3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69"
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
        if git("rev-parse", "HEAD").decode().strip() != HEAD:
            raise RuntimeError("Scientific endpoint HEAD mismatch")
        self.sources, self.values, self.cache = {}, {}, {}

    def read(self, path, figure):
        if path not in self.cache:
            raw = git("show", f"{HEAD}:{path}")  # missing/untracked sources fail closed
            local = (ROOT / path).read_bytes()
            if local.replace(b"\r\n", b"\n") != raw.replace(b"\r\n", b"\n"):
                raise RuntimeError(f"Worktree source differs from frozen artifact: {path}")
            self.cache[path] = raw.decode("utf-8")
            self.sources[path] = {"git_blob": git("rev-parse", f"{HEAD}:{path}").decode().strip(),
                                  "frozen_bytes_sha256": hashlib.sha256(raw).hexdigest(),
                                  "figures": []}
        if figure not in self.sources[path]["figures"]:
            self.sources[path]["figures"].append(figure)
        return self.cache[path]

    def record(self, key, value, path, locator, figure):
        if key in self.values:
            raise ValueError(f"Duplicate data key: {key}")
        self.values[key] = {"value": value, "source": path, "locator": locator, "figure": figure,
                            "transformation": "none; display rounding only"}
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
        value = self.record(key, float(matches[0]), SYNTH,
                            f"section {section}; exact bullet 'mean {endpoint}'", figure)
        self.values[key]["source_decimal_lexeme"] = matches[0]
        return value


def load_data():
    s = Sources()
    for fig, paths in {"fig1": [SYNTH, CORE_SYNTH, DELTA_SYNTH, LM_MD, EXT],
                       "fig2": [SYNTH], "fig3": [SYNTH, CORE_SYNTH, GEO_SYNTH],
                       "fig4": [DELTA_SYNTH, LM_MD]}.items():
        for path in paths:
            s.read(path, fig)
    d = {"scale_order": SCALES, "planes": {}, "core": {}, "geometry": {}, "objectives": {}}
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
    restoration_section = s.read(SYNTH, "fig3").split("### 6. ")[1].split("\n### ")[0]
    conclusions = re.findall(r"Frozen conclusion:\s*`([^`]+)`", restoration_section)
    if len(conclusions) != 1:
        raise ValueError("Missing/ambiguous historical 130M restoration conclusion")
    historical_status = s.record("130M.causal_result", conclusions[0], SYNTH,
                                 "section 6; Frozen conclusion", "fig3")
    d["causal_supported"] = {"130M": historical_status ==
        "PP3_RESTORATION_SUFFICIENCY_OVER_MATCHED_PP5_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT"}
    kernel_source = s.read(GEO_130_KERNEL, "fig3")
    kernel_matches = re.findall(r"^EXPECTED_LAYER17_MU_K2 = ([0-9.]+)$", kernel_source, re.M)
    if len(kernel_matches) != 1:
        raise ValueError("Missing/ambiguous frozen 130M kernel mean")
    d["geometry"]["130M"] = {
        "mu_k2": s.record("130M.mu_k2", float(kernel_matches[0]), GEO_130_KERNEL,
                          "EXPECTED_LAYER17_MU_K2", "fig3"),
        "lambda1": s.field("130M.lambda1", GEO_130_PLANES,
                           "geometry/positive_eigenvalues/0", "fig3"),
    }
    for flag in ["backward_executed", "scientific_model_forward_count", "task_heads_executed"]:
        if s.field("130M.geometry." + flag, GEO_130_PLANES,
                   "boundaries/" + flag, "fig3") not in (False, 0):
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
                if s.field(scale + ".geometry." + flag, p, flag, "fig3") is not False:
                    raise ValueError("Response-blind geometry contract mismatch")
            d["geometry"][scale] = {
                "mu_k2": s.field(scale + ".mu_k2", p, "strong_mask/mu_k2", "fig3"),
                "lambda1": s.field(scale + ".lambda1", p, "lambda_plus_by_plane/0", "fig3"),
            }
            # Bind exact raw artifacts to the rows of the frozen native bridge.
            native_section = s.read(GEO_SYNTH, "fig3").split("## 2. Frozen native Mamba-side scale values")[1].split("## 3.")[0]
            rows_md = [line.split("|") for line in native_section.splitlines()
                       if line.startswith("| " + scale + " |")]
            if len(rows_md) != 1:
                raise ValueError("Ambiguous native geometry synthesis row")
            for column, key in [(2, "mu_k2"), (4, "lambda1")]:
                if not math.isclose(float(rows_md[0][column]), d["geometry"][scale][key], rel_tol=1e-12):
                    raise ValueError("Native geometry artifact does not match synthesis")
    d["chronology"] = s.field("chronology", EXT, "five_scale_descriptive_completeness/prospective_status", "fig4")
    validate_values(d)
    return s, d


def validate_values(d):
    if d["scale_order"] != SCALES:
        raise ValueError("Scale order mismatch")
    if list(d["causal_supported"]) != SCALES or any(v is not True for v in d["causal_supported"].values()):
        raise ValueError("Frozen causal support status mismatch")
    if list(d["planes"].values()) != [["P3", "P5"], ["P3", "P5"], ["P2", "P5"], ["P5", "P4"], ["P3", "P5"]]:
        raise ValueError("Plane map mismatch")
    for objective, expected in [("contra", [1, 1, 1, -1, -1]), ("vanilla", [-1, -1, -1, 1, 1])]:
        actual = [1 if d["objectives"][x][objective] > 0 else -1 if d["objectives"][x][objective] < 0 else 0 for x in SCALES]
        if actual != expected:
            raise ValueError("Objective sign vector mismatch")
    if len(d["pairs"]) != d["behavior_n"] or len({r["source_pair_id"] for r in d["pairs"]}) != len(d["pairs"]):
        raise ValueError("Incomplete or duplicate pair rows")
    expected_ids = [f"xg1_fact_{i}" for i in range(2701, 3001)]
    if [r["source_pair_id"] for r in d["pairs"]] != expected_ids:
        raise ValueError("Pair population/order mismatch; filtering forbidden")
    if "130M" in d["core"] or list(d["geometry"]) != SCALES:
        raise ValueError("Historical 130M geometry and causal status must remain distinct")
    if d["geometry"]["130M"] != {"mu_k2": 0.027899337798707836,
                                   "lambda1": 0.8706181418918275}:
        raise ValueError("Frozen historical 130M geometry mismatch")


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
    f=Figure(STEMS[1],376)
    f.panel("A","Prospective PP3 evidence chain",18)
    f.text(15,34,"Distinct estimands and XG1 holdouts; means are not magnitude-comparable.",8.5,color=GRAY)
    labels=[("Transport",), ("Specificity",), ("Necessity",), ("Restoration", "sufficiency")]
    for i,(label,endpoint,mean,n) in enumerate(zip(labels,["C_PP3", "D_SPEC", "D_NEC", "D_SUF"],
                                                 d["chain_display_means"],d["chain_n"])):
        x=15+122*i
        f.rect(x,48,108,112)
        for j,line in enumerate(label): f.text(x+54,67+j*12,line,9.5,True,align="center")
        f.text(x+54,98,endpoint,10,align="center")
        f.text(x+54,122,mean,8.1,align="center")
        f.text(x+54,146,f"N = {n}",9,align="center")
        if i<3:f.arrow(x+110,104,x+120,104)
    f.panel("B","Behavioral restoration",194)
    f.text(15,214,"Restored PP3 minus PP5 replacement",8.5)
    f.text(15,235,"Mean D_BEH",9)
    f.text(15,255,f"{d['behavior']:+.8f}",15,True,BLUE)
    f.text(15,275,"Task-margin contrast; seed 181",8.5)
    f.text(15,291,f"N = {d['behavior_n']} paired items",8.5)
    f.text(15,308,"Behavioral consequence is measured",8,color=GRAY)
    f.text(15,320,"by margin, not accuracy improvement.",8,color=GRAY)
    f.panel("C","Readout / behavior coupling",194,x=231)
    x,y,w,h=275,230,205,94
    xmin,xmax=-.014,.022; ymin,ymax=-.03,.08
    xp=lambda v:x+(v-xmin)/(xmax-xmin)*w
    yp=lambda v:y+h-(v-ymin)/(ymax-ymin)*h
    for v in [-.01,0,.01,.02]:
        f.text(xp(v),y+h+12,f"{v:g}",7.5,align="center")
    for v in [0,.04,.08]:
        f.text(x-5,yp(v)+2,f"{v:g}",7.5,align="right")
    f.line(x,yp(0),x+w,yp(0),width=.6); f.line(xp(0),y,xp(0),y+h,width=.6)
    for r in d["pairs"]: f.dot(xp(r["Delta_L"]),yp(r["D_BEH"]),radius=1.15)
    f.text(x+w,y-5,"D_BEH",8,align="right")
    f.text(x+w/2,y+h+26,"Delta L (owned readout)",8.5,align="center")
    f.text(245,211,f"Pearson {d['metrics']['Pearson']:.5f}   Spearman {d['metrics']['Spearman']:.5f}",8.5)
    f.text(245,223,f"Sign agreement {d['metrics']['Sign agreement']:.5f}",8.5)
    f.text(15,366,"All points retained. Association values are frozen descriptive results; no fit or new test.",8,color=GRAY)
    f.save()


def figure3(d):
    f=Figure(STEMS[2],376)
    f.panel("A","Scale-local causal roles across all five Mamba-1 scales",19)
    f.text(15,37,"Selected / control ranks are local identities, not semantic homologues.",8.5,color=GRAY)
    xs=[160,234,308,382,456]
    f.rect(15,47,474,27)
    for x,scale in zip(xs,SCALES):f.text(x,65,scale,10,True,align="center")
    f.text(15,88,"Common contrast: Q(selected restored) - Q(coefficient-matched control)",8.4,True)
    rows=[("Selected / control",[" / ".join(d["planes"][s]) for s in SCALES]),
          ("Historical label",["D_SUF", "D_DOM", "D_CORE", "D_CORE", "D_CORE"]),
          ("Result",["Supported" if d["causal_supported"][s] else "Not supported" for s in SCALES])]
    for i,(label,vals) in enumerate(rows):
        yy=108+i*20;f.text(20,yy,label,8.5,True)
        for x,val in zip(xs,vals): f.text(x,yy,val,8.3,align="center")
        f.line(15,yy+7,489,yy+7,color="#DAE0E5")
    f.text(15,165,"130M restoration sufficiency; 370M two-test Holm family (residual criterion failed).",8)
    f.text(15,177,"790M-2.8B: later core confirmations. Distinct protocols; no magnitude scaling.",8)
    f.panel("B","Native kernel reorganization",198)
    f.panel("C","Native geometry reorganization",198,x=260)
    f.text(51,215,"Kernel mean squared weight (mu_k2)",8)
    f.text(301,215,"Leading plane lambda1",8)
    chart(f,51,230,183,97,[d["geometry"][s]["mu_k2"] for s in SCALES],SCALES,
          0,.03,[0,.01,.02,.03],lambda v:f"{v:.2f}",historical_index=0)
    chart(f,301,230,183,97,[d["geometry"][s]["lambda1"] for s in SCALES],SCALES,
          .85,1,[.85,.90,.95,1],lambda v:f"{v:.2f}",color=ORANGE,historical_index=0)
    f.text(15,360,"Open markers: historical response-blind 130M; filled markers: later matched 370M-2.8B bridge.",8)
    f.text(15,372,"Categorical scale positions; raw values, no scaling fit. Geometry axis begins at 0.85.",8,color=GRAY)
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
    f.text(15,364,"130M vanilla LM: later frozen completeness extension (post-primary).",8.5,True)
    f.text(15,379,"370M-2.8B: original prospective four-scale control. All five shown descriptively.",8.5)
    plane_labels = ", ".join("/".join(d["planes"][scale]) for scale in SCALES)
    f.text(15,392,f"Selected/control: {plane_labels}. No new inferential testing.",8,color=GRAY)
    f.save()


def write_readme(s, d):
    lines=["# Frozen Mamba-1 main-paper figures", "", f"Scientific authority: `{HEAD}`.", "",
           "Run from repository root (Python with reportlab; Poppler pdftoppm on PATH):", "",
           "```powershell", "python paper/iclr2027/scripts/build_figures.py",
           "python -B paper/iclr2027/scripts/check_figure_values.py --rebuild", "git diff --check", "```", "",
           "The optional --rebuild check regenerates the artifacts and requires identical bytes for all eight outputs, the manifest, and this README.",
           "Use a Python environment with reportlab, pypdf, and Pillow installed.",
           "PDFs are 7 inches wide, entirely vector, using embedded Arial fonts. PNGs are rendered from the PDFs at 400 dpi.",
           "The builder reads exact Git blobs at the pinned endpoint and rejects worktree source drift (CRLF conversion permitted).",
           "It fails on missing paths/fields, identity mismatches, changed signs, or missing renderer. No run globbing or fallback.",
           "PDF timestamps/IDs are deterministic. Reproducible byte hashes require the same ReportLab/Poppler versions.", "",
           "## Scientific boundaries and source decisions", "",
           "- Fig. 1 is a conceptual overview of the supplied narrative; arrows between scales denote study order, not parameter-count causation.",
           "- Fig. 1 specifies a task-margin consequence and a shared frozen substrate within each scale, not one literal substrate across model sizes.",
           "- Fig. 2 reads historical transport, necessity, and sufficiency means directly from uniquely anchored Markdown bullets: the raw JSON summaries contain no aggregate means. Specificity uses the frozen inference JSON. No means or correlations are recomputed.",
           "- Fig. 2 Panel A uses equal-size, equal-color evidence cards with full frozen decimal means and cohort sizes. Distinct estimands/holdouts are not encoded as comparable magnitudes. Panels B/C are unchanged.",
           "- Fig. 2 uses all 300 pair-level rows in frozen order; Delta L remains the owned readout used for the frozen correlation. Behavioral restoration is a task-margin endpoint, not an accuracy gain.",
           "- Fig. 3 uses historical D_SUF at 130M, D_DOM at 370M, and D_CORE at 790M/1.4B/2.8B; these are labeled separately. The failed historical 370M joint residual criterion is preserved.",
           "- Fig. 3 Panel A states the common selected-restored minus coefficient-matched-control contrast and retains historical D_SUF/D_DOM/D_CORE labels and separate confirmation protocols. Frozen support statuses are source-bound; no statistical decision is recomputed. Raw means remain in this provenance inventory only.",
           "- Fig. 3 Panels B/C add the frozen historical 130M mu_k2 and lambda1 with open markers. The later homogeneous response-blind bridge covers 370M-2.8B only. No 130M spectral summary scalar or D_CORE value is reconstructed.",
           "- Fig. 4 uses exact machine-readable comparator means, checked against the five-scale Delta-L synthesis (rounding tolerance 1e-12). Task and vanilla retain separate raw y-scales and zero lines.",
           "- 130M vanilla LM is a later post-primary completeness extension. Original prospective control: 370M, 790M, 1.4B, 2.8B.",
           "- No smoothing, fitted curve, normalization, new statistic, p-value, model loading, tokenizer, training, evaluation, forward or backward pass is used.",
           "- Every numerical data value, including scatter coordinates and pair identities, has a source locator in figure_data_manifest.json. Axis ticks, display sizes, and rounding are presentation choices.",
           "- Figures use embedded Arial/ASCII text; rendered output was visually inspected.", "",
           "## Exact extracted values", "", "### Figure 2", "",
           "| Endpoint | Frozen mean |", "|---|---:|"]
    for name,v in zip(["C_PP3", "D_SPEC", "D_NEC", "D_SUF"],d["chain"]): lines.append(f"| {name} | {v!r} |")
    lines += [f"| D_BEH | {d['behavior']!r} |", ""]
    for k,v in d["metrics"].items():lines.append(f"- {k}: `{v!r}`")
    lines += ["", "### Figure 3", "", "| Scale | Selected/control | Causal mean | mu_k2 | lambda1 |", "|---|---|---:|---:|---:|"]
    for scale in SCALES:
        geo=d["geometry"].get(scale,{})
        lines.append(f"| {scale} | {'/'.join(d['planes'][scale])} | {d['chain'][3] if scale=='130M' else d['core'][scale]!r} | {geo.get('mu_k2','n/a')} | {geo.get('lambda1','n/a')} |")
    lines += ["", "### Figure 4", "", "| Scale | Task forward-equivalent | Vanilla TASK_MATCHED | Pearson | Spearman | Sign agreement |", "|---|---:|---:|---:|---:|---:|"]
    for scale,o in d["objectives"].items():
        lines.append("| " + scale + " | " + " | ".join(repr(o[k]) for k in ["contra","vanilla","pearson","spearman","pair_sign_agreement_fraction"]) + " |")
    lines += ["", "## Exact source paths by figure", ""]
    for fig in ["fig1","fig2","fig3","fig4"]:
        lines += ["### " + fig, ""]
        lines += [f"- `{p}`" for p,meta in s.sources.items() if fig in meta["figures"]]
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    renderer = shutil.which("pdftoppm")
    if not renderer:
        raise RuntimeError("pdftoppm is required; no silent PNG fallback")
    s,d=load_data()
    OUT.mkdir(parents=True,exist_ok=True)
    for function in [figure1,figure2,figure3,figure4]: function(d)
    for stem in STEMS:
        subprocess.run([renderer,"-singlefile","-r","400","-png",str(OUT/(stem+".pdf")),str(OUT/stem)],check=True)
    import reportlab
    manifest={"schema_version":1,"scientific_head":HEAD,"scale_order":SCALES,
              "sources":s.sources,"extracted_values":s.values,"plot_data":d,
              "tools":{"reportlab":reportlab.Version,"pdftoppm":subprocess.run([renderer,"-v"],capture_output=True,text=True,check=True).stderr.splitlines()[0]},
              "outputs":{stem+ext:hashlib.sha256((OUT/(stem+ext)).read_bytes()).hexdigest() for stem in STEMS for ext in [".pdf",".png"]},
              "new_statistics":False,"model_execution":False,
              "notes":["Historical 130M kernel mean and leading-plane eigenvalue use the same mathematical quantities but precede the homogeneous four-scale bridge.",
                       "D_SUF, D_DOM, and D_CORE share a causal contrast but retain distinct protocol labels and inferential families.",d["chronology"]]}
    (OUT/"figure_data_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write_readme(s,d)
    print("PASS: built 4 vector PDFs + 4 PNGs (400 dpi)")
    print("PASS: source binding, all five scales, plane identities, task +,+,+,-,-; vanilla -,-,-,+,+")
    print("Frozen behavior metrics:",json.dumps(d["metrics"]))
    print(f"Manifest: {len(s.sources)} pinned source files; {len(s.values)} located values/rows")


if __name__ == "__main__":
    main()
