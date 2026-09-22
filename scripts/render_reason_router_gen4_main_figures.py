"""Static, source-bound Gen4 manuscript renderer. No scientific execution.

Only the manifest's 22 frozen sources are read. JSON statistics and Markdown
summary cells are copied, never re-estimated. Stored rows support descriptive
plots and arithmetic validation only. Requires numpy and matplotlib to render.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reports/reason_router_gen4_main_figure_plot_data_manifest_v1.json"
SCHEMA = "gen4-main-figure-plot-data-manifest-v1"
# Canonical JSON identity at authority HEAD 4ffe98c174f9ee1b48454c9bbfaecc8449eb37f4.
# Whitespace/line endings and manifest location may vary; bindings may not.
MANIFEST_DIGEST = "11ab58bfe1ff92e606b3af1ab85a2810153e8af74980cf0ba35cbd7b90fd7c1a"
OUTPUT_NAMES = tuple(
    f"figure{i}.{ext}" for i in range(1, 6) for ext in ("pdf", "png")
) + ("main_table_1.csv", "render_manifest.json")
READOUT_FROZEN = {
    "mean_370M": 0.0005089563518854174,
    "mean_1.4B": -0.0011954475058862238,
    "mean_R": 0.001704403857771641,
    "sd_R": 0.0027407135373192773,
    "t_statistic": 10.771333954019786,
    "p_value": 2.2238330789610916e-23,
}
BLUE, ORANGE, GRAY = "#21618c", "#b45127", "#65717b"


class ContractError(ValueError):
    """Frozen source or output safety contract was not satisfied."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def at(value, *keys):
    for key in keys:
        require(isinstance(value, dict) and key in value,
                f"Required source field missing: {'/'.join(map(str, keys))}")
        value = value[key]
    return value


def number(value):
    require(not isinstance(value, bool), "Boolean is not a scientific value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Required finite numeric value: {value!r}") from exc
    require(math.isfinite(result), "Nonfinite scientific value")
    return result


def close(actual, expected, label):
    # Relative tolerance avoids accepting large errors in tiny susceptibility
    # values or p-values; the absolute floor is only for arithmetic near zero.
    require(math.isclose(number(actual), number(expected), rel_tol=1e-12,
                         abs_tol=1e-30), f"Frozen arithmetic mismatch: {label}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_manifest(manifest):
    require(at(manifest, "schema_version") == SCHEMA, "Manifest schema_version differs")
    require(at(manifest, "scientific_execution") == "CLOSED",
            "Manifest scientific_execution must be CLOSED")
    for key in ("new_model_execution", "new_statistical_tests", "new_p_values"):
        require(at(manifest, key) is False, f"Manifest {key} must be false")
    digest = sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    require(digest == MANIFEST_DIGEST, "Manifest frozen bindings/content differ")


class Sources:
    def __init__(self, manifest_path=DEFAULT_MANIFEST, root=ROOT):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        validate_manifest(self.manifest)
        names = [s for f in self.manifest["figures"] for p in f["panels"]
                 for s in p["sources"]] + self.manifest["main_table_1"]["sources"]
        self.raw, self.paths = {}, {}
        for name in sorted(set(names)):
            path = (self.root / name).resolve()
            require(path.is_relative_to(self.root), f"Source escapes repository: {name}")
            require(path.is_file(), f"Missing bound source: {name}")
            self.paths[name] = path
            self.raw[name] = path.read_bytes()

    def panel_names(self, figure, panel):
        f = self.manifest["figures"][figure - 1]
        matches = [p for p in f["panels"] if p["panel"] == panel]
        require(len(matches) == 1, f"Missing/ambiguous panel {figure}{panel}")
        return matches[0]["sources"]

    def read(self, name):
        require(name in self.raw, f"Unbound source: {name}")
        text = self.raw[name].decode("utf-8")
        if name.endswith(".jsonl"):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text) if name.endswith(".json") else text

    def panel(self, figure, panel, index=0):
        return self.read(self.panel_names(figure, panel)[index])

    def hashes(self):
        return {name: sha(data) for name, data in self.raw.items()}

    def verify_unchanged(self):
        require(self.manifest_path.read_bytes() == self.manifest_bytes,
                "Manifest mutated during rendering")
        for name, path in self.paths.items():
            require(path.read_bytes() == self.raw[name], f"Source mutated: {name}")


def validate_pairs(rows, fields, first, pair_index=False):
    require(isinstance(rows, list) and len(rows) == 300, "Expected pair-level N=300")
    ids = []
    for row in rows:
        identity = at(row, "source_pair_id")
        require(isinstance(identity, str), "source_pair_id must be text")
        ids.append(identity)
        for field in fields:
            number(at(row, field))
        if pair_index:
            index = at(row, "pair_index")
            require(type(index) is int and 0 <= index < 300, "Invalid pair_index")
            require(identity == f"xg1_fact_{first + index}", "pair_index/identity mismatch")
    expected = {f"xg1_fact_{i:03d}" for i in range(first, first + 300)}
    require(len(set(ids)) == 300 and set(ids) == expected,
            "Duplicate or unexpected frozen source_pair_id")
    return sorted(rows, key=lambda row: int(row["source_pair_id"].rsplit("_", 1)[1]))


def validate_readout(rows, analysis):
    rows = validate_pairs(rows, ("Delta_L_370M", "Delta_L_1.4B", "R"), 4801)
    primary = at(analysis, "primary_test")
    require(at(primary, "n") == 300 and at(primary, "df") == 299,
            "Frozen readout N/df differs")
    require(at(primary, "test") == "paired_one_sample_student_t"
            and at(primary, "alternative") == "greater"
            and at(primary, "p_value_count") == 1, "Frozen readout test differs")
    for key in ("mean_R", "sd_R", "t_statistic", "p_value"):
        close(at(primary, key), READOUT_FROZEN[key], key)
    for scale, key in (("370M", "mamba370m_Delta_L"), ("1.4B", "mamba14b_Delta_L")):
        mean = at(analysis, "descriptive", key, "mean")
        close(mean, READOUT_FROZEN[f"mean_{scale}"], key)
        close(np.mean([r[f"Delta_L_{scale}"] for r in rows]), mean, key)
        close(at(analysis, "sign_gates", f"mean_Delta_L_{scale}"), mean, key)
    for row in rows:
        close(row["Delta_L_370M"] - row["Delta_L_1.4B"], row["R"], "R row")
    close(np.mean([r["R"] for r in rows]), primary["mean_R"], "mean R")
    close(np.std([r["R"] for r in rows], ddof=1), primary["sd_R"], "SD R")
    require(at(analysis, "sign_reversal_supported") is True, "Readout conclusion differs")
    return rows


def validate_transport(rows, analysis):
    fields = ("D_ADJ", "D_CAN_descriptive_only", "D_TRANSPORT", "G")
    rows = validate_pairs(rows, fields, 5101, pair_index=True)
    require(at(analysis, "population", "pair_count") == 300
            and at(analysis, "primary", "n") == 300, "Frozen transport N differs")
    primary = at(analysis, "primary")
    require(at(primary, "df") == 299 and at(primary, "p_value_count") == 1
            and at(primary, "alternative") == "greater"
            and at(primary, "endpoint") == "G=D_TRANSPORT-D_ADJ"
            and at(primary, "test") == "one_sample_student_t_on_paired_difference_G",
            "Frozen transport test differs")
    for key in ("p_value", "t_statistic", "mean_G", "sd_G_sample"):
        number(at(primary, key))
    require(at(analysis, "mandatory_sign_gate", "pass") is False,
            "Positive restoration gate differs")
    for row in rows:
        close(row["D_TRANSPORT"] - row["D_ADJ"], row["G"], "G row")
    for field, key in zip(fields, ("historical_D_ADJ", "historical_D_CAN", "D_TRANSPORT", "G")):
        close(np.mean([r[field] for r in rows]),
              at(analysis, "descriptive_only", key, "mean"), field)
    close(np.mean([r["G"] for r in rows]), at(analysis, "primary", "mean_G"), "mean G")
    close(np.std([r["G"] for r in rows], ddof=1),
          at(analysis, "primary", "sd_G_sample"), "SD G")
    return rows


def extract(text, pattern):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    require(len(matches) == 1, f"Missing/ambiguous frozen Markdown field: {pattern}")
    return number(matches[0])


def bullet(text, label):
    return extract(text, rf"^- {re.escape(label)}\s*(?:=|:)\s*`([^`]+)`")


def section(text, start, end):
    require(text.count(start) == 1 and text.count(end) == 1,
            f"Missing/ambiguous frozen Markdown section: {start}")
    return text.split(start, 1)[1].split(end, 1)[0]


def table(text, header, column, ranks):
    require(text.count(header) == 1, f"Missing/ambiguous table: {header}")
    block = text.split(header, 1)[1].strip().split("\n\n", 1)[0]
    result = {}
    for line in block.splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if cells[0] in ranks:
            require(cells[0] not in result and len(cells) > column, "Invalid summary table row")
            result[cells[0]] = number(cells[column])
    require(set(result) == set(ranks), f"Required table ranks missing: {header}")
    return [result[rank] for rank in ranks]


def prepare(sources):
    """Validate and extract every plotted value before creating any output."""
    d = {}
    d["xg1"] = validate_pairs(sources.panel(2, "A"), ("C_PP3",), 1)
    require(at(sources.panel(2, "A", 1), "source_pair_count") == 300, "XG1 N differs")
    spec = section(sources.panel(2, "B"), "### 4. Prospective specificity", "## Combined scientific claim")
    d["specificity"] = {k: bullet(spec, k) for k in
                        ("mean C_PP3", "mean C_PP5", "mean D_SPEC", "SD D_SPEC", "t(299)", "one-sided p")}
    for name, panel, endpoint in (("necessity", "C", "D_NEC"), ("restoration", "D", "D_SUF")):
        text = sources.panel(2, panel)
        d[name] = {"mean": bullet(text, f"mean {endpoint}"),
                   "sd": bullet(text, f"SD {endpoint}"),
                   "t": bullet(text, "t"), "p": bullet(text, "one-sided p")}
    epsilon = sources.panel(2, "E")
    eps = [at(epsilon, "epsilon_set", "reference")] + at(epsilon, "epsilon_set", "new")
    require(eps == [0.025, 0.0125, 0.00625], "Frozen epsilon set differs")
    d["epsilon"] = [(e, [number(at(epsilon, "spectral_profiles", str(e),
                                  "mean_plane_contributions", f"P{k}")) for k in range(1, 6)]) for e in eps]
    core = sources.panel(3, "A-D")
    d["core_means"] = [extract(core, r"`mean\(D_DOM\) = ([^`]+)`"),
                       extract(core, r"`mean\(D_CORE\) = ([^`]+)`")]
    d["core_p"] = [extract(core, r"Holm-adjusted `p = ([^`]+)`"),
                   extract(core, r"one-sided primary `p = ([^`]+)`")]
    residual370 = section(core, "## 3. Residual organization at 370M", "## 4. Residual organization at 1.4B")
    residual14 = section(core, "## 4. Residual organization at 1.4B", "\n## 5.")
    d["residual"] = []
    for text, ranks, mass_header, mass_col in (
        (residual370, ["P1", "P2", "P4", "P5"], "| local rank | energy share `w_k` |", 1),
        (residual14, ["P1", "P2", "P3", "P4"], "| local rank | mean `C_k` | share `w_k` |", 2),
    ):
        d["residual"].append({"ranks": ranks, "mass": table(text, mass_header, mass_col, ranks),
                              "signed": table(text, "| local rank | `v_k` |", 1, ranks)})
    adjacent = sources.panel(4, "A")
    d["adjacent_means"] = table(adjacent,
        "| Endpoint | Mean | SD | 95% t-CI | Cohen dz | Positive fraction |", 1, ["D_CAN", "D_ADJ"])
    d["adjacent_p"] = extract(adjacent, r"^- p = ([\deE.+-]+)\.$")
    geometry = sources.panel(4, "B")
    require(at(geometry, "pooled", "row_count") == 600
            and at(geometry, "pooled", "rank_counts", "2") == 600, "Transport rank/row count differs")
    d["angles"] = [number(at(geometry, "pooled", f"principal_angle_degrees_{i}", "mean")) for i in (1, 2)]
    d["overlap"] = number(at(geometry, "pooled", "projector_overlap", "mean"))
    d["transport_analysis"] = sources.panel(4, "C", 1)
    d["transport"] = validate_transport(sources.panel(4, "C"), d["transport_analysis"])
    d["readout_analysis"] = sources.panel(5, "B", 1)
    d["readout"] = validate_readout(sources.panel(5, "B"), d["readout_analysis"])
    d["context_means"] = table(sources.panel(5, "A"),
        "| Scale | Frozen population | Selected / control | Mean `Delta_L_owned` | Mean `Delta_L_forward = 2x owned` | Positive fraction | Inferential status |",
        3, ["Mamba-130M", "Mamba-370M", "Mamba-1.4B"])
    for actual, key in zip(d["context_means"][1:], ("mean_370M", "mean_1.4B")):
        close(actual, READOUT_FROZEN[key], "context readout")
    bridge = sources.panel(5, "C")
    require(at(bridge, "scientific_conclusion") == "SCALE_SPECIFIC_BEHAVIORAL_BRIDGE_ONLY",
            "Behavioral bridge conclusion differs")
    d["bridge"] = [at(bridge, "scale_results", scale, "primary_test") for scale in ("mamba370m", "mamba14b")]
    for entry in d["bridge"]:
        require(number(at(entry, "n")) == 300, "Bridge N differs")
        for key in ("mean", "p_holm"):
            number(at(entry, key))
    stage = sources.panel(5, "D")
    d["stages"] = at(stage, "stage_order")
    require(d["stages"] == ["pre_block_35"] + [f"post_block_{i}" for i in range(35, 48)]
            + ["post_final_norm"], "Frozen stage order differs")
    d["stage_means"] = [[number(at(stage, "cross_scale", "stagewise_pair_D", st, f"D_{scale}"))
                         for st in d["stages"]] for scale in ("370m", "14b")]
    d["stage_C2"] = at(stage, "cross_scale", "first_persistent_C2_D_opposition")
    d["stage_pair"] = at(stage, "cross_scale", "first_persistent_pair_D_opposition")
    require(d["stage_C2"] == "post_block_35" and d["stage_pair"] == "post_block_47",
            "Frozen localization differs")
    d["table"] = prepare_table(sources)
    return d


def prepare_table(sources):
    names = sources.manifest["main_table_1"]["sources"]
    external, negative, steering = [sources.read(name) for name in names]
    rows = []
    boundary = "Gold evidence only; no retrieval claim; no benchmark-superiority claim"
    for scale, key in (("130M", "mamba130m"), ("370M", "mamba370m"), ("1.4B", None)):
        primary = at(negative, "primary") if key is None else at(external, "scale_results", key, "primary")
        require(at(primary, "n") == 462, "External transfer N differs")
        p_key = "p_one_sided_less" if key is None else "p_holm"
        if key is None:
            require(at(negative, "primary_supported") is False, "Negative transfer gate differs")
        else:
            require(at(primary, "supported_holm") is True, "Positive transfer gate differs")
        rows.append({"evidence": "AVeriTeC gold-evidence transfer", "checkpoint": scale,
                     "N": 462, "primary_outcome": f"D_EXT={number(at(primary, 'mean')):+.2e}",
                     "effect_or_utility": f"dz={number(at(primary, 'cohen_dz')):+.3f}",
                     "frozen_p": f"{number(at(primary, p_key)):.4f}",
                     "test": "one-sided less" if key is None else "Holm (two-test family)",
                     "conclusion": "Negative transfer not established" if key is None else "Positive transfer supported",
                     "boundary": boundary, "source": names[1] if key is None else names[0]})
    primary = at(steering, "primary_endpoint")
    require(at(steering, "result") == "AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED",
            "Steering result differs")
    require(at(primary, "C_corrections") == 0 and at(primary, "D_damages") == 0
            and at(primary, "estimable") is False, "Steering counts/estimability differ")
    require(at(steering, "population", "N") == 2799, "Steering N differs")
    close(at(steering, "descriptive_diagnostics", "net_accuracy_change"), 0, "steering utility")
    rows.append({"evidence": "AVeriTeC fixed-mirror steering", "checkpoint": "370M", "N": 2799,
                 "primary_outcome": f"corrections={primary['C_corrections']}; damages={primary['D_damages']}",
                 "effect_or_utility": "net accuracy change=0", "frozen_p": "not estimable",
                 "test": "exact utility test; zero discordant pairs",
                 "conclusion": "Fixed-mirror useful steering not established", "boundary": boundary,
                 "source": names[2]})
    return rows


def note(ax, text, y=0.97):
    ax.text(0.03, y, text, transform=ax.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})


def panel(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#b9c1c7", lw=0.8, zorder=0)
    ax.tick_params(labelsize=9)


def summaries(ax, labels, values, colors=None, unit=1):
    values = np.asarray(values, dtype=float) / unit
    ax.bar(range(len(values)), values, color=colors or BLUE, width=0.55)
    ax.set_xticks(range(len(values)), labels)
    span = max(np.max(np.abs(values)), 1e-12)
    ax.set_ylim(min(0, min(values)) - 0.25 * span, max(0, max(values)) + 0.55 * span)
    for i, value in enumerate(values):
        ax.annotate(f"{value:+.3g}", (i, value), xytext=(0, 5 if value >= 0 else -5),
                    textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)


def build_figures(plt, d):
    """Artists consume only validated data; no tests, fits, or synthetic rows."""
    figs = []
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    ax.set_title("Figure 1 | What does causal-role recurrence preserve?", loc="left", fontsize=16, pad=20)
    ax.text(0.02, 0.92, "A  Checkpoint-local reconstruction and prospective intervention", fontsize=12, weight="bold")
    boxes = ["Frozen recurrent-state\ngeometry", "Independent local\nplane reconstruction", "Freeze candidate +\nresponse-blind control", "Fresh-cohort\ncausal intervention"]
    for i, label in enumerate(boxes):
        x = 0.12 + i * 0.25
        ax.text(x, 0.78, label, ha="center", va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=0.65", "facecolor": "#eaf1f6", "edgecolor": BLUE})
        if i < 3:
            ax.annotate("", xy=(x + 0.14, 0.78), xytext=(x + 0.10, 0.78),
                        arrowprops={"arrowstyle": "->", "color": GRAY})
    ax.text(0.02, 0.60, "B  Dominant ranks are local to each checkpoint", fontsize=12, weight="bold")
    for x, label in zip((0.17, 0.50, 0.83), ("Mamba-130M\nP3", "Mamba-370M\nP3", "Mamba-1.4B\nP5")):
        ax.text(x, 0.47, label, ha="center", fontsize=13, color=BLUE)
    ax.text(0.50, 0.35, "Equal rank labels do not imply semantic identity.", ha="center", fontsize=10)
    ax.text(0.02, 0.23, "C  Across the tested checkpoints", fontsize=12, weight="bold")
    ax.text(0.50, 0.11, "Causal-role recurrence does not guarantee geometric invariance\nor downstream functional invariance.",
            ha="center", va="center", fontsize=14, color=ORANGE, linespacing=1.6)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.06)
    figs.append(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    a, b, c, e, f, legend = axes.flat
    panel(a, "A  External-generator transport", "Stored C_PP3 (x 1e-8)")
    a.scatter(np.arange(1, 301), [r["C_PP3"] / 1e-8 for r in d["xg1"]], s=7, color=BLUE, alpha=0.55)
    a.set_xlabel("Frozen XG1 item (1-300)")
    note(a, "Stored item-level observations; N=300")
    spec = d["specificity"]
    panel(b, "B  Specificity", "Mean susceptibility contrast (x 1e-8)")
    summaries(b, ["PP3", "PP5 control"], [spec["mean C_PP3"], spec["mean C_PP5"]], [BLUE, GRAY], 1e-8)
    note(b, f"Frozen one-sided p={spec['one-sided p']:.2e}")
    for ax, key, letter, label in ((c, "necessity", "C", "D_NEC"), (e, "restoration", "D", "D_SUF")):
        s = d[key]
        panel(ax, f"{letter}  {'Matched-control necessity' if key == 'necessity' else 'Restoration sufficiency'}",
              "Frozen mean +/- SD (x 1e-8)")
        ax.errorbar([0], [s["mean"] / 1e-8], yerr=[s["sd"] / 1e-8], fmt="o", color=BLUE, capsize=6)
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-1, (s["mean"] + s["sd"]) / 1e-8 * 1.35)
        ax.set_xticks([0], [label])
        note(ax, f"N=300; frozen one-sided p={s['p']:.2e}\nMean={s['mean']:.2e}; SD={s['sd']:.2e}")
    panel(f, "E  Finite-epsilon robustness", "Mean plane contribution (x 1e-8)")
    for (eps, values), marker in zip(d["epsilon"], ("o", "s", "^")):
        f.plot(range(1, 6), np.asarray(values) / 1e-8, marker=marker, label=f"epsilon={eps:g}", lw=1)
    f.set_xticks(range(1, 6), [f"P{k}" for k in range(1, 6)])
    f.legend(fontsize=8, frameon=False)
    legend.axis("off")
    legend.text(0, 0.95, "Evidence boundaries", fontsize=12, weight="bold", va="top")
    legend.text(0, 0.80, "B-D: frozen summaries only;\nno synthetic item distributions.\n\nPP5: response-blind max-separation\ncontrol. Necessity uses a matched\ncoefficient-transfer control.\n\nRestoration is relative to matched\nPP5 replacement. Error bars are SD,\nnot confidence intervals.\n\nE: descriptive finite-epsilon check;\nno zero-limit fit or extrapolation.", va="top", fontsize=10, linespacing=1.35)
    fig.suptitle("Figure 2 | Prospective validation of the 130M causal component", fontsize=16, x=0.06, ha="left")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.08, wspace=0.40, hspace=0.58)
    figs.append(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    a, b, c, e = axes.flat
    panel(a, "A  CORE-STABLE | confirmatory", "Frozen mean core contrast (x 1e-8)")
    summaries(a, ["370M: P3 / P5", "1.4B: P5 / P4"], d["core_means"], [BLUE, BLUE], 1e-8)
    note(a, f"370M Holm p={d['core_p'][0]:.2e}\n1.4B one-sided p={d['core_p'][1]:.2e}")
    for ax, key, title in ((b, "mass", "B  RESIDUAL-PLASTIC | coefficient mass"),
                            (c, "signed", "C  RESIDUAL-PLASTIC | signed effects")):
        panel(ax, title, "Residual energy share" if key == "mass" else "Normalized signed residual effect")
        labels, vals, colors = [], [], []
        for scale, item, color in zip(("370M", "1.4B"), d["residual"], (BLUE, ORANGE)):
            labels.extend([f"{scale}\n{r}" for r in item["ranks"]])
            vals.extend(item[key]); colors.extend([color] * 4)
        ax.bar(range(8), vals, color=colors)
        ax.set_xticks(range(8), labels, fontsize=8)
        ax.axvline(3.5, color=GRAY, ls=":", lw=1)
    e.axis("off")
    e.text(0, 0.95, "D  Causal-role recurrence, local reorganization", va="top", fontsize=12, weight="bold")
    e.text(0, 0.80, "Blue: 370M checkpoint-local ranks\nOrange: 1.4B checkpoint-local ranks\n\nResidual comparisons are descriptive.\nNo cross-scale significance test.\nSame-numbered planes are not semantic matches.\n\n130M context: residual mass concentrates in P1.\n370M: P5; 1.4B: primarily P2, then P3.\n\nHistorical 370M joint residual criterion:\nNOT ESTABLISHED; not rescued by core recurrence.", va="top", fontsize=10, linespacing=1.4)
    fig.suptitle("Figure 3 | CORE-STABLE / RESIDUAL-PLASTIC", fontsize=16, x=0.07, ha="left")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.08, wspace=0.27, hspace=0.42)
    figs.append(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    a, b, c = axes
    panel(a, "A  Fixed adjacent-site specificity", "Frozen mean contrast (x 1e-9)")
    summaries(a, ["Canonical\n(33,34,35)", "Adjacent +1\n(34,35,36)"], d["adjacent_means"], [BLUE, GRAY], 1e-9)
    note(a, f"N=300; frozen one-sided p={d['adjacent_p']:.2e}")
    panel(b, "B  Response-free geometry transport", "Mean principal angle (degrees)")
    summaries(b, ["Angle 1", "Angle 2"], d["angles"], [BLUE, ORANGE])
    b.set_ylim(0, 120)
    b.set_yticks([0, 30, 60, 90])
    note(b, f"Rank two: 600/600 XG2/XG4 rows\nMean projector overlap={d['overlap']:.5f}")
    panel(c, "C  Transported functional response", "Stored paired response (x 1e-9)")
    keys = ["D_CAN_descriptive_only", "D_ADJ", "D_TRANSPORT"]
    for row in d["transport"]:
        c.plot(range(3), [row[k] / 1e-9 for k in keys], color=GRAY, alpha=0.07, lw=0.5)
    analysis = d["transport_analysis"]
    means = [at(analysis, "descriptive_only", k, "mean") / 1e-9
             for k in ("historical_D_CAN", "historical_D_ADJ", "D_TRANSPORT")]
    c.plot(range(3), means, "o-", color=ORANGE, lw=2)
    c.set_xticks(range(3), ["D_CAN*", "D_ADJ", "D_TRANSPORT"], fontsize=8)
    primary = analysis["primary"]
    note(c, f"Mean G={primary['mean_G']:.2e}\nFrozen one-sided p={primary['p_value']:.2e}")
    c.set_ylim(top=35)
    fig.suptitle("Figure 4 | Cross-block geometric reorientation", fontsize=16, x=0.06, ha="left")
    fig.text(0.06, 0.07, "Transport shifts the adjacent response toward canonical, but positive restoration is not established.\n*Canonical response in C is descriptive only. A tests one fixed adjacent site; no global layer-optimum claim.", fontsize=11, linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.24, wspace=0.34)
    figs.append(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    a, b, c, e = axes.flat
    panel(a, "A  Three-checkpoint context", "Mean Delta_L_owned (x 1e-3)")
    summaries(a, ["130M*", "370M", "1.4B"], d["context_means"], [GRAY, BLUE, ORANGE], 1e-3)
    note(a, "*Separate frozen population; no fitted scale trend")
    panel(b, "B  Matched readout reversal | primary", "Delta_L_owned (x 1e-3)")
    for row in d["readout"]:
        b.plot([0, 1], [row["Delta_L_370M"] / 1e-3, row["Delta_L_1.4B"] / 1e-3], color=GRAY, alpha=0.12, lw=0.5)
    b.plot([0, 1], np.asarray(d["context_means"][1:]) / 1e-3, "o-", color=ORANGE, lw=2)
    b.set_xticks([0, 1], ["370M (P3 / P5)", "1.4B (P5 / P4)"])
    p = d["readout_analysis"]["primary_test"]
    b.set_ylim(top=13)
    note(b, f"N={p['n']}; mean R={p['mean_R']:+.2e}; SD={p['sd_R']:.2e}\nFrozen t({p['df']})={p['t_statistic']:.2f}; one-sided p={p['p_value']:.2e}")
    panel(c, "C  Direct behavioral bridge", "Frozen mean D_BEH (x 1e-3)")
    summaries(c, ["370M", "1.4B"], [number(p["mean"]) for p in d["bridge"]], [BLUE, ORANGE], 1e-3)
    note(c, f"Positive-bridge Holm p: {number(d['bridge'][0]['p_holm']):.2e}, {number(d['bridge'][1]['p_holm']):.2f}\nScale-specific behavioral bridge only")
    panel(e, "D  Stagewise localization | descriptive", "Frozen pair-average D (x 1e-3)")
    for vals, label, color in zip(d["stage_means"], ("370M", "1.4B"), (BLUE, ORANGE)):
        e.plot(range(len(vals)), np.asarray(vals) / 1e-3, "o-", ms=3, label=label, color=color)
    e.set_xticks([0, 1, 5, 9, 13, 14], ["pre 35", "post 35", "39", "43", "47", "final\nnorm"])
    e.legend(loc="lower left", frameon=False, fontsize=9)
    e.set_ylim(-2.8, 3.0)
    note(e, "Persistent C2 opposition: post_block_35\nPersistent pair opposition: post_block_47")
    fig.suptitle("Figure 5 | Downstream readout reversal despite causal-role recurrence", fontsize=16, x=0.06, ha="left")
    fig.text(0.06, 0.025, "Delta_L_forward = 2 x Delta_L_owned (presentation only; frozen inference unchanged).\nInference concerns 300 matched items for two fixed checkpoints, not model seeds. Stagewise localization identifies no unique mediator.", fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.14, wspace=0.24, hspace=0.43)
    figs.append(fig)
    return figs


def validate_output_dir(output_dir, sources):
    out = Path(output_dir).resolve()
    protected = list(sources.paths.values()) + [sources.manifest_path]
    require(out not in protected, "Output directory would overwrite a frozen source")
    # Refuse *all* existing targets, including hardlinks/symlinks to unbound
    # scientific files. No existing artifact needs to be replaced for a render.
    for name in OUTPUT_NAMES:
        target = out / name
        require(target.resolve() not in protected, f"Output would overwrite a frozen source: {name}")
        require(not target.exists() and not target.is_symlink(), f"Output already exists: {target}")
    return out


def render(manifest_path, output_dir, root=ROOT):
    sources = Sources(manifest_path, root)
    d = prepare(sources)
    out = validate_output_dir(output_dir, sources)
    # Lazy import lets schema/arithmetic contracts run without a plotting stack.
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    style = {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
             "text.usetex": False, "axes.formatter.use_mathtext": False,
             "savefig.dpi": 180, "figure.facecolor": "white"}
    with plt.rc_context(style):
        figures = build_figures(plt, d)
        out.mkdir(parents=True, exist_ok=True)
        try:
            for i, fig in enumerate(figures, 1):
                for ext in ("pdf", "png"):
                    metadata = {"Creator": "ContraMamba static renderer", "CreationDate": None,
                                "ModDate": None} if ext == "pdf" else {"Software": "ContraMamba static renderer"}
                    # Exclusive creation also prevents races after safety validation.
                    with (out / f"figure{i}.{ext}").open("xb") as handle:
                        fig.savefig(handle, format=ext, metadata=metadata)
        finally:
            for fig in figures:
                plt.close(fig)
    with (out / "main_table_1.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(d["table"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(d["table"])
    sources.verify_unchanged()
    provenance = {
        "schema_version": "gen4-main-figure-render-manifest-v1",
        "scientific_execution": "CLOSED",
        "manifest_sha256": sha(sources.manifest_bytes),
        "manifest_canonical_sha256": MANIFEST_DIGEST,
        "evidence_head": sources.manifest["evidence_head"],
        "implementation_authority_head": "4ffe98c174f9ee1b48454c9bbfaecc8449eb37f4",
        "renderer_sha256": sha(Path(__file__).read_bytes()),
        "source_sha256": sources.hashes(),
        "outputs": list(OUTPUT_NAMES),
        "output_sha256": {name: sha((out / name).read_bytes()) for name in OUTPUT_NAMES[:-1]},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "new_model_execution": False, "new_statistical_tests": False, "new_p_values": False,
        "source_artifacts_byte_identical": True,
        "readout_frozen_primary_test": d["readout_analysis"]["primary_test"],
        "transport_frozen_primary": d["transport_analysis"]["primary"],
        "figure_sources": sources.manifest["figures"],
        "table_sources": sources.manifest["main_table_1"],
        "presentation": "Stored rows or frozen summaries only; no synthetic samples; no new inference.",
        "source_note": "Steering uses result and observed counts, not its stale claim_boundary.positive_claim template.",
    }
    with (out / "render_manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        render(args.manifest, args.output_dir)
    except (ContractError, OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    print("PASS: rendered " + ", ".join(OUTPUT_NAMES))


if __name__ == "__main__":
    main()
