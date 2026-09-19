#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    reason_router_gen4_mamba370m14b_stagewise_coupling_localization_fast_cuda
    as runner,
)


SCALES = runner.SCALE_ORDER
STAGES = runner.STAGE_ORDER
CELLS = runner.TARGET_CELLS
CONDITIONS = runner.CONDITIONS
PRIMITIVES = runner.PRIMITIVE_KEYS
RESULT_PASS = "PASS_STAGEWISE_BEHAVIORAL_COUPLING_LOCALIZATION_ANALYSIS"


class AnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnalysisError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL:{line_no}")
        out.append(value)
    return out


def validate_checksums(shard_dir: Path) -> None:
    sums = shard_dir / runner.CHECKSUM_FILE
    require(sums.is_file(), "SUMS_MISSING")
    seen = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        target = shard_dir / name
        require(target.is_file(), f"SUM_TARGET:{name}")
        require(sha256_file(target) == digest, f"SUM_SHA:{name}")
        seen.add(name)
    require(seen == {runner.ROW_FILE, runner.SUMMARY_FILE}, "SUM_SET")


def validate_scale(run_dir: Path, scale: str) -> list[dict[str, Any]]:
    spec = runner.bridge.scale_spec(scale)
    all_rows: list[dict[str, Any]] = []

    for shard_id, shard in enumerate(runner.SHARDS):
        shard_dir = run_dir / f"shard{shard_id}"
        validate_checksums(shard_dir)
        summary = json.loads(
            (shard_dir / runner.SUMMARY_FILE).read_text(encoding="utf-8")
        )
        require(summary["result"] == runner.RESULT_PASS, "SUMMARY_RESULT")
        require(summary["scale"] == scale, "SUMMARY_SCALE")
        require(summary["checkpoint_sha256"] == spec["checkpoint_sha256"], "SUMMARY_CKPT")
        require(summary["selected_dominant_candidate"] == spec["selected_plane"], "SUMMARY_SELECTED")
        require(summary["response_blind_control_plane"] == spec["control_plane"], "SUMMARY_CONTROL")
        require(tuple(summary["condition_order"]) == CONDITIONS, "SUMMARY_CONDITIONS")
        require(tuple(summary["stage_order"]) == STAGES, "SUMMARY_STAGES")
        require(summary["frozen_behavioral_reproduction_verified"] is True, "SUMMARY_REPRO")
        require(
            float(summary["max_frozen_behavioral_reproduction_abs_error"])
            <= runner.REPRO_TOL,
            "SUMMARY_REPRO_TOL",
        )
        require(
            float(summary["max_final_norm_lens_abs_error"])
            <= runner.LENS_TOL,
            "SUMMARY_LENS_TOL",
        )
        require(
            float(summary["max_pre_target_hidden_abs_error"])
            <= runner.PREFIX_TOL,
            "SUMMARY_PREFIX_TOL",
        )
        require(
            int(summary["scientific_full_model_forward_count_this_run"])
            == runner.FULL_FORWARDS_PER_SHARD,
            "SUMMARY_FWD",
        )
        require(
            int(summary["downstream_only_replay_count_this_run"])
            == runner.DOWNSTREAM_REPLAYS_PER_SHARD,
            "SUMMARY_REPLAY",
        )
        require(summary["p_value_count_added"] == 0, "SUMMARY_P")
        require(summary["primary_inference_executed"] is False, "SUMMARY_INFERENCE")
        require(summary["selection_reopened"] is False, "SUMMARY_SELECTION")
        require(summary["rescue_performed"] is False, "SUMMARY_RESCUE")
        require(summary["training_executed"] is False, "SUMMARY_TRAIN")
        require(summary["backward_executed"] is False, "SUMMARY_BACKWARD")

        rows = read_jsonl(shard_dir / runner.ROW_FILE)
        require(len(rows) == 300, f"ROW_COUNT:{shard_id}")
        all_rows.extend(rows)

    require(len(all_rows) == 600, "SCALE_ROW_COUNT")
    observed = set()
    for row in all_rows:
        key = (str(row["source_pair_id"]), str(row["contrast_cell_id"]))
        require(key not in observed, f"ROW_DUP:{key}")
        observed.add(key)
        require(row["scale"] == scale, f"ROW_SCALE:{key}")
        require(tuple(row["stage_order"]) == STAGES, f"ROW_STAGES:{key}")
        require(set(row["conditions"]) == set(CONDITIONS), f"ROW_CONDITIONS:{key}")
        for condition in CONDITIONS:
            value = row["conditions"][condition]
            require(
                len(value["stage_lens"]) == len(STAGES)
                and set(value["stage_lens"]) == set(STAGES),
                f"LENS_STAGES:{key}:{condition}",
            )
            require(
                int(row["scientific_full_model_forward_count"]) == 3,
                f"ROW_FWD:{key}",
            )
            require(
                int(row["downstream_only_replay_count"]) == 6,
                f"ROW_REPLAY:{key}",
            )
    expected = {
        (pair, cell)
        for pair in runner.PAIR_IDS
        for cell in CELLS
    }
    require(observed == expected, "ROW_COVERAGE")
    return all_rows


def mean(values: Sequence[float]) -> float:
    require(bool(values), "MEAN_EMPTY")
    return float(sum(float(v) for v in values) / len(values))


def median(values: Sequence[float]) -> float:
    require(bool(values), "MEDIAN_EMPTY")
    return float(statistics.median(float(v) for v in values))


def summary(values: Sequence[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    require(vals and all(math.isfinite(v) for v in vals), "SUMMARY_VALUES")
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "fraction_positive": sum(v > 0.0 for v in vals) / len(vals),
        "fraction_negative": sum(v < 0.0 for v in vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
    }


def stage_condition_metric(
    row: Mapping[str, Any],
    *,
    condition: str,
    stage: str,
    metric: str,
) -> float:
    value = float(row["conditions"][condition]["stage_lens"][stage][metric])
    require(math.isfinite(value), f"METRIC:{condition}:{stage}:{metric}")
    return value


def decompose_three(native: float, neutral: float, control: float) -> dict[str, float]:
    a_sel = native - neutral
    b_ctrl = control - neutral
    d = native - control
    require(abs((a_sel - b_ctrl) - d) <= 1.0e-12, "DECOMPOSITION_IDENTITY")
    return {
        "A_sel_native_minus_neutralized": a_sel,
        "B_ctrl_control_minus_neutralized": b_ctrl,
        "D_native_minus_control": d,
    }


def build_pair_stage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"])): row
        for row in rows
    }
    stage_results: dict[str, Any] = {}

    for stage in STAGES:
        pair_values = {
            "A": [],
            "B": [],
            "D": [],
        }
        cell_values = {
            cell: {"A": [], "B": [], "D": []}
            for cell in CELLS
        }
        primitive_values = {
            primitive: {"A": [], "B": [], "D": []}
            for primitive in PRIMITIVES
        }
        hidden_values = {
            key: {
                "attended_sequence_l2": [],
                "target_token_l2": [],
                "attended_suffix_l2": [],
                "pre_target_max_abs": [],
            }
            for key in (
                "selected_native_minus_neutralized",
                "control_minus_neutralized",
                "native_minus_control",
            )
        }

        for pair in runner.PAIR_IDS:
            condition_pair_margins = {c: [] for c in CONDITIONS}
            condition_pair_primitives = {
                p: {c: [] for c in CONDITIONS}
                for p in PRIMITIVES
            }

            for cell in CELLS:
                row = by[(pair, cell)]
                vals = {
                    c: stage_condition_metric(
                        row,
                        condition=c,
                        stage=stage,
                        metric="correct_class_logit_margin",
                    )
                    for c in CONDITIONS
                }
                dec = decompose_three(
                    vals["native"],
                    vals["dominant_neutralized"],
                    vals["dominant_control"],
                )
                cell_values[cell]["A"].append(dec["A_sel_native_minus_neutralized"])
                cell_values[cell]["B"].append(dec["B_ctrl_control_minus_neutralized"])
                cell_values[cell]["D"].append(dec["D_native_minus_control"])
                for c in CONDITIONS:
                    condition_pair_margins[c].append(vals[c])

                for primitive in PRIMITIVES:
                    for c in CONDITIONS:
                        condition_pair_primitives[primitive][c].append(
                            stage_condition_metric(
                                row,
                                condition=c,
                                stage=stage,
                                metric=primitive,
                            )
                        )

                propagation = row["hidden_propagation"][stage]
                for key in hidden_values:
                    for metric in hidden_values[key]:
                        hidden_values[key][metric].append(
                            float(propagation[key][metric])
                        )

            pair_dec = decompose_three(
                mean(condition_pair_margins["native"]),
                mean(condition_pair_margins["dominant_neutralized"]),
                mean(condition_pair_margins["dominant_control"]),
            )
            pair_values["A"].append(pair_dec["A_sel_native_minus_neutralized"])
            pair_values["B"].append(pair_dec["B_ctrl_control_minus_neutralized"])
            pair_values["D"].append(pair_dec["D_native_minus_control"])

            for primitive in PRIMITIVES:
                pdec = decompose_three(
                    mean(condition_pair_primitives[primitive]["native"]),
                    mean(condition_pair_primitives[primitive]["dominant_neutralized"]),
                    mean(condition_pair_primitives[primitive]["dominant_control"]),
                )
                primitive_values[primitive]["A"].append(
                    pdec["A_sel_native_minus_neutralized"]
                )
                primitive_values[primitive]["B"].append(
                    pdec["B_ctrl_control_minus_neutralized"]
                )
                primitive_values[primitive]["D"].append(
                    pdec["D_native_minus_control"]
                )

        stage_results[stage] = {
            "pair_margin": {
                "A_sel": summary(pair_values["A"]),
                "B_ctrl": summary(pair_values["B"]),
                "D": summary(pair_values["D"]),
            },
            "cell_margin": {
                cell: {
                    "A_sel": summary(cell_values[cell]["A"]),
                    "B_ctrl": summary(cell_values[cell]["B"]),
                    "D": summary(cell_values[cell]["D"]),
                }
                for cell in CELLS
            },
            "primitive_effects": {
                primitive: {
                    "A_sel": summary(primitive_values[primitive]["A"]),
                    "B_ctrl": summary(primitive_values[primitive]["B"]),
                    "D": summary(primitive_values[primitive]["D"]),
                }
                for primitive in PRIMITIVES
            },
            "hidden_propagation": {
                key: {
                    metric: summary(values)
                    for metric, values in metrics.items()
                }
                for key, metrics in hidden_values.items()
            },
        }

    return stage_results


def first_persistent(
    stage_results: Mapping[str, Any],
    getter,
    predicate,
) -> str | None:
    for index, stage in enumerate(STAGES):
        tail = STAGES[index:]
        if all(predicate(float(getter(stage_results[s]))) for s in tail):
            return stage
    return None


def localize(scale: str, stages: Mapping[str, Any]) -> dict[str, Any]:
    get_d = lambda x: x["pair_margin"]["D"]["mean"]
    get_a = lambda x: x["pair_margin"]["A_sel"]["mean"]
    get_b = lambda x: x["pair_margin"]["B_ctrl"]["mean"]
    get_c2d = lambda x: x["cell_margin"]["C2_NAME"]["D"]["mean"]

    if scale == "mamba14b":
        return {
            "first_persistent_D_negative":
                first_persistent(stages, get_d, lambda x: x < 0.0),
            "first_persistent_A_sel_negative":
                first_persistent(stages, get_a, lambda x: x < 0.0),
            "first_persistent_B_ctrl_positive":
                first_persistent(stages, get_b, lambda x: x > 0.0),
            "first_persistent_C2_D_negative":
                first_persistent(stages, get_c2d, lambda x: x < 0.0),
        }
    return {
        "first_persistent_D_positive":
            first_persistent(stages, get_d, lambda x: x > 0.0),
        "first_persistent_A_sel_positive":
            first_persistent(stages, get_a, lambda x: x > 0.0),
        "first_persistent_B_ctrl_negative":
            first_persistent(stages, get_b, lambda x: x < 0.0),
        "first_persistent_C2_D_positive":
            first_persistent(stages, get_c2d, lambda x: x > 0.0),
    }


def cross_scale_localization(
    s370: Mapping[str, Any],
    s14: Mapping[str, Any],
) -> dict[str, Any]:
    def persistent_opposition(cell: str | None = None) -> str | None:
        for index, stage in enumerate(STAGES):
            okay = True
            for later in STAGES[index:]:
                if cell is None:
                    d370 = float(s370[later]["pair_margin"]["D"]["mean"])
                    d14 = float(s14[later]["pair_margin"]["D"]["mean"])
                else:
                    d370 = float(s370[later]["cell_margin"][cell]["D"]["mean"])
                    d14 = float(s14[later]["cell_margin"][cell]["D"]["mean"])
                if not (d370 > 0.0 and d14 < 0.0):
                    okay = False
                    break
            if okay:
                return stage
        return None

    per_stage = {}
    for stage in STAGES:
        d370 = float(s370[stage]["pair_margin"]["D"]["mean"])
        d14 = float(s14[stage]["pair_margin"]["D"]["mean"])
        per_stage[stage] = {
            "D_370m": d370,
            "D_14b": d14,
            "D_14b_minus_370m": d14 - d370,
        }

    return {
        "first_persistent_pair_D_opposition":
            persistent_opposition(),
        "first_persistent_C2_D_opposition":
            persistent_opposition("C2_NAME"),
        "stagewise_pair_D": per_stage,
    }


def analyze(
    *,
    mamba370m_run_dir: Path,
    mamba14b_run_dir: Path,
) -> dict[str, Any]:
    rows370 = validate_scale(mamba370m_run_dir, "mamba370m")
    rows14 = validate_scale(mamba14b_run_dir, "mamba14b")

    stages370 = build_pair_stage(rows370)
    stages14 = build_pair_stage(rows14)

    result = {
        "schema_version":
            "gen4-mamba370m14b-stagewise-coupling-localization-analysis-v1",
        "result": RESULT_PASS,
        "population": "xg1_fact_4801..xg1_fact_5100",
        "stage_order": list(STAGES),
        "condition_order": list(CONDITIONS),
        "model_forward_count_added": 3600,
        "downstream_only_replay_count": 7200,
        "p_value_count_added": 0,
        "selection_reopened": False,
        "rescue_performed": False,
        "training_executed": False,
        "backward_executed": False,
        "scale_results": {
            "mamba370m": {
                "stage_results": stages370,
                "localization": localize("mamba370m", stages370),
            },
            "mamba14b": {
                "stage_results": stages14,
                "localization": localize("mamba14b", stages14),
            },
        },
        "cross_scale": cross_scale_localization(stages370, stages14),
        "interpretation_boundary": (
            "Stagewise downstream lens: intermediate residual states are "
            "read through the frozen final downstream path. A sign transition "
            "localizes when the representation becomes readable in the new "
            "orientation; it does not prove a unique causal mediator."
        ),
    }

    # Freeze the known final qualitative ordering as a validation constraint.
    final370 = stages370[runner.FINAL_NORM_STAGE]["pair_margin"]
    final14 = stages14[runner.FINAL_NORM_STAGE]["pair_margin"]
    require(float(final370["A_sel"]["mean"]) > 0.0, "FINAL_370_A_SIGN")
    require(float(final370["B_ctrl"]["mean"]) < 0.0, "FINAL_370_B_SIGN")
    require(float(final370["D"]["mean"]) > 0.0, "FINAL_370_D_SIGN")
    require(float(final14["A_sel"]["mean"]) < 0.0, "FINAL_14_A_SIGN")
    require(float(final14["B_ctrl"]["mean"]) > 0.0, "FINAL_14_B_SIGN")
    require(float(final14["D"]["mean"]) < 0.0, "FINAL_14_D_SIGN")

    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba370m-run-dir", type=Path, required=True)
    parser.add_argument("--mamba14b-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        mamba370m_run_dir=args.mamba370m_run_dir,
        mamba14b_run_dir=args.mamba14b_run_dir,
    )
    require(not args.output.exists(), "OUTPUT_COLLISION")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("RESULT=" + result["result"])
    print("P_VALUE_COUNT_ADDED=0")
    for scale in SCALES:
        final = result["scale_results"][scale]["stage_results"][
            runner.FINAL_NORM_STAGE
        ]["pair_margin"]
        loc = result["scale_results"][scale]["localization"]
        print(
            f"{scale.upper()}_FINAL_A_SEL="
            + format(float(final["A_sel"]["mean"]), ".17g")
        )
        print(
            f"{scale.upper()}_FINAL_B_CTRL="
            + format(float(final["B_ctrl"]["mean"]), ".17g")
        )
        print(
            f"{scale.upper()}_FINAL_D="
            + format(float(final["D"]["mean"]), ".17g")
        )
        print(
            f"{scale.upper()}_LOCALIZATION="
            + json.dumps(loc, sort_keys=True)
        )
    print(
        "CROSS_SCALE_LOCALIZATION="
        + json.dumps(result["cross_scale"], sort_keys=True)
    )
    print("SELECTION_REOPENED=False")
    print("RESCUE_PERFORMED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
