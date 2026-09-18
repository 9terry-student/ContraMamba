from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


N = 300
EXPECTED_CHECKPOINT_SHA256 = (
    "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
)
EXPECTED_TOTAL_FORWARDS = 50400
EXPECTED_ITEM_FORWARDS = 160


class AnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnalysisError(message)




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checksums(run_dir: Path) -> None:
    checksum_path = run_dir / "SHA256SUMS.txt"
    require(checksum_path.is_file(), "CHECKSUM_FILE_MISSING")
    seen: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        path = run_dir / name
        require(path.is_file(), f"CHECKSUM_TARGET_MISSING:{name}")
        require(sha256_file(path) == digest, f"CHECKSUM_MISMATCH:{name}")
        seen.add(name)
    required = {
        "seed181_checkpoint_replication_items.jsonl",
        "seed181_checkpoint_replication_summary.json",
        "seed181_principal_geometry.json",
        "seed181_principal_geometry.pt",
        "seed181_xg2_geometry_items.jsonl",
        "seed181_xg4_geometry_items.jsonl",
        "seed181_xg2_alignment_delta_h.pt",
        "seed181_xg4_alignment_delta_h.pt",
    }
    require(seen == required, "CHECKSUM_SET")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), "JSONL_OBJECT")
            out.append(value)
    return out


def mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def analyze(run_dir: Path) -> dict[str, Any]:
    validate_checksums(run_dir)
    items = read_jsonl(run_dir / "seed181_checkpoint_replication_items.jsonl")
    summary = json.loads(
        (run_dir / "seed181_checkpoint_replication_summary.json").read_text(
            encoding="utf-8"
        )
    )
    geometry = json.loads(
        (run_dir / "seed181_principal_geometry.json").read_text(encoding="utf-8")
    )
    require(len(items) == N, "ITEM_COUNT")
    require(summary["primary_inference_executed"] is False, "RAW_BOUNDARY")
    require(summary["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "SUMMARY_CHECKPOINT")
    require(
        int(summary["scientific_model_forward_count_this_run"]) == EXPECTED_TOTAL_FORWARDS,
        "SUMMARY_FORWARD_BUDGET",
    )
    require(geometry["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "GEOMETRY_CHECKPOINT")
    require(geometry["selection_uses_response"] is False, "GEOMETRY_SELECTION_BOUNDARY")
    homolog = int(geometry["homolog_plane_index"])
    control = int(geometry["control_plane_index"])
    require(homolog != control and 1 <= homolog <= 5 and 1 <= control <= 5, "PLANE_IDENTITIES")
    for index, row in enumerate(items):
        require(row["source_pair_id"] == f"xg1_fact_{901 + index:03d}", f"PAIR_ORDER:{index}")
        require(row["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, f"ITEM_CHECKPOINT:{index}")
        require(int(row["homolog_plane_index"]) == homolog, f"ITEM_HOMOLOG:{index}")
        require(int(row["control_plane_index"]) == control, f"ITEM_CONTROL:{index}")
        require(
            int(row["scientific_model_forward_count_this_run"]) == EXPECTED_ITEM_FORWARDS,
            f"ITEM_FORWARD_BUDGET:{index}",
        )

    d = np.asarray([float(row["D_SUF"]) for row in items], dtype=np.float64)
    require(np.isfinite(d).all(), "D_NONFINITE")
    test = stats.ttest_1samp(d, popmean=0.0, alternative="greater")

    q_native = np.asarray([float(row["Q_restored"]) for row in items], dtype=np.float64)
    q_principal = np.asarray([float(row["Q_principal"]) for row in items], dtype=np.float64)
    residual = q_native - q_principal
    rms_q = float(np.sqrt(np.mean(q_native * q_native)))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    corr = float(np.corrcoef(q_native, q_principal)[0, 1])

    mean_q_restored = mean(items, "Q_restored")
    mean_s_homolog = mean(items, "S_homolog")
    mean_d = float(np.mean(d))
    p = float(test.pvalue)
    gates = {
        "mean_Q_restored_gt_0": mean_q_restored > 0.0,
        "mean_S_homolog_gt_0": mean_s_homolog > 0.0,
        "mean_D_SUF_gt_0": mean_d > 0.0,
        "one_sided_p_lt_0_05": p < 0.05,
    }
    supported = all(gates.values())

    return {
        "schema_version": "gen4-seed181-checkpoint-replication-analysis-v1",
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "N": N,
        "df": N - 1,
        "homolog_plane_index": geometry["homolog_plane_index"],
        "control_plane_index": geometry["control_plane_index"],
        "homolog_overlap": geometry["homolog_overlap"],
        "seed180_pp3_overlap_by_seed181_plane": geometry[
            "seed180_pp3_overlap_by_seed181_plane"
        ],
        "principal_angles_deg": geometry["principal_angles_deg"],
        "positive_eigenvalues": geometry["positive_eigenvalues"],
        "mean_Q_restored": mean_q_restored,
        "mean_Q_neutralized": mean(items, "Q_neutralized"),
        "mean_Q_control": mean(items, "Q_control"),
        "mean_S_homolog": mean_s_homolog,
        "mean_S_control": mean(items, "S_control"),
        "mean_D_SUF": mean_d,
        "D_SUF_t": float(test.statistic),
        "D_SUF_one_sided_p": p,
        "positive_causal_replication_gates": gates,
        "causal_replication_supported": supported,
        "causal_result_label": (
            "SEED181_HOMOLOG_RESTORATION_REPLICATION_SUPPORTED"
            if supported
            else "SEED181_HOMOLOG_RESTORATION_REPLICATION_NOT_ESTABLISHED"
        ),
        "finite_epsilon_mean_Q_native": float(np.mean(q_native)),
        "finite_epsilon_mean_Q_principal": float(np.mean(q_principal)),
        "finite_epsilon_mean_residual": float(np.mean(residual)),
        "finite_epsilon_rmse": rmse,
        "finite_epsilon_normalized_rmse_over_rms_Q_native": (
            rmse / rms_q if rms_q != 0.0 else None
        ),
        "finite_epsilon_mae": mae,
        "finite_epsilon_pearson": corr,
        "finite_epsilon_sign_agreement": float(
            np.mean(np.signbit(q_native) == np.signbit(q_principal))
        ),
        "principal_reconstruction_inference": "DESCRIPTIVE_ONLY_NO_BINARY_THRESHOLD",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyze(args.run_dir)
    require(not args.output.exists(), "OUTPUT_COLLISION")
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("CAUSAL_RESULT=" + result["causal_result_label"])
    print("HOMOLOG_PLANE=P" + str(result["homolog_plane_index"]))
    print("D_SUF_ONE_SIDED_P=" + repr(result["D_SUF_one_sided_p"]))


if __name__ == "__main__":
    main()
