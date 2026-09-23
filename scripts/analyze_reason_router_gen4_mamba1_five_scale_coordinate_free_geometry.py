from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]

SCALE_ORDER = ("130M", "370M", "790M", "1.4B", "2.8B")
FAMILIES = ("xg2", "xg4")
N = 300

CONFIG: dict[str, dict[str, Any]] = {
    "130M": {
        "historical_provenance": True,
        "shape": [300, 395],
        "root": (
            "reports/reason_router_gen4_seed181_checkpoint_replication_runs/"
            "g4k-seed181-checkpoint-replication-8e96fd1-retry1"
        ),
        "summary": "seed181_checkpoint_replication_summary.json",
        "geometry": "seed181_principal_geometry.json",
        "xg2": {
            "tensor": "seed181_xg2_alignment_delta_h.pt",
            "tensor_sha256": "058e576c16b75d7e0944508f5a0ef41e7eb9a6dfbe1995703892cd535bcd7902",
            "items": "seed181_xg2_geometry_items.jsonl",
            "items_sha256": "79018820e1c91fd72aeaa95e827cd81aa226d03ceda6a80be3af903391400eaa",
        },
        "xg4": {
            "tensor": "seed181_xg4_alignment_delta_h.pt",
            "tensor_sha256": "90b55f36f1c70a758d5c79b16a0420770c778e240617ed8709de2080999102c6",
            "items": "seed181_xg4_geometry_items.jsonl",
            "items_sha256": "381fdcf6054cb6642b291d51bf7a2abdec0c5e482f39a18364275f6ae7ec4fd9",
        },
    },
    "370M": {
        "historical_provenance": False,
        "shape": [300, 650],
        "root": (
            "reports/reason_router_gen4_mamba370m_geometry_preparation_runs/"
            "g4k-mamba370-geometry-xg2xg4-2gpu-d8e71ad-retry2"
        ),
        "manifest": "artifact_manifest.json",
        "summary": "geometry_summary.json",
        "xg2": {
            "tensor": "xg2_alignment_delta_h.pt",
            "tensor_sha256": "5c68a9e3fe30473b4e0b2bb941c4c930236ee2fa4715a7ff693a92ed23c5a96b",
            "items": "xg2_geometry_items.jsonl",
            "items_sha256": "13bd27fe6c606988b49c5a46f108555427f8f6a222ca2559ec62656e8766cc30",
        },
        "xg4": {
            "tensor": "xg4_alignment_delta_h.pt",
            "tensor_sha256": "66b2d3634517e647e3027ffb92bd289c61b117507eaed34f54b715ba444d753c",
            "items": "xg4_geometry_items.jsonl",
            "items_sha256": "6964bb201304a379169f1512ea874c7ee65562fb4b099ec04d92a2d36778249f",
        },
    },
    "790M": {
        "historical_provenance": False,
        "shape": [300, 975],
        "root": (
            "reports/reason_router_gen4_mamba790m_geometry_preparation_runs/"
            "g4k-mamba790m-geometry-xg2xg4-2gpu-774983b-retry2"
        ),
        "manifest": "artifact_manifest.json",
        "summary": "geometry_summary.json",
        "xg2": {
            "tensor": "xg2_alignment_delta_h.pt",
            "tensor_sha256": "0d96495388e5bd0595f129328c82e151700e01e20758e2fda68f1bf5366eadc0",
            "items": "xg2_geometry_items.jsonl",
            "items_sha256": "eff15318010b51828d5dd240a08ed759f115474ef25dc60a1bfc5f5caf349f08",
        },
        "xg4": {
            "tensor": "xg4_alignment_delta_h.pt",
            "tensor_sha256": "1b71d7893f25cc7ab7d0ea39d8b644dcbb6a6bcaa0908183c28b44c17fbd2448",
            "items": "xg4_geometry_items.jsonl",
            "items_sha256": "e91210d113aa404068815c52f3a8279557528a747baf28355b33f1b441982834",
        },
    },
    "1.4B": {
        "historical_provenance": False,
        "shape": [300, 829],
        "root": (
            "reports/reason_router_gen4_mamba14b_geometry_preparation_runs/"
            "g4k-mamba14b-geometry-xg2xg4-2gpu-c758d5e-retry1"
        ),
        "manifest": "artifact_manifest.json",
        "summary": "geometry_summary.json",
        "xg2": {
            "tensor": "xg2_alignment_delta_h.pt",
            "tensor_sha256": "936c0d456859781e2bcd6f4481bdd483db9d567c5a274c8ce86f201b18cf1bd7",
            "items": "xg2_geometry_items.jsonl",
            "items_sha256": "0ba3e37fd19f99688ea4d334b51301755a7a95473b90bc87a98180bd40dbb57c",
        },
        "xg4": {
            "tensor": "xg4_alignment_delta_h.pt",
            "tensor_sha256": "d38cb80221560da8d0fc0d6f66ec906e2b38c42c3ef48f94854f551a1e4956a8",
            "items": "xg4_geometry_items.jsonl",
            "items_sha256": "b09e9ddf05a3b18bc3ba8ea8e090e6504cd56953d4e5d38d41cde1544c6e36f8",
        },
    },
    "2.8B": {
        "historical_provenance": False,
        "shape": [300, 1003],
        "root": (
            "reports/reason_router_gen4_mamba28b_geometry_preparation_runs/"
            "g4k-mamba28b-geometry-xg2xg4-2gpu-7744d94"
        ),
        "manifest": "artifact_manifest.json",
        "summary": "geometry_summary.json",
        "xg2": {
            "tensor": "xg2_alignment_delta_h.pt",
            "tensor_sha256": "54cd6b6048ab9ea2f735720ddf10d52147f15b5ca6a8ad7d970fef3e9bd17d9c",
            "items": "xg2_geometry_items.jsonl",
            "items_sha256": "6f1595154cbcafe0e8003d9bb25a22e99cc6e1d78a2f97a779b06ada4409e797",
        },
        "xg4": {
            "tensor": "xg4_alignment_delta_h.pt",
            "tensor_sha256": "ed7b74f64cfd3b3a1f84f579e86648a1ba708cf887edffa004226ea1b6a77092",
            "items": "xg4_geometry_items.jsonl",
            "items_sha256": "96749bb7dab59fe3ef248c5d87ec5dc39a05df2429c8886a7103fe61df086a3f",
        },
    },
}


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError("GIT_FAILURE:" + " ".join(args)) from exc


def git_blob_bytes(path: Path) -> bytes:
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise AnalysisError(f"GIT_BLOB_PATH_OUTSIDE_REPO:{path}") from exc

    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError(f"GIT_BLOB_READ_FAILURE:{relative}") from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl_bytes(
    raw: bytes,
    label: Path,
) -> list[dict[str, Any]]:
    rows = []
    text = raw.decode("utf-8-sig")
    for line in text.splitlines():
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL_NON_OBJECT:{label}")
            rows.append(value)
    return rows


def load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    require(torch.is_tensor(value), f"NOT_TENSOR:{path}")
    return value.detach().cpu().to(torch.float64).contiguous()


def expected_pair_ids(family: str) -> list[str]:
    return [f"{family}_fact_{i:03d}" for i in range(301, 601)]


def verify_later_manifest(root: Path, cfg: dict[str, Any]) -> None:
    manifest = read_json(root / str(cfg["manifest"]))
    require(manifest["training_executed"] is False, f"TRAINING_FLAG:{root}")
    require(manifest["backward_executed"] is False, f"BACKWARD_FLAG:{root}")
    require(manifest["response_observed"] is False, f"RESPONSE_FLAG:{root}")
    require(manifest["xg1_accessed"] is False, f"XG1_FLAG:{root}")

    summary = read_json(root / str(cfg["summary"]))
    require(
        summary["causal_response_observed"] is False,
        f"SUMMARY_RESPONSE_FLAG:{root}",
    )
    require(
        summary["control_selection_performed"] is False,
        f"CONTROL_SELECTION_FLAG:{root}",
    )

    hashes = manifest["output_file_sha256"]
    for family in FAMILIES:
        fam = cfg[family]
        require(
            hashes[fam["tensor"]] == fam["tensor_sha256"],
            f"MANIFEST_TENSOR_SHA:{root}:{family}",
        )
        require(
            hashes[fam["items"]] == fam["items_sha256"],
            f"MANIFEST_ITEMS_SHA:{root}:{family}",
        )


def verify_130m_boundary(root: Path, cfg: dict[str, Any]) -> None:
    summary = read_json(root / str(cfg["summary"]))
    geometry = read_json(root / str(cfg["geometry"]))

    require(summary["training_executed"] is False, "130M_TRAINING_FLAG")
    require(summary["backward_executed"] is False, "130M_BACKWARD_FLAG")
    require(summary["logits_read"] is False, "130M_LOGITS_FLAG")
    require(summary["task_heads_executed"] is False, "130M_TASK_HEAD_FLAG")
    require(
        geometry["selection_uses_response"] is False,
        "130M_SELECTION_RESPONSE_FLAG",
    )


def verify_and_load(
    scale: str,
    family: str,
) -> tuple[torch.Tensor, dict[str, Any], list[str]]:
    cfg = CONFIG[scale]
    fam = cfg[family]
    root = ROOT / str(cfg["root"])

    require(root.is_dir(), f"MISSING_ROOT:{scale}:{root}")

    tensor_path = root / str(fam["tensor"])
    items_path = root / str(fam["items"])

    require(tensor_path.is_file(), f"MISSING_TENSOR:{scale}:{family}")
    require(items_path.is_file(), f"MISSING_ITEMS:{scale}:{family}")

    tensor_sha = sha256_file(tensor_path)

    # Text artifacts may be checked out with platform-specific newline
    # conversion on Windows. Verify and consume the exact committed blob
    # bytes so the frozen SHA256 identifies the bytes actually analyzed.
    items_raw = git_blob_bytes(items_path)
    items_sha = sha256_bytes(items_raw)

    require(
        tensor_sha == fam["tensor_sha256"],
        f"TENSOR_SHA_MISMATCH:{scale}:{family}:{tensor_sha}",
    )
    require(
        items_sha == fam["items_sha256"],
        f"ITEMS_SHA_MISMATCH:{scale}:{family}:{items_sha}",
    )

    tensor = load_tensor(tensor_path)
    expected_shape = tuple(int(v) for v in cfg["shape"])

    require(
        tuple(tensor.shape) == expected_shape,
        f"TENSOR_SHAPE:{scale}:{family}:{tuple(tensor.shape)}",
    )
    require(
        bool(torch.isfinite(tensor).all().item()),
        f"TENSOR_NONFINITE:{scale}:{family}",
    )

    norms = torch.linalg.vector_norm(tensor, ord=2, dim=1)
    require(
        bool(torch.isfinite(norms).all().item()),
        f"NORM_NONFINITE:{scale}:{family}",
    )
    require(
        bool(torch.all(norms > 0).item()),
        f"ZERO_ROW_NORM:{scale}:{family}",
    )

    items = read_jsonl_bytes(items_raw, items_path)
    require(len(items) == N, f"ITEM_COUNT:{scale}:{family}:{len(items)}")

    pair_ids = [str(row["source_pair_id"]) for row in items]
    require(
        pair_ids == expected_pair_ids(family),
        f"PAIR_ORDER:{scale}:{family}",
    )

    for index, row in enumerate(items):
        require(
            str(row["family_key"]) == family,
            f"FAMILY_KEY:{scale}:{family}:{index}",
        )
        if "alignment_plan_index" in row:
            require(
                int(row["alignment_plan_index"]) == index,
                f"PLAN_INDEX:{scale}:{family}:{index}",
            )
        if "response_observed" in row:
            require(
                row["response_observed"] is False,
                f"ITEM_RESPONSE:{scale}:{family}:{index}",
            )
        if "xg1_accessed" in row:
            require(
                row["xg1_accessed"] is False,
                f"ITEM_XG1:{scale}:{family}:{index}",
            )

    provenance = {
        "scale": scale,
        "family": family,
        "historical_provenance": bool(cfg["historical_provenance"]),
        "tensor_path": tensor_path.relative_to(ROOT).as_posix(),
        "tensor_sha256": tensor_sha,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype_loaded": str(tensor.dtype),
        "items_path": items_path.relative_to(ROOT).as_posix(),
        "items_sha256": items_sha,
        "source_pair_first": pair_ids[0],
        "source_pair_last": pair_ids[-1],
        "source_pair_count": len(pair_ids),
    }

    return tensor, provenance, pair_ids


def row_normalize(value: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(value, ord=2, dim=1)
    return (value / norms[:, None]).contiguous()


def gram(value: torch.Tensor) -> torch.Tensor:
    result = value @ value.T
    require(bool(torch.isfinite(result).all().item()), "GRAM_NONFINITE")
    return result.contiguous()


def center_gram(value: torch.Tensor) -> torch.Tensor:
    row_mean = value.mean(dim=1, keepdim=True)
    col_mean = value.mean(dim=0, keepdim=True)
    total_mean = value.mean()
    result = value - row_mean - col_mean + total_mean
    require(bool(torch.isfinite(result).all().item()), "CENTERED_GRAM_NONFINITE")
    return result.contiguous()


def centered_linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    ca = center_gram(a)
    cb = center_gram(b)

    numerator = torch.sum(ca * cb)
    denom = torch.linalg.vector_norm(ca) * torch.linalg.vector_norm(cb)

    require(float(denom.item()) > 0.0, "CKA_ZERO_DENOMINATOR")

    value = float((numerator / denom).item())
    require(math.isfinite(value), "CKA_NONFINITE")
    require(-1e-12 <= value <= 1.0 + 1e-12, f"CKA_RANGE:{value}")
    return value


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    require(a.ndim == b.ndim == 1 and a.shape == b.shape, "PEARSON_SHAPE")

    da = a - a.mean()
    db = b - b.mean()

    denom = torch.linalg.vector_norm(da) * torch.linalg.vector_norm(db)
    require(float(denom.item()) > 0.0, "PEARSON_ZERO_DENOMINATOR")

    value = float((torch.dot(da, db) / denom).item())
    require(math.isfinite(value), "PEARSON_NONFINITE")
    require(-1.0 - 1e-12 <= value <= 1.0 + 1e-12, f"PEARSON_RANGE:{value}")
    return value


def strict_upper(value: torch.Tensor) -> torch.Tensor:
    idx = torch.triu_indices(N, N, offset=1)
    return value[idx[0], idx[1]].contiguous()


def matrix_template() -> list[list[float]]:
    return [[0.0 for _ in SCALE_ORDER] for _ in SCALE_ORDER]


def analyze_family(
    family: str,
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    normalized = {
        scale: row_normalize(tensors[scale])
        for scale in SCALE_ORDER
    }

    grams = {
        scale: gram(normalized[scale])
        for scale in SCALE_ORDER
    }

    uppers = {
        scale: strict_upper(grams[scale])
        for scale in SCALE_ORDER
    }

    cka_matrix = matrix_template()
    rsm_matrix = matrix_template()
    pairs: list[dict[str, Any]] = []

    for i, scale_a in enumerate(SCALE_ORDER):
        for j, scale_b in enumerate(SCALE_ORDER):
            if j < i:
                continue

            cka = centered_linear_cka(grams[scale_a], grams[scale_b])
            rsm = pearson(uppers[scale_a], uppers[scale_b])

            cka_matrix[i][j] = cka
            cka_matrix[j][i] = cka
            rsm_matrix[i][j] = rsm
            rsm_matrix[j][i] = rsm

            if i != j:
                pairs.append(
                    {
                        "family": family,
                        "scale_a": scale_a,
                        "scale_b": scale_b,
                        "cka": cka,
                        "cosine_rsm_pearson": rsm,
                    }
                )

    require(len(pairs) == 10, f"PAIR_COUNT:{family}:{len(pairs)}")

    return {
        "scale_order": list(SCALE_ORDER),
        "cka_matrix": cka_matrix,
        "cosine_rsm_pearson_matrix": rsm_matrix,
        "pairs": pairs,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "family",
        "scale_a",
        "scale_b",
        "cka",
        "cosine_rsm_pearson",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_matrix(matrix: list[list[float]]) -> str:
    header = "| | " + " | ".join(SCALE_ORDER) + " |\n"
    sep = "|---|" + "|".join(["---:" for _ in SCALE_ORDER]) + "|\n"
    rows = []
    for scale, values in zip(SCALE_ORDER, matrix):
        rows.append(
            "| "
            + scale
            + " | "
            + " | ".join(f"{float(v):.6f}" for v in values)
            + " |"
        )
    return header + sep + "\n".join(rows)


def write_report(
    path: Path,
    *,
    expected_head: str,
    results: dict[str, Any],
) -> None:
    blocks = [
        "# Five-Scale Coordinate-Free Native Geometry Static Analysis",
        "",
        "## Status",
        "",
        "`DESCRIPTIVE_STATIC_RESULT`",
        "",
        f"Execution commit: `{expected_head}`",
        "",
        "No model/tokenizer execution, training, evaluation, forward pass, "
        "backward pass, or new representation collection occurred.",
        "",
        "130M is historical-provenance geometry; 370M--2.8B are the later "
        "homogeneous geometry-preparation series.",
        "",
    ]

    for family in FAMILIES:
        result = results[family]
        blocks.extend(
            [
                f"## {family.upper()}",
                "",
                "### Centered linear CKA",
                "",
                fmt_matrix(result["cka_matrix"]),
                "",
                "### Cosine-RSM Pearson correlation",
                "",
                fmt_matrix(result["cosine_rsm_pearson_matrix"]),
                "",
            ]
        )

    blocks.extend(
        [
            "## Interpretation boundary",
            "",
            "These matrices are descriptive. No p-value, permutation test, "
            "threshold, monotonic trend test, or scaling-law fit was performed.",
            "",
            "Linear CKA addresses similarity up to orthogonal feature rotations "
            "and isotropic scaling; it does not establish equivalence or "
            "non-equivalence under arbitrary invertible transformations.",
            "",
            "Scientific interpretation is intentionally deferred until this "
            "static result and its provenance are reviewed.",
            "",
        ]
    )

    path.write_text("\n".join(blocks), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only five-scale coordinate-free geometry analysis from "
            "already-frozen alignment_delta_h artifacts."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    head = git("rev-parse", "HEAD")
    require(head == args.expected_head, f"HEAD_MISMATCH:{head}")

    status = git("status", "--porcelain")
    require(status == "", "WORKTREE_NOT_CLEAN")

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    ).resolve()

    require(
        ROOT.resolve() in output_dir.parents,
        "OUTPUT_OUTSIDE_REPOSITORY",
    )
    require(not output_dir.exists(), f"OUTPUT_EXISTS:{output_dir}")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    for scale in SCALE_ORDER:
        cfg = CONFIG[scale]
        root = ROOT / str(cfg["root"])
        if scale == "130M":
            verify_130m_boundary(root, cfg)
        else:
            verify_later_manifest(root, cfg)

    tensors: dict[str, dict[str, torch.Tensor]] = {
        family: {} for family in FAMILIES
    }
    provenance: list[dict[str, Any]] = []
    canonical_ids: dict[str, list[str] | None] = {
        family: None for family in FAMILIES
    }

    for scale in SCALE_ORDER:
        for family in FAMILIES:
            tensor, entry, pair_ids = verify_and_load(scale, family)
            tensors[family][scale] = tensor
            provenance.append(entry)

            if canonical_ids[family] is None:
                canonical_ids[family] = pair_ids
            else:
                require(
                    canonical_ids[family] == pair_ids,
                    f"CROSS_SCALE_PAIR_IDENTITY:{scale}:{family}",
                )

    results = {
        family: analyze_family(family, tensors[family])
        for family in FAMILIES
    }

    output_dir.mkdir(parents=True, exist_ok=False)

    payload = {
        "schema_version": (
            "gen4-mamba1-five-scale-coordinate-free-geometry-static-analysis-v1"
        ),
        "execution_head": head,
        "scientific_model_forward_count": 0,
        "training_executed": False,
        "evaluation_executed": False,
        "tokenizer_executed": False,
        "backward_executed": False,
        "primary_metric": "row-normalized-centered-linear-cka",
        "secondary_metric": "row-normalized-cosine-rsm-pearson",
        "scale_order": list(SCALE_ORDER),
        "families": results,
    }

    provenance_payload = {
        "schema_version": (
            "gen4-mamba1-five-scale-coordinate-free-geometry-provenance-v1"
        ),
        "execution_head": head,
        "N": N,
        "scale_order": list(SCALE_ORDER),
        "families": list(FAMILIES),
        "artifacts": provenance,
    }

    result_path = output_dir / "coordinate_free_geometry_result.json"
    prov_path = output_dir / "provenance_manifest.json"
    csv_path = output_dir / "pairwise_coordinate_free_geometry.csv"
    report_path = output_dir / "analysis_report.md"

    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    prov_path.write_text(
        json.dumps(
            provenance_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    pair_rows = []
    for family in FAMILIES:
        pair_rows.extend(results[family]["pairs"])
    write_csv(csv_path, pair_rows)
    write_report(
        report_path,
        expected_head=head,
        results=results,
    )

    output_files = [
        result_path,
        prov_path,
        csv_path,
        report_path,
    ]
    sums = output_dir / "SHA256SUMS.txt"
    sums.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in sorted(output_files)
        ),
        encoding="ascii",
    )

    print("RESULT=PASS_STATIC_ONLY")
    print(f"HEAD={head}")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("GPU_REQUIRED=NO")
    print(f"OUTPUT_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
