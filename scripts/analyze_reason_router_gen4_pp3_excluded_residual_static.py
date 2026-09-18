from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

EXPECTED_HEAD = "7a6c30fd6d76c74e86917641401f62308e8195c0"

HOLDOUT_XG2_ITEMS_SHA = "41677dd4e5594f9a3eb3f33477c014b44368644fe2c1fe9038fd647b61119f6e"
HOLDOUT_XG4_ITEMS_SHA = "739fdc74ae64d4a30bfc1d772c3be4516c762d58313be45397633104207a8c7a"
NECESSITY_ITEMS_SHA = "67d1d550c45e73a346a0a3b4b3fe4fea39c8d7ac89104ba74b1528b265f8cce0"
RESTORATION_ITEMS_SHA = "8c4fade5f0e2c02ed183352a573fe9dc70af210610ec0ab5d00cb2f27371444d"

EXPECTED_EIGENVALUES = (
    0.87061814189182785,
    0.94755022112376275,
    0.98692852916688512,
    0.99848952673382474,
    0.99986792842854511,
)

RESIDUAL_PLANES = (0, 1, 3, 4)
PLANE_NAMES = ("P1", "P2", "P3", "P4", "P5")
RESIDUAL_NAMES = ("P1", "P2", "P4", "P5")
K = 5

HOLDOUT_ROOT = Path(
    "reports/reason_router_gen4_xg2_basis_cross_family_holdout_c8fcb97_retry1"
)
NECESSITY_ROOT = Path("reports/reason_router_gen4_pp3_necessity_dc85079_retry2")
RESTORATION_ROOT = Path(
    "reports/reason_router_gen4_pp3_restoration_sufficiency_53b6cce_retry2"
)

OUT_JSON = Path(
    "reports/reason_router_gen4_pp3_excluded_residual_static_analysis_7a6c30f.json"
)
OUT_MD = Path(
    "reports/reason_router_gen4_pp3_excluded_residual_static_analysis_7a6c30f.md"
)


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def fmean(xs: Iterable[float]) -> float:
    values = list(xs)
    require(bool(values), "EMPTY_MEAN")
    return math.fsum(values) / len(values)


def cosine(a: list[float], b: list[float]) -> float:
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    require(na > 0.0 and nb > 0.0, "ZERO_COSINE_NORM")
    return dot / (na * nb)


def effective_count_nonnegative(values: list[float]) -> float:
    total = math.fsum(values)
    require(total > 0.0, "NONPOSITIVE_EFFECTIVE_COUNT_TOTAL")
    w = [x / total for x in values]
    denom = math.fsum(x * x for x in w)
    require(denom > 0.0, "ZERO_EFFECTIVE_COUNT_DENOM")
    return 1.0 / denom


def effective_count_abs(values: list[float]) -> float:
    return effective_count_nonnegative([abs(x) for x in values])


def normalized_abs(values: list[float]) -> list[float]:
    total = math.fsum(abs(x) for x in values)
    require(total > 0.0, "ZERO_ABS_NORMALIZATION")
    return [abs(x) / total for x in values]


def sign_code(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def extract_holdout(item: dict[str, Any]) -> tuple[list[float], list[float], float]:
    j2 = [float(row["J"]) for row in item["xg2_basis_probes"]]
    j4 = [float(row["J"]) for row in item["xg4_basis_probes"]]
    require(len(j2) == K and len(j4) == K, "HOLDOUT_J_COUNT")
    return j2, j4, float(item["Q"])


def extract_condition(
    item: dict[str, Any],
    condition_name: str,
) -> tuple[list[float], list[float], float]:
    by = {str(row["condition"]): row for row in item["conditions"]}
    require(condition_name in by, f"MISSING_CONDITION:{condition_name}")
    condition = by[condition_name]
    probes = condition["direction_probes"]
    require(len(probes) == 2 * K, f"PROBE_COUNT:{condition_name}")
    expected = [f"xg2_{i}" for i in range(K)] + [f"xg4_{i}" for i in range(K)]
    observed = [str(row["direction_key"]) for row in probes]
    require(observed == expected, f"DIRECTION_ORDER:{condition_name}")
    j2 = [float(row["J"]) for row in probes[:K]]
    j4 = [float(row["J"]) for row in probes[K:]]
    return j2, j4, float(condition["Q"])


def build_geometry(repo: Path, holdout_module: Any) -> dict[str, Any]:
    import torch

    # Windows working-tree text conversion can alter frozen JSONL bytes.
    # The family bases depend only on the binary alignment_delta_h.pt plans,
    # so read those exact bytes from the frozen Git object rather than
    # re-validating the whole Phase-1 working-tree artifact.
    prior = holdout_module.prior
    freeze = prior.PHASE1_ARTIFACT_FREEZE_COMMIT
    loaded: dict[str, Any] = {}

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", freeze, EXPECTED_HEAD],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "PHASE1_FREEZE_NOT_ANCESTOR")

    for family in ("xg2", "xg4"):
        rel = (
            prior.PHASE1_ARTIFACT_ROOT
            / family
            / prior.phase1.PLAN_FILE
        ).as_posix()
        raw = subprocess.check_output(
            ["git", "show", f"{freeze}:{rel}"],
            cwd=repo,
        )
        observed = hashlib.sha256(raw).hexdigest()
        expected = prior.PHASE1_PLAN_SHA256[family]
        require(
            observed == expected,
            f"PHASE1_PLAN_GIT_OBJECT_SHA:{family}:{observed}",
        )

        plans = torch.load(
            io.BytesIO(raw),
            map_location="cpu",
            weights_only=True,
        )
        require(torch.is_tensor(plans), f"PLAN_NOT_TENSOR:{family}")
        require(
            plans.ndim == 2
            and int(plans.shape[0]) == prior.SOURCE_PAIR_COUNT,
            f"PLAN_SHAPE:{family}:{tuple(plans.shape)}",
        )
        require(
            bool(torch.isfinite(plans).all().item()),
            f"PLAN_NONFINITE:{family}",
        )

        basis = prior.reconstruct_family_basis(family, plans)
        loaded[family] = {
            "basis": basis,
            "plan_sha256": observed,
        }

    b2 = (
        loaded["xg2"]["basis"]["basis"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    b4 = (
        loaded["xg4"]["basis"]["basis"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    require(tuple(b2.shape) == tuple(b4.shape), "BASIS_SHAPE_MISMATCH")
    require(int(b2.shape[1]) == K, "BASIS_K")

    gram2 = b2.T @ b2
    gram4 = b4.T @ b4
    eye = torch.eye(K, dtype=torch.float64)
    require(float(torch.max(torch.abs(gram2 - eye))) <= 1e-10, "B2_ORTHONORMAL")
    require(float(torch.max(torch.abs(gram4 - eye))) <= 1e-10, "B4_ORTHONORMAL")

    p2 = b2 @ b2.T
    p4 = b4 @ b4.T
    contrast = p2 - p4

    evals, evecs = torch.linalg.eigh(contrast)
    pos_idx = [i for i, x in enumerate(evals.tolist()) if x > 1e-10]
    neg_idx = [i for i, x in enumerate(evals.tolist()) if x < -1e-10]
    require(len(pos_idx) == K and len(neg_idx) == K, "CONTRAST_RANK")

    pos_idx = sorted(pos_idx, key=lambda i: float(evals[i]))
    neg_idx = sorted(neg_idx, key=lambda i: abs(float(evals[i])))

    lam_pos = [float(evals[i]) for i in pos_idx]
    lam_neg_abs = [abs(float(evals[i])) for i in neg_idx]

    for i, expected in enumerate(EXPECTED_EIGENVALUES):
        require(abs(lam_pos[i] - expected) <= 2e-12, f"POS_EIGENVALUE:P{i+1}")
        require(abs(lam_neg_abs[i] - expected) <= 2e-12, f"NEG_EIGENVALUE:P{i+1}")
        require(abs(lam_pos[i] - lam_neg_abs[i]) <= 2e-12, f"PAIR_EIGENVALUE:P{i+1}")

    plus = torch.stack([evecs[:, i] for i in pos_idx], dim=1)
    minus = torch.stack([evecs[:, i] for i in neg_idx], dim=1)

    union = torch.cat([b2, b4], dim=1)
    union_gram = union.T @ union
    require(int(torch.linalg.matrix_rank(union_gram)) == 2 * K, "UNION_RANK")

    return {
        "b2": b2,
        "b4": b4,
        "union": union,
        "union_gram": union_gram,
        "plus": plus,
        "minus": minus,
        "lambda": lam_pos,
    }


def decompose(
    geometry: dict[str, Any],
    j2: list[float],
    j4: list[float],
    q_stored: float,
) -> dict[str, Any]:
    import torch

    y = torch.tensor(j2 + j4, dtype=torch.float64)
    coeff = torch.linalg.solve(geometry["union_gram"], y)
    g_union = geometry["union"] @ coeff

    z_plus = geometry["plus"].T @ g_union
    z_minus = geometry["minus"].T @ g_union
    lam = torch.tensor(geometry["lambda"], dtype=torch.float64)

    positive = (lam * z_plus.square() / K).tolist()
    negative = (-lam * z_minus.square() / K).tolist()
    net = [p + n for p, n in zip(positive, negative, strict=True)]

    q_direct = (
        math.fsum(x * x for x in j2)
        - math.fsum(x * x for x in j4)
    ) / K
    q_reconstructed = math.fsum(net)

    require(math.isfinite(q_stored), "NONFINITE_Q_STORED")
    require(abs(q_direct - q_stored) <= 2e-18, "DIRECT_Q_MISMATCH")
    require(abs(q_reconstructed - q_stored) <= 2e-18, "RECON_Q_MISMATCH")

    return {
        "positive": [float(x) for x in positive],
        "negative": [float(x) for x in negative],
        "net": [float(x) for x in net],
        "q": float(q_stored),
        "q_reconstruction_residual": float(q_reconstructed - q_stored),
    }


def analyze_cohort(
    name: str,
    items: list[dict[str, Any]],
    extractor: Any,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for item in items:
        j2, j4, q = extractor(item)
        rows.append(decompose(geometry, j2, j4, q))

    require(len(rows) == 300, f"COHORT_SIZE:{name}")

    positive_mean = [
        fmean(row["positive"][i] for row in rows)
        for i in range(K)
    ]
    negative_mean = [
        fmean(row["negative"][i] for row in rows)
        for i in range(K)
    ]
    net_mean = [
        fmean(row["net"][i] for row in rows)
        for i in range(K)
    ]
    mean_q = fmean(row["q"] for row in rows)

    max_recon = max(abs(row["q_reconstruction_residual"]) for row in rows)
    require(abs(math.fsum(net_mean) - mean_q) <= 2e-18, f"MEAN_Q_RECON:{name}")

    residual_net = [net_mean[i] for i in RESIDUAL_PLANES]
    residual_positive = [positive_mean[i] for i in RESIDUAL_PLANES]
    residual_negative = [negative_mean[i] for i in RESIDUAL_PLANES]

    positive_net_total = math.fsum(max(x, 0.0) for x in net_mean)
    pp3_positive_net_share = (
        max(net_mean[2], 0.0) / positive_net_total
        if positive_net_total > 0.0 else None
    )

    abs_net = [abs(x) for x in residual_net]
    dominant_abs = max(range(4), key=lambda i: abs_net[i])
    dominant_positive = max(range(4), key=lambda i: residual_positive[i])

    return {
        "name": name,
        "n": len(rows),
        "mean_Q": mean_q,
        "positive_gain_mean_by_plane": dict(zip(PLANE_NAMES, positive_mean, strict=True)),
        "negative_penalty_mean_by_plane": dict(zip(PLANE_NAMES, negative_mean, strict=True)),
        "net_mean_by_plane": dict(zip(PLANE_NAMES, net_mean, strict=True)),
        "pp3_positive_net_share": pp3_positive_net_share,
        "residual_planes": list(RESIDUAL_NAMES),
        "residual_net_vector": residual_net,
        "residual_positive_gain_vector": residual_positive,
        "residual_negative_penalty_vector": residual_negative,
        "residual_mean_Q": math.fsum(residual_net),
        "residual_abs_net_weights": normalized_abs(residual_net),
        "residual_effective_abs_net_plane_count": effective_count_abs(residual_net),
        "residual_effective_positive_gain_plane_count":
            effective_count_nonnegative(residual_positive),
        "residual_net_sign_pattern": [sign_code(x) for x in residual_net],
        "dominant_residual_abs_net_plane": RESIDUAL_NAMES[dominant_abs],
        "dominant_residual_positive_gain_plane": RESIDUAL_NAMES[dominant_positive],
        "max_abs_Q_reconstruction_residual": max_recon,
    }


def pairwise(cohorts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    names = list(cohorts)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = cohorts[names[i]]
            b = cohorts[names[j]]
            sa = a["residual_net_sign_pattern"]
            sb = b["residual_net_sign_pattern"]
            sign_agreement = math.fsum(
                1.0 for x, y in zip(sa, sb, strict=True) if x == y
            ) / len(sa)
            out.append({
                "left": names[i],
                "right": names[j],
                "residual_net_cosine": cosine(
                    a["residual_net_vector"],
                    b["residual_net_vector"],
                ),
                "residual_positive_gain_cosine": cosine(
                    a["residual_positive_gain_vector"],
                    b["residual_positive_gain_vector"],
                ),
                "residual_net_sign_agreement_fraction": sign_agreement,
            })
    return out


def render_markdown(result: dict[str, Any]) -> str:
    lines = []
    lines.append("# Gen4 PP3-Excluded Residual Static Characterization")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("`STATIC_EXPLORATORY_RESIDUAL_CHARACTERIZATION_NO_INFERENCE`")
    lines.append("")
    lines.append(
        "This analysis reuses only already validated directional-Jacobian artifacts "
        "and frozen XG2/XG4 basis geometry."
    )
    lines.append("")
    lines.append("- new model forwards: `0`")
    lines.append("- checkpoint loads: `0`")
    lines.append("- GPU use: `false`")
    lines.append("- p-values: `0`")
    lines.append("- training/backward/task-head/logit analysis: `none`")
    lines.append("")
    lines.append("The purpose is hypothesis generation for the PP3-excluded distributed residual,")
    lines.append("not confirmatory promotion of any secondary plane.")
    lines.append("")
    lines.append("## Frozen decomposition")
    lines.append("")
    lines.append("For every item:")
    lines.append("")
    lines.append("`Q = (1/5) g^T (P2 - P4) g`")
    lines.append("")
    lines.append("The same frozen five principal contrast planes P1..P5 are used.")
    lines.append("P3 is excluded only after exact reconstruction; no secondary plane is selected.")
    lines.append("")
    lines.append("Frozen positive contrast eigenvalues:")
    lines.append("")
    for i, value in enumerate(result["geometry"]["positive_eigenvalues"], 1):
        lines.append(f"- P{i}: `{value:.17g}`")
    lines.append("")
    lines.append("## Cohort summaries")
    lines.append("")
    for name, c in result["cohorts"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- N: `{c['n']}`")
        lines.append(f"- mean Q: `{c['mean_Q']:.17g}`")
        lines.append(f"- mean PP3 positive-net share: `{c['pp3_positive_net_share']}`")
        lines.append(f"- PP3-excluded residual mean Q: `{c['residual_mean_Q']:.17g}`")
        lines.append(
            "- residual effective |net| plane count: "
            f"`{c['residual_effective_abs_net_plane_count']:.12g}`"
        )
        lines.append(
            "- residual effective positive-gain plane count: "
            f"`{c['residual_effective_positive_gain_plane_count']:.12g}`"
        )
        lines.append(
            "- residual net sign pattern [P1,P2,P4,P5]: "
            f"`{c['residual_net_sign_pattern']}`"
        )
        lines.append(
            "- dominant residual |net| plane (descriptive only): "
            f"`{c['dominant_residual_abs_net_plane']}`"
        )
        lines.append(
            "- dominant residual positive-gain plane (descriptive only): "
            f"`{c['dominant_residual_positive_gain_plane']}`"
        )
        lines.append(
            "- max |Q reconstruction residual|: "
            f"`{c['max_abs_Q_reconstruction_residual']:.17g}`"
        )
        lines.append("")
        lines.append("| Plane | positive gain | negative penalty | net |")
        lines.append("|---|---:|---:|---:|")
        for plane in PLANE_NAMES:
            lines.append(
                f"| {plane} | "
                f"{c['positive_gain_mean_by_plane'][plane]:.17g} | "
                f"{c['negative_penalty_mean_by_plane'][plane]:.17g} | "
                f"{c['net_mean_by_plane'][plane]:.17g} |"
            )
        lines.append("")
    lines.append("## PP3-excluded pairwise profile agreement")
    lines.append("")
    lines.append(
        "| left | right | residual net cosine | residual positive-gain cosine | "
        "net-sign agreement |"
    )
    lines.append("|---|---|---:|---:|---:|")
    for row in result["pairwise"]:
        lines.append(
            f"| {row['left']} | {row['right']} | "
            f"{row['residual_net_cosine']:.12g} | "
            f"{row['residual_positive_gain_cosine']:.12g} | "
            f"{row['residual_net_sign_agreement_fraction']:.12g} |"
        )
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append(
        "These quantities are descriptive and exploratory. They may motivate a "
        "future prospectively frozen residual object on a fresh population, but "
        "they do not establish any P1/P2/P4/P5 plane as causal, transportable, "
        "necessary, or sufficient."
    )
    lines.append("")
    lines.append(
        "No secondary plane should be promoted from this analysis without a new "
        "prospectively frozen question and fresh evidence."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    require((repo / ".git").exists(), f"NOT_REPO:{repo}")

    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain")
    require(head == EXPECTED_HEAD, f"HEAD_MISMATCH:{head}")
    require(status == "", "WORKTREE_NOT_CLEAN")

    sys.path.insert(0, str(repo))

    from scripts import reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda as holdout
    from scripts import reason_router_gen4_pp3_necessity_fast_cuda as necessity
    from scripts import reason_router_gen4_pp3_restoration_sufficiency_fast_cuda as restoration

    holdout_xg2_root = repo / HOLDOUT_ROOT / "xg2"
    holdout_xg4_root = repo / HOLDOUT_ROOT / "xg4"
    necessity_root = repo / NECESSITY_ROOT
    restoration_root = repo / RESTORATION_ROOT

    exact_items = {
        "holdout_xg2": (
            holdout_xg2_root / "xg2_basis_cross_family_items.jsonl",
            HOLDOUT_XG2_ITEMS_SHA,
        ),
        "holdout_xg4": (
            holdout_xg4_root / "xg2_basis_cross_family_items.jsonl",
            HOLDOUT_XG4_ITEMS_SHA,
        ),
        "necessity": (
            necessity_root / "pp3_necessity_items.jsonl",
            NECESSITY_ITEMS_SHA,
        ),
        "restoration": (
            restoration_root / "pp3_restoration_sufficiency_items.jsonl",
            RESTORATION_ITEMS_SHA,
        ),
    }
    for label, (path, expected) in exact_items.items():
        require(path.is_file(), f"MISSING_ITEMS:{label}:{path}")
        observed = sha256_file(path)
        require(observed == expected, f"ITEMS_SHA:{label}:{observed}")

    h2 = holdout.validate_artifact(holdout_xg2_root, "xg2")
    h4 = holdout.validate_artifact(holdout_xg4_root, "xg4")
    nec = necessity.validate_artifact(necessity_root)
    res = restoration.validate_artifact(restoration_root)

    geometry = build_geometry(repo, holdout)

    cohorts = {
        "XG2_601_900": analyze_cohort(
            "XG2_601_900",
            h2["items"],
            extract_holdout,
            geometry,
        ),
        "XG4_601_900": analyze_cohort(
            "XG4_601_900",
            h4["items"],
            extract_holdout,
            geometry,
        ),
        "XG1_601_900_NATIVE": analyze_cohort(
            "XG1_601_900_NATIVE",
            nec["items"],
            lambda x: extract_condition(x, "native"),
            geometry,
        ),
        "XG1_901_1200_NATIVE": analyze_cohort(
            "XG1_901_1200_NATIVE",
            res["items"],
            lambda x: extract_condition(x, "pp3_restored"),
            geometry,
        ),
    }

    result = {
        "schema_version": "gen4-pp3-excluded-residual-static-characterization-v1",
        "status": "STATIC_EXPLORATORY_RESIDUAL_CHARACTERIZATION_NO_INFERENCE",
        "analysis_head": EXPECTED_HEAD,
        "input_item_sha256": {
            label: expected
            for label, (_path, expected) in exact_items.items()
        },
        "geometry": {
            "positive_eigenvalues": geometry["lambda"],
            "residual_planes": list(RESIDUAL_NAMES),
        },
        "cohorts": cohorts,
        "pairwise": pairwise(cohorts),
        "boundaries": {
            "scientific_model_forward_count": 0,
            "checkpoint_load_count": 0,
            "gpu_used": False,
            "p_value_count": 0,
            "training_executed": False,
            "backward_executed": False,
            "task_heads_executed": False,
            "logits_read": False,
            "confirmatory_inference_executed": False,
            "secondary_plane_promoted": False,
        },
    }

    out_json = repo / OUT_JSON
    out_md = repo / OUT_MD
    require(not out_json.exists(), f"OUTPUT_EXISTS:{OUT_JSON}")
    require(not out_md.exists(), f"OUTPUT_EXISTS:{OUT_MD}")

    out_json.write_bytes(canonical_json_bytes(result))
    out_md.write_text(render_markdown(result), encoding="utf-8", newline="\n")

    print("=== PP3-EXCLUDED RESIDUAL STATIC ANALYSIS ===")
    print("HEAD =", head)
    print("COHORTS =", len(cohorts))
    print("MODEL_FORWARDS = 0")
    print("CHECKPOINT_LOADS = 0")
    print("GPU_USED = False")
    print("P_VALUE_COUNT = 0")
    print()
    for name, c in cohorts.items():
        print(f"[{name}]")
        print(" mean_Q =", repr(c["mean_Q"]))
        print(" pp3_positive_net_share =", repr(c["pp3_positive_net_share"]))
        print(" residual_mean_Q =", repr(c["residual_mean_Q"]))
        print(" residual_net =", [repr(x) for x in c["residual_net_vector"]])
        print(" residual_sign =", c["residual_net_sign_pattern"])
        print(
            " residual_effective_abs_net_plane_count =",
            repr(c["residual_effective_abs_net_plane_count"]),
        )
        print(
            " residual_effective_positive_gain_plane_count =",
            repr(c["residual_effective_positive_gain_plane_count"]),
        )
        print()
    print("[PAIRWISE]")
    for row in result["pairwise"]:
        print(
            row["left"],
            "<->",
            row["right"],
            "net_cos=",
            repr(row["residual_net_cosine"]),
            "positive_cos=",
            repr(row["residual_positive_gain_cosine"]),
            "sign_agreement=",
            repr(row["residual_net_sign_agreement_fraction"]),
        )
    print()
    print("JSON =", OUT_JSON.as_posix())
    print("JSON_SHA256 =", sha256_file(out_json))
    print("REPORT =", OUT_MD.as_posix())
    print("REPORT_SHA256 =", sha256_file(out_md))
    print("RESULT = STATIC_ANALYSIS_PASS")


if __name__ == "__main__":
    main()
