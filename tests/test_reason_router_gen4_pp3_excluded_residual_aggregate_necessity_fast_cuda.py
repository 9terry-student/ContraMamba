from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda
    as r
)


def synthetic_planes():
    out = {}
    residual_names = (
        "p1_plus", "p1_minus",
        "p2_plus", "p2_minus",
        "p4_plus", "p4_minus",
        "p5_plus", "p5_minus",
    )
    for index, name in enumerate(residual_names):
        value = torch.zeros(r.DIM, dtype=torch.float64)
        value[index] = 1.0
        out[name] = value
    p3p = torch.zeros(r.DIM, dtype=torch.float64)
    p3m = torch.zeros(r.DIM, dtype=torch.float64)
    p3p[8] = 1.0
    p3m[9] = 1.0
    out["pp3_plus"] = p3p
    out["pp3_minus"] = p3m
    return out


def audit(condition: str) -> dict:
    coefficients = {
        "P1": [1.0, 2.0],
        "P2": [3.0, 4.0],
        "P4": [5.0, 6.0],
        "P5": [7.0, 8.0],
    }
    l2 = sum(x * x for pair in coefficients.values() for x in pair) ** 0.5
    projections = {
        plane: [0.0, 0.0]
        for plane in r.RESIDUAL_PLANES
    }
    return {
        "condition": condition,
        "token_index": 10,
        "orientation": 1,
        "branch_sign": 1,
        "coefficient_source": "branch_local_native_residual_coordinates",
        "native_residual_coefficients": coefficients,
        "treatment_correction_l2": l2,
        "control_correction_l2": l2,
        "treatment_control_l2_mismatch": 0.0,
        "treatment_control_dot": 0.0,
        "condition_correction_l2":
            0.0 if condition == "native" else l2,
        "pp3_native_coefficients": [0.2, -0.3],
        "pp3_post_coefficients": [0.2, -0.3],
        "pp3_coefficient_drift_max_abs": 0.0,
        "residual_post_condition_projections": projections,
        "residual_neutralization_max_abs_projection":
            0.0 if condition == "residual_neutralized" else None,
        "probe_correction_l2": r.EPS,
        "applied_correction_max_abs_residual": 0.0,
    }


def signed(condition: str, orientation: int, value: float) -> dict:
    audits = {}
    for role, sign in (("tp", 1), ("tm", -1)):
        value_audit = audit(condition)
        value_audit["orientation"] = orientation
        value_audit["branch_sign"] = sign
        value_audit["token_index"] = 10 if role == "tp" else 11
        audits[role] = value_audit
    return {
        "condition": condition,
        "orientation": orientation,
        "F": value,
        "plus_path_efficiency": value,
        "minus_path_efficiency": 0.0,
        "branch_audits": audits,
        "model_forward_count": r.F_SIGNED,
    }


def direction(condition: str, key: str, j: float) -> dict:
    fp = j * r.EPS
    fm = -j * r.EPS
    j_value = (fp - fm) / (2.0 * r.EPS)
    family, index = key.split("_")
    return {
        "direction_key": key,
        "basis_family": family,
        "basis_index": int(index),
        "F_plus": fp,
        "F_minus": fm,
        "J": j_value,
        "J_squared": j_value * j_value,
        "positive_probe": signed(condition, 1, fp),
        "negative_probe": signed(condition, -1, fm),
        "model_forward_count": r.F_DIR,
    }


def condition(name: str, scale: float) -> dict:
    probes = [
        direction(name, key, scale * value)
        for key, value in zip(
            r.DIRECTIONS,
            [1, 2, 3, 4, 5, .5, 1, 1.5, 2, 2.5],
            strict=True,
        )
    ]
    e2 = sum(float(p["J_squared"]) for p in probes[:r.K]) / r.K
    e4 = sum(float(p["J_squared"]) for p in probes[r.K:]) / r.K
    return {
        "condition": name,
        "direction_order": list(r.DIRECTIONS),
        "direction_probes": probes,
        "E_XG2": e2,
        "E_XG4": e4,
        "Q": e2 - e4,
        "scientific_model_forward_count": r.F_COND,
    }


def item(index: int) -> dict:
    conditions = [
        condition("native", 1.0),
        condition("residual_neutralized", .8),
        condition("quarter_turn_control", .95),
    ]
    by = {c["condition"]: c for c in conditions}
    endpoint = r.endpoint(
        float(by["native"]["Q"]),
        float(by["residual_neutralized"]["Q"]),
        float(by["quarter_turn_control"]["Q"]),
    )
    return {
        "family_key": "xg1",
        "source_pair_id": r.expected_pairs()[index],
        "pair_index": index,
        "target_plus_anchor": 1,
        "target_minus_anchor": 1,
        "reference_plus_anchor": 1,
        "reference_minus_anchor": 1,
        "schema_version": r.ITEM_SCHEMA,
        "implementation_authority_commit": r.AUTHORITY_COMMIT,
        "static_preparation_freeze_commit": r.STATIC_COMMIT,
        "design_commit": r.DESIGN_COMMIT,
        "epsilon": r.EPS,
        "condition_order": list(r.CONDITIONS),
        "direction_order": list(r.DIRECTIONS),
        "conditions": conditions,
        **endpoint,
        "baseline_model_forward_count_this_run": 0,
        "scientific_model_forward_count_this_run": r.F_PAIR,
        "primary_inference_executed": False,
        "scientific_conclusion": None,
    }


def summary():
    checkpoint = (
        r.holdout.phase1.base.prevalence_eq.extraction
        .REPRESENTATIVE_CHECKPOINT_SHA256
    )
    shards = [
        {
            "shard_id": shard["shard_id"],
            "gpu_id": shard["gpu_id"],
            "device_name": f"synthetic-gpu-{shard['gpu_id']}",
            "pair_first": shard["pair_first"],
            "pair_last": shard["pair_last"],
            "pair_count": shard["pair_count"],
            "scientific_model_forward_count_this_run":
                shard["forward_budget"],
            "checkpoint_sha256": checkpoint,
        }
        for shard in r.SHARDS
    ]
    return {
        "schema_version": r.SUMMARY_SCHEMA,
        "result": r.RESULT_PASS,
        "execution_head": "synthetic-head",
        "design_commit": r.DESIGN_COMMIT,
        "static_preparation_freeze_commit": r.STATIC_COMMIT,
        "implementation_authority_commit": r.AUTHORITY_COMMIT,
        "source_pair_count": r.N,
        "pair_id_first": "xg1_fact_1501",
        "pair_id_last": "xg1_fact_1800",
        "epsilon": r.EPS,
        "condition_order": list(r.CONDITIONS),
        "direction_order": list(r.DIRECTIONS),
        "residual_plane_order": list(r.RESIDUAL_PLANES),
        "model_forwards_per_direction": r.F_DIR,
        "model_forwards_per_condition": r.F_COND,
        "model_forwards_per_pair": r.F_PAIR,
        "scientific_model_forward_count_this_run": r.F_TOTAL,
        "baseline_model_forward_count_this_run": 0,
        "gpu_count": r.GPU_COUNT,
        "parallelization": "independent_pair_shards_spawn",
        "shards": shards,
        "primary_endpoint_definition":
            "D_RES_NEC=(Q0-QR)-(Q0-QC)=QC-QR",
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
        "representative_checkpoint_sha256": checkpoint,
    }


def rewrite_artifact_hashes(out: Path, changed_name: str, raw: bytes):
    (out / changed_name).write_bytes(raw)
    manifest = json.loads(
        (out / r.MANIFEST_FILE).read_text(encoding="utf-8")
    )
    manifest["files"][changed_name]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["files"][changed_name]["bytes"] = len(raw)
    manifest_raw = r.canonical(manifest)
    (out / r.MANIFEST_FILE).write_bytes(manifest_raw)

    hashes = {
        r.ITEM_FILE: r.sha256_file(out / r.ITEM_FILE),
        r.SUMMARY_FILE: r.sha256_file(out / r.SUMMARY_FILE),
        r.MANIFEST_FILE: hashlib.sha256(manifest_raw).hexdigest(),
    }
    (out / r.CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def test_frozen_constants_shards_and_budget():
    assert r.DESIGN_COMMIT == "c4518d20f4417ca9f057fbd4940c28539e4ffb2c"
    assert r.STATIC_COMMIT == "4a7698264488e811370bdf071c3cde73735757e0"
    assert r.AUTHORITY_COMMIT == "7356e81d34b2883e74b8fa24b7751f725d9ca1db"
    assert r.CONDITIONS == (
        "native",
        "residual_neutralized",
        "quarter_turn_control",
    )
    assert r.F_PAIR == 120
    assert r.F_TOTAL == 36000
    assert r.SHARDS[0]["pair_first"] == "xg1_fact_1501"
    assert r.SHARDS[0]["pair_last"] == "xg1_fact_1650"
    assert r.SHARDS[1]["pair_first"] == "xg1_fact_1651"
    assert r.SHARDS[1]["pair_last"] == "xg1_fact_1800"
    assert r.SHARDS[0]["forward_budget"] == 18000
    assert r.SHARDS[1]["forward_budget"] == 18000
    r.validate_shards()



def test_frozen_static_inputs_and_residual_vector_authentication():
    validated = r.validate_static_inputs()
    assert validated["preparation"]["scientific_model_forward_count"] == 0
    assert validated["preparation"]["checkpoint_load_count"] == 0
    assert validated["preparation"]["gpu_used"] is False

    planes = r.load_planes()
    assert set(planes) == {
        "p1_plus", "p1_minus",
        "p2_plus", "p2_minus",
        "p4_plus", "p4_minus",
        "p5_plus", "p5_minus",
        "pp3_plus", "pp3_minus",
    }
    residual = torch.stack(
        [
            planes[name]
            for name in (
                "p1_plus", "p1_minus",
                "p2_plus", "p2_minus",
                "p4_plus", "p4_minus",
                "p5_plus", "p5_minus",
            )
        ],
        dim=1,
    )
    assert torch.max(
        torch.abs(
            residual.T @ residual
            - torch.eye(8, dtype=torch.float64)
        )
    ).item() <= r.TOL


def test_raw_artifact_rejects_provenance_and_order_mutations(tmp_path: Path):
    items = [item(i) for i in range(r.N)]

    out = tmp_path / "provenance"
    r.write_outputs(out, items, summary())
    value = json.loads(
        (out / r.SUMMARY_FILE).read_text(encoding="utf-8")
    )
    value["implementation_authority_commit"] = "0" * 40
    raw = r.canonical(value)
    rewrite_artifact_hashes(out, r.SUMMARY_FILE, raw)
    with pytest.raises(
        r.ResidualAggregateNecessityError,
        match="SUMMARY_PROVENANCE",
    ):
        r.validate_artifact(out)

    out2 = tmp_path / "order"
    r.write_outputs(out2, items, summary())
    value = json.loads(
        (out2 / r.SUMMARY_FILE).read_text(encoding="utf-8")
    )
    value["condition_order"] = list(reversed(r.CONDITIONS))
    raw = r.canonical(value)
    rewrite_artifact_hashes(out2, r.SUMMARY_FILE, raw)
    with pytest.raises(
        r.ResidualAggregateNecessityError,
        match="SUMMARY_ORDER",
    ):
        r.validate_artifact(out2)


def test_no_baseline_forward_accounting():
    value = item(0)
    assert value["scientific_model_forward_count_this_run"] == 120
    assert value["baseline_model_forward_count_this_run"] == 0
    assert sum(
        c["scientific_model_forward_count"]
        for c in value["conditions"]
    ) == 120

def test_residual_neutralization_and_quarter_turn_math():
    planes = synthetic_planes()
    h = torch.zeros(r.DIM, dtype=torch.float64)
    h[:10] = torch.tensor(
        [1., 2., 3., 4., 5., 6., 7., 8., .25, -.5],
        dtype=torch.float64,
    )

    native = r.condition_correction(h, "native", planes)
    neutral = r.condition_correction(
        h, "residual_neutralized", planes
    )
    control = r.condition_correction(
        h, "quarter_turn_control", planes
    )

    assert native["condition_correction_l2"] == 0.0
    assert neutral["residual_neutralization_max_abs_projection"] == 0.0
    assert control["residual_neutralization_max_abs_projection"] is None

    expected_component = torch.zeros(r.DIM, dtype=torch.float64)
    expected_component[:8] = h[:8]
    assert torch.equal(
        neutral["d"],
        -expected_component,
    )

    expected_control = torch.zeros(r.DIM, dtype=torch.float64)
    for offset in range(0, 8, 2):
        a = float(h[offset])
        b = float(h[offset + 1])
        expected_control[offset] = b
        expected_control[offset + 1] = -a
    assert torch.equal(control["d"], expected_control)

    assert neutral["treatment_control_l2_mismatch"] <= r.TOL
    assert abs(neutral["treatment_control_dot"]) <= r.TOL
    assert neutral["pp3_coefficient_drift_max_abs"] <= r.TOL
    assert control["pp3_coefficient_drift_max_abs"] <= r.TOL


def test_endpoint_identity():
    result = r.endpoint(10.0, 6.0, 8.5)
    assert result == {
        "Q0": 10.0,
        "QR": 6.0,
        "QC": 8.5,
        "A_R": 4.0,
        "A_C": 1.5,
        "D_RES_NEC": 2.5,
    }
    assert result["D_RES_NEC"] == result["QC"] - result["QR"]


def test_endpoint_canonicalizes_cancellation_sensitive_identity():
    q0 = 1.0
    qr = 0.1
    qc = 0.2

    cancellation_path = (q0 - qr) - (q0 - qc)
    canonical = qc - qr

    # Regression guard: these are mathematically identical but not
    # bitwise-identical IEEE-754 evaluations for this input.
    assert cancellation_path != canonical

    result = r.endpoint(q0, qr, qc)
    assert result["A_R"] == q0 - qr
    assert result["A_C"] == q0 - qc
    assert result["D_RES_NEC"] == canonical
    r.validate_endpoint(result)


def test_canonical_two_shard_merge():
    all_items = [item(i) for i in range(r.N)]
    checkpoint = (
        r.holdout.phase1.base.prevalence_eq.extraction
        .REPRESENTATIVE_CHECKPOINT_SHA256
    )
    payloads = []
    for shard in r.SHARDS:
        payloads.append({
            "shard_id": shard["shard_id"],
            "gpu_id": shard["gpu_id"],
            "device_name": f"gpu-{shard['gpu_id']}",
            "pair_first": shard["pair_first"],
            "pair_last": shard["pair_last"],
            "pair_count": shard["pair_count"],
            "scientific_model_forward_count_this_run":
                shard["forward_budget"],
            "checkpoint_sha256": checkpoint,
            "items": all_items[
                shard["start_index"]:shard["end_index"]
            ],
        })
    merged, meta = r.merge_shards(list(reversed(payloads)))
    assert [x["source_pair_id"] for x in merged] == list(r.expected_pairs())
    assert [x["gpu_id"] for x in meta] == [0, 1]


def test_artifact_roundtrip_and_endpoint_mutation_rejected(tmp_path: Path):
    items = [item(i) for i in range(r.N)]
    out = tmp_path / "artifact"
    r.write_outputs(out, items, summary())
    validated = r.validate_artifact(out)
    assert validated["summary"]["result"] == r.RESULT_PASS

    rows = r.read_jsonl(out / r.ITEM_FILE)
    rows[0]["D_RES_NEC"] += 1.0
    raw = r.jsonl(rows)
    rewrite_artifact_hashes(out, r.ITEM_FILE, raw)
    with pytest.raises(
        r.ResidualAggregateNecessityError,
        match="ENDPOINT:D_RES_NEC",
    ):
        r.validate_artifact(out)


def test_summary_budget_and_boundary_rejected(tmp_path: Path):
    items = [item(i) for i in range(r.N)]
    out = tmp_path / "artifact"
    r.write_outputs(out, items, summary())

    value = json.loads(
        (out / r.SUMMARY_FILE).read_text(encoding="utf-8")
    )
    value["scientific_model_forward_count_this_run"] = 35999
    raw = r.canonical(value)
    rewrite_artifact_hashes(out, r.SUMMARY_FILE, raw)
    with pytest.raises(
        r.ResidualAggregateNecessityError,
        match="SUMMARY_BUDGET",
    ):
        r.validate_artifact(out)


def test_item_boundary_rejects_inference_or_conclusion():
    value = item(0)
    value["primary_inference_executed"] = True
    with pytest.raises(
        r.ResidualAggregateNecessityError,
        match="ITEM_BOUNDARY",
    ):
        r.validate_item(value, r.expected_pairs()[0], 0)

    value = item(0)
    value["scientific_conclusion"] = "not-allowed"
    with pytest.raises(
        r.ResidualAggregateNecessityError,
        match="ITEM_BOUNDARY",
    ):
        r.validate_item(value, r.expected_pairs()[0], 0)


def test_static_hash_mismatch_fails_closed(monkeypatch):
    original = r.sha256_file
    target = r.ROOT / r.DATA_ROOT / "structured_source_facts.jsonl"

    def broken(path):
        if Path(path) == target:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(r, "sha256_file", broken)
    with pytest.raises(
        r.ResidualAggregateNecessityError,
        match="SHA:",
    ):
        r.validate_static_inputs()


def test_runner_has_no_statistical_inference_and_is_device_generic():
    source = Path(r.__file__).read_text(encoding="utf-8")
    lower = source.lower()
    compact = "".join(source.split())

    assert "scipy" not in lower
    assert "ttest" not in lower
    assert "student_t" not in lower
    assert '.to("cuda:0")' not in source
    assert "torch.cuda.set_device(0)" not in source
    assert "torch.cuda.set_device(gpu_id)" in compact
    assert "model.to(device)" in compact
    assert "torch.cuda.synchronize(device)" in compact
    assert "runtime_gate_for_device(runtime,gpu_id)" in compact
    assert "make_fast_capture_for_device(runtime,kernels,device)" in compact
    assert '"primary_inference_executed":False' in compact
    assert '"multiplicity_correction_executed":False' in compact
    assert '"scientific_conclusion":None' in compact
