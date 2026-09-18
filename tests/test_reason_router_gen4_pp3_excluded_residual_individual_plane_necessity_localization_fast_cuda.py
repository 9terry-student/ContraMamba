from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_fast_cuda
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


def synthetic_audit(condition: str) -> dict:
    plane, kind = r.parse_condition(condition)
    coeff = {
        "P1": [1.0, 2.0],
        "P2": [3.0, 4.0],
        "P4": [5.0, 6.0],
        "P5": [7.0, 8.0],
    }
    if plane is None:
        treatment = control = mismatch = dot = None
        correction_l2 = 0.0
    else:
        a, b = coeff[plane]
        treatment = (a * a + b * b) ** 0.5
        control = treatment
        mismatch = 0.0
        dot = 0.0
        correction_l2 = treatment
    return {
        "condition": condition,
        "selected_plane": plane,
        "intervention_kind": kind,
        "token_index": 10,
        "orientation": 1,
        "branch_sign": 1,
        "coefficient_source": "branch_local_native_residual_coordinates",
        "native_residual_coefficients": coeff,
        "treatment_correction_l2": treatment,
        "control_correction_l2": control,
        "treatment_control_l2_mismatch": mismatch,
        "treatment_control_dot": dot,
        "condition_correction_l2": correction_l2,
        "pp3_native_coefficients": [0.2, -0.3],
        "pp3_post_coefficients": [0.2, -0.3],
        "pp3_coefficient_drift_max_abs": 0.0,
        "residual_post_condition_projections": {
            "P1": [1.0, 2.0],
            "P2": [3.0, 4.0],
            "P4": [5.0, 6.0],
            "P5": [7.0, 8.0],
        },
        "other_residual_plane_drift_max_abs": 0.0,
        "target_plane_neutralization_max_abs_projection":
            0.0 if kind == "neutralized" else None,
        "probe_correction_l2": r.EPS,
        "applied_correction_max_abs_residual": 0.0,
    }


def signed(condition: str, orientation: int, value: float) -> dict:
    audits = {}
    for role, sign in (("tp", 1), ("tm", -1)):
        audit = synthetic_audit(condition)
        audit["orientation"] = orientation
        audit["branch_sign"] = sign
        audit["token_index"] = 10 if role == "tp" else 11
        audits[role] = audit
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


SCALES = {
    "native": 1.0,
    "p1_neutralized": .80,
    "p1_quarter_turn_control": .95,
    "p2_neutralized": .82,
    "p2_quarter_turn_control": .94,
    "p4_neutralized": .84,
    "p4_quarter_turn_control": .93,
    "p5_neutralized": .86,
    "p5_quarter_turn_control": .92,
}


def item(index: int) -> dict:
    conditions = [
        condition(name, SCALES[name])
        for name in r.CONDITIONS
    ]
    by = {c["condition"]: c for c in conditions}
    endpoints = r.endpoints_from_conditions(by)
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
        "residual_plane_order": list(r.RESIDUAL_PLANES),
        "condition_order": list(r.CONDITIONS),
        "direction_order": list(r.DIRECTIONS),
        "conditions": conditions,
        **endpoints,
        "baseline_model_forward_count_this_run": 0,
        "scientific_model_forward_count_this_run": r.F_PAIR,
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
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
        "pair_id_first": "xg1_fact_1801",
        "pair_id_last": "xg1_fact_2100",
        "epsilon": r.EPS,
        "residual_plane_order": list(r.RESIDUAL_PLANES),
        "condition_order": list(r.CONDITIONS),
        "direction_order": list(r.DIRECTIONS),
        "model_forwards_per_direction": r.F_DIR,
        "model_forwards_per_condition": r.F_COND,
        "model_forwards_per_pair": r.F_PAIR,
        "scientific_model_forward_count_this_run": r.F_TOTAL,
        "baseline_model_forward_count_this_run": 0,
        "gpu_count": r.GPU_COUNT,
        "parallelization": "independent_pair_shards_spawn",
        "shards": shards,
        "primary_endpoint_definition":
            "for each k in {P1,P2,P4,P5}: D_k=QC_k-QN_k",
        "planned_raw_confirmatory_p_value_count":
            r.PLANNED_RAW_P_VALUE_COUNT,
        "planned_multiplicity_method": r.PLANNED_MULTIPLICITY,
        "planned_familywise_alpha": r.PLANNED_FWER_ALPHA,
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
    assert r.DESIGN_COMMIT == "d046a8e03e7522a72dfbd08cc9129b769cd5686a"
    assert r.STATIC_COMMIT == "ab9a6ebbc95bec20e8682b9538365f53167f108e"
    assert r.AUTHORITY_COMMIT == "ac13d93785719345fc0362298c9988bf040e694d"
    assert r.AUTHORITY_BLOB == "97da88df71d348e1f2fd33d6b44dbf531e4dee00"
    assert r.RESIDUAL_PLANES == ("P1", "P2", "P4", "P5")
    assert r.CONDITIONS == (
        "native",
        "p1_neutralized",
        "p1_quarter_turn_control",
        "p2_neutralized",
        "p2_quarter_turn_control",
        "p4_neutralized",
        "p4_quarter_turn_control",
        "p5_neutralized",
        "p5_quarter_turn_control",
    )
    assert r.F_PAIR == 360
    assert r.F_TOTAL == 108000
    assert r.SHARDS[0]["pair_first"] == "xg1_fact_1801"
    assert r.SHARDS[0]["pair_last"] == "xg1_fact_1950"
    assert r.SHARDS[1]["pair_first"] == "xg1_fact_1951"
    assert r.SHARDS[1]["pair_last"] == "xg1_fact_2100"
    assert r.SHARDS[0]["forward_budget"] == 54000
    assert r.SHARDS[1]["forward_budget"] == 54000
    r.validate_shards()


def test_frozen_static_inputs_and_plane_authentication():
    validated = r.validate_static_inputs()
    assert validated["preparation"]["scientific_model_forward_count"] == 0
    assert validated["preparation"]["checkpoint_load_count"] == 0
    assert validated["preparation"]["gpu_used"] is False
    assert validated["preparation"]["future_execution_contract"][
        "raw_confirmatory_p_value_count"
    ] == 4

    planes = r.load_planes()
    assert set(planes) == {
        "p1_plus", "p1_minus",
        "p2_plus", "p2_minus",
        "p4_plus", "p4_minus",
        "p5_plus", "p5_minus",
        "pp3_plus", "pp3_minus",
    }


@pytest.mark.parametrize("plane", r.RESIDUAL_PLANES)
def test_individual_plane_neutralization_and_quarter_turn_math(plane):
    planes = synthetic_planes()
    h = torch.zeros(r.DIM, dtype=torch.float64)
    h[:10] = torch.tensor(
        [1., 2., 3., 4., 5., 6., 7., 8., .25, -.5],
        dtype=torch.float64,
    )

    neutral = r.condition_correction(
        h, r.neutral_condition(plane), planes
    )
    control = r.condition_correction(
        h, r.control_condition(plane), planes
    )

    offsets = {"P1": 0, "P2": 2, "P4": 4, "P5": 6}
    offset = offsets[plane]
    a = float(h[offset])
    b = float(h[offset + 1])

    expected_neutral = torch.zeros(r.DIM, dtype=torch.float64)
    expected_neutral[offset] = -a
    expected_neutral[offset + 1] = -b
    assert torch.equal(neutral["d"], expected_neutral)

    expected_control = torch.zeros(r.DIM, dtype=torch.float64)
    expected_control[offset] = b
    expected_control[offset + 1] = -a
    assert torch.equal(control["d"], expected_control)

    assert neutral["treatment_control_l2_mismatch"] <= r.TOL
    assert abs(neutral["treatment_control_dot"]) <= r.TOL
    assert neutral["pp3_coefficient_drift_max_abs"] <= r.TOL
    assert control["pp3_coefficient_drift_max_abs"] <= r.TOL
    assert neutral["other_residual_plane_drift_max_abs"] <= r.TOL
    assert control["other_residual_plane_drift_max_abs"] <= r.TOL
    assert neutral[
        "target_plane_neutralization_max_abs_projection"
    ] <= r.TOL
    assert control[
        "target_plane_neutralization_max_abs_projection"
    ] is None


def test_native_condition_has_zero_correction():
    value = r.condition_correction(
        torch.zeros(r.DIM, dtype=torch.float64),
        "native",
        synthetic_planes(),
    )
    assert torch.count_nonzero(value["d"]).item() == 0
    assert value["selected_plane"] is None
    assert value["intervention_kind"] == "native"


def test_plane_endpoint_identity():
    result = r.plane_endpoint(10.0, 6.0, 8.5)
    assert result == {
        "QN": 6.0,
        "QC": 8.5,
        "A_N": 4.0,
        "A_C": 1.5,
        "D": 2.5,
    }
    assert result["D"] == result["QC"] - result["QN"]


def test_endpoint_canonicalizes_cancellation_sensitive_identity():
    q0 = 1.0
    qn = 0.1
    qc = 0.2
    cancellation_path = (q0 - qn) - (q0 - qc)
    canonical = qc - qn
    assert cancellation_path != canonical
    result = r.plane_endpoint(q0, qn, qc)
    assert result["A_N"] == q0 - qn
    assert result["A_C"] == q0 - qc
    assert result["D"] == canonical


def test_no_baseline_forward_accounting():
    value = item(0)
    assert value["scientific_model_forward_count_this_run"] == 360
    assert value["baseline_model_forward_count_this_run"] == 0
    assert sum(
        c["scientific_model_forward_count"]
        for c in value["conditions"]
    ) == 360
    r.validate_item(value, r.expected_pairs()[0], 0)


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
    rows[0]["plane_endpoints"]["P1"]["D"] += 1.0
    raw = r.jsonl(rows)
    rewrite_artifact_hashes(out, r.ITEM_FILE, raw)
    with pytest.raises(
        r.ResidualIndividualPlaneNecessityError,
        match="ENDPOINT:P1:D",
    ):
        r.validate_artifact(out)


def test_raw_artifact_rejects_provenance_order_and_budget_mutations(tmp_path: Path):
    items = [item(i) for i in range(r.N)]

    out = tmp_path / "provenance"
    r.write_outputs(out, items, summary())
    value = json.loads((out / r.SUMMARY_FILE).read_text(encoding="utf-8"))
    value["implementation_authority_commit"] = "0" * 40
    raw = r.canonical(value)
    rewrite_artifact_hashes(out, r.SUMMARY_FILE, raw)
    with pytest.raises(
        r.ResidualIndividualPlaneNecessityError,
        match="SUMMARY_PROVENANCE",
    ):
        r.validate_artifact(out)

    out2 = tmp_path / "order"
    r.write_outputs(out2, items, summary())
    value = json.loads((out2 / r.SUMMARY_FILE).read_text(encoding="utf-8"))
    value["condition_order"] = list(reversed(r.CONDITIONS))
    raw = r.canonical(value)
    rewrite_artifact_hashes(out2, r.SUMMARY_FILE, raw)
    with pytest.raises(
        r.ResidualIndividualPlaneNecessityError,
        match="SUMMARY_ORDER",
    ):
        r.validate_artifact(out2)

    out3 = tmp_path / "budget"
    r.write_outputs(out3, items, summary())
    value = json.loads((out3 / r.SUMMARY_FILE).read_text(encoding="utf-8"))
    value["scientific_model_forward_count_this_run"] = 107999
    raw = r.canonical(value)
    rewrite_artifact_hashes(out3, r.SUMMARY_FILE, raw)
    with pytest.raises(
        r.ResidualIndividualPlaneNecessityError,
        match="SUMMARY_BUDGET",
    ):
        r.validate_artifact(out3)


def test_item_boundary_rejects_inference_multiplicity_or_conclusion():
    value = item(0)
    value["primary_inference_executed"] = True
    with pytest.raises(
        r.ResidualIndividualPlaneNecessityError,
        match="ITEM_BOUNDARY",
    ):
        r.validate_item(value, r.expected_pairs()[0], 0)

    value = item(0)
    value["multiplicity_correction_executed"] = True
    with pytest.raises(
        r.ResidualIndividualPlaneNecessityError,
        match="ITEM_BOUNDARY",
    ):
        r.validate_item(value, r.expected_pairs()[0], 0)

    value = item(0)
    value["scientific_conclusion"] = "not-allowed"
    with pytest.raises(
        r.ResidualIndividualPlaneNecessityError,
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
        r.ResidualIndividualPlaneNecessityError,
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
    assert "adjusted_p" not in lower
    assert "defholm" not in compact.lower()
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
