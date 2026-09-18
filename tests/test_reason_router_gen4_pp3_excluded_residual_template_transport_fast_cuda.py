from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_pp3_excluded_residual_template_transport_fast_cuda
    as r
)


def synthetic_geometry():
    union = torch.zeros((r.DIM, 10), dtype=torch.float64)
    for i in range(10):
        union[i, i] = 1.0
    plus = union[:, :5].clone()
    minus = union[:, 5:].clone()
    return {
        "union": union,
        "union_gram": torch.eye(10, dtype=torch.float64),
        "plus": plus,
        "minus": minus,
        "lambda": [1.0] * 5,
    }


def signed(orientation: int, value: float) -> dict:
    audits = {}
    for role, sign in (("tp", 1), ("tm", -1)):
        audits[role] = {
            "token_index": 10 if role == "tp" else 11,
            "orientation": orientation,
            "branch_sign": sign,
            "probe_correction_l2": r.EPS,
            "applied_correction_max_abs_residual": 0.0,
            "native_state_direct_correction_applied": False,
        }
    return {
        "orientation": orientation,
        "F": value,
        "plus_path_efficiency": value,
        "minus_path_efficiency": 0.0,
        "branch_audits": audits,
        "model_forward_count": r.F_SIGNED,
    }


def direction(key: str, j: float) -> dict:
    # Mirror production exactly: J is the stored result of the same
    # central-difference arithmetic applied to F+ and F-. Do not inject the
    # idealized input j directly, because binary floating arithmetic such as
    # (j*EPS - (-j*EPS))/(2*EPS) is not guaranteed bit-identical to j.
    fp = j * r.EPS
    fm = -j * r.EPS
    j_value = (fp - fm) / (2.0 * r.EPS)
    positive = signed(1, fp)
    negative = signed(-1, fm)
    family, index = key.split("_")
    return {
        "direction_key": key,
        "basis_family": family,
        "basis_index": int(index),
        "F_plus": fp,
        "F_minus": fm,
        "J": j_value,
        "J_squared": j_value * j_value,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": r.F_DIR,
    }


def item(index: int) -> dict:
    pair = r.expected_pairs()[index]
    js2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    js4 = [0.5, 1.0, 1.5, 2.0, 2.5]
    probes = [
        direction(key, value)
        for key, value in zip(
            r.DIRECTIONS,
            js2 + js4,
            strict=True,
        )
    ]
    endpoint = r.decompose_endpoint(probes, synthetic_geometry())
    return {
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": index,
        "target_plus_anchor": 1,
        "target_minus_anchor": 1,
        "reference_plus_anchor": 1,
        "reference_minus_anchor": 1,
        "schema_version": r.ITEM_SCHEMA,
        "design_commit": r.DESIGN_COMMIT,
        "static_preparation_freeze_commit": r.STATIC_COMMIT,
        "epsilon": r.EPS,
        "direction_order": list(r.DIRECTIONS),
        "direction_probes": probes,
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
    shards = []
    for shard in r.SHARDS:
        shards.append({
            "shard_id": shard["shard_id"],
            "gpu_id": shard["gpu_id"],
            "device_name": f"synthetic-gpu-{shard['gpu_id']}",
            "pair_first": shard["pair_first"],
            "pair_last": shard["pair_last"],
            "pair_count": shard["pair_count"],
            "scientific_model_forward_count_this_run":
                shard["forward_budget"],
            "checkpoint_sha256": checkpoint,
        })
    return {
        "schema_version": r.SUMMARY_SCHEMA,
        "result": r.RESULT_PASS,
        "execution_head": "synthetic-head",
        "design_commit": r.DESIGN_COMMIT,
        "static_preparation_freeze_commit": r.STATIC_COMMIT,
        "source_pair_count": r.N,
        "pair_id_first": "xg1_fact_1201",
        "pair_id_last": "xg1_fact_1500",
        "epsilon": r.EPS,
        "direction_order": list(r.DIRECTIONS),
        "residual_plane_order": list(r.RESIDUAL_PLANES),
        "model_forwards_per_direction": r.F_DIR,
        "model_forwards_per_pair": r.F_PAIR,
        "scientific_model_forward_count_this_run": r.F_TOTAL,
        "baseline_model_forward_count_this_run": 0,
        "gpu_count": r.GPU_COUNT,
        "parallelization": "independent_pair_shards_spawn",
        "shards": shards,
        "primary_endpoint_definition": "D_TEMPLATE=C_XG2-C_XG4",
        "template_cosine": r.TEMPLATE_COSINE,
        "xg2_unit_template": list(r.XG2_UNIT_TEMPLATE),
        "xg4_unit_template": list(r.XG4_UNIT_TEMPLATE),
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
        "representative_checkpoint_sha256": checkpoint,
    }


def test_frozen_constants_and_two_gpu_sharding():
    assert r.DESIGN_COMMIT == "d3cc008fad221862e6fe9718b67b6ba0c87d0368"
    assert r.STATIC_COMMIT == "6d65de694f112b051366d8ccf7e4617caafba1b6"
    assert r.SOURCE_SHA == "9b93eff2399b7f90fb3d63f28f834fb986efcac9d347038c4b265519062d71a6"
    assert r.ROWS_SHA == "f60cc028e8276279499290976f18ba503fc7a35a4232c1d768b822075d87ec5e"
    assert r.GPU_COUNT == 2
    assert r.F_PAIR == 40
    assert r.F_TOTAL == 12000
    assert r.SHARDS[0]["pair_first"] == "xg1_fact_1201"
    assert r.SHARDS[0]["pair_last"] == "xg1_fact_1350"
    assert r.SHARDS[1]["pair_first"] == "xg1_fact_1351"
    assert r.SHARDS[1]["pair_last"] == "xg1_fact_1500"
    assert r.SHARDS[0]["forward_budget"] == 6000
    assert r.SHARDS[1]["forward_budget"] == 6000
    r.validate_shards()


def test_decomposition_and_template_endpoint_identity():
    probes = [
        direction(key, value)
        for key, value in zip(
            r.DIRECTIONS,
            [1, 2, 3, 4, 5, 0.5, 1, 1.5, 2, 2.5],
            strict=True,
        )
    ]
    out = r.decompose_endpoint(probes, synthetic_geometry())

    # Mirror production exactly: expected principal-plane contributions
    # must be derived from the actual stored J values, not from idealized
    # decimal inputs that may differ by a last-bit floating representation.
    stored_j2 = [float(p["J"]) for p in probes[: r.K]]
    stored_j4 = [float(p["J"]) for p in probes[r.K :]]
    expected_plane = [
        (stored_j2[i] ** 2 - stored_j4[i] ** 2) / r.K
        for i in range(r.K)
    ]
    assert [out["principal_plane_net"][f"P{i+1}"] for i in range(5)] == expected_plane
    assert out["residual_net_vector"] == [
        expected_plane[0],
        expected_plane[1],
        expected_plane[3],
        expected_plane[4],
    ]
    assert abs(out["Q_reconstruction_residual"]) <= r.Q_RECON_TOL
    assert out["D_TEMPLATE"] == out["C_XG2"] - out["C_XG4"]


def test_apply_probe_hook_isolates_target_coordinate():
    runtime = r.holdout.phase1.base.prevalence_eq
    core = runtime.core
    width = core.INTERMEDIATE_SIZE

    before = (
        torch.arange(1 * 4 * 2 * width, dtype=torch.float32)
        .reshape(1, 4, 2 * width)
        / 1000.0
    )
    mask = torch.zeros(width, dtype=torch.bool)
    mask[: r.DIM] = True
    direction_value = torch.zeros(r.DIM, dtype=torch.float64)
    direction_value[3] = 1.0
    audit = {}

    after = r.apply_probe_hook(
        before,
        token_index=2,
        strong_mask=mask,
        direction=direction_value,
        orientation=1,
        branch_sign=-1,
        audit=audit,
    )
    realized = (
        after[0, 2, :width][mask]
        - before[0, 2, :width][mask]
    ).to(torch.float64)
    expected = torch.zeros(r.DIM, dtype=torch.float64)
    expected[3] = -r.EPS

    assert torch.allclose(
        realized,
        expected.to(torch.float32).to(torch.float64),
        atol=1e-6,
        rtol=0,
    )
    assert torch.equal(after[:, :, width:], before[:, :, width:])
    assert torch.equal(after[:, :2, :], before[:, :2, :])
    assert torch.equal(after[:, 3:, :], before[:, 3:, :])
    assert audit["native_state_direct_correction_applied"] is False
    assert abs(audit["probe_correction_l2"] - r.EPS) <= r.TOL


def test_canonical_two_shard_merge_matches_full_order():
    all_items = [item(i) for i in range(r.N)]
    payloads = []
    checkpoint = (
        r.holdout.phase1.base.prevalence_eq.extraction
        .REPRESENTATIVE_CHECKPOINT_SHA256
    )
    for shard in r.SHARDS:
        shard_items = all_items[
            shard["start_index"]:shard["end_index"]
        ]
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
            "items": shard_items,
        })

    merged, meta = r.merge_shards(list(reversed(payloads)))
    assert [x["source_pair_id"] for x in merged] == list(r.expected_pairs())
    assert len(meta) == 2
    assert meta[0]["gpu_id"] == 0
    assert meta[1]["gpu_id"] == 1


def test_merge_rejects_overlap_or_wrong_shard_identity():
    checkpoint = (
        r.holdout.phase1.base.prevalence_eq.extraction
        .REPRESENTATIVE_CHECKPOINT_SHA256
    )
    payloads = []
    for shard in r.SHARDS:
        payloads.append({
            "shard_id": shard["shard_id"],
            "gpu_id": shard["gpu_id"],
            "device_name": "gpu",
            "pair_first": shard["pair_first"],
            "pair_last": shard["pair_last"],
            "pair_count": shard["pair_count"],
            "scientific_model_forward_count_this_run":
                shard["forward_budget"],
            "checkpoint_sha256": checkpoint,
            "items": [
                item(i)
                for i in range(
                    shard["start_index"],
                    shard["end_index"],
                )
            ],
        })

    broken = copy.deepcopy(payloads)
    broken[1]["items"][0]["source_pair_id"] = "xg1_fact_1350"
    with pytest.raises(r.ResidualTemplateTransportError):
        r.merge_shards(broken)


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


def test_artifact_roundtrip_and_endpoint_mutation_rejection(tmp_path: Path):
    items = [item(i) for i in range(r.N)]
    out = tmp_path / "artifact"
    r.write_outputs(out, items, summary())
    validated = r.validate_artifact(out)
    assert validated["summary"]["result"] == r.RESULT_PASS

    rows = r.read_jsonl(out / r.ITEM_FILE)
    rows[0]["D_TEMPLATE"] += 1.0
    raw = r.jsonl(rows)
    rewrite_artifact_hashes(out, r.ITEM_FILE, raw)

    with pytest.raises(
        r.ResidualTemplateTransportError,
        match="TEMPLATE_ENDPOINT",
    ):
        r.validate_artifact(out)


def test_summary_budget_and_gpu_topology_mutation_rejected(tmp_path: Path):
    items = [item(i) for i in range(r.N)]
    out = tmp_path / "artifact"
    r.write_outputs(out, items, summary())

    value = json.loads(
        (out / r.SUMMARY_FILE).read_text(encoding="utf-8")
    )
    value["scientific_model_forward_count_this_run"] = 11999
    raw = r.canonical(value)
    rewrite_artifact_hashes(out, r.SUMMARY_FILE, raw)
    with pytest.raises(
        r.ResidualTemplateTransportError,
        match="SUMMARY_BUDGET",
    ):
        r.validate_artifact(out)


def test_runner_contains_no_statistical_inference():
    source = Path(r.__file__).read_text(encoding="utf-8").lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert "student_t" not in source
    compact = source.replace(" ", "")
    assert '"primary_inference_executed":false' in compact
    assert '"multiplicity_correction_executed":false' in compact
    assert '"scientific_conclusion":none' in compact



def test_runner_two_gpu_runtime_and_fast_capture_are_device_generic():
    source = Path(r.__file__).read_text(encoding="utf-8")
    compact = "".join(source.split())

    assert "runtime.backend.runtime_gate()" not in source
    assert "runtime.backend._make_fast_capture(kernels)" not in source
    assert '.to("cuda:0")' not in source
    assert "torch.cuda.set_device(0)" not in source

    assert "runtime_gate_for_device(runtime,gpu_id)" in compact
    assert "make_fast_capture_for_device(runtime,kernels,device)" in compact
    assert "input_ids=input_ids.detach().to(device).contiguous()" in compact
    assert "torch.cuda.synchronize(device)" in compact
    assert "torch.cuda.get_device_name(gpu_id)" in compact
    assert "torch.cuda.get_device_capability(gpu_id)" in compact
