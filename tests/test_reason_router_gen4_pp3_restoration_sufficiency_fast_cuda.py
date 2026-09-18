from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from scripts import reason_router_gen4_pp3_restoration_sufficiency_fast_cuda as r


def planes() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, index in (
        ("pp3_plus", 0),
        ("pp3_minus", 1),
        ("pp5_plus", 2),
        ("pp5_minus", 3),
    ):
        value = torch.zeros(r.DIM, dtype=torch.float64)
        value[index] = 1.0
        out[name] = value
    return out


def test_frozen_constants_population_budget_and_order():
    assert r.AUTHORITY_COMMIT == "cae566ed458e5c6f93c86ce03b029950035652dc"
    assert r.AUTHORITY_BLOB == "090b68503c3f10327065bc275f7b6daffdccfc59"
    assert r.STATIC_COMMIT == "0f907574f1ca25ec12e35573a83e1b499ed1b53b"
    assert r.SOURCE_SHA == "2c700452d818531c46a8ffd473eb6d64d8af29f3a9284da8371f5c9eb2610c21"
    assert r.ROWS_SHA == "7ec2ea86f35562394244f6df6e8b098ea3ba8a9bf358868fc614f5029746241c"
    assert r.STRUCT_SHA == "a1c6957d7d48fb93b53f7b93bb5378caf0bbdc0adb96c438f6aa56a779ae9922"
    assert r.ANCHOR_SHA == "dc4f2cd4ca2806249467407c7c980411d8fa02051418f9a8b625c1b1c4756253"
    assert r.ELIG_SHA == "0fb8da67687f223b6c72ea7bc946542e6543bb85c050163056696a8e945e5a89"
    assert r.expected_pairs()[0] == "xg1_fact_901"
    assert r.expected_pairs()[-1] == "xg1_fact_1200"
    assert len(r.expected_pairs()) == 300
    assert r.CONDITIONS == (
        "pp3_neutralized",
        "pp3_restored",
        "pp5_replacement",
    )
    assert r.DIRECTIONS == tuple(
        [f"xg2_{i}" for i in range(5)]
        + [f"xg4_{i}" for i in range(5)]
    )
    assert r.F_SIGNED == 2
    assert r.F_DIR == 4
    assert r.F_COND == 40
    assert r.F_PAIR == 120
    assert r.F_TOTAL == 36000


def test_condition_formulas_and_matched_restoration_norm():
    p = planes()
    h = torch.zeros(r.DIM, dtype=torch.float64)
    h[0] = 3.0
    h[1] = -4.0
    h[2] = 100.0
    h[3] = -200.0

    neutralized = r.condition_correction(
        h,
        "pp3_neutralized",
        p,
    )
    restored = r.condition_correction(
        h,
        "pp3_restored",
        p,
    )
    replacement = r.condition_correction(
        h,
        "pp5_replacement",
        p,
    )

    for value in (neutralized, restored, replacement):
        assert value["a"] == 3.0
        assert value["b"] == -4.0
        assert value["c3_l2"] == 5.0
        assert value["c5_l2"] == 5.0
        assert value["restoration_addition_norm_mismatch"] == 0.0

    expected_b = torch.zeros(r.DIM, dtype=torch.float64)
    expected_b[0] = -3.0
    expected_b[1] = 4.0
    assert torch.equal(neutralized["d"], expected_b)

    assert torch.equal(
        restored["d"],
        torch.zeros(r.DIM, dtype=torch.float64),
    )
    assert restored["r3_native_state_max_abs_residual"] == 0.0

    expected_r5 = torch.zeros(r.DIM, dtype=torch.float64)
    expected_r5[0] = -3.0
    expected_r5[1] = 4.0
    expected_r5[2] = 3.0
    expected_r5[3] = -4.0
    assert torch.equal(replacement["d"], expected_r5)
    assert replacement["r5_construction_max_abs_residual"] == 0.0


def test_direct_hook_formula_and_isolation():
    p = planes()
    runtime = r.holdout.phase1.base.prevalence_eq
    core = runtime.core
    width = core.INTERMEDIATE_SIZE

    before = (
        torch.arange(
            1 * 4 * 2 * width,
            dtype=torch.float32,
        ).reshape(1, 4, 2 * width)
        / 1000.0
    )

    mask = torch.zeros(width, dtype=torch.bool)
    mask[: r.DIM] = True

    direction = torch.zeros(r.DIM, dtype=torch.float64)
    direction[4] = 1.0

    target = 2
    native = before[
        0,
        target,
        :width,
    ][mask].to(torch.float64)
    a = float(native[0])
    b = float(native[1])

    audit: dict = {}
    after = r.apply_hook(
        before,
        token_index=target,
        strong_mask=mask,
        condition="pp5_replacement",
        planes=p,
        direction=direction,
        orientation=1,
        branch_sign=-1,
        audit=audit,
    )

    expected = torch.zeros(r.DIM, dtype=torch.float64)
    expected[0] = -a
    expected[1] = -b
    expected[2] = a
    expected[3] = b
    expected[4] = -r.EPS

    realized = (
        after[0, target, :width][mask]
        - before[0, target, :width][mask]
    ).to(torch.float64)

    assert torch.allclose(
        realized,
        expected.to(torch.float32).to(torch.float64),
        atol=1e-6,
        rtol=0,
    )
    assert torch.equal(after[:, :, width:], before[:, :, width:])
    assert torch.equal(after[:, :target, :], before[:, :target, :])
    assert torch.equal(
        after[:, target + 1 :, :],
        before[:, target + 1 :, :],
    )
    assert torch.equal(
        after[:, :, :width][:, :, ~mask],
        before[:, :, :width][:, :, ~mask],
    )
    assert audit["coefficient_source"] == "native_pp3_coordinates"
    assert audit["condition"] == "pp5_replacement"
    assert audit["branch_sign"] == -1
    assert audit["orientation"] == 1
    assert (
        audit["restoration_addition_norm_mismatch"]
        <= r.TOL
    )
    assert abs(audit["probe_correction_l2"] - r.EPS) <= r.TOL


def branch_audit(condition: str, sign: int) -> dict:
    a = 2.0
    b = -1.0
    component_l2 = math.sqrt(5.0)

    if condition == "pp3_neutralized":
        direct_l2 = component_l2
        post_plus = 0.0
        post_minus = 0.0
        neutral_residual = 0.0
        r3_residual = None
        r5_residual = None
    elif condition == "pp3_restored":
        direct_l2 = 0.0
        post_plus = a
        post_minus = b
        neutral_residual = None
        r3_residual = 0.0
        r5_residual = None
    else:
        direct_l2 = math.sqrt(10.0)
        post_plus = 0.0
        post_minus = 0.0
        neutral_residual = None
        r3_residual = None
        r5_residual = 0.0

    return {
        "condition": condition,
        "token_index": 10 if sign == 1 else 11,
        "orientation": 1,
        "branch_sign": sign,
        "coefficient_source": "native_pp3_coordinates",
        "native_pp3_a": a,
        "native_pp3_b": b,
        "pp3_component_l2": component_l2,
        "pp5_component_l2": component_l2,
        "restoration_addition_norm_mismatch": 0.0,
        "direct_final_state_correction_l2": direct_l2,
        "pp3_post_condition_residual_plus": post_plus,
        "pp3_post_condition_residual_minus": post_minus,
        "neutralized_construction_max_abs_residual":
            neutral_residual,
        "r3_native_state_max_abs_residual": r3_residual,
        "r5_construction_max_abs_residual": r5_residual,
        "probe_correction_l2": r.EPS,
        "applied_correction_max_abs_residual": 0.0,
    }


def signed(condition: str, orientation: int) -> dict:
    value = 1.0 if orientation == 1 else 0.0
    audits = {
        "tp": branch_audit(condition, 1),
        "tm": branch_audit(condition, -1),
    }
    for audit in audits.values():
        audit["orientation"] = orientation

    return {
        "condition": condition,
        "orientation": orientation,
        "plus_path_efficiency": value,
        "minus_path_efficiency": 0.0,
        "F": value,
        "branch_audits": audits,
        "model_forward_count": r.F_SIGNED,
    }


def direction(condition: str, key: str) -> dict:
    positive = signed(condition, 1)
    negative = signed(condition, -1)
    j_value = (
        positive["F"] - negative["F"]
    ) / (2.0 * r.EPS)
    family, index = key.split("_")

    return {
        "direction_key": key,
        "basis_family": family,
        "basis_index": int(index),
        "F_plus": positive["F"],
        "F_minus": negative["F"],
        "J": j_value,
        "J_squared": j_value * j_value,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": r.F_DIR,
    }


def condition(condition_name: str) -> dict:
    probes = [
        direction(condition_name, key)
        for key in r.DIRECTIONS
    ]
    e_xg2 = sum(
        value["J_squared"]
        for value in probes[: r.K]
    ) / r.K
    e_xg4 = sum(
        value["J_squared"]
        for value in probes[r.K :]
    ) / r.K

    return {
        "condition": condition_name,
        "direction_order": list(r.DIRECTIONS),
        "direction_probes": probes,
        "E_XG2": e_xg2,
        "E_XG4": e_xg4,
        "Q": e_xg2 - e_xg4,
        "scientific_model_forward_count": r.F_COND,
    }


def item(pair: str, index: int) -> dict:
    conditions = [
        condition(name)
        for name in r.CONDITIONS
    ]
    # These synthetic conditions all have Q=0, giving an exact endpoint.
    endpoint = r.endpoint(
        conditions[0]["Q"],
        conditions[1]["Q"],
        conditions[2]["Q"],
    )
    return {
        "schema_version": r.ITEM_SCHEMA,
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": index,
        "target_plus_anchor": 1,
        "target_minus_anchor": 1,
        "reference_plus_anchor": 1,
        "reference_minus_anchor": 1,
        "implementation_authority_commit": r.AUTHORITY_COMMIT,
        "static_preparation_freeze_commit": r.STATIC_COMMIT,
        "epsilon": r.EPS,
        "condition_order": list(r.CONDITIONS),
        "direction_order": list(r.DIRECTIONS),
        "conditions": conditions,
        **endpoint,
        "baseline_model_forward_count_this_run": 0,
        "scientific_model_forward_count_this_run": r.F_PAIR,
    }


def summary() -> dict:
    return {
        "schema_version": r.SUMMARY_SCHEMA,
        "result": r.RESULT_PASS,
        "execution_head": "synthetic-test-head",
        "implementation_authority_commit": r.AUTHORITY_COMMIT,
        "static_preparation_freeze_commit": r.STATIC_COMMIT,
        "source_pair_count": r.N,
        "pair_id_first": "xg1_fact_901",
        "pair_id_last": "xg1_fact_1200",
        "epsilon": r.EPS,
        "condition_order": list(r.CONDITIONS),
        "direction_order": list(r.DIRECTIONS),
        "model_forwards_per_direction": r.F_DIR,
        "model_forwards_per_condition": r.F_COND,
        "model_forwards_per_pair": r.F_PAIR,
        "representative_checkpoint_sha256": (
            r.holdout.phase1.base.prevalence_eq.extraction
            .REPRESENTATIVE_CHECKPOINT_SHA256
        ),
        "scientific_model_forward_count_this_run": r.F_TOTAL,
        "baseline_model_forward_count_this_run": 0,
        "primary_endpoint_definition":
            "D_SUF=(Q_R3-Q_B)-(Q_R5-Q_B)=Q_R3-Q_R5",
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
    }


def rewrite_artifact_hashes(out: Path, changed_name: str, raw: bytes):
    (out / changed_name).write_bytes(raw)

    manifest = json.loads(
        (out / r.MANIFEST_FILE).read_text(encoding="utf-8")
    )
    manifest["files"][changed_name]["sha256"] = hashlib.sha256(
        raw
    ).hexdigest()
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


def test_endpoint_algebra_and_mutation_rejection():
    endpoint = r.endpoint(1.0, 3.0, 2.0)
    assert endpoint == {
        "Q_B": 1.0,
        "Q_R3": 3.0,
        "Q_R5": 2.0,
        "S3": 2.0,
        "S5": 1.0,
        "D_SUF": 1.0,
    }
    r.validate_endpoint(endpoint)

    changed = dict(endpoint)
    changed["D_SUF"] = 2.0
    with pytest.raises(
        r.PP3RestorationSufficiencyError,
        match="ENDPOINT",
    ):
        r.validate_endpoint(changed)


def test_item_validator_rejects_provenance_and_budget_mutation():
    base = item("xg1_fact_901", 0)
    r.validate_item(base, "xg1_fact_901", 0)

    mutations = (
        ("family_key", "xg2"),
        ("implementation_authority_commit", "deadbeef"),
        ("static_preparation_freeze_commit", "deadbeef"),
        ("epsilon", 0.05),
        ("scientific_model_forward_count_this_run", 119),
    )
    for field, value in mutations:
        changed = copy.deepcopy(base)
        changed[field] = value
        with pytest.raises(r.PP3RestorationSufficiencyError):
            r.validate_item(changed, "xg1_fact_901", 0)


def test_artifact_roundtrip_and_endpoint_mutation_rejection(
    tmp_path: Path,
):
    items = [
        item(pair, index)
        for index, pair in enumerate(r.expected_pairs())
    ]
    out = tmp_path / "artifact"
    r.write_outputs(out, items, summary())

    validated = r.validate_artifact(out)
    assert validated["summary"]["result"] == r.RESULT_PASS

    rows = r.read_jsonl(out / r.ITEM_FILE)
    rows[0]["D_SUF"] = 1.0
    raw = r.jsonl(rows)
    rewrite_artifact_hashes(out, r.ITEM_FILE, raw)

    with pytest.raises(
        r.PP3RestorationSufficiencyError,
        match="ENDPOINT",
    ):
        r.validate_artifact(out)


def test_artifact_validator_rejects_summary_budget_mutation(
    tmp_path: Path,
):
    items = [
        item(pair, index)
        for index, pair in enumerate(r.expected_pairs())
    ]
    out = tmp_path / "artifact-budget"
    r.write_outputs(out, items, summary())

    value = json.loads(
        (out / r.SUMMARY_FILE).read_text(encoding="utf-8")
    )
    value["scientific_model_forward_count_this_run"] = 35999
    raw = r.canonical(value)
    rewrite_artifact_hashes(out, r.SUMMARY_FILE, raw)

    with pytest.raises(
        r.PP3RestorationSufficiencyError,
        match="SUMMARY_BUDGET",
    ):
        r.validate_artifact(out)


def test_checksum_mutation_rejected(tmp_path: Path):
    items = [
        item(pair, index)
        for index, pair in enumerate(r.expected_pairs())
    ]
    out = tmp_path / "artifact-checksum"
    r.write_outputs(out, items, summary())

    checksum = out / r.CHECKSUM_FILE
    lines = checksum.read_text(encoding="utf-8").splitlines()
    lines[0] = "0" * 64 + lines[0][64:]
    checksum.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        r.PP3RestorationSufficiencyError,
        match="CHECKSUMS",
    ):
        r.validate_artifact(out)


def test_runner_contains_no_statistical_inference_implementation():
    source = Path(r.__file__).read_text(
        encoding="utf-8"
    ).lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert "student_t" not in source
    compact = source.replace(" ", "")
    assert '"primary_inference_executed":false' in compact
    assert '"multiplicity_correction_executed":false' in compact
    assert '"scientific_conclusion":none' in compact
