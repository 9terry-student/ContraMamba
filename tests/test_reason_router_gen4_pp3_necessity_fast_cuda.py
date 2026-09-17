from __future__ import annotations
import copy, hashlib, json, math
from pathlib import Path

import pytest
import torch

from scripts import reason_router_gen4_pp3_necessity_fast_cuda as r


def planes() -> dict[str, torch.Tensor]:
    out = {}
    for name, index in (
        ("pp3_plus", 0),
        ("pp3_minus", 1),
        ("pp5_plus", 2),
        ("pp5_minus", 3),
    ):
        v = torch.zeros(r.DIM, dtype=torch.float64)
        v[index] = 1.0
        out[name] = v
    return out


def test_frozen_constants_population_budget_and_order():
    assert r.AUTHORITY_COMMIT == "0f428d84a5065268c9a4dfaf80855483c4e99c42"
    assert r.STATIC_COMMIT == "e470f132a37c731754feb4333dcb2e48e29af53b"
    assert r.SOURCE_SHA == "49bec37150630d31bb5f502f49ef23ffc9a75bb93e079127c8c430aae3da6abd"
    assert r.ROWS_SHA == "e03534599c07201e371eb07938d8492d22a30de39dcbbb3c0c22300a4ff94224"
    assert r.STRUCT_SHA == "219994e7ec757148ffcae30a7d50357e347530e06cecbb618db619b73bbe8b76"
    assert r.ANCHOR_SHA == "67d74cd2b3ab8daa8065227b1e85ea125ca88af4de64c872e5d47f60d9c40ec2"
    assert r.PLAN_SHA["xg2"] == "b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c"
    assert r.PLAN_SHA["xg4"] == "792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f"
    assert r.expected_pairs()[0] == "xg1_fact_601"
    assert r.expected_pairs()[-1] == "xg1_fact_900"
    assert len(r.expected_pairs()) == 300
    assert r.CONDITIONS == ("native", "pp3_neutralized", "pp5_coefficient_control")
    assert r.DIRECTIONS == tuple([f"xg2_{i}" for i in range(5)] + [f"xg4_{i}" for i in range(5)])
    assert r.F_SIGNED == 2
    assert r.F_DIR == 4
    assert r.F_COND == 40
    assert r.F_PAIR == 120
    assert r.F_TOTAL == 36000


def test_pp3_neutralization_and_matched_pp5_coefficient_transfer():
    p = planes()
    h = torch.zeros(r.DIM, dtype=torch.float64)
    h[0], h[1] = 3.0, -4.0
    h[2], h[3] = 100.0, -200.0  # deliberately different native PP5 coordinates

    c3 = r.condition_correction(h, "pp3_neutralized", p)
    c5 = r.condition_correction(h, "pp5_coefficient_control", p)

    assert c3["a"] == 3.0 and c3["b"] == -4.0
    assert c5["a"] == 3.0 and c5["b"] == -4.0
    assert torch.equal(c3["d"], torch.tensor([-3.0, 4.0] + [0.0] * (r.DIM - 2), dtype=torch.float64))
    expected5 = torch.zeros(r.DIM, dtype=torch.float64)
    expected5[2], expected5[3] = -3.0, 4.0
    assert torch.equal(c5["d"], expected5)
    assert c3["l2"] == 5.0
    assert c5["l2"] == 5.0
    assert abs(c3["res_plus"]) <= r.TOL
    assert abs(c3["res_minus"]) <= r.TOL


def test_direct_correction_formula_and_hook_isolation():
    p = planes()
    runtime = r.holdout.phase1.base.prevalence_eq
    core = runtime.core
    width = core.INTERMEDIATE_SIZE

    before = torch.arange(
        1 * 4 * 2 * width,
        dtype=torch.float32,
    ).reshape(1, 4, 2 * width) / 1000.0

    mask = torch.zeros(width, dtype=torch.bool)
    mask[: r.DIM] = True

    direction = torch.zeros(r.DIM, dtype=torch.float64)
    direction[4] = 1.0

    # Native coefficients a,b are nonzero at target strong channels 0,1.
    target = 2
    native = before[0, target, :width][mask].to(torch.float64)
    a = float(native[0])
    b = float(native[1])

    audit = {}
    after = r.apply_hook(
        before,
        token_index=target,
        strong_mask=mask,
        condition="pp3_neutralized",
        planes=p,
        direction=direction,
        orientation=1,
        branch_sign=-1,
        audit=audit,
    )

    expected = torch.zeros(r.DIM, dtype=torch.float64)
    expected[0] = -a
    expected[1] = -b
    expected[4] = -r.EPS
    realized = (
        after[0, target, :width][mask]
        - before[0, target, :width][mask]
    ).to(torch.float64)

    assert torch.allclose(realized, expected.to(torch.float32).to(torch.float64), atol=1e-6, rtol=0)
    assert torch.equal(after[:, :, width:], before[:, :, width:])  # gate half exact
    assert torch.equal(after[:, :target, :], before[:, :target, :])
    assert torch.equal(after[:, target + 1 :, :], before[:, target + 1 :, :])
    assert torch.equal(
        after[:, :, :width][:, :, ~mask],
        before[:, :, :width][:, :, ~mask],
    )
    assert audit["coefficient_source"] == "native_pp3_coordinates"
    assert audit["branch_sign"] == -1
    assert audit["orientation"] == 1
    assert abs(audit["probe_correction_l2"] - r.EPS) <= r.TOL


def branch_audit(condition: str, sign: int) -> dict:
    return {
        "condition": condition,
        "token_index": 10 if sign == 1 else 11,
        "orientation": 1,
        "branch_sign": sign,
        "coefficient_source": "native_pp3_coordinates",
        "native_pp3_a": 2.0,
        "native_pp3_b": -1.0,
        "condition_correction_l2": 0.0 if condition == "native" else math.sqrt(5.0),
        "probe_correction_l2": r.EPS,
        "pp3_post_condition_residual_plus": 0.0,
        "pp3_post_condition_residual_minus": 0.0,
        "applied_correction_max_abs_residual": 0.0,
    }


def signed(condition: str, orientation: int) -> dict:
    # Keep F_plus/F_minus algebra simple and finite.
    value = 1.0 if orientation == 1 else 0.0
    audits = {
        "tp": branch_audit(condition, 1),
        "tm": branch_audit(condition, -1),
    }
    for a in audits.values():
        a["orientation"] = orientation
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
    pos = signed(condition, 1)
    neg = signed(condition, -1)
    j = (pos["F"] - neg["F"]) / (2.0 * r.EPS)
    family, index = key.split("_")
    return {
        "direction_key": key,
        "basis_family": family,
        "basis_index": int(index),
        "F_plus": pos["F"],
        "F_minus": neg["F"],
        "J": j,
        "J_squared": j * j,
        "positive_probe": pos,
        "negative_probe": neg,
        "model_forward_count": r.F_DIR,
    }


def condition(condition_name: str) -> dict:
    ps = [direction(condition_name, key) for key in r.DIRECTIONS]
    e2 = sum(x["J_squared"] for x in ps[: r.K]) / r.K
    e4 = sum(x["J_squared"] for x in ps[r.K :]) / r.K
    return {
        "condition": condition_name,
        "direction_order": list(r.DIRECTIONS),
        "direction_probes": ps,
        "E_XG2": e2,
        "E_XG4": e4,
        "Q": e2 - e4,
        "scientific_model_forward_count": r.F_COND,
    }


def item(pair: str, index: int) -> dict:
    cs = [condition(c) for c in r.CONDITIONS]
    ep = r.endpoint(cs[0]["Q"], cs[1]["Q"], cs[2]["Q"])
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
        "conditions": cs,
        **ep,
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
        "pair_id_first": "xg1_fact_601",
        "pair_id_last": "xg1_fact_900",
        "epsilon": r.EPS,
        "condition_order": list(r.CONDITIONS),
        "direction_order": list(r.DIRECTIONS),
        "model_forwards_per_direction": r.F_DIR,
        "model_forwards_per_condition": r.F_COND,
        "model_forwards_per_pair": r.F_PAIR,
        "representative_checkpoint_sha256": (
            r.holdout.phase1.base.prevalence_eq.extraction.REPRESENTATIVE_CHECKPOINT_SHA256
        ),
        "scientific_model_forward_count_this_run": r.F_TOTAL,
        "baseline_model_forward_count_this_run": 0,
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
    }


def test_endpoint_algebra_rejects_mutation():
    x = item("xg1_fact_601", 0)
    r.validate_endpoint(x)
    x["D_NEC"] = 1.0
    with pytest.raises(r.PP3NecessityError):
        r.validate_endpoint(x)


def test_artifact_validator_rejects_altered_endpoint_algebra(tmp_path: Path):
    items = [item(pair, i) for i, pair in enumerate(r.expected_pairs())]
    out = tmp_path / "artifact"
    r.write_outputs(out, items, summary())
    assert r.validate_artifact(out)["summary"]["result"] == r.RESULT_PASS

    rows = r.read_jsonl(out / r.ITEM_FILE)
    rows[0]["D_NEC"] = 1.0
    raw = r.jsonl(rows)
    (out / r.ITEM_FILE).write_bytes(raw)

    manifest = json.loads((out / r.MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["files"][r.ITEM_FILE]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["files"][r.ITEM_FILE]["bytes"] = len(raw)
    manifest_raw = r.canonical(manifest)
    (out / r.MANIFEST_FILE).write_bytes(manifest_raw)

    hashes = {
        r.ITEM_FILE: hashlib.sha256(raw).hexdigest(),
        r.SUMMARY_FILE: r.sha256_file(out / r.SUMMARY_FILE),
        r.MANIFEST_FILE: hashlib.sha256(manifest_raw).hexdigest(),
    }
    (out / r.CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(r.PP3NecessityError, match="ENDPOINT"):
        r.validate_artifact(out)



def test_item_validator_rejects_provenance_mutation():
    base = item("xg1_fact_601", 0)
    r.validate_item(base, "xg1_fact_601", 0)

    mutations = (
        ("family_key", "xg2"),
        ("implementation_authority_commit", "deadbeef"),
        ("static_preparation_freeze_commit", "deadbeef"),
        ("epsilon", 0.05),
    )
    for field, value in mutations:
        changed = copy.deepcopy(base)
        changed[field] = value
        with pytest.raises(r.PP3NecessityError):
            r.validate_item(changed, "xg1_fact_601", 0)


def test_artifact_validator_rejects_summary_provenance_mutation(tmp_path: Path):
    items = [item(pair, i) for i, pair in enumerate(r.expected_pairs())]
    out = tmp_path / "artifact-summary-provenance"
    r.write_outputs(out, items, summary())

    s = json.loads((out / r.SUMMARY_FILE).read_text(encoding="utf-8"))
    s["implementation_authority_commit"] = "deadbeef"
    raw = r.canonical(s)
    (out / r.SUMMARY_FILE).write_bytes(raw)

    manifest = json.loads((out / r.MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["files"][r.SUMMARY_FILE]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["files"][r.SUMMARY_FILE]["bytes"] = len(raw)
    manifest_raw = r.canonical(manifest)
    (out / r.MANIFEST_FILE).write_bytes(manifest_raw)

    hashes = {
        r.ITEM_FILE: r.sha256_file(out / r.ITEM_FILE),
        r.SUMMARY_FILE: hashlib.sha256(raw).hexdigest(),
        r.MANIFEST_FILE: hashlib.sha256(manifest_raw).hexdigest(),
    }
    (out / r.CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(r.PP3NecessityError, match="SUMMARY_AUTHORITY"):
        r.validate_artifact(out)

def test_runner_contains_no_statistical_inference_implementation():
    source = Path(r.__file__).read_text(encoding="utf-8").lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert '"primary_inference_executed":false' in source.replace(" ", "")
    assert '"multiplicity_correction_executed":false' in source.replace(" ", "")
    assert '"scientific_conclusion":none' in source.replace(" ", "")


def test_static_tests_do_not_execute_model_or_cuda(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("scientific runtime must not execute in unit tests")

    monkeypatch.setattr(r, "run_observation", blocked)
    # Pure helpers remain usable and no checkpoint/model/CUDA path is invoked.
    p = planes()
    h = torch.zeros(r.DIM, dtype=torch.float64)
    assert r.condition_correction(h, "native", p)["l2"] == 0.0
