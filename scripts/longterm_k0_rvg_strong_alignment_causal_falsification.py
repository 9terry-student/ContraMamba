#!/usr/bin/env python3
"""Frozen K0 strong-side directional-alignment causal-falsification runner.

Authority:
  edc951f8e845e5e41e4f17ad3678eed297c62970

The only intervention boundary is the layer-22 mixer x-branch, immediately
after in_proj and before convolution, at relative coordinate k=2 and the
frozen strong-240 channels.

No tokenizer, logits, task head, training, parameter mutation, channel search,
or post-hoc intervention tuning is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

DESIGN_COMMIT = "edc951f8e845e5e41e4f17ad3678eed297c62970"
DESIGN_BLOB = "fcfa00d3de9e8b9c1cba200626fd7a6ee4b73220"
STOP_COMMIT = "f5cf85692bf8900f019a06979c54e637144b2d68"

GEOMETRY_EVIDENCE_COMMIT = "431e8faa6e5c82a20d87f532b4ab960fcf641ec2"
GEOMETRY_IMPLEMENTATION_COMMIT = "0b1168182a265bc405652d2ce519f63310433c3e"
GEOMETRY_RUNNER_REL = (
    "scripts/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_audit.py"
)
GEOMETRY_RUNNER_SHA256 = (
    "0c487e3d29236e298f8c74cbdbcdfce3599af587dfd27a6e5fcf169df64eade5"
)
GEOMETRY_RUNNER_BLOB = "007d9ec21487ccfe7c56d6410d1dac6f832a9a22"
GEOMETRY_RUN_DIR = (
    "reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1"
)
GEOMETRY_ITEM_REL = (
    GEOMETRY_RUN_DIR + "/layer20_y20_strong_interaction_geometry_item_metrics.jsonl"
)
GEOMETRY_CHANNEL_REL = (
    GEOMETRY_RUN_DIR + "/layer20_y20_strong_interaction_geometry_channel_validation.jsonl"
)
GEOMETRY_SUMMARY_REL = GEOMETRY_RUN_DIR + "/summary.json"
GEOMETRY_MANIFEST_REL = GEOMETRY_RUN_DIR + "/execution_manifest.json"
GEOMETRY_ITEM_SHA256 = (
    "e7002e03bd170c05ea068e70eb32cc1f7a8b0d4f569ac44e531a42a8a4e1ebfe"
)
GEOMETRY_CHANNEL_SHA256 = (
    "111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942"
)
GEOMETRY_SUMMARY_SHA256 = (
    "35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9a51467477163"
)
GEOMETRY_MANIFEST_SHA256 = (
    "1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee"
)

CARRY_EVIDENCE_COMMIT = "21b36fa4579bee53644fa7f3da84cdc947ddef5b"
CARRY_IMPLEMENTATION_COMMIT = "378ac88cf3413dc46c9c7fa153ef60785fd370a4"
CARRY_RUN_DIR = (
    "reports/longterm_k0_rvg_layer22_carry_write_factorization_378ac88_v1"
)
CARRY_ITEM_REL = CARRY_RUN_DIR + "/layer22_carry_write_factorization_metrics.jsonl"
CARRY_SUMMARY_REL = CARRY_RUN_DIR + "/summary.json"
CARRY_ITEM_SHA256 = (
    "c9aa47f5cff0b8d4bd7ca81dfcef886e47fef02df9d6ca4e8e2dc2419367f2ad"
)
CARRY_SUMMARY_SHA256 = (
    "b17b8bc0a6aa679f0f18e0eb3f064d1d9e50b5bea135ecb026836f4fd28461f4"
)

RAW_OBSERVER_REL = "scripts/longterm_k0_rvg_raw_recurrence_observer.py"

AUTHORITY_REL = (
    "reports/longterm_k0_rvg_strong_alignment_causal_falsification_"
    "static_design_candidate.md"
)
RUNNER_REL = "scripts/longterm_k0_rvg_strong_alignment_causal_falsification.py"
TEST_REL = "tests/test_longterm_k0_rvg_strong_alignment_causal_falsification.py"

ITEM_FILE = "strong_alignment_causal_falsification_item_metrics.jsonl"
SUMMARY_FILE = "summary.json"
MANIFEST_FILE = "execution_manifest.json"

ITEM_SCHEMA = "k0-rvg-strong-alignment-causal-falsification-item-v1"
SUMMARY_SCHEMA = "k0-rvg-strong-alignment-causal-falsification-summary-v1"
MANIFEST_SCHEMA = "k0-rvg-strong-alignment-causal-falsification-execution-manifest-v1"

SOURCE_BLOCK = 20
TARGET_RESIDUAL_LAYER = 21
INTERVENTION_LAYER = 22
TARGET_K = 2
HIDDEN = 768
INTERMEDIATE = 1536
STATE_SIZE = 16
LAYER_COUNT = 24

STRONG_COUNT = 240
WEAK_COUNT = 1296
EQUAL_COUNT = 0

FULL_FORWARD_BUDGET = 2640
PREFLIGHT_FORWARD_BUDGET = 16

COSINE_SLACK = 1e-12
BRIDGE_REL_TOL = 1e-12
BRIDGE_ABS_TOL = 1e-12
VECTOR_ABS_TOL = 5e-12
RUNTIME_CAST_TOL = 5e-6
MIDPOINT_TOL = 5e-6

EXPECTED_HANDOFF_SHA256 = (
    "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
)
EXPECTED_ENCODER_CANONICAL_DIGEST = (
    "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
)
EXPECTED_ENCODER_RAW_CONCAT_DIGEST = (
    "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"
)
EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

UNRELATED_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
    "validate_frozen_strong_alignment_sign_contribution_artifacts.py",
    "validate_frozen_strong_channel_mass_breadth_concentration_artifacts.py",
    "validate_frozen_strong_sign_contribution_itemwise_breadth_artifacts.py",
}

# These exact provenance booleans are intentionally public only when False.
# Their names contain otherwise-forbidden substrings, so the serializer must
# distinguish negative execution attestations from payload-bearing fields.
PUBLIC_NEGATIVE_BOOLEAN_FLAGS = frozenset({
    "logits_read",
    "raw_vectors_persisted",
})


class FalsificationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FalsificationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FalsificationError("GIT_FAILURE:" + " ".join(args)) from exc


def _git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FalsificationError("GIT_SHOW_FAILURE:" + spec) from exc


def _ancestor(root: Path, commit: str, head: str) -> bool:
    return (
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def _status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"').replace("\\", "/")


def _import_file(path: Path, name: str):
    require(path.is_file(), "MODULE_MISSING:" + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "MODULE_SPEC_FAILURE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_frozen(root: Path, commit: str, rel: str, name: str):
    frozen = _git_bytes(root, f"{commit}:{rel}")
    path = root / rel
    require(path.is_file(), "FROZEN_PARENT_MISSING:" + rel)
    require(path.read_bytes() == frozen, "FROZEN_PARENT_WORKTREE_DRIFT:" + rel)
    return _import_file(path, name)


def _frozen_bytes(
    root: Path,
    commit: str,
    rel: str,
    label: str,
) -> tuple[bytes, str, str]:
    """Authenticate frozen commit:path bytes and derive exact SHA256/blob."""
    raw = _git_bytes(root, f"{commit}:{rel}")
    blob = _git(root, "rev-parse", f"{commit}:{rel}")
    path = root / rel
    require(path.is_file(), label + "_MISSING")
    require(path.read_bytes() == raw, label + "_WORKTREE_DRIFT")
    return raw, sha256_bytes(raw), blob


def _jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise FalsificationError(f"{label}_JSONL_PARSE_FAILURE:{line_no}") from exc
        require(isinstance(row, dict), f"{label}_ROW_NOT_OBJECT:{line_no}")
        rows.append(row)
    return rows


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise FalsificationError(label + "_JSON_PARSE_FAILURE") from exc
    require(isinstance(obj, dict), label + "_NOT_OBJECT")
    return obj


class ForwardBudget:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.count = 0

    def consume(self, n: int = 1) -> None:
        require(n >= 0, "NEGATIVE_FORWARD_CONSUMPTION")
        require(self.count + n <= self.limit, "FORWARD_BUDGET_EXCEEDED")
        self.count += n


def authenticate_static(root: Path, *, runtime: bool = False) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")

    for commit, label in (
        (DESIGN_COMMIT, "DESIGN"),
        (STOP_COMMIT, "STOP"),
        (GEOMETRY_EVIDENCE_COMMIT, "GEOMETRY_EVIDENCE"),
        (GEOMETRY_IMPLEMENTATION_COMMIT, "GEOMETRY_IMPLEMENTATION"),
        (CARRY_EVIDENCE_COMMIT, "CARRY_EVIDENCE"),
        (CARRY_IMPLEMENTATION_COMMIT, "CARRY_IMPLEMENTATION"),
    ):
        require(_ancestor(root, commit, head), label + "_NOT_ANCESTOR")

    require(
        _git(root, "rev-parse", f"{DESIGN_COMMIT}:{AUTHORITY_REL}") == DESIGN_BLOB,
        "DESIGN_BLOB_MISMATCH",
    )
    require(
        (root / AUTHORITY_REL).read_bytes()
        == _git_bytes(root, f"{DESIGN_COMMIT}:{AUTHORITY_REL}"),
        "DESIGN_WORKTREE_DRIFT",
    )

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()

    active = {RUNNER_REL, TEST_REL}
    for line in status:
        path = _status_path(line)
        xy = line[:2]
        if path in UNRELATED_UNTRACKED:
            require(xy == "??", "UNRELATED_STATE_CHANGED:" + line)
            continue
        if not runtime and path in active:
            continue
        raise FalsificationError("UNEXPECTED_WORKTREE_CHANGE:" + line)

    info: dict[str, Any] = {
        "branch": branch,
        "head": head,
        "authority_blob": DESIGN_BLOB,
        "runtime_mode": runtime,
    }

    if runtime:
        for rel, label in ((RUNNER_REL, "RUNNER"), (TEST_REL, "TEST")):
            require(
                subprocess.call(
                    ["git", "ls-files", "--error-unmatch", "--", rel],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                == 0,
                "RUNTIME_REQUIRES_TRACKED_" + label,
            )
            require(
                subprocess.call(
                    ["git", "diff", "--quiet", "--", rel],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                == 0,
                "RUNTIME_" + label + "_WORKTREE_DRIFT",
            )
        # Git's clean/smudge filters (notably Windows CRLF conversion) may make
        # worktree bytes differ from committed blob bytes while `git diff` is clean.
        # Runtime provenance therefore treats the committed blob as canonical and
        # uses Git's filtered diff check above for worktree equivalence.
        runner_blob_bytes = _git_bytes(root, f"{head}:{RUNNER_REL}")
        test_blob_bytes = _git_bytes(root, f"{head}:{TEST_REL}")
        info["implementation_commit"] = head
        info["runner_blob"] = _git(root, "rev-parse", f"{head}:{RUNNER_REL}")
        info["test_blob"] = _git(root, "rev-parse", f"{head}:{TEST_REL}")
        info["runner_sha256"] = sha256_bytes(runner_blob_bytes)
        info["test_sha256"] = sha256_bytes(test_blob_bytes)

    return info


def common_pairs(
    geometry_rows: Sequence[Mapping[str, Any]],
    carry_rows: Sequence[Mapping[str, Any]],
):
    geo = {int(r["local_template_index"]): dict(r) for r in geometry_rows}
    carry = {
        (int(r["local_template_index"]), str(r["role"])): dict(r)
        for r in carry_rows
        if int(r.get("relative_coordinate", -99)) == TARGET_K
        and bool(r.get("in_common_ddsssss_cohort"))
    }
    keys = sorted(
        i for i in geo if (i, "corr") in carry and (i, "ctrl") in carry
    )
    require(len(keys) == 330, "COMMON_330_PAIR_MISMATCH")
    for i in keys:
        stable = geo[i]["stable_item_id"]
        require(carry[i, "corr"]["stable_item_id"] == stable, "CORR_STABLE_ID_MISMATCH")
        require(carry[i, "ctrl"]["stable_item_id"] == stable, "CTRL_STABLE_ID_MISMATCH")
    return keys, geo, carry


def load_authenticated_parent_stack(root: Path, *, runtime: bool = False):
    parent = import_frozen(
        root,
        GEOMETRY_IMPLEMENTATION_COMMIT,
        GEOMETRY_RUNNER_REL,
        "k0_rvg_frozen_geometry_parent",
    )
    require(
        sha256_file(root / GEOMETRY_RUNNER_REL) == GEOMETRY_RUNNER_SHA256,
        "GEOMETRY_RUNNER_SHA256_MISMATCH",
    )
    require(
        _git(root, "rev-parse", f"{GEOMETRY_IMPLEMENTATION_COMMIT}:{GEOMETRY_RUNNER_REL}")
        == GEOMETRY_RUNNER_BLOB,
        "GEOMETRY_RUNNER_BLOB_MISMATCH",
    )

    immediate = parent.load_immediate_parent(root)
    summary, manifest, parent_items, parent_channels = (
        parent.authenticate_immediate_parent_evidence(root)
    )
    stack = parent.build_parent_stack(root, immediate) if runtime else None

    geometry_raw, geometry_item_sha, geometry_item_blob = _frozen_bytes(
        root, GEOMETRY_EVIDENCE_COMMIT, GEOMETRY_ITEM_REL, "GEOMETRY_ITEM"
    )
    _, geometry_channel_sha, geometry_channel_blob = _frozen_bytes(
        root, GEOMETRY_EVIDENCE_COMMIT, GEOMETRY_CHANNEL_REL, "GEOMETRY_CHANNEL"
    )
    _, geometry_summary_sha, geometry_summary_blob = _frozen_bytes(
        root, GEOMETRY_EVIDENCE_COMMIT, GEOMETRY_SUMMARY_REL, "GEOMETRY_SUMMARY"
    )
    _, geometry_manifest_sha, geometry_manifest_blob = _frozen_bytes(
        root, GEOMETRY_EVIDENCE_COMMIT, GEOMETRY_MANIFEST_REL, "GEOMETRY_MANIFEST"
    )

    carry_raw, carry_item_sha, carry_item_blob = _frozen_bytes(
        root, CARRY_EVIDENCE_COMMIT, CARRY_ITEM_REL, "CARRY_ITEM"
    )
    _, carry_summary_sha, carry_summary_blob = _frozen_bytes(
        root, CARRY_EVIDENCE_COMMIT, CARRY_SUMMARY_REL, "CARRY_SUMMARY"
    )

    geometry_rows = _jsonl(geometry_raw, "GEOMETRY_ITEM")
    carry_rows = _jsonl(carry_raw, "CARRY_ITEM")
    keys, geo, carry = common_pairs(geometry_rows, carry_rows)

    require(
        all(
            r.get("schema_version")
            == "k0-rvg-layer22-carry-write-factorization-row-v1"
            for r in carry_rows
        ),
        "CARRY_SCHEMA_MISMATCH",
    )

    return {
        "geometry_parent": parent,
        "geometry_immediate_parent": immediate,
        "geometry_stack": stack,
        "geometry_parent_summary": summary,
        "geometry_parent_manifest": manifest,
        "geometry_parent_items": parent_items,
        "geometry_parent_channels": parent_channels,
        "common_indices": keys,
        "geometry_rows": geo,
        "carry_rows": carry,
        "artifact_identities": {
            "geometry_item": {"sha256": geometry_item_sha, "blob": geometry_item_blob},
            "geometry_channel": {"sha256": geometry_channel_sha, "blob": geometry_channel_blob},
            "geometry_summary": {"sha256": geometry_summary_sha, "blob": geometry_summary_blob},
            "geometry_manifest": {"sha256": geometry_manifest_sha, "blob": geometry_manifest_blob},
            "carry_item": {"sha256": carry_item_sha, "blob": carry_item_blob},
            "carry_summary": {"sha256": carry_summary_sha, "blob": carry_summary_blob},
        },
    }


def _expected_strong_mask(parent_channels: Mapping[int, Mapping[str, Any]]):
    import torch

    require(len(parent_channels) == INTERMEDIATE, "PARENT_CHANNEL_COUNT_MISMATCH")
    mask = torch.zeros(INTERMEDIATE, dtype=torch.bool)
    for j in range(INTERMEDIATE):
        row = parent_channels[j]
        part = str(row["downstream_partition"])
        require(part in {"strong", "weak", "equal"}, f"BAD_PARENT_PARTITION:{j}")
        if part == "strong":
            mask[j] = True
    require(int(mask.sum().item()) == STRONG_COUNT, "FROZEN_STRONG_COUNT_MISMATCH")
    return mask


def resolve_runtime(root: Path, ctx: Mapping[str, Any], handoff: Path):
    parent = ctx["geometry_parent"]
    immediate = ctx["geometry_immediate_parent"]
    stack = ctx["geometry_stack"]
    require(stack is not None, "RUNTIME_STACK_MISSING")

    runtime = parent.resolve_runtime(root, immediate, stack, handoff)
    model = runtime["model"]
    backbone = model.mamba
    require(len(backbone.layers) == LAYER_COUNT, "BACKBONE_LAYER_COUNT_MISMATCH")

    mixer22 = backbone.layers[INTERVENTION_LAYER].mixer
    require(hasattr(mixer22, "in_proj"), "LAYER22_INPROJ_MISSING")
    require(int(mixer22.intermediate_size) == INTERMEDIATE, "LAYER22_INTERMEDIATE_MISMATCH")

    op = runtime["operator"]
    strong_mask = op["strong_mask"].detach().cpu().bool().contiguous()
    require(strong_mask.numel() == INTERMEDIATE, "RUNTIME_STRONG_MASK_WIDTH_MISMATCH")
    require(int(strong_mask.sum().item()) == STRONG_COUNT, "RUNTIME_STRONG_COUNT_MISMATCH")
    require(
        int(op["weak_mask"].sum().item()) == WEAK_COUNT,
        "RUNTIME_WEAK_COUNT_MISMATCH",
    )
    require(
        int(op["equal_mask"].sum().item()) == EQUAL_COUNT,
        "RUNTIME_EQUAL_COUNT_MISMATCH",
    )
    frozen_mask = _expected_strong_mask(ctx["geometry_parent_channels"])
    require(bool((strong_mask == frozen_mask).all().item()), "FROZEN_STRONG_MASK_IDENTITY_MISMATCH")

    require(
        runtime["binding"].source_sha256 == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )
    require(
        runtime["handoff"]["zip_sha256"] == EXPECTED_HANDOFF_SHA256,
        "HANDOFF_SHA256_MISMATCH",
    )
    require(
        runtime["handoff"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "CHECKPOINT_SHA256_MISMATCH",
    )
    require(
        runtime["encoder"]["canonical_digest"] == EXPECTED_ENCODER_CANONICAL_DIGEST,
        "ENCODER_CANONICAL_DIGEST_MISMATCH",
    )
    require(
        runtime["encoder"]["raw_concat_digest"] == EXPECTED_ENCODER_RAW_CONCAT_DIGEST,
        "ENCODER_RAW_CONCAT_DIGEST_MISMATCH",
    )

    observer = import_frozen(
        root,
        CARRY_IMPLEMENTATION_COMMIT,
        RAW_OBSERVER_REL,
        "k0_rvg_frozen_raw_recurrence_observer",
    )
    observer_binding = observer.resolve_source_binding()
    require(
        observer_binding.source_sha256 == EXPECTED_MAMBA_SOURCE_SHA256,
        "OBSERVER_SOURCE_SHA256_MISMATCH",
    )
    require(observer_binding.update_line == 350, "OBSERVER_UPDATE_LINE_MISMATCH")
    require(observer_binding.readout_line == 351, "OBSERVER_READOUT_LINE_MISMATCH")
    layer_map = observer.registered_mamba_layers(model)
    require(len(layer_map) == LAYER_COUNT, "OBSERVER_LAYER_MAP_COUNT_MISMATCH")
    require(INTERVENTION_LAYER in set(layer_map.values()), "LAYER22_NOT_REGISTERED")

    deep_ctx = stack["stack"]["ctx"]
    require("base" in deep_ctx, "DIRECT_FORWARD_BASE_MISSING")
    base = deep_ctx["base"]
    require(
        hasattr(base, "direct_backbone_forward"),
        "DIRECT_BACKBONE_FORWARD_MISSING",
    )

    layer20 = backbone.layers[SOURCE_BLOCK]
    norm22 = backbone.layers[INTERVENTION_LAYER].norm
    require(
        runtime.get("layer20") is layer20,
        "LAYER20_RUNTIME_IDENTITY_MISMATCH",
    )
    require(
        runtime.get("norm22") is norm22,
        "LAYER22_NORM_RUNTIME_IDENTITY_MISMATCH",
    )
    require(
        runtime.get("mixer22") is mixer22,
        "LAYER22_MIXER_RUNTIME_IDENTITY_MISMATCH",
    )
    require(
        float(norm22.variance_epsilon) == 1e-5,
        "LAYER22_RMS_EPS_MISMATCH",
    )

    parent_layer_map = runtime.get("layer_map")
    if parent_layer_map is not None:
        require(
            dict(parent_layer_map) == dict(layer_map),
            "OBSERVER_PARENT_LAYER_MAP_MISMATCH",
        )

    return {
        "parent_runtime": runtime,
        "model": model,
        "backbone": backbone,
        "base": base,
        "layer20": layer20,
        "norm22": norm22,
        "mixer22": mixer22,
        "strong_mask": strong_mask,
        "observer": observer,
        "observer_binding": observer_binding,
        "layer_map": layer_map,
    }


def common_plan_pairs(ctx: Mapping[str, Any]):
    stack = ctx["geometry_stack"]
    require(stack is not None, "RUNTIME_STACK_MISSING")
    plan = stack["stack"]["ctx"]["plan"]
    wanted = set(int(i) for i in ctx["common_indices"])
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in plan:
        idx = int(row["local_template_index"])
        role = str(row["role"])
        if idx in wanted and role in {"corr", "ctrl"}:
            key = (idx, role)
            require(key not in by_key, f"PLAN_DUPLICATE:{idx}:{role}")
            by_key[key] = row
    require(len(by_key) == 660, "COMMON_PLAN_ROLE_COUNT_MISMATCH")
    return {
        idx: {"corr": by_key[idx, "corr"], "ctrl": by_key[idx, "ctrl"]}
        for idx in sorted(wanted)
    }


def _prefixes(ctx: Mapping[str, Any], row: Mapping[str, Any]):
    parent = ctx["geometry_parent"]
    immediate = ctx["geometry_immediate_parent"]
    stack = ctx["geometry_stack"]
    matched, swapped = parent._prefixes(immediate, stack, row)
    return tuple(matched), tuple(swapped)


def _finite_tensor(t, label: str) -> None:
    import torch

    require(torch.is_tensor(t), label + "_NOT_TENSOR")
    require(bool(torch.isfinite(t).all().item()), label + "_NONFINITE")


def _cos(x, y) -> float:
    import torch

    _finite_tensor(x, "COS_X")
    _finite_tensor(y, "COS_Y")
    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0 and b > 0.0, "ZERO_SOURCE_NORM")
    out = float(torch.dot(x, y).item() / (a * b))
    require(
        -1.0 - COSINE_SLACK <= out <= 1.0 + COSINE_SLACK,
        "COSINE_OUT_OF_RANGE",
    )
    return min(1.0, max(-1.0, out))


def reconstruct_geometry(
    row: Mapping[str, Any],
    matched_outputs,
    swapped_outputs,
    runtime: Mapping[str, Any],
):
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])

    (_mpr, mrms, _ml21, ml20) = matched_outputs
    (_spr, srms, _sl21, sl20) = swapped_outputs

    gamma = runtime["parent_runtime"]["norm22"].weight.detach().cpu().to(torch.float64)
    w = runtime["parent_runtime"]["operator"]["w_hidden64"]
    mask = runtime["strong_mask"]

    r20m = ml20["R20"].to(torch.float64).contiguous()
    r20s = sl20["R20"].to(torch.float64).contiguous()
    y20m = ml20["Y20"].to(torch.float64).contiguous()
    y20s = sl20["Y20"].to(torch.float64).contiguous()

    xm = mrms["X32"].to(torch.float64).contiguous()
    xs = srms["X32"].to(torch.float64).contiguous()
    sm = float(mrms["rms_scale32"].item())
    ss = float(srms["rms_scale32"].item())
    sbar = 0.5 * (sm + ss)

    dr20 = r20m - r20s
    dy20 = y20m - y20s
    dx = xm - xs
    d = float(torch.linalg.vector_norm(dx).item())
    require(d > 0.0 and math.isfinite(d), f"DELTA_X22_ZERO_OR_NONFINITE:{idx}:{role}")

    q_r = gamma * (sbar * dr20)
    q_y = gamma * (sbar * dy20)
    h_r = torch.mv(w, q_r)
    h_y = torch.mv(w, q_y)

    x = (h_r[mask] / d).detach().cpu().to(torch.float64).contiguous()
    y = (h_y[mask] / d).detach().cpu().to(torch.float64).contiguous()
    require(x.numel() == STRONG_COUNT and y.numel() == STRONG_COUNT, "STRONG_WIDTH_MISMATCH")
    _finite_tensor(x, "GEOM_X")
    _finite_tensor(y, "GEOM_Y")

    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0 and b > 0.0, "ZERO_SOURCE_NORM")
    c = _cos(x, y)
    i = 2.0 * float(torch.dot(x, y).item())

    return {
        "idx": idx,
        "stable_item_id": row["stable_item_id"],
        "role": role,
        "x": x,
        "y": y,
        "d": d,
        "A": a,
        "B": b,
        "C": c,
        "I": i,
    }


def bridge_geometry(
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    frozen: Mapping[str, Any],
):
    residuals: dict[str, float] = {}
    for role, rec in (("corr", corr), ("ctrl", ctrl)):
        for field in ("A", "B", "C", "I"):
            got = float(rec[field])
            exp = float(frozen[f"{field}_{role}"])
            residuals[f"{field}_{role}"] = abs(got - exp)
            require(
                math.isclose(
                    got, exp, rel_tol=BRIDGE_REL_TOL, abs_tol=BRIDGE_ABS_TOL
                ),
                f"GEOMETRY_BRIDGE_FAILURE:{field}:{role}",
            )

    delta_i = float(corr["I"]) - float(ctrl["I"])
    for field in ("reconstructed_delta_I", "parent_delta_ry20_strong"):
        if field in frozen:
            exp = float(frozen[field])
            residuals[field] = abs(delta_i - exp)
            require(
                math.isclose(
                    delta_i, exp, rel_tol=BRIDGE_REL_TOL, abs_tol=BRIDGE_ABS_TOL
                ),
                "GEOMETRY_PARENT_TARGET_BRIDGE_FAILURE:" + field,
            )
    return residuals


def alignment_delta(x_c, y_c, c_target: float):
    import torch

    x = x_c.detach().cpu().to(torch.float64).contiguous()
    y = y_c.detach().cpu().to(torch.float64).contiguous()
    require(x.numel() == STRONG_COUNT and y.numel() == STRONG_COUNT, "STRONG_WIDTH_MISMATCH")
    _finite_tensor(x, "ALIGN_X")
    _finite_tensor(y, "ALIGN_Y")

    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0 and b > 0.0, "ZERO_SOURCE_NORM")

    c = _cos(x, y)
    require(
        -1.0 - COSINE_SLACK <= float(c_target) <= 1.0 + COSINE_SLACK,
        "TARGET_COSINE_OUT_OF_RANGE",
    )
    ct = min(1.0, max(-1.0, float(c_target)))

    u = x / a
    raw = y / b - c * u
    n = float(torch.linalg.vector_norm(raw).item())
    require(math.isfinite(n) and n > COSINE_SLACK, "DEGENERATE_ORTHOGONAL_COMPONENT")
    v = raw / n

    y_star = b * (ct * u + math.sqrt(max(0.0, 1.0 - ct * ct)) * v)
    _finite_tensor(y_star, "ALIGN_Y_STAR")
    realized = _cos(x, y_star)
    a_resid = abs(float(torch.linalg.vector_norm(x).item()) - a)
    b_resid = abs(float(torch.linalg.vector_norm(y_star).item()) - b)

    require(a_resid <= VECTOR_ABS_TOL, "ALIGN_A_PRESERVATION_FAILURE")
    require(b_resid <= VECTOR_ABS_TOL, "ALIGN_B_PRESERVATION_FAILURE")
    require(abs(realized - ct) <= VECTOR_ABS_TOL, "ALIGN_TARGET_COSINE_FAILURE")

    return y_star - y, {
        "A": a,
        "B": b,
        "C": c,
        "target_cosine": ct,
        "realized_cosine": realized,
        "a_residual": a_resid,
        "b_residual": b_resid,
    }


def magnitude_delta(x_c, y_c, a_target: float, b_target: float):
    import torch

    x = x_c.detach().cpu().to(torch.float64).contiguous()
    y = y_c.detach().cpu().to(torch.float64).contiguous()
    require(x.numel() == STRONG_COUNT and y.numel() == STRONG_COUNT, "STRONG_WIDTH_MISMATCH")
    _finite_tensor(x, "MAG_X")
    _finite_tensor(y, "MAG_Y")

    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0 and b > 0.0, "ZERO_SOURCE_NORM")
    require(math.isfinite(float(a_target)) and float(a_target) > 0.0, "BAD_TARGET_A")
    require(math.isfinite(float(b_target)) and float(b_target) > 0.0, "BAD_TARGET_B")

    xm = (float(a_target) / a) * x
    ym = (float(b_target) / b) * y
    realized_a = float(torch.linalg.vector_norm(xm).item())
    realized_b = float(torch.linalg.vector_norm(ym).item())
    base_c = _cos(x, y)
    realized_c = _cos(xm, ym)

    require(abs(realized_a - float(a_target)) <= VECTOR_ABS_TOL, "MAG_A_TARGET_FAILURE")
    require(abs(realized_b - float(b_target)) <= VECTOR_ABS_TOL, "MAG_B_TARGET_FAILURE")
    require(abs(realized_c - base_c) <= VECTOR_ABS_TOL, "MAG_COSINE_PRESERVATION_FAILURE")

    return (xm - x) + (ym - y), {
        "target_A": float(a_target),
        "target_B": float(b_target),
        "realized_A": realized_a,
        "realized_B": realized_b,
        "baseline_cosine": base_c,
        "realized_cosine": realized_c,
        "cosine_residual": abs(realized_c - base_c),
    }


def _apply_intervention(
    inproj_output,
    *,
    token_index: int,
    strong_mask,
    delta_h,
    matched: bool,
    audit: dict[str, Any],
):
    import torch

    require(torch.is_tensor(inproj_output), "INPROJ_OUTPUT_NOT_TENSOR")
    require(
        inproj_output.ndim == 3
        and inproj_output.shape[0] == 1
        and inproj_output.shape[-1] == 2 * INTERMEDIATE,
        "INPROJ_OUTPUT_SHAPE_MISMATCH",
    )
    require(0 <= int(token_index) < inproj_output.shape[1], "TARGET_TOKEN_OUT_OF_RANGE")

    mask_cpu = strong_mask.detach().cpu().bool().contiguous()
    require(mask_cpu.numel() == INTERMEDIATE, "STRONG_MASK_WIDTH_MISMATCH")
    require(int(mask_cpu.sum().item()) == STRONG_COUNT, "STRONG_MASK_COUNT_MISMATCH")
    mask = mask_cpu.to(device=inproj_output.device)

    correction64 = delta_h.detach().cpu().to(torch.float64).contiguous()
    require(correction64.numel() == STRONG_COUNT, "DELTA_H_WIDTH_MISMATCH")
    _finite_tensor(correction64, "DELTA_H")

    before = inproj_output.detach().clone()
    expected = before.clone()
    out = inproj_output.clone()
    sign = 0.5 if matched else -0.5
    correction = correction64.to(device=out.device, dtype=out.dtype)

    expected_x = expected[0, int(token_index), :INTERMEDIATE]
    expected_x[mask] = expected_x[mask] + sign * correction

    out_x = out[0, int(token_index), :INTERMEDIATE]
    out_x[mask] = out_x[mask] + sign * correction

    require(torch.equal(out, expected), "UNAUTHORIZED_INPROJ_CHANGE")

    before_strong = before[0, int(token_index), :INTERMEDIATE][mask].detach().cpu()
    after_strong = out[0, int(token_index), :INTERMEDIATE][mask].detach().cpu()
    intended = (sign * correction).detach().cpu()
    applied = after_strong - before_strong
    applied_resid = float(
        torch.max(
            torch.abs(applied.to(torch.float64) - intended.to(torch.float64))
        ).item()
    )
    require(applied_resid <= RUNTIME_CAST_TOL, "APPLIED_CORRECTION_MISMATCH")

    # Explicit manipulation-boundary checks.
    require(
        torch.equal(
            out[:, :, INTERMEDIATE:],
            before[:, :, INTERMEDIATE:],
        ),
        "Z_BRANCH_CHANGED",
    )
    weak = ~mask
    require(
        torch.equal(
            out[:, :, :INTERMEDIATE][:, :, weak],
            before[:, :, :INTERMEDIATE][:, :, weak],
        ),
        "WEAK_OR_EQUAL_X_CHANGED",
    )
    if int(token_index) > 0:
        require(
            torch.equal(out[:, : int(token_index), :], before[:, : int(token_index), :]),
            "EARLIER_TOKEN_CHANGED",
        )
    if int(token_index) + 1 < out.shape[1]:
        require(
            torch.equal(
                out[:, int(token_index) + 1 :, :],
                before[:, int(token_index) + 1 :, :],
            ),
            "LATER_TOKEN_CHANGED",
        )

    audit.clear()
    audit.update(
        {
            "token_index": int(token_index),
            "matched": bool(matched),
            "before_strong": before_strong.to(torch.float64),
            "after_strong": after_strong.to(torch.float64),
            "applied_correction_max_abs_residual": applied_resid,
            "runtime_correction_l2": float(
                torch.linalg.vector_norm(correction64).item()
            ),
        }
    )
    return out


def install_inproj_hook(
    mixer,
    *,
    token_index: int,
    strong_mask,
    delta_h,
    matched: bool,
    audit: dict[str, Any],
):
    def hook(_module, _args, output):
        return _apply_intervention(
            output,
            token_index=token_index,
            strong_mask=strong_mask,
            delta_h=delta_h,
            matched=matched,
            audit=audit,
        )

    return mixer.in_proj.register_forward_hook(hook)


def _cpu_float32(value, label: str):
    import torch

    require(torch.is_tensor(value), label + "_NOT_TENSOR")
    out = value.detach().cpu().contiguous().clone()
    require(out.dtype == torch.float32, label + "_DTYPE_NOT_FLOAT32")
    require(tuple(out.shape) == (1, INTERMEDIATE, STATE_SIZE), label + "_SHAPE_MISMATCH")
    require(bool(torch.isfinite(out).all().item()), label + "_NONFINITE")
    return out


def _recurrence_record(runtime: Mapping[str, Any], target_abs: int, collector):
    key = (INTERVENTION_LAYER, int(target_abs))
    require(collector.records is not None, "RECURRENCE_RECORDS_MISSING")
    require(key in collector.records, "LAYER22_RECURRENCE_RECORD_MISSING")
    rec = collector.records[key]
    s_prev = _cpu_float32(rec.s_prev, "S_PREV")
    g = _cpu_float32(rec.g, "G")
    w = _cpu_float32(rec.w, "W")
    s_post = _cpu_float32(rec.s_post, "S_POST")
    carry = (g * s_prev).contiguous()
    return {"S_POST32": s_post, "W32": w, "CARRY32": carry}


def capture_branch(
    ctx: Mapping[str, Any],
    runtime: Mapping[str, Any],
    row: Mapping[str, Any],
    token_ids: Sequence[int],
    budget: ForwardBudget,
    *,
    delta_h=None,
    matched: bool | None = None,
):
    """One physical model forward with all causal-falsification observers.

    Important: do not call the frozen geometry parent's `run_branch` here.
    That branch recursively enters an authenticated parent capture chain whose
    write-factor collector installs its own CPython trace.  Nesting a second
    RawRecurrenceCollector outside that chain replaces the outer trace and
    loses the layer-22 recurrence record.  Instead, reuse the same authenticated
    `base.direct_backbone_forward` boundary used by the frozen native-state
    parents, while reproducing the frozen R20/Y20/X22 hook observations locally.
    """
    import torch

    target_abs = int(row["anchor"]) + TARGET_K
    collector = runtime["observer"].RawRecurrenceCollector(
        runtime["observer_binding"],
        runtime["layer_map"],
        (target_abs,),
    )

    layer20 = runtime["layer20"]
    norm22 = runtime["norm22"]
    holders: dict[str, Any] = {}
    counts = {
        "layer20_pre": 0,
        "layer20_mixer_post": 0,
        "norm22_pre": 0,
        "norm22_post": 0,
    }

    def layer20_pre_hook(_module, args):
        counts["layer20_pre"] += 1
        require(
            counts["layer20_pre"] == 1,
            "DUPLICATE_LAYER20_PRE_HOOK",
        )
        require(len(args) >= 1, "LAYER20_PRE_ARG_COUNT_MISMATCH")
        full = args[0].detach().cpu().contiguous().clone()
        require(full.dtype == torch.float32, "R20_DTYPE_MISMATCH")
        require(
            full.ndim == 3 and full.shape[0] == 1 and full.shape[-1] == HIDDEN,
            "R20_FULL_SHAPE_MISMATCH",
        )
        require(0 <= target_abs < full.shape[1], "R20_TARGET_OUT_OF_RANGE")
        holders["R20"] = full[0, target_abs, :].contiguous().clone()

    def layer20_mixer_post_hook(_module, _args, output):
        counts["layer20_mixer_post"] += 1
        require(
            counts["layer20_mixer_post"] == 1,
            "DUPLICATE_LAYER20_MIXER_POST_HOOK",
        )
        full = output.detach().cpu().contiguous().clone()
        require(full.dtype == torch.float32, "Y20_DTYPE_MISMATCH")
        require(
            full.ndim == 3 and full.shape[0] == 1 and full.shape[-1] == HIDDEN,
            "Y20_FULL_SHAPE_MISMATCH",
        )
        require(0 <= target_abs < full.shape[1], "Y20_TARGET_OUT_OF_RANGE")
        holders["Y20"] = full[0, target_abs, :].contiguous().clone()

    def norm22_pre_hook(_module, args):
        counts["norm22_pre"] += 1
        require(counts["norm22_pre"] == 1, "DUPLICATE_NORM22_PRE_HOOK")
        require(len(args) == 1, "NORM22_PRE_ARG_COUNT_MISMATCH")
        full = args[0].detach().cpu().contiguous().clone()
        require(full.dtype == torch.float32, "R22_DTYPE_MISMATCH")
        require(
            full.ndim == 3 and full.shape[0] == 1 and full.shape[-1] == HIDDEN,
            "R22_FULL_SHAPE_MISMATCH",
        )
        require(0 <= target_abs < full.shape[1], "R22_TARGET_OUT_OF_RANGE")
        holders["R22"] = full[0, target_abs, :].contiguous().clone()

    def norm22_post_hook(_module, _args, output):
        counts["norm22_post"] += 1
        require(counts["norm22_post"] == 1, "DUPLICATE_NORM22_POST_HOOK")
        full = output.detach().cpu().contiguous().clone()
        require(full.dtype == torch.float32, "X22_DTYPE_MISMATCH")
        require(
            full.ndim == 3 and full.shape[0] == 1 and full.shape[-1] == HIDDEN,
            "X22_FULL_SHAPE_MISMATCH",
        )
        require(0 <= target_abs < full.shape[1], "X22_TARGET_OUT_OF_RANGE")
        holders["X22"] = full[0, target_abs, :].contiguous().clone()

    handles = [
        layer20.register_forward_pre_hook(layer20_pre_hook),
        layer20.mixer.register_forward_hook(layer20_mixer_post_hook),
        norm22.register_forward_pre_hook(norm22_pre_hook),
        norm22.register_forward_hook(norm22_post_hook),
    ]

    audit: dict[str, Any] | None = None
    if delta_h is not None:
        require(matched is not None, "INTERVENTION_BRANCH_ROLE_REQUIRED")
        audit = {}
        handles.append(
            install_inproj_hook(
                runtime["mixer22"],
                token_index=target_abs,
                strong_mask=runtime["strong_mask"],
                delta_h=delta_h,
                matched=bool(matched),
                audit=audit,
            )
        )

    prior_trace = sys.gettrace()
    budget.consume(1)
    try:
        with collector.capture():
            runtime["base"].direct_backbone_forward(
                runtime["model"],
                token_ids,
            )
    finally:
        for handle in reversed(handles):
            handle.remove()

    require(sys.gettrace() is prior_trace, "TRACE_RESTORATION_FAILURE")
    require(
        counts
        == {
            "layer20_pre": 1,
            "layer20_mixer_post": 1,
            "norm22_pre": 1,
            "norm22_post": 1,
        },
        "DIRECT_CAPTURE_HOOK_COUNT_FAILURE",
    )
    require(
        set(holders) == {"R20", "Y20", "R22", "X22"},
        "DIRECT_CAPTURE_MISSING",
    )

    r22 = holders["R22"]
    variance32 = r22.pow(2).mean()
    scale32 = torch.rsqrt(variance32 + 1e-5).detach().cpu().clone()
    reconstructed32 = (
        runtime["norm22"].weight.detach().cpu().contiguous()
        * (r22 * scale32)
    ).to(torch.float32).contiguous()
    rms_residual = float(
        torch.linalg.vector_norm(
            reconstructed32.to(torch.float64)
            - holders["X22"].to(torch.float64)
        ).item()
    ) / max(
        float(
            torch.linalg.vector_norm(
                holders["X22"].to(torch.float64)
            ).item()
        ),
        1e-12,
    )
    require(rms_residual <= 1e-6, "DIRECT_RMS_RECONSTRUCTION_FAILURE")

    outputs = (
        None,
        {
            "X32": holders["X22"],
            "rms_scale32": scale32,
            "rms_branch_reconstruction_relative_residual": rms_residual,
        },
        None,
        {
            "R20": holders["R20"],
            "Y20": holders["Y20"],
        },
    )
    endpoint = _recurrence_record(runtime, target_abs, collector)

    if delta_h is not None:
        require(audit is not None and bool(audit), "INTERVENTION_HOOK_NOT_OBSERVED")
        require(audit["token_index"] == target_abs, "INTERVENTION_TOKEN_AUDIT_MISMATCH")

    return outputs, endpoint, audit


def endpoint_metrics(matched: Mapping[str, Any], swapped: Mapping[str, Any]):
    import torch

    out: dict[str, float] = {}
    for source, name in (
        ("S_POST32", "delta_s22_post_l2"),
        ("W32", "delta_w_l2"),
        ("CARRY32", "delta_carry_l2"),
    ):
        dm = matched[source].to(torch.float64) - swapped[source].to(torch.float64)
        value = float(torch.linalg.vector_norm(dm).item())
        require(math.isfinite(value), "NONFINITE_ENDPOINT:" + name)
        out[name] = value
    return out


def bridge_carry(
    runtime_metrics: Mapping[str, float],
    frozen: Mapping[str, Any],
):
    residuals = {}
    for field in ("delta_s22_post_l2", "delta_w_l2", "delta_carry_l2"):
        got = float(runtime_metrics[field])
        exp = float(frozen[field])
        residuals[field] = abs(got - exp)
        require(
            math.isclose(got, exp, rel_tol=BRIDGE_REL_TOL, abs_tol=BRIDGE_ABS_TOL),
            "CARRY_BRIDGE_FAILURE:" + field,
        )
    return residuals


def midpoint_residual(m_audit: Mapping[str, Any], s_audit: Mapping[str, Any]) -> float:
    import torch

    before_mid = 0.5 * (
        m_audit["before_strong"] + s_audit["before_strong"]
    )
    after_mid = 0.5 * (
        m_audit["after_strong"] + s_audit["after_strong"]
    )
    value = float(torch.max(torch.abs(after_mid - before_mid)).item())
    require(value <= MIDPOINT_TOL, "BRANCH_MIDPOINT_PRESERVATION_FAILURE")
    return value


def _run_baseline_role(
    ctx: Mapping[str, Any],
    runtime: Mapping[str, Any],
    row: Mapping[str, Any],
    budget: ForwardBudget,
):
    matched_ids, swapped_ids = _prefixes(ctx, row)
    mo, me, _ = capture_branch(ctx, runtime, row, matched_ids, budget)
    so, se, _ = capture_branch(ctx, runtime, row, swapped_ids, budget)
    geom = reconstruct_geometry(row, mo, so, runtime)
    endpoints = endpoint_metrics(me, se)
    return geom, endpoints


def _run_intervention_role(
    ctx: Mapping[str, Any],
    runtime: Mapping[str, Any],
    row: Mapping[str, Any],
    delta_h,
    budget: ForwardBudget,
):
    matched_ids, swapped_ids = _prefixes(ctx, row)
    _mo, me, ma = capture_branch(
        ctx, runtime, row, matched_ids, budget, delta_h=delta_h, matched=True
    )
    _so, se, sa = capture_branch(
        ctx, runtime, row, swapped_ids, budget, delta_h=delta_h, matched=False
    )
    require(ma is not None and sa is not None, "INTERVENTION_AUDIT_MISSING")
    mid = midpoint_residual(ma, sa)
    endpoints = endpoint_metrics(me, se)
    applied_resid = max(
        float(ma["applied_correction_max_abs_residual"]),
        float(sa["applied_correction_max_abs_residual"]),
    )
    require(applied_resid <= RUNTIME_CAST_TOL, "APPLIED_CORRECTION_GATE_FAILURE")
    return endpoints, {
        "midpoint_max_abs_residual": mid,
        "applied_correction_max_abs_residual": applied_resid,
        "runtime_correction_l2": float(ma["runtime_correction_l2"]),
    }


def _row_for_item(
    idx: int,
    pair: Mapping[str, Mapping[str, Any]],
    ctx: Mapping[str, Any],
    runtime: Mapping[str, Any],
    budget: ForwardBudget,
):
    corr_row = pair["corr"]
    ctrl_row = pair["ctrl"]
    stable = ctx["geometry_rows"][idx]["stable_item_id"]
    require(corr_row["stable_item_id"] == stable, "CORR_PLAN_STABLE_ID_MISMATCH")
    require(ctrl_row["stable_item_id"] == stable, "CTRL_PLAN_STABLE_ID_MISMATCH")

    corr_geom, corr0 = _run_baseline_role(ctx, runtime, corr_row, budget)
    ctrl_geom, ctrl0 = _run_baseline_role(ctx, runtime, ctrl_row, budget)

    geom_resid = bridge_geometry(
        corr_geom,
        ctrl_geom,
        ctx["geometry_rows"][idx],
    )
    corr_carry_resid = bridge_carry(corr0, ctx["carry_rows"][idx, "corr"])
    ctrl_carry_resid = bridge_carry(ctrl0, ctx["carry_rows"][idx, "ctrl"])

    align_norm, align_check = alignment_delta(
        corr_geom["x"],
        corr_geom["y"],
        float(ctrl_geom["C"]),
    )
    align_h = float(corr_geom["d"]) * align_norm
    align_ep, align_runtime = _run_intervention_role(
        ctx, runtime, corr_row, align_h, budget
    )

    mag_norm, mag_check = magnitude_delta(
        corr_geom["x"],
        corr_geom["y"],
        float(ctrl_geom["A"]),
        float(ctrl_geom["B"]),
    )
    mag_h = float(corr_geom["d"]) * mag_norm
    mag_ep, mag_runtime = _run_intervention_role(
        ctx, runtime, corr_row, mag_h, budget
    )

    max_geom_bridge = max(geom_resid.values()) if geom_resid else 0.0
    max_carry_bridge = max(
        list(corr_carry_resid.values()) + list(ctrl_carry_resid.values())
    )

    row = {
        "schema_version": ITEM_SCHEMA,
        "local_template_index": int(idx),
        "stable_item_id": stable,
        "in_common_ddsssss_cohort": True,
        "source_block": SOURCE_BLOCK,
        "target_residual_layer": TARGET_RESIDUAL_LAYER,
        "intervention_layer": INTERVENTION_LAYER,
        "relative_coordinate": TARGET_K,

        "A_corr": float(corr_geom["A"]),
        "A_ctrl": float(ctrl_geom["A"]),
        "B_corr": float(corr_geom["B"]),
        "B_ctrl": float(ctrl_geom["B"]),
        "C_corr": float(corr_geom["C"]),
        "C_ctrl": float(ctrl_geom["C"]),
        "I_corr": float(corr_geom["I"]),
        "I_ctrl": float(ctrl_geom["I"]),
        "geometry_bridge_max_abs_residual": float(max_geom_bridge),
        "native_bridge_max_abs_residual": float(max_carry_bridge),

        "alignment_target_cosine": float(align_check["target_cosine"]),
        "alignment_realized_cosine": float(align_check["realized_cosine"]),
        "alignment_A_preservation_abs_residual": float(align_check["a_residual"]),
        "alignment_B_preservation_abs_residual": float(align_check["b_residual"]),
        "alignment_midpoint_max_abs_residual": float(
            align_runtime["midpoint_max_abs_residual"]
        ),
        "alignment_runtime_correction_l2": float(
            align_runtime["runtime_correction_l2"]
        ),
        "alignment_applied_correction_max_abs_residual": float(
            align_runtime["applied_correction_max_abs_residual"]
        ),

        "magnitude_target_A": float(mag_check["target_A"]),
        "magnitude_target_B": float(mag_check["target_B"]),
        "magnitude_realized_A": float(mag_check["realized_A"]),
        "magnitude_realized_B": float(mag_check["realized_B"]),
        "magnitude_realized_cosine": float(mag_check["realized_cosine"]),
        "magnitude_cosine_preservation_abs_residual": float(
            mag_check["cosine_residual"]
        ),
        "magnitude_midpoint_max_abs_residual": float(
            mag_runtime["midpoint_max_abs_residual"]
        ),
        "magnitude_runtime_correction_l2": float(
            mag_runtime["runtime_correction_l2"]
        ),
        "magnitude_applied_correction_max_abs_residual": float(
            mag_runtime["applied_correction_max_abs_residual"]
        ),

        "baseline_corr_delta_s22_post_l2": float(corr0["delta_s22_post_l2"]),
        "baseline_corr_delta_w_l2": float(corr0["delta_w_l2"]),
        "baseline_corr_delta_carry_l2": float(corr0["delta_carry_l2"]),
        "baseline_ctrl_delta_s22_post_l2": float(ctrl0["delta_s22_post_l2"]),
        "baseline_ctrl_delta_w_l2": float(ctrl0["delta_w_l2"]),
        "baseline_ctrl_delta_carry_l2": float(ctrl0["delta_carry_l2"]),
        "alignment_corr_delta_s22_post_l2": float(align_ep["delta_s22_post_l2"]),
        "alignment_corr_delta_w_l2": float(align_ep["delta_w_l2"]),
        "alignment_corr_delta_carry_l2": float(align_ep["delta_carry_l2"]),
        "magnitude_corr_delta_s22_post_l2": float(mag_ep["delta_s22_post_l2"]),
        "magnitude_corr_delta_w_l2": float(mag_ep["delta_w_l2"]),
        "magnitude_corr_delta_carry_l2": float(mag_ep["delta_carry_l2"]),
    }

    row.update(
        {
            "S_c0_minus_ctrl": row["baseline_corr_delta_s22_post_l2"]
            - row["baseline_ctrl_delta_s22_post_l2"],
            "S_cA_minus_ctrl": row["alignment_corr_delta_s22_post_l2"]
            - row["baseline_ctrl_delta_s22_post_l2"],
            "S_cM_minus_ctrl": row["magnitude_corr_delta_s22_post_l2"]
            - row["baseline_ctrl_delta_s22_post_l2"],
            "W_c0_minus_ctrl": row["baseline_corr_delta_w_l2"]
            - row["baseline_ctrl_delta_w_l2"],
            "W_cA_minus_ctrl": row["alignment_corr_delta_w_l2"]
            - row["baseline_ctrl_delta_w_l2"],
            "W_cM_minus_ctrl": row["magnitude_corr_delta_w_l2"]
            - row["baseline_ctrl_delta_w_l2"],
            "C_c0_minus_cA": row["baseline_corr_delta_carry_l2"]
            - row["alignment_corr_delta_carry_l2"],
            "C_c0_minus_cM": row["baseline_corr_delta_carry_l2"]
            - row["magnitude_corr_delta_carry_l2"],
            "S_c0_minus_cA": row["baseline_corr_delta_s22_post_l2"]
            - row["alignment_corr_delta_s22_post_l2"],
            "S_c0_minus_cM": row["baseline_corr_delta_s22_post_l2"]
            - row["magnitude_corr_delta_s22_post_l2"],
            "W_c0_minus_cA": row["baseline_corr_delta_w_l2"]
            - row["alignment_corr_delta_w_l2"],
            "W_c0_minus_cM": row["baseline_corr_delta_w_l2"]
            - row["magnitude_corr_delta_w_l2"],
        }
    )

    for k, v in row.items():
        if isinstance(v, float):
            require(math.isfinite(v), "NONFINITE_PUBLIC_METRIC:" + k)
    return row


def classify(summary: Mapping[str, Any], manipulation_ok: bool) -> str:
    if not manipulation_ok:
        return "Invalid"
    sa = float(summary["R_SA"])
    wa = float(summary["R_WA"])
    sm = float(summary["R_SM"])
    wm = float(summary["R_WM"])
    if sa > 0.0 and wa > 0.0:
        return "A" if sa > sm and wa > wm else "B"
    if (sa > 0.0) != (wa > 0.0):
        return "C"
    return "D"


def _aggregate(values) -> dict[str, Any]:
    vals = [float(v) for v in values]
    require(bool(vals), "EMPTY_AGGREGATE")
    require(all(math.isfinite(v) for v in vals), "NONFINITE_AGGREGATE")
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    forward_count: int,
) -> dict[str, Any]:
    require(len(rows) == 330, "COMMON_330_REQUIRED")
    require(forward_count == FULL_FORWARD_BUDGET, "FULL_FORWARD_COUNT_MISMATCH")

    def mean(k: str) -> float:
        return float(statistics.fmean(float(r[k]) for r in rows))

    out: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "population_size": 330,
        "model_forward_count": int(forward_count),
        "G_S0": mean("S_c0_minus_ctrl"),
        "G_SA": mean("S_cA_minus_ctrl"),
        "G_SM": mean("S_cM_minus_ctrl"),
        "G_W0": mean("W_c0_minus_ctrl"),
        "G_WA": mean("W_cA_minus_ctrl"),
        "G_WM": mean("W_cM_minus_ctrl"),
        "R_CA": mean("C_c0_minus_cA"),
        "R_CM": mean("C_c0_minus_cM"),
        "max_geometry_bridge_abs_residual": max(
            float(r["geometry_bridge_max_abs_residual"]) for r in rows
        ),
        "max_native_bridge_abs_residual": max(
            float(r["native_bridge_max_abs_residual"]) for r in rows
        ),
        "max_alignment_A_preservation_abs_residual": max(
            float(r["alignment_A_preservation_abs_residual"]) for r in rows
        ),
        "max_alignment_B_preservation_abs_residual": max(
            float(r["alignment_B_preservation_abs_residual"]) for r in rows
        ),
        "max_alignment_cosine_abs_residual": max(
            abs(float(r["alignment_realized_cosine"]) - float(r["alignment_target_cosine"]))
            for r in rows
        ),
        "max_alignment_midpoint_abs_residual": max(
            float(r["alignment_midpoint_max_abs_residual"]) for r in rows
        ),
        "max_alignment_applied_correction_abs_residual": max(
            float(r["alignment_applied_correction_max_abs_residual"]) for r in rows
        ),
        "max_magnitude_A_abs_residual": max(
            abs(float(r["magnitude_realized_A"]) - float(r["magnitude_target_A"]))
            for r in rows
        ),
        "max_magnitude_B_abs_residual": max(
            abs(float(r["magnitude_realized_B"]) - float(r["magnitude_target_B"]))
            for r in rows
        ),
        "max_magnitude_cosine_preservation_abs_residual": max(
            float(r["magnitude_cosine_preservation_abs_residual"]) for r in rows
        ),
        "max_magnitude_midpoint_abs_residual": max(
            float(r["magnitude_midpoint_max_abs_residual"]) for r in rows
        ),
        "max_magnitude_applied_correction_abs_residual": max(
            float(r["magnitude_applied_correction_max_abs_residual"]) for r in rows
        ),
        "alignment_runtime_correction_l2": _aggregate(
            r["alignment_runtime_correction_l2"] for r in rows
        ),
        "magnitude_runtime_correction_l2": _aggregate(
            r["magnitude_runtime_correction_l2"] for r in rows
        ),
        "state_alignment_reduction": _aggregate(
            r["S_c0_minus_cA"] for r in rows
        ),
        "state_magnitude_reduction": _aggregate(
            r["S_c0_minus_cM"] for r in rows
        ),
        "write_alignment_reduction": _aggregate(
            r["W_c0_minus_cA"] for r in rows
        ),
        "write_magnitude_reduction": _aggregate(
            r["W_c0_minus_cM"] for r in rows
        ),
        "itemwise_counts": {},
        "forbidden_action_flags": {
            "training_executed": False,
            "task_heads_executed": False,
            "logits_read": False,
            "tokenizer_invoked": False,
            "raw_vectors_persisted": False,
            "k1_executed": False,
        },
    }

    out["R_SA"] = out["G_S0"] - out["G_SA"]
    out["R_SM"] = out["G_S0"] - out["G_SM"]
    out["R_WA"] = out["G_W0"] - out["G_WA"]
    out["R_WM"] = out["G_W0"] - out["G_WM"]
    out["F_SA"] = out["R_SA"] / out["G_S0"] if out["G_S0"] > 0.0 else None

    def counts(field: str):
        vals = [float(r[field]) for r in rows]
        return {
            "decrease": sum(v > 0.0 for v in vals),
            "increase": sum(v < 0.0 for v in vals),
            "equal": sum(v == 0.0 for v in vals),
        }

    out["itemwise_counts"]["state_alignment"] = counts("S_c0_minus_cA")
    out["itemwise_counts"]["state_magnitude"] = counts("S_c0_minus_cM")
    out["itemwise_counts"]["write_alignment"] = counts("W_c0_minus_cA")
    out["itemwise_counts"]["write_magnitude"] = counts("W_c0_minus_cM")

    gt = [r for r in rows if float(r["C_corr"]) > float(r["C_ctrl"])]
    lt = [r for r in rows if float(r["C_corr"]) < float(r["C_ctrl"])]
    eq = [r for r in rows if float(r["C_corr"]) == float(r["C_ctrl"])]
    out["alignment_sign_strata"] = {
        "corr_gt_ctrl": {
            "count": len(gt),
            "state": counts_for_subset(gt, "S_c0_minus_cA"),
            "write": counts_for_subset(gt, "W_c0_minus_cA"),
        },
        "corr_lt_ctrl": {
            "count": len(lt),
            "state": counts_for_subset(lt, "S_c0_minus_cA"),
            "write": counts_for_subset(lt, "W_c0_minus_cA"),
        },
        "equal": {
            "count": len(eq),
            "state": counts_for_subset(eq, "S_c0_minus_cA"),
            "write": counts_for_subset(eq, "W_c0_minus_cA"),
        },
    }

    manipulation_ok = (
        out["max_geometry_bridge_abs_residual"] <= BRIDGE_ABS_TOL
        and out["max_native_bridge_abs_residual"] <= BRIDGE_ABS_TOL
        and out["max_alignment_A_preservation_abs_residual"] <= VECTOR_ABS_TOL
        and out["max_alignment_B_preservation_abs_residual"] <= VECTOR_ABS_TOL
        and out["max_alignment_cosine_abs_residual"] <= VECTOR_ABS_TOL
        and out["max_alignment_midpoint_abs_residual"] <= MIDPOINT_TOL
        and out["max_alignment_applied_correction_abs_residual"] <= RUNTIME_CAST_TOL
        and out["max_magnitude_A_abs_residual"] <= VECTOR_ABS_TOL
        and out["max_magnitude_B_abs_residual"] <= VECTOR_ABS_TOL
        and out["max_magnitude_cosine_preservation_abs_residual"] <= VECTOR_ABS_TOL
        and out["max_magnitude_midpoint_abs_residual"] <= MIDPOINT_TOL
        and out["max_magnitude_applied_correction_abs_residual"] <= RUNTIME_CAST_TOL
    )
    out["all_mandatory_manipulation_checks_pass"] = bool(manipulation_ok)
    out["outcome"] = classify(out, manipulation_ok)

    for k, v in out.items():
        if isinstance(v, float):
            require(math.isfinite(v), "NONFINITE_SUMMARY:" + k)
    return out


def counts_for_subset(rows: Sequence[Mapping[str, Any]], field: str):
    vals = [float(r[field]) for r in rows]
    return {
        "decrease": sum(v > 0.0 for v in vals),
        "increase": sum(v < 0.0 for v in vals),
        "equal": sum(v == 0.0 for v in vals),
    }


def _reject_private_payload(obj: Any, path: str = "root") -> None:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(obj):
        raise FalsificationError("RAW_TENSOR_IN_PUBLIC_ARTIFACT:" + path)
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            lower = str(key).lower()
            field_path = path + "." + str(key)
            if lower in PUBLIC_NEGATIVE_BOOLEAN_FLAGS:
                require(
                    value is False,
                    "PUBLIC_NEGATIVE_FLAG_NOT_FALSE:" + field_path,
                )
            else:
                for forbidden in (
                    "raw_vector",
                    "activation_vector",
                    "logits",
                    "checkpoint_bytes",
                ):
                    require(
                        forbidden not in lower,
                        "FORBIDDEN_PUBLIC_FIELD:" + field_path,
                    )
            _reject_private_payload(value, field_path)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _reject_private_payload(value, f"{path}[{i}]")


def _json_bytes(obj: Any) -> bytes:
    _reject_private_payload(obj)
    return (
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def publish(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
):
    require(len(rows) == 330, "PUBLIC_ITEM_COUNT_MISMATCH")
    final = output_dir.resolve()
    partial = Path(str(final) + ".partial")
    require(not final.exists() and not partial.exists(), "OUTPUT_OR_PARTIAL_EXISTS")

    # Validate and serialize the complete public payload before creating any
    # filesystem state.  A schema/privacy rejection must not leave a stale
    # `.partial` directory that blocks the next authenticated execution.
    metrics_bytes = _jsonl_bytes(rows)
    summary_bytes = _json_bytes(summary)
    m = dict(manifest)
    m["outputs"] = {
        ITEM_FILE: sha256_bytes(metrics_bytes),
        SUMMARY_FILE: sha256_bytes(summary_bytes),
    }
    manifest_bytes = _json_bytes(m)

    partial.mkdir(parents=True, exist_ok=False)
    metrics = partial / ITEM_FILE
    summary_path = partial / SUMMARY_FILE
    manifest_path = partial / MANIFEST_FILE

    metrics.write_bytes(metrics_bytes)
    summary_path.write_bytes(summary_bytes)
    manifest_path.write_bytes(manifest_bytes)

    require(
        {p.name for p in partial.iterdir()}
        == {ITEM_FILE, SUMMARY_FILE, MANIFEST_FILE},
        "OUTPUT_SET_MISMATCH",
    )
    os.replace(partial, final)


def _manifest(
    repo: Mapping[str, Any],
    runtime: Mapping[str, Any],
    ctx: Mapping[str, Any],
    *,
    forward_count: int,
):
    require(forward_count == FULL_FORWARD_BUDGET, "MANIFEST_FORWARD_COUNT_MISMATCH")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "runtime_branch": repo["branch"],
        "runtime_git_head": repo["head"],
        "design_freeze_commit": DESIGN_COMMIT,
        "design_blob": DESIGN_BLOB,
        "implementation_commit": repo["implementation_commit"],
        "runner_rel": RUNNER_REL,
        "runner_sha256": repo["runner_sha256"],
        "runner_blob": repo["runner_blob"],
        "stopping_boundary_freeze": STOP_COMMIT,
        "geometry_evidence_freeze": GEOMETRY_EVIDENCE_COMMIT,
        "geometry_implementation_commit": GEOMETRY_IMPLEMENTATION_COMMIT,
        "geometry_runner_sha256": GEOMETRY_RUNNER_SHA256,
        "geometry_runner_blob": GEOMETRY_RUNNER_BLOB,
        "geometry_item_sha256": ctx["artifact_identities"]["geometry_item"]["sha256"],
        "geometry_item_blob": ctx["artifact_identities"]["geometry_item"]["blob"],
        "geometry_channel_sha256": ctx["artifact_identities"]["geometry_channel"]["sha256"],
        "geometry_channel_blob": ctx["artifact_identities"]["geometry_channel"]["blob"],
        "geometry_summary_sha256": ctx["artifact_identities"]["geometry_summary"]["sha256"],
        "geometry_summary_blob": ctx["artifact_identities"]["geometry_summary"]["blob"],
        "geometry_manifest_sha256": ctx["artifact_identities"]["geometry_manifest"]["sha256"],
        "geometry_manifest_blob": ctx["artifact_identities"]["geometry_manifest"]["blob"],
        "carry_evidence_freeze": CARRY_EVIDENCE_COMMIT,
        "carry_implementation_commit": CARRY_IMPLEMENTATION_COMMIT,
        "carry_item_sha256": ctx["artifact_identities"]["carry_item"]["sha256"],
        "carry_item_blob": ctx["artifact_identities"]["carry_item"]["blob"],
        "carry_summary_sha256": ctx["artifact_identities"]["carry_summary"]["sha256"],
        "carry_summary_blob": ctx["artifact_identities"]["carry_summary"]["blob"],
        "handoff_zip_sha256": runtime["parent_runtime"]["handoff"]["zip_sha256"],
        "checkpoint_sha256": runtime["parent_runtime"]["handoff"]["checkpoint_sha256"],
        "encoder_canonical_digest": runtime["parent_runtime"]["encoder"]["canonical_digest"],
        "encoder_raw_concat_digest": runtime["parent_runtime"]["encoder"]["raw_concat_digest"],
        "mamba_source_sha256": runtime["parent_runtime"]["binding"].source_sha256,
        "model_forward_count": int(forward_count),
        "training_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "tokenizer_invoked": False,
        "causal_intervention_executed": True,
        "raw_vectors_persisted": False,
        "k1_executed": False,
    }


def runtime_preflight(root: Path, handoff: Path, output_dir=None):
    require(output_dir is None, "PREFLIGHT_OUTPUT_DIR_FORBIDDEN")
    repo = authenticate_static(root, runtime=True)
    ctx = load_authenticated_parent_stack(root, runtime=True)
    runtime = resolve_runtime(root, ctx, handoff)
    pairs = common_plan_pairs(ctx)

    selected = list(sorted(pairs))[:2]
    require(len(selected) == 2, "PREFLIGHT_SELECTION_FAILURE")
    budget = ForwardBudget(PREFLIGHT_FORWARD_BUDGET)

    rows = [
        _row_for_item(idx, pairs[idx], ctx, runtime, budget)
        for idx in selected
    ]
    require(budget.count == PREFLIGHT_FORWARD_BUDGET, "PREFLIGHT_FORWARD_ACCOUNTING_FAILURE")

    print("PASS_STRONG_ALIGNMENT_CAUSAL_FALSIFICATION_RUNTIME_PREFLIGHT")
    print("runtime_git_head =", repo["head"])
    print("selected_local_template_indices =", selected)
    print("model_forward_count =", budget.count)
    print("scientific_evidence_emitted = False")
    print("output_files_written = 0")
    print("max_geometry_bridge_abs_residual =", max(r["geometry_bridge_max_abs_residual"] for r in rows))
    print("max_native_bridge_abs_residual =", max(r["native_bridge_max_abs_residual"] for r in rows))
    print("max_alignment_midpoint_abs_residual =", max(r["alignment_midpoint_max_abs_residual"] for r in rows))
    print("max_magnitude_midpoint_abs_residual =", max(r["magnitude_midpoint_max_abs_residual"] for r in rows))


def execute(root: Path, handoff: Path, output_dir: Path):
    repo = authenticate_static(root, runtime=True)
    ctx = load_authenticated_parent_stack(root, runtime=True)
    runtime = resolve_runtime(root, ctx, handoff)
    pairs = common_plan_pairs(ctx)

    require(not output_dir.exists(), "OUTPUT_DIR_EXISTS")
    require(not Path(str(output_dir) + ".partial").exists(), "PARTIAL_OUTPUT_EXISTS")

    budget = ForwardBudget(FULL_FORWARD_BUDGET)
    rows = []
    for n, idx in enumerate(sorted(pairs), start=1):
        rows.append(_row_for_item(idx, pairs[idx], ctx, runtime, budget))
        if n % 10 == 0 or n == 330:
            print(
                f"PROGRESS items={n}/330 model_forwards={budget.count}",
                flush=True,
            )

    require(budget.count == FULL_FORWARD_BUDGET, "FULL_FORWARD_ACCOUNTING_FAILURE")
    summary = build_summary(rows, forward_count=budget.count)
    require(
        bool(summary["all_mandatory_manipulation_checks_pass"]),
        "MANDATORY_MANIPULATION_CHECK_FAILURE",
    )
    manifest = _manifest(repo, runtime, ctx, forward_count=budget.count)
    publish(output_dir, rows, summary, manifest)

    print("PASS_STRONG_ALIGNMENT_CAUSAL_FALSIFICATION_EXECUTION")
    print("output_dir =", output_dir.resolve())
    print("model_forward_count =", budget.count)
    print("outcome =", summary["outcome"])
    print("R_SA =", summary["R_SA"])
    print("R_WA =", summary["R_WA"])
    print("R_SM =", summary["R_SM"])
    print("R_WM =", summary["R_WM"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--runtime-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    require(
        sum((args.static_preflight, args.runtime_preflight, args.execute)) == 1,
        "SELECT_ONE_MODE",
    )
    root = Path(__file__).resolve().parents[1]

    if args.static_preflight:
        repo = authenticate_static(root, runtime=False)
        ctx = load_authenticated_parent_stack(root, runtime=False)
        print(
            "PASS_STATIC_AUTHENTICATION",
            json.dumps(
                {
                    **repo,
                    "common_330": len(ctx["common_indices"]),
                    "geometry_item_sha256": ctx["artifact_identities"]["geometry_item"]["sha256"],
                    "geometry_item_blob": ctx["artifact_identities"]["geometry_item"]["blob"],
                    "carry_item_sha256": ctx["artifact_identities"]["carry_item"]["sha256"],
                    "carry_item_blob": ctx["artifact_identities"]["carry_item"]["blob"],
                },
                sort_keys=True,
            ),
        )
        return 0

    require(args.handoff is not None, "HANDOFF_REQUIRED")
    require(args.handoff.is_file(), "HANDOFF_MISSING")

    if args.runtime_preflight:
        runtime_preflight(root, args.handoff, args.output_dir)
        return 0

    require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")
    execute(root, args.handoff, args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FalsificationError as exc:
        print("BLOCKED:", exc)
        raise SystemExit(2)
