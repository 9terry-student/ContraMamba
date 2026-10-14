#!/usr/bin/env python3
"""Gen5 G1 frozen-P0-factor CPU installation/restore diagnostic.

Uses three authentic SHA256-verified Phase3A P0 correction factor pairs and
frozen historical owner bases. Uses a *synthetic* parent/layer-22 mixer and
synthetic hidden input. Never loads a pretrained parent, tokenizer, task data,
or performs scientific evaluation, gradients, optimization, or GPU execution.
Writes only a standalone diagnostic JSON outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
from torch import nn

HEAD = "7e19ecbd61c1108282b1ad2602c8d6dfc94c7469"
BRANCH = "gen5-causal-role-state-ownership"
EXECUTION_HEAD = "d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e"
CHECKPOINT_ROOT = (
    "reports/reason_router_gen5_phase3a_training_runs/"
    "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells"
)
CKPT_SHA = {
    6201: "157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf",
    6202: "1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214",
    6203: "c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770",
}
PREFLIGHT_SHA = "82f93e692a84c6890c36179fe522f1ee30e61ccf002f07836cfbe93f26a46641"
PARENT_CKPT_SHA = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
R22_SHA = "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
C22_SHA = "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"


def require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError("BLOCKED:" + why)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    require(t.device.type == "cpu", "NON_CPU_TENSOR")
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def compare(label: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    require(actual.device.type == expected.device.type == "cpu", f"NONCPU:{label}")
    require(actual.shape == expected.shape, f"SHAPE:{label}")
    require(bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
            f"NONFINITE:{label}")
    torch.testing.assert_close(actual, expected, rtol=4e-4, atol=2e-6, msg=label)
    error = actual.double() - expected.double()
    max_abs = float(error.abs().max())
    relative = float(torch.linalg.vector_norm(error)) / max(
        1e-12, float(torch.linalg.vector_norm(expected.double()))
    )
    return {"max_abs": max_abs, "relative_frobenius": relative}


def load_real_sources(repo: Path, r22: torch.Tensor, c22: torch.Tensor):
    from contramamba.gen5_phase2_state_update_ownership import StateWriteCorrection
    from contramamba.gen5_shared_bypass_functional_core_diagnostic import EXPECTED_SOURCE_TENSOR_SHA256
    loaded = {}
    for seed in sorted(CKPT_SHA):
        ckpt = repo / CHECKPOINT_ROOT / f"seed{seed}" / "P0" / "final_correction.pt"
        require(ckpt.is_file(), f"CHECKPOINT_NOT_FOUND:{seed}")
        require(sha_file(ckpt) == CKPT_SHA[seed], f"CHECKPOINT_SHA:{seed}")
        payload = torch.load(ckpt, weights_only=True, map_location="cpu")
        require(type(payload) is dict, f"PAYLOAD_TYPE:{seed}")
        for field, expected in (
            ("schema_version", "GEN5_PHASE3A_FINAL_CORRECTION_V1"),
            ("execution_commit", EXECUTION_HEAD),
            ("parent_checkpoint_sha256", PARENT_CKPT_SHA),
            ("r22_sha256", R22_SHA), ("c22_sha256", C22_SHA),
            ("arm", "G5-C0"), ("pressure", "P0"), ("seed", seed),
        ):
            require(payload.get(field) == expected, f"PAYLOAD_{field}:{seed}")
        tensors = payload.get("state_dict")
        require(isinstance(tensors, dict) and
                set(tensors) == {"A_theta.weight", "B_theta.weight"},
                f"STATE_DICT:{seed}")
        declared = payload.get("tensor_sha256")
        require(isinstance(declared, dict), f"TENSOR_SHA_DICT:{seed}")
        for key, shape in (
            ("A_theta.weight", (2, 768)),
            ("B_theta.weight", (24576, 2)),
        ):
            tensor = tensors[key]
            require(isinstance(tensor, torch.Tensor) and
                    tensor.dtype == torch.float32 and tuple(tensor.shape) == shape,
                    f"FACTOR_SHAPE_DTYPE:{seed}:{key}")
            require(bool(torch.isfinite(tensor).all()), f"FACTOR_NONFINITE:{seed}:{key}")
            expected_sha = EXPECTED_SOURCE_TENSOR_SHA256[seed][key]
            require(sha_tensor(tensor) == declared.get(key) == expected_sha,
                    f"FACTOR_IDENTITY:{seed}:{key}")
        module = StateWriteCorrection(arm="G5-C0", r22=r22, c22=c22, seed=seed)
        with torch.no_grad():
            module.A_theta.weight.copy_(tensors["A_theta.weight"])
            module.B_theta.weight.copy_(tensors["B_theta.weight"])
        module.requires_grad_(False).eval()
        for key, weight in (
            ("A_theta.weight", module.A_theta.weight),
            ("B_theta.weight", module.B_theta.weight),
        ):
            require(sha_tensor(weight) == EXPECTED_SOURCE_TENSOR_SHA256[seed][key],
                    f"LOADED_FACTOR:{seed}:{key}")
        loaded[seed] = module
    return loaded


def synthetic_parent(mixer: nn.Module):
    class Block(nn.Module):
        def __init__(self, idx: int):
            super().__init__()
            self.mixer = mixer if idx == 22 else nn.Identity()

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=False)
            self.layers = nn.ModuleList(Block(i) for i in range(24))

        def forward(self, input_ids: torch.Tensor):
            # Synthetic path only; this is NOT the pretrained Mamba backbone.
            return self.layers[22].mixer(input_ids)

    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.mamba = Backbone()

    return Parent().requires_grad_(False).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    out = args.out.resolve()
    require(git(repo, "rev-parse", "HEAD") == HEAD, "HEAD_MISMATCH")
    require(git(repo, "branch", "--show-current") == BRANCH, "BRANCH_MISMATCH")
    require(git(repo, "status", "--porcelain") == "", "WORKTREE_DIRTY")
    require(not out.exists(), "OUTPUT_COLLISION")
    require(not out.is_relative_to(repo), "OUTPUT_MUST_BE_OFF_REPO")
    require(args.preflight.is_file(), "PREFLIGHT_MISSING")
    require(sha_file(args.preflight) == PREFLIGHT_SHA, "PREFLIGHT_SHA")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    require(preflight.get("status") == "STATIC_ARM_IDENTITY_PASS_NOT_EVALUATION_AUTHORITY",
            "PREFLIGHT_STATUS")
    require(preflight.get("seeds") == sorted(CKPT_SHA), "PREFLIGHT_SEEDS")
    for seed, sha in CKPT_SHA.items():
        require(preflight["checkpoint_identities"][str(seed)]["checkpoint_sha256"] == sha,
                f"PREFLIGHT_CHECKPOINT_IDENTITY:{seed}")
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "tests"))
    torch.set_num_threads(2)

    from contramamba import gen5_shared_bypass_functional_core_diagnostic as diag
    from contramamba.gen5_phase2_state_update_ownership import (
        Phase2Layer22MixerWrapper, accelerated_correction_mixer_contribution,
        load_frozen_owner_bases, parent_parameter_fingerprint, phase2_active_mask,
    )
    from contramamba.gen5_shared_bypass_g1_installation_diagnostic import (
        _native_buffer_sha256, installed_g1_synthetic_diagnostic,
        g1_companion_active_mask,
    )
    from test_gen5_shared_bypass_runtime_equivalence_synthetic import SyntheticLayer22Mixer

    require(tuple(diag.SEEDS) == (6201, 6202, 6203), "SEED_CONTRACT")
    require(len(diag.ARMS) == 6, "ARM_CONTRACT")
    r22, c22, _ = load_frozen_owner_bases(repo)
    require(r22.device.type == c22.device.type == "cpu", "BASIS_NON_CPU")
    sources = load_real_sources(repo, r22, c22)
    diag.validate_frozen_sources(sources)
    for heldout in diag.SEEDS:
        for arm in diag.ARMS:
            actual = diag.select_frozen_coefficients(heldout, arm)
            expected = preflight["arm_contract"][str(heldout)]["arm_operator_coefficient_vectors"][arm]
            require(len(expected) == len(actual) == 3 and
                    all(math.isclose(a, e, abs_tol=1e-13, rel_tol=0)
                        for a, e in zip(actual, expected)),
                    f"COEFFICIENT_MISMATCH:{heldout}:{arm}")

    native = SyntheticLayer22Mixer().requires_grad_(False).eval()
    model = synthetic_parent(native)
    base_params = parent_parameter_fingerprint(model)
    base_buffers = _native_buffer_sha256(model)
    original_ids = tuple(id(layer.mixer) for layer in model.mamba.layers)
    factor_sha_before = {
        str(s): {
            "A": sha_tensor(src.A_theta.weight),
            "B": sha_tensor(src.B_theta.weight),
        } for s, src in sources.items()
    }
    x = torch.randn((1, 3, 768), generator=torch.Generator(device="cpu").manual_seed(7)) * .6
    mask = torch.tensor([[1, 0, 1]], dtype=torch.long)
    zeros = torch.zeros_like(mask)
    arm_results = {}
    count = 0
    rollback_on_exception = False
    double_install_rejected = False
    old_mask_helper_rejected = False
    max_companion_error = 0.0

    with torch.inference_mode():
        native_out = native(x)
        correction_out = {
            s: accelerated_correction_mixer_contribution(
                native, x, src, attention_mask=mask, checkpoint_recompute=False,
            ) for s, src in sources.items()
        }
        for heldout in diag.SEEDS:
            seed_rows = {}
            for arm in diag.ARMS:
                before_calls = native.forward_calls
                with installed_g1_synthetic_diagnostic(
                    model, source_corrections=sources, heldout_seed=heldout, arm=arm,
                ) as wrapper:
                    require(model.mamba.layers[22].mixer is wrapper, f"INSTALL_OBJECT:{heldout}:{arm}")
                    require(wrapper.native_mixer is native, f"INSTALL_NATIVE:{heldout}:{arm}")
                    require(not wrapper.training and not any(p.requires_grad for p in model.parameters()),
                            f"INSTALL_EVAL_FROZEN:{heldout}:{arm}")
                    if heldout == 6201 and arm == "FULL":
                        try:
                            from contramamba.gen5_shared_bypass_g1_installation_diagnostic import (
                                install_g1_synthetic_diagnostic,
                            )
                            install_g1_synthetic_diagnostic(
                                model, source_corrections=sources,
                                heldout_seed=heldout, arm=arm,
                            )
                        except diag.DiagnosticContractError as err:
                            require("G1_ALREADY_WRAPPED" in str(err), "DOUBLE_INSTALL_WRONG_REJECTION")
                            double_install_rejected = True
                        try:
                            with phase2_active_mask(model, mask):
                                pass
                        except Exception as err:
                            require("LAYER22_NOT_WRAPPED" in str(err), "OLD_MASK_WRONG_REJECTION")
                            old_mask_helper_rejected = True
                    with g1_companion_active_mask(model, mask):
                        y = model.mamba(input_ids=x)
                    require(wrapper._phase2_active_mask is None, f"MASK_LEAK:{heldout}:{arm}")
                    require(native.forward_masks[-1] is None, f"NATIVE_MASK_ALTERED:{heldout}:{arm}")
                require(model.mamba.layers[22].mixer is native, f"RESTORE_REF:{heldout}:{arm}")
                require(tuple(id(layer.mixer) for layer in model.mamba.layers) == original_ids,
                        f"RESTORE_IDS:{heldout}:{arm}")
                require(parent_parameter_fingerprint(model) == base_params,
                        f"PARENT_PARAMETER_MUTATION:{heldout}:{arm}")
                require(_native_buffer_sha256(model) == base_buffers,
                        f"PARENT_BUFFER_MUTATION:{heldout}:{arm}")
                require(native.forward_calls == before_calls + 1,
                        f"NATIVE_CALL_COUNT:{heldout}:{arm}")
                weights = diag.select_frozen_coefficients(heldout, arm)
                expected_delta = sum((correction_out[s] * c
                                      for s, c in zip(diag.SEEDS, weights) if c != 0),
                                     torch.zeros_like(native_out))
                expected = native_out + expected_delta
                output_check = compare(f"ARM:{heldout}:{arm}", y, expected)
                delta_check = compare(f"DELTA:{heldout}:{arm}", y - native_out, expected_delta)
                if torch.linalg.vector_norm(expected_delta.double()).item() > 1e-7:
                    require(delta_check["relative_frobenius"] < 0.02,
                            f"CORRECTION_RELATIVE_ERROR:{heldout}:{arm}")
                max_companion_error = max(max_companion_error, delta_check["relative_frobenius"])
                seed_rows[arm] = {"output": output_check, "correction_only": delta_check}
                count += 1

            # Compare original FULL wrapper without changing installed model.
            old = Phase2Layer22MixerWrapper(native_mixer=native, correction=sources[heldout]).eval()
            with old.active_mask(mask):
                old_y = old(x)
            expected_full = native_out + correction_out[heldout]
            seed_rows["OLD_FULL_REFERENCE"] = compare(
                f"OLD_FULL:{heldout}", old_y, expected_full,
            )
            seed_rows["RECONSTRUCTION"] = compare(
                f"RECON:{heldout}",
                native_out + (correction_out[heldout]),
                native_out + sum((correction_out[s] * c for s, c in
                                  zip(diag.SEEDS, diag.select_frozen_coefficients(heldout,"SHARED_ONLY"))),
                                 torch.zeros_like(native_out)) +
                             sum((correction_out[s] * c for s, c in
                                  zip(diag.SEEDS, diag.select_frozen_coefficients(heldout,"RESIDUAL_ONLY"))),
                                 torch.zeros_like(native_out)),
            )
            arm_results[str(heldout)] = seed_rows

        # A controlled user exception must restore the same native mixer.
        try:
            with installed_g1_synthetic_diagnostic(
                model, source_corrections=sources,
                heldout_seed=6201, arm="SHARED_ONLY",
            ):
                with g1_companion_active_mask(model, zeros):
                    raise ValueError("G1_INTENTIONAL_ABORT")
        except ValueError as err:
            require(str(err) == "G1_INTENTIONAL_ABORT", "EXCEPTION_MISMATCH")
            rollback_on_exception = True
    require(count == 18, "ARM_COUNT")
    require(rollback_on_exception and double_install_rejected and old_mask_helper_rejected,
            "NEGATIVE_CONTROLS_MISSING")
    require(model.mamba.layers[22].mixer is native, "FINAL_NATIVE_NOT_RESTORED")
    require(tuple(id(layer.mixer) for layer in model.mamba.layers) == original_ids,
            "FINAL_LAYER_IDENTITIES_CHANGED")
    require(parent_parameter_fingerprint(model) == base_params, "FINAL_PARENT_PARAMETERS_CHANGED")
    require(_native_buffer_sha256(model) == base_buffers, "FINAL_PARENT_BUFFERS_CHANGED")
    diag.validate_frozen_sources(sources)
    factor_sha_after = {
        str(s): {"A": sha_tensor(src.A_theta.weight), "B": sha_tensor(src.B_theta.weight)}
        for s, src in sources.items()
    }
    require(factor_sha_after == factor_sha_before, "FINAL_FACTOR_MUTATION")
    require(all(t.device.type == "cpu" for t in model.parameters()), "FINAL_PARENT_NONCPU")
    artifact = {
        "schema": "GEN5_G1_FROZEN_P0_CPU_SYNTHETIC_PARENT_INSTALL_V1",
        "status": "PASS_G1_FROZEN_FACTORS_SYNTHETIC_PARENT_NOT_PRETRAINED",
        "repo_head": HEAD,
        "historical_execution_head": EXECUTION_HEAD,
        "frozen_checkpoint_sha256": {str(k): v for k, v in CKPT_SHA.items()},
        "preflight_json_sha256": PREFLIGHT_SHA,
        "source_factor_sha256": factor_sha_after,
        "installed_and_restored_arm_count": count,
        "legacy_mask_helper_rejected": old_mask_helper_rejected,
        "duplicate_install_rejected": double_install_rejected,
        "exception_rollback_pass": rollback_on_exception,
        "parent_parameter_buffer_identity_unchanged": True,
        "frozen_factor_identity_unchanged": True,
        "max_correction_only_relative_error": max_companion_error,
        "per_seed": arm_results,
        "synthetic_parent": True,
        "pretrained_model_loaded": False,
        "pretrained_model_forward": False,
        "task_evaluation": False,
        "training": False,
        "gpu": False,
        "limits": [
            "G1 installation is on synthetic parent with pretrained-shaped layer22, not original pretrained model",
            "Actual frozen P0 correction A/B factor bytes and exact frozen owner bases",
            "Checkpoint geometry and synthetic functional insertion, not task-functional evidence",
            "All numerical checks use synthetic hidden input and synthetic native mixer",
            "No evaluation authority is granted by this PASS",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("GEN5_G1_FROZEN_CHECKPOINT_CPU_INSTALLATION_PASS")
    print(f"HEAD={HEAD}; CHECKPOINTS=3; INSTALLED_RESTORED_ARMS={count}; PARENT_UNCHANGED=YES")
    print("DOUBLE_INSTALL=BLOCKED; OLD_MASK_HELPER=BLOCKED; EXCEPTION_ROLLBACK=PASS")
    print("PRETRAINED_BACKBONE=NO; PRETRAINED_FORWARD=NO; TASK_EVALUATION=NO; TRAINING=NO; GPU=NO")
    print(f"MAX_CORRECTION_ONLY_REL={max_companion_error:.10g}")
    print(f"RESULT_JSON={out}")
    print(f"RESULT_SHA256={sha_file(out)}")

if __name__ == "__main__":
    main()
