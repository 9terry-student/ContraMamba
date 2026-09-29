#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts import reason_router_gen5_phase1b_r22_c22_construction as runner

RUNNER_PATH = ROOT / "scripts/reason_router_gen5_phase1b_r22_c22_construction.py"
EXPECTED_CANONICAL_RUNNER_SHA256 = "d92f829a4351b631d213c86755c754e7e5fd5f09240be4e1be6382bbc0421236"


class VerificationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise VerificationError(message)


def canonical_source_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return text.encode("utf-8")


def verify_source_identity_and_firewall() -> None:
    raw = canonical_source_bytes(RUNNER_PATH)
    require(
        hashlib.sha256(raw).hexdigest() == EXPECTED_CANONICAL_RUNNER_SHA256,
        "RUNNER_CANONICAL_SHA256",
    )
    text = raw.decode("utf-8")
    require(
        "reason_router_gen5_phase1b_xg1_necessity_confirmation_v1" not in text,
        "NECESSITY_CONFIRMATION_PATH_PRESENT",
    )
    require(
        "reason_router_gen5_phase1b_xg1_restoration_confirmation_v1" not in text,
        "RESTORATION_CONFIRMATION_PATH_PRESENT",
    )
    tree = ast.parse(text)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_canonical_analysis_tokenizer"
    ]
    require(len(calls) == 1, "TOKENIZER_LOAD_CALL_COUNT")
    require(runner.FULL_FORWARD_BUDGET == 1800, "FORWARD_BUDGET")
    require(runner.LAYER22 == 22, "LAYER22")
    require(runner.RANK == 2, "RANK")


def verify_frozen_dependency_blobs() -> None:
    for path, expected in runner.FROZEN_BLOBS.items():
        observed = runner.git("rev-parse", f"HEAD:{path}")
        require(observed == expected, f"FROZEN_BLOB:{path}")


def verify_synthetic_native_write_observer() -> None:
    class Mixer:
        def slow(self, discrete_A, deltaB_u):
            ssm_state = torch.zeros((1, 1536, 16), dtype=torch.float32)
            for i in range(discrete_A.shape[2]):
                ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
                scan_output = ssm_state.sum()
                del scan_output
            return ssm_state

    source, first = inspect.getsourcelines(Mixer.slow)
    update = None
    readout = None
    for offset, line in enumerate(source):
        if "ssm_state = discrete_A" in line:
            update = first + offset
        if "scan_output = ssm_state.sum()" in line:
            readout = first + offset
    require(update is not None and readout is not None, "SYNTHETIC_LINE_BINDING")

    mixer = Mixer()
    g = torch.ones((1, 1536, 2, 16), dtype=torch.float32)
    w = torch.zeros((1, 1536, 2, 16), dtype=torch.float32)
    w[:, :, 0, :] = 1.25
    collector = runner.Layer22NativeWriteCollector(
        code=Mixer.slow.__code__,
        update_line=int(update),
        readout_line=int(readout),
        mixer22=mixer,
        target_indices=(0,),
    )
    with collector.capture():
        _ = mixer.slow(g, w)
    require(collector.records is not None and set(collector.records) == {0}, "OBSERVER_RECORD")
    record = collector.records[0]
    require(torch.equal(record.w, w[:, :, 0, :]), "WRITE_CAPTURE")
    require(torch.equal(record.s_post, w[:, :, 0, :]), "POST_STATE_CAPTURE")
    require(record.reconstruction_relative_residual == 0.0, "RECONSTRUCTION")


def verify_artifact_boundary() -> None:
    allowed = {
        runner.R22_FILE,
        runner.C22_FILE,
        runner.ITEM_FILE,
        runner.SUMMARY_FILE,
        runner.MANIFEST_FILE,
        runner.CHECKSUM_FILE,
    }
    require(len(allowed) == 6, "ARTIFACT_FILE_COUNT")
    forbidden = {
        "raw_w.npy",
        "raw_s.npy",
        "dw.npy",
        "ds.npy",
        "logits.json",
        "predictions.json",
    }
    require(not (allowed & forbidden), "RAW_OR_TASK_ARTIFACT_ALLOWED")


def main() -> int:
    verify_source_identity_and_firewall()
    verify_frozen_dependency_blobs()
    verify_synthetic_native_write_observer()
    verify_artifact_boundary()
    print("RESULT=PASS_GEN5_PHASE1B_R22_C22_IMPLEMENTATION_VERIFICATION")
    print("MODEL_LOADED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("SCIENTIFIC_EXECUTION=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
