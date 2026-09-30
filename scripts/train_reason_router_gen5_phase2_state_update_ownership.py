"""Future Gen5 Phase 2 training harness.

Implementation is present for provenance/static validation only.
Scientific training remains fail-closed until a later training execution
authority explicitly amends this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from contramamba.gen5_phase2_state_update_ownership import ARMS


IMPLEMENTATION_AUTHORITY_COMMIT = (
    "e2f8975d8271c0e95c92b9389dfc4717a221f7df"
)
TRAINING_EXECUTION_AUTHORITY_COMMIT: str | None = None

FROZEN_TRAINING_CONTRACT = {
    "train_rows": 2880,
    "dev_rows": 720,
    "split_seed": 8192,
    "optimizer": "torch.optim.AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "scheduler": None,
    "epochs": 20,
    "optimizer_steps": 20,
    "gradient_clip_norm": 5.0,
    "training_seeds": [5201, 5202, 5203],
    "checkpoint_selection": "FINAL_FIXED_STEP_ONLY",
    "objective": "FINAL_3WAY_CROSS_ENTROPY_ONLY",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gen5 Phase 2 state-update ownership training harness."
    )
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(5201, 5202, 5203), required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--execution-authority-commit")
    parser.add_argument(
        "--print-frozen-contract",
        action="store_true",
        help="Static-only: print the frozen training contract and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.print_frozen_contract:
        print(json.dumps(FROZEN_TRAINING_CONTRACT, sort_keys=True, indent=2))
        print("TRAINING_EXECUTED=False")
        return

    if TRAINING_EXECUTION_AUTHORITY_COMMIT is None:
        raise RuntimeError(
            "GEN5_PHASE2_TRAINING_NOT_AUTHORIZED:"
            "implementation authority does not authorize training"
        )
    if args.execution_authority_commit != TRAINING_EXECUTION_AUTHORITY_COMMIT:
        raise RuntimeError("GEN5_PHASE2_TRAINING_AUTHORITY_COMMIT_MISMATCH")

    raise RuntimeError(
        "GEN5_PHASE2_TRAINING_RUNNER_NOT_OPENED:"
        "a later execution-authority amendment must explicitly enable execution"
    )


if __name__ == "__main__":
    main()
