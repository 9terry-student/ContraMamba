"""Stage26-C: v7 real-Mamba forward-contract validation utility.

Validates that ContraMambaV7Hierarchical with a real Mamba backbone satisfies
the v7 output contract on a small clean-data batch.

This utility is for VALIDATION ONLY. It does NOT:
  - train the model
  - update weights
  - evaluate OOD data
  - perform checkpoint selection
  - save model checkpoints

Historical framing:
  ContraMamba originated the six-axis epistemic framework. EpistemicBERT was a
  pragmatic detour/testbed that operationalized the framework in a strong-backbone
  annotation setting. Stage26 returns the clarified hierarchy to ContraMamba.

Label order (matches FinalLabel in src/contramamba/labels.py):
  REFUTE=0 / NOT_ENTITLED=1 / SUPPORT=2

Usage (once Mamba weights are available locally or via HuggingFace):
  python tools/validate_v7_forward_contract.py

Dummy backbone is refused unless --allow-dummy is explicitly passed. Dummy
results are plumbing-only and must not be treated as model evidence.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import train_controlled_v5 as v5  # noqa: E402
from contramamba.modeling_v7_hierarchical import (  # noqa: E402
    ContraMambaV7Hierarchical,
    V7_REQUIRED_OUTPUT_KEYS,
    validate_v7_output_contract,
)


# ---------------------------------------------------------------------------
# Provenance constants (hardcoded for Stage26-C; never data-driven)
# ---------------------------------------------------------------------------
_STAGE15_USED = False
_OOD_USED = False
_TIME_SWAP_USED_IN_MAIN_CLEAN_DATA = False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Minimal JSONL loader — no schema validation required for a forward check."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_mamba_model(
    model_name: str,
    freeze_a_log: bool = True,
) -> ContraMambaV7Hierarchical:
    """Build v7 with real Mamba backbone. Standard production dimensions."""
    return ContraMambaV7Hierarchical(
        model_name=model_name,
        frame_size=128,
        predicate_size=128,
        sufficiency_size=128,
        polarity_size=64,
        dropout=0.1,
        freeze_a_log=freeze_a_log,
        # All ablation flags default to False (full hierarchical model)
    )


def _build_dummy_model() -> ContraMambaV7Hierarchical:
    """Build v7 with dummy backbone (plumbing-only — not model evidence)."""
    from scripts.train_controlled_v6b_minimal import build_v7_model  # lazy import
    # dummy uses tiny vocab; we only need a non-zero value; real plumbing check
    return build_v7_model(vocab_size=256, max_length=128)


def _encode_batch(
    records: list[dict[str, Any]],
    backbone: str,
    model_name: str,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Encode records using the same path as the training script."""
    if backbone == "dummy":
        vocab = v5.build_vocab(records)
        bundle = v5.encode_records(records, vocab)
    else:
        from transformers import AutoTokenizer  # lazy — only when mamba is used
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Mamba tokenizer has neither pad_token nor eos_token"
                )
            tokenizer.pad_token = tokenizer.eos_token
        bundle = v5.encode_mamba_records(records, tokenizer, max_length)
    # Move all tensors to device
    return v5.move_inputs(bundle["model_inputs"], device)


def _validate_output(
    output: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """Run all contract checks. Returns a dict of check results (no exceptions)."""
    results: dict[str, Any] = {}

    # ── Key contract ──────────────────────────────────────────────────────────
    try:
        validate_v7_output_contract(output)
        results["required_keys_missing"] = []
        results["key_contract_passed"] = True
    except KeyError as exc:
        missing = [k for k in V7_REQUIRED_OUTPUT_KEYS if k not in output]
        results["required_keys_missing"] = missing
        results["key_contract_passed"] = False
        results["key_contract_error"] = str(exc)

    # ── Shape checks ──────────────────────────────────────────────────────────
    shape_errors: list[str] = []

    logits = output.get("logits")
    if logits is not None:
        results["logits_shape"] = list(logits.shape)
        if logits.shape != (batch_size, 3):
            shape_errors.append(
                f"logits shape {tuple(logits.shape)} != ({batch_size}, 3)"
            )
    else:
        results["logits_shape"] = None
        shape_errors.append("logits missing")

    base_logits = output.get("base_logits")
    if base_logits is not None:
        results["base_logits_shape"] = list(base_logits.shape)
        if base_logits.shape != (batch_size, 3):
            shape_errors.append(
                f"base_logits shape {tuple(base_logits.shape)} != ({batch_size}, 3)"
            )
    else:
        results["base_logits_shape"] = None
        shape_errors.append("base_logits missing")

    preds = output.get("predictions")
    if preds is not None:
        results["predictions_shape"] = list(preds.shape)
        if preds.shape != (batch_size,):
            shape_errors.append(
                f"predictions shape {tuple(preds.shape)} != ({batch_size},)"
            )
    else:
        results["predictions_shape"] = None
        shape_errors.append("predictions missing")

    results["shape_errors"] = shape_errors
    results["shape_contract_passed"] = len(shape_errors) == 0

    # ── NaN/Inf check ─────────────────────────────────────────────────────────
    nan_inf_keys: list[str] = []
    if logits is not None:
        if not torch.isfinite(logits).all():
            nan_inf_keys.append("logits")
    results["nan_or_inf_found"] = len(nan_inf_keys) > 0
    results["nan_or_inf_keys"] = nan_inf_keys

    # ── Probability range checks [0, 1] ───────────────────────────────────────
    prob_range_errors: list[str] = []
    prob_keys = [
        "v7_frame_prob",
        "v7_predicate_prob",
        "v7_sufficiency_prob",
        "v7_entitlement_prob",
    ]
    # Temporal prob only if temporal channel is active (may be None)
    if output.get("v7_temporal_prob") is not None:
        prob_keys.append("v7_temporal_prob")

    for key in prob_keys:
        prob = output.get(key)
        if prob is None:
            prob_range_errors.append(f"{key}: missing")
            continue
        if not torch.isfinite(prob).all():
            prob_range_errors.append(f"{key}: non-finite values")
        elif prob.min().item() < 0.0 or prob.max().item() > 1.0:
            prob_range_errors.append(
                f"{key}: out of [0,1] range "
                f"[{prob.min().item():.4f}, {prob.max().item():.4f}]"
            )

    results["prob_range_errors"] = prob_range_errors
    results["prob_range_passed"] = len(prob_range_errors) == 0

    # ── base_logits == logits check ───────────────────────────────────────────
    if logits is not None and base_logits is not None:
        results["base_logits_equals_logits"] = torch.equal(logits, base_logits)
    else:
        results["base_logits_equals_logits"] = None

    # ── Temporal channel active ───────────────────────────────────────────────
    results["temporal_channel_active"] = output.get("v7_temporal_prob") is not None

    # ── v7_final_logit_composition ────────────────────────────────────────────
    results["v7_final_logit_composition"] = output.get(
        "v7_final_logit_composition", "unknown"
    )

    return results


def _overall_status(check_results: dict[str, Any]) -> str:
    passed = (
        check_results.get("key_contract_passed", False)
        and check_results.get("shape_contract_passed", False)
        and not check_results.get("nan_or_inf_found", True)
        and check_results.get("prob_range_passed", False)
    )
    return "PASS" if passed else "FAIL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage26-C: Validate v7 forward-pass output contract on a small clean-data batch."
            " Does NOT train. Does NOT update weights."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/controlled_v5_v3_without_time_swap.jsonl"),
        help="Path to clean main data JSONL (default: data/controlled_v5_v3_without_time_swap.jsonl).",
    )
    parser.add_argument(
        "--model-name",
        default="state-spaces/mamba-130m-hf",
        help="HuggingFace model name for real Mamba backbone (default: state-spaces/mamba-130m-hf).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device to use: 'cuda', 'cpu', etc. "
            "Default: cuda if available, else cpu."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max tokenization length (default: 128).",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=4,
        help="Number of records to use for the forward pass (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for record selection (default: 1).",
    )
    parser.add_argument(
        "--backbone",
        choices=("mamba", "dummy"),
        default="mamba",
        help=(
            "Backbone to use. 'mamba' loads real weights (default). "
            "'dummy' is plumbing-only and requires --allow-dummy."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=("v7_hierarchical",),
        default="v7_hierarchical",
        help="Architecture to validate. Only v7_hierarchical is supported here (default).",
    )
    parser.add_argument(
        "--allow-dummy",
        action="store_true",
        default=False,
        help=(
            "Allow dummy backbone. Required only when --backbone dummy is passed. "
            "Dummy results are plumbing-only and must not be treated as v7 model evidence."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Dummy backbone guard ──────────────────────────────────────────────────
    # Dummy backbone is intentionally refused unless --allow-dummy is explicit.
    # This prevents dummy plumbing results from being confused with real-Mamba evidence.
    if args.backbone == "dummy" and not args.allow_dummy:
        sys.exit(
            "ERROR: dummy backbone is plumbing-only and cannot be used as v7 model evidence.\n"
            "Pass --allow-dummy explicitly if you intentionally want dummy validation."
        )

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Resolve data path relative to repo root if not absolute ──────────────
    data_path = args.data
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    # ── Load records ──────────────────────────────────────────────────────────
    all_records = _load_jsonl(data_path)
    rng = random.Random(args.seed)
    n = min(args.num_records, len(all_records))
    records = rng.sample(all_records, n)
    batch_size = len(records)

    print(f"[validate_v7_forward_contract] loaded {len(all_records)} records, "
          f"sampled {batch_size} with seed={args.seed}")

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"[validate_v7_forward_contract] building v7 model "
          f"(backbone={args.backbone}, model_name={args.model_name!r})")
    if args.backbone == "dummy":
        model = _build_dummy_model()
        dummy_used = True
    else:
        model = _build_mamba_model(args.model_name)
        dummy_used = False

    model.eval()
    model.to(device)

    # ── Encode batch ──────────────────────────────────────────────────────────
    print(f"[validate_v7_forward_contract] encoding {batch_size} records "
          f"(backbone={args.backbone}, max_length={args.max_length})")
    full_inputs = _encode_batch(
        records,
        backbone=args.backbone,
        model_name=args.model_name,
        max_length=args.max_length,
        device=device,
    )

    # Pass only the 4 model feature keys to forward; v7 ignores v6B-specific kwargs
    feature_inputs = v5.model_feature_inputs(full_inputs)

    # ── Forward pass (no gradients, no weight updates) ────────────────────────
    print("[validate_v7_forward_contract] running single forward pass (no_grad)")
    with torch.no_grad():
        output = model(**feature_inputs)

    # ── Contract validation ───────────────────────────────────────────────────
    check_results = _validate_output(output, batch_size)
    status = _overall_status(check_results)

    # ── Build summary ─────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "status": status,
        "architecture": args.architecture,
        "backbone": args.backbone,
        "model_name": args.model_name,
        "device": str(device),
        "data": str(data_path),
        "num_records": len(all_records),
        "batch_size": batch_size,
        "seed": args.seed,
        "max_length": args.max_length,
        # Shape results
        "logits_shape": check_results["logits_shape"],
        "base_logits_shape": check_results["base_logits_shape"],
        "predictions_shape": check_results.get("predictions_shape"),
        # Label order (hardcoded; validated from labels.py)
        "label_order": {
            "dim_0": "REFUTE (FinalLabel.REFUTE=0)",
            "dim_1": "NOT_ENTITLED (FinalLabel.NOT_ENTITLED=1)",
            "dim_2": "SUPPORT (FinalLabel.SUPPORT=2)",
        },
        # Contract results
        "key_contract_passed": check_results["key_contract_passed"],
        "required_keys_missing": check_results["required_keys_missing"],
        "shape_errors": check_results["shape_errors"],
        "nan_or_inf_found": check_results["nan_or_inf_found"],
        "nan_or_inf_keys": check_results["nan_or_inf_keys"],
        "prob_range_passed": check_results["prob_range_passed"],
        "prob_range_errors": check_results["prob_range_errors"],
        "base_logits_equals_logits": check_results["base_logits_equals_logits"],
        "temporal_channel_active": check_results["temporal_channel_active"],
        "v7_final_logit_composition": check_results["v7_final_logit_composition"],
        # Provenance / exclusion fields
        "dummy_used": dummy_used,
        "dummy_evidence_allowed": args.allow_dummy,
        "stage15_used": _STAGE15_USED,
        "ood_used": _OOD_USED,
        "time_swap_used_in_main_clean_data": _TIME_SWAP_USED_IN_MAIN_CLEAN_DATA,
    }

    print("\n" + json.dumps(summary, indent=2))

    if status == "PASS":
        print("\n[validate_v7_forward_contract] Contract: PASS")
    else:
        print("\n[validate_v7_forward_contract] Contract: FAIL — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
