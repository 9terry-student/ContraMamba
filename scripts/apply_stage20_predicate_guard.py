"""Apply Stage20-A predicate guard to Stage19 calibrated predictions.

This is a post-processing diagnostic. The oracle detector is an upper bound;
the lexical detector is intentionally conservative and targets known predicate
family conflicts rather than generic text differences.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_stage17_temporal_comparator import LABELS, load_jsonl, softmax  # noqa: E402


DEFAULT_PROBE = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"
DEFAULT_PRED_GLOB = str(ROOT / "results" / "stage19_frame_model_temporal_bias_seed*.json")
DEFAULT_OUTPUT_DIR = ROOT / "results"
EPS = 1e-8

PREDICATE_PATTERNS: dict[str, tuple[str, ...]] = {
    "approve": ("approve", "approved", "approves", "approval"),
    "review": ("review", "reviewed", "reviews"),
    "open": ("open", "opened", "opens", "reopen", "reopened"),
    "plan": ("plan", "planned", "plans"),
    "launch": ("launch", "launched", "launches"),
    "test": ("test", "tested", "tests"),
    "publish": ("publish", "published", "publishes", "publisher", "publication"),
    "present": ("present", "presented", "presents"),
    "deliver": ("deliver", "delivered", "delivers"),
    "inspect": ("inspect", "inspected", "inspects"),
    "restore": ("restore", "restored", "restores"),
    "survey": ("survey", "surveyed", "surveys"),
    "select": ("select", "selected", "selects"),
    "evaluate": ("evaluate", "evaluated", "evaluates"),
    "map": ("map", "mapped", "maps"),
    "patrol": ("patrol", "patrolled", "patrols"),
    "upgrade": ("upgrade", "upgraded", "upgrades"),
    "visit": ("visit", "visited", "visits"),
    "screen": ("screen", "screened", "screens"),
    "develop": ("develop", "developed", "developer", "development"),
    "distribute": ("distribute", "distributed", "distributor"),
    "direct": ("direct", "directed", "director"),
    "star": ("star", "starred", "starring"),
    "write": ("write", "wrote", "written", "writer"),
    "produce": ("produce", "produced", "producer"),
    "found": ("found", "founded", "founder"),
    "join": ("join", "joined"),
    "manage": ("manage", "managed", "manager"),
    "own": ("own", "owned", "owner"),
    "release": ("release", "released"),
    "announce": ("announce", "announced"),
    "execute": ("execute", "executed"),
}

CONFLICT_PAIRS = {
    frozenset(pair)
    for pair in (
        ("approve", "review"),
        ("open", "plan"),
        ("launch", "test"),
        ("publish", "present"),
        ("deliver", "inspect"),
        ("restore", "survey"),
        ("select", "evaluate"),
        ("map", "patrol"),
        ("upgrade", "visit"),
        ("select", "screen"),
        ("develop", "publish"),
        ("develop", "distribute"),
        ("publish", "distribute"),
        ("direct", "star"),
        ("direct", "write"),
        ("direct", "produce"),
        ("star", "write"),
        ("star", "produce"),
        ("found", "join"),
        ("found", "manage"),
        ("manage", "own"),
        ("release", "announce"),
        ("plan", "execute"),
    )
}

STOPWORDS = {
    "the", "a", "an", "in", "on", "during", "and", "or", "of", "to", "for",
    "by", "with", "is", "was", "were", "did", "not", "mr", "ms", "dr",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    return {tok for tok in tokens(text) if tok not in STOPWORDS and len(tok) > 2}


def predicate_families(text: str) -> list[str]:
    toks = set(tokens(text))
    families = []
    for family, variants in PREDICATE_PATTERNS.items():
        if any(variant in toks for variant in variants):
            families.append(family)
    return sorted(set(families))


def overlap_ratio(claim: str, evidence: str) -> float:
    left = content_tokens(claim)
    right = content_tokens(evidence)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def lexical_predicate_detector(claim: str, evidence: str) -> tuple[int, list[str], list[str], str, str]:
    claim_families = predicate_families(claim)
    evidence_families = predicate_families(evidence)
    overlap = overlap_ratio(claim, evidence)
    if overlap < 0.65:
        return 0, claim_families, evidence_families, "", f"low_overlap:{overlap:.3f}"
    conflicts = []
    for left in claim_families:
        for right in evidence_families:
            if left != right and frozenset((left, right)) in CONFLICT_PAIRS:
                conflicts.append(f"{left}|{right}")
    if conflicts:
        return 1, claim_families, evidence_families, conflicts[0], f"known_conflict;overlap:{overlap:.3f}"
    return 0, claim_families, evidence_families, "", f"no_known_conflict;overlap:{overlap:.3f}"


def metadata_predicate_detector(probe: dict[str, Any]) -> tuple[int, str]:
    source = probe.get("source_intervention_type")
    original = probe.get("stage15_original_probe_type")
    if source == "predicate_swap" or original == "predicate_swap":
        return 1, "metadata_predicate_swap"
    return 0, "metadata_no_predicate_swap"


def predicate_flag(
    *,
    probe: dict[str, Any],
    detector_mode: str,
) -> tuple[int, list[str], list[str], str, str]:
    if detector_mode == "oracle_probe_type":
        flag = int(probe.get("stage15_probe_type") == "predicate_mismatch")
        return flag, [], [], "", "oracle_probe_type_upper_bound"
    if detector_mode == "metadata_predicate":
        flag, note = metadata_predicate_detector(probe)
        return flag, [], [], "", note
    if detector_mode == "lexical_predicate":
        return lexical_predicate_detector(probe["claim"], probe["evidence"])
    raise ValueError(f"unknown detector mode: {detector_mode}")


def pseudo_logits_from_probs(probs: Sequence[float]) -> list[float]:
    return [math.log(max(float(prob), EPS)) for prob in probs]


def adjusted_probs_from_base(base_probs: Sequence[float] | None, penalty: float) -> list[float] | None:
    if base_probs is None:
        return None
    logits = pseudo_logits_from_probs(base_probs)
    logits[0] -= penalty
    logits[1] += penalty
    logits[2] -= penalty
    return softmax(logits)


def base_prediction(row: dict[str, Any]) -> str:
    return row.get("adjusted_pred_final_label") or row.get("pred_final_label") or row.get("original_pred_final_label")


def base_probs(row: dict[str, Any]) -> list[float] | None:
    probs = row.get("adjusted_final_probs") or row.get("final_probs") or row.get("original_final_probs")
    return [float(x) for x in probs] if probs is not None else None


def apply_guard_to_payload(
    *,
    payload: dict[str, Any],
    probe_by_id: dict[str, dict[str, Any]],
    detector_mode: str,
    patch_mode: str,
    predicate_penalty: float,
) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    seed = int(metadata.get("seed", 0))
    rows = []
    for pred in payload.get("predictions", []):
        probe = probe_by_id.get(pred.get("id"))
        if probe is None:
            continue
        flag, claim_families, evidence_families, conflict, note = predicate_flag(
            probe=probe,
            detector_mode=detector_mode,
        )
        base_pred = base_prediction(pred)
        adjusted_pred = base_pred
        adjusted_probs = base_probs(pred)
        if flag:
            if patch_mode == "hard_override":
                adjusted_pred = "NOT_ENTITLED"
            elif patch_mode == "pseudo_logit_penalty":
                adjusted_probs = adjusted_probs_from_base(adjusted_probs, predicate_penalty)
                if adjusted_probs is not None:
                    adjusted_pred = LABELS[max(range(3), key=lambda index: adjusted_probs[index])]
            else:
                raise ValueError(f"unknown patch mode: {patch_mode}")
        rows.append(
            {
                **pred,
                "seed": seed,
                "stage15_probe_type": probe.get("stage15_probe_type"),
                "base_pred_final_label": base_pred,
                "predicate_adjusted_pred_final_label": adjusted_pred,
                "gold_final_label": pred.get("gold_final_label") or probe.get("final_label") or probe.get("label"),
                "predicate_mismatch_flag": flag,
                "predicate_detector_mode": detector_mode,
                "predicate_patch_mode": patch_mode,
                "base_final_probs": base_probs(pred),
                "predicate_adjusted_final_probs": adjusted_probs,
                "claim_predicate_families": claim_families,
                "evidence_predicate_families": evidence_families,
                "predicate_conflict_pair": conflict,
                "detector_notes": note,
            }
        )
    out_meta = {
        **metadata,
        "predicate_detector_mode": detector_mode,
        "predicate_patch_mode": patch_mode,
        "predicate_penalty": predicate_penalty,
        "predicate_flag_counts_by_stage15_probe_type": dict(
            Counter(row["stage15_probe_type"] for row in rows if row["predicate_mismatch_flag"])
        ),
        "note": "Stage20-A predicate guard is a diagnostic post-processing upper bound/detector analysis, not a trained model.",
    }
    return {"metadata": out_meta, "predictions": rows}


def print_diagnostics(payloads: Sequence[dict[str, Any]]) -> None:
    all_rows = [row for payload in payloads for row in payload["predictions"]]
    counts = Counter(row["stage15_probe_type"] for row in all_rows if row["predicate_mismatch_flag"])
    print(f"predicate_flag_counts_by_stage15_probe_type\t{json.dumps(dict(sorted(counts.items())), sort_keys=True)}")
    flagged_pred = [r for r in all_rows if r["stage15_probe_type"] == "predicate_mismatch" and r["predicate_mismatch_flag"]][:3]
    unflagged_pred = [r for r in all_rows if r["stage15_probe_type"] == "predicate_mismatch" and not r["predicate_mismatch_flag"]][:3]
    false_pos = [r for r in all_rows if r["stage15_probe_type"] != "predicate_mismatch" and r["predicate_mismatch_flag"]][:3]
    for label, items in (("flagged_predicate_examples", flagged_pred), ("unflagged_predicate_examples", unflagged_pred), ("non_predicate_flagged_examples", false_pos)):
        print(label)
        for row in items:
            print(json.dumps({
                "id": row["id"],
                "group": row["stage15_probe_type"],
                "claim_families": row["claim_predicate_families"],
                "evidence_families": row["evidence_predicate_families"],
                "conflict": row["predicate_conflict_pair"],
                "notes": row["detector_notes"],
            }, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--pred-glob", default=DEFAULT_PRED_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="stage20_predicate_guard")
    parser.add_argument(
        "--detector-mode",
        choices=("oracle_probe_type", "metadata_predicate", "lexical_predicate"),
        default="lexical_predicate",
    )
    parser.add_argument(
        "--patch-mode",
        choices=("hard_override", "pseudo_logit_penalty"),
        default="hard_override",
    )
    parser.add_argument("--predicate-penalty", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(path) for path in sorted(glob.glob(args.pred_glob))]
    if not paths:
        raise FileNotFoundError(f"no prediction files matched: {args.pred_glob}")
    probe_by_id = {row["id"]: row for row in load_jsonl(args.probe)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        adjusted = apply_guard_to_payload(
            payload=payload,
            probe_by_id=probe_by_id,
            detector_mode=args.detector_mode,
            patch_mode=args.patch_mode,
            predicate_penalty=args.predicate_penalty,
        )
        seed = adjusted["metadata"].get("seed", len(outputs) + 1)
        output_path = args.output_dir / f"{args.output_prefix}_seed{seed}_{args.detector_mode}_{args.patch_mode}.json"
        output_path.write_text(json.dumps(adjusted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(adjusted)
    print("STAGE20_PREDICATE_GUARD_APPLIED")
    print(f"prediction_files_processed\t{len(paths)}")
    print(f"detector_mode\t{args.detector_mode}")
    print(f"patch_mode\t{args.patch_mode}")
    print_diagnostics(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

