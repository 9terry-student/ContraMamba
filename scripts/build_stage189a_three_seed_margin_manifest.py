#!/usr/bin/env python3
"""Build six Stage189-B paired-run manifests; never execute training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any, Iterable

DATA_REL = "data/controlled_v5_v3_without_time_swap.jsonl"
DATA_SHA = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
SIDECAR_REL = (
    "reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/"
    "stage185a_controlled_train_integrity_sidecar.jsonl"
)
SIDECAR_SHA = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
HELPER_REL = "scripts/export_external_scalars_from_checkpoint.py"
STAGE189C_REL = "scripts/export_stage189c_posthoc_margin_reference.py"
SEEDS = (174, 175, 176)
ARMS = ("baseline", "intervention")
READY = "STAGE189A_THREE_SEED_MARGIN_REPLICATION_AND_POSTHOC_REFERENCE_SPEC_READY"
BLOCKED = "STAGE189A_THREE_SEED_MARGIN_REPLICATION_SPEC_BLOCKED"
NEXT = "STAGE189B_THREE_SEED_PAIRED_TRAINING"
CLOSURE = "STAGE188B_SINGLE_SEED_MARGIN_MIXED_NO_REPLICATION_YET"
ALLOWED_ARM_DIFFS = {
    "compatible_positive_margin_weight",
    "controlled_integrity_sidecar_path",
    "expected_integrity_sidecar_semantic_sha256",
    "output_json",
    "output_predictions_json",
    "stage115_clean_dev_scalar_output_jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage188b-closure", type=Path, required=True)
    parser.add_argument("--stage185a-dir", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--current-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", default=DATA_SHA)
    parser.add_argument("--expected-sidecar-semantic-sha256", default=SIDECAR_SHA)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sidecar(path: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"sidecar line {number} is not an object")
            rows.append(value)
    semantic = [{key: row[key] for key in sorted(row) if key != "created_at"} for row in rows]
    encoded = json.dumps(semantic, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), rows


def sidecar_row_id(row: dict[str, Any]) -> str | None:
    value = row.get("id", row.get("row_id"))
    return str(value) if value is not None else None


def is_train_compatible(row: dict[str, Any]) -> bool:
    value = row.get("frame_compatible_label")
    return row.get("split") == "train" and type(value) is int and value == 1


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def parser_options(repo: Path, trainer: Path) -> tuple[set[str], list[str]]:
    sources = [trainer, repo / "scripts" / "train_controlled_v5.py"]
    pattern = re.compile(
        r"""add_argument\(\s*["'](--[a-z0-9-]+)["']""",
        re.IGNORECASE,
    )
    found: set[str] = set()
    evidence: list[str] = []
    for source in sources:
        if not source.is_file():
            continue
        options = set(pattern.findall(source.read_text(encoding="utf-8")))
        found.update(options)
        evidence.append(f"{source}:{len(options)}")
    return found, evidence


def set_option(argv: list[str], option: str, value: Any | None, flag: bool = False) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            index += 1
            if index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        result.append(token)
        index += 1
    if flag and value:
        result.append(option)
    elif value is not None:
        result.extend([option, str(value)])
    return result


def argv_map(argv: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional token {token!r}")
        key = token[2:].replace("-", "_")
        if index + 1 == len(argv) or argv[index + 1].startswith("--"):
            result[key] = True
            index += 1
        else:
            result[key] = argv[index + 1]
            index += 2
    return result


def common_argv(seed: int) -> list[str]:
    return [
        "--data", DATA_REL,
        "--architecture", "v6b_minimal",
        "--backbone", "mamba",
        "--model-name", "state-spaces/mamba-130m-hf",
        "--device", "cuda",
        "--seed", str(seed),
        "--epochs", "20",
        "--select-metric", "final_macro_f1",
        "--flag-source", "controlled_heuristic",
        "--stage174c-clean-pairwise-mode", "off",
        "--stage174c-clean-pairwise-weight", "0.0",
        "--stage175b-support-anchor-mode", "off",
        "--stage175b-support-anchor-weight", "0.0",
        "--stage177c-frame-pairwise-mode", "off",
        "--stage177c-frame-pairwise-weight", "0.0",
        "--compatible-positive-margin-logit", "0.0",
        "--save-selected-checkpoint",
        "--selected-checkpoint-filename", "selected_checkpoint.pt",
    ]


def run_argv(seed: int, arm: str, output: Path) -> tuple[list[str], Path]:
    run_dir = output / f"stage189b_seed{seed}_{arm}"
    argv = common_argv(seed)
    settings = [
        ("--compatible-positive-margin-weight", 0.0 if arm == "baseline" else 0.05),
        ("--controlled-integrity-sidecar-path", None if arm == "baseline" else SIDECAR_REL),
        ("--expected-integrity-sidecar-semantic-sha256", None if arm == "baseline" else SIDECAR_SHA),
        ("--output-json", run_dir / "training_report.json"),
        ("--output-predictions-json", run_dir / "clean_dev_predictions.json"),
        ("--stage115-clean-dev-scalar-output-jsonl", run_dir / "clean_dev_scalars.jsonl"),
    ]
    for option, value in settings:
        argv = set_option(argv, option, value)
    return argv, run_dir


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trainer = args.trainer_source.resolve()
    helper = (repo / HELPER_REL).resolve()
    stage189c_source = (repo / STAGE189C_REL).resolve()
    closure_path = args.stage188b_closure.resolve()
    sidecar = args.stage185a_dir.resolve() / "stage185a_controlled_train_integrity_sidecar.jsonl"
    dataset = repo / DATA_REL
    blockers: list[str] = []
    gates: list[dict[str, Any]] = []

    def gate(name: str, required: Any, observed: Any, passed: bool, reason: str) -> None:
        gates.append({"gate": name, "required": json.dumps(required, sort_keys=True),
                      "observed": json.dumps(observed, sort_keys=True), "passed": passed,
                      "blocking_reason": "" if passed else reason})
        if not passed:
            blockers.append(f"{name}: {reason}")

    try:
        closure = read_json(closure_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        closure = {}
        blockers.append(f"closure unreadable: {exc}")
    gate("stage188b_closure", CLOSURE, closure.get("decision"), closure.get("decision") == CLOSURE,
         "Stage188-B mixed closure mismatch")
    gate("stage188b_no_blockers", [], closure.get("blocking_reasons"), closure.get("blocking_reasons") == [],
         "Stage188-B closure contains blockers")
    gate("stage188b_single_seed", True, closure.get("single_seed_only"), closure.get("single_seed_only") is True,
         "closure must remain single-seed evidence")

    observed_data_sha = file_sha(dataset) if dataset.is_file() else None
    trainer_sha = file_sha(trainer) if trainer.is_file() else None
    try:
        sidecar_sha, sidecar_rows = semantic_sidecar(sidecar)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sidecar_sha, sidecar_rows = None, []
        blockers.append(f"sidecar unreadable: {exc}")
    sidecar_ids = [sidecar_row_id(row) for row in sidecar_rows]
    train_rows = [row for row in sidecar_rows if row.get("split") == "train"]
    dev_rows = [row for row in sidecar_rows if row.get("split") == "dev"]
    train_compatible_rows = [row for row in sidecar_rows if is_train_compatible(row)]
    train_incompatible_rows = [row for row in train_rows if not is_train_compatible(row)]
    status_counts = {status: sum(row.get("integrity_status") == status for row in train_compatible_rows)
                     for status in ("ELIGIBLE", "INELIGIBLE", "UNRESOLVED")}
    eligible = [row for row in train_compatible_rows if row.get("integrity_status") == "ELIGIBLE"]
    positive_margin_rows = [
        row for row in sidecar_rows if row.get("eligible_for_positive_margin") is True
    ]
    full_sidecar_topology = {
        "total_rows": len(sidecar_rows),
        "unique_row_ids": len(set(sidecar_ids)) if None not in sidecar_ids else len(set(sidecar_ids)) - 1,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
    }
    topology = {"train_rows": len(train_compatible_rows), "eligible_rows": len(eligible),
                "eligible_pairs": len({row.get("pair_id") for row in eligible}),
                "eligible_families": len({row.get("family_contract_id") for row in eligible}),
                "incompatible_train_rows": len(train_incompatible_rows),
                **{key.lower(): value for key, value in status_counts.items()}}
    gate("dataset_sha256", args.expected_dataset_sha256, observed_data_sha,
         observed_data_sha == args.expected_dataset_sha256, "dataset identity mismatch")
    gate("sidecar_semantic_sha256", args.expected_sidecar_semantic_sha256, sidecar_sha,
         sidecar_sha == args.expected_sidecar_semantic_sha256, "sidecar identity mismatch")
    gate("full_sidecar_topology",
         {"total_rows": 3600, "unique_row_ids": 3600, "train_rows": 2880, "dev_rows": 720},
         full_sidecar_topology,
         full_sidecar_topology == {
             "total_rows": 3600, "unique_row_ids": 3600, "train_rows": 2880, "dev_rows": 720
         } and None not in sidecar_ids,
         "full Stage185-A sidecar topology mismatch")
    gate("sidecar_train_topology",
         {"train_rows": 1440, "eligible_rows": 605, "eligible_pairs": 121, "eligible_families": 5,
          "incompatible_train_rows": 1440,
          "eligible": 605, "ineligible": 716, "unresolved": 119}, topology,
         topology == {"train_rows": 1440, "eligible_rows": 605, "eligible_pairs": 121,
                      "eligible_families": 5, "incompatible_train_rows": 1440,
                      "eligible": 605, "ineligible": 716, "unresolved": 119},
         "authoritative train-compatible integrity topology mismatch")
    compatible_eligible_ids = {sidecar_row_id(row) for row in eligible}
    positive_margin_ids = {sidecar_row_id(row) for row in positive_margin_rows}
    gate("positive_margin_eligibility_identity",
         {"row_count": 605, "same_as_train_compatible_eligible_ids": True},
         {"row_count": len(positive_margin_rows),
          "same_as_train_compatible_eligible_ids": positive_margin_ids == compatible_eligible_ids},
         len(positive_margin_rows) == 605
         and None not in positive_margin_ids
         and positive_margin_ids == compatible_eligible_ids,
         "positive-margin eligibility IDs differ from train-compatible ELIGIBLE IDs")
    gate("current_git_commit", "non-empty", args.current_git_commit.strip(),
         bool(args.current_git_commit.strip()), "current commit is empty")
    gate("trainer_source", "existing file", trainer_sha, trainer_sha is not None, "trainer source missing")
    helper_sha = file_sha(helper) if helper.is_file() else None
    helper_text = helper.read_text(encoding="utf-8") if helper.is_file() else ""
    trainer_text = trainer.read_text(encoding="utf-8") if trainer.is_file() else ""
    stage189c_text = stage189c_source.read_text(encoding="utf-8") if stage189c_source.is_file() else ""
    required_helper_functions = ("load_checkpoint", "merged_runner_args", "build_eval_model", "export_rows")
    missing_helper_functions = [
        name for name in required_helper_functions
        if re.search(rf"^def\s+{re.escape(name)}\s*\(", helper_text, re.MULTILINE) is None
    ]
    gate("checkpoint_helper_functions", [], missing_helper_functions, not missing_helper_functions,
         "checkpoint helper required functions are missing")
    gate("checkpoint_helper_sha256", "current helper bytes", helper_sha, helper_sha is not None,
         "checkpoint helper is missing")
    checkpoint_contract = {
        "save_selected_checkpoint_store_true": bool(re.search(
            r'add_argument\(\s*["\']--save-selected-checkpoint["\'][\s\S]{0,180}?action\s*=\s*["\']store_true["\']',
            trainer_text)),
        "selected_checkpoint_filename_takes_value": bool(re.search(
            r'add_argument\(\s*["\']--selected-checkpoint-filename["\'][\s\S]{0,180}?type\s*=\s*str',
            trainer_text)),
        "selected_state_source": "_ood_best_state" in trainer_text
            and "model.load_state_dict({key: value.to(device) for key, value in _ood_best_state.items()})" in trainer_text,
        "selected_epoch_source": '"selected_epoch": _stage174a_selected_epoch' in trainer_text
            and "_stage174a_selected_epoch = _ood_best_epoch" in trainer_text,
        "selected_not_final": '"model_state_dict": _stage176a0_cpu_state_dict(selected_state_dict)' in trainer_text
            and "selected_state_dict=_ood_best_state" in trainer_text,
        "selected_flag": '"checkpoint_is_selected_clean_dev_state": True' in trainer_text,
        "provenance_run_directory": "_stage174a_run_dir / args.selected_checkpoint_filename" in trainer_text,
        "provenance_checkpoint_path_sha": '_stage176a0_saved_checkpoint["sha256"]' in trainer_text
            and '"selected_checkpoint": _stage174a_selected_checkpoint' in trainer_text,
        "reload_verification": "loaded = torch.load(checkpoint_path, map_location=" in trainer_text
            and 'loaded_metadata.get("selected_epoch") != selected_epoch' in trainer_text,
        "stage189_metadata": all(token in trainer_text for token in (
            '"selection_metric_name": _stage189_selection_metric_key',
            '"selection_metric_value": float(_stage189_selection_metric_value)',
            '"compatible_positive_margin":',
            '"trainer_sha256":',
            '"source_git_commit":',
            '"training_args":',
        )),
    }
    gate("selected_checkpoint_source_contract", {key: True for key in checkpoint_contract},
         checkpoint_contract, all(checkpoint_contract.values()),
         "selected-checkpoint persistence source contract is not statically proven")
    direct_loop_contract = all(token in stage189c_text for token in (
        'output["frame_logit"]', 'output["frame_prob"]', ".detach().cpu().reshape(-1)",
        "local_direct_loop_contract", "--expected-checkpoint-helper-sha256",
    ))
    gate("stage189c_direct_loop_source_contract", True, direct_loop_contract, direct_loop_contract,
         "Stage189-C direct tensor export contract is not statically proven")

    available, option_evidence = parser_options(repo, trainer)
    manifests: dict[tuple[int, str], dict[str, Any]] = {}
    matrix: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    all_emitted: set[str] = set()
    for seed in SEEDS:
        arm_maps: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            argv, run_dir = run_argv(seed, arm, output)
            all_emitted.update(token for token in argv if token.startswith("--"))
            mapping = argv_map(argv)
            arm_maps[arm] = mapping
            checkpoint = run_dir / "selected_checkpoint.pt"
            manifest = {
                "stage": "Stage189-A", "seed": seed, "arm": arm, "runnable": False,
                "fresh_training_required": True, "stage188_checkpoint_reuse_forbidden": True,
                "current_git_commit": args.current_git_commit.strip(),
                "trainer_path": str(trainer), "trainer_sha256": trainer_sha,
                "checkpoint_helper_path": str(helper), "checkpoint_helper_sha256": helper_sha,
                "dataset_path": DATA_REL, "dataset_sha256": observed_data_sha,
                "sidecar_semantic_sha256": sidecar_sha if arm == "intervention" else None,
                "baseline_sidecar_access_during_training": False if arm == "baseline" else None,
                "argv": argv, "parsed_argv_contract": mapping,
                "run_directory": str(run_dir), "selected_checkpoint_path": str(checkpoint),
                "posthoc_reference_required": True,
            }
            manifests[(seed, arm)] = manifest
            matrix.append({"seed": seed, "arm": arm, "margin_weight": mapping.get("compatible_positive_margin_weight"),
                           "margin_logit": mapping.get("compatible_positive_margin_logit"),
                           "sidecar_path": mapping.get("controlled_integrity_sidecar_path"),
                           "run_directory": str(run_dir), "selected_checkpoint": str(checkpoint),
                           "training_status": "not_run_manifest_only"})
        left, right = arm_maps["baseline"], arm_maps["intervention"]
        for field in sorted(set(left) | set(right)):
            if left.get(field) == right.get(field):
                continue
            allowed = field in ALLOWED_ARM_DIFFS
            difference_rows.append({"seed": seed, "field": field,
                                    "baseline": json.dumps(left.get(field)),
                                    "intervention": json.dumps(right.get(field)),
                                    "allowed": allowed,
                                    "reason": "precommitted arm/output difference" if allowed else "forbidden difference"})
            if not allowed:
                blockers.append(f"seed {seed} forbidden arm argv difference: {field}")
        baseline_isolated = (
            left.get("compatible_positive_margin_weight") == "0.0"
            and "controlled_integrity_sidecar_path" not in left
            and "expected_integrity_sidecar_semantic_sha256" not in left
        )
        gate(f"seed{seed}_baseline_sidecar_isolation", True, baseline_isolated, baseline_isolated,
             "baseline argv includes sidecar access or nonzero margin")
        same_common = all(left.get(key) == right.get(key) for key in set(left) | set(right)
                          if key not in ALLOWED_ARM_DIFFS)
        gate(f"seed{seed}_paired_common_argv", True, same_common, same_common,
             "paired common argv mismatch")

    missing_options = sorted(all_emitted - available)
    gate("current_parser_options", [], missing_options, not missing_options,
         "one or more emitted options are absent from the statically inspected parser")
    gate("six_run_matrix", 6, len(matrix), len(matrix) == 6, "six manifests were not constructed")
    decision = READY if not blockers else BLOCKED
    runnable = decision == READY
    for manifest in manifests.values():
        manifest["runnable"] = runnable
        manifest["planned_argv"] = list(manifest["argv"])
        if not runnable:
            manifest["argv"] = []
        manifest["command_preview"] = (
            shlex.join(["python", str(trainer), *manifest["argv"]]) if runnable else None
        )
    for (seed, arm), manifest in manifests.items():
        write_json(output / f"stage189a_seed{seed}_{arm}_manifest.json", manifest)

    identity_rows = [
        {"artifact": "dataset", "path": str(dataset), "expected_identity": args.expected_dataset_sha256,
         "observed_identity": observed_data_sha, "passed": observed_data_sha == args.expected_dataset_sha256},
        {"artifact": "sidecar", "path": str(sidecar), "expected_identity": args.expected_sidecar_semantic_sha256,
         "observed_identity": sidecar_sha, "passed": sidecar_sha == args.expected_sidecar_semantic_sha256},
        {"artifact": "trainer", "path": str(trainer), "expected_identity": "current bytes",
         "observed_identity": trainer_sha, "passed": trainer_sha is not None},
        {"artifact": "checkpoint_helper", "path": str(helper), "expected_identity": "current bytes",
         "observed_identity": helper_sha, "passed": helper_sha is not None and not missing_helper_functions},
        {"artifact": "git_commit", "path": "caller supplied", "expected_identity": "non-empty",
         "observed_identity": args.current_git_commit.strip(), "passed": bool(args.current_git_commit.strip())},
    ]
    report = {
        "stage": "Stage189-A", "decision": decision, "authorized_next": NEXT if runnable else None,
        "training_performed": False, "checkpoint_evaluation_performed": False,
        "manifest_count": len(manifests), "seeds": list(SEEDS), "arms": list(ARMS),
        "blocking_reasons": blockers, "trainer_sha256": trainer_sha,
        "checkpoint_helper_path": str(helper), "checkpoint_helper_sha256": helper_sha,
        "checkpoint_helper_required_functions": list(required_helper_functions),
        "selected_checkpoint_source_contract": checkpoint_contract,
        "stage189c_local_direct_loop_contract": direct_loop_contract,
        "current_git_commit": args.current_git_commit.strip(),
        "dataset_sha256": observed_data_sha, "sidecar_semantic_sha256": sidecar_sha,
        "full_sidecar_topology": full_sidecar_topology,
        "train_compatible_topology": topology,
        "sidecar_train_topology": topology, "parser_static_evidence": option_evidence,
        "missing_emitted_options": missing_options,
        "baseline_sidecar_isolation": "weight zero; sidecar path and expected SHA options absent",
        "selected_checkpoint_contract": "opt-in saved internally selected best clean-dev state",
        "posthoc_selection_policy": "evaluation-only; never used for checkpoint selection",
    }
    write_json(output / "stage189a_manifest_report.json", report)
    write_csv(output / "authoritative_input_identity.csv",
              ["artifact", "path", "expected_identity", "observed_identity", "passed"], identity_rows)
    write_csv(output / "paired_argv_difference_audit.csv",
              ["seed", "field", "baseline", "intervention", "allowed", "reason"], difference_rows)
    write_csv(output / "six_run_matrix.csv",
              ["seed", "arm", "margin_weight", "margin_logit", "sidecar_path", "run_directory",
               "selected_checkpoint", "training_status"], matrix)
    write_csv(output / "stage189b_gate.csv",
              ["gate", "required", "observed", "passed", "blocking_reason"], gates)
    markdown = f"""# Stage189-A six-run manifest\n\n**Decision:** `{decision}`\n\n- Seeds: 174, 175, 176\n- Runs: six paired fresh-training manifests\n- Training performed: no\n- Selected checkpoint: opt-in internally selected clean-dev state\n- Baseline sidecar access: prohibited and argv-isolated\n- Authorized next: `{NEXT if runnable else None}`\n\n## Blocking reasons\n\n{chr(10).join('- ' + item for item in blockers) if blockers else '- None.'}\n"""
    (output / "stage189a_manifest_report.md").write_text(markdown, encoding="utf-8")
    return 0 if runnable else 2


if __name__ == "__main__":
    raise SystemExit(main())
