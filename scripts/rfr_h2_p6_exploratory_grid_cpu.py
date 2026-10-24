"""RFR-H2-P6 frozen-grid evaluator; NO model inference by default.

Exploratory non-confirmatory fixed-law Boolean diagnostic. This is not a
recurrent-state mediation test, independent-root cohort, or unknown-gold test.
An explicit --execute --approved-commit ... call is a technical interlock only;
execution still requires a separate research authorization.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Callable

from scripts import rfr_h2_p1_temporal_serialization_contract_cpu as p1
from scripts import rfr_h2_p3_tokenizer_continuation_conformance_cpu as p3

LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
TOKEN_IDS = (16708, 32471, 11793)
ARMS = ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")
READOUTS = ("PRE", "POST", "DELAYED")
QUERIES = ("A", "B", "U")
PRE_ARM = "PRE_NO_T1"
BASELINES = (
    "ALWAYS_REFUTE", "ALWAYS_NOT_ENTITLED", "ALWAYS_SUPPORT",
    "T0_ONLY", "LATEST_ASSERTION", "IGNORE_C", "QUERY_ONLY", "AUTH_FLAG_ONLY",
)
MODEL_SHA = "1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea"
RUNTIME_SHA = "0fb3d343b2016aa4abe0c493ae143dc2193e1148bdd8792e9486ad04f5e30d71"
PINNED_FILES = {
    "scripts/rfr_h2_p1_temporal_serialization_contract_cpu.py": "be7831c13787ce079daece50620a34bd5704a5679a31d2f8bf5d502d402c49a0",
    "scripts/rfr_h2_p3_tokenizer_continuation_conformance_cpu.py": "47313a6d765727c8d5a6d18183a1129739eaf07d7444634c0b7ce4b10f27af5d",
    "reports/rfr_h2_p5_prospective_behavioral_contrasts_and_baselines_design_candidate.md": "edeaf052f2326830299c1b651482812e2137c916912bb56ef4906e58b002227d",
    "reports/rfr_h2_p6_unified_preexperiment_protocol_and_execution_contract_candidate.md": "1a567f6f0cb16270f794ce0edd24fb52ca45132a235174d29931fdef0b4c43b0",
}
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"


class ContractError(ValueError):
    pass


def need(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError("H2_P6_" + code)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_repository_files(repo: Path) -> dict[str, str]:
    observed = {}
    for name, digest in PINNED_FILES.items():
        path = repo / name
        need(path.is_file(), "SOURCE_MISSING_" + Path(name).name)
        actual = sha_file(path)
        need(actual == digest, "SOURCE_DRIFT_" + Path(name).name)
        observed[name] = actual
    return observed


def expected_label(a: bool, c: bool, u: bool, arm: str, readout: str, query: str) -> str:
    need(arm in ARMS and readout in READOUTS and query in QUERIES, "FACTOR_INVALID")
    active_a = not a if arm == "VALID_FLIP" and readout != "PRE" else a
    values = {"A": active_a, "B": active_a and c, "U": u}
    return "SUPPORT" if values[query] else "REFUTE"


def _default_oracle(world: dict[str, bool], arm: str, readout: str, query: str) -> dict:
    """The frozen D1 oracle, through P1's existing cross-oracle consistency check."""
    return p1._compare_oracle(world, arm, readout, query)


def build_grid(oracle_fn: Callable | None = None) -> list[dict]:
    """Construct ephemeral 168 IDs, not a materialized independent cohort.

    oracle_fn injection is strictly for model-free tests; normal calls use D1.
    """
    need(p1.ARMS == ARMS and p1.READOUTS == READOUTS and p1.QUERIES == QUERIES,
         "P1_FACTORS_DRIFT")
    need(p1.CLASS_LABELS == LABELS and p3.EXPECTED_LABEL_ORDER == LABELS,
         "LABEL_ORDER_DRIFT")
    need(p1.RULE_TEXT.startswith("Rule: B is true exactly when both A and C are true.\n"),
         "RULE_DRIFT")
    need(p1.DELAY_TEXT == "Neutral interval: an unrelated blank sheet was turned.\n",
         "DELAY_DRIFT")
    need(p3.REVISION == "5708daa364c50b880e7bd92eab456e0d34492ee9",
         "REVISION_DRIFT")
    need(tuple(p3.CONTINUATIONS[k] for k in LABELS) ==
         (" False", " Unknown", " True"), "CONTINUATION_DRIFT")
    need(p3.QUERY_SUFFIX_ORIGINAL == "Answer: " and
         p3.QUERY_SUFFIX_REASSIGNED == "Answer:", "SEAM_DRIFT")
    need(p1.READOUT_SPEC["evaluation_enabled"] is False and
         p1.READOUT_SPEC["candidate_continuations"] == "UNSET_NO_TOKENIZER",
         "P1_PREMATURE_PROMOTION")
    oracle = _default_oracle if oracle_fn is None else oracle_fn
    records = []
    for a, c, u in itertools.product((False, True), repeat=3):
        world = {"A": a, "C": c, "U": u}
        stratum = f"A{int(a)}C{int(c)}U{int(u)}"
        seen_pre = {}
        for readout in READOUTS:
            arms = (ARMS[0],) if readout == "PRE" else ARMS
            for arm in arms:
                rule, t0, t1, delay = p3._segments(world, arm, readout)
                need(rule == p1.RULE_TEXT and t0 == p1._t0_text(world), "SEGMENT_DRIFT")
                need(t1 == ("" if readout == "PRE" else p1._t1_text(world, arm)),
                     "T1_OR_PRE_LEAK")
                need(delay == (p1.DELAY_TEXT if readout == "DELAYED" else ""),
                     "DELAY_SURFACE_DRIFT")
                prefix = rule + t0 + t1 + delay
                for query in QUERIES:
                    original = prefix + p1._query_text(query)
                    need(original.endswith("Answer: "), "QUERY_SUFFIX_DRIFT")
                    prompt = original[:-1]
                    need(prompt.endswith("\nAnswer:"), "QUERY_SEAM_DRIFT")
                    need(prompt + p3.CONTINUATIONS["SUPPORT"] == original + "True",
                         "VISIBLE_TEXT_DRIFT")
                    gold = oracle(world, arm, readout, query)
                    predicted_gold = expected_label(a, c, u, arm, readout, query)
                    need(type(gold) is dict and gold.get("final_label") == predicted_gold and
                         gold.get("final_label_id") == LABELS.index(predicted_gold),
                         "D1_ORACLE_CONTRADICTION")
                    if readout == "PRE":
                        seen_pre[query] = prompt
                    actual_arm = PRE_ARM if readout == "PRE" else arm
                    key = f"{stratum}|{readout}|{actual_arm}|{query}"
                    records.append({
                        "case_id": key, "stratum": stratum, "a0": a, "c0": c, "u0": u,
                        "arm": actual_arm, "readout": readout, "query": query,
                        "gold": predicted_gold, "gold_id": LABELS.index(predicted_gold),
                        "prompt": prompt, "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
                    })
        need(len(seen_pre) == 3, "PRE_COVERAGE")
        for query, original in seen_pre.items():
            for arm in ARMS[1:]:
                pre = p3._segments(world, arm, "PRE")
                need("".join(pre) + p1._query_text(query)[:-1] == original,
                     "PRE_ARM_LEAKAGE")
    keys = [x["case_id"] for x in records]
    need(len(records) == 168 and len(set(keys)) == 168, "UNIQUE_168")
    need(len({x["prompt_sha256"] for x in records}) == 168, "PROMPT_COLLISION")
    need(Counter(x["gold"] for x in records) ==
         Counter({"SUPPORT": 70, "REFUTE": 98}), "GOLD_COVERAGE_DRIFT")
    need(all(x["gold"] != "NOT_ENTITLED" for x in records), "UNKNOWN_GOLD_INTRUSION")
    return records


def baseline_prediction(case: dict, baseline: str) -> str:
    need(baseline in BASELINES, "UNKNOWN_BASELINE")
    if baseline in ("ALWAYS_REFUTE", "ALWAYS_NOT_ENTITLED", "ALWAYS_SUPPORT"):
        return baseline.removeprefix("ALWAYS_")
    a, c, u = case["a0"], case["c0"], case["u0"]
    query, readout, arm = case["query"], case["readout"], case["arm"]
    if baseline == "QUERY_ONLY":
        return {"A": "REFUTE", "B": "REFUTE", "U": "SUPPORT"}[query]
    if baseline == "AUTH_FLAG_ONLY":
        return "REFUTE" if readout == "PRE" or arm == "INVALID_FLIP" else "SUPPORT"
    if baseline == "T0_ONLY":
        current_a = a
    elif baseline == "LATEST_ASSERTION":
        current_a = (not a) if (readout != "PRE" and arm in
                                ("VALID_FLIP", "INVALID_FLIP")) else a
    else:
        current_a = not a if readout != "PRE" and arm == "VALID_FLIP" else a
    vals = {"A": current_a, "B": current_a if baseline == "IGNORE_C" else current_a and c,
            "U": u}
    return "SUPPORT" if vals[query] else "REFUTE"


def baseline_rows(cases: list[dict]) -> list[dict]:
    need(len(cases) == 168 and len({x["case_id"] for x in cases}) == 168,
         "BASELINE_INPUT_COVERAGE")
    output = []
    for case in cases:
        result = {"case_id": case["case_id"], "stratum": case["stratum"],
                  "gold": case["gold"], "predictions": {}}
        for name in BASELINES:
            guess = baseline_prediction(case, name)
            result["predictions"][name] = {
                "prediction": guess, "correct": guess == case["gold"],
            }
        output.append(result)
    return output


def score_triplet(values: tuple[float, ...], *, gold: str) -> dict:
    need(gold in LABELS, "GOLD_LABEL_UNKNOWN")
    need(type(values) in (tuple, list) and len(values) == 3 and
         all(type(x) in (int, float) for x in values), "SCORE_SCHEMA")
    z = tuple(float(x) for x in values)
    if not all(math.isfinite(x) for x in z):
        return {"logits": [x if math.isfinite(x) else ("NaN" if math.isnan(x) else ("+Infinity" if x > 0 else "-Infinity")) for x in z],
                "valid": False, "reason": "NONFINITE", "prediction": None,
                "correct": False, "gold_margin": None, "candidate_softmax": None}
    maxima = max(z)
    if z.count(maxima) != 1:
        return {"logits": list(z), "valid": False, "reason": "TIED_MAX",
                "prediction": None, "correct": False, "gold_margin": None,
                "candidate_softmax": None}
    winner = LABELS[z.index(maxima)]
    gold_id = LABELS.index(gold)
    margin = z[gold_id] - max(x for i, x in enumerate(z) if i != gold_id)
    if not math.isfinite(margin):
        return {"logits": list(z), "valid": False, "reason": "MARGIN_OVERFLOW",
                "prediction": None, "correct": False, "gold_margin": None,
                "candidate_softmax": None}
    centered = [math.exp(x - maxima) for x in z]
    denom = sum(centered)
    return {"logits": list(z), "valid": True, "reason": None,
            "prediction": winner, "correct": winner == gold,
            "gold_margin": margin, "candidate_softmax": [x / denom for x in centered]}


def score_report(cases: list[dict], scores: dict[str, tuple[float, ...]]) -> dict:
    need(len(cases) == 168 and len({x["case_id"] for x in cases}) == 168,
         "CASE_COUNT")
    need(type(scores) is dict and set(scores) == {x["case_id"] for x in cases},
         "SCORE_COVERAGE")
    predictions = []
    for case in cases:
        s = score_triplet(scores[case["case_id"]], gold=case["gold"])
        predictions.append({**{k: v for k, v in case.items() if k != "prompt"}, **s})
    worlds = {}
    for key in sorted({c["stratum"] for c in cases}):
        group = [p for p in predictions if p["stratum"] == key]
        pre = [p for p in group if p["readout"] == "PRE"]
        post = [p for p in group if p["readout"] == "POST"]
        delayed = [p for p in group if p["readout"] == "DELAYED"]
        need((len(pre), len(post), len(delayed)) == (3, 9, 9), "STRATUM_SHAPE")
        worlds[key] = {
            "PRE_joint": all(p["correct"] for p in pre),
            "POST_joint": all(p["correct"] for p in post),
            "DELAYED_joint": all(p["correct"] for p in delayed),
            "CASE_joint": all(p["correct"] for p in group),
        }
    confusion = Counter((p["gold"], p["prediction"] or "INVALID") for p in predictions)
    cell_failures = [
        {k: p[k] for k in ("case_id", "stratum", "arm", "readout", "query",
                             "a0", "c0", "u0", "gold", "prediction", "reason")}
        for p in predictions if not p["correct"]
    ]
    conditional = {}
    predicate_map = {
        "A_REVISION": lambda x: x["query"] == "A" and x["readout"] != "PRE",
        "B_DEPENDENCY_C_TRUE": lambda x: x["query"] == "B" and x["c0"] and x["readout"] != "PRE",
        "B_BLOCKED_C_FALSE": lambda x: x["query"] == "B" and not x["c0"] and x["readout"] != "PRE",
        "U_RETENTION": lambda x: x["query"] == "U",
        "INVALID_RESISTANCE": lambda x: x["arm"] == "INVALID_FLIP",
        "POST_DELAYED": lambda x: x["readout"] in ("POST", "DELAYED"),
    }
    for name, pred in predicate_map.items():
        selected = [x for x in predictions if pred(x)]
        conditional[name] = {"correct": sum(x["correct"] for x in selected),
                             "total": len(selected), "all_correct": all(x["correct"] for x in selected)}
    b_rows = baseline_rows(cases)
    baseline_summary = {
        name: {"correct": sum(x["predictions"][name]["correct"] for x in b_rows),
               "total": 168} for name in BASELINES
    }
    summary = {
        "status": "EXPLORATORY_FIXED_GRID_ONLY",
        "n_unique_contexts": 168, "n_logical_strata": 8,
        "independently_sampled_roots": 0,
        "gold_coverage": dict(Counter(p["gold"] for p in predictions)),
        "prediction_counts": dict(Counter(p["prediction"] or "INVALID" for p in predictions)),
        "confusion": [{"gold": a, "prediction": b, "count": count}
                      for (a, b), count in sorted(confusion.items())],
        "correct": sum(p["correct"] for p in predictions),
        "CASE_joint_count": sum(x["CASE_joint"] for x in worlds.values()),
        "CASE_joint_denominator": 8, "strata": worlds,
        "conditional_diagnostics": conditional,
        "baseline_summary": baseline_summary,
        "failures": cell_failures,
        "three_way_gold_coverage": "ABSENT",
        "source_alias_confound": "UNRESOLVED",
        "candidate_prior_control": "NOT_DESIGNED",
        "native_state_causal_mediation": "NOT_ESTABLISHED",
        "phase3b": "BLOCKED_UNCHANGED",
    }
    return {"predictions": predictions, "baseline_predictions": b_rows,
            "summary": summary}


def git_output(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8")
    return proc.stdout.strip()


def authentication(repo: Path, snapshot: Path, approved_commit: str) -> dict:
    need(re.fullmatch(r"[0-9a-f]{40}", approved_commit) is not None, "COMMIT_SCHEMA")
    need(git_output(repo, "rev-parse", "--show-toplevel").replace("\\", "/").casefold()
         == str(repo.resolve()).replace("\\", "/").casefold(), "REPO_ROOT")
    need(git_output(repo, "branch", "--show-current") == EXPECTED_BRANCH, "BRANCH")
    head = git_output(repo, "rev-parse", "HEAD")
    need(head == approved_commit, "HEAD_MISMATCH")
    need(not git_output(repo, "status", "--porcelain=v1", "--untracked-files=all"), "DIRTY")
    need(git_output(repo, "rev-parse", "HEAD") == head, "HEAD_RACE")
    tracked = authenticate_repository_files(repo)
    need(snapshot.is_dir() and snapshot.name == p3.REVISION, "SNAPSHOT_REVISION")
    config = p3.authenticate_snapshot(snapshot)
    weight = snapshot / "model.safetensors"
    need(weight.is_file() and weight.stat().st_size == 516567560, "WEIGHT_MISSING")
    need(sha_file(weight) == MODEL_SHA, "WEIGHT_SHA")
    import importlib.metadata as md
    import sys
    need(sys.version_info[:3] == (3, 13, 2), "PYTHON_DRIFT")
    versions = {x: md.version(x) for x in
                ("torch", "transformers", "tokenizers", "safetensors")}
    need(versions == {"torch": "2.12.1", "transformers": "4.45.0",
                      "tokenizers": "0.20.3", "safetensors": "0.8.0"}, "PACKAGE_DRIFT")
    model_source = Path(md.distribution("transformers").locate_file(
        "transformers/models/mamba/modeling_mamba.py"))
    need(model_source.is_file() and sha_file(model_source) == RUNTIME_SHA,
         "MAMBA_SOURCE_DRIFT")
    return {"head": head, "branch": EXPECTED_BRANCH, "source_sha256": tracked,
            "model_sha256": MODEL_SHA, "runtime_source_sha256": RUNTIME_SHA,
            "tokenizer_hashes": dict(p3.SNAPSHOT_FILES_SHA256),
            "revision": p3.REVISION, "versions": versions,
            "config_vocab_size": config["vocab_size"]}


def tokenize_for_run(cases: list[dict], snapshot: Path) -> dict[str, list[int]]:
    from transformers import AutoTokenizer
    verified = p3.audit_snapshot(snapshot)
    need(verified["result"] == p3.PASS_STATUS and
         verified["candidate_token_ids"] == dict(zip(LABELS, TOKEN_IDS)) and
         verified["prompt_token_length_range"][-1] <= 150 and
         verified["complete_token_length_range"][-1] <= 151,
         "P3_CONFORMANCE")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    need(type(tokenizer).__name__ == "GPTNeoXTokenizerFast", "TOKENIZER_TYPE")
    outputs = {}
    for case in cases:
        ids = tokenizer.encode(case["prompt"], add_special_tokens=False,
                               truncation=False)
        need(0 < len(ids) <= 150, "INPUT_LENGTH")
        need(all(type(i) is int and 0 <= i < 50280 for i in ids), "TOKEN_ID_RANGE")
        for label, token_id in zip(LABELS, TOKEN_IDS):
            full = tokenizer.encode(case["prompt"] + p3.CONTINUATIONS[label],
                                    add_special_tokens=False, truncation=False)
            need(full == ids + [token_id], "TOKEN_BOUNDARY_OR_ID_DRIFT")
        outputs[case["case_id"]] = ids
    need(len(outputs) == 168, "TOKENIZED_CASE_COUNT")
    return outputs


def evaluate_on_cpu(cases: list[dict], tokenized: dict[str, list[int]], snapshot: Path) -> dict:
    import torch
    from transformers import MambaForCausalLM
    torch.set_num_threads(4)
    with torch.inference_mode():
        model, info = MambaForCausalLM.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, output_loading_info=True)
        need(all(not info.get(k) for k in ("missing_keys", "unexpected_keys",
                                            "mismatched_keys", "error_msgs")),
             "CHECKPOINT_LOAD_MISMATCH")
        model = model.to("cpu")
        model.eval()
        need(model.get_input_embeddings().weight.data_ptr() ==
             model.get_output_embeddings().weight.data_ptr(), "UNTIED_LM_HEAD")
        need(len(model.backbone.layers) == 24, "LAYER_COUNT")
        scores = {}
        for case in cases:
            ids = tokenized[case["case_id"]]
            batch = torch.tensor([ids], dtype=torch.long, device="cpu")
            output = model(input_ids=batch, use_cache=False, labels=None,
                           output_hidden_states=False, return_dict=True)
            logits = output.logits
            need(tuple(logits.shape) == (1, len(ids), 50280), "FORWARD_SHAPE")
            need(output.cache_params is None, "CACHE_PRESENT")
            triplet = tuple(float(logits[0, -1, i].item()) for i in TOKEN_IDS)
            scores[case["case_id"]] = triplet
        del model
    return score_report(cases, scores)


def json_bytes(item: object) -> bytes:
    return (json.dumps(item, sort_keys=True, ensure_ascii=False,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(items: list[dict]) -> bytes:
    return b"".join(json.dumps(item, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
                    for item in items)


def save_evidence(output_root: Path, run_name: str, manifest: dict, result: dict) -> Path:
    need(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_name)), "RUN_NAME")
    root = Path(output_root)
    final = root / run_name
    partial = root / ("." + run_name + ".partial")
    need(not final.exists() and not partial.exists(), "RUN_DIRECTORY_EXISTS")
    summary = result["summary"]
    report_lines = [
        "# RFR-H2 fixed-grid diagnostic (EXPLORATORY ONLY)", "",
        f"Run: `{run_name}`; commit: `{manifest['head']}`.", "",
        f"Unique contexts: {summary['n_unique_contexts']}; "
        f"CASE joint: {summary['CASE_joint_count']}/8.", "",
        "## Eight linked Boolean strata", "",
        "| Stratum | PRE joint | POST joint | DELAYED joint | CASE joint |",
        "|---|---|---|---|---|",
    ]
    for key, values in sorted(summary["strata"].items()):
        report_lines.append("| " + " | ".join([key] +
                            [str(values[x]) for x in
                             ("PRE_joint", "POST_joint", "DELAYED_joint", "CASE_joint")]) + " |")
    report_lines += ["", "## Conditional contrasts", "",
                     "| Condition | Correct | Total | All correct |",
                     "|---|---:|---:|---|"]
    for key, values in sorted(summary["conditional_diagnostics"].items()):
        report_lines.append(f"| {key} | {values['correct']} | {values['total']} | "
                            f"{values['all_correct']} |")
    report_lines += ["", "## Fixed deterministic baselines", "",
                     "| Baseline | Oracle matches / 168 |",
                     "|---|---:|"]
    for key, values in sorted(summary["baseline_summary"].items()):
        report_lines.append(f"| {key} | {values['correct']} / {values['total']} |")
    report_lines += ["", "## Error and limitation summary", "",
                     f"Correct contexts: {summary['correct']}/168; "
                     f"failed contexts: {len(summary['failures'])}.", "",
                     "No independent roots, confidence intervals or population claim.",
                     "No NOT_ENTITLED gold; source alias and authentication confounded.",
                     "No independent token-prior control or H3 state mediation.",
                     "Full condition/label-level errors, raw logits and candidate-only "
                     "softmax are in predictions.jsonl and summary.json.", ""]
    files = {
        "manifest.json": json_bytes(manifest),
        "predictions.jsonl": jsonl_bytes(result["predictions"]),
        "baseline_predictions.jsonl": jsonl_bytes(result["baseline_predictions"]),
        "summary.json": json_bytes(result["summary"]),
        "report.md": ("\n".join(report_lines)).encode("utf-8"),
    }
    checksums = "".join(f"{sha_bytes(content)}  {name}\n" for name, content in sorted(files.items()))
    root.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        for name, content in files.items():
            (partial / name).write_bytes(content)
        (partial / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8", newline="\n")
        for name, content in files.items():
            need(sha_file(partial / name) == sha_bytes(content), "ARTIFACT_HASH")
        need(set(p.name for p in partial.iterdir()) == {*files, "SHA256SUMS.txt"},
             "ARTIFACT_SET")
        need(not final.exists(), "OVERWRITE_BLOCKED")
        partial.rename(final)
    except Exception:
        # Keep partial directory for forensic inspection; never mark it complete.
        raise
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Only with separate execution authority")
    parser.add_argument("--approved-commit", default=None)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    try:
        # No tokenizer, torch, transformer, checkpoint, or filesystem writes on default path.
        cases = build_grid()
        baselines = baseline_rows(cases)
        counts = {name: sum(x["predictions"][name]["correct"] for x in baselines)
                  for name in BASELINES}
        if not args.execute:
            need(all(x is None for x in (args.approved_commit, args.snapshot,
                                           args.run_name, args.output_root)),
                 "EXECUTION_OPTIONS_WITHOUT_EXECUTE")
            print(json.dumps({"status": "H2_P6_CONTRACT_ONLY_NO_MODEL",
                              "unique_contexts": len(cases), "nominal_contexts": 216,
                              "deduplicated_PRE": 48, "logical_strata": 8,
                              "independent_roots": 0, "baseline_oracle_matches": counts,
                              "model_loaded": False, "model_evaluation": False,
                              "phase3b": "BLOCKED_UNCHANGED"}, indent=2))
            return 0
        need(args.approved_commit is not None and args.snapshot is not None and
             args.run_name is not None and args.output_root is not None,
             "EXECUTION_ARGUMENTS")
        # Hard fail before imports involving model or tokenizer.
        auth = authentication(args.repo_root.resolve(), args.snapshot.resolve(), args.approved_commit)
        tokenized = tokenize_for_run(cases, args.snapshot.resolve())
        output = evaluate_on_cpu(cases, tokenized, args.snapshot.resolve())
        for record in output["predictions"]:
            record["token_count"] = len(tokenized[record["case_id"]])
        manifest = {**auth, "run_name": args.run_name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": "EXPLORATORY_NONCONFIRMATORY", "candidate_token_ids":
                    dict(zip(LABELS, TOKEN_IDS)), "use_cache": False,
                    "device": "cpu", "dtype": "float32", "model_evaluation": True,
                    "independent_roots": 0, "phase3b": "BLOCKED_UNCHANGED"}
        location = save_evidence(args.output_root, args.run_name, manifest, output)
        print(json.dumps({"status": "EXPLORATORY_ARTIFACT_WRITTEN_NOT_VALIDATED",
                          "artifact_path": str(location),
                          "case_joint_count": output["summary"]["CASE_joint_count"],
                          "phase3b": "BLOCKED_UNCHANGED"}, indent=2))
        return 0
    except (ContractError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print("H2_P6_RESULT=BLOCKED")
        print("REASON=" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
