"""Prospective 16-item P/Q Boolean readout eligibility control, CPU-only.

This is NOT a new RFR experiment, model calibration, prompt sweep, or H3 test.
The default path is model-/tokenizer-free and writes nothing. --execute is a
technical interlock; it is NOT permission to conduct a scientific model run.
The new prompt surface is frozen by the committed script, not selected by scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from scripts import rfr_h2_p3_tokenizer_continuation_conformance_cpu as p3
from scripts import rfr_h2_p6_exploratory_grid_cpu as p6

MODEL_ID = p3.MODEL_ID
REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
TOKEN_IDS = (16708, 32471, 11793)
FACTS_HEADER = "Facts:\n"
FACT_TEMPLATE = "{key} is {value}.\n"
QUESTION_TEMPLATE = "Question: Is {query} true?\nAnswer:"
RECORD_ORDERS = (("P", "Q"), ("Q", "P"))
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
SCIENTIFIC_STATUS = "PROSPECTIVE_BASIC_READOUT_ELIGIBILITY_ONLY"


class EligibilityError(ValueError):
    pass


def need(condition: bool, code: str) -> None:
    if not condition:
        raise EligibilityError("H2_BASIC_" + code)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_template() -> dict[str, str]:
    return {"facts_header": FACTS_HEADER, "fact_template": FACT_TEMPLATE,
            "question_template": QUESTION_TEMPLATE,
            "record_orders": "PQ/QP"}


def make_prompt(p: bool, q: bool, order: tuple[str, str], query: str) -> str:
    need(type(p) is bool and type(q) is bool, "STRICT_BOOLEAN")
    need(order in RECORD_ORDERS and query in ("P", "Q"), "FACTOR_SCHEMA")
    values = {"P": p, "Q": q}
    return (FACTS_HEADER + "".join(FACT_TEMPLATE.format(
        key=k, value="TRUE" if values[k] else "FALSE") for k in order)
        + QUESTION_TEMPLATE.format(query=query))


def build_grid() -> list[dict[str, Any]]:
    need(p3.REVISION == REVISION and
         tuple(p3.CONTINUATIONS[k] for k in LABELS) ==
         (" False", " Unknown", " True") and
         p3.EXPECTED_LABEL_ORDER == LABELS and
         p6.TOKEN_IDS == TOKEN_IDS and
         p6.LABELS == LABELS, "FROZEN_READOUT_CHANGED")
    # The exact one-template lexical contract. No alternative phrasing allowed.
    need(source_template() == {
        "facts_header": "Facts:\n", "fact_template": "{key} is {value}.\n",
        "question_template": "Question: Is {query} true?\nAnswer:",
        "record_orders": "PQ/QP"}, "TEMPLATE_CHANGED")
    rows = []
    for p, q in itertools.product((False, True), repeat=2):
        for order in RECORD_ORDERS:
            for query in ("P", "Q"):
                prompt = make_prompt(p, q, order, query)
                gold = "SUPPORT" if {"P": p, "Q": q}[query] else "REFUTE"
                key = f"P{int(p)}Q{int(q)}|ORDER_{''.join(order)}|QUERY_{query}"
                rows.append({
                    "case_id": key, "p": p, "q": q,
                    "order": "".join(order), "query": query,
                    "gold": gold, "gold_id": LABELS.index(gold),
                    "prompt": prompt, "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
                })
    need(len(rows) == 16 and len({x["case_id"] for x in rows}) == 16 and
         len({x["prompt_sha256"] for x in rows}) == 16, "EXACT_16")
    need(Counter(x["gold"] for x in rows) == Counter({"SUPPORT": 8, "REFUTE": 8}),
         "GOLD_BALANCE")
    return rows


def parse_scores(cases: list[dict], raw: dict[str, tuple[float, float, float]]) -> dict:
    need(len(cases) == 16 and type(raw) is dict and
         set(raw) == {x["case_id"] for x in cases}, "SCORES_COVERAGE")
    predictions = []
    for case in cases:
        scored = p6.score_triplet(raw[case["case_id"]], gold=case["gold"])
        predictions.append({**{k: v for k, v in case.items() if k != "prompt"}, **scored})
    by_key = {(x["p"], x["q"], x["order"], x["query"]): x for x in predictions}
    opposite_pairs = []
    for p, q in ((False, True), (True, False)):
        for order in ("PQ", "QP"):
            first, second = (by_key[(p, q, order, target)] for target in ("P", "Q"))
            opposite_pairs.append({"p": p, "q": q, "order": order,
                                   "both_correct": first["correct"] and second["correct"],
                                   "predictions_differ": first["prediction"] != second["prediction"]})
    order_pairs = []
    for p, q in itertools.product((False, True), repeat=2):
        for query in ("P", "Q"):
            a, b = (by_key[(p, q, order, query)] for order in ("PQ", "QP"))
            order_pairs.append({"p": p, "q": q, "query": query,
                                "both_correct": a["correct"] and b["correct"],
                                "predictions_agree": a["prediction"] == b["prediction"]})
    correct = sum(x["correct"] for x in predictions)
    status = "ELIGIBLE_BASIC_READOUT_ONLY" if correct == 16 else "INELIGIBLE_STOP_RFR_PATH"
    summary = {
        "status": status, "scope": SCIENTIFIC_STATUS,
        "correct": correct, "total": 16, "strict_pass": correct == 16,
        "gold_counts": dict(Counter(x["gold"] for x in predictions)),
        "prediction_counts": dict(Counter(x["prediction"] or "INVALID" for x in predictions)),
        "opposite_facts_query_switch_pairs": opposite_pairs,
        "record_order_pairs": order_pairs,
        "constant_support_correct": 8, "constant_refute_correct": 8,
        "has_training": False, "h2_rfr_revalidated": False,
        "h3_causal_mediation": "NOT_ESTABLISHED", "phase3b": "BLOCKED_UNCHANGED",
        "limitations": "Pass means basic value retrieval only; failure does not distinguish lexical priors from missing competence.",
    }
    return {"predictions": predictions, "summary": summary}


def authenticate(repo: Path, snapshot: Path, approved_commit: str) -> dict:
    # Reuse the exact H2-P6 fail-closed repository/runtime/hash authenticator.
    auth = p6.authentication(repo, snapshot, approved_commit)
    need(auth["branch"] == EXPECTED_BRANCH, "BRANCH_DRIFT")
    new_source_hashes = {}
    for relative in ("scripts/rfr_h2_basic_readout_eligibility_cpu.py",
                     "tests/test_rfr_h2_basic_readout_eligibility_cpu.py"):
        local = repo / relative
        need(local.is_file(), "NEW_SOURCE_MISSING")
        # Require source to be tracked at the approved HEAD (not an uncommitted file).
        object_content = subprocess.run(
            ["git", "-C", str(repo), "show", f"{approved_commit}:{relative}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        local_sha = sha_file(local)
        need(local_sha == sha_bytes(object_content), "NEW_SOURCE_HEAD_DRIFT")
        new_source_hashes[relative] = local_sha
    auth["eligibility_source_sha256"] = new_source_hashes
    return auth


def tokenize(cases: list[dict], snapshot: Path) -> dict[str, list[int]]:
    from transformers import AutoTokenizer
    # P3's pinned tokenizer check on prior surfaces still holds; this *new*
    # prompt surface is validated separately in the next loop.
    audited = p3.audit_snapshot(snapshot)
    need(audited["result"] == p3.PASS_STATUS and
         audited["candidate_token_ids"] == dict(zip(LABELS, TOKEN_IDS)),
         "P3_TOKEN_ID_DRIFT")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    need(type(tokenizer).__name__ == "GPTNeoXTokenizerFast", "TOKENIZER_CLASS")
    reserved = set(tokenizer.all_special_ids)
    tokenized = {}
    for case in cases:
        prompt = case["prompt"]
        need(prompt.endswith("\nAnswer:"), "ANSWER_SEAM")
        ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        need(type(ids) is list and 0 < len(ids) <= 150 and
             all(type(x) is int and 0 <= x < 50280 for x in ids), "TOKENIZED_PROMPT")
        for label, tid in zip(LABELS, TOKEN_IDS):
            need(tid not in reserved, "ANSWER_SPECIAL_TOKEN")
            full = tokenizer.encode(prompt + p3.CONTINUATIONS[label],
                                    add_special_tokens=False, truncation=False)
            need(full == ids + [tid], "NEW_SURFACE_CONTINUATION_MISMATCH")
        tokenized[case["case_id"]] = ids
    need(len(tokenized) == 16, "TOKENIZED_16")
    return tokenized


def evaluate(cases: list[dict], tokenized: dict[str, list[int]], snapshot: Path) -> dict:
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
             model.get_output_embeddings().weight.data_ptr(), "UNTIED_HEAD")
        need(len(model.backbone.layers) == 24, "LAYER_COUNT")
        scores = {}
        for case in cases:
            ids = tokenized[case["case_id"]]
            output = model(input_ids=torch.tensor([ids], dtype=torch.long, device="cpu"),
                           use_cache=False, labels=None, output_hidden_states=False,
                           return_dict=True)
            need(tuple(output.logits.shape) == (1, len(ids), 50280), "FORWARD_SHAPE")
            need(output.cache_params is None, "CACHE_PRESENT")
            scores[case["case_id"]] = tuple(
                float(output.logits[0, -1, i].item()) for i in TOKEN_IDS)
        del model
    result = parse_scores(cases, scores)
    for pred in result["predictions"]:
        pred["token_count"] = len(tokenized[pred["case_id"]])
    return result


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def save_run(root: Path, name: str, manifest: dict, result: dict) -> Path:
    need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name) is not None, "RUN_NAME")
    root = Path(root)
    final = root / name
    partial = root / ("." + name + ".partial")
    need(not final.exists() and not partial.exists(), "RUN_EXISTS")
    s = result["summary"]
    report = ("# Prospective 16-item P/Q readout eligibility (exploratory)\n\n"
              + f"Run: `{name}`. Approved commit: `{manifest['head']}`.\n\n"
              + f"**Correct: {s['correct']}/16; status: {s['status']}.**\n\n"
              + "This is basic fact readout, not RFR, H3, or independent generalization.\n"
              + "The 16/16 criterion was fixed before observing these new logits.\n")
    files = {
        "manifest.json": canonical_json(manifest),
        "predictions.jsonl": b"".join(json.dumps(x, ensure_ascii=False,
            sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
            for x in result["predictions"]),
        "summary.json": canonical_json(s), "report.md": report.encode("utf-8"),
    }
    checksums = "".join(f"{sha_bytes(data)}  {fname}\n" for fname, data in sorted(files.items()))
    root.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        for fname, data in files.items():
            (partial / fname).write_bytes(data)
        (partial / "SHA256SUMS.txt").write_bytes(checksums.encode("utf-8"))
        for fname, data in files.items():
            need(sha_file(partial / fname) == sha_bytes(data), "ARTIFACT_HASH")
        need({x.name for x in partial.iterdir()} == {*files, "SHA256SUMS.txt"},
             "ARTIFACT_COVERAGE")
        need(not final.exists(), "RUN_OVERWRITE")
        partial.rename(final)
    except Exception:
        raise   # Preserve incomplete .partial for forensic inspection.
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-commit")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    try:
        cases = build_grid()
        if not args.execute:
            need(all(x is None for x in (args.approved_commit, args.snapshot,
                                        args.output_root, args.run_name)),
                 "EXECUTE_REQUIRED")
            print(json.dumps({"status": "STATIC_ELIGIBILITY_CONTRACT_ONLY_NO_MODEL",
                              "contexts": 16, "truth_worlds": 4, "record_orders": 2,
                              "queries": 2, "gold": {"REFUTE": 8, "SUPPORT": 8},
                              "template_example": make_prompt(False, True, ("P", "Q"), "P"),
                              "strict_acceptance": "16/16",
                              "model_evaluation": False, "phase3b": "BLOCKED_UNCHANGED"},
                             indent=2))
            return 0
        need(all(x is not None for x in (args.approved_commit, args.snapshot,
                                       args.output_root, args.run_name)), "EXECUTION_ARGUMENTS")
        need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_name) is not None,
             "RUN_NAME")
        repo, snapshot = args.repo_root.resolve(), args.snapshot.resolve()
        dest = args.output_root.resolve() / args.run_name
        need(not dest.exists() and
             not (args.output_root.resolve() / ("." + args.run_name + ".partial")).exists(),
             "RUN_EXISTS")
        auth = authenticate(repo, snapshot, args.approved_commit)
        tokens = tokenize(cases, snapshot)
        result = evaluate(cases, tokens, snapshot)
        manifest = {**auth, "scope": SCIENTIFIC_STATUS,
                    "head": args.approved_commit, "run_name": args.run_name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model_id": MODEL_ID, "revision": REVISION, "dtype": "float32",
                    "device": "cpu", "model_evaluation": True, "training": False,
                    "gold": {"REFUTE": 8, "SUPPORT": 8},
                    "token_ids": dict(zip(LABELS, TOKEN_IDS)),
                    "template": source_template(), "strict_acceptance": "16/16",
                    "independent_roots": 0, "h2_fixed_grid_outcome": "NO_SUPPORT_UNCHANGED",
                    "phase3b": "BLOCKED_UNCHANGED"}
        dest = save_run(args.output_root, args.run_name, manifest, result)
        print(json.dumps({"status": "ARTIFACT_WRITTEN_NOT_INDEPENDENTLY_VALIDATED",
                          "artifact_path": str(dest),
                          "correct": result["summary"]["correct"],
                          "strict_pass": result["summary"]["strict_pass"],
                          "phase3b": "BLOCKED_UNCHANGED"}, indent=2))
        return 0
    except (EligibilityError, p6.ContractError, ValueError, OSError,
            subprocess.CalledProcessError) as exc:
        print("H2_BASIC_RESULT=BLOCKED")
        print("REASON=" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
