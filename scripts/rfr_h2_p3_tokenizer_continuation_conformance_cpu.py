"""RFR-H2-P3: pinned-tokenizer-only, no-model continuation boundary preflight.

Scope: 8 hypothetical Boolean strata; 216 ephemeral prompt contexts; no corpus,
RNG, sampled root, model, logits, checkpoint, trained readout, or native state.
Frozen H2-P1 remains unchanged. This tests a *readout seam*, not RFR behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from scripts import rfr_h2_p1_temporal_serialization_contract_cpu as p1

MODEL_ID = "state-spaces/mamba-130m-hf"
REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
SNAPSHOT_FILES_SHA256 = {
    "tokenizer.json": "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8",
    "tokenizer_config.json": "fcd5669efe1150240c13ee4bd863316de4f2abd14cb1806a8cdbcbea6577bc99",
    "special_tokens_map.json": "10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c",
    "config.json": "784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a",
}
CONTINUATIONS = {
    "REFUTE": " False",
    "NOT_ENTITLED": " Unknown",
    "SUPPORT": " True",
}
EXPECTED_LABEL_ORDER = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
QUERY_SUFFIX_ORIGINAL = "Answer: "
QUERY_SUFFIX_REASSIGNED = "Answer:"
SCORING_RULE = "ONE_NEXT_TOKEN_LOGIT_AT_FINAL_QUERY_POSITION_NO_EOS_NO_NORMALIZATION"
PASS_STATUS = "H2_P3_TOKEN_BOUNDARY_CONFORMANCE_PASS_NO_MODEL"


class ConformanceError(ValueError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ConformanceError("H2_P3_" + code)


def authenticate_snapshot(snapshot: Path) -> dict[str, Any]:
    """Hash local files before tokenizer import. No downloads or writes."""
    snapshot = Path(snapshot)
    require(snapshot.is_dir(), "SNAPSHOT_DIRECTORY_MISSING")
    require(snapshot.name == REVISION, "REVISION_MISMATCH")
    for name, expected in SNAPSHOT_FILES_SHA256.items():
        file = snapshot / name
        require(file.is_file(), "SNAPSHOT_FILE_MISSING_" + name)
        require(hashlib.sha256(file.read_bytes()).hexdigest() == expected,
                "SNAPSHOT_SHA256_MISMATCH_" + name)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    require(type(config) is dict and type(config.get("vocab_size")) is int,
            "CONFIG_VOCAB_SCHEMA")
    require(config["vocab_size"] == 50280, "CONFIG_VOCAB_INVALID")
    require(config.get("architectures") == ["MambaForCausalLM"],
            "CONFIG_NOT_CAUSAL_LM")
    return config


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    result = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    require(isinstance(result, (list, tuple)), "TOKENIZER_OUTPUT_TYPE")
    require(bool(result) and all(type(t) is int and t >= 0 for t in result),
            "TOKENIZER_OUTPUT_IDS")
    return tuple(result)


def _segments(world: dict[str, bool], arm: str, readout: str) -> tuple[str, str, str, str]:
    rule = p1.RULE_TEXT
    t0 = p1._t0_text(world)
    t1 = "" if readout == "PRE" else p1._t1_text(world, arm)
    delay = p1.DELAY_TEXT if readout == "DELAYED" else ""
    return rule, t0, t1, delay


def audit_hypothetical_conformance(
    tokenizer: Any, *, vocab_size: int, reserved_token_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Fail-closed boundary verification using an injected read-only tokenizer.

    This is NOT proof of actual LM weights, logits, class priors, or scientific
    validity. Tests can inject synthetic tokenizers without loading a model.
    """
    require(type(vocab_size) is int and vocab_size > 3, "VOCAB_SIZE")
    require(type(reserved_token_ids) is tuple and all(
        type(value) is int and value >= 0 for value in reserved_token_ids
    ), "SPECIAL_TOKEN_SCHEMA")
    reserved = set(reserved_token_ids)
    require(tuple(p1.READOUT_SPEC["label_order"]) == EXPECTED_LABEL_ORDER,
            "P1_LABEL_CONTRACT_DRIFT")
    require(p1.READOUT_SPEC["candidate_continuations"] == "UNSET_NO_TOKENIZER" and
            p1.READOUT_SPEC["evaluation_enabled"] is False,
            "P1_READOUT_PROMOTION")
    require(set(CONTINUATIONS) == set(EXPECTED_LABEL_ORDER) and
            all(type(s) is str and s.startswith(" ") and s.count(" ") == 1
                for s in CONTINUATIONS.values()), "CONTINUATION_CONTRACT")

    prompt_lengths: list[int] = []
    complete_lengths: list[int] = []
    candidate_id_sets: dict[str, set[int]] = {k: set() for k in EXPECTED_LABEL_ORDER}
    checked_contexts = 0
    checked_continuations = 0
    checked_anchors = 0
    common_prefixes: dict[tuple[bool, bool, bool, str, str], tuple[int, ...]] = {}
    pre_prompts: dict[tuple[bool, bool, bool, str], tuple[int, ...]] = {}
    post_prefixes: dict[tuple[bool, bool, bool, str], tuple[int, ...]] = {}
    delayed_prefixes: dict[tuple[bool, bool, bool, str], tuple[int, ...]] = {}

    for a, c, u in itertools.product((False, True), repeat=3):
        world = {"A": a, "C": c, "U": u}
        world_key = (a, c, u)
        for arm, readout in itertools.product(p1.ARMS, p1.READOUTS):
            rule, t0, t1, delay = _segments(world, arm, readout)
            prefix = rule + t0 + t1 + delay
            prefix_ids = _encode(tokenizer, prefix)

            if readout in ("POST", "DELAYED"):
                # Check that the same t1 record is reused verbatim.
                require(t1 == p1._t1_text(world, arm), "T1_DRIFT")
                key = (*world_key, arm)
                if readout == "POST":
                    post_prefixes[key] = prefix_ids
                else:
                    delayed_prefixes[key] = prefix_ids

            for query in p1.QUERIES:
                original_prompt = prefix + p1._query_text(query)
                require(original_prompt.endswith(QUERY_SUFFIX_ORIGINAL),
                        "P1_QUERY_SCAFFOLD_CHANGED")
                revised_prompt = original_prompt[:-1]
                require(revised_prompt.endswith(QUERY_SUFFIX_REASSIGNED),
                        "REASSIGNED_QUERY_SUFFIX")
                prompt_ids = _encode(tokenizer, revised_prompt)
                prompt_lengths.append(len(prompt_ids))
                checked_contexts += 1

                for text in (rule, rule + t0,
                             *( (rule + t0 + t1,) if readout != "PRE" else () ),
                             *( (prefix,) if readout == "DELAYED" else () ),
                             prefix):
                    anchor = _encode(tokenizer, text)
                    checked_anchors += 1
                    require(prompt_ids[:len(anchor)] == anchor,
                            "TOKEN_ANCHOR_MISMATCH")

                prefix_key = (*world_key, arm, readout)
                existing = common_prefixes.setdefault(prefix_key, prefix_ids)
                require(existing == prefix_ids, "QUERY_PREFIX_CHANGED")
                if readout == "PRE":
                    pre_key = (*world_key, query)
                    previous = pre_prompts.setdefault(pre_key, prompt_ids)
                    require(previous == prompt_ids, "PRE_ARM_LEAKAGE")

                observed: dict[str, int] = {}
                for label in EXPECTED_LABEL_ORDER:
                    answer = CONTINUATIONS[label]
                    require(revised_prompt + answer == original_prompt + answer[1:],
                            "VISIBLE_TEXT_CHANGED")
                    full_ids = _encode(tokenizer, revised_prompt + answer)
                    checked_continuations += 1
                    require(full_ids[:len(prompt_ids)] == prompt_ids,
                            "ANSWER_BOUNDARY_MISMATCH")
                    suffix = full_ids[len(prompt_ids):]
                    require(len(suffix) == 1, "CONTINUATION_NOT_SINGLE_TOKEN")
                    token_id = suffix[0]
                    require(token_id < vocab_size, "CANDIDATE_OUTSIDE_MODEL_VOCAB")
                    require(token_id not in reserved, "ANSWER_IS_SPECIAL_TOKEN")
                    candidate_id_sets[label].add(token_id)
                    observed[label] = token_id
                    complete_lengths.append(len(full_ids))
                require(len(set(observed.values())) == len(EXPECTED_LABEL_ORDER),
                        "CANDIDATE_TOKEN_COLLISION")

    for key, post in post_prefixes.items():
        delayed = delayed_prefixes.get(key)
        require(delayed is not None and delayed[:len(post)] == post,
                "DELAY_DID_NOT_PRESERVE_POST_TOKENS")
    require(len(post_prefixes) == 24 and len(delayed_prefixes) == 24,
            "POST_DELAY_COVERAGE")
    require(checked_contexts == 216 and checked_continuations == 648,
            "FACTORIAL_COVERAGE")
    require(all(len(s) == 1 for s in candidate_id_sets.values()),
            "CANDIDATE_TOKEN_CONTEXT_DRIFT")
    final_ids = {label: next(iter(candidate_id_sets[label]))
                 for label in EXPECTED_LABEL_ORDER}
    require(len(set(final_ids.values())) == 3, "CANDIDATE_TOKEN_COLLISION")
    return {
        "result": PASS_STATUS,
        "hypothetical_boolean_strata": 8,
        "independently_sampled_roots": 0,
        "prompts_checked": checked_contexts,
        "candidate_checks": checked_continuations,
        "anchors_checked": checked_anchors,
        "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
        "complete_token_length_range": [min(complete_lengths), max(complete_lengths)],
        "candidate_token_ids": final_ids,
        "candidate_token_lengths": {k: 1 for k in EXPECTED_LABEL_ORDER},
        "visible_completed_text_unchanged": True,
        "scoring_rule_proposed_not_executed": SCORING_RULE,
        "actual_model_input_budget": "NOT_VERIFIED",
        "lm_head_loaded": False,
        "model_evaluation": False,
        "score_calibration_and_class_priors": "NOT_ESTABLISHED",
        "unknown_class_gold_coverage": "ABSENT_IN_PRIMARY_BOOLEAN_WORLDS",
        "native_state_mediation": "NOT_ESTABLISHED",
        "phase3b": "BLOCKED_UNCHANGED",
    }


def audit_snapshot(snapshot: Path) -> dict[str, Any]:
    config = authenticate_snapshot(snapshot)
    # Intentionally lazy: importing this module does not load transformers.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        use_fast=True,
    )
    require(getattr(tokenizer, "is_fast", False) is True,
            "NOT_FAST_TOKENIZER")
    require(type(tokenizer).__name__ == "GPTNeoXTokenizerFast",
            "TOKENIZER_TYPE_MISMATCH")
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = config.get(key)
        if type(value) is int:
            special_ids.add(value)
    output = audit_hypothetical_conformance(
        tokenizer, vocab_size=config["vocab_size"],
        reserved_token_ids=tuple(sorted(special_ids)),
    )
    output["model_id"] = MODEL_ID
    output["snapshot_revision"] = REVISION
    output["snapshot_hash_check"] = "PASS"
    output["tokenizer_class"] = type(tokenizer).__name__
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-snapshot", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = audit_snapshot(args.tokenizer_snapshot)
    except (ConformanceError, ValueError, OSError) as exc:
        print("H2_P3_RESULT=BLOCKED")
        print("REASON=" + str(exc))
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
