import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_longterm_o0b_matched_controls.py"
SPEC = importlib.util.spec_from_file_location("o0b_validator", MODULE_PATH)
validator = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(validator)


class OffsetTokenizer:
    is_fast = True
    def __init__(self, spans=None): self.vocab = {}; self.calls = 0; self.spans = spans
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        assert add_special_tokens is False; self.calls += 1
        if self.spans is not None: return {"input_ids": list(range(1, len(self.spans) + 1)), "offset_mapping": self.spans}
        spans = []; ids = []
        starts = [0]
        for i, ch in enumerate(text):
            if ch == " ": starts.append(i + 1)
        # Token spans include their preceding separator, making the first evidence span cross the boundary.
        words = list(__import__("re").finditer(r"\S+", text))
        for n, match in enumerate(words):
            a = match.start() - (1 if match.start() else 0); b = match.end(); spans.append([a, b])
            token = text[a:b]
            ids.append(self.vocab.setdefault(token, len(self.vocab) + 1))
        return {"input_ids": ids, "offset_mapping": spans}


def record(pair_id):
    return {"schema_version": validator.SCHEMA_VERSION, "pair_id": pair_id, "claim": "Shared claim text.", "reference_sufficient": "alpha beta gamma delta", "paraphrase_sufficient": "alpha beta theta delta", "insufficient_matched": "alpha beta omega delta", "surface_null_matched": "alpha beta sigma delta", "insufficiency_rationale": "The control withholds the decisive support.", "paraphrase_rationale": "The wording changes while support remains.", "surface_null_rationale": "The wording changes while sufficiency remains."}

def records(): return [record(x) for x in validator.PAIR_IDS]
def raw_records(rs): return ("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rs)).encode()
def loader_factory(tokenizer, calls=None):
    def loader(*args, **kwargs):
        if calls is not None: calls.append((args, kwargs))
        return tokenizer
    return loader
def run(rs=None, tokenizer=None):
    rs = rs or records(); raw = raw_records(rs); t = tokenizer or OffsetTokenizer()
    with patch.object(validator, "read_dataset_bytes", return_value=raw):
        return validator.validate_to_bytes(dataset_path=Path("memory.jsonl"), repository_head=validator.BOUNDARY_RECOVERY_AUTHORITY_COMMIT, tokenizer_loader=loader_factory(t), check_existing_artifact=False)

def test_valid_metadata_and_canonical_bytes():
    payload, encoded = run(); assert payload["overall"] == "PASS"; assert payload["tokenized_text_coordinate_domain"] == "printable_ascii_u0020_u007e"; assert payload["boundary_recovery_authority_commit"] == validator.BOUNDARY_RECOVERY_AUTHORITY_COMMIT
    assert encoded == validator.canonical_json_bytes(json.loads(encoded.decode())); assert b"\r" not in encoded

@pytest.mark.parametrize("bad", ["é", "漢", "🙂", "\nalpha", " alpha", "alpha ", "alpha\r", "alpha\t", "smart’", "en–dash", "zero\u200bwidth"])
def test_domain_guard_rejects_before_tokenizer(bad):
    rs = records(); rs[0]["reference_sufficient"] = bad; t = OffsetTokenizer(); calls = []
    with patch.object(validator, "read_dataset_bytes", return_value=raw_records(rs)):
        with pytest.raises(validator.ValidationError): validator.validate_to_bytes(dataset_path=Path("memory.jsonl"), repository_head=validator.BOUNDARY_RECOVERY_AUTHORITY_COMMIT, tokenizer_loader=loader_factory(t, calls), check_existing_artifact=False)
    assert calls == []; assert t.calls == 0

def test_clean_domain_allows_tokenizer():
    t = OffsetTokenizer(); run(tokenizer=t); assert t.calls > 0

def test_schema_ids_and_equal_counts():
    rs = records(); rs[0]["pair_id"] = "bad"
    with pytest.raises(validator.ValidationError): validator.validate_records(rs)
    rs = records(); rs[0]["paraphrase_sufficient"] += " extra"
    with pytest.raises(validator.ValidationError, match="unequal"): run(rs)

def test_offset_fail_closed_cases():
    base = [[0, 1], [2, 3], [4, 5]]
    for spans in ([[0, 0], [2, 3]], [[0, 2], [1, 3]], [[2, 3], [1, 4]], [[0, 3], [2, 4]], [[0, 1, 2]], [[0, 1], [2, 99]]):
        with pytest.raises(validator.ValidationError): validator.derive_member_tokens(OffsetTokenizer(spans), "A", "B")

def test_boundary_start_and_crossing_are_recorded():
    t = OffsetTokenizer(); member = validator.derive_member_tokens(t, "A", "B")
    assert member["evidence_start_index"] == 3; assert member["boundary_crossing"] is True
    boundary = len(validator.prefix_text("A")); exact = [[0, 1], [2, boundary], [boundary, boundary + 1]]
    # The exact-boundary case is exercised directly with a valid synthetic span layout.
    member = validator.derive_member_tokens(OffsetTokenizer(exact), "A", "B")
    assert member["boundary_crossing"] is False

def test_invariants_divergence_and_anchors():
    payload, _ = run(); pair = payload["pairs"][0]; assert pair["matched_set_invariants"]["pre_evidence_token_id_invariant"]
    for c in validator.CONDITIONS[1:]:
        cmp = pair["comparisons_to_reference"][c]; assert cmp["first_divergent_token_index"] >= pair["matched_set_invariants"]["common_evidence_start_index"]; assert set(cmp["anchor_indices"]) == set(validator.ANCHOR_KEYS)

def test_tokenizer_security_and_settings():
    calls = []; t = OffsetTokenizer(); validator.safe_load_tokenizer(loader_factory(t, calls), tokenizer_id=validator.TOKENIZER_ID, revision=validator.TOKENIZER_REVISION, trust_remote_code=False, add_special_tokens=False)
    assert calls[0][1] == {"revision": validator.TOKENIZER_REVISION, "trust_remote_code": False, "add_special_tokens": False, "use_fast": True}
    with pytest.raises(validator.ValidationError): validator.safe_load_tokenizer(loader_factory(t), tokenizer_id="wrong", revision=validator.TOKENIZER_REVISION, trust_remote_code=False, add_special_tokens=False)


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("revision", "wrong-revision", "wrong tokenizer revision"),
        ("add_special_tokens", True, "add_special_tokens must be False"),
        ("trust_remote_code", True, "trust_remote_code must be False"),
    ],
)
def test_tokenizer_contract_rejects_each_forbidden_setting(setting, value, message):
    calls = []
    kwargs = dict(tokenizer_id=validator.TOKENIZER_ID, revision=validator.TOKENIZER_REVISION,
                  trust_remote_code=False, add_special_tokens=False)
    kwargs[setting] = value
    with pytest.raises(validator.ValidationError, match=message):
        validator.safe_load_tokenizer(loader_factory(OffsetTokenizer(), calls), **kwargs)
    assert calls == []


def test_boundary_without_covering_token_fails():
    boundary = len(validator.prefix_text("A"))
    text_length = len(validator.serialize_member("A", "B"))
    spans = [[0, boundary], [boundary + 1, text_length]]
    with pytest.raises(validator.ValidationError, match="zero-length|uniquely covered"):
        validator.derive_member_tokens(OffsetTokenizer(spans), "A", "B")


def test_boundary_with_multiple_covering_tokens_fails_closed():
    boundary = len(validator.prefix_text("A"))
    spans = [[0, boundary + 1], [boundary, boundary + 2]]
    with pytest.raises(validator.ValidationError):
        validator.derive_member_tokens(OffsetTokenizer(spans), "A", "B")


class DivergenceTokenizer:
    is_fast = True
    def __init__(self, mode): self.mode = mode; self.calls = 0
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        self.calls += 1
        boundary = len(validator.prefix_text("Shared claim text."))
        ids = [1, 2, 3]
        if self.calls > 1 and self.mode == "before": ids[0] = 99
        elif self.calls > 1 and self.mode == "terminal": ids[-1] = 99
        return {"input_ids": ids, "offset_mapping": [[0, boundary], [boundary, boundary + 1], [boundary + 1, boundary + 2]]}


@pytest.mark.parametrize("mode,match", [("none", "missing divergence"), ("before", "pre-evidence|claim/scaffold"), ("terminal", "terminal relation")])
def test_divergence_contract_rejects_invalid_first_divergence(mode, match):
    with pytest.raises(validator.ValidationError, match=match):
        validator.validate_pair_tokens(record(validator.PAIR_IDS[0]), DivergenceTokenizer(mode))


def test_anchor_contract_rejects_adversarial_metadata():
    divergence, terminal = 5, 10
    valid = validator.anchor_dict(divergence, terminal)
    assert valid["anchor_post_plus_4"] == divergence + 4
    cases = []
    wrong_offset = dict(valid); wrong_offset["anchor_divergence"] += 1; cases.append(wrong_offset)
    missing_key = dict(valid); del missing_key["anchor_pre_minus_1"]; cases.append(missing_key)
    extra_key = dict(valid); extra_key["unexpected"] = 1; cases.append(extra_key)
    nonnull_out_of_range = dict(valid); nonnull_out_of_range["anchor_post_plus_4"] = terminal + 1; cases.append(nonnull_out_of_range)
    null_in_range = dict(valid); null_in_range["anchor_post_plus_1"] = None; cases.append(null_in_range)
    wrong_terminal = dict(valid); wrong_terminal["anchor_terminal"] = terminal - 1; cases.append(wrong_terminal)
    for anchors in cases:
        with pytest.raises(validator.ValidationError, match="anchor"):
            validator.validate_anchor_dict(anchors, divergence, terminal)


def _canonical_artifact_fixture(path):
    path.write_bytes((Path(__file__).resolve().parents[1] / "reports" / "longterm_o0b_matched_controls_v1_validation.json").read_bytes())


@pytest.mark.parametrize(
    "field",
    ["dataset_sha256", "scientific_design_authority_commit", "implementation_authority_commit", "boundary_recovery_authority_commit", "repository_head"],
)
def test_canonical_provenance_rejects_runtime_or_authority_mismatch(field):
    root = Path(__file__).resolve().parents[1]
    artifact = root / f".o0b_adversarial_{field}_artifact.json"
    dataset = root / f".o0b_adversarial_{field}_dataset.jsonl"
    try:
        _canonical_artifact_fixture(artifact)
        dataset.write_bytes((root / "data" / "longterm_o0b_matched_controls_v1.jsonl").read_bytes())
        kwargs = dict(dataset_path=dataset, artifact_path=artifact, repository_head=validator.BOUNDARY_RECOVERY_AUTHORITY_COMMIT, check_existing_artifact=True)
        if field == "dataset_sha256":
            dataset.write_bytes(dataset.read_bytes().replace(b"December 25, 2021", b"December 25, 2022", 1))
        else:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload[field] = "a" * (64 if field == "dataset_sha256" else 40)
            artifact.write_bytes(validator.canonical_json_bytes(payload))
        with pytest.raises(validator.ValidationError, match="mismatch"):
            validator.validate_to_bytes(**kwargs)
    finally:
        artifact.unlink(missing_ok=True)
        dataset.unlink(missing_ok=True)


def test_deterministic_bytes_have_frozen_encoding_contract():
    _, first = run(); _, second = run()
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert first.startswith(b"{") and not first.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in first and first.endswith(b"\n") and not first.endswith(b"\n\n")
    assert first == validator.canonical_json_bytes(json.loads(first.decode("utf-8")))


def _validator_uses_prohibited_model_loading(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prohibited_names = {"AutoModel", "AutoModelForCausalLM", "MambaModel", "torch", "safetensors", "pipeline"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = [alias.name.rsplit(".", 1)[-1] for alias in node.names]
            if any(name in prohibited_names for name in imported):
                return True
        if isinstance(node, ast.Name) and node.id in prohibited_names:
            return True
        if isinstance(node, ast.Attribute) and node.attr in prohibited_names:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"load", "load_file", "from_pretrained"}:
            receiver = node.func.value
            is_auto_tokenizer = isinstance(receiver, ast.Name) and receiver.id == "AutoTokenizer"
            is_qualified_auto_tokenizer = isinstance(receiver, ast.Attribute) and receiver.attr == "AutoTokenizer"
            if node.func.attr != "from_pretrained" or not (is_auto_tokenizer or is_qualified_auto_tokenizer):
                return True
    return False


def test_validator_has_no_prohibited_model_loading_path():
    assert _validator_uses_prohibited_model_loading(MODULE_PATH) is False

def test_existing_artifact_and_determinism():
    p1, b1 = run(); p2, b2 = run(); assert p1 == p2 and b1 == b2 and hashlib.sha256(b1).hexdigest() == hashlib.sha256(b2).hexdigest()

def test_real_tokenizer_contract_if_available():
    pytest.importorskip("transformers"); t = validator.load_tokenizer(); assert t.is_fast is True
    encoded = t("Claim: A\nEvidence: B", add_special_tokens=False, return_offsets_mapping=True); assert len(encoded["input_ids"]) == len(encoded["offset_mapping"])
