"""Fail-closed, token-aligned O0b native Mamba hidden-state observer.

This module is deliberately import-safe: transformers and torch are imported only
inside their loader/forward functions.  The scientific entry point is complete;
the accompanying tests use synthetic tokenizer/model objects.
"""
from __future__ import annotations

import argparse, hashlib, io, json, os, re, subprocess, sys, zipfile
from collections.abc import Sequence as SequenceABC
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCIENTIFIC_DESIGN_AUTHORITY_COMMIT = "df461469cb087f7f5db1e41a2b08e65ea517ad8"
BOUNDARY_RECOVERY_AUTHORITY_COMMIT = "2ed4439e511f7534186cbd5df9110e45fdc1d66c"
INPUT_IMPLEMENTATION_FREEZE_COMMIT = "7ce4e0cd05d87118c29526a53ab5178dc722db27"
OBSERVER_IMPLEMENTATION_AUTHORITY_COMMIT = "65881cf398d26b136e4984686b14f7d40b939c3e"
DATASET_PATH = "data/longterm_o0b_matched_controls_v1.jsonl"
DATASET_SHA256 = "75a675bee49cb26eb0935d364f0f5d090922dd01576dfc23294961b28394aec2"
VALIDATION_ARTIFACT_PATH = "reports/longterm_o0b_matched_controls_v1_validation.json"
VALIDATION_ARTIFACT_SHA256 = "e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff"
MODEL_ID = TOKENIZER_ID = "state-spaces/mamba-130m-hf"
MODEL_REVISION = TOKENIZER_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
COMPARISON_ORDER = ("insufficient_matched", "paraphrase_sufficient", "surface_null_matched")
ANCHOR_ORDER = ("anchor_pre_minus_1", "anchor_divergence", "anchor_post_plus_1", "anchor_post_plus_2", "anchor_post_plus_4", "anchor_terminal")
PRE_DIVERGENCE_RTOL, PRE_DIVERGENCE_ATOL, COSINE_REDUNDANCY_ATOL = 0.0, 1e-6, 1e-12
REQUIRED_ARTIFACTS = ("manifest.json", "anchor_observations.jsonl", "anchor_hidden_states.npz", "paired_distances.jsonl", "summary.json", "report.md", "SHA256SUMS.txt")
OBSERVATION_KEYS = ("schema_version", "pair_id", "comparison_id", "reference_condition", "member_condition", "vector_role", "condition", "anchor_name", "absolute_token_index", "layer_index", "layer_role", "state_source", "vector_index")
DISTANCE_KEYS = ("schema_version", "pair_id", "comparison_id", "reference_condition", "member_condition", "anchor_name", "absolute_token_index", "reference_absolute_token_index", "member_absolute_token_index", "layer_index", "layer_role", "state_source", "reference_vector_index", "member_vector_index", "normalized_l2_distance", "cosine_distance", "cosine_redundancy_error", "pre_divergence_integrity_status")
SUMMARY_KEYS = ("layer_index", "layer_role", "state_source", "anchor_name", "a_available_pair_ids", "a_available_pair_count", "a_mean", "a_median", "b_available_pair_ids", "b_available_pair_count", "b_mean", "b_median", "c_available_pair_ids", "c_available_pair_count", "c_mean", "c_median", "a_gt_b_comparable_pair_ids", "a_gt_b_denominator", "a_gt_b_count", "a_gt_c_comparable_pair_ids", "a_gt_c_denominator", "a_gt_c_count")
SCHEMA_VERSION = "longterm_o0b_token_aligned_native_mamba_state_dynamics_v1"
PAIR_ORDER = ("o0b_pair_001", "o0b_pair_002", "o0b_pair_003")
REFERENCE_CONDITION = "reference_sufficient"
LAYER_KEYS = ("layer_index", "layer_role", "state_source")
MANIFEST_KEYS = ("schema_version", "experiment_name", "scientific_design_authority_commit", "boundary_recovery_authority_commit", "input_implementation_freeze_commit", "observer_implementation_authority_commit", "observer_implementation_commit", "observer_script_sha256", "dataset_path", "dataset_sha256", "validation_artifact_path", "validation_artifact_sha256", "validation_artifact_repository_head", "model_id", "model_revision", "model_trust_remote_code", "tokenizer_id", "tokenizer_revision", "tokenizer_trust_remote_code", "tokenizer_use_fast", "add_special_tokens", "device", "dtype", "python_version", "numpy_version", "torch_version", "transformers_version", "serialization_template", "comparison_order", "anchor_order", "layer_descriptors", "pre_divergence_rtol", "pre_divergence_atol", "cosine_redundancy_atol", "exact_command", "run_name", "required_artifacts", "execution_status")
RUNTIME_VERSION_KEYS = ("python_version", "numpy_version", "torch_version", "transformers_version")
FORBIDDEN_RUNTIME_VERSION_PLACEHOLDERS = frozenset({"unknown", "n/a", "none"})

class ContractError(RuntimeError): pass
def require(ok: bool, msg: str) -> None:
    if not ok: raise ContractError(msg)
def sha256_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha256_file(p: Path) -> str: return sha256_bytes(p.read_bytes())
def canonical_json(v: Any) -> bytes: return (json.dumps(v, ensure_ascii=False, sort_keys=True, indent=2, separators=(",", ": "), allow_nan=False) + "\n").encode()
def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode() for r in rows)

def validate_tokenizer_kwargs(kwargs: Mapping[str, Any]) -> None:
    require(dict(kwargs) == {"revision": TOKENIZER_REVISION, "trust_remote_code": False, "use_fast": True}, "tokenizer loader kwargs")

def validate_model_kwargs(kwargs: Mapping[str, Any], torch_module: Any) -> None:
    require(kwargs.get("revision") == MODEL_REVISION and kwargs.get("torch_dtype") is torch_module.float32 and kwargs.get("trust_remote_code") is False and set(kwargs) == {"revision", "torch_dtype", "trust_remote_code"}, "model loader kwargs")

def load_tokenizer(factory: Any | None = None):
    if factory is None:
        from transformers import AutoTokenizer
        factory=AutoTokenizer
    kwargs = {"revision": TOKENIZER_REVISION, "trust_remote_code": False, "use_fast": True}; validate_tokenizer_kwargs(kwargs)
    tok = factory.from_pretrained(TOKENIZER_ID, **kwargs)
    require(getattr(tok, "is_fast", False) is True, "slow tokenizer")
    return tok
def load_model(factory: Any | None = None, torch_module: Any | None = None):
    if torch_module is None:
        import torch
        torch_module=torch
    if factory is None:
        from transformers import MambaModel
        factory=MambaModel
    kwargs = {"revision": MODEL_REVISION, "torch_dtype": torch_module.float32, "trust_remote_code": False}; validate_model_kwargs(kwargs, torch_module)
    model = factory.from_pretrained(MODEL_ID, **kwargs)
    model = model.to("cpu"); model.eval(); model.requires_grad_(False)
    return model

def validate_sources(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == 3, "exactly three pairs required")
    expected = {"o0b_pair_001", "o0b_pair_002", "o0b_pair_003"}
    require({r["pair_id"] for r in rows} == expected, "pair set mismatch")
    for row in rows:
        require(set(row) >= {"pair_id", "claim", *COMPARISON_ORDER, "reference_sufficient"}, "dataset fields")
        require(row["pair_id"].startswith("o0b_pair_") and len(row["pair_id"]) == 12, "pair id")
        for k in ("claim", "reference_sufficient", *COMPARISON_ORDER):
            v = row[k]; require(isinstance(v, str) and v and v == v.strip(), f"invalid source {k}")
            require(all(" " <= c <= "~" for c in v), f"non printable ASCII source {k}")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    require(all(isinstance(x, dict) for x in rows), "JSONL objects required"); return rows

def validate_artifact(a: Mapping[str, Any], dataset_sha: str) -> None:
    require(a.get("dataset_path") == DATASET_PATH and a.get("dataset_sha256") == dataset_sha, "artifact input provenance")
    require(a.get("scientific_design_authority_commit") == SCIENTIFIC_DESIGN_AUTHORITY_COMMIT, "design authority")
    require(a.get("boundary_recovery_authority_commit") == BOUNDARY_RECOVERY_AUTHORITY_COMMIT, "boundary authority")
    require(a.get("repository_head") == BOUNDARY_RECOVERY_AUTHORITY_COMMIT and a.get("overall") == "PASS", "validation provenance")
    require(a.get("pair_ids") == ["o0b_pair_001", "o0b_pair_002", "o0b_pair_003"], "artifact pairs")

def tokenize_full(tokenizer: Any, text: str) -> dict[str, Any]:
    out = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(out["input_ids"]); offsets = [list(x) for x in out["offset_mapping"]]
    require(len(ids) == len(offsets) and ids, "tokenization shape")
    return {"ids": [int(x) for x in ids], "offsets": offsets}

def validate_tokenized_members(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    validate_sources(rows); byid = {r["pair_id"]: r for r in rows}; result = []
    for p in artifact["pairs"]:
        row = byid[p["pair_id"]]
        for cond in ("reference_sufficient", *COMPARISON_ORDER):
            text = f"Claim: {row['claim']}\nEvidence: {row[cond]}"; got = tokenize_full(tokenizer, text); exp = p["conditions"][cond]
            require(got["ids"] == exp["full_serialized_token_ids"] and len(got["ids"]) == exp["full_token_count"], "token IDs/count mismatch")
            require(got["offsets"] == exp["full_offset_mapping"], "offset mismatch")
            evidence_char_start = len(f"Claim: {row['claim']}\nEvidence: ")
            start = exp["evidence_start_index"]
            require(exp["evidence_char_start"] == evidence_char_start and start < len(got["ids"]), "evidence boundary")
            require(got["offsets"][start] == [exp["evidence_start_offset_start"], exp["evidence_start_offset_end"]], "evidence offset")
            require(exp["evidence_token_count"] == len(got["ids"]) - start and exp["terminal_index"] == len(got["ids"]) - 1, "terminal metadata")
            require(exp["boundary_crossing"] is False, "boundary crossing")
            result.append({"pair": p, "condition": cond, "ids": got["ids"], "text": text})
        ref_ids = p["conditions"]["reference_sufficient"]["full_serialized_token_ids"]
        for c in COMPARISON_ORDER:
            cmp = p["comparisons_to_reference"][c]; member_ids = p["conditions"][c]["full_serialized_token_ids"]
            require(set(cmp["anchor_indices"]) == set(ANCHOR_ORDER), "anchor schema")
            divergence = next((i for i,(x,y) in enumerate(zip(ref_ids, member_ids)) if x != y), None)
            require(divergence == cmp["first_divergent_token_index"], "divergence mismatch")
            require(cmp["anchor_indices"]["anchor_pre_minus_1"] == divergence-1 and cmp["anchor_indices"]["anchor_divergence"] == divergence, "anchor mismatch")
            require(cmp["anchor_indices"]["anchor_terminal"] == len(ref_ids)-1, "terminal anchor mismatch")
    return result

def exposed_hidden_states(outputs: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    hs = getattr(outputs, "hidden_states", None); last = getattr(outputs, "last_hidden_state", None)
    require(hs is not None and len(hs) > 0, "missing hidden_states"); require(last is not None, "missing last_hidden_state")
    layers = list(hs)
    def arr(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    if not (np.shape(arr(layers[-1])) == np.shape(arr(last)) and np.array_equal(arr(layers[-1]), arr(last))): layers.append(last)
    desc = []
    for i, _ in enumerate(layers):
        role = "embedding_or_initial_hidden_state" if i == 0 else "output_hidden_state" if i == len(layers)-1 else "intermediate_hidden_state"
        desc.append({"layer_index": i, "layer_role": role, "state_source": f"hidden_states[{i}]" if i < len(hs) else "last_hidden_state"})
    return layers, desc

def run_native_forward(model: Any, input_ids: Sequence[int]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    import torch
    x = torch.tensor([list(input_ids)], dtype=torch.long, device="cpu")
    with torch.inference_mode(): out = model(input_ids=x, output_hidden_states=True, return_dict=True, use_cache=False)
    layers, desc = exposed_hidden_states(out); arrays = []
    for x in layers:
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        require(a.ndim == 3 and a.shape[0] == 1 and a.shape[1] == len(input_ids) and a.shape[2] > 0, "hidden shape")
        require(a.dtype == np.float32 and np.isfinite(a).all(), "hidden dtype/finite")
        arrays.append(np.array(a[0], dtype=np.float32, copy=True))
    return arrays, desc

def assert_pre_divergence(reference: np.ndarray, member: np.ndarray, index: int, ids_ref: Sequence[int], ids_member: Sequence[int]) -> None:
    require(index >= 0 and index < len(ids_ref) and index < len(ids_member), "pre-divergence index")
    require(list(ids_ref[:index+1]) == list(ids_member[:index+1]), "pre-divergence token mismatch")
    require(np.allclose(reference[index], member[index], rtol=0.0, atol=PRE_DIVERGENCE_ATOL), "pre-divergence hidden mismatch")

def distance(member_vector: Any, reference_vector: Any) -> dict[str, float]:
    m = np.asarray(member_vector, dtype=np.float64); r = np.asarray(reference_vector, dtype=np.float64)
    require(m.ndim == r.ndim == 1 and m.size and r.size == m.size, "vector shape")
    require(np.isfinite(m).all() and np.isfinite(r).all(), "nonfinite vector")
    mn, rn = float(np.linalg.norm(m)), float(np.linalg.norm(r)); require(np.isfinite(mn) and np.isfinite(rn) and mn > 0 and rn > 0, "zero norm")
    mu, ru = m/mn, r/rn; d = float(np.linalg.norm(mu-ru)); cos = float(1.0-np.dot(mu, ru)); err = abs(d*d-2.0*cos)
    require(np.isfinite(d) and np.isfinite(cos) and np.isfinite(err) and err <= COSINE_REDUNDANCY_ATOL, "cosine redundancy")
    return {"normalized_l2_distance": float(d), "cosine_distance": float(cos), "cosine_redundancy_error": float(err)}

def deterministic_npz(vectors: Sequence[Any]) -> bytes:
    a = np.ascontiguousarray(np.asarray(vectors, dtype=np.dtype("<f4"))); require(a.ndim == 2 and a.shape[0] > 0 and a.shape[1] > 0, "vector matrix")
    raw = io.BytesIO(); np.lib.format.write_array(raw, a, version=(1, 0), allow_pickle=False)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=False, compresslevel=None) as z:
        info = zipfile.ZipInfo("vectors.npy", (1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_STORED; info.create_system=3; info.create_version=20; info.extract_version=20; info.external_attr=0o100644 << 16; info.internal_attr=0; info.extra=b""; info.comment=b""; info.flag_bits=0; z.comment=b""; z.writestr(info, raw.getvalue())
    return out.getvalue()

def checksum_text(files: Mapping[str, bytes]) -> bytes:
    names = sorted(files); require(names == sorted(set(names)) and "SHA256SUMS.txt" not in names, "checksum set")
    return "".join(f"{sha256_bytes(files[n])}  {n}\n" for n in names).encode("ascii")

def parse_checksums(data: bytes, files: Mapping[str, bytes]) -> None:
    require(data.endswith(b"\n") and b"\r" not in data, "checksum line endings")
    lines=data.split(b"\n")[:-1]; allowed=set(REQUIRED_ARTIFACTS[:-1]); require(len(lines)==6, "checksum line count")
    names=[]
    for line in lines:
        require(re.fullmatch(rb"[0-9a-f]{64}  [^\r\n]+", line) is not None, "checksum syntax")
        digest, name=line.split(b"  "); name=name.decode("ascii"); require(name in allowed and name not in names, "checksum filename"); names.append(name); require(digest.decode()==sha256_bytes(files[name]), "checksum digest")
    require(names==sorted(names) and set(names)==allowed, "checksum ordering/set")

def descriptive_summary(rows: Sequence[Mapping[str, Any]], pair_ids: Sequence[str], layer: Mapping[str, Any], anchor: str) -> dict[str, Any]:
    out = dict(layer); out["anchor_name"] = anchor
    vals = {k: [(r["pair_id"], float(r["normalized_l2_distance"])) for r in rows if r["comparison_id"] == k and r["anchor_name"] == anchor and r["layer_index"] == layer["layer_index"]] for k in COMPARISON_ORDER}
    for prefix, key in zip(("a", "b", "c"), COMPARISON_ORDER):
        ids, vs = [x[0] for x in vals[key]], [x[1] for x in vals[key]]; require(ids==list(dict.fromkeys(ids)), "duplicate summary pair"); require(set(ids)<=set(pair_ids), "summary pair"); out[f"{prefix}_available_pair_ids"] = ids; out[f"{prefix}_available_pair_count"] = len(vs); out[f"{prefix}_mean"] = float(np.mean(vs)) if vs else None; out[f"{prefix}_median"] = float(np.median(vs)) if vs else None
    for prefix, left, right in (("a_gt_b", "insufficient_matched", "paraphrase_sufficient"),("a_gt_c", "insufficient_matched", "surface_null_matched")):
        lefts = {r["pair_id"]: float(r["normalized_l2_distance"]) for r in rows if r["comparison_id"] == left and r["anchor_name"] == anchor and r["layer_index"] == layer["layer_index"]}; rights = {r["pair_id"]: float(r["normalized_l2_distance"]) for r in rows if r["comparison_id"] == right and r["anchor_name"] == anchor and r["layer_index"] == layer["layer_index"]}; ids = [p for p in pair_ids if p in lefts and p in rights]
        out[f"{prefix}_comparable_pair_ids"] = ids; out[f"{prefix}_denominator"] = len(ids); out[f"{prefix}_count"] = sum(lefts[i] > rights[i] for i in ids)
    return out

def _pair_records(coordinates: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    pairs = coordinates.get("pairs", coordinates) if isinstance(coordinates, Mapping) else coordinates
    by_id = {p["pair_id"]: p for p in pairs}
    require(tuple(by_id) == PAIR_ORDER or set(by_id) == set(PAIR_ORDER), "pair order")
    return [by_id[p] for p in PAIR_ORDER]

def validate_layer_descriptors(descriptors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out=[]; seen=set()
    for i, d in enumerate(descriptors):
        require(set(d) == set(LAYER_KEYS), "layer descriptor keys"); require(type(d["layer_index"]) is int and d["layer_index"] == i and i not in seen, "layer index")
        require(d["layer_role"] in {"embedding_or_initial_hidden_state", "intermediate_hidden_state", "output_hidden_state"}, "layer role")
        require(isinstance(d["state_source"], str) and (d["state_source"].startswith("hidden_states[") or d["state_source"] == "last_hidden_state"), "state source")
        seen.add(i); out.append(dict(d))
    require(bool(out), "empty layer descriptors"); return out

def _anchor_coordinates(coordinates: Mapping[str, Any], pair_id: str, comparison: str) -> Mapping[str, Any]:
    p = next(x for x in _pair_records(coordinates) if x["pair_id"] == pair_id)
    c = p["comparisons_to_reference"][comparison]
    require(c.get("reference_condition") == REFERENCE_CONDITION, "reference condition")
    return c.get("anchor_indices", c)

def validate_observation_records(rows: Sequence[Mapping[str, Any]], coordinates: Mapping[str, Any], descriptors: Sequence[Mapping[str, Any]], vector_count: int | None = None) -> None:
    descriptors=validate_layer_descriptors(descriptors); seen=set(); expected=[]
    for p in PAIR_ORDER:
        for comp in COMPARISON_ORDER:
            for anchor in ANCHOR_ORDER:
                idx=_anchor_coordinates(coordinates,p,comp).get(anchor)
                if idx is None: continue
                for layer in descriptors:
                    for role, cond in (("reference","reference_sufficient"),("member",comp)):
                        expected.append((p,comp,anchor,layer["layer_index"],role,cond,idx))
    require(len(rows)==len(expected), "observation row count"); physical={}; reverse={}
    for row, exp in zip(rows, expected):
        require(set(row)==set(OBSERVATION_KEYS), "observation keys")
        p,comp,anchor,li,role,cond,idx=exp
        require((row["pair_id"],row["comparison_id"],row["anchor_name"],row["layer_index"],row["vector_role"],row["condition"],row["absolute_token_index"])==exp, "observation order/identity")
        require(row["schema_version"]==SCHEMA_VERSION and row["reference_condition"]==REFERENCE_CONDITION and row["member_condition"]==comp, "observation provenance")
        require(row["layer_role"]==descriptors[li]["layer_role"] and row["state_source"]==descriptors[li]["state_source"], "observation layer")
        require(type(row["absolute_token_index"]) is int and row["absolute_token_index"]>=0 and type(row["vector_index"]) is int and row["vector_index"]>=0, "observation index")
        require(row["vector_index"] < vector_count if vector_count is not None else True, "observation vector bounds")
        ident=(p,comp,anchor,li,role); require(ident not in seen, "duplicate observation"); seen.add(ident)
        physical_key=(p,row["condition"],row["absolute_token_index"],li); vi=row["vector_index"]
        if physical_key not in physical: require(vi==len(physical), "non-deterministic vector walk")
        require(physical.get(physical_key,vi)==vi and reverse.get(vi,physical_key)==physical_key, "physical vector correspondence"); physical[physical_key]=vi; reverse[vi]=physical_key

def validate_distance_records(rows: Sequence[Mapping[str, Any]], coordinates: Mapping[str, Any], descriptors: Sequence[Mapping[str, Any]], vector_count: int | None = None) -> None:
    descriptors=validate_layer_descriptors(descriptors); expected=[]
    for p in PAIR_ORDER:
        for comp in COMPARISON_ORDER:
            for anchor in ANCHOR_ORDER:
                idx=_anchor_coordinates(coordinates,p,comp).get(anchor)
                if idx is not None:
                    expected.extend((p,comp,anchor,layer["layer_index"],idx) for layer in descriptors)
    require(len(rows)==len(expected), "distance row count"); seen=set()
    for row, (p,comp,anchor,li,idx) in zip(rows,expected):
        require(set(row)==set(DISTANCE_KEYS), "distance keys"); require((row["pair_id"],row["comparison_id"],row["anchor_name"],row["layer_index"],row["absolute_token_index"])==(p,comp,anchor,li,idx), "distance order/identity")
        require(row["schema_version"]==SCHEMA_VERSION and row["reference_condition"]==REFERENCE_CONDITION and row["member_condition"]==comp, "distance provenance")
        require(row["reference_absolute_token_index"]==idx and row["member_absolute_token_index"]==idx, "distance coordinate")
        require(row["layer_role"]==descriptors[li]["layer_role"] and row["state_source"]==descriptors[li]["state_source"], "distance layer")
        for k in ("absolute_token_index","reference_absolute_token_index","member_absolute_token_index","layer_index","reference_vector_index","member_vector_index"): require(type(row[k]) is int and row[k]>=0, "distance index")
        for k in ("normalized_l2_distance","cosine_distance","cosine_redundancy_error"): require(type(row[k]) is float and np.isfinite(row[k]), "distance metric")
        require(row["cosine_redundancy_error"]<=COSINE_REDUNDANCY_ATOL, "distance redundancy"); require(row["pre_divergence_integrity_status"] == ("PASS" if anchor=="anchor_pre_minus_1" else "NOT_APPLICABLE"), "distance integrity")
        if vector_count is not None: require(row["reference_vector_index"]<vector_count and row["member_vector_index"]<vector_count, "distance vector bounds")
        ident=(p,comp,anchor,li); require(ident not in seen, "duplicate distance"); seen.add(ident)

def assemble_observations(coordinates: Mapping[str, Any], hidden_states: Mapping[tuple[str,str], Sequence[Any]], layer_descriptors: Sequence[Mapping[str, Any]], token_ids: Mapping[tuple[str,str], Sequence[int]] | None = None) -> dict[str, Any]:
    desc=validate_layer_descriptors(layer_descriptors); observations=[]; distances=[]; vectors=[]; keys={}; pair_ids=list(PAIR_ORDER)
    def get_vec(pair, cond, idx, li):
        key=(pair,cond,idx,li)
        if key in hidden_states:
            raw=hidden_states[key]
        elif (pair,cond) in hidden_states:
            raw=np.asarray(hidden_states[(pair,cond)])
            require(raw.ndim==3, "hidden state layout"); raw=raw[li,idx]
        else:
            require(False, "missing hidden state")
        arr=np.asarray(raw if np.asarray(raw).ndim==1 else np.asarray(raw)[idx], dtype=np.float32)
        require(arr.ndim==1 and arr.size and np.isfinite(arr).all(), "hidden vector")
        if key not in keys: keys[key]=len(vectors); vectors.append(np.array(arr,dtype=np.float32,copy=True))
        else: require(np.array_equal(vectors[keys[key]],arr), "physical vector mismatch")
        return keys[key]
    for p in pair_ids:
        for comp in COMPARISON_ORDER:
            for anchor in ANCHOR_ORDER:
                idx=_anchor_coordinates(coordinates,p,comp).get(anchor)
                if idx is None: continue
                for layer in desc:
                    refs=get_vec(p,REFERENCE_CONDITION,idx,layer["layer_index"]); mems=get_vec(p,comp,idx,layer["layer_index"])
                    for role,cond,vi in (("reference",REFERENCE_CONDITION,refs),("member",comp,mems)):
                        observations.append({"schema_version":SCHEMA_VERSION,"pair_id":p,"comparison_id":comp,"reference_condition":REFERENCE_CONDITION,"member_condition":comp,"vector_role":role,"condition":cond,"anchor_name":anchor,"absolute_token_index":idx,"layer_index":layer["layer_index"],"layer_role":layer["layer_role"],"state_source":layer["state_source"],"vector_index":vi})
                    if anchor=="anchor_pre_minus_1":
                        require(token_ids is not None, "validated token IDs required")
                        assert_pre_divergence(np.asarray(hidden_states[(p,REFERENCE_CONDITION)][layer["layer_index"]]), np.asarray(hidden_states[(p,comp)][layer["layer_index"]]), idx, token_ids[(p,REFERENCE_CONDITION)], token_ids[(p,comp)])
                    m=distance(vectors[mems],vectors[refs]); distances.append({"schema_version":SCHEMA_VERSION,"pair_id":p,"comparison_id":comp,"reference_condition":REFERENCE_CONDITION,"member_condition":comp,"anchor_name":anchor,"absolute_token_index":idx,"reference_absolute_token_index":idx,"member_absolute_token_index":idx,"layer_index":layer["layer_index"],"layer_role":layer["layer_role"],"state_source":layer["state_source"],"reference_vector_index":refs,"member_vector_index":mems,**m,"pre_divergence_integrity_status":"PASS" if anchor=="anchor_pre_minus_1" else "NOT_APPLICABLE"})
    validate_observation_records(observations,coordinates,desc,len(vectors)); validate_distance_records(distances,coordinates,desc,len(vectors))
    require(all(r["pre_divergence_integrity_status"]=="PASS" for r in distances if r["anchor_name"]=="anchor_pre_minus_1"), "pre-divergence")
    return {"observations":observations,"vectors":np.ascontiguousarray(np.asarray(vectors,dtype=np.dtype("<f4"))),"distances":distances,"layer_descriptors":desc,"pair_ids":pair_ids,"token_ids":token_ids}

def build_summary(distances: Sequence[Mapping[str, Any]], pair_ids: Sequence[str], layer_descriptors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregates=[descriptive_summary(distances,pair_ids,l,a) for a in ANCHOR_ORDER for l in layer_descriptors]
    return {"schema_version":SCHEMA_VERSION,"comparison_order":list(COMPARISON_ORDER),"anchor_order":list(ANCHOR_ORDER),"integrity":{"pre_divergence_all_pass":all(r["pre_divergence_integrity_status"]=="PASS" for r in distances if r["anchor_name"]=="anchor_pre_minus_1"),"pair_ids":list(pair_ids),"layer_descriptors":[dict(x) for x in layer_descriptors]},"aggregates":aggregates}

def validate_summary(summary: Mapping[str, Any], pair_ids: Sequence[str], layer_descriptors: Sequence[Mapping[str, Any]], distances: Sequence[Mapping[str, Any]] | None = None) -> None:
    require(set(summary)=={"schema_version","comparison_order","anchor_order","integrity","aggregates"} and summary["schema_version"]==SCHEMA_VERSION, "summary keys/version")
    require(summary["comparison_order"]==list(COMPARISON_ORDER) and summary["anchor_order"]==list(ANCHOR_ORDER), "summary order"); integ=summary["integrity"]; require(set(integ)=={"pre_divergence_all_pass","pair_ids","layer_descriptors"} and integ["pre_divergence_all_pass"] is True and integ["pair_ids"]==list(pair_ids), "summary integrity")
    require(integ["layer_descriptors"]==[dict(x) for x in layer_descriptors], "summary layers"); require(len(summary["aggregates"])==len(ANCHOR_ORDER)*len(layer_descriptors), "summary aggregate count")
    expected=list(summary["aggregates"]) if distances is None else [descriptive_summary(distances,pair_ids,l,a) for a in ANCHOR_ORDER for l in layer_descriptors]
    require(summary["aggregates"]==expected, "summary semantic reconstruction")
    for r in summary["aggregates"]:
        require(set(r)==set(SUMMARY_KEYS), "summary aggregate keys")
        for prefix in ("a","b","c"):
            require(type(r[f"{prefix}_available_pair_ids"]) is list and r[f"{prefix}_available_pair_count"]==len(r[f"{prefix}_available_pair_ids"]), "summary counts")

def reconstruct_summary(distances: Sequence[Mapping[str, Any]], pair_ids: Sequence[str], layer_descriptors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return build_summary(distances,pair_ids,layer_descriptors)

def canonical_observer_argv(argv: Sequence[str]) -> str:
    require(isinstance(argv, SequenceABC) and not isinstance(argv, (str, bytes, bytearray)), "argv sequence")
    require(all(type(x) is str for x in argv), "argv strings")
    exact_command_indices = [i for i, x in enumerate(argv) if x == "--exact-command"]
    require(len(exact_command_indices) == 1, "exact command occurrence")
    index = exact_command_indices[0]
    require(index + 1 < len(argv), "exact command payload")
    remaining = [x for i, x in enumerate(argv) if i not in (index, index + 1)]
    return json.dumps(remaining, ensure_ascii=False, separators=(",",":"))

def actual_command(argv: Sequence[str]) -> str: return canonical_observer_argv(argv)
def verify_runtime_provenance(expected_observer_implementation_commit: str, expected_observer_script_sha256: str, script_path: Path, run_name: str, exact_command: str, argv: Sequence[str] | None = None, actual_head: str | None = None) -> dict[str, str]:
    require(isinstance(expected_observer_implementation_commit,str) and len(expected_observer_implementation_commit)==40 and all(c in "0123456789abcdef" for c in expected_observer_implementation_commit), "observer commit")
    require(isinstance(expected_observer_script_sha256,str) and len(expected_observer_script_sha256)==64 and all(c in "0123456789abcdef" for c in expected_observer_script_sha256), "observer script SHA")
    head=actual_head or subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(script_path).resolve().parents[1],text=True).strip(); actual=sha256_file(Path(script_path)); require(head==expected_observer_implementation_commit and actual==expected_observer_script_sha256, "runtime provenance")
    canonical=canonical_observer_argv(sys.argv if argv is None else argv); require(type(exact_command) is str and exact_command == canonical, "exact command provenance")
    return {"observer_implementation_commit":head,"observer_script_sha256":actual,"run_name":run_name,"exact_command":canonical}

def capture_runtime_versions() -> dict[str, str]:
    import torch as torch_module
    import transformers as transformers_module
    return {"python_version":sys.version.split()[0],"numpy_version":np.__version__,"torch_version":torch_module.__version__,"transformers_version":transformers_module.__version__}

def validate_runtime_versions(versions: Mapping[str, Any]) -> None:
    for key in RUNTIME_VERSION_KEYS:
        value=versions.get(key) if isinstance(versions, Mapping) else None
        require(type(value) is str and value and value == value.strip() and value.strip().lower() not in FORBIDDEN_RUNTIME_VERSION_PLACEHOLDERS, f"runtime version {key}")

def build_manifest(fields: Mapping[str, Any]) -> dict[str, Any]:
    m=dict(fields); require(set(m)==set(MANIFEST_KEYS), "manifest keys"); require(m["schema_version"]==SCHEMA_VERSION and m["scientific_design_authority_commit"]==SCIENTIFIC_DESIGN_AUTHORITY_COMMIT and m["boundary_recovery_authority_commit"]==BOUNDARY_RECOVERY_AUTHORITY_COMMIT and m["input_implementation_freeze_commit"]==INPUT_IMPLEMENTATION_FREEZE_COMMIT and m["observer_implementation_authority_commit"]==OBSERVER_IMPLEMENTATION_AUTHORITY_COMMIT, "manifest constants")
    validate_runtime_versions(m)
    require(m["required_artifacts"]==list(REQUIRED_ARTIFACTS) and m["execution_status"]=="COMPLETE", "manifest status/artifacts"); require(m["model_trust_remote_code"] is False and m["tokenizer_trust_remote_code"] is False and m["tokenizer_use_fast"] is True and m["add_special_tokens"] is False and m["device"]=="cpu" and m["dtype"]=="float32", "manifest runtime constants"); require(m["pre_divergence_rtol"]==0.0 and m["pre_divergence_atol"]==1e-6 and m["cosine_redundancy_atol"]==1e-12, "manifest tolerances")
    require(m["validation_artifact_repository_head"]==BOUNDARY_RECOVERY_AUTHORITY_COMMIT and m["dataset_path"]==DATASET_PATH and m["dataset_sha256"]==DATASET_SHA256 and m["validation_artifact_path"]==VALIDATION_ARTIFACT_PATH and m["validation_artifact_sha256"]==VALIDATION_ARTIFACT_SHA256 and m["model_id"]==MODEL_ID and m["tokenizer_id"]==TOKENIZER_ID and m["comparison_order"]==list(COMPARISON_ORDER) and m["anchor_order"]==list(ANCHOR_ORDER) and m["serialization_template"]=="canonical-json-v1/deterministic-npz-v1", "manifest frozen fields")
    validate_layer_descriptors(m["layer_descriptors"])
    for k,n in (("observer_implementation_commit",40),("observer_script_sha256",64),("dataset_sha256",64),("validation_artifact_sha256",64)): require(type(m[k]) is str and len(m[k])==n and all(c in "0123456789abcdef" for c in m[k]), "manifest SHA")
    require(all(x not in json.dumps(m) for x in ("timestamp","hostname","username","uuid")), "volatile manifest"); return m

def render_report(manifest: Mapping[str,Any], distances: Sequence[Mapping[str,Any]], summary: Mapping[str,Any]) -> bytes:
    lines=["# O0b Token-Aligned Native Hidden-State Proxy Screening","","## Provenance","",f"- Run name: {manifest['run_name']}",f"- Observer implementation commit: {manifest['observer_implementation_commit']}","","## Integrity Checks","",f"- Pre-divergence all pass: {'true' if summary['integrity']['pre_divergence_all_pass'] else 'false'}","","## Pair-Level Distances","","| pair_id | comparison_id | anchor_name | layer_index | normalized_l2_distance | cosine_distance | status |","|---|---|---|---:|---:|---:|---|"]
    for r in distances: lines.append(f"| {r['pair_id']} | {r['comparison_id']} | {r['anchor_name']} | {r['layer_index']} | {format(r['normalized_l2_distance'],'.17g')} | {format(r['cosine_distance'],'.17g')} | {r['pre_divergence_integrity_status']} |")
    lines += ["","## Descriptive Summaries","","| anchor_name | layer_index | a_mean | b_mean | c_mean | a_gt_b_count | a_gt_c_count |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["aggregates"]:
        f=lambda x: "NA" if x is None else format(x,".17g"); lines.append(f"| {r['anchor_name']} | {r['layer_index']} | {f(r['a_mean'])} | {f(r['b_mean'])} | {f(r['c_mean'])} | {r['a_gt_b_count']} | {r['a_gt_c_count']} |")
    lines += ["","## Scientific Boundary","","This artifact is a descriptive hidden-state observation bundle; it does not establish causal sufficiency or a deployment decision.",""]
    return "\n".join(lines).encode("utf-8")

class FileSystemAdapter:
    def exists(self,p): return Path(p).exists()
    def mkdir(self,p): Path(p).mkdir()
    def write_bytes(self,p,b): Path(p).write_bytes(b)
    def read_bytes(self,p): return Path(p).read_bytes()
    def list_dir(self,p): return [x.name for x in Path(p).iterdir()]
    def rename(self,a,b): os.rename(a,b)

def _parse_npz(data: bytes) -> np.ndarray:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            require(z.namelist()==["vectors.npy"], "NPZ members"); info=z.infolist()[0]; require(info.date_time==(1980,1,1,0,0,0) and info.compress_type==zipfile.ZIP_STORED and info.create_system==3 and info.create_version==20 and info.extract_version==20 and info.external_attr==0o100644<<16 and info.internal_attr==0 and info.extra==b"" and info.comment==b"" and info.flag_bits==0, "NPZ metadata")
            raw=z.read("vectors.npy")
        require(raw[:6]==b"\x93NUMPY" and raw[6:8]==b"\x01\x00" and b"descr': '<f4'" in raw[:256] and b"fortran_order': False" in raw[:256] and b"object" not in raw[:256], "NPY header")
        loaded=np.load(io.BytesIO(data), allow_pickle=False); require(loaded.files==["vectors"], "NPZ array member"); arr=loaded["vectors"]
    except (ValueError, OSError, zipfile.BadZipFile, KeyError) as e: raise ContractError("invalid NPZ") from e
    require(isinstance(arr,np.ndarray) and arr.dtype==np.dtype("<f4") and arr.ndim==2 and arr.shape[0]>0 and arr.flags.c_contiguous and np.isfinite(arr).all(), "NPZ array")
    return np.array(arr, dtype=np.dtype("<f4"), copy=False)

def validate_publication(fs: Any, staging: Any, files: Mapping[str,bytes], coordinates: Mapping[str,Any], descriptors: Sequence[Mapping[str,Any]]) -> None:
    require(set(fs.list_dir(staging))==set(REQUIRED_ARTIFACTS), "published set")
    actual={n:fs.read_bytes(staging/n) for n in REQUIRED_ARTIFACTS}; parse_checksums(actual["SHA256SUMS.txt"],{n:actual[n] for n in REQUIRED_ARTIFACTS[:-1]})
    try: manifest=json.loads(actual["manifest.json"]); obs=[json.loads(x) for x in actual["anchor_observations.jsonl"].decode("utf-8").splitlines()]; dist=[json.loads(x) for x in actual["paired_distances.jsonl"].decode("utf-8").splitlines()]; summary=json.loads(actual["summary.json"])
    except (UnicodeDecodeError,json.JSONDecodeError) as e: raise ContractError("malformed JSON artifact") from e
    require(canonical_json(manifest)==actual["manifest.json"], "manifest canonical"); require(manifest["layer_descriptors"]==[dict(x) for x in descriptors], "manifest layers"); build_manifest(manifest)
    vectors=_parse_npz(actual["anchor_hidden_states.npz"]); validate_observation_records(obs,coordinates,descriptors,vectors.shape[0]); validate_distance_records(dist,coordinates,descriptors,vectors.shape[0])
    obs_by_key={(r["pair_id"],r["comparison_id"],r["anchor_name"],r["layer_index"],r["vector_role"]):r for r in obs}
    for r in dist:
        base=(r["pair_id"],r["comparison_id"],r["anchor_name"],r["layer_index"])
        require(base+("reference",) in obs_by_key and base+("member",) in obs_by_key, "missing observation linkage"); require(obs_by_key[base+("reference",)]["vector_index"]==r["reference_vector_index"], "reference vector correspondence")
        member=obs_by_key[base+("member",)]
        require(member["vector_index"]==r["member_vector_index"], "member vector correspondence")
    validate_summary(summary,summary["integrity"]["pair_ids"],descriptors,dist); require(canonical_jsonl(obs)==actual["anchor_observations.jsonl"] and canonical_jsonl(dist)==actual["paired_distances.jsonl"] and canonical_json(summary)==actual["summary.json"], "canonical artifacts")
    used=set();
    for r in obs: used.add(r["vector_index"])
    require(used==set(range(vectors.shape[0])), "unreferenced vector row")
    for r in dist:
        rv=vectors[r["reference_vector_index"]]; mv=vectors[r["member_vector_index"]]; fresh=distance(mv,rv)
        require(all(r[k]==fresh[k] for k in fresh), "distance does not match NPZ")
        if r["anchor_name"]=="anchor_pre_minus_1": require(np.allclose(rv,mv,rtol=0.0,atol=PRE_DIVERGENCE_ATOL), "published pre-divergence vectors")
    require(render_report(manifest,dist,summary)==actual["report.md"], "report bytes")

def build_artifact_bundle(manifest: Mapping[str,Any], assembled: Mapping[str,Any]) -> dict[str,bytes]:
    m=build_manifest(manifest); summary=build_summary(assembled["distances"],assembled["pair_ids"],assembled["layer_descriptors"]); validate_summary(summary,assembled["pair_ids"],assembled["layer_descriptors"]); obs=canonical_jsonl(assembled["observations"]); dist=canonical_jsonl(assembled["distances"]); six={"manifest.json":canonical_json(m),"anchor_observations.jsonl":obs,"anchor_hidden_states.npz":deterministic_npz(assembled["vectors"]),"paired_distances.jsonl":dist,"summary.json":canonical_json(summary)}; six["report.md"]=render_report(m,assembled["distances"],summary); six["SHA256SUMS.txt"]=checksum_text(six); return six

def publish_bundle(output_dir: Any, files: Mapping[str,bytes], coordinates: Mapping[str,Any], descriptors: Sequence[Mapping[str,Any]], fs: Any | None=None) -> None:
    fs=fs or FileSystemAdapter(); staging=output_dir.with_name(output_dir.name+".tmp"); require(not fs.exists(output_dir) and not fs.exists(staging), "output/staging collision"); require(set(files)==set(REQUIRED_ARTIFACTS), "bundle names"); fs.mkdir(staging)
    for n in REQUIRED_ARTIFACTS: fs.write_bytes(staging/n,files[n])
    validate_publication(fs,staging,files,coordinates,descriptors); require(not fs.exists(output_dir), "output appeared"); fs.rename(staging,output_dir)

# Stable descriptive aliases for future execution-authority callers.
validate_anchor_observations = validate_observation_records
validate_paired_distances = validate_distance_records
validate_summary_schema = validate_summary
build_complete_artifact_bundle = build_artifact_bundle
capture_exact_command = actual_command

def publish(output_dir: Path, files: Mapping[str, bytes], coordinates: Mapping[str,Any] | None=None, descriptors: Sequence[Mapping[str,Any]] | None=None, fs: Any | None=None) -> None:
    require(coordinates is not None and descriptors is not None, "authoritative publication inputs required")
    publish_bundle(output_dir, files, coordinates, descriptors, fs)

def parse_args(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--run-name", required=True); p.add_argument("--exact-command", required=True); p.add_argument("--observer-implementation-commit", required=True); p.add_argument("--observer-script-sha256", required=True); return p.parse_args(argv)

def run_observer(args: argparse.Namespace, dependencies: Mapping[str, Any] | None=None) -> dict[str, Any]:
    d=dict(dependencies or {}); root=Path(d.get("root",Path(__file__).resolve().parents[1])); data=Path(d.get("dataset_path",root/DATASET_PATH)); art=Path(d.get("artifact_path",root/VALIDATION_ARTIFACT_PATH))
    expected_head=d.get("repository_head"); prov=verify_runtime_provenance(args.observer_implementation_commit,args.observer_script_sha256,Path(__file__),args.run_name,args.exact_command,argv=d.get("argv"),actual_head=expected_head)
    db=data.read_bytes(); require(sha256_bytes(db)==DATASET_SHA256, "dataset SHA256"); rows=read_jsonl(data); validate_sources(rows)
    ab=art.read_bytes(); require(sha256_bytes(ab)==VALIDATION_ARTIFACT_SHA256, "artifact SHA256"); artifact=json.loads(ab); validate_artifact(artifact, DATASET_SHA256)
    tok=(d.get("tokenizer_loader") or load_tokenizer)(); tokenized=validate_tokenized_members(rows,artifact,tok)
    tokens={(x["pair"]["pair_id"],x["condition"]):x["ids"] for x in tokenized}; model=(d.get("model_loader") or load_model)(); states={}; descriptors=None; forward_records=[]
    for x in tokenized:
        arrays, desc=run_native_forward(model,x["ids"]); states[(x["pair"]["pair_id"],x["condition"])] = arrays; descriptors=descriptors or desc; require(desc==descriptors, "layer layout mismatch"); forward_records.append((x["pair"]["pair_id"],x["condition"]))
    require(len(forward_records)==12 and len(set(forward_records))==12, "exactly twelve full forwards")
    assembled=assemble_observations(artifact,states,descriptors,tokens)
    versions=d.get("runtime_versions") or capture_runtime_versions()
    fields={"schema_version":SCHEMA_VERSION,"experiment_name":"longterm_o0b_token_aligned_native_mamba_state_dynamics","scientific_design_authority_commit":SCIENTIFIC_DESIGN_AUTHORITY_COMMIT,"boundary_recovery_authority_commit":BOUNDARY_RECOVERY_AUTHORITY_COMMIT,"input_implementation_freeze_commit":INPUT_IMPLEMENTATION_FREEZE_COMMIT,"observer_implementation_authority_commit":OBSERVER_IMPLEMENTATION_AUTHORITY_COMMIT,**prov,"dataset_path":DATASET_PATH,"dataset_sha256":DATASET_SHA256,"validation_artifact_path":VALIDATION_ARTIFACT_PATH,"validation_artifact_sha256":VALIDATION_ARTIFACT_SHA256,"validation_artifact_repository_head":BOUNDARY_RECOVERY_AUTHORITY_COMMIT,"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"model_trust_remote_code":False,"tokenizer_id":TOKENIZER_ID,"tokenizer_revision":TOKENIZER_REVISION,"tokenizer_trust_remote_code":False,"tokenizer_use_fast":True,"add_special_tokens":False,"device":"cpu","dtype":"float32","python_version":versions.get("python_version","unknown"),"numpy_version":versions.get("numpy_version",np.__version__),"torch_version":versions.get("torch_version","unknown"),"transformers_version":versions.get("transformers_version","unknown"),"serialization_template":"canonical-json-v1/deterministic-npz-v1","comparison_order":list(COMPARISON_ORDER),"anchor_order":list(ANCHOR_ORDER),"layer_descriptors":descriptors,"pre_divergence_rtol":0.0,"pre_divergence_atol":1e-6,"cosine_redundancy_atol":1e-12,"run_name":args.run_name,"required_artifacts":list(REQUIRED_ARTIFACTS),"execution_status":"COMPLETE"}
    files=build_artifact_bundle(fields,assembled); fs=d.get("filesystem"); publish_bundle(args.output_dir,files,artifact,descriptors,fs); return {"files":files,"assembled":assembled,"forward_records":forward_records,"manifest":fields}

def main(argv=None, _test_dependencies=None):
    return run_observer(parse_args(argv),_test_dependencies)
if __name__ == "__main__": main()
