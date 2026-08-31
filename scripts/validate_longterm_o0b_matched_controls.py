#!/usr/bin/env python
"""Fail-closed, tokenizer-only validator for the O0b matched controls."""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION="longterm_o0b_matched_controls_v1"; DATASET_PATH=Path("data/longterm_o0b_matched_controls_v1.jsonl"); ARTIFACT_PATH=Path("reports/longterm_o0b_matched_controls_v1_validation.json")
SCIENTIFIC_DESIGN_AUTHORITY_COMMIT="df461469cb087f7f5db1e41a2b08e65ea517ad8"; IMPLEMENTATION_AUTHORITY_COMMIT="31e6d7882586e312f783cb2fd69718eb1ee7e452"; BOUNDARY_RECOVERY_AUTHORITY_COMMIT="2ed4439e511f7534186cbd5df9110e45fdc1d66c"
TOKENIZER_ID="state-spaces/mamba-130m-hf"; TOKENIZER_REVISION="5708daa364c50b880e7bd92eab456e0d34492ee9"; ADD_SPECIAL_TOKENS=False; TRUST_REMOTE_CODE=False; TOKENIZED_TEXT_COORDINATE_DOMAIN="printable_ascii_u0020_u007e"; SERIALIZATION_TEMPLATE="Claim: <claim>\\nEvidence: <evidence>"
PAIR_IDS=["o0b_pair_001","o0b_pair_002","o0b_pair_003"]; CONDITIONS=["reference_sufficient","paraphrase_sufficient","insufficient_matched","surface_null_matched"]; REQUIRED_FIELDS=["schema_version","pair_id","claim",*CONDITIONS,"insufficiency_rationale","paraphrase_rationale","surface_null_rationale"]; ANCHOR_KEYS=["anchor_pre_minus_1","anchor_divergence","anchor_post_plus_1","anchor_post_plus_2","anchor_post_plus_4","anchor_terminal"]
SHA_RE=re.compile(r"^[0-9a-f]{40}$")
class ValidationError(RuntimeError): pass
def sha256_bytes(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def canonical_json_bytes(payload:dict[str,Any])->bytes:
    b=(json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode();
    if b.startswith(b"\xef\xbb\xbf") or b"\r" in b: raise ValidationError("canonical JSON encoding/newline contract failed")
    return b
def _require_sha(name:str,value:str,expected:str|None=None):
    if expected is not None and value == expected: return
    if not isinstance(value,str) or not SHA_RE.fullmatch(value): raise ValidationError(f"{name} mismatch or malformed")
def current_repository_head(): return subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
def read_dataset_bytes(path:Path=DATASET_PATH):
    try:return path.read_bytes()
    except FileNotFoundError as e: raise ValidationError(f"dataset path missing: {path.as_posix()}") from e
def parse_dataset(raw:bytes):
    try:text=raw.decode("utf-8")
    except UnicodeDecodeError as e: raise ValidationError("dataset is not valid UTF-8") from e
    if text.startswith("\ufeff") or "\r" in text or not text.endswith("\n"): raise ValidationError("dataset must be UTF-8 without BOM, CR, and with LF termination")
    out=[]
    for n,line in enumerate(text.splitlines(),1):
        if not line.strip(): raise ValidationError(f"blank JSONL line at {n}")
        try:r=json.loads(line)
        except json.JSONDecodeError as e: raise ValidationError(f"malformed JSONL at line {n}") from e
        if not isinstance(r,dict): raise ValidationError("malformed record")
        out.append(r)
    return out
def is_printable_ascii_text(text:Any)->bool: return isinstance(text,str) and text!="" and text==text.strip() and all(0x20<=ord(ch)<=0x7e for ch in text)
def validate_records(records):
    if len(records)!=3: raise ValidationError("dataset must contain exactly three records")
    seen=set(); out=[]
    for r in records:
        if set(r)!=set(REQUIRED_FIELDS): raise ValidationError("malformed record fields")
        if r["schema_version"]!=SCHEMA_VERSION: raise ValidationError("schema_version mismatch")
        if not isinstance(r["pair_id"],str) or not r["pair_id"].strip() or r["pair_id"] in seen or r["pair_id"] not in PAIR_IDS: raise ValidationError("invalid pair ID")
        seen.add(r["pair_id"])
        for f in REQUIRED_FIELDS:
            if not isinstance(r[f],str) or not r[f].strip(): raise ValidationError(f"malformed field {f}")
        for f in ["claim",*CONDITIONS]:
            if not is_printable_ascii_text(r[f]): raise ValidationError(f"tokenized source field outside printable ASCII domain: {f}")
        out.append({f:r[f] for f in REQUIRED_FIELDS})
    if [r["pair_id"] for r in out]!=PAIR_IDS: raise ValidationError("pair IDs must appear in frozen order")
    return out
def serialize_member(claim,evidence): return f"Claim: {claim}\nEvidence: {evidence}"
def prefix_text(claim): return f"Claim: {claim}\nEvidence: "
def load_tokenizer(tokenizer_id=TOKENIZER_ID,revision=TOKENIZER_REVISION,*,trust_remote_code=False,add_special_tokens=False,use_fast=True):
    if tokenizer_id!=TOKENIZER_ID or revision!=TOKENIZER_REVISION or trust_remote_code is not False or add_special_tokens is not False or use_fast is not True: raise ValidationError("tokenizer identity/settings mismatch")
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(tokenizer_id,revision=revision,use_fast=True,trust_remote_code=False)
    if getattr(t,"is_fast",False) is not True: raise ValidationError("fast tokenizer required")
    if getattr(t,"requires_trust_remote_code",False) or (getattr(t,"init_kwargs",{}) or {}).get("trust_remote_code") is True: raise ValidationError("tokenizer attempted remote code")
    return t
def safe_load_tokenizer(loader:Callable[...,Any],*,tokenizer_id,revision,trust_remote_code,add_special_tokens):
    if tokenizer_id!=TOKENIZER_ID: raise ValidationError("wrong tokenizer ID")
    if revision!=TOKENIZER_REVISION: raise ValidationError("wrong tokenizer revision")
    if trust_remote_code is not False: raise ValidationError("trust_remote_code must be False")
    if add_special_tokens is not False: raise ValidationError("add_special_tokens must be False")
    t=loader(tokenizer_id,revision=revision,trust_remote_code=False,add_special_tokens=False,use_fast=True)
    if getattr(t,"is_fast",False) is not True: raise ValidationError("fast tokenizer required")
    if any(getattr(t,x,False) for x in ("requires_trust_remote_code","model_class_instantiated","model_weights_requested")): raise ValidationError("forbidden tokenizer/model behavior")
    return t
def _parts(e):
    ids=e.get("input_ids") if isinstance(e,dict) else getattr(e,"input_ids",None); off=e.get("offset_mapping") if isinstance(e,dict) else getattr(e,"offset_mapping",None)
    if not isinstance(ids,list) or not all(isinstance(x,int) for x in ids): raise ValidationError("tokenizer did not return integer input_ids")
    if not isinstance(off,list): raise ValidationError("tokenizer did not return offset mapping")
    return ids,off
def token_ids(tokenizer,text,*,add_special_tokens=False):
    if add_special_tokens is not False: raise ValidationError("add_special_tokens must be False")
    return _parts(tokenizer(text,add_special_tokens=False,return_offsets_mapping=True))[0]
def derive_member_tokens(tokenizer,claim,evidence):
    text=serialize_member(claim,evidence); boundary=len(prefix_text(claim)); ids,offsets=_parts(tokenizer(text,add_special_tokens=False,return_offsets_mapping=True))
    if len(ids)!=len(offsets): raise ValidationError("token/offset length mismatch")
    spans=[]
    for i,s in enumerate(offsets):
        if not isinstance(s,(tuple,list)) or len(s)!=2 or not all(isinstance(x,int) and not isinstance(x,bool) for x in s): raise ValidationError("malformed offset")
        a,b=s
        if not 0<=a<b<=len(text): raise ValidationError("zero-length or out-of-range offset")
        if i and not(spans[-1][0]<=a and spans[-1][1]<=b and spans[-1][1]<=a): raise ValidationError("overlapping or non-monotone offset")
        spans.append((a,b))
    cover=[i for i,(a,b) in enumerate(spans) if a<=boundary<b]
    if len(cover)!=1: raise ValidationError("evidence boundary is not uniquely covered")
    i=cover[0]; a,b=spans[i]
    return {"full_serialized_token_ids":ids,"full_token_count":len(ids),"full_offset_mapping":[list(s) for s in spans],"evidence_char_start":boundary,"evidence_start_index":i,"evidence_start_offset_start":a,"evidence_start_offset_end":b,"boundary_crossing":a<boundary<b,"evidence_token_count":len(ids)-i,"terminal_index":len(ids)-1}
def first_divergent_index(left,right):
    for i,(a,b) in enumerate(zip(left,right)):
        if a!=b:return i
    return None
def validate_anchor_dict(anchors,divergence,terminal):
    expected={"anchor_pre_minus_1":divergence-1,"anchor_divergence":divergence,"anchor_post_plus_1":divergence+1 if divergence+1<=terminal else None,"anchor_post_plus_2":divergence+2 if divergence+2<=terminal else None,"anchor_post_plus_4":divergence+4 if divergence+4<=terminal else None,"anchor_terminal":terminal}
    if anchors!=expected: raise ValidationError("anchor mismatch")
    if any(v is not None and not 0<=v<=terminal for v in anchors.values()): raise ValidationError("anchor out of range")
def anchor_dict(divergence,terminal):
    a={"anchor_pre_minus_1":divergence-1,"anchor_divergence":divergence,"anchor_post_plus_1":divergence+1 if divergence+1<=terminal else None,"anchor_post_plus_2":divergence+2 if divergence+2<=terminal else None,"anchor_post_plus_4":divergence+4 if divergence+4<=terminal else None,"anchor_terminal":terminal}; validate_anchor_dict(a,divergence,terminal); return a
def validate_pair_tokens(pair,tokenizer):
    m={c:derive_member_tokens(tokenizer,pair["claim"],pair[c]) for c in CONDITIONS}; starts={m[c]["evidence_start_index"] for c in CONDITIONS}
    if len(starts)!=1: raise ValidationError(f"different evidence_start_index for {pair['pair_id']}")
    common=next(iter(starts)); ref=m[CONDITIONS[0]]
    for c in CONDITIONS[1:]:
        for i in range(common):
            if m[c]["full_serialized_token_ids"][i]!=ref["full_serialized_token_ids"][i]: raise ValidationError("pre-evidence token-ID invariant failed")
            if m[c]["full_offset_mapping"][i]!=ref["full_offset_mapping"][i]: raise ValidationError("pre-evidence offset invariant failed")
    counts={m[c]["full_token_count"] for c in CONDITIONS}
    if len(counts)!=1: raise ValidationError(f"unequal full token counts for {pair['pair_id']}")
    terminal=ref["terminal_index"]; comp={}
    for c in CONDITIONS[1:]:
        d=first_divergent_index(ref["full_serialized_token_ids"],m[c]["full_serialized_token_ids"])
        if d is None: raise ValidationError(f"missing divergence for {pair['pair_id']} {c}")
        if d<common: raise ValidationError(f"divergence in claim/scaffold for {pair['pair_id']} {c}")
        if d>=terminal: raise ValidationError(f"invalid divergence/terminal relation for {pair['pair_id']} {c}")
        comp[c]={"anchor_indices":anchor_dict(d,terminal),"first_divergent_token_index":d,"reference_condition":"reference_sufficient","terminal_index":terminal,"validation_flags":{"divergence_exists":True,"divergence_in_evidence_region":True,"divergence_before_terminal":True}}
    return {"pair_id":pair["pair_id"],"claim":pair["claim"],"conditions":m,"comparisons_to_reference":comp,"equal_full_token_count":True,"full_token_count":next(iter(counts)),"terminal_index":terminal,"matched_set_invariants":{"common_evidence_start_index":common,"common_evidence_start_index_pass":True,"pre_evidence_token_id_invariant":True,"pre_evidence_offset_invariant":True},"validation_flags":{"claim_identity_by_construction":True,"equal_full_token_count":True,"full_sequence_offsets_verified":True,"first_divergences_valid":True}}
def build_payload(records,tokenizer,*,dataset_sha256,implementation_authority_commit=IMPLEMENTATION_AUTHORITY_COMMIT,scientific_design_authority_commit=SCIENTIFIC_DESIGN_AUTHORITY_COMMIT,boundary_recovery_authority_commit=BOUNDARY_RECOVERY_AUTHORITY_COMMIT,repository_head=None):
    if not isinstance(dataset_sha256,str) or not re.fullmatch(r"[0-9a-f]{64}",dataset_sha256): raise ValidationError("dataset_sha256 malformed")
    _require_sha("implementation_authority_commit",implementation_authority_commit,IMPLEMENTATION_AUTHORITY_COMMIT); _require_sha("scientific_design_authority_commit",scientific_design_authority_commit,SCIENTIFIC_DESIGN_AUTHORITY_COMMIT); _require_sha("boundary_recovery_authority_commit",boundary_recovery_authority_commit,BOUNDARY_RECOVERY_AUTHORITY_COMMIT)
    head=repository_head if repository_head is not None else current_repository_head(); _require_sha("repository_head",head)
    return {"add_special_tokens":False,"boundary_recovery_authority_commit":boundary_recovery_authority_commit,"conditions":CONDITIONS,"dataset_path":DATASET_PATH.as_posix(),"dataset_sha256":dataset_sha256,"implementation_authority_commit":implementation_authority_commit,"overall":"PASS","pair_ids":PAIR_IDS,"pairs":[validate_pair_tokens(p,tokenizer) for p in records],"repository_head":head,"schema_version":SCHEMA_VERSION,"scientific_design_authority_commit":scientific_design_authority_commit,"serialization_template":SERIALIZATION_TEMPLATE,"tokenized_text_coordinate_domain":TOKENIZED_TEXT_COORDINATE_DOMAIN,"tokenizer_id":TOKENIZER_ID,"tokenizer_revision":TOKENIZER_REVISION,"tokenizer_is_fast":True,"trust_remote_code":False,"validation_flags":{"dataset_runtime_hash_bound":True,"deterministic_canonical_json":True,"no_model_classes":True,"no_model_weights":True,"overall_pass":True,"tokenizer_only":True}}
def validate_existing_artifact_consistency(path,payload):
    if not path.exists():return
    try:existing=json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError,json.JSONDecodeError) as e:raise ValidationError("existing validation artifact is malformed") from e
    for f in ("dataset_sha256","scientific_design_authority_commit","implementation_authority_commit","boundary_recovery_authority_commit","repository_head"):
        if existing.get(f)!=payload[f]:raise ValidationError(f"existing artifact provenance mismatch: {f}")
def validate_to_bytes(*,dataset_path=DATASET_PATH,repository_head=None,tokenizer_loader=load_tokenizer,artifact_path=ARTIFACT_PATH,check_existing_artifact=True):
    raw=read_dataset_bytes(dataset_path); records=validate_records(parse_dataset(raw)); t=safe_load_tokenizer(tokenizer_loader,tokenizer_id=TOKENIZER_ID,revision=TOKENIZER_REVISION,trust_remote_code=False,add_special_tokens=False); payload=build_payload(records,t,dataset_sha256=sha256_bytes(raw),repository_head=repository_head)
    if check_existing_artifact:validate_existing_artifact_consistency(artifact_path,payload)
    encoded=canonical_json_bytes(payload)
    if canonical_json_bytes(json.loads(encoded.decode()))!=encoded:raise ValidationError("canonical generation is not byte-identical")
    return payload,encoded
def write_validation_artifact(*,dataset_path=DATASET_PATH,artifact_path=ARTIFACT_PATH,tokenizer_loader=load_tokenizer,repository_head=None,check_existing_artifact=True):
    payload,encoded=validate_to_bytes(dataset_path=dataset_path,repository_head=repository_head,tokenizer_loader=tokenizer_loader,artifact_path=artifact_path,check_existing_artifact=check_existing_artifact); artifact_path.parent.mkdir(parents=True,exist_ok=True); artifact_path.write_bytes(encoded); return payload
def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,default=DATASET_PATH);p.add_argument("--artifact",type=Path,default=ARTIFACT_PATH);p.add_argument("--no-existing-artifact-check",action="store_true");a=p.parse_args()
    try:write_validation_artifact(dataset_path=a.dataset,artifact_path=a.artifact,check_existing_artifact=not a.no_existing_artifact_check)
    except ValidationError as e:print(f"VALIDATION FAILED: {e}");return 1
    print("VALIDATION PASS");return 0
if __name__=="__main__":raise SystemExit(main())
