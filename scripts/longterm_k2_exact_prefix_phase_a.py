"""Frozen K2 Phase-A literal-prefix observer screen (no native-state code)."""
from __future__ import annotations
import argparse, gc, hashlib, io, json, platform, re, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

K2_PREREG_AUTHORITY_COMMIT="f2543949a23749f9a1119f88b06105d217a6172a"; SOURCE_BLOB_COMMIT="8eb7386e0344117d026c0e6ab172018bb98a698e"; A0_COMMIT="55debe94f0d19d16a334395e8561901fed6b52fa"
SOURCE_REL="reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"; SOURCE_PHYSICAL="eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"; SOURCE_SEMANTIC="3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
HF_MODEL="state-spaces/mamba-130m-hf"; HF_REVISION="5708daa364c50b880e7bd92eab456e0d34492ee9"; ENCODER_CANONICAL_DIGEST="48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"; ENCODER_RAW_CONCAT_DIGEST="968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"; ENCODER_TENSOR_COUNT=242; ENCODER_TOTAL_NUMEL=129135360; ENCODER_TOTAL_RAW_BYTES=516541440; CLASS_ORDER=("REFUTE","NOT_ENTITLED","SUPPORT")
OBSERVER_INPUT_CONTRACT="K2_PROSPECTIVE_LITERAL_PREFIX_WITH_FROZEN_A0_HEADS"; HISTORICAL_A0_SERIALIZATION_EQUIVALENCE_CLAIMED=False; HANDOFF_SCHEMA="contramamba-handoff-v3"; HANDOFF_JSON_MAX_BYTES=2*1024*1024
CANDIDATE_SCHEMA="k2ep-candidate-pool-v1"; SCREEN_SCHEMA="k2ep-screening-v1"; ELIGIBLE_SCHEMA="k2ep-eligible-ids-v1"; FINAL_SCHEMA="k2ep-final-ids-v1"; MANIFEST_SCHEMA="k2ep-phase-a-manifest-v1"
EXPECTED_ZIP_SHA256={"seed180":"96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861","seed181":"cd0070658d0df99675e4310dd9fc994f59985160849ef5e2be6dab9ff7c10596","seed182":"af5c720c7e8df99780b68414ce13d2416741efe573d48b05faf79427ab3caf7d"}; EXPECTED_CHECKPOINT_SHA256={"seed180":"4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c","seed181":"e9af8ca6e5c1ed62b8ec3014401d79ca0f8ab869a88db4ee7f18685a46f78e00","seed182":"a8c1b03095ec75d5e28eca9d8dc1dd83af309c93828ff4f12453fcdf36c61b63"}
FIELDS=("id","pair_id","claim","evidence","final_label","frame_compatible_label","predicate_covered_label","sufficiency_label","polarity_label","primary_failure_type","intervention_type"); ALLOWED_UNTRACKED={"scripts/longterm_k1_native_state_kinematics.py","tests/test_longterm_k1_native_state_kinematics.py"}; ALLOWED_TRACKED_WIP={"scripts/longterm_k2_exact_prefix_phase_a.py","tests/test_longterm_k2_exact_prefix_phase_a.py"}
def canonical_json(x:Any)->bytes:return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def canonical_jsonl(rows:list[dict[str,Any]])->bytes:return b"".join(canonical_json(r)+b"\n" for r in rows)
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def file_sha(p:Path)->str:return sha(p.read_bytes())
def semantic_sha(rows:list[dict[str,Any]])->str:
 for r in rows:
  if any(k not in r for k in FIELDS):raise ValueError("SOURCE_SEMANTIC_FIELD_MISSING")
  if any(type(r[k]) is not int or r[k] not in(0,1) for k in("frame_compatible_label","predicate_covered_label","sufficiency_label")):raise ValueError("SOURCE_EXACT_BINARY_INVALID")
 return sha(canonical_json([{k:r[k] for k in FIELDS} for r in rows]))
def require_frozen_source_path(root:Path,supplied:Path)->Path:
 actual=supplied.resolve(); expected=(root/SOURCE_REL).resolve()
 if actual!=expected:raise ValueError("SOURCE_FROZEN_PATH_MISMATCH")
 return actual
def load_source(root:Path,path:Path)->list[dict[str,Any]]:
 try:raw=subprocess.check_output(["git","show",f"{SOURCE_BLOB_COMMIT}:{SOURCE_REL}"],cwd=root)
 except (OSError,subprocess.CalledProcessError) as e:raise ValueError("SOURCE_GIT_SHOW_FAILED") from e
 if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):raise ValueError("SOURCE_PHYSICAL_JSONL_FORMAT_INVALID")
 if sha(raw)!=SOURCE_PHYSICAL:raise ValueError("SOURCE_PHYSICAL_SHA256_MISMATCH")
 lines=raw[:-1].split(b"\n")
 if not lines or any(not x for x in lines):raise ValueError("SOURCE_PHYSICAL_JSONL_FORMAT_INVALID")
 try:rows=[json.loads(x.decode("utf-8","strict")) for x in lines]
 except (UnicodeDecodeError,json.JSONDecodeError) as e:raise ValueError("SOURCE_PHYSICAL_JSONL_FORMAT_INVALID") from e
 if any(not isinstance(x,dict) for x in rows):raise ValueError("SOURCE_PHYSICAL_JSONL_FORMAT_INVALID")
 if semantic_sha(rows)!=SOURCE_SEMANTIC:raise ValueError("SOURCE_SEMANTIC_SHA256_MISMATCH")
 return rows
def _fail(pair:str,label:str)->dict[str,Any]:return {"pair_id":pair,"construction_status":"invalid","construction_failure_label":label}
def recipe(pair:str,rows:list[dict[str,Any]])->dict[str,Any]:
 ts=[r for r in rows if r.get("intervention_type")=="evidence_truncation"]; es=[r for r in rows if r.get("intervention_type")=="entity_swap"]; qs=[r for r in rows if r.get("intervention_type")=="polarity_flip" and r.get("final_label")=="REFUTE"] or [r for r in rows if r.get("intervention_type")=="none" and r.get("final_label")=="REFUTE"]
 if len(ts)!=1 or len(es)!=1 or len(qs)!=1:return _fail(pair,"INVALID_SOURCE_MULTIPLICITY")
 t,e,q=ts[0],es[0],qs[0]
 if any(not isinstance(r.get("claim"),str) for r in(t,e,q)) or len({r["claim"].encode() for r in(t,e,q)})!=1:return _fail(pair,"INVALID_SOURCE_CLAIM_IDENTITY")
 if not(t.get("final_label")=="NOT_ENTITLED" and t.get("primary_failure_type")=="sufficiency" and t.get("sufficiency_label")==0):return _fail(pair,"INVALID_TRUNCATION_SEMANTICS")
 if not(e.get("final_label")=="NOT_ENTITLED" and e.get("primary_failure_type")=="frame" and e.get("polarity_label")=="NONE" and e.get("evidence","").encode()!=t.get("evidence","").encode()):return _fail(pair,"INVALID_CONTROL_SEMANTICS")
 if not(q.get("final_label")=="REFUTE" and q.get("polarity_label")=="REFUTE"):return _fail(pair,"INVALID_CORRECTION_SEMANTICS")
 core={"schema_version":CANDIDATE_SCHEMA,"pair_id":pair,"truncation_source_id":t["id"],"refute_source_id":q["id"],"control_source_id":e["id"],"prefix_text":"Claim: "+t["claim"]+"\nEvidence: "+t["evidence"]+"\nAdditional evidence:","correction_text":" This is false: "+q["evidence"],"control_text":" A separate event: "+e["evidence"]}
 return {**core,"stable_item_id":"k2ep-v1:"+sha(canonical_json(core)),"base_claim_sha256":sha(t["claim"].encode()),"source_dataset_physical_sha256":SOURCE_PHYSICAL,"source_dataset_semantic_sha256":SOURCE_SEMANTIC,"construction_status":"valid","construction_failure_label":None}
def _full_invalid(x:dict[str,Any])->dict[str,Any]:
 if x["construction_status"]=="valid":return x
 x.update({"schema_version":CANDIDATE_SCHEMA,"stable_item_id":"k2ep-invalid-v1:"+sha(canonical_json({"pair_id":x["pair_id"],"failure":x["construction_failure_label"]})),"base_claim_sha256":None,"truncation_source_id":None,"refute_source_id":None,"control_source_id":None,"prefix_text":None,"correction_text":None,"control_text":None,"source_dataset_physical_sha256":SOURCE_PHYSICAL,"source_dataset_semantic_sha256":SOURCE_SEMANTIC});return x
def construct(rows:list[dict[str,Any]])->list[dict[str,Any]]:
 groups={}
 for r in rows:groups.setdefault(r.get("pair_id"),[]).append(r)
 return sorted((_full_invalid(recipe(p,rs)) for p,rs in groups.items()),key=lambda x:x["stable_item_id"])
def _ids(e:Any)->list[int]:return list(e["input_ids"] if isinstance(e,Mapping) else e.input_ids)
def tokenize_prefix_bundle(tok:Any,prefix:str)->dict[str,Any]:
 if not getattr(tok,"is_fast",False):raise ValueError("TOKENIZER_OFFSET_MAPPING_REQUIRED")
 enc=tok(prefix,add_special_tokens=False,return_offsets_mapping=True); ids=_ids(enc); offsets=[tuple(x) for x in enc["offset_mapping"]]; cs=len("Claim: "); ce=prefix.index("\nEvidence: "); es=ce+len("\nEvidence: "); ee=prefix.index("\nAdditional evidence:"); cm=[a<ce and b>cs for a,b in offsets]; em=[a<ee and b>es for a,b in offsets]
 if not ids or len(ids)!=len(offsets) or any(a>=b for a,b in offsets) or not any(cm) or not any(em) or any(a and b for a,b in zip(cm,em)):raise ValueError("PREFIX_MASK_INVALID")
 return {"input_ids":ids,"attention_mask":[1]*len(ids),"position_ids":list(range(len(ids))),"offset_mapping":offsets,"claim_mask":cm,"evidence_mask":em,"tau":len(ids)-1}
def prefix_model_inputs(prefix_bundle:dict[str,Any])->dict[str,Any]:
 """Return the already-tokenized P_i bundle; never tokenize P_i downstream."""
 return prefix_bundle
def _branch(tok:Any,text:str)->dict[str,list[int]]:
 ids=_ids(tok(text,add_special_tokens=False));return {"input_ids":ids,"attention_mask":[1]*len(ids),"position_ids":list(range(len(ids)))}
def token_contract(row:dict[str,Any],tok:Any,prefix_bundle:dict[str,Any]|None=None)->dict[str,Any]:
 p=prefix_bundle if prefix_bundle is not None else tokenize_prefix_bundle(tok,row["prefix_text"]); corr,ctrl=_branch(tok,row["prefix_text"]+row["correction_text"]),_branch(tok,row["prefix_text"]+row["control_text"]); tau=p["tau"]
 if tau<1 or any(b[key][:tau+1]!=p[key] for b in(corr,ctrl) for key in("input_ids","attention_mask","position_ids")):return {"ok":False,"failure":"INVALID_EXACT_PREFIX"}
 lc,ln=len(corr["input_ids"])-tau-1,len(ctrl["input_ids"])-tau-1
 if lc!=ln or not 8<=lc<=24 or corr["input_ids"][tau+1]==ctrl["input_ids"][tau+1]:return {"ok":False,"failure":"INVALID_CONTINUATION_TOKEN_CONTRACT"}
 return {"ok":True,"tau":tau,"continuation_length":lc,"prefix_bundle":p,"corr":corr,"ctrl":ctrl}
def prepare_candidate_pool(rows:list[dict[str,Any]],tok:Any)->tuple[list[dict[str,Any]],dict[str,dict[str,Any]]]:
 pool,bundles=construct(rows),{}
 for x in pool:
  if x["construction_status"]=="valid":
   check=token_contract(x,tok,tokenize_prefix_bundle(tok,x["prefix_text"]))
   if not check["ok"]:x.update(construction_status="invalid",construction_failure_label=check["failure"])
   else:bundles[x["stable_item_id"]]=check
 buckets={}
 for x in pool:
  if x["construction_status"]=="valid":buckets.setdefault(x["base_claim_sha256"],[]).append(x)
 for xs in buckets.values():
  for x in sorted(xs,key=lambda z:(z["pair_id"],z["stable_item_id"]))[1:]:x.update(construction_status="excluded",construction_failure_label="DUPLICATE_BASE_CLAIM_EXCLUDED");bundles.pop(x["stable_item_id"],None)
 return sorted(pool,key=lambda x:x["stable_item_id"]),bundles
def _safe_member(name:str)->str:
 if not name or "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:",name) or ":" in name:raise ValueError("HANDOFF_MEMBER_PATH_INVALID")
 parts=PurePosixPath(name).parts
 if not parts or any(not p or p in(".","..") for p in parts) or "/".join(parts)!=name:raise ValueError("HANDOFF_MEMBER_PATH_INVALID")
 return name
def _safe_zip_names(z:zipfile.ZipFile)->list[str]:
 names=[_safe_member(info.filename) for info in z.infolist()]
 if len(names)!=len(set(names)):raise ValueError("HANDOFF_MEMBER_DUPLICATE")
 return names
def discover_handoff_manifest(z:zipfile.ZipFile)->tuple[str,dict[str,Any]]:
 """Discover the authenticated handoff-v3 manifest by schema, never by name."""
 names=_safe_zip_names(z);found=[]
 for info,name in zip(z.infolist(),names):
  if not name.endswith(".json") or info.file_size>HANDOFF_JSON_MAX_BYTES:continue
  try:value=json.loads(z.read(info).decode("utf-8","strict"))
  except (UnicodeDecodeError,json.JSONDecodeError):continue
  if isinstance(value,dict) and value.get("schema")==HANDOFF_SCHEMA:found.append((name,value))
 if not found:raise ValueError("HANDOFF_MANIFEST_MISSING")
 if len(found)!=1:raise ValueError("HANDOFF_MANIFEST_AMBIGUOUS")
 return found[0]
def manifest_checkpoint_identity(manifest:Mapping[str,Any],seed:str)->dict[str,Any]:
 """Bind the frozen selected checkpoint through v3's authenticated files table."""
 if manifest.get("schema")!=HANDOFF_SCHEMA:raise ValueError("HANDOFF_SCHEMA_MISMATCH")
 if manifest.get("expected_commit")!=A0_COMMIT or manifest.get("actual_commit")!=A0_COMMIT:raise ValueError("HANDOFF_COMMIT_MISMATCH")
 files=manifest.get("files")
 if not isinstance(files,list):raise ValueError("HANDOFF_CHECKPOINT_MANIFEST_MISSING")
 selected=[]
 for record in files:
  if not isinstance(record,Mapping) or not isinstance(record.get("path"),str) or not isinstance(record.get("sha256"),str) or not isinstance(record.get("size_bytes"),int):raise ValueError("HANDOFF_CHECKPOINT_MANIFEST_MISSING")
  if record["sha256"]==EXPECTED_CHECKPOINT_SHA256[seed]:selected.append(dict(record))
 if len(selected)!=1:raise ValueError("HANDOFF_CHECKPOINT_MANIFEST_MISSING" if not selected else "HANDOFF_CHECKPOINT_AMBIGUOUS")
 return selected[0]
def audit_handoff(path:Path,seed:str,expected_zip_sha:str|None=None)->dict[str,Any]:
 if seed not in EXPECTED_ZIP_SHA256:raise ValueError("HANDOFF_SEED_INVALID")
 zsha=file_sha(path)
 if zsha!=(expected_zip_sha or EXPECTED_ZIP_SHA256[seed]):raise ValueError("HANDOFF_ZIP_SHA256_MISMATCH")
 with zipfile.ZipFile(path) as z:
  names=_safe_zip_names(z);manifest_member,m=discover_handoff_manifest(z);ident=manifest_checkpoint_identity(m,seed);cp=_safe_member("files/"+_safe_member(ident["path"]))
  if cp not in names:raise ValueError("HANDOFF_CHECKPOINT_MISSING")
  data=z.read(cp)
  if len(data)!=ident["size_bytes"] or sha(data)!=ident["sha256"] or ident["sha256"]!=EXPECTED_CHECKPOINT_SHA256[seed]:raise ValueError("HANDOFF_CHECKPOINT_IDENTITY_MISMATCH")
 return {"seed":seed,"zip_path":str(path.resolve()),"zip_sha256":zsha,"checkpoint_member":cp,"checkpoint_sha256":sha(data),"checkpoint_size_bytes":len(data),"manifest_member":manifest_member}
def load_checkpoint_cpu_bytes(data:bytes)->Any:
 import torch
 return torch.load(io.BytesIO(data),map_location="cpu",weights_only=True)
def load_authenticated_checkpoint(handoff:Mapping[str,Any])->Mapping[str,Any]:
 with zipfile.ZipFile(handoff["zip_path"]) as z:checkpoint=load_checkpoint_cpu_bytes(z.read(handoff["checkpoint_member"]))
 if not isinstance(checkpoint,Mapping) or checkpoint.get("schema_version")!="stage176a0_selected_checkpoint_v1" or not isinstance(checkpoint.get("model_state_dict"),Mapping) or not isinstance(checkpoint.get("metadata"),Mapping):raise ValueError("CHECKPOINT_SCHEMA_INVALID")
 return checkpoint
def strict_load(model:Any,state:Mapping[str,Any])->None:model.load_state_dict(state,strict=True)
def encoder_fingerprint(state:Mapping[str,Any])->dict[str,Any]:
 values={};raw=[];numel=0;dtypes=[]
 for key,value in sorted(state.items()):
  if key.startswith("mamba."):
   if not hasattr(value,"detach"):raise ValueError("MODEL_STATE_NOT_TENSORS")
   tensor=value.detach().cpu().contiguous();data=tensor.numpy().tobytes();values[key]=sha(data);raw.append(data);numel+=tensor.numel();dtypes.append(str(tensor.dtype))
 if not values:raise ValueError("MODEL_STATE_NOT_TENSORS")
 return {"algorithm":"mamba_state_per_tensor_sha256_canonical_json_v1","canonical_digest":sha(canonical_json(values)),"raw_concat_digest":sha(b"".join(raw)),"tensor_count":len(values),"total_numel":numel,"total_raw_bytes":sum(map(len,raw)),"dtypes":sorted(set(dtypes))}
def verify_encoder_fingerprint(state:Mapping[str,Any])->dict[str,Any]:
 fp=encoder_fingerprint(state)
 if (fp["canonical_digest"],fp["tensor_count"],fp["total_numel"],fp["total_raw_bytes"],fp["dtypes"]) != (ENCODER_CANONICAL_DIGEST,ENCODER_TENSOR_COUNT,ENCODER_TOTAL_NUMEL,ENCODER_TOTAL_RAW_BYTES,["torch.float32"]):raise ValueError("COMMON_ENCODER_CANONICAL_CONTRACT_MISMATCH")
 return fp
def resolve_hf_snapshot(revision:str)->tuple[Path,dict[str,Any]]:
 if revision!=HF_REVISION:raise ValueError("HF_REVISION_MISMATCH")
 from huggingface_hub import snapshot_download,__version__ as hv
 snapshot=Path(snapshot_download(repo_id=HF_MODEL,revision=HF_REVISION,allow_patterns=["config.json","tokenizer*","special_tokens_map.json","vocab.*","merges.txt"],local_files_only=False))
 if snapshot.name!=HF_REVISION:raise ValueError("HF_RESOLVED_REVISION_MISMATCH")
 from transformers import AutoConfig,AutoTokenizer,__version__ as tv
 config=AutoConfig.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False);tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True);files=[{"path":str(x.resolve()),"size_bytes":x.stat().st_size,"sha256":file_sha(x)} for x in sorted(snapshot.rglob("*")) if x.is_file() and(x.name=="config.json" or "tokenizer" in x.name or x.name in{"vocab.json","merges.txt","special_tokens_map.json"})]
 if not files:raise ValueError("HF_REQUIRED_FILE_MISSING")
 return snapshot,{"hf_model_id":HF_MODEL,"requested_hf_revision":HF_REVISION,"resolved_hf_revision":HF_REVISION,"resolved_snapshot_path":str(snapshot.resolve()),"tokenizer_class":type(tok).__name__,"tokenizer_is_fast":bool(getattr(tok,"is_fast",False)),"tokenizer_backend_class":type(getattr(tok,"backend_tokenizer",None)).__name__ if getattr(tok,"backend_tokenizer",None) is not None else None,"transformers_version":tv,"huggingface_hub_version":hv,"tokenizer_files":files,"tokenizer":tok,"config":config}
def normalized_a0_constructor_args(checkpoint:Mapping[str,Any],backbone:Any)->dict[str,Any]:
 metadata,state=checkpoint.get("metadata"),checkpoint.get("model_state_dict");args=metadata.get("training_args") if isinstance(metadata,Mapping) else None
 if not isinstance(args,Mapping) or not isinstance(state,Mapping) or any(not hasattr(v,"shape") for v in state.values()):raise ValueError("CHECKPOINT_SCHEMA_INVALID")
 names=set(state); flags={"use_boundary_head":"boundary_head.","use_frame_violation_head":"frame_violation_head.","use_predicate_isolation_head":"predicate_isolation_head.","use_preservation_entitlement_head":"preservation_entitlement_head.","use_temporal_diagnostic_head":"temporal_diagnostic_head.","use_temporal_residual_adapter":"temporal_residual_adapter.","use_temporal_channel":"temporal_channel_v1."}
 return {"backbone":backbone,"frame_size":128,"predicate_size":128,"sufficiency_size":128,"energy_size":64,"dropout":.1,"freeze_a_log":False,"decision_mode":"explicit_product","reason_router_epsilon":float(args.get("reason_router_epsilon",1e-8)),"use_temporal_comparator":"alpha_temporal_raw" in names,"use_predicate_comparator":"alpha_predicate_raw" in names,"alpha_temporal_init":1.25,"alpha_predicate_init":1.25,**{key:any(n.startswith(prefix) for n in names) for key,prefix in flags.items()}}
def build_a0_model(snapshot:Path,checkpoint:Mapping[str,Any])->Any:
 from transformers import MambaConfig,MambaModel
 root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/"src")]
 from contramamba.modeling_v6b_minimal import ContraMambaV6BMinimal
 state=checkpoint["model_state_dict"]
 config=MambaConfig.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False);model=ContraMambaV6BMinimal(**normalized_a0_constructor_args(checkpoint,MambaModel(config)));strict_load(model,state);model.eval();return model
def forward_prefix(model:Any,inputs:dict[str,Any],seed:str,checkpoint_sha:str,device:str)->dict[str,Any]:
 import torch
 ts={key:torch.tensor(value,dtype=torch.long,device=device).unsqueeze(0) for key,value in inputs.items() if key in{"input_ids","attention_mask","position_ids","claim_mask","evidence_mask"}}
 with torch.inference_mode():out=model(**ts)
 logits=out["logits"] if isinstance(out,Mapping) else out.logits;probs=torch.softmax(logits,dim=-1)[0].cpu().tolist();i=max(range(3),key=lambda x:probs[x]);ordered=sorted(probs,reverse=True);return {"seed":seed,"checkpoint_sha256":checkpoint_sha,"predicted_final_label":CLASS_ORDER[i],"probabilities":dict(zip(CLASS_ORDER,probs)),"confidence":probs[i],"margin":ordered[0]-ordered[1]}
def screen_one_seed(seed:str,handoff:Mapping[str,Any],snapshot:Path,screenable:Mapping[str,dict[str,Any]],device:str)->dict[str,dict[str,Any]]:
 checkpoint=load_authenticated_checkpoint(handoff);model=build_a0_model(snapshot,checkpoint)
 try:
  handoff["encoder"]=verify_encoder_fingerprint(checkpoint["model_state_dict"])
  return {item:forward_prefix(model,check["prefix_bundle"],seed,handoff["checkpoint_sha256"],device) for item,check in screenable.items()}
 finally:del model,checkpoint;gc.collect()
def synthetic_forward_sentinel(model:Any,tok:Any)->None:
 """Non-study plumbing check; it never receives controlled-source candidate text."""
 import torch
 bundle=tokenize_prefix_bundle(tok,"Claim: blorp\nEvidence: snarp\nAdditional evidence:")
 tensors={key:torch.tensor(value,dtype=torch.long).unsqueeze(0) for key,value in bundle.items() if key in{"input_ids","attention_mask","claim_mask","evidence_mask"}}
 with torch.inference_mode():out=model(**tensors)
 logits=out["logits"] if isinstance(out,Mapping) else out.logits
 if tuple(logits.shape)!=(1,3) or not torch.isfinite(logits).all().item():raise ValueError("SYNTHETIC_FORWARD_CONTRACT_MISMATCH")
def preflight_one_seed(seed:str,handoff:Mapping[str,Any],snapshot:Path,tok:Any,synthetic:bool=False)->dict[str,Any]:
 checkpoint=load_authenticated_checkpoint(handoff);model=build_a0_model(snapshot,checkpoint)
 try:
  fp=verify_encoder_fingerprint(checkpoint["model_state_dict"])
  if synthetic:synthetic_forward_sentinel(model,tok)
  return {**handoff,"encoder":fp,"strict_load":"PASS","metadata_training_args":"PASS","synthetic_forward":"PASS" if synthetic else "NOT_RUN"}
 finally:del model,checkpoint;gc.collect()
def join_screening(pool:list[dict[str,Any]],seed_outputs:Mapping[str,Mapping[str,dict[str,Any]]],checked:Mapping[str,dict[str,Any]])->list[dict[str,Any]]:
 out=[]
 for x in pool:
  r={"schema_version":SCREEN_SCHEMA,"stable_item_id":x["stable_item_id"],"construction_status":x["construction_status"],"construction_failure_label":x["construction_failure_label"],"prefix_gold_label":"NOT_ENTITLED","tau":None,"continuation_length":None,"head_outputs":[],"eligible":False}
  if x["construction_status"]=="valid":r.update(tau=checked[x["stable_item_id"]]["tau"],continuation_length=checked[x["stable_item_id"]]["continuation_length"]);r["head_outputs"]=[seed_outputs[s][x["stable_item_id"]] for s in EXPECTED_ZIP_SHA256];r["eligible"]=all(y["predicted_final_label"]=="SUPPORT" for y in r["head_outputs"])
  out.append(r)
 return out
def id_lists(screening:list[dict[str,Any]])->tuple[list[dict[str,Any]],list[dict[str,Any]],str]:
 e=sorted(({"schema_version":ELIGIBLE_SCHEMA,"stable_item_id":x["stable_item_id"]} for x in screening if x["eligible"]),key=lambda x:x["stable_item_id"])
 if len(e)<30:return e,[],"INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE"
 chosen=e if len(e)<=64 else sorted(e,key=lambda x:(sha(canonical_json(x)),x["stable_item_id"]))[:64];return e,[{"schema_version":FINAL_SCHEMA,"stable_item_id":x["stable_item_id"]} for x in chosen],"PHASE_B_ELIGIBLE"
def git_provenance(root:Path)->dict[str,Any]:
 status=subprocess.check_output(["git","status","--porcelain=v1"],cwd=root,text=True).splitlines();branch=subprocess.check_output(["git","branch","--show-current"],cwd=root,text=True).strip();head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
 if branch!="longterm-k-series-native-state-kinematics" or subprocess.call(["git","merge-base","--is-ancestor",K2_PREREG_AUTHORITY_COMMIT,head],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)!=0:raise ValueError("GIT_PROVENANCE_MISMATCH")
 for line in status:
  path=line[3:].replace("\\","/")
  if line[:2]=="??" and path in ALLOWED_UNTRACKED:continue
  if line[:2] in {" M","M "} and path in ALLOWED_TRACKED_WIP:continue
  raise ValueError("GIT_DIRTY_CONTRACT_MISMATCH")
 if subprocess.call(["git","ls-files","--error-unmatch","scripts/longterm_k2_exact_prefix_phase_a.py"],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)!=0:raise ValueError("K2_SCRIPT_NOT_TRACKED")
 return {"runtime_git_head":head,"runtime_branch":branch,"runtime_dirty_contract":status,"allowed_untracked_policy":sorted(ALLOWED_UNTRACKED)}
def require_safe_output_dir(root:Path,output:Path)->Path:
 target=output.resolve();base=root.resolve()
 if target==base or base in target.parents:raise ValueError("OUTPUT_DIRECTORY_PROTECTED")
 if target.exists() and any(target.iterdir()):raise ValueError("OUTPUT_DIRECTORY_NONEMPTY")
 return target
def package_provenance()->dict[str,str]:
 import torch,transformers,huggingface_hub,tokenizers
 return {"python_version":sys.version,"python_executable":sys.executable,"platform":platform.platform(),"torch_version":torch.__version__,"transformers_version":transformers.__version__,"huggingface_hub_version":huggingface_hub.__version__,"tokenizers_version":tokenizers.__version__}
def write_phase_a_outputs(output:Path,pool:list[dict[str,Any]],screening:list[dict[str,Any]],handoffs:dict[str,Any],hf:dict[str,Any],runtime:dict[str,Any])->dict[str,Any]:
 output=require_safe_output_dir(Path(__file__).resolve().parents[1],output);e,f,v=id_lists(screening);blobs={"candidate_pool.jsonl":canonical_jsonl(sorted(pool,key=lambda x:x["stable_item_id"])),"screening.jsonl":canonical_jsonl(sorted(screening,key=lambda x:x["stable_item_id"])),"eligible_ids.jsonl":canonical_jsonl(e),"final_confirmatory_ids.jsonl":canonical_jsonl(f)};output.mkdir(parents=True,exist_ok=True)
 for n,b in blobs.items():(output/n).write_bytes(b)
 m={"schema_version":MANIFEST_SCHEMA,"k2_prereg_authority_commit":K2_PREREG_AUTHORITY_COMMIT,"observer_input_contract":OBSERVER_INPUT_CONTRACT,"historical_a0_serialization_equivalence_claimed":False,"add_special_tokens":False,"trust_remote_code":False,"source_path":runtime["actual_source_path"],"frozen_source_rel":SOURCE_REL,"source_git_commit":SOURCE_BLOB_COMMIT,"source_git_path":SOURCE_REL,"source_byte_origin":"FROZEN_GIT_BLOB","source_physical_sha256":SOURCE_PHYSICAL,"source_semantic_sha256":SOURCE_SEMANTIC,"hf_model_id":HF_MODEL,"requested_hf_revision":HF_REVISION,"resolved_hf_revision":hf["resolved_hf_revision"],"resolved_snapshot_path":hf["resolved_snapshot_path"],"tokenizer_class":hf["tokenizer_class"],"tokenizer_is_fast":hf["tokenizer_is_fast"],"tokenizer_backend_class":hf["tokenizer_backend_class"],"tokenizer_files":hf["tokenizer_files"],"handoffs":handoffs,"common_encoder": {"algorithm":"mamba_state_per_tensor_sha256_canonical_json_v1","normative_canonical_digest":ENCODER_CANONICAL_DIGEST,"secondary_raw_concat_digest":ENCODER_RAW_CONCAT_DIGEST,"tensor_count":ENCODER_TENSOR_COUNT,"total_numel":ENCODER_TOTAL_NUMEL,"total_raw_bytes":ENCODER_TOTAL_RAW_BYTES,"dtype":"torch.float32"},"script_sha256":file_sha(Path(__file__)),"candidate_pool_sha256":sha(blobs["candidate_pool.jsonl"]),"screening_artifact_sha256":sha(blobs["screening.jsonl"]),"eligible_id_list_sha256":sha(blobs["eligible_ids.jsonl"]),"final_confirmatory_id_list_sha256":sha(blobs["final_confirmatory_ids.jsonl"]),"N_total_attempts":len(pool),"N_construction_valid":sum(x["construction_status"]=="valid" for x in pool),"N_duplicate_excluded":sum(x["construction_failure_label"]=="DUPLICATE_BASE_CLAIM_EXCLUDED" for x in pool),"N_eligible":len(e),"N_final":len(f),"phase_a_verdict":v,**runtime};(output/"phase_a_manifest.json").write_bytes(canonical_json(m)+b"\n");return m
def parser()->argparse.ArgumentParser:
 p=argparse.ArgumentParser();p.add_argument("--source-data");p.add_argument("--seed180-handoff",required=True);p.add_argument("--seed181-handoff",required=True);p.add_argument("--seed182-handoff",required=True);p.add_argument("--hf-revision",required=True);p.add_argument("--output-dir");p.add_argument("--device",default="cpu");p.add_argument("--integration-preflight",action="store_true");return p
def main(argv:list[str]|None=None)->int:
 a=parser().parse_args(argv);root=Path(__file__).resolve().parents[1]
 if a.hf_revision!=HF_REVISION:raise ValueError("HF_REVISION_MISMATCH")
 if a.device!="cpu":raise ValueError("CUDA_REQUIRES_FUTURE_AUTHORIZATION")
 runtime=git_provenance(root);snapshot,hf=resolve_hf_snapshot(a.hf_revision);handoffs={seed:audit_handoff(Path(getattr(a,seed+"_handoff")),seed) for seed in EXPECTED_ZIP_SHA256}
 if a.integration_preflight:
  # Validate the frozen blob only; deliberately do not construct or tokenize K2 candidates.
  load_source(root,Path(a.source_data) if a.source_data else root/SOURCE_REL)
  result={seed:preflight_one_seed(seed,handoffs[seed],snapshot,hf["tokenizer"],synthetic=seed=="seed180") for seed in EXPECTED_ZIP_SHA256}
  print(canonical_json({"integration_preflight":"PASS","handoffs":result,"hf_requested_revision":HF_REVISION,"hf_resolved_revision":hf["resolved_hf_revision"],"source_git_commit":SOURCE_BLOB_COMMIT,"k2_prereg_authority_commit":K2_PREREG_AUTHORITY_COMMIT}).decode())
  return 0
 if not a.source_data or not a.output_dir:raise ValueError("PHASE_A_SOURCE_AND_OUTPUT_REQUIRED")
 source=require_frozen_source_path(root,Path(a.source_data));pool,checked=prepare_candidate_pool(load_source(root,source),hf["tokenizer"]);outputs={}
 for seed in EXPECTED_ZIP_SHA256:outputs[seed]=screen_one_seed(seed,handoffs[seed],snapshot,checked,a.device)
 runtime.update(package_provenance(),actual_source_path=str(source));return (write_phase_a_outputs(Path(a.output_dir),pool,join_screening(pool,outputs,checked),handoffs,hf,runtime),0)[1]
if __name__=="__main__":raise SystemExit(main())
