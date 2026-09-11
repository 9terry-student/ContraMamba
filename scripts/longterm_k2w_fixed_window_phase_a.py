"""Frozen K2W Phase-A exact-prefix screen; deliberately contains no native-state code."""
from __future__ import annotations
import argparse, gc, hashlib, io, json, platform, re, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

K2W_PREREG_AUTHORITY_COMMIT="cc5386e730c333209eb070b14025c5368038e247"; K2_CLOSURE_COMMIT="386ef0763a0dd8470c22617581af196a324f222f"; SOURCE_BLOB_COMMIT="8eb7386e0344117d026c0e6ab172018bb98a698e"; A0_COMMIT="55debe94f0d19d16a334395e8561901fed6b52fa"
SOURCE_REL="reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"; SOURCE_PHYSICAL_SHA256="eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"; SOURCE_SEMANTIC_SHA256="3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
HF_MODEL="state-spaces/mamba-130m-hf"; HF_REVISION="5708daa364c50b880e7bd92eab456e0d34492ee9"; COMMON_ENCODER_CANONICAL_SHA256="48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"; COMMON_ENCODER_RAW_CONCAT_SHA256="968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"; COMMON_ENCODER_TENSOR_COUNT=242; COMMON_ENCODER_NUMEL=129135360; COMMON_ENCODER_RAW_BYTES=516541440
CLASS_ORDER=("REFUTE","NOT_ENTITLED","SUPPORT"); W=8; OBSERVER_INPUT_CONTRACT="K2_PROSPECTIVE_LITERAL_PREFIX_WITH_FROZEN_A0_HEADS"; HISTORICAL_A0_SERIALIZATION_EQUIVALENCE_CLAIMED=False
CANDIDATE_SCHEMA="k2w-candidate-pool-v1"; SCREENING_SCHEMA="k2w-screening-v1"; ELIGIBLE_SCHEMA="k2w-eligible-ids-v1"; FINAL_SCHEMA="k2w-final-ids-v1"; MANIFEST_SCHEMA="k2w-phase-a-manifest-v1"; HANDOFF_SCHEMA="contramamba-handoff-v3"; HANDOFF_JSON_MAX_BYTES=2*1024*1024
EXPECTED_ZIP_SHA256={"seed180":"96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861","seed181":"cd0070658d0df99675e4310dd9fc994f59985160849ef5e2be6dab9ff7c10596","seed182":"af5c720c7e8df99780b68414ce13d2416741efe573d48b05faf79427ab3caf7d"}; EXPECTED_CHECKPOINT_SHA256={"seed180":"4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c","seed181":"e9af8ca6e5c1ed62b8ec3014401d79ca0f8ab869a88db4ee7f18685a46f78e00","seed182":"a8c1b03095ec75d5e28eca9d8dc1dd83af309c93828ff4f12453fcdf36c61b63"}
FIELDS=("id","pair_id","claim","evidence","final_label","frame_compatible_label","predicate_covered_label","sufficiency_label","polarity_label","primary_failure_type","intervention_type"); HISTORICAL_K1_UNTRACKED={"scripts/longterm_k1_native_state_kinematics.py","tests/test_longterm_k1_native_state_kinematics.py"}; K2W_UNTRACKED={"scripts/longterm_k2w_fixed_window_phase_a.py","tests/test_longterm_k2w_fixed_window_phase_a.py"}

def canonical_json(x:Any)->bytes:return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def sha256(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def file_sha256(p:Path)->str:return sha256(p.read_bytes())
def canonical_jsonl(rows:list[dict[str,Any]])->bytes:return b"".join(canonical_json(x)+b"\n" for x in rows)
def semantic_sha256(rows:list[dict[str,Any]])->str:
 for r in rows:
  if any(k not in r for k in FIELDS):raise ValueError("SOURCE_SEMANTIC_FIELD_MISSING")
  if any(type(r[k]) is not int or r[k] not in (0,1) for k in ("frame_compatible_label","predicate_covered_label","sufficiency_label")):raise ValueError("SOURCE_EXACT_BINARY_INVALID")
 return sha256(canonical_json([{k:r[k] for k in FIELDS} for r in rows]))
def require_frozen_source_path(root:Path,supplied:Path)->Path:
 if supplied.resolve()!=(root/SOURCE_REL).resolve():raise ValueError("SOURCE_FROZEN_PATH_MISMATCH")
 return supplied.resolve()
def parse_frozen_jsonl(raw:bytes)->list[dict[str,Any]]:
 if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):raise ValueError("SOURCE_PHYSICAL_JSONL_FORMAT_INVALID")
 lines=raw[:-1].split(b"\n")
 if not lines or any(not line for line in lines):raise ValueError("SOURCE_PHYSICAL_JSONL_FORMAT_INVALID")
 try: rows=[json.loads(x.decode("utf-8","strict")) for x in lines]
 except (UnicodeDecodeError,json.JSONDecodeError) as e:raise ValueError("SOURCE_PHYSICAL_JSONL_FORMAT_INVALID") from e
 if any(not isinstance(x,dict) for x in rows):raise ValueError("SOURCE_PHYSICAL_JSONL_FORMAT_INVALID")
 return rows
def load_source(root:Path,supplied:Path)->list[dict[str,Any]]:
 require_frozen_source_path(root,supplied)
 try: raw=subprocess.check_output(["git","show",f"{SOURCE_BLOB_COMMIT}:{SOURCE_REL}"],cwd=root)
 except (OSError,subprocess.CalledProcessError) as e:raise ValueError("SOURCE_GIT_SHOW_FAILED") from e
 if sha256(raw)!=SOURCE_PHYSICAL_SHA256:raise ValueError("SOURCE_PHYSICAL_SHA256_MISMATCH")
 rows=parse_frozen_jsonl(raw)
 if semantic_sha256(rows)!=SOURCE_SEMANTIC_SHA256:raise ValueError("SOURCE_SEMANTIC_SHA256_MISMATCH")
 return rows
def _invalid(pair:str,label:str)->dict[str,Any]:return {"schema_version":CANDIDATE_SCHEMA,"pair_id":pair,"construction_status":"invalid","construction_failure_label":label,"stable_item_id":"k2w-invalid-v1:"+sha256(canonical_json({"pair_id":pair,"failure":label})),"base_claim_sha256":None,"truncation_source_id":None,"refute_source_id":None,"control_source_id":None,"prefix_text":None,"correction_text":None,"control_text":None,"source_dataset_physical_sha256":SOURCE_PHYSICAL_SHA256,"source_dataset_semantic_sha256":SOURCE_SEMANTIC_SHA256}
def recipe(pair:str,rows:list[dict[str,Any]])->dict[str,Any]:
 t=[x for x in rows if x.get("intervention_type")=="evidence_truncation"]; e=[x for x in rows if x.get("intervention_type")=="entity_swap"]; q=[x for x in rows if x.get("intervention_type")=="polarity_flip" and x.get("final_label")=="REFUTE"]
 if not q:q=[x for x in rows if x.get("intervention_type")=="none" and x.get("final_label")=="REFUTE"]
 if len(t)!=1 or len(e)!=1 or len(q)!=1:return _invalid(pair,"INVALID_SOURCE_MULTIPLICITY")
 t,e,q=t[0],e[0],q[0]
 if any(not isinstance(x.get("claim"),str) for x in(t,e,q)) or len({x["claim"].encode("utf-8") for x in(t,e,q)})!=1:return _invalid(pair,"INVALID_SOURCE_CLAIM_IDENTITY")
 if not(t.get("final_label")=="NOT_ENTITLED" and t.get("primary_failure_type")=="sufficiency" and t.get("sufficiency_label")==0):return _invalid(pair,"INVALID_TRUNCATION_SEMANTICS")
 if not(e.get("final_label")=="NOT_ENTITLED" and e.get("primary_failure_type")=="frame" and e.get("polarity_label")=="NONE" and isinstance(e.get("evidence"),str) and e["evidence"].encode()!=str(t.get("evidence","")).encode()):return _invalid(pair,"INVALID_CONTROL_SEMANTICS")
 if not(q.get("final_label")=="REFUTE" and q.get("polarity_label")=="REFUTE" and isinstance(q.get("evidence"),str) and isinstance(t.get("evidence"),str)):return _invalid(pair,"INVALID_CORRECTION_SEMANTICS")
 core={"schema_version":CANDIDATE_SCHEMA,"pair_id":pair,"truncation_source_id":t["id"],"refute_source_id":q["id"],"control_source_id":e["id"],"prefix_text":"Claim: "+t["claim"]+"\nEvidence: "+t["evidence"]+"\nAdditional evidence:\n","correction_text":q["evidence"],"control_text":e["evidence"],"source_dataset_physical_sha256":SOURCE_PHYSICAL_SHA256,"source_dataset_semantic_sha256":SOURCE_SEMANTIC_SHA256}
 return {**core,"stable_item_id":"k2w-v1:"+sha256(canonical_json(core)),"base_claim_sha256":sha256(t["claim"].encode("utf-8")),"construction_status":"valid","construction_failure_label":None,"p":None,"d":None,"tau":None,"tau_minus_p":None,"corr_post_tau_available":None,"ctrl_post_tau_available":None}
def construct(rows:list[dict[str,Any]])->list[dict[str,Any]]:
 groups={}
 for r in rows:groups.setdefault(r.get("pair_id"),[]).append(r)
 return sorted((recipe(str(k),v) for k,v in groups.items()),key=lambda x:x["stable_item_id"])
def _ids(x:Any)->list[int]:return list(x["input_ids"] if isinstance(x,Mapping) else x.input_ids)
def tokenize_prefix_bundle(tok:Any,text:str)->dict[str,Any]:
 if not getattr(tok,"is_fast",False):raise ValueError("TOKENIZER_OFFSET_MAPPING_REQUIRED")
 x=tok(text,add_special_tokens=False,return_offsets_mapping=True); ids=_ids(x); offsets=[tuple(y) for y in x["offset_mapping"]]; claim_start=len("Claim: "); claim_end=text.index("\nEvidence: "); evidence_start=claim_end+len("\nEvidence: "); evidence_end=text.index("\nAdditional evidence:\n")
 cm=[a<claim_end and b>claim_start for a,b in offsets]; em=[a<evidence_end and b>evidence_start for a,b in offsets]
 if not ids or len(ids)!=len(offsets) or any(a>=b for a,b in offsets) or not any(cm) or not any(em) or any(a and b for a,b in zip(cm,em)):raise ValueError("PREFIX_MASK_INVALID")
 return {"input_ids":ids,"attention_mask":[1]*len(ids),"position_ids":list(range(len(ids))),"offset_mapping":offsets,"claim_mask":cm,"evidence_mask":em}
def _branch(tok:Any,text:str)->dict[str,list[int]]:
 ids=_ids(tok(text,add_special_tokens=False));return {"input_ids":ids,"attention_mask":[1]*len(ids),"position_ids":list(range(len(ids)))}
def token_contract(row:dict[str,Any],tok:Any,prefix_bundle:dict[str,Any]|None=None)->dict[str,Any]:
 p=prefix_bundle or tokenize_prefix_bundle(tok,row["prefix_text"]); plen=len(p["input_ids"]); pidx=plen-1; corr=_branch(tok,row["prefix_text"]+row["correction_text"]); ctrl=_branch(tok,row["prefix_text"]+row["control_text"])
 if any(b[k][:plen]!=p[k] for b in(corr,ctrl) for k in ("input_ids","attention_mask","position_ids")):return {"ok":False,"failure":"INVALID_EXACT_PREFIX"}
 d=next((i for i in range(pidx+1,min(len(corr["input_ids"]),len(ctrl["input_ids"]))) if corr["input_ids"][i]!=ctrl["input_ids"][i]),None)
 if d is None:return {"ok":False,"failure":"INVALID_EVENT_DIVERGENCE"}
 tau=d-1
 if tau<1:return {"ok":False,"failure":"INVALID_PRESTATE"}
 if any(b[k][:tau+1]!=corr[k][:tau+1] for b in (ctrl,) for k in ("input_ids","attention_mask","position_ids")):return {"ok":False,"failure":"INVALID_EXACT_PREFIX"}
 ca=len(corr["input_ids"])-tau-1; na=len(ctrl["input_ids"])-tau-1
 if ca<W or na<W:return {"ok":False,"failure":"INVALID_WINDOW_AVAILABILITY"}
 return {"ok":True,"p":pidx,"d":d,"tau":tau,"tau_minus_p":tau-pidx,"corr_post_tau_available":ca,"ctrl_post_tau_available":na,"prefix_bundle":p,"corr":corr,"ctrl":ctrl}
def prepare_candidate_pool(rows:list[dict[str,Any]],tok:Any)->tuple[list[dict[str,Any]],dict[str,dict[str,Any]]]:
 pool=construct(rows); checked={}
 for x in pool:
  if x["construction_status"]=="valid":
   result=token_contract(x,tok,tokenize_prefix_bundle(tok,x["prefix_text"]))
   if result["ok"]:
    x.update({k:result[k] for k in ("p","d","tau","tau_minus_p","corr_post_tau_available","ctrl_post_tau_available")});checked[x["stable_item_id"]]=result
   else:x.update(construction_status="invalid",construction_failure_label=result["failure"])
 buckets={}
 for x in pool:
  if x["construction_status"]=="valid":buckets.setdefault(x["base_claim_sha256"],[]).append(x)
 for xs in buckets.values():
  for x in sorted(xs,key=lambda z:(z["pair_id"],z["stable_item_id"]))[1:]:x.update(construction_status="excluded",construction_failure_label="DUPLICATE_BASE_CLAIM_EXCLUDED");checked.pop(x["stable_item_id"],None)
 return sorted(pool,key=lambda x:x["stable_item_id"]),checked
def construction_summary(pool:list[dict[str,Any]],checked:Mapping[str,dict[str,Any]])->dict[str,Any]:
 def dist(k:str)->dict[str,int]:
  out={}
  for x in pool:
   if x.get(k) is not None:out[str(x[k])]=out.get(str(x[k]),0)+1
  return dict(sorted(out.items(),key=lambda x:int(x[0])))
 semantic_failures={"INVALID_SOURCE_MULTIPLICITY","INVALID_SOURCE_CLAIM_IDENTITY","INVALID_TRUNCATION_SEMANTICS","INVALID_CONTROL_SEMANTICS","INVALID_CORRECTION_SEMANTICS"}
 def passed(x:dict[str,Any],stage:str)->bool:
  failure=x["construction_failure_label"]
  if failure in semantic_failures:return False
  if stage=="semantic":return True
  if failure=="INVALID_EXACT_PREFIX":return False
  if stage=="exact":return True
  if failure=="INVALID_EVENT_DIVERGENCE":return False
  if stage=="divergence":return True
  return failure not in {"INVALID_PRESTATE","INVALID_WINDOW_AVAILABILITY"}
 return {"N_total_attempts":len(pool),"N_semantically_valid":sum(passed(x,"semantic") for x in pool),"N_exact_prefix_valid":sum(passed(x,"exact") for x in pool),"N_event_divergence_valid":sum(passed(x,"divergence") for x in pool),"N_window8_available":sum(passed(x,"window") for x in pool),"N_construction_valid":sum(passed(x,"window") for x in pool),"N_duplicate_excluded":sum(x["construction_status"]=="excluded" for x in pool),"tau_minus_p_distribution":dist("tau_minus_p"),"corr_post_tau_available_distribution":dist("corr_post_tau_available"),"ctrl_post_tau_available_distribution":dist("ctrl_post_tau_available")}

def _safe_member(name:str)->str:
 if not name or "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:",name) or ":" in name:raise ValueError("HANDOFF_MEMBER_PATH_INVALID")
 parts=PurePosixPath(name).parts
 if not parts or any(p in (".","..") for p in parts) or "/".join(parts)!=name:raise ValueError("HANDOFF_MEMBER_PATH_INVALID")
 return name
def _safe_zip_names(z:zipfile.ZipFile)->list[str]:
 names=[_safe_member(x.filename) for x in z.infolist()]
 if len(names)!=len(set(names)):raise ValueError("HANDOFF_MEMBER_DUPLICATE")
 return names
def discover_handoff_manifest(z:zipfile.ZipFile)->tuple[str,dict[str,Any]]:
 found=[]
 for i in z.infolist():
  name=_safe_member(i.filename)
  if name.endswith(".json") and i.file_size<=HANDOFF_JSON_MAX_BYTES:
   try:x=json.loads(z.read(i).decode("utf-8","strict"))
   except (UnicodeDecodeError,json.JSONDecodeError):continue
   if isinstance(x,dict) and x.get("schema")==HANDOFF_SCHEMA:found.append((name,x))
 if len(found)!=1:raise ValueError("HANDOFF_MANIFEST_MISSING" if not found else "HANDOFF_MANIFEST_AMBIGUOUS")
 return found[0]
def audit_handoff(path:Path,seed:str)->dict[str,Any]:
 if sha256(path.read_bytes())!=EXPECTED_ZIP_SHA256[seed]:raise ValueError("HANDOFF_ZIP_SHA256_MISMATCH")
 with zipfile.ZipFile(path) as z:
  names=_safe_zip_names(z); mm,m=discover_handoff_manifest(z)
  if m.get("expected_commit")!=A0_COMMIT or m.get("actual_commit")!=A0_COMMIT:raise ValueError("HANDOFF_COMMIT_MISMATCH")
  fs=m.get("files"); found=[x for x in fs if isinstance(x,Mapping) and x.get("sha256")==EXPECTED_CHECKPOINT_SHA256[seed]] if isinstance(fs,list) else []
  if len(found)!=1:raise ValueError("HANDOFF_CHECKPOINT_MANIFEST_MISSING" if not found else "HANDOFF_CHECKPOINT_AMBIGUOUS")
  r=found[0]
  if not isinstance(r.get("path"),str) or not isinstance(r.get("size_bytes"),int):raise ValueError("HANDOFF_CHECKPOINT_MANIFEST_MISSING")
  cp=_safe_member("files/"+_safe_member(r["path"]))
  if cp not in names:raise ValueError("HANDOFF_CHECKPOINT_MISSING")
  data=z.read(cp)
  if len(data)!=r["size_bytes"] or sha256(data)!=EXPECTED_CHECKPOINT_SHA256[seed]:raise ValueError("HANDOFF_CHECKPOINT_IDENTITY_MISMATCH")
 return {"seed":seed,"zip_path":str(path.resolve()),"zip_sha256":EXPECTED_ZIP_SHA256[seed],"manifest_member":mm,"checkpoint_member":cp,"checkpoint_sha256":EXPECTED_CHECKPOINT_SHA256[seed],"checkpoint_size_bytes":len(data)}
def load_checkpoint_cpu_bytes(data:bytes)->Any:
 import torch
 return torch.load(io.BytesIO(data),map_location="cpu",weights_only=True)
def load_authenticated_checkpoint(handoff:Mapping[str,Any])->Mapping[str,Any]:
 with zipfile.ZipFile(handoff["zip_path"]) as z:x=load_checkpoint_cpu_bytes(z.read(handoff["checkpoint_member"]))
 if not isinstance(x,Mapping) or x.get("schema_version")!="stage176a0_selected_checkpoint_v1" or not isinstance(x.get("model_state_dict"),Mapping) or not isinstance(x.get("metadata"),Mapping) or not isinstance(x["metadata"].get("training_args"),Mapping):raise ValueError("CHECKPOINT_SCHEMA_INVALID")
 return x
def encoder_fingerprint(state:Mapping[str,Any])->dict[str,Any]:
 vals={}; raw=[]; numel=0; dtypes=[]
 for k,v in sorted(state.items()):
  if k.startswith("mamba."):
   t=v.detach().cpu().contiguous(); b=t.numpy().tobytes(); vals[k]=sha256(b);raw.append(b);numel+=t.numel();dtypes.append(str(t.dtype))
 fp={"canonical_digest":sha256(canonical_json(vals)),"raw_concat_digest":sha256(b"".join(raw)),"tensor_count":len(vals),"total_numel":numel,"total_raw_bytes":sum(map(len,raw)),"dtypes":sorted(set(dtypes))}
 if (fp["canonical_digest"],fp["tensor_count"],fp["total_numel"],fp["total_raw_bytes"],fp["dtypes"]) != (COMMON_ENCODER_CANONICAL_SHA256,COMMON_ENCODER_TENSOR_COUNT,COMMON_ENCODER_NUMEL,COMMON_ENCODER_RAW_BYTES,["torch.float32"]):raise ValueError("COMMON_ENCODER_CANONICAL_CONTRACT_MISMATCH")
 return fp
def resolve_hf_snapshot(revision:str)->tuple[Path,dict[str,Any]]:
 if revision!=HF_REVISION:raise ValueError("HF_REVISION_MISMATCH")
 from huggingface_hub import snapshot_download,__version__ as hv
 p=Path(snapshot_download(repo_id=HF_MODEL,revision=revision,allow_patterns=["config.json","tokenizer*","special_tokens_map.json","vocab.*","merges.txt"]))
 if p.name!=revision:raise ValueError("HF_RESOLVED_REVISION_MISMATCH")
 from transformers import AutoConfig,AutoTokenizer,__version__ as tv
 config=AutoConfig.from_pretrained(str(p),local_files_only=True,trust_remote_code=False); tok=AutoTokenizer.from_pretrained(str(p),local_files_only=True,trust_remote_code=False,use_fast=True)
 if not tok.is_fast:raise ValueError("TOKENIZER_OFFSET_MAPPING_REQUIRED")
 files=[{"path":str(x.resolve()),"size_bytes":x.stat().st_size,"sha256":file_sha256(x)} for x in sorted(p.rglob("*")) if x.is_file() and (x.name=="config.json" or "tokenizer" in x.name or x.name in {"vocab.json","merges.txt","special_tokens_map.json"})]
 return p,{"hf_model_id":HF_MODEL,"requested_hf_revision":revision,"resolved_hf_revision":revision,"tokenizer_files":files,"tokenizer_class":type(tok).__name__,"tokenizer_backend_class":type(tok.backend_tokenizer).__name__,"tokenizer_is_fast":True,"tokenizer":tok,"config":config,"transformers_version":tv,"huggingface_hub_version":hv}
def normalized_a0_constructor_args(checkpoint:Mapping[str,Any],backbone:Any)->dict[str,Any]:
 state=checkpoint["model_state_dict"]; args=checkpoint["metadata"]["training_args"]; names=set(state); flags={"use_boundary_head":"boundary_head.","use_frame_violation_head":"frame_violation_head.","use_predicate_isolation_head":"predicate_isolation_head.","use_preservation_entitlement_head":"preservation_entitlement_head.","use_temporal_diagnostic_head":"temporal_diagnostic_head.","use_temporal_residual_adapter":"temporal_residual_adapter.","use_temporal_channel":"temporal_channel_v1."}
 return {"backbone":backbone,"frame_size":128,"predicate_size":128,"sufficiency_size":128,"energy_size":64,"dropout":.1,"freeze_a_log":False,"decision_mode":"explicit_product","reason_router_epsilon":float(args.get("reason_router_epsilon",1e-8)),"use_temporal_comparator":"alpha_temporal_raw" in names,"use_predicate_comparator":"alpha_predicate_raw" in names,"alpha_temporal_init":1.25,"alpha_predicate_init":1.25,**{k:any(n.startswith(v) for n in names) for k,v in flags.items()}}
def build_a0_model(snapshot:Path,checkpoint:Mapping[str,Any])->Any:
 from transformers import MambaConfig,MambaModel
 root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/"src")]
 from contramamba.modeling_v6b_minimal import ContraMambaV6BMinimal
 model=ContraMambaV6BMinimal(**normalized_a0_constructor_args(checkpoint,MambaModel(MambaConfig.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False))));model.load_state_dict(checkpoint["model_state_dict"],strict=True);model.eval();return model
def forward_prefix(model:Any,bundle:dict[str,Any],seed:str,checkpoint_sha:str)->dict[str,Any]:
 import torch
 x={k:torch.tensor(v,dtype=torch.long).unsqueeze(0) for k,v in bundle.items() if k in {"input_ids","attention_mask","claim_mask","evidence_mask"}}
 with torch.inference_mode():out=model(**x)
 logits=out["logits"] if isinstance(out,Mapping) else out.logits; probs=torch.softmax(logits,dim=-1)[0].tolist(); i=max(range(3),key=lambda j:probs[j]); ordered=sorted(probs,reverse=True)
 return {"seed":seed,"checkpoint_sha256":checkpoint_sha,"predicted_final_label":CLASS_ORDER[i],"probabilities":dict(zip(CLASS_ORDER,probs)),"confidence":probs[i],"margin":ordered[0]-ordered[1]}
def synthetic_forward_sentinel(model:Any,tok:Any)->None:
 import torch
 b=tokenize_prefix_bundle(tok,"Claim: blorp\nEvidence: snarp\nAdditional evidence:\n"); x={k:torch.tensor(v,dtype=torch.long).unsqueeze(0) for k,v in b.items() if k in {"input_ids","attention_mask","claim_mask","evidence_mask"}}
 with torch.inference_mode():out=model(**x)
 logits=out["logits"] if isinstance(out,Mapping) else out.logits
 if tuple(logits.shape)!=(1,3) or not torch.isfinite(logits).all().item():raise ValueError("SYNTHETIC_FORWARD_CONTRACT_MISMATCH")
def preflight_one_seed(seed:str,handoff:Mapping[str,Any],snapshot:Path,tok:Any,synthetic:bool=False)->dict[str,Any]:
 cp=load_authenticated_checkpoint(handoff);model=build_a0_model(snapshot,cp)
 try:
  fp=encoder_fingerprint(cp["model_state_dict"])
  if synthetic:synthetic_forward_sentinel(model,tok)
  return {**handoff,"encoder":fp,"strict_load":"PASS","metadata_training_args":"PASS","synthetic_forward":"PASS" if synthetic else "NOT_RUN"}
 finally:del model,cp;gc.collect()
def screen_one_seed(seed:str,handoff:Mapping[str,Any],snapshot:Path,checked:Mapping[str,dict[str,Any]])->dict[str,Any]:
 cp=load_authenticated_checkpoint(handoff);model=build_a0_model(snapshot,cp)
 try:
  fp=encoder_fingerprint(cp["model_state_dict"])
  return {"head_outputs":{i:forward_prefix(model,x["prefix_bundle"],seed,handoff["checkpoint_sha256"]) for i,x in checked.items()},"handoff":{**handoff,"encoder":fp,"strict_load":"PASS","metadata_training_args":"PASS"}}
 finally:del model,cp;gc.collect()
def join_screening(pool:list[dict[str,Any]],outputs:Mapping[str,Mapping[str,dict[str,Any]]],checked:Mapping[str,dict[str,Any]])->list[dict[str,Any]]:
 out=[]
 for x in pool:
  r={"schema_version":SCREENING_SCHEMA,"stable_item_id":x["stable_item_id"],"construction_status":x["construction_status"],"construction_failure_label":x["construction_failure_label"],"prefix_gold_label":"NOT_ENTITLED","p":None,"d":None,"tau":None,"tau_minus_p":None,"corr_post_tau_available":None,"ctrl_post_tau_available":None,"head_outputs":[],"eligible":False}
  if x["construction_status"]=="valid":
   q=checked[x["stable_item_id"]];r.update({k:q[k] for k in ("p","d","tau","tau_minus_p","corr_post_tau_available","ctrl_post_tau_available")});r["head_outputs"]=[outputs[s][x["stable_item_id"]] for s in EXPECTED_ZIP_SHA256];r["eligible"]=all(y["predicted_final_label"]=="SUPPORT" for y in r["head_outputs"])
  out.append(r)
 return out
def id_lists(screening:list[dict[str,Any]],pool:Mapping[str,dict[str,Any]])->tuple[list[dict[str,Any]],list[dict[str,Any]],str]:
 e=sorted(({"schema_version":ELIGIBLE_SCHEMA,"stable_item_id":x["stable_item_id"]} for x in screening if x["eligible"]),key=lambda x:x["stable_item_id"])
 if len(e)<30:return e,[],"INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE"
 chosen=e if len(e)<=64 else sorted(e,key=lambda x:(sha256(canonical_json({k:pool[x["stable_item_id"]][k] for k in ("schema_version","pair_id","truncation_source_id","refute_source_id","control_source_id","prefix_text","correction_text","control_text","source_dataset_physical_sha256","source_dataset_semantic_sha256")})),x["stable_item_id"]))[:64]
 return e,[{"schema_version":FINAL_SCHEMA,"stable_item_id":x["stable_item_id"]} for x in chosen],"PHASE_B_ELIGIBLE"
def git_provenance(root:Path,integration_preflight:bool=False)->dict[str,Any]:
 status=subprocess.check_output(["git","status","--porcelain=v1"],cwd=root,text=True).splitlines(); branch=subprocess.check_output(["git","branch","--show-current"],cwd=root,text=True).strip(); head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
 if branch!="longterm-k-series-native-state-kinematics" or subprocess.call(["git","merge-base","--is-ancestor",K2W_PREREG_AUTHORITY_COMMIT,head],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL):raise ValueError("GIT_PROVENANCE_MISMATCH")
 allowed=HISTORICAL_K1_UNTRACKED|(K2W_UNTRACKED if integration_preflight else set())
 for line in status:
  if line[:2]!="??" or line[3:].replace("\\","/") not in allowed:raise ValueError("GIT_DIRTY_CONTRACT_MISMATCH")
 return {"runtime_branch":branch,"runtime_git_head":head,"runtime_dirty_contract":status}
def phase_a_manifest(handoffs:dict[str,Any],hf:dict[str,Any],runtime:dict[str,Any],summary:dict[str,Any],blobs:Mapping[str,bytes],eligible:list[dict[str,Any]],final:list[dict[str,Any]],verdict:str)->dict[str,Any]:
 import torch,transformers,huggingface_hub,tokenizers
 return {"schema_version":MANIFEST_SCHEMA,"k2w_prereg_authority_commit":K2W_PREREG_AUTHORITY_COMMIT,"k2_closure_commit":K2_CLOSURE_COMMIT,"script_sha256":file_sha256(Path(__file__)),"source_byte_origin":"FROZEN_GIT_BLOB","source_git_commit":SOURCE_BLOB_COMMIT,"source_git_path":SOURCE_REL,"source_physical_sha256":SOURCE_PHYSICAL_SHA256,"source_semantic_sha256":SOURCE_SEMANTIC_SHA256,"hf_model_id":HF_MODEL,"requested_hf_revision":HF_REVISION,"resolved_hf_revision":hf["resolved_hf_revision"],"tokenizer_files":hf["tokenizer_files"],"tokenizer_class":hf["tokenizer_class"],"tokenizer_backend_class":hf["tokenizer_backend_class"],"tokenizer_is_fast":True,"trust_remote_code":False,"add_special_tokens":False,"observer_input_contract":OBSERVER_INPUT_CONTRACT,"historical_a0_serialization_equivalence_claimed":False,"handoffs":handoffs,"common_encoder":{"canonical_digest":COMMON_ENCODER_CANONICAL_SHA256,"secondary_raw_digest":COMMON_ENCODER_RAW_CONCAT_SHA256,"tensor_count":COMMON_ENCODER_TENSOR_COUNT,"numel":COMMON_ENCODER_NUMEL,"raw_bytes":COMMON_ENCODER_RAW_BYTES},"candidate_pool_sha256":sha256(blobs["candidate_pool.jsonl"]),"screening_artifact_sha256":sha256(blobs["screening.jsonl"]),"eligible_id_list_sha256":sha256(blobs["eligible_ids.jsonl"]),"final_confirmatory_id_list_sha256":sha256(blobs["final_confirmatory_ids.jsonl"]),"N_eligible":len(eligible),"N_final":len(final),"phase_a_verdict":verdict,"python_version":sys.version,"platform":platform.platform(),"torch_version":torch.__version__,"transformers_version":transformers.__version__,"tokenizers_version":tokenizers.__version__,"huggingface_hub_version":huggingface_hub.__version__,**summary,**runtime}
def write_phase_a_outputs(output:Path,pool:list[dict[str,Any]],screening:list[dict[str,Any]],handoffs:dict[str,Any],hf:dict[str,Any],runtime:dict[str,Any],summary:dict[str,Any])->dict[str,Any]:
 root=Path(__file__).resolve().parents[1]; target=output.resolve()
 if target==root.resolve() or root.resolve() in target.parents or (target.exists() and any(target.iterdir())):raise ValueError("OUTPUT_DIRECTORY_PROTECTED")
 e,f,v=id_lists(screening,{x["stable_item_id"]:x for x in pool}); blobs={"candidate_pool.jsonl":canonical_jsonl(pool),"screening.jsonl":canonical_jsonl(screening),"eligible_ids.jsonl":canonical_jsonl(e),"final_confirmatory_ids.jsonl":canonical_jsonl(f)};target.mkdir(parents=True,exist_ok=True)
 for n,b in blobs.items():(target/n).write_bytes(b)
 m=phase_a_manifest(handoffs,hf,runtime,summary,blobs,e,f,v);(target/"phase_a_manifest.json").write_bytes(canonical_json(m)+b"\n");return m
def parser()->argparse.ArgumentParser:
 p=argparse.ArgumentParser();p.add_argument("--source-data");p.add_argument("--seed180-handoff",required=True);p.add_argument("--seed181-handoff",required=True);p.add_argument("--seed182-handoff",required=True);p.add_argument("--hf-revision",required=True);p.add_argument("--output-dir");p.add_argument("--integration-preflight",action="store_true");return p
def main(argv:list[str]|None=None)->int:
 a=parser().parse_args(argv);root=Path(__file__).resolve().parents[1]; runtime=git_provenance(root,integration_preflight=a.integration_preflight);snapshot,hf=resolve_hf_snapshot(a.hf_revision);source=Path(a.source_data) if a.source_data else root/SOURCE_REL;handoffs={s:audit_handoff(Path(getattr(a,s+"_handoff")),s) for s in EXPECTED_ZIP_SHA256};rows=load_source(root,source);pool,checked=prepare_candidate_pool(rows,hf["tokenizer"]);summary=construction_summary(pool,checked)
 expected={"N_total_attempts":300,"N_semantically_valid":300,"N_exact_prefix_valid":300,"N_event_divergence_valid":300,"N_window8_available":300,"N_construction_valid":300,"N_duplicate_excluded":0,"tau_minus_p_distribution":{"1":149,"2":151},"corr_post_tau_available_distribution":{"19":4,"20":5,"21":7,"22":118,"23":61,"24":70,"25":34,"26":1},"ctrl_post_tau_available_distribution":{"16":2,"17":3,"18":7,"19":38,"20":53,"21":116,"22":47,"23":12,"24":22}}
 if summary!=expected:raise ValueError("K2W_FROZEN_CONSTRUCTION_FEASIBILITY_MISMATCH")
 if a.integration_preflight:
  result={s:preflight_one_seed(s,handoffs[s],snapshot,hf["tokenizer"],synthetic=s=="seed180") for s in EXPECTED_ZIP_SHA256};print(canonical_json({"integration_preflight":"PASS","construction":summary,"handoffs":result}).decode());return 0
 if not a.output_dir:raise ValueError("PHASE_A_OUTPUT_REQUIRED")
 screened={s:screen_one_seed(s,handoffs[s],snapshot,checked) for s in EXPECTED_ZIP_SHA256};outputs={s:screened[s]["head_outputs"] for s in EXPECTED_ZIP_SHA256};observed_handoffs={s:screened[s]["handoff"] for s in EXPECTED_ZIP_SHA256};write_phase_a_outputs(Path(a.output_dir),pool,join_screening(pool,outputs,checked),observed_handoffs,hf,runtime,summary);return 0
if __name__=="__main__":raise SystemExit(main())
