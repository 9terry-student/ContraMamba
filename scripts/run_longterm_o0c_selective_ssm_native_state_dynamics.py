"""Fail-closed O0c runner; importing this module performs no ML work."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path

OBSERVER_COMMIT="e03b38fa0681fd04d8633a7de184559bd0473133"
DATASET=Path("data/longterm_o0b_matched_controls_v1.jsonl"); VALIDATION=Path("reports/longterm_o0b_matched_controls_v1_validation.json")
PAIR_ORDER=("o0b_pair_001","o0b_pair_002","o0b_pair_003"); CONDITION_ORDER=("reference_sufficient","insufficient_matched","paraphrase_sufficient","surface_null_matched")
COMPARISONS=(("comparison-A","insufficient_matched"),("comparison-B","paraphrase_sufficient"),("comparison-C","surface_null_matched"))
class ContractError(RuntimeError): pass
def require(ok,msg):
 if not ok: raise ContractError(msg)
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def parser():
 p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
 for n in ("output-dir","run-name","exact-command","expected-runner-commit","expected-runner-sha256","expected-observer-commit","expected-observer-sha256"): p.add_argument("--"+n,required=True)
 return p
def canonical_exact_command(argv):
 hits=[i for i,x in enumerate(argv) if x=="--exact-command"]; require(len(hits)==1 and hits[0]+1<len(argv),"exact-command")
 i=hits[0]; return json.dumps(list(argv[:i])+list(argv[i+2:]),ensure_ascii=False,separators=(",",":"))
def _strings(ns):
 for n in ("run_name","exact_command","expected_runner_commit","expected_runner_sha256","expected_observer_commit","expected_observer_sha256"): require(isinstance(getattr(ns,n,None),str) and getattr(ns,n).strip(),n)
def parse_args(argv=None):
 actual=list(sys.argv if argv is None else argv); ns=parser().parse_args(actual[1:] if actual and not actual[0].startswith("--") else actual); _strings(ns); require(ns.exact_command==canonical_exact_command(actual),"exact-command mismatch"); return ns
def load_observer():
 path=Path(__file__).with_name("observe_longterm_o0c_selective_ssm_native_state_dynamics.py"); spec=importlib.util.spec_from_file_location("o0c_observer",path); require(spec and spec.loader,"observer import"); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def git_head(): return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
def git_canonical_input(path):
 rel=Path(path).as_posix()
 tracked=subprocess.run(["git","ls-files","--error-unmatch","--",rel],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 require(tracked.returncode==0,"input not tracked at HEAD")
 dirty=subprocess.run(["git","diff","--quiet","HEAD","--",rel],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 require(dirty.returncode==0,"input differs from HEAD" if dirty.returncode==1 else "git diff failed")
 shown=subprocess.run(["git","show",f"HEAD:{rel}"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 require(shown.returncode==0,"git show failed"); require(bool(shown.stdout),"empty canonical input")
 return shown.stdout
def preflight(ns,observer,head=git_head,file_digest=digest,canonical_input=git_canonical_input):
 _strings(ns); output=Path(ns.output_dir); staging=output.with_name(output.name+".staging"); require(head()==ns.expected_runner_commit,"runner HEAD"); require(file_digest(Path(__file__))==ns.expected_runner_sha256,"runner SHA256"); require(ns.expected_observer_commit==OBSERVER_COMMIT,"observer commit"); identity=observer.observer_script_identity(); require(identity["observer_script_sha256"]==ns.expected_observer_sha256,"observer SHA256"); require(not output.exists(),"output collision"); require(not staging.exists(),"staging collision"); dataset_bytes=canonical_input(DATASET); require(hashlib.sha256(dataset_bytes).hexdigest()==observer.DATASET_SHA256,"dataset SHA256"); validation_bytes=canonical_input(VALIDATION); require(hashlib.sha256(validation_bytes).hexdigest()==observer.VALIDATION_ARTIFACT_SHA256,"validation SHA256"); observer.runtime_gate(); return identity,dataset_bytes,validation_bytes
def read_inputs(observer,dataset_bytes,validation_bytes):
 rows=[json.loads(x) for x in dataset_bytes.decode("utf-8").splitlines()]; require(len(rows)==3 and [x.get("pair_id") for x in rows]==list(PAIR_ORDER),"dataset order"); fields=("schema_version","pair_id","claim",*CONDITION_ORDER)
 for row in rows: require(all(isinstance(row.get(k),str) and row[k] and all(32<=ord(c)<=126 for c in row[k]) for k in fields),"dataset text")
 artifact=json.loads(validation_bytes.decode("utf-8")); require(isinstance(artifact.get("pairs"),list),"validation artifact"); return rows,artifact
def serialize(row,condition): return "Claim: "+row["claim"]+"\nEvidence: "+row[condition]
def validate_tokens(tokenizer,rows,artifact):
 require(tokenizer.is_fast is True,"tokenizer fast"); pairs={x["pair_id"]:x for x in artifact["pairs"]}; require(set(pairs)==set(PAIR_ORDER),"artifact pairs"); out={}; keys=("full_serialized_token_ids","full_offset_mapping","full_token_count","terminal_index","evidence_char_start","evidence_start_index","evidence_start_offset_start","evidence_start_offset_end","evidence_token_count","boundary_crossing")
 for row in rows:
  out[row["pair_id"]]={}
  for condition in CONDITION_ORDER:
   frozen=pairs[row["pair_id"]]["conditions"][condition]; require(all(k in frozen for k in keys),"frozen token fields"); text=serialize(row,condition); got=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True); ids=list(got["input_ids"]); offsets=[list(x) for x in got["offset_mapping"]]; start=len("Claim: "+row["claim"]+"\nEvidence: "); idx=frozen["evidence_start_index"]
   require(ids==frozen["full_serialized_token_ids"],"token ids"); require(offsets==frozen["full_offset_mapping"],"offsets"); require(len(ids)==frozen["full_token_count"] and frozen["terminal_index"]==len(ids)-1,"token count/terminal"); require(frozen["evidence_char_start"]==start and type(idx) is int and 0<=idx<len(offsets) and offsets[idx]==[frozen["evidence_start_offset_start"],frozen["evidence_start_offset_end"]] and offsets[idx][0]==start and frozen["evidence_token_count"]==len(ids)-idx and frozen["boundary_crossing"] is False,"evidence boundary"); out[row["pair_id"]][condition]=ids
 return out
def comparison_anchors(observer,artifact):
 pairs={x["pair_id"]:x for x in artifact["pairs"]}; require(set(pairs)==set(PAIR_ORDER),"anchor pairs"); anchors={}
 for pair in PAIR_ORDER:
  item=pairs[pair]; require(set(item["comparisons_to_reference"])==set(x[1] for x in COMPARISONS),"comparisons")
  for comparison,member in COMPARISONS:
   c=item["comparisons_to_reference"][member]; a=c["anchor_indices"]; ref=item["conditions"]["reference_sufficient"]["full_serialized_token_ids"]; other=item["conditions"][member]["full_serialized_token_ids"]; require(len(ref)==len(other),"trajectory length"); d=next((i for i,(x,y) in enumerate(zip(ref,other)) if x!=y),None)
   require(c["reference_condition"]=="reference_sufficient" and d is not None and d==c["first_divergent_token_index"],"divergence"); require(set(a)==set(observer.ANCHOR_ORDER) and a["anchor_pre_minus_1"]==d-1 and a["anchor_divergence"]==d and [a[k] for k in ("anchor_post_plus_1","anchor_post_plus_2","anchor_post_plus_4")]==[d+1,d+2,d+4] and a["anchor_terminal"]==item["conditions"][member]["terminal_index"],"anchors"); require(all(type(v) is int and 0<=v<len(ref) and v<len(other) for v in a.values()),"anchor range"); anchors[(pair,comparison)]=dict(a)
 require(len(anchors)==9,"anchor count"); return anchors
def discover_layers(model,MambaModel,MambaBlock,MambaMixer,ModuleList):
 require(type(model) is MambaModel and type(model.layers) is ModuleList,"model/layers type"); n=model.config.num_hidden_layers; require(len(model.layers)==n and n>0,"layers"); mixers=[]
 for i,block in enumerate(model.layers): require(type(block) is MambaBlock and block.layer_idx==i,"block structure"); require(type(block.mixer) is MambaMixer and block.mixer.layer_idx==i,"mixer structure"); mixers.append(block.mixer)
 actual=[x for x in model.modules() if type(x) is MambaMixer]; require(len({id(x) for x in mixers})==len(mixers),"duplicate mixer"); require(len(actual)==len(mixers) and {id(x) for x in actual}=={id(x) for x in mixers},"mixer completeness"); desc=[{"layer_index":i,"layer_role":"mamba_mixer"} for i in range(n)]; return desc,{id(x):d for x,d in zip(mixers,desc)}
def validate_model_state(model,torch):
 model.eval(); model.requires_grad_(False); params=list(model.parameters()); require(params,"model parameters")
 for p in params: require(p.device.type=="cpu" and p.dtype==torch.float32 and p.requires_grad is False,"parameter state")
 for b in model.buffers(): require(b.device.type=="cpu" and (not b.is_floating_point() or b.dtype==torch.float32),"buffer state")
 require(model.training is False and all(not getattr(m,"training",False) for m in model.modules()),"module eval")
def _production_factories():
 import torch
 from transformers import AutoTokenizer,MambaModel
 from transformers.models.mamba.modeling_mamba import MambaBlock,MambaMixer
 return torch,AutoTokenizer,MambaModel,MambaBlock,MambaMixer,torch.nn.ModuleList
def run(ns=None,*,observer=None,factories=None,head=git_head,file_digest=digest,canonical_input=git_canonical_input,runtime_info=None):
 ns=parse_args() if ns is None else ns; observer=observer or load_observer(); identity,dataset_bytes,validation_bytes=preflight(ns,observer,head,file_digest,canonical_input); rows,artifact=read_inputs(observer,dataset_bytes,validation_bytes); anchors=comparison_anchors(observer,artifact); torch,AutoTokenizer,MambaModel,MambaBlock,MambaMixer,ModuleList=factories or _production_factories(); tokenizer=AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf",revision="5708daa364c50b880e7bd92eab456e0d34492ee9",trust_remote_code=False,use_fast=True); ids=validate_tokens(tokenizer,rows,artifact); model=MambaModel.from_pretrained("state-spaces/mamba-130m-hf",revision="5708daa364c50b880e7bd92eab456e0d34492ee9",torch_dtype=torch.float32,trust_remote_code=False); validate_model_state(model,torch); descriptors,registration=discover_layers(model,MambaModel,MambaBlock,MambaMixer,ModuleList); members=[]; trajectories={}
 for pair in PAIR_ORDER:
  for condition in CONDITION_ORDER:
   inst=observer.create_native_state_observer(registration,enabled=True); tensor=torch.tensor([ids[pair][condition]],dtype=torch.long)
   with inst.capture():
    with torch.inference_mode(): model(input_ids=tensor,use_cache=False)
   snapshots=inst.snapshots; require(isinstance(snapshots,dict) and {k[0] for k in snapshots}=={1},"forward identity"); captures={d["layer_index"]:{t:snapshots[(1,d["layer_index"],t)] for t in range(len(ids[pair][condition]))} for d in descriptors}; observer.validate_captures(captures,len(ids[pair][condition]),descriptors); members.append({"pair_id":pair,"condition":condition,"captures":captures,"fresh_capture_collection":True,"cache_reused":False})
 observer.validate_member_orchestration(members,descriptors)
 for m in members:
  for layer,states in m["captures"].items(): trajectories[(m["pair_id"],m["condition"],layer)]=[states[i] for i in sorted(states)]
 state_rows,vectors=observer.state_rows(trajectories,descriptors); measured=observer.measurements(state_rows,vectors,anchors); summary=observer.build_summary(measured,state_rows)
 if runtime_info is None:
  import platform,importlib.metadata,transformers,numpy
  runtime_info=(platform.python_version(),numpy.__version__,transformers.__version__,str(Path(importlib.metadata.distribution("transformers").locate_file("transformers")).resolve()),str(Path(transformers.__file__).resolve().parent))
 py,npv,tv,dist,root=runtime_info
 fields={"schema_version":observer.SCHEMA_VERSION,"experiment_name":observer.EXPERIMENT_NAME,"scientific_design_authority_commit":observer.SCIENTIFIC_DESIGN_AUTHORITY_COMMIT,"implementation_authority_commit":observer.IMPLEMENTATION_AUTHORITY_COMMIT,"observer_implementation_commit":OBSERVER_COMMIT,**identity,"dataset_path":observer.DATASET_PATH,"dataset_sha256":observer.DATASET_SHA256,"validation_artifact_path":observer.VALIDATION_ARTIFACT_PATH,"validation_artifact_sha256":observer.VALIDATION_ARTIFACT_SHA256,"model_id":observer.MODEL_ID,"model_revision":observer.MODEL_REVISION,"tokenizer_id":observer.TOKENIZER_ID,"tokenizer_revision":observer.TOKENIZER_REVISION,"model_trust_remote_code":False,"tokenizer_trust_remote_code":False,"add_special_tokens":False,"device":"cpu","dtype":"float32","expected_python_version":observer.EXPECTED_VERSIONS["python"],"observed_python_version":py,"expected_numpy_version":observer.EXPECTED_VERSIONS["numpy"],"observed_numpy_version":npv,"expected_torch_version":observer.EXPECTED_VERSIONS["torch"],"observed_torch_version":torch.__version__,"expected_transformers_version":observer.EXPECTED_VERSIONS["transformers"],"observed_transformers_version":tv,"transformers_distribution_root":dist,"transformers_import_root":root,"source_resolution_classification":"PASS_RECONCILED_UNIQUE_TRANSFORMERS_SOURCE","backend_classification":"BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN","mamba_source_module":observer.MAMBA_MODULE,"mamba_source_sha256":observer.MAMBA_SHA256,"mamba_source_bytes":observer.MAMBA_BYTES,"cache_source_module":observer.CACHE_MODULE,"cache_source_sha256":observer.CACHE_SHA256,"cache_source_bytes":observer.CACHE_BYTES,"capture_source_qualname":observer.CAPTURE_QUALNAME,"capture_source_line":observer.CAPTURE_LINE,"capture_state_source":"native_selective_ssm_recurrent_state","capture_state_timing":"post_consumption_s_t","pair_order":list(PAIR_ORDER),"condition_order":list(CONDITION_ORDER),"comparison_order":[x[0] for x in COMPARISONS],"anchor_order":list(observer.ANCHOR_ORDER),"layer_descriptors":descriptors,"serialization_template":"canonical-json-v1/deterministic-npz-v1","exact_command":ns.exact_command,"run_name":ns.run_name,"required_artifacts":list(observer.REQUIRED_ARTIFACTS),"equivalence_gate_status":"PASS_EXACT_EQUIVALENCE_NONINTERFERENCE","capture_completeness_status":"PASS_COMPLETE_NATIVE_STATE_CAPTURE","provenance_status":"PASS_PROVENANCE_VALIDATED","execution_status":"PASS_EXECUTION_COMPLETE","blocker":None}; manifest=observer.build_manifest(fields); bundle=observer.build_bundle(manifest,state_rows,vectors,measured,summary); require(tuple(bundle)==tuple(observer.REQUIRED_ARTIFACTS),"artifact set"); observer.validate_bundle(bundle); observer.publish_bundle(Path(ns.output_dir),bundle)
def main(argv=None): run(parse_args(argv))
if __name__=="__main__": main()
