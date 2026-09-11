import atexit, hashlib, importlib.util, io, json, shutil, subprocess, sys, tempfile, types, zipfile
from pathlib import Path
import pytest

P=Path(__file__).parents[1]/"scripts"/"longterm_k2_exact_prefix_phase_a.py"; spec=importlib.util.spec_from_file_location("k2",P); k=importlib.util.module_from_spec(spec);sys.modules["k2"]=k;spec.loader.exec_module(k)
SCRATCH_ROOT=Path(__file__).parent/"_k2_phase_a_test_tmp";atexit.register(lambda: shutil.rmtree(SCRATCH_ROOT,ignore_errors=True))
def scratch(name):
 p=SCRATCH_ROOT/name
 if p.exists():shutil.rmtree(p) if p.is_dir() else p.unlink()
 p.parent.mkdir(exist_ok=True);return p
def row(i,pair,typ,lab="NOT_ENTITLED",claim="claim",ev="evidence",**kw):
 r={"id":i,"pair_id":pair,"claim":claim,"evidence":ev,"final_label":lab,"frame_compatible_label":0,"predicate_covered_label":1,"sufficiency_label":0,"polarity_label":"NONE","primary_failure_type":"sufficiency","intervention_type":typ};r.update(kw);return r
def good(pair="p",claim="claim",flip=True):
 q=row("q",pair,"polarity_flip" if flip else "none","REFUTE",claim,"false",polarity_label="REFUTE")
 return [row("t",pair,"evidence_truncation",claim=claim,ev="short"),row("e",pair,"entity_swap",claim=claim,ev="other",primary_failure_type="frame"),q]
class OffsetTok:
 is_fast=True
 def __call__(self,text,**kw):
  # Template-only, boundary (leading space + semantic), and semantic tokens.
  spans=[]; start=0
  for part in ["Claim:"," claim","\nEvidence:"," evidence","\nAdditional"," evidence:"]:
   spans.append((start,start+len(part)));start+=len(part)
  return {"input_ids":list(range(len(spans))),"offset_mapping":spans} if kw.get("return_offsets_mapping") else {"input_ids":list(range(max(2,len(text))) )}
class BranchTok:
 is_fast=True
 def __init__(self,n=8,same=False,prefix_bad=False):self.n=n;self.same=same;self.prefix_bad=prefix_bad
 def __call__(self,s,**kw):
  if kw.get("return_offsets_mapping"): return OffsetTok()(s,**kw)
  if s.endswith(" CORR") or " This is false:" in s: return {"input_ids":list(range(6))+[3]*self.n}
  if s.endswith(" CTRL") or " A separate event:" in s: return {"input_ids":([9]+list(range(1,6)) if self.prefix_bad else list(range(6)))+([3]*self.n if self.same else [4]*self.n)}
  return {"input_ids":list(range(6))}
def candidate():
 x=k.recipe("p",good());x["correction_text"]=" CORR";x["control_text"]=" CTRL";return x

def test_literal_recipe_preference_and_exact_utf8():
 x=k.recipe("p",good());assert x["prefix_text"]=="Claim: claim\nEvidence: short\nAdditional evidence:";assert x["correction_text"]==" This is false: false";assert x["control_text"]==" A separate event: other";assert x["refute_source_id"]=="q"
 fallback=k.recipe("p",good(flip=False));assert fallback["refute_source_id"]=="q"
 assert k.canonical_jsonl([{"b":"é","a":1}])==b'{"a":1,"b":"\xc3\xa9"}\n'
def test_recipe_invalid_matrix():
 cases=[]
 cases.append((good()+[good()[0]],"INVALID_SOURCE_MULTIPLICITY"))
 a=good();a[2]["claim"]="other";cases.append((a,"INVALID_SOURCE_CLAIM_IDENTITY"))
 a=good();a[0]["sufficiency_label"]=1;cases.append((a,"INVALID_TRUNCATION_SEMANTICS"))
 a=good();a[1]["polarity_label"]="REFUTE";cases.append((a,"INVALID_CONTROL_SEMANTICS"))
 a=good();a[2]["polarity_label"]="NONE";cases.append((a,"INVALID_CORRECTION_SEMANTICS"))
 for rows,label in cases:assert k.recipe("p",rows)["construction_failure_label"]==label
def test_stable_identity_duplicate_and_order():
 pool,_=k.prepare_candidate_pool(good("z","same")+good("a","same"),BranchTok());assert [x["stable_item_id"] for x in pool]==sorted(x["stable_item_id"] for x in pool);assert sum(x["construction_failure_label"]=="DUPLICATE_BASE_CLAIM_EXCLUDED" for x in pool)==1
 x=k.recipe("p",good());core={key:x[key] for key in ("schema_version","pair_id","truncation_source_id","refute_source_id","control_source_id","prefix_text","correction_text","control_text")};assert x["stable_item_id"]=="k2ep-v1:"+hashlib.sha256(k.canonical_json(core)).hexdigest();assert x["base_claim_sha256"]==hashlib.sha256(b"claim").hexdigest()
def test_source_format_and_hash_contract(monkeypatch):
 raw=(json.dumps(good()[0],separators=(",",":"))+"\n").encode();monkeypatch.setattr(k,"SOURCE_PHYSICAL",k.sha(raw));monkeypatch.setattr(k,"SOURCE_SEMANTIC",k.semantic_sha([good()[0]]));p=scratch("source");p.write_bytes(raw);assert k.load_source(p)[0]["id"]=="t"
 for broken in [raw.replace(b"\n",b"\r\n"),raw[:-1],b"\n",b"not-json\n"]:
  p.write_bytes(broken)
  with pytest.raises(ValueError):k.load_source(p)
 p.write_bytes(b"\xef\xbb\xbf"+raw)
 with pytest.raises(ValueError,match="SOURCE_PHYSICAL_JSONL_FORMAT_INVALID"):k.load_source(p)
 p.write_bytes(raw+b"\n"+raw)
 monkeypatch.setattr(k,"SOURCE_PHYSICAL",k.sha(raw+b"\n"+raw))
 with pytest.raises(ValueError,match="SOURCE_PHYSICAL_JSONL_FORMAT_INVALID"):k.load_source(p)
 p.write_bytes(raw);monkeypatch.setattr(k,"SOURCE_PHYSICAL","0"*64)
 with pytest.raises(ValueError,match="PHYSICAL"):k.load_source(p)
 monkeypatch.setattr(k,"SOURCE_PHYSICAL",k.sha(raw));monkeypatch.setattr(k,"SOURCE_SEMANTIC","1"*64)
 with pytest.raises(ValueError,match="SEMANTIC"):k.load_source(p)
def test_prefix_masks_are_literal_overlap_not_retokenized():
 prefix="Claim: claim\nEvidence: evidence\nAdditional evidence:";out=k.prefix_model_inputs(k.tokenize_prefix_bundle(OffsetTok(),prefix))
 assert out["input_ids"]==list(range(6));assert out["attention_mask"]==[1]*6;assert out["claim_mask"]==[False,True,False,False,False,False];assert out["evidence_mask"]==[False,False,False,True,False,False];assert not any(a and b for a,b in zip(out["claim_mask"],out["evidence_mask"]))
 assert "eos" not in P.read_text().lower() and k.HISTORICAL_A0_SERIALIZATION_EQUIVALENCE_CLAIMED is False
def test_token_contract_full_boundary_matrix():
 x=candidate()
 for n,ok in [(7,False),(8,True),(24,True),(25,False)]:assert k.token_contract(x,BranchTok(n))["ok"] is ok
 assert k.token_contract(x,BranchTok(8,same=True))["failure"]=="INVALID_CONTINUATION_TOKEN_CONTRACT"
 assert k.token_contract(x,BranchTok(8,prefix_bad=True))["failure"]=="INVALID_EXACT_PREFIX"
 r=k.token_contract(x,BranchTok(8));assert r["tau"]==5 and r["prefix_bundle"]["attention_mask"]==[1]*6 and r["prefix_bundle"]["position_ids"]==list(range(6))
def zip_bytes(manifest, members=None):
 b=io.BytesIO()
 with zipfile.ZipFile(b,"w") as z:
  z.writestr("handoff_manifest.json",json.dumps(manifest));
  for n,v in (members or {"checkpoint.pt":b"ckpt"}).items():z.writestr(n,v)
 return b.getvalue()
def test_handoff_rejection_matrix(monkeypatch):
 cp=b"ckpt";h=k.sha(cp);m={"schema_version":"contramamba-handoff-v3","expected_commit":k.A0_COMMIT,"actual_commit":k.A0_COMMIT,"selected_checkpoint":{"path":"checkpoint.pt","sha256":h,"size_bytes":4}}
 p=scratch("h.zip");p.write_bytes(zip_bytes(m));monkeypatch.setitem(k.EXPECTED_CHECKPOINT_SHA256,"seed180",h);monkeypatch.setitem(k.EXPECTED_ZIP_SHA256,"seed180",k.file_sha(p));assert k.audit_handoff(p,"seed180")["checkpoint_member"]=="checkpoint.pt"
 variants=[({}, {"checkpoint.pt":cp}),({**m,"schema_version":"bad"},{"checkpoint.pt":cp}),({**m,"actual_commit":"bad"},{"checkpoint.pt":cp}),({"schema_version":"contramamba-handoff-v3","expected_commit":k.A0_COMMIT,"actual_commit":k.A0_COMMIT},{"checkpoint.pt":cp}),({**m,"selected_checkpoint":{**m["selected_checkpoint"],"size_bytes":3}},{"checkpoint.pt":cp}),({**m,"selected_checkpoint":{**m["selected_checkpoint"],"sha256":""}},{"checkpoint.pt":cp}),({**m,"selected_checkpoint":{**m["selected_checkpoint"],"path":"../x"}},{"checkpoint.pt":cp})]
 for manifest,members in variants:
  p.write_bytes(zip_bytes(manifest,members));monkeypatch.setitem(k.EXPECTED_ZIP_SHA256,"seed180",k.file_sha(p))
  with pytest.raises(ValueError):k.audit_handoff(p,"seed180")
 p.write_bytes(b"wrong");
 with pytest.raises(ValueError,match="ZIP"):k.audit_handoff(p,"seed180")
def test_checkpoint_load_weights_only_and_strict(monkeypatch):
 calls={}
 class Torch:
  @staticmethod
  def load(*a,**kw):calls.update(kw);return "ok"
 monkeypatch.setitem(sys.modules,"torch",Torch);assert k.load_checkpoint_cpu_bytes(b"x")=="ok" and calls=={"map_location":"cpu","weights_only":True}
 class M:
  def load_state_dict(self,s,strict):assert strict is True;raise RuntimeError("shape")
 with pytest.raises(RuntimeError):k.strict_load(M(),{})
def test_encoder_digest_synthetic_and_phase_b_absence():
 class T:
  def __init__(self,b):self.b=b
  def detach(self):return self
  def cpu(self):return self
  def contiguous(self):return self
  def numpy(self):return self
  def tobytes(self):return self.b
 a={"mamba.a":T(b"a")};assert k.encoder_digest(a)==k.encoder_digest(a);assert k.encoder_digest(a)!=k.encoder_digest({"mamba.a":T(b"b")})
 source=P.read_text().lower();assert "native state" not in source and "holm" not in source and "sign test" not in source
def test_hf_resolved_revision_mismatch_and_exact_positive(monkeypatch):
 root=scratch("hf");exact=root/k.HF_REVISION;exact.mkdir(parents=True);(exact/"config.json").write_text("{}")
 hub=types.ModuleType("huggingface_hub");hub.__version__="hub-test";hub.snapshot_download=lambda **_:str(exact)
 transformers=types.ModuleType("transformers");transformers.__version__="transformers-test"
 class Config:
  @staticmethod
  def from_pretrained(*_,**__):return object()
 class Tok:
  is_fast=True;backend_tokenizer=object()
  @staticmethod
  def from_pretrained(*_,**__):return Tok()
 transformers.AutoConfig=Config;transformers.AutoTokenizer=Tok
 monkeypatch.setitem(sys.modules,"huggingface_hub",hub);monkeypatch.setitem(sys.modules,"transformers",transformers)
 snapshot,hf=k.resolve_hf_snapshot(k.HF_REVISION);assert snapshot==exact and hf["requested_hf_revision"]==k.HF_REVISION and hf["resolved_hf_revision"]==k.HF_REVISION
 wrong=root/"not-the-requested-revision";wrong.mkdir();hub.snapshot_download=lambda **_:str(wrong)
 with pytest.raises(ValueError,match="HF_RESOLVED_REVISION_MISMATCH"):k.resolve_hf_snapshot(k.HF_REVISION)
def test_build_a0_model_enforces_frozen_encoder_digest(monkeypatch):
 events=[]
 class Tensor:shape=(1,)
 class Model:
  def load_state_dict(self,state,strict):events.append(("strict",strict))
  def eval(self):events.append(("eval",));return self
 class MambaConfig:
  @staticmethod
  def from_pretrained(*_,**kw):events.append(("config",kw));return "config"
 class MambaModel:
  def __init__(self,config):events.append(("backbone",config))
 transformers=types.ModuleType("transformers");transformers.MambaConfig=MambaConfig;transformers.MambaModel=MambaModel
 package=types.ModuleType("contramamba");package.__path__=[];module=types.ModuleType("contramamba.modeling_v6b_minimal");module.ContraMambaV6BMinimal=lambda **kw:events.append(("construct",kw)) or Model()
 monkeypatch.setitem(sys.modules,"transformers",transformers);monkeypatch.setitem(sys.modules,"contramamba",package);monkeypatch.setitem(sys.modules,"contramamba.modeling_v6b_minimal",module)
 checkpoint={"training_args":{},"model_state_dict":{"mamba.weight":Tensor()}}
 reported=[];monkeypatch.setattr(k,"encoder_digest",lambda state:reported.append("67bfc8cb253fef88b2b8936d442468b9ddcbffa8b79582ba3e2432cb271a937b") or reported[-1])
 assert reported==[] and k.build_a0_model(Path("snapshot"),checkpoint).__class__ is Model and reported==[k.ENCODER_DIGEST] and ("strict",True) in events and ("eval",) in events
 events.clear();monkeypatch.setattr(k,"encoder_digest",lambda state:"0"*64)
 with pytest.raises(ValueError,match="COMMON_ENCODER_DIGEST_MISMATCH"):k.build_a0_model(Path("snapshot"),checkpoint)
 assert events==[]
def test_screen_unanimity_gold_and_all_attempts():
 pool,checked=k.prepare_candidate_pool(good("ok")+good("bad"),BranchTok(8)); pool[1]["construction_status"]="excluded"; checked.pop(pool[1]["stable_item_id"],None)
 outputs={seed:{pool[0]["stable_item_id"]:{"seed":seed,"predicted_final_label":"SUPPORT","checkpoint_sha256":seed,"probabilities":{},"confidence":1,"margin":1}} for seed in k.EXPECTED_ZIP_SHA256}
 out=k.join_screening(pool,outputs,checked);assert out[0]["eligible"] and not out[1]["eligible"] and len(out)==2 and out[0]["prefix_gold_label"]=="NOT_ENTITLED"
 outputs["seed182"][pool[0]["stable_item_id"]]["predicted_final_label"]="REFUTE";assert not k.join_screening([pool[0]],outputs,checked)[0]["eligible"]
def test_final_n_matrix_and_finalized_byte_hashes():
 def ss(n):return [{"stable_item_id":f"{i:03}","eligible":True} for i in range(n)]
 for n,want,verdict in [(29,0,"INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE"),(30,30,"PHASE_B_ELIGIBLE"),(64,64,"PHASE_B_ELIGIBLE"),(65,64,"PHASE_B_ELIGIBLE")]:assert len(k.id_lists(ss(n))[1])==want and k.id_lists(ss(n))[2]==verdict
 pool=k.construct(good());screen=[{"stable_item_id":pool[0]["stable_item_id"],"eligible":False,"construction_status":"valid","construction_failure_label":None}];out=Path(tempfile.gettempdir())/"k2_phase_a_test_output";shutil.rmtree(out,ignore_errors=True);hf={"resolved_hf_revision":k.HF_REVISION,"resolved_snapshot_path":"x","tokenizer_class":"x","tokenizer_is_fast":True,"tokenizer_backend_class":"x","tokenizer_files":[]};m=k.write_phase_a_outputs(out,pool,screen,{},hf,{"actual_source_path":"x"})
 for f,key in [("candidate_pool.jsonl","candidate_pool_sha256"),("screening.jsonl","screening_artifact_sha256"),("eligible_ids.jsonl","eligible_id_list_sha256"),("final_confirmatory_ids.jsonl","final_confirmatory_id_list_sha256")]:assert k.sha((out/f).read_bytes())==m[key]
 with pytest.raises(ValueError):k.write_phase_a_outputs(out,pool,screen,{}, {"resolved_hf_revision":k.HF_REVISION}, {})
def test_complete_phase_a_manifest_and_runtime_package_provenance():
 packages=k.package_provenance()
 for key in ("python_version","torch_version","transformers_version","huggingface_hub_version","tokenizers_version"):assert isinstance(packages[key],str) and packages[key]
 pool=k.construct(good());screen=[{"stable_item_id":pool[0]["stable_item_id"],"eligible":False,"construction_status":"valid","construction_failure_label":None}]
 handoffs={seed:{"seed":seed,"zip_sha256":"zip-"+seed,"checkpoint_sha256":"checkpoint-"+seed} for seed in k.EXPECTED_ZIP_SHA256}
 hf={"resolved_hf_revision":k.HF_REVISION,"resolved_snapshot_path":"snapshot","tokenizer_class":"SyntheticTokenizer","tokenizer_is_fast":True,"tokenizer_backend_class":"SyntheticBackend","tokenizer_files":[]}
 runtime={**packages,"actual_source_path":"C:/frozen/source.jsonl","runtime_branch":"longterm-k-series-native-state-kinematics","runtime_git_head":k.AUTHORITY_COMMIT,"runtime_dirty_contract":["?? scripts/longterm_k2_exact_prefix_phase_a.py"],"allowed_untracked_policy":sorted(k.ALLOWED_UNTRACKED)}
 out=Path(tempfile.gettempdir())/"k2_complete_manifest_test_output";shutil.rmtree(out,ignore_errors=True);m=k.write_phase_a_outputs(out,pool,screen,handoffs,hf,runtime)
 assert m["schema_version"]==k.MANIFEST_SCHEMA and m["authority_prereg_commit"]==k.AUTHORITY_COMMIT and m["runtime_branch"]==runtime["runtime_branch"] and m["runtime_git_head"]==runtime["runtime_git_head"] and m["script_sha256"]==k.file_sha(P)
 assert m["observer_input_contract"]==k.OBSERVER_INPUT_CONTRACT and m["historical_a0_serialization_equivalence_claimed"] is False and m["source_path"]==runtime["actual_source_path"] and m["source_physical_sha256"]==k.SOURCE_PHYSICAL and m["source_semantic_sha256"]==k.SOURCE_SEMANTIC
 assert m["hf_model_id"]==k.HF_MODEL and m["requested_hf_revision"]==k.HF_REVISION and m["resolved_hf_revision"]==k.HF_REVISION and m["add_special_tokens"] is False and m["trust_remote_code"] is False and m["tokenizer_class"]=="SyntheticTokenizer"
 assert set(m["handoffs"])==set(k.EXPECTED_ZIP_SHA256) and all(m["handoffs"][seed]["zip_sha256"]=="zip-"+seed and m["handoffs"][seed]["checkpoint_sha256"]=="checkpoint-"+seed for seed in k.EXPECTED_ZIP_SHA256)
 assert m["common_encoder_digest"]==k.ENCODER_DIGEST and all(m[key] for key in ("candidate_pool_sha256","screening_artifact_sha256","eligible_id_list_sha256","final_confirmatory_id_list_sha256")) and (m["N_total_attempts"],m["N_construction_valid"],m["N_duplicate_excluded"],m["N_eligible"],m["N_final"],m["phase_a_verdict"])==(1,1,0,0,0,"INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE")
 assert all(m[key]==runtime[key] for key in ("python_version","torch_version","transformers_version","huggingface_hub_version","tokenizers_version","runtime_dirty_contract","allowed_untracked_policy"))
def test_cli_help_is_side_effect_free_and_frozen_options():
 output=scratch("help_output");before=set(SCRATCH_ROOT.rglob("*"));run=subprocess.run([sys.executable,str(P),"--help"],cwd=P.parents[1],capture_output=True,text=True)
 assert run.returncode==0 and "--source-data" in run.stdout and "--seed180-handoff" in run.stdout and "--hf-revision" in run.stdout and "--output-dir" in run.stdout and not output.exists() and set(SCRATCH_ROOT.rglob("*"))==before and "phase_a_manifest.json" not in run.stdout
 with pytest.raises(ValueError,match="HF_REVISION"):k.main(["--source-data","x","--seed180-handoff","x","--seed181-handoff","x","--seed182-handoff","x","--hf-revision","bad","--output-dir","x"])
 assert k.CLASS_ORDER==("REFUTE","NOT_ENTITLED","SUPPORT") and k.OBSERVER_INPUT_CONTRACT=="K2_PROSPECTIVE_LITERAL_PREFIX_WITH_FROZEN_A0_HEADS"

def test_one_prefix_tokenization_and_invalid_pool_before_screening():
 class Counting(BranchTok):
  def __init__(self):super().__init__(8);self.prefix_calls=0
  def __call__(self,text,**kw):
   if text.startswith("Claim:") and " This is false:" not in text and " A separate event:" not in text:self.prefix_calls+=1
   return super().__call__(text,**kw)
 tok=Counting();pool,bundles=k.prepare_candidate_pool(good(),tok);assert tok.prefix_calls==1 and len(bundles)==1
 bad,bundles=k.prepare_candidate_pool(good(),BranchTok(7));assert bad[0]["construction_status"]=="invalid" and bad[0]["construction_failure_label"]=="INVALID_CONTINUATION_TOKEN_CONTRACT" and not bundles

def test_source_path_and_output_safety():
 root=scratch("path_repo");root.mkdir(); expected=root/k.SOURCE_REL;expected.parent.mkdir(parents=True);expected.write_text("x")
 assert k.require_frozen_source_path(root,expected)==expected.resolve()
 with pytest.raises(ValueError,match="FROZEN"):k.require_frozen_source_path(root,scratch("copy.jsonl"))
 external=Path(tempfile.gettempdir())/"k2_output_safety_test";shutil.rmtree(external,ignore_errors=True);assert k.require_safe_output_dir(root,external)==external.resolve();external.mkdir();assert k.require_safe_output_dir(root,external)==external.resolve();(external/"x").write_text("x")
 with pytest.raises(ValueError):k.require_safe_output_dir(root,external)
 for protected in (root,root/"reports",root/"anything"):
  with pytest.raises(ValueError):k.require_safe_output_dir(root,protected)

def test_zip_member_safety_exact_manifest_and_all_seed_bindings(monkeypatch):
 for name in ("../x","/absolute/x","C:/x","C:\\x","foo\\bar"):
  with pytest.raises(ValueError):k._safe_member(name)
 cp=b"ckpt"; cp_sha=k.sha(cp)
 for seed in k.EXPECTED_ZIP_SHA256:
  manifest={"schema_version":"contramamba-handoff-v3","expected_commit":k.A0_COMMIT,"actual_commit":k.A0_COMMIT,"selected_checkpoint":{"path":"checkpoint.pt","sha256":cp_sha,"size_bytes":4}}
  p=scratch(seed+".zip");p.write_bytes(zip_bytes(manifest));monkeypatch.setitem(k.EXPECTED_ZIP_SHA256,seed,k.file_sha(p));monkeypatch.setitem(k.EXPECTED_CHECKPOINT_SHA256,seed,cp_sha);assert k.audit_handoff(p,seed)["manifest_member"]=="handoff_manifest.json"
 manifest={"schema_version":"contramamba-handoff-v3","expected_commit":k.A0_COMMIT,"actual_commit":k.A0_COMMIT,"selected_checkpoint":{"path":"checkpoint.pt","sha256":cp_sha,"size_bytes":4}}
 p=scratch("wrong_manifest.zip");b=io.BytesIO()
 with zipfile.ZipFile(b,"w") as z:z.writestr("nested/handoff_manifest.json",json.dumps(manifest));z.writestr("checkpoint.pt",cp)
 p.write_bytes(b.getvalue());monkeypatch.setitem(k.EXPECTED_ZIP_SHA256,"seed180",k.file_sha(p))
 with pytest.raises(ValueError,match="MANIFEST"):k.audit_handoff(p,"seed180")

def test_explicit_branch_mask_and_position_mismatches(monkeypatch):
 x=candidate();bundle=k.tokenize_prefix_bundle(BranchTok(),x["prefix_text"])
 good_branch={"input_ids":list(range(6))+[3]*8,"attention_mask":[1]*14,"position_ids":list(range(14))}
 for field in ("attention_mask","position_ids"):
  broken={key:list(value) for key,value in good_branch.items()};broken[field][0]=99
  calls=iter((good_branch,broken));monkeypatch.setattr(k,"_branch",lambda *_:next(calls))
  assert k.token_contract(x,BranchTok(),bundle)["failure"]=="INVALID_EXACT_PREFIX"

def test_one_seed_at_a_time_releases_before_next_loader(monkeypatch):
 events=[]
 class Model: pass
 monkeypatch.setattr(k,"load_authenticated_checkpoint",lambda handoff:events.append("load-"+handoff["seed"]) or {"model_state_dict":{}})
 monkeypatch.setattr(k,"build_a0_model",lambda snapshot,checkpoint:events.append("build") or Model())
 monkeypatch.setattr(k,"forward_prefix",lambda model,inputs,seed,checkpoint,device:{"seed":seed,"predicted_final_label":"SUPPORT"})
 monkeypatch.setattr(k.gc,"collect",lambda:events.append("gc"))
 outputs={}
 for seed in k.EXPECTED_ZIP_SHA256:outputs[seed]=k.screen_one_seed(seed,{"seed":seed,"checkpoint_sha256":seed},Path("snapshot"),{"x":{"prefix_bundle":{}}},"cpu")
 assert list(outputs)==["seed180","seed181","seed182"] and events==["load-seed180","build","gc","load-seed181","build","gc","load-seed182","build","gc"]

def test_git_provenance_allows_authority_commit_and_descendant(monkeypatch):
 root=Path.cwd();base={"status":"","head":k.AUTHORITY_COMMIT,"ancestry":0};calls=[]
 def check(argv,**kw):
  if argv[1:]==["status","--porcelain=v1"]:return base["status"]
  if argv[1:]==["branch","--show-current"]:return "longterm-k-series-native-state-kinematics\n"
  if argv[1:]==["rev-parse","HEAD"]:return base["head"]+"\n"
  raise AssertionError(argv)
 def call(argv,**kw):
  calls.append(argv)
  if argv==["git","merge-base","--is-ancestor",k.AUTHORITY_COMMIT,base["head"]]:return base["ancestry"]
  if argv==["git","ls-files","--error-unmatch","scripts/longterm_k2_exact_prefix_phase_a.py"]:return 0
  raise AssertionError(argv)
 monkeypatch.setattr(k.subprocess,"check_output",check);monkeypatch.setattr(k.subprocess,"call",call)
 assert k.git_provenance(root)["runtime_git_head"]==k.AUTHORITY_COMMIT
 base["head"]="0123456789abcdef0123456789abcdef01234567"
 assert k.git_provenance(root)["runtime_git_head"]==base["head"]
 assert calls==[["git","merge-base","--is-ancestor",k.AUTHORITY_COMMIT,k.AUTHORITY_COMMIT],["git","ls-files","--error-unmatch","scripts/longterm_k2_exact_prefix_phase_a.py"],["git","merge-base","--is-ancestor",k.AUTHORITY_COMMIT,base["head"]],["git","ls-files","--error-unmatch","scripts/longterm_k2_exact_prefix_phase_a.py"]]

def test_git_provenance_rejects_non_descendant_and_dirty_states(monkeypatch):
 root=Path.cwd();base={"status":"","head":"0123456789abcdef0123456789abcdef01234567","ancestry":1}
 def check(argv,**kw):
  if argv[1:]==["status","--porcelain=v1"]:return base["status"]
  if argv[1:]==["branch","--show-current"]:return "longterm-k-series-native-state-kinematics\n"
  if argv[1:]==["rev-parse","HEAD"]:return base["head"]+"\n"
  raise AssertionError(argv)
 def call(argv,**kw):
  if argv==["git","merge-base","--is-ancestor",k.AUTHORITY_COMMIT,base["head"]]:return base["ancestry"]
  if argv==["git","ls-files","--error-unmatch","scripts/longterm_k2_exact_prefix_phase_a.py"]:return 0
  raise AssertionError(argv)
 monkeypatch.setattr(k.subprocess,"check_output",check);monkeypatch.setattr(k.subprocess,"call",call)
 with pytest.raises(ValueError,match="GIT_PROVENANCE_MISMATCH"):k.git_provenance(root)
 base["ancestry"]=0
 for status in ("?? unexpected.txt\n","M  tracked.py\n"," M tracked.py\n"):
  base["status"]=status
  with pytest.raises(ValueError):k.git_provenance(root)
