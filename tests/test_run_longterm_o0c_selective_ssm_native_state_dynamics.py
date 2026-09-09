import ast, dataclasses, hashlib, importlib.util, json, subprocess, sys
from contextlib import contextmanager
from pathlib import Path
import pytest
import numpy as np
P=Path(__file__).parents[1]/"scripts"/"run_longterm_o0c_selective_ssm_native_state_dynamics.py"
spec=importlib.util.spec_from_file_location("runner",P); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
# This separately registered real_observer import cannot exercise the production
# o0c_observer first-load path, so it previously masked the missing registration.
OP=P.with_name("observe_longterm_o0c_selective_ssm_native_state_dynamics.py")
observer_spec=importlib.util.spec_from_file_location("real_observer",OP); real_observer=importlib.util.module_from_spec(observer_spec); sys.modules[observer_spec.name]=real_observer; observer_spec.loader.exec_module(real_observer)

def argv(extra=()):
 a=["runner.py","--output-dir","out","--run-name","run","--expected-runner-commit","head","--expected-runner-sha256","sha","--expected-observer-commit",r.OBSERVER_COMMIT,"--expected-observer-sha256","observer",*extra,"--exact-command",""]
 a[-1]=r.canonical_exact_command(a); return a
def test_import_safety():
 tree=ast.parse(P.read_text(encoding="utf-8")); names={n.name for n in tree.body if isinstance(n,ast.Import) for n in n.names}; assert "torch" not in names and "transformers" not in names
def test_production_load_observer_registers_before_dataclass_execution(monkeypatch):
 monkeypatch.delitem(sys.modules,"o0c_observer",raising=False)
 observer=r.load_observer()
 assert observer.__name__=="o0c_observer"
 assert sys.modules["o0c_observer"] is observer
 assert dataclasses.is_dataclass(observer._SyntheticTraceBinding)
def test_canonical_argv_preserves_program_and_strings():
 a=argv(["--run-name","A B/Case"]); a[-1]=r.canonical_exact_command(a); assert json.loads(r.canonical_exact_command(a))[0]=="runner.py"; assert r.parse_args(a).run_name=="A B/Case"
@pytest.mark.parametrize("bad",[["x"],["x","--exact-command","a","--exact-command","b"]])
def test_exact_command_cardinality(bad):
 with pytest.raises(r.ContractError): r.canonical_exact_command(bad)
@pytest.mark.parametrize("flag",["--out","--run","--expected-runner-sh"])
def test_abbreviations_rejected(flag):
 with pytest.raises(SystemExit): r.parse_args([flag,"x"])
@pytest.mark.parametrize("field",["run-name","exact-command","expected-runner-commit","expected-runner-sha256","expected-observer-commit","expected-observer-sha256"])
def test_blank_semantic_cli_rejected(field):
 a=argv(); i=a.index("--"+field)+1; a[i]=" ";
 if field!="exact-command": a[-1]=r.canonical_exact_command(a)
 with pytest.raises(r.ContractError): r.parse_args(a)
def test_serialization_no_terminal_newline(): assert r.serialize({"claim":"c","reference_sufficient":"e"},"reference_sufficient")=="Claim: c\nEvidence: e"

class Mix:
 def __init__(self,i): self.layer_idx=i
class Block:
 def __init__(self,i,m=None): self.layer_idx=i; self.mixer=Mix(i) if m is None else m
class ML(list): pass
class Model:
 def __init__(self,layers): self.layers=layers; self.config=type("C",(),{"num_hidden_layers":len(layers)})(); self.training=False
 def modules(self): return [self,*sum(([b,b.mixer] for b in self.layers),[])]
def test_exact_modulelist_contract_and_descriptors():
 d,m=r.discover_layers(Model(ML([Block(0),Block(1)])),Model,Block,Mix,ML); assert d==[{"layer_index":0,"layer_role":"mamba_mixer"},{"layer_index":1,"layer_role":"mamba_mixer"}] and len(m)==2
@pytest.mark.parametrize("maker",[lambda:Model([Block(0)]),lambda:Model(ML([Block(1)])),lambda:Model(ML([Block(0,type("Other",(),{"layer_idx":0})())]))])
def test_bad_layer_shapes_rejected(maker):
 with pytest.raises(r.ContractError): r.discover_layers(maker(),Model,Block,Mix,ML)
def test_extra_and_duplicate_mixers_rejected():
 m=Model(ML([Block(0),Block(1)])); m.modules=lambda:[m,*m.layers,*[x.mixer for x in m.layers],Mix(9)]
 with pytest.raises(r.ContractError): r.discover_layers(m,Model,Block,Mix,ML)
 same=Mix(0); m=Model(ML([Block(0,same),Block(1,same)]))
 with pytest.raises(r.ContractError): r.discover_layers(m,Model,Block,Mix,ML)

class Tensor:
 def __init__(self,device="cpu",dtype="f32",grad=False,floaty=True): self.device=type("D",(),{"type":device})(); self.dtype=dtype; self.requires_grad=grad; self._float=floaty
 def is_floating_point(self): return self._float
class Tiny(Model):
 def __init__(self,ps,bs=(),sub=False): super().__init__(ML([Block(0)])); self.ps=ps; self.bs=bs; self.sub=type("S",(),{"training":sub})()
 def eval(self): self.training=False; return self
 def requires_grad_(self,x):
  for p in self.ps: p.requires_grad=x
 def parameters(self): return iter(self.ps)
 def buffers(self): return iter(self.bs)
 def modules(self): return [self,self.sub]
class Torch: float32="f32"
@pytest.mark.parametrize("model",[Tiny([Tensor(),Tensor(dtype="bad")]),Tiny([Tensor(),Tensor(device="cuda")]),Tiny([Tensor(),Tensor(grad=True)]),Tiny([Tensor()],[Tensor(dtype="bad")]),Tiny([Tensor()],[Tensor(device="cuda",floaty=False)]),Tiny([Tensor()],sub=True)])
def test_full_model_state_attack_rejected(model):
 # The grad attack restores grad after model.requires_grad_(False), mimicking a hostile module.
 if model.ps[-1].requires_grad: model.requires_grad_=lambda x: None
 with pytest.raises(r.ContractError): r.validate_model_state(model,Torch)
def test_full_model_state_accepts_integer_buffer(): r.validate_model_state(Tiny([Tensor()],[Tensor(dtype="i64",floaty=False)]),Torch)

def row(pair="o0b_pair_001"):
 return {"schema_version":"v","pair_id":pair,"claim":"c","reference_sufficient":"e","insufficient_matched":"e","paraphrase_sufficient":"e","surface_null_matched":"e"}
def frozen(): return {"full_serialized_token_ids":[1,2],"full_offset_mapping":[[0,19],[19,20]],"full_token_count":2,"terminal_index":1,"evidence_char_start":19,"evidence_start_index":1,"evidence_start_offset_start":19,"evidence_start_offset_end":20,"evidence_token_count":1,"boundary_crossing":False}
def artifact(rows): return {"pairs":[{"pair_id":x["pair_id"],"conditions":{c:frozen() for c in r.CONDITION_ORDER}} for x in rows]}
class Tok:
 def __init__(self,fast=True,change=None): self.is_fast=fast; self.change=change; self.calls=[]
 def __call__(self,text,**kw):
  self.calls.append(kw); z=frozen();
  if self.change: z[self.change]=False if self.change=="boundary_crossing" else (9 if self.change in ("full_token_count","terminal_index","evidence_start_index") else [[0,8],[9,10]] if self.change=="full_offset_mapping" else [9,2])
  return {"input_ids":z["full_serialized_token_ids"],"offset_mapping":z["full_offset_mapping"]}
def test_token_path_contract_and_calls():
 rows=[row(p) for p in r.PAIR_ORDER]; t=Tok(); r.validate_tokens(t,rows,artifact(rows)); assert len(t.calls)==12 and all(x=={"add_special_tokens":False,"return_offsets_mapping":True} for x in t.calls)
@pytest.mark.parametrize("fast,change",[(False,None),(True,"full_serialized_token_ids"),(True,"full_offset_mapping"),(True,"full_token_count"),(True,"terminal_index"),(True,"evidence_start_index"),(True,"boundary_crossing")])
def test_token_attacks_fail(fast,change):
 rows=[row(p) for p in r.PAIR_ORDER]; a=artifact(rows)
 # IDs/offsets are runtime-tokenizer attacks; the remaining fields are frozen-artifact attacks.
 if change in ("full_token_count","terminal_index","evidence_start_index","boundary_crossing"):
  for pair in a["pairs"]:
   for condition in r.CONDITION_ORDER: pair["conditions"][condition][change]=3 if change!="boundary_crossing" else True
  change=None
 with pytest.raises(r.ContractError): r.validate_tokens(Tok(fast,change),rows,a)

def anchors():
 class O: ANCHOR_ORDER=("anchor_pre_minus_1","anchor_divergence","anchor_post_plus_1","anchor_post_plus_2","anchor_post_plus_4","anchor_terminal")
 pairs=[]
 for p in r.PAIR_ORDER:
  conditions={"reference_sufficient":{"full_serialized_token_ids":[0]*40,"terminal_index":39}}; comps={}
  for n,c,d in (("comparison-A","insufficient_matched",25),("comparison-B","paraphrase_sufficient",18),("comparison-C","surface_null_matched",17)):
   x=[0]*40; x[d]=1; conditions[c]={"full_serialized_token_ids":x,"terminal_index":39}; comps[c]={"reference_condition":"reference_sufficient","first_divergent_token_index":d,"anchor_indices":dict(zip(O.ANCHOR_ORDER,[d-1,d,d+1,d+2,d+4,39]))}
  pairs.append({"pair_id":p,"conditions":conditions,"comparisons_to_reference":comps})
 return O,{"pairs":pairs}
def test_anchor_actual_construction():
 O,a=anchors(); x=r.comparison_anchors(O,a); assert len(x)==9 and [x[(r.PAIR_ORDER[0],c)]["anchor_divergence"] for c,_ in r.COMPARISONS]==[25,18,17]
@pytest.mark.parametrize("edit",["first_divergent_token_index","anchor_pre_minus_1","anchor_divergence","anchor_post_plus_1","anchor_post_plus_2","anchor_post_plus_4","anchor_terminal"])
def test_anchor_attacks_fail(edit):
 O,a=anchors(); c=a["pairs"][0]["comparisons_to_reference"]["insufficient_matched"]; target=c if edit=="first_divergent_token_index" else c["anchor_indices"]; target[edit]+=1
 with pytest.raises(r.ContractError): r.comparison_anchors(O,a)

def test_preflight_failures_before_loaders():
 class O:
  DATASET_SHA256="d"; VALIDATION_ARTIFACT_SHA256="v"
  def observer_script_identity(self): return {"observer_script_sha256":"o"}
  def runtime_gate(self): raise AssertionError("runtime should not run")
 ns=type("N",(),{"output_dir":"preflight-never-created","run_name":"x","exact_command":"x","expected_runner_commit":"wrong","expected_runner_sha256":"x","expected_observer_commit":r.OBSERVER_COMMIT,"expected_observer_sha256":"o"})()
 with pytest.raises(r.ContractError): r.preflight(ns,O(),head=lambda:"head",file_digest=lambda _:"x")

class _LoaderSpies:
 def __init__(self): self.tokenizer_calls=[]; self.model_calls=[]
 def factories(self):
  calls=self
  class T:
   @classmethod
   def from_pretrained(cls,*a,**kw): calls.tokenizer_calls.append((a,kw)); raise AssertionError("tokenizer loader ran")
  class M:
   @classmethod
   def from_pretrained(cls,*a,**kw): calls.model_calls.append((a,kw)); raise AssertionError("model loader ran")
  return type("Torch",(),{"float32":"f32"}),T,M,Block,Mix,ML

class _PreflightObserver:
 DATASET_SHA256=hashlib.sha256(b"dataset").hexdigest(); VALIDATION_ARTIFACT_SHA256=hashlib.sha256(b"validation").hexdigest()
 def __init__(self,identity="observer",runtime=None): self.identity=identity; self.runtime=runtime or (lambda:None)
 def observer_script_identity(self): return {"observer_script_sha256":self.identity}
 def runtime_gate(self): return self.runtime()

VALIDATED_OBSERVER_SHA256="94f1d65408980dfe70a5cd855035b5273c7d82744288bb06eaba01d519f97067"

def test_observer_binding_is_exact_validated_commit():
 assert r.OBSERVER_COMMIT=="60a53b6f5d1db8d7cbecded5b94d5231adcfc520"

def test_preflight_rejects_expected_observer_commit_mismatch():
 ns=_preload_ns(); ns.expected_observer_commit="wrong"
 with pytest.raises(r.ContractError,match="observer commit"):
  r.preflight(ns,_PreflightObserver(),head=lambda:"head",file_digest=lambda _:"runner",canonical_input=_canonical_preload_input)

def test_preflight_rejects_expected_observer_sha256_mismatch_against_identity():
 ns=_preload_ns(); ns.expected_observer_sha256="wrong"
 with pytest.raises(r.ContractError,match="observer SHA256"):
  r.preflight(ns,_PreflightObserver(VALIDATED_OBSERVER_SHA256),head=lambda:"head",file_digest=lambda _:"runner",canonical_input=_canonical_preload_input)

def test_preflight_accepts_exact_validated_observer_sha256_identity():
 ns=_preload_ns(); ns.expected_observer_sha256=VALIDATED_OBSERVER_SHA256
 identity,_,_=r.preflight(ns,_PreflightObserver(VALIDATED_OBSERVER_SHA256),head=lambda:"head",file_digest=lambda _:"runner",canonical_input=_canonical_preload_input)
 assert identity["observer_script_sha256"]==VALIDATED_OBSERVER_SHA256

def _preload_ns():
 output=Path.cwd()/"__o0c_test_only_published__"
 assert not output.exists()
 a=argv(["--output-dir",str(output)]); ns=r.parse_args(a); ns.expected_runner_sha256="runner"; return ns

def _canonical_preload_input(path): return b"dataset" if path==r.DATASET else b"validation"

def _git_result(args,code=0,stdout=b""): return subprocess.CompletedProcess(args,code,stdout,b"")

def test_windows_crlf_worktree_uses_git_canonical_bytes_for_runner_parsing(monkeypatch):
 canonical=b"a\nb\n"; worktree=b"a\r\nb\r\n"; dataset=r.DATASET; validation=r.VALIDATION
 assert hashlib.sha256(canonical).hexdigest()!=hashlib.sha256(worktree).hexdigest()
 def fake_run(args,**kw):
  if args[1]=="ls-files": return _git_result(args)
  if args[1]=="diff": return _git_result(args)
  assert args[1]=="show"
  return _git_result(args,stdout=canonical)
 monkeypatch.setattr(r.subprocess,"run",fake_run)
 assert r.git_canonical_input(dataset)==canonical
 seen=[]
 monkeypatch.setattr(r,"read_inputs",lambda _o,d,v: (seen.append((d,v)),(_ for _ in ()).throw(r.ContractError("parsed canonical bytes")))[1])
 observer=_PreflightObserver(); observer.DATASET_SHA256=hashlib.sha256(canonical).hexdigest(); observer.VALIDATION_ARTIFACT_SHA256=hashlib.sha256(canonical).hexdigest(); spies=_LoaderSpies(); ns=_preload_ns()
 with pytest.raises(r.ContractError,match="parsed canonical bytes"): r.run(ns,observer=observer,factories=spies.factories(),head=lambda:"head",file_digest=lambda _:"runner")
 assert seen==[(canonical,canonical)] and spies.tokenizer_calls==[] and spies.model_calls==[]

@pytest.mark.parametrize("failure",("dirty","show"))
def test_git_canonical_input_failures_reject_before_loaders(monkeypatch,failure):
 canonical=b"dataset"
 def fake_run(args,**kw):
  if args[1]=="ls-files": return _git_result(args)
  if args[1]=="diff": return _git_result(args,1 if failure=="dirty" else 0)
  return _git_result(args,1 if failure=="show" else 0,canonical)
 monkeypatch.setattr(r.subprocess,"run",fake_run); spies=_LoaderSpies(); ns=_preload_ns(); observer=_PreflightObserver()
 with pytest.raises(r.ContractError): r.run(ns,observer=observer,factories=spies.factories(),head=lambda:"head",file_digest=lambda _:"runner")
 assert spies.tokenizer_calls==[] and spies.model_calls==[]

@pytest.mark.parametrize("case",("runner_head","runner_sha","observer_commit","observer_sha","blank_run_name","bad_exact_command","output_exists","staging_exists","dataset_sha","validation_sha","runtime_gate"))
def test_every_preload_gate_keeps_both_loaders_uninvoked(case):
 """Each failure reaches the real gate (or its real CLI gate) before factories load ML."""
 spies=_LoaderSpies()
 if case=="bad_exact_command":
  bad=argv(); bad[-1]="not-the-canonical-command"
  with pytest.raises(r.ContractError): r.parse_args(bad)
  assert spies.tokenizer_calls==[] and spies.model_calls==[]
  return
 ns=_preload_ns(); observer=_PreflightObserver(); head=lambda:"head"; digests=lambda _:"runner"
 if case=="runner_head": ns.expected_runner_commit="wrong"
 elif case=="runner_sha": ns.expected_runner_sha256="wrong"
 elif case=="observer_commit": ns.expected_observer_commit="wrong"
 elif case=="observer_sha": ns.expected_observer_sha256="wrong"
 elif case=="blank_run_name": ns.run_name=" "
 elif case=="output_exists": Path(ns.output_dir).mkdir()
 elif case=="staging_exists": Path(ns.output_dir).with_name(Path(ns.output_dir).name+".staging").mkdir()
 elif case=="dataset_sha": observer.DATASET_SHA256="wrong"
 elif case=="validation_sha": observer.VALIDATION_ARTIFACT_SHA256="wrong"
 elif case=="runtime_gate": observer.runtime=lambda: (_ for _ in ()).throw(r.ContractError("runtime"))
 try:
  with pytest.raises(r.ContractError): r.run(ns,observer=observer,factories=spies.factories(),head=head,file_digest=digests,canonical_input=_canonical_preload_input)
  assert spies.tokenizer_calls==[] and spies.model_calls==[]
 finally:
  for p in (Path(ns.output_dir),Path(ns.output_dir).with_name(Path(ns.output_dir).name+".staging")):
   if p.exists(): p.rmdir()

def _e2e_rows_and_artifact():
 rows=[row(p) for p in r.PAIR_ORDER]; pairs=[]
 for pair in r.PAIR_ORDER:
  conditions={}; comparisons={}
  for i,condition in enumerate(r.CONDITION_ORDER):
   ids=[0]*40
   if i: ids={"insufficient_matched":[0]*25+[1]+[0]*14,"paraphrase_sufficient":[0]*18+[2]+[0]*21,"surface_null_matched":[0]*17+[3]+[0]*22}[condition]
   conditions[condition]={"full_serialized_token_ids":ids,"full_offset_mapping":[[0,19]]+[[19+i,20+i] for i in range(39)],"full_token_count":40,"terminal_index":39,"evidence_char_start":19,"evidence_start_index":1,"evidence_start_offset_start":19,"evidence_start_offset_end":20,"evidence_token_count":39,"boundary_crossing":False}
  for name,condition,d in (("comparison-A","insufficient_matched",25),("comparison-B","paraphrase_sufficient",18),("comparison-C","surface_null_matched",17)):
   comparisons[condition]={"reference_condition":"reference_sufficient","first_divergent_token_index":d,"anchor_indices":{"anchor_pre_minus_1":d-1,"anchor_divergence":d,"anchor_post_plus_1":d+1,"anchor_post_plus_2":d+2,"anchor_post_plus_4":d+4,"anchor_terminal":39}}
  pairs.append({"pair_id":pair,"conditions":conditions,"comparisons_to_reference":comparisons})
 return rows,{"pairs":pairs}

def test_fake_success_orchestrates_exactly_twelve_members_and_seven_artifacts(monkeypatch):
 rows,fixture=_e2e_rows_and_artifact(); events=[]; loader=_LoaderSpies()
 expected_comparison_anchors=None; actual_comparison_anchors=r.comparison_anchors
 def capture_comparison_anchors(observer,artifact):
  nonlocal expected_comparison_anchors
  expected_comparison_anchors=actual_comparison_anchors(observer,artifact)
  return expected_comparison_anchors
 monkeypatch.setattr(r,"comparison_anchors",capture_comparison_anchors)
 class FakeTokenizer:
  is_fast=True
  def __init__(self): self.calls=[]
  def __call__(self,text,**kw):
   self.calls.append((text,kw)); pair,condition=divmod(len(self.calls)-1,4); z=fixture["pairs"][pair]["conditions"][r.CONDITION_ORDER[condition]]; return {"input_ids":z["full_serialized_token_ids"],"offset_mapping":z["full_offset_mapping"]}
 class AutoTokenizer:
  @classmethod
  def from_pretrained(cls,*a,**kw): loader.tokenizer_calls.append((a,kw)); loader.tokenizer=FakeTokenizer(); return loader.tokenizer
 class Param:
  device=type("D",(),{"type":"cpu"})(); dtype="f32"; requires_grad=False
 class FModel(Model):
  def __init__(self): super().__init__(ML([Block(0),Block(1)])); self.ps=[Param()]; self.forward_calls=[]
  def eval(self): self.training=False; return self
  def requires_grad_(self,value): self.ps[0].requires_grad=value
  def parameters(self): return iter(self.ps)
  def buffers(self): return iter(())
  def __call__(self,**kw):
   assert mode.active; self.forward_calls.append(kw); events.append("forward")
   inst=observer.active; assert inst is not None; self.forward_instances.append(inst.n)
   inst.snapshots={(1,layer,token):(layer,token) for layer in range(2) for token in range(40)}
 class MambaModel(FModel):
  @classmethod
  def from_pretrained(cls,*a,**kw): loader.model_calls.append((a,kw)); loader.model=cls(); loader.model.forward_instances=[]; return loader.model
 class Mode:
  active=False
  @contextmanager
  def inference_mode(self):
   self.active=True
   try: yield
   finally: self.active=False
 mode=Mode()
 class Torch(Mode):
  float32="f32"; long="i64"; __version__="torch-version"
  def tensor(self,value,**kw): return (value,kw)
 torch=Torch(); torch.inference_mode=mode.inference_mode
 class Instance:
  def __init__(self,n): self.n=n; self.snapshots={}
  @contextmanager
  def capture(self):
   assert observer.active is None; observer.active=self
   try: yield
   finally: observer.active=None
 class Observer(_PreflightObserver):
  SCHEMA_VERSION="schema"; EXPERIMENT_NAME="experiment"; SCIENTIFIC_DESIGN_AUTHORITY_COMMIT="242ad9ed70fc995ebda560911a7d0dfd2f18f9b3"; IMPLEMENTATION_AUTHORITY_COMMIT="implementation"; DATASET_PATH="data/longterm_o0b_matched_controls_v1.jsonl"; DATASET_SHA256="75a675bee49cb26eb0935d364f0f5d090922dd01576dfc23294961b28394aec2"; VALIDATION_ARTIFACT_PATH="reports/longterm_o0b_matched_controls_v1_validation.json"; VALIDATION_ARTIFACT_SHA256="e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff"; MODEL_ID=TOKENIZER_ID="state-spaces/mamba-130m-hf"; MODEL_REVISION=TOKENIZER_REVISION="5708daa364c50b880e7bd92eab456e0d34492ee9"; EXPECTED_VERSIONS={"python":"py","numpy":"np","torch":"torch","transformers":"transformers"}; MAMBA_MODULE="mamba"; MAMBA_SHA256="msha"; MAMBA_BYTES=1; CACHE_MODULE="cache"; CACHE_SHA256="csha"; CACHE_BYTES=2; CAPTURE_QUALNAME="capture"; CAPTURE_LINE=3; ANCHOR_ORDER=("anchor_pre_minus_1","anchor_divergence","anchor_post_plus_1","anchor_post_plus_2","anchor_post_plus_4","anchor_terminal"); REQUIRED_ARTIFACTS=real_observer.REQUIRED_ARTIFACTS
  def __init__(self): super().__init__(); self.active=None; self.instances=[]; self.validations=[]
  def create_native_state_observer(self,registration,enabled):
   assert enabled and list(registration.values())==[{"layer_index":0,"layer_role":"mamba_mixer"},{"layer_index":1,"layer_role":"mamba_mixer"}]; x=Instance(len(self.instances)); self.instances.append(x); return x
  def validate_captures(self,captures,n,descriptors): assert n==40 and list(captures)==[0,1] and len(captures[0])==40; self.validations.append(captures); events.append("capture")
  def validate_member_orchestration(self,members,descriptors): assert len(members)==12 and len({id(x["captures"]) for x in members})==12; events.append("orchestration")
  def state_rows(self,trajectories,descriptors): assert len(trajectories)==24; events.append("state_rows"); return "rows","vectors"
  def measurements(self,state_rows,vectors,anchors):
   assert anchors is expected_comparison_anchors and len(anchors)==9
   assert set(anchors)=={(pair,comparison) for pair in r.PAIR_ORDER for comparison in ("comparison-A","comparison-B","comparison-C")}
   assert dict(r.COMPARISONS)=={"comparison-A":"insufficient_matched","comparison-B":"paraphrase_sufficient","comparison-C":"surface_null_matched"}
   assert all(tuple(anchor)==self.ANCHOR_ORDER for anchor in anchors.values())
   assert [(anchors[(r.PAIR_ORDER[0],comparison)]["anchor_divergence"],anchors[(r.PAIR_ORDER[0],comparison)]["anchor_pre_minus_1"]) for comparison in ("comparison-A","comparison-B","comparison-C")]==[(25,24),(18,17),(17,16)]
   self.anchors=anchors; events.append("measurements"); return "measured"
  def build_summary(self,measured,state_rows): events.append("summary"); return "summary"
  def build_manifest(self,fields): self.manifest_fields=fields; events.append("manifest"); return "manifest"
  def build_bundle(self,*x): events.append("bundle"); return {name:name for name in real_observer.REQUIRED_ARTIFACTS}
  def validate_bundle(self,bundle): assert tuple(bundle)==tuple(real_observer.REQUIRED_ARTIFACTS) and len(bundle)==7; events.append("validate_bundle")
  def publish_bundle(self,path,bundle): assert tuple(bundle)==tuple(real_observer.REQUIRED_ARTIFACTS) and len(bundle)==7; events.append("publish")
 observer=Observer()
 factories=(torch,AutoTokenizer,MambaModel,Block,Mix,ML)
 ns=_preload_ns(); digests=lambda p:{r.DATASET:"dataset",r.VALIDATION:"validation"}.get(p,"runner")
 monkeypatch.setattr(r,"read_inputs",lambda _o,_d,_v: (rows,fixture))
 r.run(ns,observer=observer,factories=factories,head=lambda:"head",file_digest=digests,runtime_info=("py","np","transformers","dist","root"))
 expected=("state-spaces/mamba-130m-hf",); revision="5708daa364c50b880e7bd92eab456e0d34492ee9"
 assert loader.tokenizer_calls==[(expected,{"revision":revision,"trust_remote_code":False,"use_fast":True})]
 assert loader.model_calls==[(expected,{"revision":revision,"torch_dtype":"f32","trust_remote_code":False})]
 assert len(loader.tokenizer.calls)==12 and all(kw=={"add_special_tokens":False,"return_offsets_mapping":True} for _,kw in loader.tokenizer.calls)
 assert len(observer.instances)==12 and len({id(x) for x in observer.instances})==12 and len(loader.model.forward_calls)==12
 assert loader.model.forward_instances==list(range(12))
 assert [call["input_ids"][0][0] for call in loader.model.forward_calls]==[fixture["pairs"][p]["conditions"][c]["full_serialized_token_ids"] for p in range(3) for c in r.CONDITION_ORDER]
 assert all(x["use_cache"] is False and x["input_ids"][0] and x["input_ids"][1]=={"dtype":"i64"} for x in loader.model.forward_calls)
 assert len(observer.validations)==12 and events.index("orchestration")>max(i for i,x in enumerate(events) if x=="forward")
 assert events[-7:]==["state_rows","measurements","summary","manifest","bundle","validate_bundle","publish"]
 f=observer.manifest_fields; assert tuple(real_observer.REQUIRED_ARTIFACTS)==("manifest.json","state_rows.jsonl","full_recurrent_states.npz","paired_measurements.jsonl","summary.json","report.md","SHA256SUMS.txt") and f["scientific_design_authority_commit"]==observer.SCIENTIFIC_DESIGN_AUTHORITY_COMMIT=="242ad9ed70fc995ebda560911a7d0dfd2f18f9b3" and f["observer_implementation_commit"]==r.OBSERVER_COMMIT and f["implementation_authority_commit"]==observer.IMPLEMENTATION_AUTHORITY_COMMIT and f["dataset_path"]==observer.DATASET_PATH and f["dataset_sha256"]==observer.DATASET_SHA256 and f["validation_artifact_path"]==observer.VALIDATION_ARTIFACT_PATH and f["validation_artifact_sha256"]==observer.VALIDATION_ARTIFACT_SHA256 and f["exact_command"]==ns.exact_command and f["run_name"]=="run" and [f[x] for x in ("equivalence_gate_status","capture_completeness_status","provenance_status","execution_status")]==["PASS_EXACT_EQUIVALENCE_NONINTERFERENCE","PASS_COMPLETE_NATIVE_STATE_CAPTURE","PASS_PROVENANCE_VALIDATED","PASS_EXECUTION_COMPLETE"] and f["blocker"] is None and f["required_artifacts"]==list(real_observer.REQUIRED_ARTIFACTS)

def test_runner_real_manifest_and_publication_integration_normalizes_str_subclasses(monkeypatch):
 """The runner must deliver exact built-in strings to the unmodified observer."""
 rows,fixture=_e2e_rows_and_artifact()
 class TorchVersionLike(str): pass
 class RuntimeValue(str): pass
 class Tokenizer:
  is_fast=True
  def __init__(self): self.calls=0
  def __call__(self,_text,**_kw):
   pair,condition=divmod(self.calls,4); self.calls+=1
   return dict(fixture["pairs"][pair]["conditions"][r.CONDITION_ORDER[condition]]) | {"input_ids":fixture["pairs"][pair]["conditions"][r.CONDITION_ORDER[condition]]["full_serialized_token_ids"],"offset_mapping":fixture["pairs"][pair]["conditions"][r.CONDITION_ORDER[condition]]["full_offset_mapping"]}
 class AutoTokenizer:
  @classmethod
  def from_pretrained(cls,*_a,**_kw): return Tokenizer()
 class Param:
  device=type("D",(),{"type":"cpu"})(); dtype="f32"; requires_grad=False
 class MambaModel:
  calls=0
  def __init__(self): self.layers=ML([Block(0),Block(1)]); self.config=type("C",(),{"num_hidden_layers":2})(); self.training=False; self.parameter=Param()
  @classmethod
  def from_pretrained(cls,*_a,**_kw): return cls()
  def eval(self): self.training=False; return self
  def requires_grad_(self,value): self.parameter.requires_grad=value
  def parameters(self): return iter((self.parameter,))
  def buffers(self): return iter(())
  def modules(self): return [self,*self.layers,*[x.mixer for x in self.layers]]
  def __call__(self,**kw):
   pair,condition=divmod(self.calls,4); type(self).calls+=1; inst=observer.active; assert inst is not None
   divergence=(40,25,18,17)[condition]
   inst.snapshots={(1,layer,t):np.asarray([t+1+pair/10+layer,2*(t+1)+pair/10+layer]+([condition/10,condition/20] if t>=divergence else [0,0]),dtype="<f4") for layer in range(2) for t in range(40)}
 class Torch:
  float32="f32"; long="i64"; __version__=TorchVersionLike("2.10.0+cpu")
  def tensor(self,value,**_kw): return value
  @contextmanager
  def inference_mode(self): yield
 torch=Torch()
 class Instance:
  def __init__(self): self.snapshots={}
  @contextmanager
  def capture(self):
   assert observer.active is None; observer.active=self
   try: yield
   finally: observer.active=None
 class Observer:
  def __init__(self): self.active=None; self.manifest_fields=None
  def __getattr__(self,name): return getattr(real_observer,name)
  def runtime_gate(self): return None
  def create_native_state_observer(self,_registration,enabled): assert enabled is True; return Instance()
  def build_manifest(self,fields): self.manifest_fields=dict(fields); return real_observer.build_manifest(fields)
 observer=Observer()
 assert type(torch.__version__) is not str
 monkeypatch.setattr(r,"read_inputs",lambda _o,_d,_v:(rows,fixture))
 output=Path.cwd()/"__o0c_runner_manifest_integration_test_output__"; assert not output.exists() and not output.with_name(output.name+".staging").exists(); a=argv(["--output-dir",str(output)]); ns=r.parse_args(a)
 ns.expected_runner_commit="723f655295430c434da4ecccdad79934360c9dcc"; ns.expected_runner_sha256=r.digest(P); ns.expected_observer_sha256=real_observer.observer_script_identity()["observer_script_sha256"]
 runtime_info=tuple(RuntimeValue(value) for value in ("3.12.13","2.0.2","5.0.0",str(Path.cwd().resolve()),str(Path.cwd().resolve())))
 try:
  r.run(ns,observer=observer,factories=(torch,AutoTokenizer,MambaModel,Block,Mix,ML),head=lambda:"723f655295430c434da4ecccdad79934360c9dcc",file_digest=r.digest,canonical_input=r.git_canonical_input,runtime_info=runtime_info)
  fields=observer.manifest_fields; required=("observed_python_version","observed_numpy_version","observed_torch_version","observed_transformers_version","transformers_distribution_root","transformers_import_root")
  assert all(type(fields[key]) is str for key in required)
  assert real_observer.build_manifest(fields)==fields
  for key in (*required,"exact_command","run_name"):
   for bad_value in (""," ","unknown","n/a"):
    with pytest.raises(real_observer.ContractError,match="manifest required string"):
     real_observer.build_manifest({**fields,key:bad_value})
  published={name:(output/name).read_bytes() for name in real_observer.REQUIRED_ARTIFACTS}
  assert len(published)==7 and set(published)==set(real_observer.REQUIRED_ARTIFACTS)
  real_observer.validate_bundle(published)
  assert not output.with_name(output.name+".staging").exists()
 finally:
  if output.exists():
   for name in real_observer.REQUIRED_ARTIFACTS: (output/name).unlink()
   output.rmdir()
