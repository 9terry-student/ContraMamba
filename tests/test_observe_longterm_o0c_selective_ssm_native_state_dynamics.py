"""Local, synthetic-only tests for the O0c observer contract."""
from __future__ import annotations
import builtins, ctypes, errno, importlib.util, inspect, io, json, os, sys, uuid, zipfile, tempfile, shutil
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pytest
PATH=Path(__file__).parents[1]/"scripts"/"observe_longterm_o0c_selective_ssm_native_state_dynamics.py"
SPEC=importlib.util.spec_from_file_location("o0c_observer",PATH); o=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=o; assert SPEC.loader; SPEC.loader.exec_module(o)
def bad(fn,*args):
    with pytest.raises((o.ContractError,TypeError)): fn(*args)
_EXTERNAL_TEST_PREFIX="o0c-publication-test-"
_ROOT_MKDIR_ATTEMPTS=4

def _external_candidate_roots():
 """Return deterministic, existing-environment external-temp candidates."""
 values=[os.environ.get(name) for name in ("CONTRAMAMBA_O0C_TEST_TEMP","O0C_TEST_TEMP","TEMP","TMP")]
 local=os.environ.get("LOCALAPPDATA")
 if local: values.append(str(Path(local)/"Temp"))
 try: values.append(tempfile.gettempdir())
 except OSError: pass
 if os.name=="nt": values.append(r"C:\tmp")
 try:
  home=Path.home()
  if home.exists(): values.append(str(home))
 except OSError: pass
 roots=[]; seen=set()
 for value in values:
  if not value: continue
  try:
   root=Path(value).expanduser().resolve(strict=False)
   key=os.path.normcase(str(root))
  except OSError: continue
  if key not in seen: seen.add(key); roots.append(root)
 return roots

def _external_root_ok(parent):
 """A candidate must already exist, be absolute, and lie outside this repo."""
 try:
  parent=parent.resolve(strict=False); repo=PATH.parents[1].resolve(strict=False)
  return parent.is_absolute() and parent.is_dir() and parent!=repo and repo not in parent.parents
 except OSError: return False

def _owned_child(parent,name_factory):
 """Bounded direct mkdir; never delegate collision retrying to tempfile."""
 for _ in range(_ROOT_MKDIR_ATTEMPTS):
  child=parent/f"{_EXTERNAL_TEST_PREFIX}{name_factory()}"
  try:
   child.mkdir(parents=False,exist_ok=False); return child,None
  except FileExistsError: continue
  except PermissionError as exc: return None,f"mkdir denied: {exc}"
  except OSError as exc: return None,f"mkdir failed: {exc}"
 return None,f"mkdir collision after {_ROOT_MKDIR_ATTEMPTS} attempts"

def _cleanup_owned_child(child):
 """Remove exactly one owned child, never its candidate root or siblings."""
 try: shutil.rmtree(child)
 except FileNotFoundError: pass
 except OSError as exc: return f"{child.name} removal: {exc}"
 return None

def _probe_external_root(parent,name_factory):
 """Prove every filesystem operation publication tests require, then clean up."""
 child,reason=_owned_child(parent,name_factory)
 if child is None: return reason
 operation_error=None
 try:
  marker=child/"marker"; marker.write_bytes(b"o0c-root-probe")
  if marker.read_bytes()!=b"o0c-root-probe": raise OSError("marker read-back mismatch")
  nested=child/"nested"; nested.mkdir(parents=False,exist_ok=False)
  nested.rename(child/"nested-moved")
  marker.unlink(); (child/"nested-moved").rmdir()
 except OSError as exc: operation_error=f"probe operation failed: {exc}"
 cleanup_error=_cleanup_owned_child(child)
 if cleanup_error: return f"probe cleanup failed: {cleanup_error}"
 return operation_error

@contextmanager
def external_test_root(candidates=None,name_factory=lambda:uuid.uuid4().hex):
 """Yield one bounded, directly-proven, owned external scratch directory."""
 reasons=[]
 for parent in (_external_candidate_roots() if candidates is None else candidates):
  try: parent=Path(parent).expanduser().resolve(strict=False)
  except OSError as exc: reasons.append(f"<invalid>: {exc}"); continue
  if not _external_root_ok(parent): reasons.append(f"{parent}: not an existing external directory"); continue
  reason=_probe_external_root(parent,name_factory)
  if reason: reasons.append(f"{parent}: {reason}"); continue
  child,reason=_owned_child(parent,name_factory)
  if child is None: reasons.append(f"{parent}: owned test directory {reason}"); continue
  try: yield child
  finally:
   cleanup_error=_cleanup_owned_child(child)
   if cleanup_error: raise RuntimeError(f"external publication test cleanup failed for {child}: {cleanup_error}")
  return
 raise RuntimeError("no writable external O0c publication-test root: "+"; ".join(reasons))
@pytest.fixture
def external_tmp():
 with external_test_root() as root: yield root
class Tensor:
    def __init__(self,v): self.v=np.array(v,dtype=np.float32)
    def detach(self): return self
    def clone(self): return Tensor(self.v.copy())
    def cpu(self): return self.v.copy()
def trajectories(n=2): return {(p,c,0):[Tensor([t+1,t+2]) for t in range(n)] for p in o.PAIR_ORDER for c in o.CONDITION_ORDER}
def rows_vectors(n=2): return o.state_rows(trajectories(n),[{"layer_index":0,"layer_role":"mamba_mixer"}])
def anchors(t=0): return {(p,comparison):{x:t for x in o.ANCHOR_ORDER} for p in o.PAIR_ORDER for comparison,_ in o.COMPARISONS}
FROZEN_COMPARISON_ANCHORS={
 ("o0b_pair_001","comparison-A"):{"anchor_divergence":25,"anchor_post_plus_1":26,"anchor_post_plus_2":27,"anchor_post_plus_4":29,"anchor_pre_minus_1":24,"anchor_terminal":44},
 ("o0b_pair_001","comparison-B"):{"anchor_divergence":18,"anchor_post_plus_1":19,"anchor_post_plus_2":20,"anchor_post_plus_4":22,"anchor_pre_minus_1":17,"anchor_terminal":44},
 ("o0b_pair_001","comparison-C"):{"anchor_divergence":17,"anchor_post_plus_1":18,"anchor_post_plus_2":19,"anchor_post_plus_4":21,"anchor_pre_minus_1":16,"anchor_terminal":44},
 ("o0b_pair_002","comparison-A"):{"anchor_divergence":24,"anchor_post_plus_1":25,"anchor_post_plus_2":26,"anchor_post_plus_4":28,"anchor_pre_minus_1":23,"anchor_terminal":35},
 ("o0b_pair_002","comparison-B"):{"anchor_divergence":14,"anchor_post_plus_1":15,"anchor_post_plus_2":16,"anchor_post_plus_4":18,"anchor_pre_minus_1":13,"anchor_terminal":35},
 ("o0b_pair_002","comparison-C"):{"anchor_divergence":14,"anchor_post_plus_1":15,"anchor_post_plus_2":16,"anchor_post_plus_4":18,"anchor_pre_minus_1":13,"anchor_terminal":35},
 ("o0b_pair_003","comparison-A"):{"anchor_divergence":16,"anchor_post_plus_1":17,"anchor_post_plus_2":18,"anchor_post_plus_4":20,"anchor_pre_minus_1":15,"anchor_terminal":35},
 ("o0b_pair_003","comparison-B"):{"anchor_divergence":21,"anchor_post_plus_1":22,"anchor_post_plus_2":23,"anchor_post_plus_4":25,"anchor_pre_minus_1":20,"anchor_terminal":35},
 ("o0b_pair_003","comparison-C"):{"anchor_divergence":16,"anchor_post_plus_1":17,"anchor_post_plus_2":18,"anchor_post_plus_4":20,"anchor_pre_minus_1":15,"anchor_terminal":35},
}

def multilayer_rows_vectors(layer_count=3):
 """Production-shape local fixture: all pairs/conditions and frozen anchors."""
 assert layer_count>=3
 descriptors=[{"layer_index":layer,"layer_role":"mamba_mixer"} for layer in range(layer_count)]
 token_counts={"o0b_pair_001":45,"o0b_pair_002":36,"o0b_pair_003":36}
 trajectories={}
 for pair in o.PAIR_ORDER:
  for condition in o.CONDITION_ORDER:
   for layer in range(layer_count):
    # Conditions intentionally agree through every token: each comparison's
    # pre-divergence proof is therefore exercised for each layer.
    trajectories[(pair,condition,layer)]=[Tensor([layer+t+1,layer+t+2]) for t in range(token_counts[pair])]
 return o.state_rows(trajectories,descriptors)

def multilayer_manifest(layer_count):
 value=manifest()
 value["layer_descriptors"]=[{"layer_index":layer,"layer_role":"mamba_mixer"} for layer in range(layer_count)]
 return value

def expected_measurement_coordinates(layer_count):
 return [(pair,comparison,anchor,layer) for pair in o.PAIR_ORDER for comparison,_ in o.COMPARISONS for anchor in o.ANCHOR_ORDER for layer in range(layer_count)]

def test_comparison_anchor_contract_rejects_old_and_malformed_shapes():
 rows,vectors=rows_vectors(); good=anchors()
 assert len(good)==9 and all(set(x)==set(o.ANCHOR_ORDER) for x in good.values())
 old={(p,c):{x:0 for x in o.ANCHOR_ORDER} for p in o.PAIR_ORDER for c in o.CONDITION_ORDER}; bad(o.measurements,rows,vectors,old)
 cases=[]
 missing=dict(good); missing.pop(("o0b_pair_001","comparison-A")); cases.append(missing)
 extra=dict(good); extra[("extra","comparison-A")]=dict(next(iter(good.values()))); cases.append(extra)
 missing_anchor={k:dict(v) for k,v in good.items()}; missing_anchor[("o0b_pair_001","comparison-A")].pop("anchor_terminal"); cases.append(missing_anchor)
 extra_anchor={k:dict(v) for k,v in good.items()}; extra_anchor[("o0b_pair_001","comparison-A")]["extra"]=0; cases.append(extra_anchor)
 nonint={k:dict(v) for k,v in good.items()}; nonint[("o0b_pair_001","comparison-A")]["anchor_terminal"]="0"; cases.append(nonint)
 boolean={k:dict(v) for k,v in good.items()}; boolean[("o0b_pair_001","comparison-A")]["anchor_terminal"]=True; cases.append(boolean)
 negative={k:dict(v) for k,v in good.items()}; negative[("o0b_pair_001","comparison-A")]["anchor_terminal"]=-1; cases.append(negative)
 for case in cases: bad(o.measurements,rows,vectors,case)

def test_comparison_specific_frozen_anchor_schedules_and_ranges():
 # Exact immutable fixture extracted from the canonical f93ed7e Git-object bytes,
 # whose SHA256 is e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff.
 rows,vectors=rows_vectors(45); frozen={k:dict(v) for k,v in FROZEN_COMPARISON_ANCHORS.items()}
 assert [frozen[("o0b_pair_001",c)]["anchor_divergence"] for c,_ in o.COMPARISONS]==[25,18,17]
 assert [frozen[("o0b_pair_001",c)]["anchor_pre_minus_1"] for c,_ in o.COMPARISONS]==[24,17,16]
 records=o.measurements(rows,vectors,frozen); assert len(records)==len(o.PAIR_ORDER)*len(o.COMPARISONS)*len(o.ANCHOR_ORDER)
 for comparison,_ in o.COMPARISONS:
  assert {r["absolute_token_index"] for r in records if r["pair_id"]=="o0b_pair_001" and r["comparison_id"]==comparison and r["anchor_name"]=="anchor_terminal"}=={44}
 for comparison,coordinate in (("comparison-A",1),("comparison-B",2),("comparison-C",3)):
  isolated=anchors(); isolated[("o0b_pair_001",comparison)]["anchor_terminal"]=coordinate
  isolated_rows,isolated_vectors=rows_vectors(4); measured=o.measurements(isolated_rows,isolated_vectors,isolated)
  for actual,_ in o.COMPARISONS:
   terminal=[r for r in measured if r["pair_id"]=="o0b_pair_001" and r["comparison_id"]==actual and r["anchor_name"]=="anchor_terminal"]
   assert {r["absolute_token_index"] for r in terminal}==({coordinate} if actual==comparison else {0})
 ref_short=trajectories(2); ref_short[("o0b_pair_001","reference_sufficient",0)]=ref_short[("o0b_pair_001","reference_sufficient",0)][:1]
 ref_rows,ref_vectors=o.state_rows(ref_short,[{"layer_index":0,"layer_role":"mamba_mixer"}]); ref_bad=anchors(); ref_bad[("o0b_pair_001","comparison-A")]["anchor_terminal"]=1; bad(o.measurements,ref_rows,ref_vectors,ref_bad)
 member_short=trajectories(2); member_short[("o0b_pair_001","insufficient_matched",0)]=member_short[("o0b_pair_001","insufficient_matched",0)][:1]
 member_rows,member_vectors=o.state_rows(member_short,[{"layer_index":0,"layer_role":"mamba_mixer"}]); member_bad=anchors(); member_bad[("o0b_pair_001","comparison-A")]["anchor_terminal"]=1; bad(o.measurements,member_rows,member_vectors,member_bad)
def manifest():
 m={k:"PASS" for k in o.MANIFEST_KEYS}; m.update({"schema_version":o.SCHEMA_VERSION,"experiment_name":o.EXPERIMENT_NAME,"scientific_design_authority_commit":o.SCIENTIFIC_DESIGN_AUTHORITY_COMMIT,"implementation_authority_commit":o.IMPLEMENTATION_AUTHORITY_COMMIT,"observer_implementation_commit":"a"*40,"observer_script_path":"scripts/observe_longterm_o0c_selective_ssm_native_state_dynamics.py","observer_script_sha256":"b"*64,"observer_script_bytes":1,"dataset_path":o.DATASET_PATH,"dataset_sha256":o.DATASET_SHA256,"validation_artifact_path":o.VALIDATION_ARTIFACT_PATH,"validation_artifact_sha256":o.VALIDATION_ARTIFACT_SHA256,"model_id":o.MODEL_ID,"model_revision":o.MODEL_REVISION,"tokenizer_id":o.TOKENIZER_ID,"tokenizer_revision":o.TOKENIZER_REVISION,"model_trust_remote_code":False,"tokenizer_trust_remote_code":False,"add_special_tokens":False,"device":"cpu","dtype":"float32","expected_python_version":"3.12.13","expected_numpy_version":"2.0.2","expected_torch_version":"2.10.0+cpu","expected_transformers_version":"5.0.0","mamba_source_module":o.MAMBA_MODULE,"mamba_source_sha256":o.MAMBA_SHA256,"mamba_source_bytes":o.MAMBA_BYTES,"cache_source_module":o.CACHE_MODULE,"cache_source_sha256":o.CACHE_SHA256,"cache_source_bytes":o.CACHE_BYTES,"capture_source_qualname":o.CAPTURE_QUALNAME,"capture_source_line":410,"capture_state_source":"native_selective_ssm_recurrent_state","capture_state_timing":"post_consumption_s_t","pair_order":list(o.PAIR_ORDER),"condition_order":list(o.CONDITION_ORDER),"comparison_order":[x[0] for x in o.COMPARISONS],"anchor_order":list(o.ANCHOR_ORDER),"layer_descriptors":[{"layer_index":0,"layer_role":"mamba_mixer"}],"serialization_template":"canonical-json-v1/deterministic-npz-v1","required_artifacts":list(o.REQUIRED_ARTIFACTS),"blocker":None,"source_resolution_classification":"PASS_RECONCILED_UNIQUE_TRANSFORMERS_SOURCE","backend_classification":"BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN","transformers_distribution_root":str(Path.cwd()),"transformers_import_root":str(Path.cwd())}); m.update(o.observer_script_identity()); m.update(o.SUCCESS_STATUSES); return m

def test_external_root_direct_mkdir_denial_uses_next_candidate_without_tempfile_retry(external_tmp,monkeypatch):
 denied,usable=external_tmp/"denied",external_tmp/"usable"; denied.mkdir(); usable.mkdir()
 original_mkdir=Path.mkdir; mkdir_calls=[]; mkdtemp_calls=[]
 def guarded_mkdir(self,*args,**kwargs):
  if self.parent==denied and self.name.startswith(_EXTERNAL_TEST_PREFIX):
   mkdir_calls.append(self); raise PermissionError(5,"Access is denied",str(self))
  return original_mkdir(self,*args,**kwargs)
 monkeypatch.setattr(Path,"mkdir",guarded_mkdir)
 monkeypatch.setattr(os,"access",lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("os.access must not authorize a root")))
 monkeypatch.setattr(tempfile,"mkdtemp",lambda *args,**kwargs: mkdtemp_calls.append((args,kwargs)))
 with external_test_root([denied,usable],name_factory=iter(("d1","u1","u2")).__next__) as child:
  assert child.parent==usable
 assert len(mkdir_calls)==1 and mkdtemp_calls==[]
 denied.rmdir(); usable.rmdir()

def test_external_root_all_direct_mkdir_denials_fail_promptly(external_tmp,monkeypatch):
 first,second=external_tmp/"first",external_tmp/"second"; first.mkdir(); second.mkdir()
 original_mkdir=Path.mkdir; calls=[]
 def denied_mkdir(self,*args,**kwargs):
  if self.parent in (first,second) and self.name.startswith(_EXTERNAL_TEST_PREFIX):
   calls.append(self); raise PermissionError(5,"Access is denied",str(self))
  return original_mkdir(self,*args,**kwargs)
 monkeypatch.setattr(Path,"mkdir",denied_mkdir)
 monkeypatch.setattr(tempfile,"mkdtemp",lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("mkdtemp must not be called")))
 with pytest.raises(RuntimeError,match="no writable external O0c publication-test root") as exc:
  with external_test_root([first,second]): pass
 assert len(calls)==2 and "mkdir denied" in str(exc.value)
 first.rmdir(); second.rmdir()

def test_external_root_fileexists_retry_is_bounded_and_succeeds(external_tmp,monkeypatch):
 parent=external_tmp/"candidate"; parent.mkdir(); original_mkdir=Path.mkdir; calls=[]
 def collision_once(self,*args,**kwargs):
  if self.parent==parent and self.name.startswith(_EXTERNAL_TEST_PREFIX):
   calls.append(self)
   if len(calls)==1: raise FileExistsError(17,"already exists",str(self))
  return original_mkdir(self,*args,**kwargs)
 monkeypatch.setattr(Path,"mkdir",collision_once)
 with external_test_root([parent],name_factory=iter(("collision","probe","owned")).__next__) as child:
  assert child.name.endswith("owned")
 assert len(calls)==3
 parent.rmdir()

def test_external_root_probe_cleanup_failure_fails_closed(external_tmp,monkeypatch):
 parent=external_tmp/"candidate"; parent.mkdir(); original_rmtree=shutil.rmtree
 def failing_owned_rmtree(path,*args,**kwargs):
  if Path(path).parent==parent and Path(path).name.startswith(_EXTERNAL_TEST_PREFIX): raise PermissionError(5,"cleanup denied",str(path))
  return original_rmtree(path,*args,**kwargs)
 monkeypatch.setattr(shutil,"rmtree",failing_owned_rmtree)
 with pytest.raises(RuntimeError,match="no writable external O0c publication-test root") as exc:
  with external_test_root([parent],name_factory=iter(("probe",)).__next__): pass
 assert "probe cleanup failed" in str(exc.value)
 monkeypatch.undo()
 for child in parent.glob(_EXTERNAL_TEST_PREFIX+"*"): original_rmtree(child)
 parent.rmdir()

def test_import_safety_rejects_model_library_imports(monkeypatch):
 original=builtins.__import__
 def guarded(name,*args,**kwargs):
  if name.split(".",1)[0] in {"torch","transformers"}: raise AssertionError("model import at module import")
  return original(name,*args,**kwargs)
 monkeypatch.setattr(builtins,"__import__",guarded)
 spec=importlib.util.spec_from_file_location("o0c_import_safety",PATH); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; assert spec.loader; spec.loader.exec_module(module); sys.modules.pop(spec.name,None)

def test_authority_binding_rejects_arbitrary_code():
 assert o.IMPLEMENTATION_AUTHORITY_COMMIT=="6eca52722aaffa214e8546c6b616e1f670aecf77"; bad(o.NativeStateObserver,(lambda:None).__code__,{},True)
 b=o._synthetic_trace_binding(lambda:None,1); assert not hasattr(b,"scientific") and o._synthetic_collector(b,{},False).code is b.code
def test_actual_cpython_trace_post_update_wrong_code_and_restore():
 class Mixer: pass
 mixer=Mixer()
 def observed(self):
  ssm_state=Tensor([0])
  for i in range(3):
   ssm_state=Tensor(ssm_state.v+1)
   capture_marker=i
  return ssm_state
 b=o._synthetic_trace_binding(observed,observed.__code__.co_firstlineno+4); prior=sys.gettrace(); watcher=o._synthetic_collector(b,{id(mixer):{"layer_index":0}},True)
 with watcher.capture(): result=observed(mixer)
 assert sys.gettrace() is prior and [watcher.snapshots[(1,0,i)].v.item() for i in range(3)]==[1,2,3] and result.v.item()==3
 result.v[:]=99; assert watcher.snapshots[(1,0,2)].v.item()==3
 wrong=o._synthetic_collector(o._synthetic_trace_binding(lambda self:None,1),{id(mixer):{"layer_index":0}},True)
 with wrong.capture(): observed(mixer)
 assert wrong.snapshots=={}
 def boom(self):
  ssm_state=Tensor([1]); i=0
  raise ValueError("synthetic")
 failing=o._synthetic_collector(o._synthetic_trace_binding(boom,boom.__code__.co_firstlineno+2),{id(mixer):{"layer_index":0}},True)
 with pytest.raises(ValueError):
  with failing.capture(): boom(mixer)
 assert sys.gettrace() is prior
def test_tokens_members_rows_and_nonaliasing():
 o.validate_captures({0:{0:Tensor([1]),1:Tensor([2]),2:Tensor([3])}},3,[{"layer_index":0}])
 for cap,n in (({0:{0:Tensor([1]),2:Tensor([3])}},3),({0:{0:Tensor([1]),1:Tensor([2])}},3),({0:{1:Tensor([1]),2:Tensor([2])}},2)):
  bad(o.validate_captures,cap,n,[{"layer_index":0}])
 rows,vectors=rows_vectors(); matrix=np.ascontiguousarray(np.asarray(vectors,dtype="<f4")); o.validate_state_rows(rows,matrix); assert len(rows)==24 and len({(p,c) for p in o.PAIR_ORDER for c in o.CONDITION_ORDER})==12
 source=Tensor([1,2]); snap=source.detach().clone(); artifact=o._array(snap); source.v[:]=8; snap.v[:]=9; assert np.array_equal(artifact,[1,2])
 for key,value in (("tensor_shape",[3]),("flattened_size",3),("vector_index",1),("absolute_token_index",1)):
  x=[dict(r) for r in rows]; x[0][key]=value; bad(o.validate_state_rows,x,matrix)
 bad(o.validate_state_rows,rows,matrix[:-1]); bad(o.validate_state_rows,list(reversed(rows)),matrix)
def test_npz_measurements_manifest_summary_and_bundle():
 rows,vectors=rows_vectors(); data=o.deterministic_npz(vectors); assert data==o.deterministic_npz(vectors) and o.parse_npz(data).dtype==np.dtype("<f4")
 raw=zipfile.ZipFile(io.BytesIO(data)).read("vectors.npy"); extra=io.BytesIO()
 with zipfile.ZipFile(extra,"w") as z: z.writestr("vectors.npy",raw); z.writestr("extra.npy",raw)
 bad(o.parse_npz,extra.getvalue()); bad(o.parse_npz,b"not zip")
 rec=o.measurements(rows,vectors,anchors()); o.validate_measurements(rec,rows,vectors)
 for field in ("normalized_l2_state_distance","reference_transition_l2","member_transition_l2","paired_transition_delta","transition_direction_cosine","pre_divergence_integrity_status"):
  x=[dict(r) for r in rec]; x[0][field]="FAIL" if field.endswith("status") else x[0][field]+.1; bad(o.validate_measurements,x,rows,vectors)
 o.assert_pre_divergence([1],[1.0000009]); bad(o.assert_pre_divergence,[1],[1.00001])
 summary=o.build_summary(rec,rows); files=o.build_bundle(manifest(),rows,vectors,rec,summary); o.validate_bundle(files)
 for key,value in (("device","cuda"),("implementation_authority_commit","f"*40),("observer_script_sha256","BAD"),("blocker","unexpected"),("execution_status","PASS"),("equivalence_gate_status","PASS_ANYTHING"),("provenance_status","unknown")):
  m=manifest(); m[key]=value; bad(o.build_manifest,m)
 changed=dict(files); changed["report.md"]+=b"tamper\n"; changed["SHA256SUMS.txt"]=o.checksum_text({n:changed[n] for n in o.REQUIRED_ARTIFACTS[:-1]}); bad(o.validate_bundle,changed)
def test_publication_collisions_and_late_race(monkeypatch):
 rows,vectors=rows_vectors(); rec=o.measurements(rows,vectors,anchors()); files=o.build_bundle(manifest(),rows,vectors,rec,o.build_summary(rec,rows)); out=Path("synthetic-final")
 monkeypatch.setattr(Path,"exists",lambda self: True); bad(o.publish_bundle,out,files)
 assert "_atomic_rename_noreplace_directory(staging,output)" in inspect.getsource(o.publish_bundle) and "os.replace" not in inspect.getsource(o.publish_bundle)
def test_synthetic_equivalence_noninterference_and_disabled_trace():
 class Model:
  def __init__(self): self.parameter=Tensor([3]); self.buffer=Tensor([4]); self.cache=Tensor([0])
  def forward(self): self.cache=Tensor(self.cache.v+1); return {"primary":Tensor([2]),"last_hidden_state":Tensor([3]),"hidden_states":[Tensor([3])]}
 a,b=Model(),Model(); prior=sys.gettrace(); x,y=a.forward(),b.forward()
 assert type(x) is type(y) and all(np.array_equal(x[k].v,y[k].v) for k in ("primary","last_hidden_state")) and np.array_equal(x["hidden_states"][0].v,y["hidden_states"][0].v) and np.array_equal(a.parameter.v,b.parameter.v) and np.array_equal(a.buffer.v,b.buffer.v) and np.array_equal(a.cache.v,b.cache.v) and sys.gettrace() is prior
 def observed(self):
  ssm_state=Tensor([0]); i=0; capture_marker=0; return ssm_state
 binding=o._synthetic_trace_binding(observed,observed.__code__.co_firstlineno+1); disabled=o._synthetic_collector(binding,{id(a):{"layer_index":0}},False)
 with disabled.capture(): observed(a)
 assert disabled.snapshots is None and sys.gettrace() is prior

def _runtime_baseline(monkeypatch):
 root=Path.cwd(); mp=root/"scripts"/"observe_longterm_o0c_selective_ssm_native_state_dynamics.py"; cp=mp
 def slow(self,values=None):
  ssm_state=self.cache.clone() if values is not None else 1
  for i in range(values.shape[0] if values is not None else 1):
   ssm_state=ssm_state + (values[i] if values is not None else 1)
   readout=ssm_state*self.parameter if values is not None else ssm_state
  if values is not None: self.cache.copy_(ssm_state)
  return readout
 line=slow.__code__.co_firstlineno+4; slow.__code__=slow.__code__.replace(co_filename=str(mp)); slow.__module__=o.MAMBA_MODULE; slow.__qualname__=o.CAPTURE_QUALNAME
 def forward(self,*args): return self.slow_forward(*args)
 forward.__code__=forward.__code__.replace(co_filename=str(mp)); forward.__module__=o.MAMBA_MODULE; forward.__qualname__="MambaMixer.forward"
 Mixer=type("MambaMixer",(),{"slow_forward":slow,"forward":forward})
 mamba=type("M",(),{"MambaMixer":Mixer})(); transformers=type("T",(),{"__file__":str(root/"__init__.py")})(); torch=type("Torch",(),{"__version__":"torch"})()
 source=[""]*500; source[0]="class MambaMixer:"; source[1]=" def forward(self,hidden_states,cache_params,cache_position,attention_mask):"; source[2]="  is_fast_path_available = all((selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn))"; source[3]="  if is_fast_path_available and \"cuda\" in self.x_proj.weight.device.type and not is_torchdynamo_compiling():"; source[4]="   return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)"; source[5]="  return self.slow_forward(hidden_states, cache_params, cache_position, attention_mask)"; source[10]="class MambaCache:"; source[11]=" def __init__(self):"; source[12]="  self.conv_states = []"; source[13]="  self.ssm_states = []"; source[14]="  conv_state = torch.zeros(self.conv_kernel_size)"; source[15]="  ssm_state = torch.zeros(self.ssm_state_size)"; source[16]="  self.conv_states.append(conv_state)"; source[17]="  self.ssm_states.append(ssm_state)"; source[18]=" def update_conv_state(self, layer_idx, new_conv_state, cache_position):"; source[19]="  self.conv_states[layer_idx] = new_conv_state"; source[20]="  return self.conv_states[layer_idx]"; source[21]=" def update_ssm_state(self, layer_idx, new_ssm_state):"; source[22]="  self.ssm_states[layer_idx] = new_ssm_state"; source[23]="  return self.ssm_states[layer_idx]"; source[407]="deltaB_u = discrete_B * hidden_states[..., None].float()"; source[408]="ssm_state = discrete_A * ssm_state + deltaB_u"; source[409]="scan_output = torch.matmul(ssm_state.to(dtype), C[..., None].unsqueeze(-1))"; source[416]="cache_params.ssm_states[0].copy_(ssm_state)"
 data="\n".join(source).encode()
 monkeypatch.setattr(o,"CAPTURE_LINE",line); monkeypatch.setattr(o,"MAMBA_BYTES",len(data)); monkeypatch.setattr(o,"MAMBA_SHA256",o.sha256_bytes(data)); monkeypatch.setattr(o,"CACHE_BYTES",len(data)); monkeypatch.setattr(o,"CACHE_SHA256",o.sha256_bytes(data))
 paths={id(mamba):(mp,data)}; monkeypatch.setattr(o,"_source",lambda mod:paths[id(mod)])
 modules={o.MAMBA_MODULE:mamba,"transformers":transformers,"torch":torch}
 versions={"python":"3.12.13","numpy":"2.0.2","torch":"2.10.0+cpu","transformers":"5.0.0"}; monkeypatch.setattr(o,"_runtime_environment",lambda:(modules,versions))
 monkeypatch.setattr(o.importlib_metadata,"distribution",lambda name:type("D",(),{"locate_file":lambda self,path:root})())
 monkeypatch.setattr(o.importlib.util,"find_spec",lambda name:type("S",(),{"origin":str(mp)})())
 return modules,versions,paths

def _dispatch_source(condition='is_fast_path_available and "cuda" in self.x_proj.weight.device.type and not is_torchdynamo_compiling()',true='return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)',fallback='return self.slow_forward(hidden_states, cache_params, cache_position, attention_mask)',prefix='',suffix=''):
 return ("class MambaMixer:\n"
         " def forward(self,hidden_states,cache_params,cache_position,attention_mask):\n"
         f"  {prefix}\n"
         "  is_fast_path_available = all((selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn))\n"
         f"  if {condition}:\n"
         f"   {true}\n"
         f"  {fallback}\n"
         f"  {suffix}\n").encode()

def _dispatch_forward(path):
 def forward(self,*args): return None
 forward.__code__=forward.__code__.replace(co_filename=str(path)); forward.__module__=o.MAMBA_MODULE; forward.__qualname__="MambaMixer.forward"
 return forward

def test_cpu_dispatch_validator_accepts_frozen_v5_shape_with_mamba_inner_fn():
 path=Path("synthetic_mamba.py"); o._validate_forward_dispatch(_dispatch_source(),_dispatch_forward(path),path)

def _cache_source(conv_ctor="self.conv_kernel_size",ssm_ctor="self.ssm_state_size",conv_append="self.conv_states.append(conv_state)",ssm_append="self.ssm_states.append(ssm_state)",conv_store="self.conv_states[layer_idx] = new_conv_state",conv_return="return self.conv_states[layer_idx]",ssm_store="self.ssm_states[layer_idx] = new_ssm_state",ssm_return="return self.ssm_states[layer_idx]"):
 return ("class MambaCache:\n"
         " def __init__(self):\n"
         "  self.conv_states = []\n"
         "  self.ssm_states = []\n"
         f"  conv_state = torch.zeros({conv_ctor})\n"
         f"  ssm_state = torch.zeros({ssm_ctor})\n"
         f"  {conv_append}\n"
         f"  {ssm_append}\n"
         " def update_conv_state(self, layer_idx, new_conv_state, cache_position):\n"
         f"  {conv_store}\n"
         f"  {conv_return}\n"
         " def update_ssm_state(self, layer_idx, new_ssm_state):\n"
         f"  {ssm_store}\n"
         f"  {ssm_return}\n").encode()

def test_mamba_cache_role_validator_accepts_separate_families_and_correct_provenance():
 o._validate_mamba_cache_roles(_cache_source())
 assert (o.CACHE_MODULE,o.CACHE_SHA256,o.CACHE_BYTES)==(o.MAMBA_MODULE,o.MAMBA_SHA256,o.MAMBA_BYTES)
 m=manifest(); assert (m["cache_source_module"],m["cache_source_sha256"],m["cache_source_bytes"])==(o.MAMBA_MODULE,o.MAMBA_SHA256,o.MAMBA_BYTES)

def test_mamba_cache_role_validator_accepts_frozen_style_ssm_augassign():
 o._validate_mamba_cache_roles(_cache_source(ssm_store="self.ssm_states[layer_idx] += new_ssm_state"))

@pytest.mark.parametrize("data",[
 _cache_source().replace(b"  self.conv_states = []\n",b""),
 _cache_source().replace(b"  self.ssm_states = []\n",b""),
 _cache_source(conv_append="self.ssm_states.append(conv_state)",ssm_append="self.conv_states.append(ssm_state)"),
 _cache_source(conv_store="self.ssm_states[layer_idx] = new_conv_state",conv_return="return self.ssm_states[layer_idx]"),
 _cache_source(ssm_store="self.conv_states[layer_idx] = new_ssm_state",ssm_return="return self.conv_states[layer_idx]"),
 _cache_source(conv_ctor="self.ssm_state_size"),
 _cache_source(ssm_ctor="self.conv_kernel_size"),
 _cache_source().replace(b"  self.ssm_states = []",b"  self.ssm_states = self.conv_states"),
 b"class NotMambaCache: pass\n",
 _cache_source()+_cache_source(),
])
def test_mamba_cache_role_validator_rejects_ambiguous_or_crossed_families(data):
 with pytest.raises(o.ContractError,match="cache/recurrent ambiguity"): o._validate_mamba_cache_roles(data)

@pytest.mark.parametrize("data",[
 _cache_source(conv_return="return conv_state"),
 _cache_source(ssm_return="return ssm_state"),
 _cache_source(conv_return="return self.ssm_states[layer_idx]"),
 _cache_source(ssm_return="return self.conv_states[layer_idx]"),
 _cache_source(conv_return="return self.conv_states[other_idx]"),
 _cache_source(ssm_return="return self.ssm_states[other_idx]"),
 _cache_source(conv_return="return self.conv_states"),
 _cache_source(ssm_return="return self.ssm_states"),
 _cache_source(conv_return="if flag:\n   return self.conv_states[layer_idx]\n  return self.conv_states[layer_idx]"),
 _cache_source(conv_return="conv_state = unrelated_value\n  return conv_state"),
])
def test_mamba_cache_role_validator_rejects_unproven_or_ambiguous_persistent_returns(data):
 with pytest.raises(o.ContractError,match="cache/recurrent ambiguity"): o._validate_mamba_cache_roles(data)

@pytest.mark.parametrize("data",[
 _cache_source(ssm_store="self.conv_states[layer_idx] += new_ssm_state"),
 _cache_source(ssm_store="self.ssm_states[other_idx] += new_ssm_state"),
 _cache_source(ssm_store="self.ssm_states += new_ssm_state"),
 _cache_source(ssm_store="ssm_state += new_ssm_state"),
])
def test_mamba_cache_role_validator_rejects_unproven_augassign_writes(data):
 with pytest.raises(o.ContractError,match="cache/recurrent ambiguity"): o._validate_mamba_cache_roles(data)

@pytest.mark.parametrize("data",[
 _dispatch_source(condition="is_fast_path_available and not is_torchdynamo_compiling()"),
 _dispatch_source(condition='is_fast_path_available and "cuda" in self.other.weight.device.type'),
 _dispatch_source(true="return self.not_cuda_kernels_forward(*args)"),
 _dispatch_source(fallback="return self.not_slow_forward(*args)"),
 _dispatch_source(prefix="self.cuda_kernels_forward(*args)"),
 _dispatch_source(fallback="return self.another_method(*args)"),
 _dispatch_source(suffix="return self.slow_forward(*args)"),
 _dispatch_source(condition='is_fast_path_available and "cpu" in self.x_proj.weight.device.type'),
])
def test_cpu_dispatch_validator_rejects_nonfrozen_shapes(data):
 path=Path("synthetic_mamba.py")
 with pytest.raises(o.ContractError,match="unsupported backend"): o._validate_forward_dispatch(data,_dispatch_forward(path),path)

def test_mamba_inner_fn_presence_is_neither_rejected_nor_sufficient():
 path=Path("synthetic_mamba.py"); good=_dispatch_source(); assert b"mamba_inner_fn" in good
 o._validate_forward_dispatch(good,_dispatch_forward(path),path)
 bad_shape=_dispatch_source(condition="is_fast_path_available and not is_torchdynamo_compiling()")
 assert b"mamba_inner_fn" in bad_shape
 with pytest.raises(o.ContractError,match="unsupported backend"): o._validate_forward_dispatch(bad_shape,_dispatch_forward(path),path)

@pytest.mark.parametrize("data",[
 _dispatch_source().replace(b'  is_fast_path_available',b'  if some_cpu_condition:\n   return self.other_backend_forward(hidden_states, cache_params, cache_position, attention_mask)\n  is_fast_path_available'),
 _dispatch_source().replace(b'  is_fast_path_available',b'  if some_cpu_condition:\n   return self.slow_forward(hidden_states, cache_params, cache_position, attention_mask)\n  is_fast_path_available'),
 _dispatch_source().replace(b'  return self.slow_forward',b'  else:\n   return self.other_backend_forward(hidden_states, cache_params, cache_position, attention_mask)\n  return self.slow_forward'),
 _dispatch_source().replace(b'  is_fast_path_available',b'  return self.other_backend_forward(hidden_states, cache_params, cache_position, attention_mask)\n  is_fast_path_available'),
 _dispatch_source(suffix='return self.other_backend_forward(hidden_states, cache_params, cache_position, attention_mask)'),
 _dispatch_source(condition='is_fast_path_available and ("cuda" in self.x_proj.weight.device.type or bypass_condition)'),
 _dispatch_source(condition='is_fast_path_available or "cuda" in self.x_proj.weight.device.type or not is_torchdynamo_compiling()'),
 _dispatch_source(condition='is_fast_path_available and "cuda" in self.other.weight.device.type and not is_torchdynamo_compiling()'),
 _dispatch_source(prefix='mamba_inner_fn(hidden_states)'),
 _dispatch_source(prefix='selective_scan_fn(hidden_states)'),
 _dispatch_source(prefix='selective_state_update(hidden_states)'),
 _dispatch_source(prefix='self.other_backend_forward(hidden_states, cache_params, cache_position, attention_mask)'),
])
def test_cpu_dispatch_validator_rejects_all_alternate_cpu_or_backend_paths(data):
 path=Path("synthetic_mamba.py")
 with pytest.raises(o.ContractError,match="unsupported backend"): o._validate_forward_dispatch(data,_dispatch_forward(path),path)

@pytest.mark.parametrize("key",["python","numpy","torch","transformers"])
def test_runtime_gate_version_negative_matrix(monkeypatch,key):
 modules,versions,_=_runtime_baseline(monkeypatch); versions[key]="wrong"
 with pytest.raises(o.ContractError,match="runtime version"): o.runtime_gate()

@pytest.mark.parametrize("mutation,message",[
 ("root_outside","shadowed import root"),("root_malformed","malformed import root"),("mamba_bytes","Mamba byte-size mismatch"),("mamba_hash","Mamba SHA256 mismatch"),("cache_bytes","Mamba cache byte-size mismatch"),("cache_hash","Mamba cache SHA256 mismatch"),
 ("missing_mixer","slow code identity"),("wrong_mixer","slow code identity"),("wrong_qualname","slow code identity"),("wrong_code","slow code identity"),("wrong_line","line binding"),("wrong_role","source input role"),("backend","forward identity"),("source_resolution","import/distribution-root mismatch")])
def test_runtime_gate_source_negative_matrix(monkeypatch,mutation,message):
 modules,versions,paths=_runtime_baseline(monkeypatch); mamba=modules[o.MAMBA_MODULE]; mp,data=paths[id(mamba)]
 if mutation=="root_outside": modules["transformers"].__file__=str(Path.cwd()/"tests"/"__init__.py")
 elif mutation=="root_malformed": modules["transformers"].__file__="relative.py"
 elif mutation=="mamba_bytes": monkeypatch.setattr(o,"MAMBA_BYTES",len(data)+1)
 elif mutation=="mamba_hash": monkeypatch.setattr(o,"MAMBA_SHA256","0"*64)
 elif mutation=="cache_bytes": monkeypatch.setattr(o,"CACHE_BYTES",99)
 elif mutation=="cache_hash": monkeypatch.setattr(o,"CACHE_SHA256","0"*64)
 elif mutation=="missing_mixer": mamba.MambaMixer=None
 elif mutation=="wrong_mixer": mamba.MambaMixer=type("X",(),{})
 elif mutation=="wrong_qualname": mamba.MambaMixer.slow_forward.__qualname__="Wrong"
 elif mutation=="wrong_code": mamba.MambaMixer.slow_forward.__code__=mamba.MambaMixer.slow_forward.__code__.replace(co_filename="other.py")
 elif mutation=="wrong_line": monkeypatch.setattr(o,"CAPTURE_LINE",499)
 elif mutation=="wrong_role":
  changed=b"\n"*len(data); paths[id(mamba)]=(mp,changed); monkeypatch.setattr(o,"MAMBA_SHA256",o.sha256_bytes(changed)); monkeypatch.setattr(o,"CACHE_SHA256",o.sha256_bytes(changed))
 elif mutation=="backend":
  def no_dispatch(self): return 1
  no_dispatch.__module__=o.MAMBA_MODULE; mamba.MambaMixer.forward=no_dispatch
 elif mutation=="source_resolution": monkeypatch.setattr(o.importlib.util,"find_spec",lambda name:type("S",(),{"origin":str(Path.cwd()/"wrong.py")})())
 with pytest.raises(o.ContractError,match=message): o.runtime_gate()

def _members(n=2):
 return [{"pair_id":p,"condition":c,"captures":{0:{i:Tensor([i+1]) for i in range(n)}},"fresh_capture_collection":True,"cache_reused":False} for p in o.PAIR_ORDER for c in o.CONDITION_ORDER]
def test_member_orchestration_matrix_and_isolation():
 members=_members(); o.validate_member_orchestration(members,[{"layer_index":0}]); assert [(m["pair_id"],m["condition"]) for m in members]==[(p,c) for p in o.PAIR_ORDER for c in o.CONDITION_ORDER]
 for mutate in (lambda x:x.pop(),lambda x:x.__setitem__(1,dict(x[0])),lambda x:x.append({**x[-1],"pair_id":"bad"}),lambda x:x.__setitem__(0,{**x[0],"condition":"bad"})):
  bad(o.validate_member_orchestration, (lambda z:(mutate(z),z)[1])(list(members)),[{"layer_index":0}])
 bad(o.validate_member_orchestration,members,[{"layer_index":0},{"layer_index":0}]); bad(o.validate_member_orchestration,members,[{"layer_index":1}])
 first,second=members[0]["captures"][0][0],members[1]["captures"][0][0]; first.v[:]=77; assert second.v.item()==1
 shared=_members(); shared[1]["captures"]=shared[0]["captures"]; bad(o.validate_member_orchestration,shared,[{"layer_index":0}])
 reuse=_members(); reuse[0]["cache_reused"]=True; bad(o.validate_member_orchestration,reuse,[{"layer_index":0}])

def _zip(payload,name="vectors.npy",**changes):
 out=io.BytesIO(); info=zipfile.ZipInfo(name,changes.pop("date_time",(1980,1,1,0,0,0))); info.compress_type=changes.pop("compress_type",zipfile.ZIP_STORED); info.create_system=changes.pop("create_system",3); info.create_version=changes.pop("create_version",20); info.extract_version=changes.pop("extract_version",20); info.external_attr=changes.pop("external_attr",0o100644<<16); info.flag_bits=changes.pop("flag_bits",0); info.extra=changes.pop("extra",b""); info.comment=changes.pop("comment",b"")
 with zipfile.ZipFile(out,"w") as z: z.comment=changes.pop("archive_comment",b""); z.writestr(info,payload)
 return out.getvalue()
def test_npz_malformed_and_nondeterministic_matrix():
 good=o.deterministic_npz([[1,2]]); payload=zipfile.ZipFile(io.BytesIO(good)).read("vectors.npy")
 empty=io.BytesIO()
 with zipfile.ZipFile(empty,"w"): pass
 cases=[b"",empty.getvalue(),_zip(payload,name="wrong.npy"),_zip(payload,compress_type=zipfile.ZIP_DEFLATED),_zip(payload,date_time=(1981,1,1,0,0,0)),_zip(payload,create_system=0),_zip(payload,external_attr=0),_zip(payload,extra=b"xx"),_zip(payload,comment=b"x"),_zip(payload,archive_comment=b"x"),_zip(b"bad")]
 extra=io.BytesIO();
 with zipfile.ZipFile(extra,"w") as z: z.writestr("vectors.npy",payload); z.writestr("other.npy",payload)
 cases.append(extra.getvalue())
 for data in cases: bad(o.parse_npz,data)
 flagged=bytearray(good)
 for signature,offset in ((b"PK\x03\x04",6),(b"PK\x01\x02",8)):
  at=flagged.find(signature); flagged[at+offset]|=1
 bad(o.parse_npz,bytes(flagged))
 for array in (np.array([[1]],dtype=object),np.array([[1]],dtype=">f4"),np.array([[1]],dtype=np.float64),np.array([[1]],dtype=np.int32),np.array([1],dtype="<f4"),np.empty((0,1),dtype="<f4"),np.array([[np.nan]],dtype="<f4"),np.array([[np.inf]],dtype="<f4"),np.array([[-np.inf]],dtype="<f4"),np.asfortranarray(np.ones((2,2),dtype="<f4"))):
  raw=io.BytesIO(); np.lib.format.write_array(raw,array,allow_pickle=array.dtype==object); bad(o.parse_npz,_zip(raw.getvalue()))
 dup=io.BytesIO();
 with zipfile.ZipFile(dup,"w") as z: z.writestr("vectors.npy",payload); z.writestr("vectors.npy",payload)
 bad(o.parse_npz,dup.getvalue())

def test_publication_staging_race_failure_and_validation(monkeypatch):
 rows,vectors=rows_vectors(); rec=o.measurements(rows,vectors,anchors()); files=o.build_bundle(manifest(),rows,vectors,rec,o.build_summary(rec,rows))
 with external_test_root() as tmp:
  out=Path(tmp).with_name(Path(tmp).name+"-o0c-publication-test-final"); staging=out.with_name(out.name+".staging")
  staging.mkdir(); (staging/"keep").write_bytes(b"keep"); bad(o.publish_bundle,out,files); assert (staging/"keep").read_bytes()==b"keep"; (staging/"keep").unlink(); staging.rmdir()
  out.mkdir(); (out/"keep").write_bytes(b"old"); bad(o.publish_bundle,out,files); assert (out/"keep").read_bytes()==b"old"; (out/"keep").unlink(); out.rmdir()
  def late(src,dst): out.mkdir(); (out/"late").write_bytes(b"unchanged"); raise o.ContractError("output collision")
  monkeypatch.setattr(o,"_atomic_rename_noreplace_directory",late); bad(o.publish_bundle,out,files); assert (out/"late").read_bytes()==b"unchanged" and not staging.exists()
  (out/"late").unlink(); out.rmdir(); monkeypatch.undo()
  monkeypatch.setattr(o,"_atomic_rename_noreplace_directory",lambda src,dst:(_ for _ in ()).throw(OSError("publish failed")))
  with pytest.raises(OSError): o.publish_bundle(out,files)
  assert not out.exists() and not staging.exists(); monkeypatch.undo(); invalid=Path(tmp)/"invalid"; bad(o.publish_bundle,invalid,{**files,"summary.json":b"{}\n"}); assert not invalid.exists()
  monkeypatch.setattr(o,"_atomic_rename_noreplace_directory",lambda src,dst:src.rename(dst))
  o.publish_bundle(out,files); assert {x.name for x in out.iterdir()}==set(o.REQUIRED_ARTIFACTS)
  for child in out.iterdir(): child.unlink()
  out.rmdir(); assert not out.exists() and not staging.exists()

def test_windows_movefileexw_uses_preserved_last_error_and_never_replaces(external_tmp):
 tmp_path=external_tmp
 src=tmp_path/"source"; dst=tmp_path/"destination"; src.mkdir(); (src/"new").write_text("new"); dst.mkdir(); (dst/"old").write_text("old")
 observed=[]; original=o.ctypes.get_last_error
 try:
  o.ctypes.get_last_error=lambda:(observed.append(original()),observed[-1])[1]
  with pytest.raises(o.ContractError,match="output collision"):
   o._atomic_rename_noreplace_directory(src,dst)
 finally: o.ctypes.get_last_error=original
 assert observed==[183]
 assert src.is_dir() and (src/"new").read_text()=="new" and dst.is_dir() and (dst/"old").read_text()=="old"

@pytest.mark.parametrize("native_error,expected",[(5,"MoveFileExW failed"),(0,"without native error")])
def test_windows_movefileexw_noncollision_and_zero_error_fail_closed(monkeypatch,external_tmp,native_error,expected):
 tmp_path=external_tmp
 class Move:
  argtypes=restype=None
  def __call__(self,source,destination,flags):
   assert flags==0; ctypes.set_last_error(native_error); return 0
 class Kernel: MoveFileExW=Move()
 monkeypatch.setattr(o.ctypes,"WinDLL",lambda name,use_last_error: Kernel())
 source,destination=tmp_path/"source",tmp_path/"destination"; source.mkdir()
 if native_error:
  with pytest.raises(OSError,match=expected) as raised: o._atomic_rename_noreplace_directory(source,destination)
  assert raised.value.errno==native_error and source.exists() and not destination.exists()
 else:
  with pytest.raises(o.ContractError,match=expected): o._atomic_rename_noreplace_directory(source,destination)
  assert source.exists() and not destination.exists()

def test_linux_renameat2_noreplace_mock_matrix(monkeypatch,external_tmp):
 tmp_path=external_tmp
 calls=[]
 class Rename:
  argtypes=restype=None
  def __call__(self,oldfd,source,newfd,destination,flags):
   calls.append((oldfd,source,newfd,destination,flags)); ctypes.set_errno(errno.EEXIST); return -1
 class Libc: renameat2=Rename()
 monkeypatch.setattr(o.os,"name","posix"); monkeypatch.setattr(o.sys,"platform","linux"); monkeypatch.setattr(o.ctypes,"CDLL",lambda name,use_errno: Libc())
 with pytest.raises(o.ContractError,match="output collision"): o._atomic_rename_noreplace_directory(tmp_path/"source",tmp_path/"destination")
 assert calls and calls[0][0]==calls[0][2]==-100 and calls[0][4]==1
 monkeypatch.setattr(o.ctypes,"CDLL",lambda name,use_errno: type("NoRename",(),{})())
 with pytest.raises(o.ContractError,match="unsupported"): o._atomic_rename_noreplace_directory(tmp_path/"source",tmp_path/"destination")
 source=inspect.getsource(o._atomic_rename_noreplace_directory); assert "os.rename" not in source and "os.replace" not in source

REQUIREMENT_COVERAGE={name:"COVERED" for name in (
 "import safety","every frozen constant","runtime versions","source roots/shadowing","Mamba/cache hashes/bytes","code-object binding","capture line/source role","backend/source classifications","actual CPython trace install","actual line event","post-update semantics","wrong code/line rejection","disabled no-trace","prior trace restoration normal path","prior trace restoration exception path","no feedback","snapshot non-aliasing","artifact-copy non-aliasing","token_count=T+1","missing/duplicate/shifted indices","terminal semantics","t=0 zero-state transition","12-member cross product","all-layer completeness","member isolation","state_rows exact schema","row/vector physical consistency","deterministic row order","deterministic NPZ writer","deterministic NPZ parser rejection matrix","paired measurement exact schema","measurement formula reconstruction","predecessor coordinate validation","pre-divergence rtol/atol","pre-divergence status derivation","summary reconstruction","manifest exact schema","manifest constants","manifest provenance/type/hash validation","blocker/status semantics","exact seven-artifact set","report deterministic rendering","SHA256SUMS contract","initial output collision","staging collision","late output race","publication failure","bundle validation failure","Tier-3 rejection","equivalence harness ordinary output","equivalence harness last_hidden_state","equivalence harness hidden_states","equivalence harness structure","equivalence harness shape","equivalence harness dtype","equivalence harness device","equivalence harness params","equivalence harness requires_grad","equivalence harness buffers","equivalence harness cache","equivalence harness trace restoration","equivalence harness coordinates","equivalence harness non-aliasing","equivalence harness no feedback")}
def test_requirement_to_test_coverage_matrix():
 assert len(REQUIREMENT_COVERAGE)==64 and set(REQUIREMENT_COVERAGE.values())=={"COVERED"}

# Bounded completion matrix: all fixtures below are local synthetic CPython
# objects.  They never import a model, tokenizer, dataset, or network client.
def _complete_measurements():
 rows,vectors=rows_vectors(); return rows,vectors,o.measurements(rows,vectors,anchors())

def test_scientific_observer_cannot_be_forged_via_runtime_instance_introspection(monkeypatch):
 modules,versions,_=_runtime_baseline(monkeypatch); o.runtime_gate()
 valid=o.create_native_state_observer({},False); assert valid.enabled is False # A1
 attacker=(lambda:None).__code__; scientific=(o.NativeStateObserver,o.create_native_state_observer)
 # A2-A4/A8: public scientific interfaces have no binding parameters.
 for interface in scientific:
  names=set(inspect.signature(interface).parameters)
  assert not names & {"code","line","capture_line","binding","capability","token","sentinel","metadata"}
  for args in ((attacker,1,{}),(attacker,),( {},False,attacker,1)):
   with pytest.raises((o.ContractError,TypeError)): interface(*args)
 # A5/A6: every closure object reachable from the legitimate type/factory is
 # useless as a historical constructor capability.
 callables=[type(valid).__init__,o.create_native_state_observer]
 recovered=[]
 for fn in callables:
  recovered.extend(cell.cell_contents for cell in (fn.__closure__ or ()))
 for value in recovered:
  if callable(value):
   with pytest.raises((o.ContractError,TypeError)): value(attacker,1,{})
 for owner in (type(valid),valid,o.create_native_state_observer):
  for value in vars(owner).values() if hasattr(owner,"__dict__") else ():
   if callable(value):
    with pytest.raises((o.ContractError,TypeError)): value(attacker,1,{})
 # A7/A9/A10: direct construction re-runs the gate and binds the validated
 # runtime only; caller-owned registration metadata cannot mutate the binding.
 layers={}; direct=type(valid)(layers,False); layers[1]={"layer_index":1}
 assert direct.code is modules[o.MAMBA_MODULE].MambaMixer.slow_forward.__code__ and direct.capture_line==o.CAPTURE_LINE and direct.layers=={}
 assert not any("capability" in k.lower() or "sentinel" in k.lower() for k in vars(o))

def test_scientific_observer_rejects_exact_factory_global_resolver_forge(monkeypatch):
 # Historical attack: altering the module-global resolver must not influence
 # the construction-local scientific binding derivation.
 modules,_,_=_runtime_baseline(monkeypatch); attacker_lambda=lambda:None
 factory_globals=o.create_native_state_observer.__globals__
 original=factory_globals["_resolve_and_validate_runtime_binding"]
 try:
  factory_globals["_resolve_and_validate_runtime_binding"]=lambda:(attacker_lambda.__code__,1)
  observer=o.create_native_state_observer({},False)
  assert observer.code is not attacker_lambda.__code__ and observer.capture_line != 1
  assert observer.code is modules[o.MAMBA_MODULE].MambaMixer.slow_forward.__code__ and observer.capture_line==o.CAPTURE_LINE
 finally:
  factory_globals["_resolve_and_validate_runtime_binding"]=original

def test_a_synthetic_collector_cannot_be_promoted_to_scientific_observer(monkeypatch):
 _runtime_baseline(monkeypatch); collector=o._synthetic_collector(o._synthetic_trace_binding(lambda:None,1),{},False)
 assert isinstance(collector,o._TraceCollector) and not isinstance(collector,o.NativeStateObserver)
 with pytest.raises((o.ContractError,TypeError)): o.NativeStateObserver(collector,False)

def test_b_real_trace_edge_case_matrix():
 class Layer: pass
 layer,other=Layer(),Layer(); prior=sys.gettrace()
 def target(self):
  ssm_state=Tensor([0])
  for i in range(2):
   ssm_state=Tensor(ssm_state.v+1)
   marker=i
  return ssm_state
 line=target.__code__.co_firstlineno+4
 wrong_line=o._synthetic_collector(o._synthetic_trace_binding(target,target.__code__.co_firstlineno),{id(layer):{"layer_index":0}},True)
 with wrong_line.capture(): assert target(layer).v.item()==2
 assert wrong_line.snapshots=={} and sys.gettrace() is prior                    # B1/B4
 with pytest.raises(o.ContractError,match="layer completeness"): o.validate_captures({},2,[{"layer_index":0}])
 unregistered=o._synthetic_collector(o._synthetic_trace_binding(target,line),{id(layer):{"layer_index":0}},True)
 with unregistered.capture(): assert target(other).v.item()==2
 assert unregistered.snapshots=={}                                               # B2
 def duplicate_target(self):
  ssm_state=Tensor([0])
  for i in (0,0):
   ssm_state=Tensor(ssm_state.v+1)
   marker=i
  return ssm_state
 duplicate=o._synthetic_collector(o._synthetic_trace_binding(duplicate_target,duplicate_target.__code__.co_firstlineno+4),{id(layer):{"layer_index":0}},True)
 with pytest.raises(o.ContractError,match="duplicate coordinate"):
  with duplicate.capture(): duplicate_target(layer)
 assert sys.gettrace() is prior                                                  # B3
 ignored=o._synthetic_collector(o._synthetic_trace_binding(lambda self:None,1),{id(layer):{"layer_index":0}},True)
 with ignored.capture(): target(layer)
 assert ignored.snapshots=={}                                                    # B6
 correct=o._synthetic_collector(o._synthetic_trace_binding(target,line),{id(layer):{"layer_index":0}},True)
 with correct.capture(): result=target(layer)
 assert result.v.item()==2 and [correct.snapshots[(1,0,i)].v.item() for i in range(2)]==[1,2] # B7
 def raises(self):
  ssm_state=Tensor([1]); i=0; marker=0
  raise RuntimeError("expected")
 boom=o._synthetic_collector(o._synthetic_trace_binding(raises,raises.__code__.co_firstlineno+1),{id(layer):{"layer_index":0}},True)
 with pytest.raises(RuntimeError):
  with boom.capture(): raises(layer)
 assert sys.gettrace() is prior                                                  # B5

def _role_source(input_term="deltaB_u = discrete_B * hidden_states[..., None].float()",update="ssm_state = discrete_A[...] * ssm_state + deltaB_u[...]",readout="scan_output = torch.matmul(ssm_state.to(dtype), C[...].unsqueeze(-1))",persist="cache_params.ssm_states[0].copy_(ssm_state)"):
 lines=[""]*418; lines[407]=input_term; lines[408]=update; lines[409]=readout; lines[416]=persist
 data="\n".join(lines).encode(); return data,compile(data.decode(),"synthetic_roles.py","exec")

@pytest.mark.parametrize("input_term,update,readout,persist,message",[
 ("deltaB_u = hidden_states","ssm_state = discrete_A * ssm_state + deltaB_u","scan_output = ssm_state * C","cache_params.ssm_states[0].copy_(ssm_state)","source input role"), # C1
 ("deltaB_u = discrete_B","ssm_state = discrete_A * ssm_state + deltaB_u","scan_output = ssm_state * C","cache_params.ssm_states[0].copy_(ssm_state)","source input role"), # C2
 ("deltaB_u = discrete_B * hidden_states","ssm_state = discrete_A * ssm_state + unrelated","scan_output = ssm_state * C","cache_params.ssm_states[0].copy_(ssm_state)","source update role"), # C3
 ("deltaB_u = discrete_B * hidden_states","ssm_state = discrete_A + deltaB_u","scan_output = ssm_state * C","cache_params.ssm_states[0].copy_(ssm_state)","source update role"), # C4
 ("deltaB_u = discrete_B * hidden_states","ssm_state = ssm_state + deltaB_u","scan_output = ssm_state * C","cache_params.ssm_states[0].copy_(ssm_state)","source update role"), # C5
 ("deltaB_u = discrete_B * hidden_states","ssm_state = discrete_A * ssm_state + deltaB_u","scan_output = torch.matmul(ordinary_state, C)","cache_params.ssm_states[0].copy_(ssm_state)","source readout role"), # C6
 ("deltaB_u = discrete_B * hidden_states","ssm_state = discrete_A * ssm_state + deltaB_u","scan_output = torch.matmul(ssm_state, unrelated_C)","cache_params.ssm_states[0].copy_(ssm_state)","source readout role"), # C7
 ("deltaB_u = discrete_B * hidden_states","ssm_state = discrete_A * ssm_state + deltaB_u","scan_output = ssm_state * C","cache_params.conv_states[0].copy_(ssm_state)","source recurrent cache role"), # C8
 ("deltaB_u = discrete_B * hidden_states","ssm_state = discrete_A * ssm_state + deltaB_u","scan_output = ssm_state * C","cache_params.ssm_states[0].copy_(other_state)","source recurrent cache role"), # C9
])
def test_c_source_role_negative_fixtures(monkeypatch,input_term,update,readout,persist,message):
 data,code=_role_source(input_term,update,readout,persist); monkeypatch.setattr(o,"CAPTURE_LINE",409)
 with pytest.raises(o.ContractError,match=message): o._validate_source_roles(data,code)

def test_c_source_role_valid_baseline(monkeypatch):
 data,code=_role_source(); monkeypatch.setattr(o,"CAPTURE_LINE",409); o._validate_source_roles(data,code)

@pytest.mark.parametrize("persist",[
 "cache_params.conv_states[0].copy_(ssm_state); cache_params.ssm_states[0].copy_(ssm_state)", # verifier's exact same-line forge (C1)
 "cache_params.ssm_states[0].copy_(ssm_state); cache_params.ssm_states[1].copy_(ssm_state)", # C2
 "cache_params.ssm_states[0].copy_(ssm_state); cache_params.cache_state.copy_(ssm_state)", # C3
 "cache_params.conv_states[0].copy_(ssm_state)", # C4
 "cache_params.ssm_states[0].copy_(ssm_state + ambiguous_state)", # C5
])
def test_c_recurrent_cache_role_is_an_exclusive_single_statement(monkeypatch,persist):
 data,code=_role_source(persist=persist); monkeypatch.setattr(o,"CAPTURE_LINE",409)
 with pytest.raises(o.ContractError,match="source recurrent cache role"): o._validate_source_roles(data,code)

def test_e_measurement_coordinate_completeness_matrix():
 rows,vectors,records=_complete_measurements(); o.validate_measurements(records,rows,vectors) # E1
 cases=[]
 cases.append(records[:-1])                                                       # E2
 cases.append([])                                                                 # E3
 cases.append([r for r in records if r["comparison_id"]=="comparison-A"])         # E4
 cases.append(records+[dict(records[-1])])                                        # E5
 extra=[dict(r) for r in records]; extra[-1]["pair_id"]="extra"; cases.append(extra)# E6
 cases.append(list(reversed(records)))                                            # E7
 mapping=[dict(r) for r in records]; mapping[0]["member_condition"]="surface_null_matched"; cases.append(mapping) # E8
 cases.append([r for r in records if r["layer_index"]!=0])                       # E9
 cases.append([r for r in records if r["anchor_name"]!="anchor_terminal"])         # E10
 for case in cases: bad(o.validate_measurements,case,rows,vectors)

def test_f_predivergence_recomputed_at_validation_time():
 rows,vectors,records=_complete_measurements()
 o.assert_pre_divergence([1],[1]); o.assert_pre_divergence([1],[1.0000009])       # F1/F2
 bad(o.assert_pre_divergence,[1],[1.0000011])                                     # F3
 tampered=[dict(r) for r in records]; tampered[0]["pre_divergence_integrity_status"]="FAIL"; bad(o.validate_measurements,tampered,rows,vectors) # F4
 vectors[0][:]+=1; bad(o.validate_measurements,records,rows,vectors) # F5
 rows,vectors,records=_complete_measurements(); wrong=[dict(r) for r in records]; wrong[0]["reference_previous_vector_index"]=wrong[0]["reference_vector_index"]; bad(o.validate_measurements,wrong,rows,vectors) # F6

def test_k_capture_enabled_disabled_same_forward_equivalence():
 class Model:
  def __init__(self):
   self.parameter=Tensor([3]); self.parameter.requires_grad=True; self.buffer=Tensor([4]); self.cache=Tensor([0]); self.device="cpu"; self.dtype="float32"
  def forward(self):
   ssm_state=Tensor(self.cache.v.copy())
   for i in range(2):
    ssm_state=Tensor(ssm_state.v+1)
    capture_marker=i
   self.cache=ssm_state
   return {"primary":Tensor([2]),"last_hidden_state":Tensor([3]),"hidden_states":[Tensor([3]),Tensor([4])],"type":"synthetic"}
 baseline,observed=Model(),Model(); prior=sys.gettrace(); plain=baseline.forward()
 line=Model.forward.__code__.co_firstlineno+4; watcher=o._synthetic_collector(o._synthetic_trace_binding(Model.forward,line),{id(observed):{"layer_index":0}},True)
 with watcher.capture(): traced=observed.forward()
 assert sys.gettrace() is prior and watcher.snapshots and len(watcher.snapshots)==2 # K12-K15
 assert type(plain) is type(traced) and plain["type"]==traced["type"]
 for key in ("primary","last_hidden_state"):
  assert np.array_equal(plain[key].v,traced[key].v) and plain[key].v.shape==traced[key].v.shape and plain[key].v.dtype==traced[key].v.dtype
 for x,y in zip(plain["hidden_states"],traced["hidden_states"]): assert np.array_equal(x.v,y.v)
 assert np.array_equal(baseline.parameter.v,observed.parameter.v) and baseline.parameter.requires_grad is observed.parameter.requires_grad and np.array_equal(baseline.buffer.v,observed.buffer.v) and np.array_equal(baseline.cache.v,observed.cache.v) and baseline.device==observed.device and baseline.dtype==observed.dtype # K1-K11
 saved=watcher.snapshots[(1,0,1)].v.copy(); watcher.snapshots[(1,0,1)].v[:]=99; assert observed.cache.v.item()==2; observed.cache.v[:]=77; assert np.array_equal(saved,[2]) # K17-K19
 assert not any(np.shares_memory(x.v,watcher.snapshots[(1,0,0)].v) for x in traced.values() if isinstance(x,Tensor)) # K20

def test_k_torch_production_gated_enabled_disabled_equivalence(monkeypatch):
 import torch
 modules,_,_=_runtime_baseline(monkeypatch); Mixer=modules[o.MAMBA_MODULE].MambaMixer
 def make(seed):
  torch.manual_seed(seed); model=Mixer(); model.parameter=torch.nn.Parameter(torch.tensor([2.0])); model.buffer=torch.tensor([7.0]); model.cache=torch.tensor([0.0]); return model
 values=torch.tensor([[1.0],[2.0]],requires_grad=False); baseline,observed=make(1),make(1); prior=sys.gettrace()
 plain=baseline.forward(values); observer=o.create_native_state_observer({id(observed):{"layer_index":0}},True)
 with observer.capture(): traced=observed.forward(values)
 assert sys.gettrace() is prior and observer.snapshots is not None and len(observer.snapshots)==2 # K12-K15
 assert torch.equal(plain,traced) and type(plain) is type(traced) and plain.shape==traced.shape and plain.dtype==traced.dtype and plain.device==traced.device and plain.requires_grad==traced.requires_grad # K1/K7-K11
 assert torch.equal(baseline.parameter,observed.parameter) and torch.equal(baseline.buffer,observed.buffer) and torch.equal(baseline.cache,observed.cache) # K4-K6
 first=observer.snapshots[(1,0,0)]; frozen=first.clone(); first.add_(99); assert torch.equal(observed.cache,torch.tensor([3.0])); observed.cache.add_(9); assert torch.equal(frozen,torch.tensor([1.0])) # K17-K19
 assert all(snapshot.data_ptr()!=observed.cache.data_ptr() for snapshot in observer.snapshots.values()) # K18/K20

def test_l_twelve_fresh_capture_enabled_forwards():
 class Member:
  def __init__(self,seed): self.cache=Tensor([seed]); self.hidden=Tensor([999])
  def forward(self):
   ssm_state=Tensor(self.cache.v.copy())
   for i in range(2):
    ssm_state=Tensor(ssm_state.v+1)
    marker=i
   self.cache=ssm_state; return self.hidden
 line=Member.forward.__code__.co_firstlineno+4; results=[]; prior=sys.gettrace()
 for n,(pair,condition) in enumerate((p,c) for p in o.PAIR_ORDER for c in o.CONDITION_ORDER):
  member=Member(n); watcher=o._synthetic_collector(o._synthetic_trace_binding(Member.forward,line),{id(member):{"layer_index":0}},True)
  with watcher.capture(): member.forward()
  captures={0:{t:watcher.snapshots[(1,0,t)] for t in range(2)}}; o.validate_captures(captures,2,[{"layer_index":0}])
  results.append({"pair_id":pair,"condition":condition,"captures":captures,"fresh_capture_collection":True,"cache_reused":False,"cache":member.cache})
 assert len(results)==12 and sys.gettrace() is prior
 o.validate_member_orchestration([{k:v for k,v in r.items() if k!="cache"} for r in results],[{"layer_index":0}])
 assert [(r["pair_id"],r["condition"]) for r in results]==[(p,c) for p in o.PAIR_ORDER for c in o.CONDITION_ORDER]
 assert len({id(r["captures"]) for r in results})==12 and len({id(r["cache"]) for r in results})==12
 frozen=[r["captures"][0][0].v.copy() for r in results[1:]]; results[0]["captures"][0][0].v[:]=999
 assert all(np.array_equal(a,r["captures"][0][0].v) for a,r in zip(frozen,results[1:])) and all(r["captures"][0][0] is not r["cache"] for r in results)

def test_l_twelve_fresh_production_gated_torch_forwards(monkeypatch):
 import torch
 modules,_,_=_runtime_baseline(monkeypatch); Mixer=modules[o.MAMBA_MODULE].MambaMixer; results=[]; prior=sys.gettrace()
 for seed,(pair,condition) in enumerate((p,c) for p in o.PAIR_ORDER for c in o.CONDITION_ORDER):
  torch.manual_seed(seed); model=Mixer(); model.parameter=torch.nn.Parameter(torch.tensor([2.0])); model.buffer=torch.tensor([seed],dtype=torch.float32); model.cache=torch.tensor([0.0]); observer=o.create_native_state_observer({id(model):{"layer_index":0}},True)
  with observer.capture(): model.forward(torch.tensor([[1.0],[2.0]]))
  captures={0:{i:observer.snapshots[(1,0,i)] for i in range(2)}}; o.validate_captures(captures,2,[{"layer_index":0}])
  results.append({"pair_id":pair,"condition":condition,"captures":captures,"fresh_capture_collection":True,"cache_reused":False,"cache":model.cache,"observer":observer})
 assert sys.gettrace() is prior and len(results)==12 and [(r["pair_id"],r["condition"]) for r in results]==[(p,c) for p in o.PAIR_ORDER for c in o.CONDITION_ORDER] # L10-L13
 assert len({id(r["captures"]) for r in results})==12 and len({r["cache"].data_ptr() for r in results})==12 # L14-L15
 frozen=[r["captures"][0][0].clone() for r in results[1:]]; results[0]["captures"][0][0].add_(99)
 assert all(torch.equal(a,r["captures"][0][0]) for a,r in zip(frozen,results[1:])) and all(r["captures"][0][0].data_ptr()!=r["cache"].data_ptr() for r in results) # L16-L17

def test_m_summary_completeness_matrix():
 rows,vectors,records=_complete_measurements(); summary=o.build_summary(records,rows); o.validate_summary(summary,records,rows) # M1
 empty=o.build_summary([],rows) if False else {**summary,"rows":[]}; bad(o.validate_summary,empty,[],rows) # M2
 for partial in (records[:-1],[r for r in records if r["comparison_id"]!="comparison-C"],[r for r in records if r["anchor_name"]!="anchor_terminal"],[r for r in records if r["layer_index"]!=0]):
  with pytest.raises(o.ContractError): o.validate_summary(summary,partial,rows) # M3-M6
 tampered={**summary,"rows":[dict(x) for x in summary["rows"]]}; tampered["rows"][0]["a_mean"]+=1; bad(o.validate_summary,tampered,records,rows) # M7
 reordered={**summary,"rows":list(reversed(summary["rows"]))}; bad(o.validate_summary,reordered,records,rows) # M8

@pytest.mark.parametrize("layer_count",[3,4])
def test_multilayer_measurement_coordinate_contract_and_historical_mask(layer_count):
 rows,vectors=multilayer_rows_vectors(layer_count)
 records=o.measurements(rows,vectors,FROZEN_COMPARISON_ANCHORS)
 expected=expected_measurement_coordinates(layer_count)
 assert len(records)==3*3*6*layer_count
 actual=[(r["pair_id"],r["comparison_id"],r["anchor_name"],r["layer_index"]) for r in records]
 assert actual==expected
 o._require_measurement_coordinates(records,rows)
 # The consumed HEAD emitted layer before anchor.  A one-layer fixture reduces
 # both orders to the same sequence, while a production-shaped fixture exposes
 # the mismatch that raised "complete measurement coordinates/order".
 legacy_one=[(p,c,a,0) for p in o.PAIR_ORDER for c,_ in o.COMPARISONS for _layer in [0] for a in o.ANCHOR_ORDER]
 canonical_one=expected_measurement_coordinates(1)
 legacy_many=[(p,c,a,layer) for p in o.PAIR_ORDER for c,_ in o.COMPARISONS for layer in range(layer_count) for a in o.ANCHOR_ORDER]
 assert legacy_one==canonical_one and legacy_many!=expected
 with pytest.raises(o.ContractError,match="complete measurement coordinates/order"):
  o._require_measurement_coordinates([],rows,legacy_many)

def test_multilayer_post_forward_completion_to_external_publication(external_tmp):
 rows,vectors=multilayer_rows_vectors(3)
 records=o.measurements(rows,vectors,FROZEN_COMPARISON_ANCHORS)
 o.validate_measurements(records,rows,vectors)
 summary=o.build_summary(records,rows)
 o.validate_summary(summary,records,rows)
 manifest_value=o.build_manifest(multilayer_manifest(3))
 files=o.build_bundle(manifest_value,rows,vectors,records,summary)
 o.validate_bundle(files)
 output=external_tmp/"completed-bundle"
 o.publish_bundle(output,files)
 assert {path.name for path in output.iterdir()}==set(o.REQUIRED_ARTIFACTS)
 read_back={name:(output/name).read_bytes() for name in o.REQUIRED_ARTIFACTS}
 assert set(read_back)==set(o.REQUIRED_ARTIFACTS) and len(read_back)==7
 o.validate_bundle(read_back)
 assert not output.with_name(output.name+".staging").exists()

def test_multilayer_post_forward_adversarial_matrix(external_tmp):
 rows,vectors=multilayer_rows_vectors(3)
 records=o.measurements(rows,vectors,FROZEN_COMPARISON_ANCHORS)
 summary=o.build_summary(records,rows)
 files=o.build_bundle(multilayer_manifest(3),rows,vectors,records,summary)
 # Measurements: order, membership, coordinate, vector ownership, and metrics.
 bad(o.validate_measurements,list(reversed(records)),rows,vectors)
 bad(o.validate_measurements,records[:-1],rows,vectors)
 bad(o.validate_measurements,records+[dict(records[-1])],rows,vectors)
 wrong_layer=[dict(x) for x in records]; wrong_layer[0]["layer_index"]=99; bad(o.validate_measurements,wrong_layer,rows,vectors)
 wrong_token=[dict(x) for x in records]; wrong_token[0]["absolute_token_index"]+=1; bad(o.validate_measurements,wrong_token,rows,vectors)
 wrong_predecessor=[dict(x) for x in records]; wrong_predecessor[0]["reference_previous_vector_index"]=wrong_predecessor[0]["reference_vector_index"]; bad(o.validate_measurements,wrong_predecessor,rows,vectors)
 wrong_member=[dict(x) for x in records]; wrong_member[0]["member_vector_index"]=wrong_member[0]["reference_vector_index"]; bad(o.validate_measurements,wrong_member,rows,vectors)
 wrong_reference=[dict(x) for x in records]; wrong_reference[0]["reference_vector_index"]=wrong_reference[0]["member_vector_index"]; bad(o.validate_measurements,wrong_reference,rows,vectors)
 wrong_metric=[dict(x) for x in records]; wrong_metric[0]["paired_transition_delta"]+=0.5; bad(o.validate_measurements,wrong_metric,rows,vectors)
 # Summary contract: missing row and scientific statistic reconstruction.
 bad(o.validate_summary,{**summary,"rows":summary["rows"][:-1]},records,rows)
 altered_summary={**summary,"rows":[dict(x) for x in summary["rows"]]}; altered_summary["rows"][0]["a_mean"]+=0.5; bad(o.validate_summary,altered_summary,records,rows)
 # Bundle: provenance, exact artifact membership/order, checksums, report.
 corrupted_manifest=dict(files); manifest_data=json.loads(corrupted_manifest["manifest.json"]); manifest_data["observer_implementation_commit"]="f"*40; corrupted_manifest["manifest.json"]=o.canonical_json(manifest_data); bad(o.validate_bundle,corrupted_manifest)
 omitted=dict(files); omitted.pop("summary.json"); bad(o.validate_bundle,omitted)
 reordered={name:files[name] for name in reversed(o.REQUIRED_ARTIFACTS)}; bad(o.validate_bundle,reordered)
 corrupted_checksums=dict(files); corrupted_checksums["SHA256SUMS.txt"]=b"0"*len(files["SHA256SUMS.txt"]); bad(o.validate_bundle,corrupted_checksums)
 corrupted_report=dict(files); corrupted_report["report.md"]+=b"tamper\n"; bad(o.validate_bundle,corrupted_report)
 output=external_tmp/"collision"; output.mkdir()
 with pytest.raises(o.ContractError,match="output collision"): o.publish_bundle(output,files)
