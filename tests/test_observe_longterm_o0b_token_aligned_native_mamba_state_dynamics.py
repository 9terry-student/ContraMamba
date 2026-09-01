"""Adversarial contract tests for the O0b observer; all model objects are fakes."""
import ast, hashlib, io, json, re, sys, zipfile
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import observe_longterm_o0b_token_aligned_native_mamba_state_dynamics as o

def test_01_model_loader_security_is_behavioral(monkeypatch):
    class Torch: float32=object()
    class Model:
        calls=[]
        @classmethod
        def from_pretrained(cls, *args, **kwargs): cls.calls.append((args, kwargs)); return cls()
        def to(self, device): assert device == "cpu"; return self
        def eval(self): return self
        def requires_grad_(self, value): assert value is False; return self
    o.load_model(Model, Torch)
    assert Model.calls == [( (o.MODEL_ID,), {"revision":o.MODEL_REVISION, "torch_dtype":Torch.float32, "trust_remote_code":False})]
    with pytest.raises(o.ContractError):
        o.validate_model_kwargs({"revision":o.MODEL_REVISION, "torch_dtype":Torch.float32, "trust_remote_code":True}, Torch)

def test_02_loader_revision_validation_rejects_wrong_and_missing_revision():
    class Torch: float32=object()
    good={"revision":o.MODEL_REVISION, "torch_dtype":Torch.float32, "trust_remote_code":False}
    o.validate_model_kwargs(good, Torch)
    for bad in ({**good, "revision":"0"*40}, {**good, "revision":None}, {k:v for k,v in good.items() if k != "revision"}):
        with pytest.raises(o.ContractError): o.validate_model_kwargs(bad, Torch)
    class Tok:
        is_fast=True; calls=[]
        @classmethod
        def from_pretrained(cls, *args, **kwargs): cls.calls.append((args, kwargs)); return cls()
    o.load_tokenizer(Tok)
    assert Tok.calls[0] == ((o.TOKENIZER_ID,), {"revision":o.TOKENIZER_REVISION, "trust_remote_code":False, "use_fast":True})
    with pytest.raises(o.ContractError): o.validate_tokenizer_kwargs({"revision":"0"*40, "trust_remote_code":False, "use_fast":True})
def test_03_pair001_divergence_ownership_exact():
    a=json.loads((ROOT/'reports/longterm_o0b_matched_controls_v1_validation.json').read_text()); p=a['pairs'][0]['comparisons_to_reference']; assert [p[x]['first_divergent_token_index'] for x in o.COMPARISON_ORDER] == [25,18,17]
def test_04_same_anchor_different_absolute_index_not_same_key():
    coords,_,assembled,_=_valid_bundle(); rows=[r for r in assembled['observations'] if r['pair_id']=='o0b_pair_001' and r['comparison_id']=='insufficient_matched' and r['vector_role']=='member' and r['layer_index']==0]
    assert len({(r['anchor_name'],r['absolute_token_index'],r['vector_index']) for r in rows})==len(rows)
def test_05_reference_reuse_requires_complete_physical_key():
    coords,_,assembled,_=_valid_bundle(); rows=[r for r in assembled['observations'] if r['pair_id']=='o0b_pair_001' and r['vector_role']=='reference' and r['layer_index']==0]
    by_anchor={r['anchor_name']:r for r in rows}; assert by_anchor['anchor_pre_minus_1']['vector_index'] != by_anchor['anchor_divergence']['vector_index']
    same=[r for r in rows if r['absolute_token_index']==by_anchor['anchor_pre_minus_1']['absolute_token_index']]; assert len({r['vector_index'] for r in same})==1
def test_06_observation_identity_schema():
    assert {'pair_id','comparison_id','anchor_name','absolute_token_index','layer_index'} <= set(o.OBSERVATION_KEYS)
def test_07_persisted_layer_provenance_is_validated():
    coords,layers,assembled,_=_valid_bundle()
    for row in assembled["observations"] + assembled["distances"]:
        d=layers[row["layer_index"]]
        assert (row["layer_index"],row["layer_role"],row["state_source"]) == (d["layer_index"],d["layer_role"],d["state_source"])
    for field, value in (("layer_role", "wrong"), ("state_source", "wrong"), ("layer_index", 1)):
        obs=list(assembled["observations"]); bad=dict(obs[0]); bad[field]=value; obs[0]=bad
        with pytest.raises(o.ContractError): o.validate_observation_records(obs,coords,layers,len(assembled["vectors"]))
        dist=list(assembled["distances"]); bad=dict(dist[0]); bad[field]=value; dist[0]=bad
        with pytest.raises(o.ContractError): o.validate_distance_records(dist,coords,layers,len(assembled["vectors"]))

def test_08_metric_matches_independent_float64_pipeline():
    member=np.array([1.0e19, 3.0, -7.0, 1.0], dtype=np.float32)
    reference=np.array([1.0e19, -5.0, 11.0, 2.0], dtype=np.float32)
    member64=np.asarray(member,dtype=np.float64); reference64=np.asarray(reference,dtype=np.float64)
    mn=np.linalg.norm(member64); rn=np.linalg.norm(reference64); mu=member64/mn; ru=reference64/rn; delta=mu-ru
    expected={"normalized_l2_distance":float(np.linalg.norm(delta)), "cosine_distance":float(1.0-np.dot(mu,ru))}
    expected["cosine_redundancy_error"]=float(abs(expected["normalized_l2_distance"]**2-2.0*expected["cosine_distance"]))
    got=o.distance(member,reference)
    assert got == expected and all(type(x) is float for x in got.values())
def test_09_metric_outputs_are_python_float():
    assert all(type(v) is float for v in o.distance([1.,0.],[0.,1.]).values())
def test_10_zero_norm_fails_without_epsilon():
    with pytest.raises(o.ContractError): o.distance([0.,0.],[1.,0.])
def test_11_redundancy_error_strict_bound():
    coords,layers,assembled,files=_valid_bundle(); row=dict(assembled['distances'][0]); row['cosine_redundancy_error']=o.COSINE_REDUNDANCY_ATOL+1e-9
    with pytest.raises(o.ContractError): o.validate_distance_records([row]+list(assembled['distances'][1:]),coords,layers,len(assembled['vectors']))
def test_12_null_anchor_has_no_substitution():
    coords,states,layers,tokens=_synthetic_bundle_inputs(); before=o.assemble_observations(coords,states,layers,tokens)
    import copy; altered=copy.deepcopy(coords); del altered['pairs'][0]['comparisons_to_reference']['insufficient_matched']['anchor_indices']['anchor_terminal']; after=o.assemble_observations(altered,states,layers,tokens)
    assert not any(r['pair_id']=='o0b_pair_001' and r['comparison_id']=='insufficient_matched' and r['anchor_name']=='anchor_terminal' for r in after['observations'])
    assert len(after['vectors']) < len(before['vectors'])
def test_13_summary_zero_denominator_behavior():
    layer={'layer_index':0,'layer_role':'r','state_source':'s'}; x=o.descriptive_summary([],[],layer,'anchor_terminal'); assert x['a_available_pair_count']==0 and x['a_mean'] is None and x['a_median'] is None

def test_13_nonempty_summary_reconstructs_ids_counts_means_medians_and_comparisons():
    coords,layers,assembled,_=_valid_bundle(); rows=[]
    for r in assembled["distances"]:
        if r["layer_index"] == 0 and r["anchor_name"] == "anchor_divergence" and r["pair_id"] in o.PAIR_ORDER[:2]:
            x=dict(r); values={"insufficient_matched": {"o0b_pair_001":3.0,"o0b_pair_002":1.0}, "paraphrase_sufficient": {"o0b_pair_001":2.0,"o0b_pair_002":2.0}, "surface_null_matched": {"o0b_pair_001":1.0,"o0b_pair_002":2.0}}
            x["normalized_l2_distance"]=values[x["comparison_id"]][x["pair_id"]]; rows.append(x)
    actual=o.descriptive_summary(rows,list(o.PAIR_ORDER),layers[0],"anchor_divergence")
    assert actual["a_available_pair_ids"] == ["o0b_pair_001","o0b_pair_002"] and actual["a_available_pair_count"] == 2
    assert actual["a_mean"] == 2.0 and actual["a_median"] == 2.0
    assert actual["b_available_pair_ids"] == ["o0b_pair_001","o0b_pair_002"] and actual["b_mean"] == actual["b_median"] == 2.0
    assert actual["c_available_pair_ids"] == ["o0b_pair_001","o0b_pair_002"] and actual["c_mean"] == actual["c_median"] == 1.5
    assert actual["a_gt_b_comparable_pair_ids"] == ["o0b_pair_001","o0b_pair_002"] and actual["a_gt_b_denominator"] == 2 and actual["a_gt_b_count"] == 1
    assert actual["a_gt_c_comparable_pair_ids"] == ["o0b_pair_001","o0b_pair_002"] and actual["a_gt_c_denominator"] == 2 and actual["a_gt_c_count"] == 1
    summary=o.reconstruct_summary(rows,list(o.PAIR_ORDER),layers[:1]); o.validate_summary(summary,list(o.PAIR_ORDER),layers[:1],rows)
    for key in ("a_available_pair_ids","a_available_pair_count","a_gt_b_denominator"):
        bad=json.loads(json.dumps(summary)); bad["aggregates"][1][key] = [] if isinstance(bad["aggregates"][1][key],list) else 99
        with pytest.raises(o.ContractError): o.validate_summary(bad,list(o.PAIR_ORDER),layers[:1],rows)
def test_14_canonical_json_is_repeatable():
    p={'z':1,'é':'ok'}; assert o.canonical_json(p)==o.canonical_json(p); assert hashlib.sha256(o.canonical_json(p)).digest()==hashlib.sha256(o.canonical_json(p)).digest()
def test_15_canonical_jsonl_order_and_bytes():
    rows=[{'i':2},{'i':1}]; b=o.canonical_jsonl(rows); assert b.decode().splitlines()==['{"i":2}','{"i":1}']; assert b==o.canonical_jsonl(rows)
def test_16_report_template_is_deterministic_when_generated_from_same_bytes():
    assert o.canonical_json({'report':['# O0b Token-Aligned Native Hidden-State Proxy Screening']}) == o.canonical_json({'report':['# O0b Token-Aligned Native Hidden-State Proxy Screening']})
def test_17_npy_format_is_1_0():
    z=o.deterministic_npz([[1.,2.]]); b=zipfile.ZipFile(io.BytesIO(z)).read('vectors.npy'); assert b[6:8] == bytes((1,0))
def test_18_npz_contains_exactly_vectors_member():
    with zipfile.ZipFile(io.BytesIO(o.deterministic_npz([[1,2]]))) as z: assert z.namelist()==['vectors.npy']
def test_19_zipinfo_frozen_fields():
    with zipfile.ZipFile(io.BytesIO(o.deterministic_npz([[1,2]]))) as z:
        i=z.infolist()[0]; assert (i.filename,i.date_time,i.compress_type,i.create_system,i.create_version,i.extract_version,i.external_attr,i.internal_attr,i.extra,i.comment,i.flag_bits)==('vectors.npy',(1980,1,1,0,0,0),zipfile.ZIP_STORED,3,20,20,0o100644<<16,0,b'',b'',0)
def test_20_zip64_largezipfile_fails_closed_and_is_disabled(monkeypatch):
    calls=[]; original=zipfile.ZipFile
    class SpyZipFile(original):
        def __init__(self,*args,**kwargs): calls.append(kwargs.copy()); super().__init__(*args,**kwargs)
        def writestr(self,*args,**kwargs): raise zipfile.LargeZipFile("simulated")
    monkeypatch.setattr(o.zipfile, "ZipFile", SpyZipFile)
    with pytest.raises(zipfile.LargeZipFile): o.deterministic_npz([[1,2]])
    assert calls == [{"compression":zipfile.ZIP_STORED,"allowZip64":False,"compresslevel":None}]
def test_21_npz_is_byte_and_hash_deterministic():
    a=o.deterministic_npz([[1,2],[3,4]]); b=o.deterministic_npz([[1,2],[3,4]]); assert a==b and hashlib.sha256(a).digest()==hashlib.sha256(b).digest()
def test_22_checksums_strict_parser_rejects_full_malformed_matrix():
    _,_,_,files=_valid_bundle(); good=files["SHA256SUMS.txt"]; six={n:files[n] for n in o.REQUIRED_ARTIFACTS[:-1]}
    lines=good.splitlines()
    assert len(lines)==6 and all(re.fullmatch(rb"[0-9a-f]{64}  [^\r\n]+",x) for x in lines)
    assert [x.split(b"  ",1)[1] for x in lines] == sorted(x.split(b"  ",1)[1] for x in lines)
    assert good.endswith(b"\n") and b"\r" not in good and good.count(b"  ")==6
    cases={
        "uppercase":good.upper(), "63-char":b"0"*63+good[64:], "65-char":b"0"*65+good[64:],
        "one-space":good.replace(b"  ",b" ",1), "three-space":good.replace(b"  ",b"   ",1),
        "crlf":good.replace(b"\n",b"\r\n"), "reordered":b"".join(reversed([x+b"\n" for x in lines])),
        "duplicate":b"\n".join([lines[0],lines[0],*lines[2:]])+b"\n",
        "missing":b"\n".join(lines[:5])+b"\n", "extra":good+b"0"*64+b"  extra.txt\n",
        "self-entry":good+b"0"*64+b"  SHA256SUMS.txt\n", "blank":good.replace(b"\n",b"\n\n",1),
        "non-hex":b"g"+good[1:], "wrong-digest":b"0"*64+good[64:]
    }
    o.parse_checksums(good,six)
    for name,bad in cases.items():
        with pytest.raises(o.ContractError):
            o.parse_checksums(bad,six)
def test_23_existing_output_fails_before_writing():
    class Existing:
        name="out"
        def exists(self): return True
        def with_name(self, name): return self
    out=Existing()
    with pytest.raises(o.ContractError): o.publish(out,{})
def test_24_existing_staging_fails_and_is_preserved():
    class ExistingStage:
        name="out"
        def exists(self): return False
        def with_name(self, name): return self
        def __truediv__(self, value): return self
        def mkdir(self): raise AssertionError("staging must be rejected before mkdir")
    class Stage(ExistingStage):
        def exists(self): return True
    ExistingStage.with_name=lambda self, name: Stage()
    with pytest.raises(o.ContractError): o.publish(ExistingStage(),{})
def test_25_partial_staging_is_not_publishable():
    with pytest.raises(o.ContractError): o.publish(Path("new-output"), {'manifest.json':b'{}'})
def test_26_checksums_are_derived_after_six_files():
    files={'manifest.json':b'1','anchor_observations.jsonl':b'2','anchor_hidden_states.npz':b'3','paired_distances.jsonl':b'4','summary.json':b'5','report.md':b'6'}; assert b'SHA256SUMS.txt' not in o.checksum_text(files)
def test_27_publication_gate_rejects_complete_20_case_mutation_matrix():
    coords,layers,assembled,base=_valid_bundle()
    def fresh(f):
        f=dict(f); f["SHA256SUMS.txt"]=o.checksum_text({n:f[n] for n in o.REQUIRED_ARTIFACTS[:-1]}); return f
    def json_file(f,name,mutate):
        f=dict(f); x=json.loads(f[name]); mutate(x); f[name]=o.canonical_json(x); return fresh(f)
    def jsonl_file(f,name,index,mutate):
        f=dict(f); x=[json.loads(z) for z in f[name].decode().splitlines()]; mutate(x[index]); f[name]=o.canonical_jsonl(x); return fresh(f)
    def reject(name,f,checksum_expected=False):
        if checksum_expected: o.parse_checksums(f["SHA256SUMS.txt"],{n:f[n] for n in o.REQUIRED_ARTIFACTS[:-1]})
        fs=_MemFS()
        with pytest.raises(o.ContractError): o.publish_bundle(_MemPath("mut-"+name),f,coords,layers,fs)
        assert fs.renames == []
    cases={
        "missing": {n:base[n] for n in o.REQUIRED_ARTIFACTS[:-1]},
        "extra": {**base,"extra.txt":b"x"},
        "bad-checksum": {**base,"SHA256SUMS.txt":base["SHA256SUMS.txt"].replace(b"0",b"1",1)},
        "malformed-manifest": json_file(base,"manifest.json",lambda x:x.pop("run_name")),
        "wrong-execution-status": json_file(base,"manifest.json",lambda x:x.__setitem__("execution_status","FAILED")),
        "malformed-observation": jsonl_file(base,"anchor_observations.jsonl",0,lambda x:x.pop("vector_index")),
        "out-of-range-index": jsonl_file(base,"anchor_observations.jsonl",0,lambda x:x.__setitem__("vector_index",len(assembled["vectors"]))),
        "aliased-physical-key": jsonl_file(base,"anchor_observations.jsonl",1,lambda x:x.__setitem__("vector_index",0)),
        "malformed-distance": jsonl_file(base,"paired_distances.jsonl",0,lambda x:x.pop("cosine_distance")),
        "wrong-metric": jsonl_file(base,"paired_distances.jsonl",0,lambda x:x.__setitem__("normalized_l2_distance",x["normalized_l2_distance"]+0.1)),
        "swapped-vector-index": jsonl_file(base,"paired_distances.jsonl",0,lambda x:x.__setitem__("reference_vector_index",x["member_vector_index"])),
        "malformed-summary": json_file(base,"summary.json",lambda x:x["aggregates"][0].pop("a_mean")),
        "wrong-aggregate-mean": json_file(base,"summary.json",lambda x:x["aggregates"][0].__setitem__("a_mean",99.0)),
        "wrong-aggregate-denominator": json_file(base,"summary.json",lambda x:x["aggregates"][0].__setitem__("a_gt_b_denominator",99)),
        "nonfinite-metric": dict(base),
        "failed-pre-divergence": jsonl_file(base,"paired_distances.jsonl",0,lambda x:x.__setitem__("pre_divergence_integrity_status","FAILED")),
        "pre-divergence-false": json_file(base,"summary.json",lambda x:x["integrity"].__setitem__("pre_divergence_all_pass",False)),
        "corrupt-npz": {**base,"anchor_hidden_states.npz":b"not-an-npz"},
        "changed-npz": fresh({**base,"anchor_hidden_states.npz":_changed_npz(base["anchor_hidden_states.npz"])}),
        "wrong-zipinfo": fresh({**base,"anchor_hidden_states.npz":_bad_zipinfo(base["anchor_hidden_states.npz"])}),
    }
    nonfinite=[dict(json.loads(z)) for z in base["paired_distances.jsonl"].decode().splitlines()]; nonfinite[0]["normalized_l2_distance"]=float("nan")
    cases["nonfinite-metric"]={**base,"paired_distances.jsonl":(json.dumps(nonfinite[0],sort_keys=True,separators=(",",":"),allow_nan=True)+"\n").encode()+o.canonical_jsonl(nonfinite[1:])}
    fresh_names={"malformed-manifest","wrong-execution-status","malformed-observation","out-of-range-index","aliased-physical-key","malformed-distance","wrong-metric","swapped-vector-index","malformed-summary","wrong-aggregate-mean","wrong-aggregate-denominator","failed-pre-divergence","pre-divergence-false","changed-npz","wrong-zipinfo"}
    for name,f in cases.items(): reject(name,f,name in fresh_names)

def test_pre_model_validation_rejects_bad_dataset_before_load():
    with pytest.raises(o.ContractError): o.validate_sources([{'pair_id':'bad'}])
def test_loader_paths_are_lazy_and_forward_is_inference_only():
    tree=ast.parse(Path(o.__file__).read_text()); names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}; assert 'transformers' not in names
    src=Path(o.__file__).read_text(); assert 'output_hidden_states=True' in src and 'use_cache=False' in src and 'torch.inference_mode()' in src
def test_pre_divergence_exact_tolerance_and_rtol():
    ref=np.array([[1.,2.]],np.float32); mem=np.array([[1.0000005,2.]],np.float32); o.assert_pre_divergence(ref,mem,0,[1],[1])
    with pytest.raises(o.ContractError): o.assert_pre_divergence(ref,np.array([[1.001,2.]],np.float32),0,[1],[1])

def _synthetic_bundle_inputs():
    coords=json.loads((ROOT/'reports/longterm_o0b_matched_controls_v1_validation.json').read_text())
    layers=[{"layer_index":0,"layer_role":"embedding_or_initial_hidden_state","state_source":"hidden_states[0]"},{"layer_index":1,"layer_role":"output_hidden_state","state_source":"last_hidden_state"}]
    states={}; tokens={}
    for pi,p in enumerate(o.PAIR_ORDER):
        n=45; ref=np.ones((2,n,3),np.float32)*(pi+1); states[(p,'reference_sufficient')]=ref; tokens[(p,'reference_sufficient')]=list(range(100*pi,100*pi+n))
        for comp in o.COMPARISON_ORDER:
            mem=ref.copy(); div=coords['pairs'][pi]['comparisons_to_reference'][comp]['first_divergent_token_index']; mem[:,div:,:]+=0.25; states[(p,comp)]=mem
            ids=list(tokens[(p,'reference_sufficient')]); ids[div:]=[x+1000 for x in ids[div:]]; tokens[(p,comp)]=ids
    return coords,states,layers,tokens

def _manifest(layers):
    return {"schema_version":o.SCHEMA_VERSION,"experiment_name":"synthetic-o0b","scientific_design_authority_commit":o.SCIENTIFIC_DESIGN_AUTHORITY_COMMIT,"boundary_recovery_authority_commit":o.BOUNDARY_RECOVERY_AUTHORITY_COMMIT,"input_implementation_freeze_commit":o.INPUT_IMPLEMENTATION_FREEZE_COMMIT,"observer_implementation_authority_commit":o.OBSERVER_IMPLEMENTATION_AUTHORITY_COMMIT,"observer_implementation_commit":"a"*40,"observer_script_sha256":"b"*64,"dataset_path":o.DATASET_PATH,"dataset_sha256":o.DATASET_SHA256,"validation_artifact_path":o.VALIDATION_ARTIFACT_PATH,"validation_artifact_sha256":o.VALIDATION_ARTIFACT_SHA256,"validation_artifact_repository_head":o.BOUNDARY_RECOVERY_AUTHORITY_COMMIT,"model_id":o.MODEL_ID,"model_revision":o.MODEL_REVISION,"model_trust_remote_code":False,"tokenizer_id":o.TOKENIZER_ID,"tokenizer_revision":o.TOKENIZER_REVISION,"tokenizer_trust_remote_code":False,"tokenizer_use_fast":True,"add_special_tokens":False,"device":"cpu","dtype":"float32","python_version":"3.13","numpy_version":"2","torch_version":"2","transformers_version":"4","serialization_template":"canonical-json-v1/deterministic-npz-v1","comparison_order":list(o.COMPARISON_ORDER),"anchor_order":list(o.ANCHOR_ORDER),"layer_descriptors":layers,"pre_divergence_rtol":0.0,"pre_divergence_atol":1e-6,"cosine_redundancy_atol":1e-12,"exact_command":"[\"python\",\"observer.py\"]","run_name":"synthetic-run","required_artifacts":list(o.REQUIRED_ARTIFACTS),"execution_status":"COMPLETE"}

class _MemPath:
    def __init__(self,name): self.name=name
    def with_name(self,name): return _MemPath(name)
    def __truediv__(self,name): return self.name+'/'+str(name)
    def __repr__(self): return self.name

class _MemFS:
    def __init__(self): self.dirs=set(); self.data={}; self.writes=[]; self.renames=[]
    def exists(self,p): return str(p) in self.dirs or str(p) in self.data
    def mkdir(self,p): self.dirs.add(str(p))
    def write_bytes(self,p,b): self.data[str(p)]=bytes(b); self.writes.append(str(p).split('/')[-1])
    def read_bytes(self,p): return self.data[str(p)]
    def list_dir(self,p): return [k.split('/')[-1] for k in self.data if k.startswith(str(p)+'/')]
    def rename(self,a,b): self.dirs.add(str(b)); self.dirs.discard(str(a)); self.renames.append((str(a),str(b)))

def test_complete_observation_distance_summary_assembly():
    coords,states,layers,tokens=_synthetic_bundle_inputs(); assembled=o.assemble_observations(coords,states,layers,tokens)
    assert assembled['observations'] and assembled['distances']; assert assembled['observations'][0]['pair_id']=='o0b_pair_001'
    assert [x['absolute_token_index'] for x in assembled['distances'] if x['pair_id']=='o0b_pair_001' and x['anchor_name']=='anchor_divergence' and x['layer_index']==0] == [25,18,17]
    summary=o.build_summary(assembled['distances'],assembled['pair_ids'],layers); o.validate_summary(summary,assembled['pair_ids'],layers)

def test_runtime_provenance_head_and_script_sha_fail_closed():
    path=Path(o.__file__); digest=o.sha256_file(path)
    assert o.verify_runtime_provenance('a'*40,digest,path,'run',['python','observer.py'],actual_head='a'*40)['run_name']=='run'
    with pytest.raises(o.ContractError): o.verify_runtime_provenance('b'*40,digest,path,'run',actual_head='a'*40)
    with pytest.raises(o.ContractError): o.verify_runtime_provenance('a'*40,'x'*64,path,'run',actual_head='a'*40)
    with pytest.raises(o.ContractError): o.verify_runtime_provenance('a'*40,'c'*64,path,'run',actual_head='a'*40)

def test_capture_runtime_versions_reads_all_four_runtime_sources(monkeypatch):
    monkeypatch.setattr(o.sys, "version", "3.99.1 synthetic build")
    monkeypatch.setattr(o.np, "__version__", "9.8.7")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.7.6+cpu"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="4.55.44"))
    assert o.capture_runtime_versions() == {
        "python_version":"3.99.1",
        "numpy_version":"9.8.7",
        "torch_version":"2.7.6+cpu",
        "transformers_version":"4.55.44",
    }

def test_parse_args_exposes_no_runtime_version_override_flags():
    required=["--output-dir","out","--run-name","run","--exact-command","[]","--observer-implementation-commit","a"*40,"--observer-script-sha256","b"*64]
    assert o.parse_args(required).run_name == "run"
    for flag in ("--python-version","--numpy-version","--torch-version","--transformers-version","--runtime-versions","--test-dependencies"):
        with pytest.raises(SystemExit):
            o.parse_args([*required, flag, "override"])

def test_manifest_runtime_versions_accept_concrete_controlled_values():
    _,_,layers,_=_synthetic_bundle_inputs()
    m=_manifest(layers)
    m.update({"python_version":"3.13.2","numpy_version":"2.2.1","torch_version":"2.6.0+cpu","transformers_version":"4.49.0"})
    assert {k:o.build_manifest(m)[k] for k in o.RUNTIME_VERSION_KEYS} == {k:m[k] for k in o.RUNTIME_VERSION_KEYS}

def test_manifest_runtime_versions_fail_closed_for_missing_none_empty_and_whitespace():
    _,_,layers,_=_synthetic_bundle_inputs()
    for key in o.RUNTIME_VERSION_KEYS:
        missing=_manifest(layers); missing.pop(key)
        with pytest.raises(o.ContractError): o.build_manifest(missing)
        for value in (None, "", " ", "\t\n"):
            bad=_manifest(layers); bad[key]=value
            with pytest.raises(o.ContractError): o.build_manifest(bad)

def test_manifest_runtime_versions_reject_forbidden_placeholders_exhaustively():
    _,_,layers,_=_synthetic_bundle_inputs()
    for key in o.RUNTIME_VERSION_KEYS:
        for value in ("unknown", "UNKNOWN", "n/a", "N/A", "none", "None"):
            bad=_manifest(layers); bad[key]=value
            with pytest.raises(o.ContractError): o.build_manifest(bad)

def test_manifest_runtime_versions_reject_non_string_fields():
    _,_,layers,_=_synthetic_bundle_inputs()
    for key in o.RUNTIME_VERSION_KEYS:
        for value in (1, 2.0, ["3.13"], {"version":"3.13"}):
            bad=_manifest(layers); bad[key]=value
            with pytest.raises(o.ContractError): o.build_manifest(bad)

def test_publication_manifest_validation_rejects_runtime_placeholders():
    coords,layers,assembled,files=_valid_bundle()
    manifest=json.loads(files["manifest.json"]); manifest["torch_version"]="unknown"
    bad=dict(files); bad["manifest.json"]=o.canonical_json(manifest); bad["SHA256SUMS.txt"]=o.checksum_text({n:bad[n] for n in o.REQUIRED_ARTIFACTS[:-1]})
    with pytest.raises(o.ContractError): o.publish_bundle(_MemPath("bad-runtime-version"),bad,coords,layers,_MemFS())

def test_manifest_exact_command_and_run_name_provenance():
    _,_,layers,_=_synthetic_bundle_inputs(); m=_manifest(layers); assert o.build_manifest(m)['run_name']=='synthetic-run'; assert o.actual_command(['python','observer.py','--run-name','synthetic-run'])=='["python","observer.py","--run-name","synthetic-run"]'

def test_publish_validator_rejects_schema_integrity_and_nonfinite_failures():
    coords,states,layers,tokens=_synthetic_bundle_inputs(); a=o.assemble_observations(coords,states,layers,tokens); files=o.build_artifact_bundle(_manifest(layers),a); fs=_MemFS(); out=_MemPath('out'); o.publish_bundle(out,files,coords,layers,fs); assert fs.renames
    bad=dict(files); bad['summary.json']=bad['summary.json'].replace(b'"pre_divergence_all_pass": true',b'"pre_divergence_all_pass": false'); fs2=_MemFS()
    with pytest.raises(o.ContractError): o.publish_bundle(_MemPath('bad'),bad,coords,layers,fs2)

def test_complete_synthetic_bundle_publishes_only_after_validation():
    coords,states,layers,tokens=_synthetic_bundle_inputs(); a=o.assemble_observations(coords,states,layers,tokens); files=o.build_artifact_bundle(_manifest(layers),a); fs=_MemFS(); o.publish_bundle(_MemPath('published'),files,coords,layers,fs)
    assert fs.writes==list(o.REQUIRED_ARTIFACTS); assert fs.writes[-1]=='SHA256SUMS.txt'; assert fs.renames==[('published.tmp','published')]
    writes,renames=list(fs.writes),list(fs.renames)
    with pytest.raises(o.ContractError): o.publish_bundle(_MemPath('published'),files,coords,layers,fs)
    assert fs.writes==writes and fs.renames==renames
    fs2=_MemFS(); fs2.dirs.add('staged.tmp'); before=(list(fs2.writes),list(fs2.renames),set(fs2.dirs))
    with pytest.raises(o.ContractError): o.publish_bundle(_MemPath('staged'),files,coords,layers,fs2)
    assert (fs2.writes,fs2.renames,set(fs2.dirs))==before

def _valid_bundle():
    coords,states,layers,tokens=_synthetic_bundle_inputs(); assembled=o.assemble_observations(coords,states,layers,tokens); files=o.build_artifact_bundle(_manifest(layers),assembled); return coords,layers,assembled,files

def _bad_zipinfo(data):
    with zipfile.ZipFile(io.BytesIO(data)) as src: raw=src.read("vectors.npy")
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_STORED,allowZip64=False) as z:
        info=zipfile.ZipInfo("vectors.npy",(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_STORED; info.create_system=3; info.external_attr=0
        z.writestr(info,raw)
    return out.getvalue()

def _changed_npz(data):
    arr=np.array(np.load(io.BytesIO(data),allow_pickle=False)["vectors"],copy=True,dtype=np.float32); arr[0,0]+=0.5
    return o.deterministic_npz(arr)

def test_pre_divergence_checks_real_prefix_and_never_uses_empty_placeholders():
    ref=np.ones((4,3),np.float32); mem=ref.copy(); o.assert_pre_divergence(ref,mem,2,[1,2,3,9],[1,2,3,8])
    with pytest.raises(o.ContractError): o.assert_pre_divergence(ref,mem,2,[1,2,4,9],[1,2,3,8])
    with pytest.raises(o.ContractError): o.assert_pre_divergence(ref,np.array([[1,1,1],[1,1,1],[1.01,1,1],[1,1,1]],np.float32),2,[1,2,3,9],[1,2,3,8])

def test_summary_reconstruction_rejects_semantic_mutations():
    coords,layers,assembled,files=_valid_bundle(); summary=json.loads(files['summary.json']); summary['aggregates'][0]['a_mean'] += 1.0
    with pytest.raises(o.ContractError): o.validate_summary(summary,assembled['pair_ids'],layers,assembled['distances'])
    summary=json.loads(files['summary.json']); summary['aggregates'][0]['a_gt_b_denominator'] += 1
    with pytest.raises(o.ContractError): o.validate_summary(summary,assembled['pair_ids'],layers,assembled['distances'])

def test_npz_and_distance_semantics_reject_fresh_checksum_corruption():
    coords,layers,assembled,files=_valid_bundle(); fs=_MemFS(); out=_MemPath('semantic'); o.publish_bundle(out,files,coords,layers,fs)
    bad=dict(files); arr=np.load(io.BytesIO(bad['anchor_hidden_states.npz']),allow_pickle=False)['vectors']; arr=np.array(arr,copy=True); arr[0,0]+=0.5; bad['anchor_hidden_states.npz']=o.deterministic_npz(arr); bad['SHA256SUMS.txt']=o.checksum_text({n:bad[n] for n in o.REQUIRED_ARTIFACTS[:-1]})
    with pytest.raises(o.ContractError): o.publish_bundle(_MemPath('bad-npz'),bad,coords,layers,_MemFS())
    bad=dict(files); dist=[json.loads(x) for x in bad['paired_distances.jsonl'].decode().splitlines()]; dist[0]['normalized_l2_distance'] += 0.1; bad['paired_distances.jsonl']=o.canonical_jsonl(dist); bad['SHA256SUMS.txt']=o.checksum_text({n:bad[n] for n in o.REQUIRED_ARTIFACTS[:-1]})
    with pytest.raises(o.ContractError): o.publish_bundle(_MemPath('bad-distance'),bad,coords,layers,_MemFS())

def test_strict_checksum_parser_rejects_format_mutations():
    _,_,_,files=_valid_bundle(); good=files['SHA256SUMS.txt'];
    for bad in (good.replace(b'  ',b' ',1),good.replace(b'  ',b'   ',1),good.replace(b'\n',b'\r\n'),good.upper(),good.replace(b'\n',b'',1)):
        with pytest.raises(o.ContractError): o.parse_checksums(bad,{n:files[n] for n in o.REQUIRED_ARTIFACTS[:-1]})

def test_report_is_deterministic_and_uses_authority_rendering():
    coords,layers,assembled,files=_valid_bundle(); manifest=json.loads(files['manifest.json']); summary=json.loads(files['summary.json']); summary['aggregates'][0]['a_mean']=None; one=o.render_report(manifest,assembled['distances'],summary); two=o.render_report(manifest,assembled['distances'],summary)
    assert one==two and hashlib.sha256(one).digest()==hashlib.sha256(two).digest() and one.endswith(b'\n') and b'NA' in one and b'1.0000000000000000' not in one

def test_loader_factories_capture_exact_security_kwargs():
    class T:
        is_fast=True
        @classmethod
        def from_pretrained(cls,*a,**kw): cls.kw=kw; return cls()
    t=o.load_tokenizer(T); assert T.kw=={'revision':o.TOKENIZER_REVISION,'trust_remote_code':False,'use_fast':True}
    class Torch: float32=object()
    class M:
        @classmethod
        def from_pretrained(cls,*a,**kw): cls.kw=kw; return cls()
        def to(self,*a): return self
        def eval(self): return self
        def requires_grad_(self,*a): return self
    o.load_model(M,Torch); assert M.kw=={'revision':o.MODEL_REVISION,'torch_dtype':Torch.float32,'trust_remote_code':False}
    with pytest.raises(o.ContractError): o.validate_tokenizer_kwargs({'revision':o.TOKENIZER_REVISION,'trust_remote_code':True,'use_fast':True})

def test_zip64_failure_is_fail_closed(monkeypatch):
    original=zipfile.ZipFile.writestr
    def fail(self,*a,**kw): raise zipfile.LargeZipFile('simulated')
    monkeypatch.setattr(zipfile.ZipFile,'writestr',fail)
    with pytest.raises(zipfile.LargeZipFile): o.deterministic_npz([[1,2]])
    assert original is not None

def test_main_orchestration_uses_exactly_twelve_full_forwards_and_publishes_seven():
    coords=json.loads((ROOT/'reports/longterm_o0b_matched_controls_v1_validation.json').read_text()); rows=o.read_jsonl(ROOT/o.DATASET_PATH); byid={r['pair_id']:r for r in rows}; token_map={};
    for p in coords['pairs']:
        for c in ('reference_sufficient',*o.COMPARISON_ORDER):
            e=p['conditions'][c]; token_map[f"Claim: {byid[p['pair_id']]['claim']}\nEvidence: {byid[p['pair_id']][c]}"]=(e['full_serialized_token_ids'],e['full_offset_mapping'])
    class Tok:
        is_fast=True
        def __call__(self,text,**kw): ids,offs=token_map[text]; return {'input_ids':ids,'offset_mapping':offs}
    class Model:
        def __init__(self): self.calls=[]
        def __call__(self,**kw):
            self.calls.append(kw); ids=kw['input_ids'].detach().cpu().numpy()[0].astype(np.float32); n=len(ids); h0=np.stack((ids+1,ids+2,ids+3),1)[None,:,:]; cs=np.cumsum(ids+1).astype(np.float32); h1=np.stack((cs,cs+1,cs+2),1)[None,:,:]; return SimpleNamespace(hidden_states=[h0,h1],last_hidden_state=h1)
    model=Model(); fs=_MemFS(); args=SimpleNamespace(output_dir=_MemPath('main-out'),run_name='synthetic-run',exact_command='["python","observer.py"]',observer_implementation_commit=o.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),observer_script_sha256=o.sha256_file(Path(o.__file__)))
    result=o.run_observer(args,{'root':ROOT,'repository_head':args.observer_implementation_commit,'tokenizer_loader':lambda:Tok(),'model_loader':lambda:model,'filesystem':fs,'runtime_versions':{'python_version':'3.13','numpy_version':'2','torch_version':'2','transformers_version':'4'}})
    assert len(model.calls)==12 and all(set(x)=={'input_ids','output_hidden_states','return_dict','use_cache'} and x['output_hidden_states'] is True and x['return_dict'] is True and x['use_cache'] is False for x in model.calls)
    assert set(fs.writes)==set(o.REQUIRED_ARTIFACTS) and fs.writes[-1]=='SHA256SUMS.txt' and set(result['files'])==set(o.REQUIRED_ARTIFACTS)

def test_run_observer_without_runtime_version_injection_uses_production_capture(monkeypatch):
    coords=json.loads((ROOT/'reports/longterm_o0b_matched_controls_v1_validation.json').read_text()); rows=o.read_jsonl(ROOT/o.DATASET_PATH); byid={r['pair_id']:r for r in rows}; token_map={}
    for p in coords['pairs']:
        for c in ('reference_sufficient',*o.COMPARISON_ORDER):
            e=p['conditions'][c]; token_map[f"Claim: {byid[p['pair_id']]['claim']}\nEvidence: {byid[p['pair_id']][c]}"]=(e['full_serialized_token_ids'],e['full_offset_mapping'])
    class Tok:
        is_fast=True
        def __call__(self,text,**kw): ids,offs=token_map[text]; return {'input_ids':ids,'offset_mapping':offs}
    class Model:
        def __call__(self,**kw):
            ids=kw['input_ids'].detach().cpu().numpy()[0].astype(np.float32); h0=np.stack((ids+1,ids+2,ids+3),1)[None,:,:]; h1=np.stack((ids+4,ids+5,ids+6),1)[None,:,:]; return SimpleNamespace(hidden_states=[h0,h1],last_hidden_state=h1)
    calls=[]
    expected_versions={'python_version':'3.13.2','numpy_version':'2.2.1','torch_version':'2.6.0+cpu','transformers_version':'4.49.0'}
    def fake_capture():
        calls.append("capture")
        return expected_versions
    monkeypatch.setattr(o, "capture_runtime_versions", fake_capture)
    fs=_MemFS(); args=SimpleNamespace(output_dir=_MemPath('capture-out'),run_name='synthetic-run',exact_command='["python","observer.py"]',observer_implementation_commit=o.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),observer_script_sha256=o.sha256_file(Path(o.__file__)))
    result=o.run_observer(args,{'root':ROOT,'repository_head':args.observer_implementation_commit,'tokenizer_loader':lambda:Tok(),'model_loader':lambda:Model(),'filesystem':fs})
    assert calls == ["capture"] and {k:result["manifest"][k] for k in o.RUNTIME_VERSION_KEYS} == expected_versions
