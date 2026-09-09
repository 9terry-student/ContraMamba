"""Fail-closed, default-disabled observer for native Mamba recurrent states.

No torch/Transformers import occurs at import time.  This is a bounded
instrumentation and serialization library; its CLI refuses scientific runs.
"""
from __future__ import annotations
import argparse, ast, ctypes, ctypes.wintypes, errno, hashlib, importlib, importlib.util, inspect, io, json, os, re, shutil, sys, zipfile
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

IMPLEMENTATION_AUTHORITY_COMMIT="6eca52722aaffa214e8546c6b616e1f670aecf77"
SCIENTIFIC_DESIGN_AUTHORITY_COMMIT="242ad9ed70fc995ebda560911a7d0dfd2f18f9b3"
SCHEMA_VERSION="longterm_o0c_selective_ssm_native_state_dynamics_v1"; EXPERIMENT_NAME="longterm_o0c_selective_ssm_native_state_dynamics"
MODEL_ID=TOKENIZER_ID="state-spaces/mamba-130m-hf"; MODEL_REVISION=TOKENIZER_REVISION="5708daa364c50b880e7bd92eab456e0d34492ee9"
DATASET_PATH="data/longterm_o0b_matched_controls_v1.jsonl"; DATASET_SHA256="75a675bee49cb26eb0935d364f0f5d090922dd01576dfc23294961b28394aec2"
VALIDATION_ARTIFACT_PATH="reports/longterm_o0b_matched_controls_v1_validation.json"; VALIDATION_ARTIFACT_SHA256="e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff"
MAMBA_MODULE="transformers.models.mamba.modeling_mamba"; MAMBA_SHA256="4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83"; MAMBA_BYTES=39500
CACHE_MODULE=MAMBA_MODULE; CACHE_SHA256=MAMBA_SHA256; CACHE_BYTES=MAMBA_BYTES
CAPTURE_QUALNAME="MambaMixer.slow_forward"; CAPTURE_LINE=410
EXPECTED_VERSIONS={"python":"3.12.13","numpy":"2.0.2","torch":"2.10.0+cpu","transformers":"5.0.0"}
PAIR_ORDER=("o0b_pair_001","o0b_pair_002","o0b_pair_003"); CONDITION_ORDER=("reference_sufficient","insufficient_matched","paraphrase_sufficient","surface_null_matched")
COMPARISONS=(("comparison-A","insufficient_matched"),("comparison-B","paraphrase_sufficient"),("comparison-C","surface_null_matched")); ANCHOR_ORDER=("anchor_pre_minus_1","anchor_divergence","anchor_post_plus_1","anchor_post_plus_2","anchor_post_plus_4","anchor_terminal")
REQUIRED_ARTIFACTS=("manifest.json","state_rows.jsonl","full_recurrent_states.npz","paired_measurements.jsonl","summary.json","report.md","SHA256SUMS.txt")
STATE_KEYS=("schema_version","pair_id","condition","token_count","terminal_index","absolute_token_index","layer_index","layer_role","state_source","state_timing","source_module","source_qualname","source_line","vector_index","tensor_shape","flattened_size","dtype","device")
MEASUREMENT_KEYS=("schema_version","pair_id","comparison_id","reference_condition","member_condition","anchor_name","absolute_token_index","layer_index","reference_vector_index","member_vector_index","reference_previous_vector_index","member_previous_vector_index","normalized_l2_state_distance","reference_transition_l2","member_transition_l2","paired_transition_delta","transition_direction_cosine","pre_divergence_integrity_status")
SUMMARY_KEYS=("schema_version","experiment_name","measurement_order","comparison_order","anchor_order","rows")
SUMMARY_ROW_KEYS=("layer_index","layer_role","anchor_name","measurement_name","a_available_pair_ids","a_available_pair_count","a_mean","a_median","b_available_pair_ids","b_available_pair_count","b_mean","b_median","c_available_pair_ids","c_available_pair_count","c_mean","c_median","a_gt_b_comparable_pair_ids","a_gt_b_denominator","a_gt_b_count","a_gt_c_comparable_pair_ids","a_gt_c_denominator","a_gt_c_count")
MANIFEST_KEYS=("schema_version","experiment_name","scientific_design_authority_commit","implementation_authority_commit","observer_implementation_commit","observer_script_path","observer_script_sha256","observer_script_bytes","dataset_path","dataset_sha256","validation_artifact_path","validation_artifact_sha256","model_id","model_revision","tokenizer_id","tokenizer_revision","model_trust_remote_code","tokenizer_trust_remote_code","add_special_tokens","device","dtype","expected_python_version","observed_python_version","expected_numpy_version","observed_numpy_version","expected_torch_version","observed_torch_version","expected_transformers_version","observed_transformers_version","transformers_distribution_root","transformers_import_root","source_resolution_classification","backend_classification","mamba_source_module","mamba_source_sha256","mamba_source_bytes","cache_source_module","cache_source_sha256","cache_source_bytes","capture_source_qualname","capture_source_line","capture_state_source","capture_state_timing","pair_order","condition_order","comparison_order","anchor_order","layer_descriptors","serialization_template","exact_command","run_name","required_artifacts","equivalence_gate_status","capture_completeness_status","provenance_status","execution_status","blocker")
SUCCESS_STATUSES={"equivalence_gate_status":"PASS_EXACT_EQUIVALENCE_NONINTERFERENCE","capture_completeness_status":"PASS_COMPLETE_NATIVE_STATE_CAPTURE","provenance_status":"PASS_PROVENANCE_VALIDATED","execution_status":"PASS_EXECUTION_COMPLETE"}
class ContractError(RuntimeError): pass
def require(ok: bool, msg: str)->None:
    if not ok: raise ContractError(msg)
def sha256_bytes(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def canonical_json(value:Any)->bytes: return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode("utf-8")
def canonical_jsonl(rows:Sequence[Mapping[str,Any]])->bytes: return b"".join(canonical_json(dict(x)) for x in rows)
def _source(module:Any)->tuple[Path,bytes]:
    path=Path(getattr(module,"__file__","")).resolve(); require(path.is_file(),"source root"); return path,path.read_bytes()

@dataclass(frozen=True)
class _SyntheticTraceBinding:
    """Private test-only target.  It is deliberately not a scientific binding."""
    function: Any
    code: Any
    capture_line: int

def _synthetic_trace_binding(function:Any,capture_line:int,registered_role:str="synthetic_post_update") -> _SyntheticTraceBinding:
    """Private test-only target; its role argument is not scientific evidence."""
    require(inspect.isfunction(function) and type(capture_line) is int and capture_line > 0,"synthetic binding")
    return _SyntheticTraceBinding(function,function.__code__,capture_line)

def _validate_source_roles(data:bytes,code:Any)->None:
    """AST proof of the frozen recurrence -> readout -> cache sequence."""
    try: tree=ast.parse(data.decode("utf-8"))
    except (UnicodeDecodeError,SyntaxError) as e: raise ContractError("source role syntax") from e
    nodes={getattr(n,"lineno",None):n for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.Expr))}
    update,readout,persist=(nodes.get(409),nodes.get(410),nodes.get(417))
    def names(n:Any)->set[str]: return {x.id for x in ast.walk(n) if isinstance(x,ast.Name)}
    def target(n:Any)->str:
        t=n.targets[0] if isinstance(n,ast.Assign) else n.target
        return t.id if isinstance(t,ast.Name) else (t.attr if isinstance(t,ast.Attribute) else "")
    def value(n:Any)->Any: return n.value if isinstance(n,(ast.Assign,ast.AnnAssign)) else None
    # The frozen implementation factors its input term before the affine
    # recurrence: deltaB_u = discrete_B * hidden_states, then
    # ssm_state = discrete_A * old_ssm_state + deltaB_u.  Prove both ordered
    # AST roles, rather than accepting a matching collection of names.
    factored=[n for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AnnAssign)) and target(n)=="deltaB_u"]
    require(len(factored)==1,"source input role")
    input_term=factored[0]; input_names=names(value(input_term))
    require(getattr(input_term,"lineno",0)<409 and isinstance(value(input_term),ast.BinOp) and isinstance(value(input_term).op,ast.Mult) and {"discrete_B","hidden_states"} <= input_names,"source input role")
    update_names=names(value(update)) if update is not None else set()
    require(isinstance(update,(ast.Assign,ast.AnnAssign)) and target(update)=="ssm_state" and isinstance(value(update),ast.BinOp) and isinstance(value(update).op,ast.Add) and {"ssm_state","discrete_A","deltaB_u"} <= update_names,"source update role")
    readout_names=names(value(readout)) if readout is not None else set()
    require(isinstance(readout,(ast.Assign,ast.AnnAssign)) and target(readout)=="scan_output" and {"ssm_state","C"} <= readout_names,"source readout role")
    # A semicolon makes two ``Expr`` nodes share one line.  The old mapping
    # silently retained one of them, accepting a convolution write followed by
    # the valid recurrent write.  The frozen persistent-role line is one,
    # exclusive statement: exactly the recurrent cache copy below.
    persist_nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.Expr)) and getattr(n,"lineno",None)==417]
    require(len(persist_nodes)==1 and persist_nodes[0] is persist,"source recurrent cache role")
    call=persist.value if isinstance(persist,ast.Expr) else None
    cache_target=call.func.value if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=="copy_" else None
    require(isinstance(cache_target,ast.Subscript) and isinstance(cache_target.value,ast.Attribute) and cache_target.value.attr=="ssm_states" and isinstance(cache_target.value.value,ast.Name) and cache_target.value.value.id=="cache_params" and len(call.args)==1 and isinstance(call.args[0],ast.Name) and call.args[0].id=="ssm_state","source recurrent cache role")
    require(CAPTURE_LINE in {n for _,_,n in code.co_lines() if n is not None},"line binding")

def _validate_mamba_cache_roles(data:bytes)->None:
    """Fail closed unless frozen ``MambaCache`` keeps conv and SSM state distinct."""
    try: tree=ast.parse(data.decode("utf-8"))
    except (UnicodeDecodeError,SyntaxError) as e: raise ContractError("cache role syntax") from e
    caches=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="MambaCache"]
    require(len(caches)==1,"cache/recurrent ambiguity")
    cache=caches[0]
    def methods(name:str)->list[ast.FunctionDef]: return [n for n in cache.body if isinstance(n,ast.FunctionDef) and n.name==name]
    def self_attr(n:Any,name:str)->bool:
        return isinstance(n,ast.Attribute) and n.attr==name and isinstance(n.value,ast.Name) and n.value.id=="self"
    def names(n:Any)->set[str]: return {x.id for x in ast.walk(n) if isinstance(x,ast.Name)} | {x.attr for x in ast.walk(n) if isinstance(x,ast.Attribute)}
    def scoped_nodes(function:ast.FunctionDef)->list[ast.AST]:
        """Return nodes in this method, excluding nested function/lambda scopes."""
        nodes=[]
        def visit(node:ast.AST)->None:
            nodes.append(node)
            for child in ast.iter_child_nodes(node):
                if not isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda)): visit(child)
        for statement in function.body: visit(statement)
        return nodes
    def writes_family(function:ast.FunctionDef,family:str)->bool:
        for node in scoped_nodes(function):
            if isinstance(node,ast.AugAssign):
                target=node.target
                if isinstance(target,ast.Subscript) and self_attr(target.value,family) and isinstance(target.slice,ast.Name) and target.slice.id=="layer_idx": return True
            target=node.target if isinstance(node,ast.AnnAssign) else (node.targets[0] if isinstance(node,ast.Assign) and len(node.targets)==1 else None)
            if self_attr(target,family) or (isinstance(target,ast.Subscript) and self_attr(target.value,family)): return True
        return False
    def returns_persistent_family(function:ast.FunctionDef,family:str)->bool:
        returns=[node for node in scoped_nodes(function) if isinstance(node,ast.Return)]
        if len(returns)!=1: return False
        value=returns[0].value
        return isinstance(value,ast.Subscript) and self_attr(value.value,family) and isinstance(value.slice,ast.Name) and value.slice.id=="layer_idx"
    init=methods("__init__"); require(len(init)==1,"cache/recurrent ambiguity")
    init=init[0]
    family_assignments={"conv_states":[],"ssm_states":[]}
    for node in ast.walk(init):
        target=node.target if isinstance(node,ast.AnnAssign) else (node.targets[0] if isinstance(node,ast.Assign) and len(node.targets)==1 else None)
        if self_attr(target,"conv_states") or self_attr(target,"ssm_states"):
            family_assignments[target.attr].append(node.value)
    require(all(len(values)==1 and isinstance(values[0],ast.List) and not values[0].elts for values in family_assignments.values()),"cache/recurrent ambiguity")
    constructors={"conv_state":[],"ssm_state":[]}
    for node in ast.walk(init):
        target=node.target if isinstance(node,ast.AnnAssign) else (node.targets[0] if isinstance(node,ast.Assign) and len(node.targets)==1 else None)
        if isinstance(target,ast.Name) and target.id in constructors: constructors[target.id].append(node.value)
    require(all(len(values)==1 and isinstance(values[0],ast.Call) for values in constructors.values()),"cache/recurrent ambiguity")
    conv_names,ssm_names=names(constructors["conv_state"][0]),names(constructors["ssm_state"][0])
    require("conv_kernel_size" in conv_names and "ssm_state_size" not in conv_names and "ssm_state_size" in ssm_names and "conv_kernel_size" not in ssm_names,"cache/recurrent ambiguity")
    appends=[]
    for node in ast.walk(init):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="append" and len(node.args)==1 and not node.keywords:
            receiver=node.func.value; argument=node.args[0]
            if isinstance(receiver,ast.Attribute) and isinstance(receiver.value,ast.Name) and receiver.value.id=="self" and isinstance(argument,ast.Name): appends.append((receiver.attr,argument.id))
    require(appends.count(("conv_states","conv_state"))==1 and appends.count(("ssm_states","ssm_state"))==1 and len(appends)==2,"cache/recurrent ambiguity")
    for method_name,family,other in (("update_conv_state","conv_states","ssm_states"),("update_ssm_state","ssm_states","conv_states")):
        method=methods(method_name); require(len(method)==1,"cache/recurrent ambiguity")
        method=method[0]
        require(writes_family(method,family) and returns_persistent_family(method,family) and not any(self_attr(n,other) for n in scoped_nodes(method)),"cache/recurrent ambiguity")

def _validate_forward_dispatch(data:bytes,forward:Any,source_path:Path)->None:
    """Prove the frozen Mamba CPU dispatch falls through to ``slow_forward``."""
    require(inspect.isfunction(forward) and forward.__module__==MAMBA_MODULE and forward.__qualname__=="MambaMixer.forward" and forward.__code__.co_filename==str(source_path),"forward identity")
    try: tree=ast.parse(data.decode("utf-8"))
    except (UnicodeDecodeError,SyntaxError) as e: raise ContractError("forward dispatch syntax") from e
    mixers=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="MambaMixer"]
    require(len(mixers)==1,"unsupported backend")
    forwards=[n for n in mixers[0].body if isinstance(n,ast.FunctionDef) and n.name=="forward"]
    require(len(forwards)==1,"unsupported backend")
    function=forwards[0]
    body=list(function.body)
    if body and isinstance(body[0],ast.Expr) and isinstance(getattr(body[0],"value",None),ast.Constant) and isinstance(body[0].value.value,str): body=body[1:]
    def path(n:Any,parts:tuple[str,...])->bool:
        for part in reversed(parts):
            if not isinstance(n,ast.Attribute) or n.attr!=part: return False
            n=n.value
        return isinstance(n,ast.Name) and n.id=="self"
    def method_call(n:Any,name:str)->bool:
        return isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==name and isinstance(n.func.value,ast.Name) and n.func.value.id=="self"
    def cuda_guard(n:Any)->bool:
        return isinstance(n,ast.Compare) and len(n.ops)==len(n.comparators)==1 and isinstance(n.ops[0],ast.In) and isinstance(n.left,ast.Constant) and n.left.value=="cuda" and path(n.comparators[0],("x_proj","weight","device","type"))
    def expected_call(n:Any,name:str)->bool:
        return method_call(n,name) and not n.keywords and len(n.args)==4 and all(isinstance(arg,ast.Name) and arg.id==expected for arg,expected in zip(n.args,("hidden_states","cache_params","cache_position","attention_mask")))
    expected_kernels=("selective_state_update","selective_scan_fn","causal_conv1d_fn","causal_conv1d_update","mamba_inner_fn")
    def availability(n:Any)->bool:
        return isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="all" and not n.keywords and len(n.args)==1 and isinstance(n.args[0],(ast.Tuple,ast.List)) and [item.id if isinstance(item,ast.Name) else None for item in n.args[0].elts]==list(expected_kernels)
    # The frozen function has no executable wrapper work beyond this exact
    # assignment, CUDA dispatch, and unconditional CPU fallback.
    require(len(body)==3 and isinstance(body[0],ast.Assign) and len(body[0].targets)==1 and isinstance(body[0].targets[0],ast.Name) and body[0].targets[0].id=="is_fast_path_available" and availability(body[0].value),"unsupported backend")
    branch,fallback=body[1],body[2]
    require(isinstance(branch,ast.If) and not branch.orelse and isinstance(branch.test,ast.BoolOp) and isinstance(branch.test.op,ast.And) and len(branch.test.values)==3,"unsupported backend")
    guard_values=branch.test.values
    require(isinstance(guard_values[0],ast.Name) and guard_values[0].id=="is_fast_path_available" and cuda_guard(guard_values[1]) and isinstance(guard_values[2],ast.UnaryOp) and isinstance(guard_values[2].op,ast.Not) and isinstance(guard_values[2].operand,ast.Call) and isinstance(guard_values[2].operand.func,ast.Name) and guard_values[2].operand.func.id=="is_torchdynamo_compiling" and not guard_values[2].operand.args and not guard_values[2].operand.keywords,"unsupported backend")
    require(len(branch.body)==1 and isinstance(branch.body[0],ast.Return) and expected_call(branch.body[0].value,"cuda_kernels_forward"),"unsupported backend")
    require(isinstance(fallback,ast.Return) and expected_call(fallback.value,"slow_forward"),"unsupported backend")
    # Exact top-level shape above excludes nested definitions, so every return
    # reachable in this wrapper is visible in this function-only walk.
    returns=[n for n in ast.walk(function) if isinstance(n,ast.Return)]
    require(len(returns)==2 and set(returns)=={branch.body[0],fallback},"unsupported backend")
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call)]
    cuda_calls=[n for n in calls if method_call(n,"cuda_kernels_forward")]; slow_calls=[n for n in calls if method_call(n,"slow_forward")]
    require(len(cuda_calls)==1 and cuda_calls[0] is branch.body[0].value and len(slow_calls)==1 and slow_calls[0] is fallback.value,"unsupported backend")
    backend_calls=[n for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="self" and n.func.attr.endswith("_forward")]
    require(len(backend_calls)==2 and set(backend_calls)==set(cuda_calls+slow_calls),"unsupported backend")
    require(not any(isinstance(n.func,ast.Name) and n.func.id in {"mamba_inner_fn","selective_scan_fn","selective_state_update"} for n in calls),"unsupported backend")

def _runtime_environment()->tuple[Mapping[str,Any],Mapping[str,str]]:
    """The sole environmental lookup boundary used by the scientific gate."""
    modules={MAMBA_MODULE:importlib.import_module(MAMBA_MODULE),"transformers":importlib.import_module("transformers"),"torch":importlib.import_module("torch")}
    return modules,{"python":".".join(map(str,sys.version_info[:3])),"numpy":np.__version__,"torch":modules["torch"].__version__,"transformers":modules["transformers"].__version__}

def _resolve_and_validate_runtime_binding()->tuple[Any,int]:
    """Derive the sole scientific trace binding from the live validated runtime."""
    modules,versions=_runtime_environment()
    require(set(versions)==set(EXPECTED_VERSIONS) and dict(versions)==EXPECTED_VERSIONS,"runtime version")
    mp,mb=_source(modules[MAMBA_MODULE]); root_text=getattr(modules["transformers"],"__file__",None)
    require(isinstance(root_text,str) and Path(root_text).is_absolute(),"malformed import root")
    root=Path(root_text).resolve().parent
    require(root.is_dir(),"transformers distribution root")
    require(mp.is_relative_to(root),"shadowed import root")
    try: distribution_root=Path(importlib_metadata.distribution("transformers").locate_file("transformers")).resolve()
    except importlib_metadata.PackageNotFoundError as e: raise ContractError("transformers distribution unavailable") from e
    require(distribution_root == root,"import/distribution-root mismatch")
    spec=importlib.util.find_spec(MAMBA_MODULE); require(spec is None or Path(str(spec.origin)).resolve()==mp,"import/distribution-root mismatch")
    require(len(mb)==MAMBA_BYTES,"Mamba byte-size mismatch"); require(sha256_bytes(mb)==MAMBA_SHA256,"Mamba SHA256 mismatch")
    require(len(mb)==CACHE_BYTES,"Mamba cache byte-size mismatch"); require(sha256_bytes(mb)==CACHE_SHA256,"Mamba cache SHA256 mismatch")
    mixer=getattr(modules[MAMBA_MODULE],"MambaMixer",None); slow=getattr(mixer,"slow_forward",None); forward=getattr(mixer,"forward",None)
    require(inspect.isfunction(slow) and slow.__module__==MAMBA_MODULE and slow.__qualname__==CAPTURE_QUALNAME and slow.__code__.co_filename==str(mp),"slow code identity")
    _validate_source_roles(mb,slow.__code__)
    _validate_forward_dispatch(mb,forward,mp)
    _validate_mamba_cache_roles(mb)
    return slow.__code__,CAPTURE_LINE

def runtime_gate()->None:
    """Validate the actual runtime; it deliberately exposes no binding object."""
    _resolve_and_validate_runtime_binding()

class _TraceCollector:
    """Shared low-level CPython line collector; scientific policy lives above it."""
    def __init__(self,code:Any,capture_line:int,registered_layers:Mapping[int,Mapping[str,Any]],enabled:bool=False):
        self.code,self.capture_line,self.layers,self.enabled=code,capture_line,dict(registered_layers),bool(enabled); indexes=[x["layer_index"] for x in self.layers.values()]; require(len(indexes)==len(set(indexes)),"duplicate layer identity"); self.snapshots=None; self._prior=None; self._forward_id=0
    def _trace(self,frame:Any,event:str,arg:Any):
        if frame.f_code is self.code and event=="line" and frame.f_lineno==self.capture_line:
            layer=self.layers.get(id(frame.f_locals.get("self")))
            if layer is None: return self._trace
            index=frame.f_locals.get("i"); require(type(index) is int and index>=0,"ambiguous token index")
            state=frame.f_locals.get("ssm_state"); require(state is not None and hasattr(state,"detach") and hasattr(state,"clone"),"not native recurrent state")
            snap=state.detach().clone(); require(snap is not state,"snapshot alias")
            key=(self._forward_id,int(layer["layer_index"]),index); require(self.snapshots is not None and key not in self.snapshots,"duplicate coordinate"); self.snapshots[key]=snap
        return self._trace
    @contextmanager
    def capture(self):
        if not self.enabled: yield self; return
        require(self.snapshots is None,"observer reuse"); self.snapshots={}; self._forward_id+=1; self._prior=sys.gettrace(); sys.settrace(self._trace)
        try: yield self
        finally: sys.settrace(self._prior); self.layers.clear()

def _synthetic_collector(binding:_SyntheticTraceBinding,registered_layers:Mapping[int,Mapping[str,Any]],enabled:bool=True)->_TraceCollector:
    require(isinstance(binding,_SyntheticTraceBinding),"synthetic trace binding")
    return _TraceCollector(binding.code,binding.capture_line,registered_layers,enabled)

class NativeStateObserver(_TraceCollector):
    """Scientific observer bound only by revalidating the current runtime."""
    def __init__(self,registered_layers:Mapping[int,Mapping[str,Any]],enabled:bool=False):
        # Keep derivation in this construction implementation.  Replacing the
        # ordinary module-global resolver therefore cannot supply a code object
        # or source line to scientific construction.
        modules,versions=_runtime_environment()
        require(set(versions)==set(EXPECTED_VERSIONS) and dict(versions)==EXPECTED_VERSIONS,"runtime version")
        mp,mb=_source(modules[MAMBA_MODULE]); root_text=getattr(modules["transformers"],"__file__",None)
        require(isinstance(root_text,str) and Path(root_text).is_absolute(),"malformed import root")
        root=Path(root_text).resolve().parent
        require(root.is_dir(),"transformers distribution root")
        require(mp.is_relative_to(root),"shadowed import root")
        try: distribution_root=Path(importlib_metadata.distribution("transformers").locate_file("transformers")).resolve()
        except importlib_metadata.PackageNotFoundError as e: raise ContractError("transformers distribution unavailable") from e
        require(distribution_root == root,"import/distribution-root mismatch")
        spec=importlib.util.find_spec(MAMBA_MODULE); require(spec is None or Path(str(spec.origin)).resolve()==mp,"import/distribution-root mismatch")
        require(len(mb)==MAMBA_BYTES,"Mamba byte-size mismatch"); require(sha256_bytes(mb)==MAMBA_SHA256,"Mamba SHA256 mismatch")
        require(len(mb)==CACHE_BYTES,"Mamba cache byte-size mismatch"); require(sha256_bytes(mb)==CACHE_SHA256,"Mamba cache SHA256 mismatch")
        mixer=getattr(modules[MAMBA_MODULE],"MambaMixer",None); slow=getattr(mixer,"slow_forward",None); forward=getattr(mixer,"forward",None)
        require(inspect.isfunction(slow) and slow.__module__==MAMBA_MODULE and slow.__qualname__==CAPTURE_QUALNAME and slow.__code__.co_filename==str(mp),"slow code identity")
        _validate_source_roles(mb,slow.__code__)
        _validate_forward_dispatch(mb,forward,mp)
        _validate_mamba_cache_roles(mb)
        code,line=slow.__code__,CAPTURE_LINE
        super().__init__(code,line,registered_layers,enabled)

def create_native_state_observer(registered_layers:Mapping[int,Mapping[str,Any]],enabled:bool=False)->NativeStateObserver:
    """Construct a scientific observer; no caller may provide binding material."""
    return NativeStateObserver(registered_layers,enabled)

def validate_captures(captures:Mapping[int,Mapping[int,Any]],token_count:int,eligible_layers:Sequence[Mapping[str,Any]])->None:
    require(type(token_count) is int and token_count>0,"token_count"); ids=[int(x["layer_index"]) for x in eligible_layers]; require(ids==sorted(ids) and len(ids)==len(set(ids)) and set(captures)==set(ids),"layer completeness")
    for layer in ids: require(list(sorted(captures[layer]))==list(range(token_count)) and len(captures[layer])==token_count and max(captures[layer])==token_count-1,"captured indices")
def validate_member_orchestration(members:Sequence[Mapping[str,Any]],eligible_layers:Sequence[Mapping[str,Any]])->None:
    """Validate the frozen 3x4 independent-forward collection before assembly.

    This intentionally accepts only observer-owned synthetic collections; it
    neither runs a model nor infers missing members/layers.
    """
    layers=[int(x["layer_index"]) for x in eligible_layers]
    require(layers==sorted(layers) and len(layers)==len(set(layers)) and layers,"eligible layer order")
    expected=[(p,c) for p in PAIR_ORDER for c in CONDITION_ORDER]
    require(len(members)==len(expected),"member count")
    seen_members=set(); seen_collections=set(); seen_inner=set(); seen_states=set()
    for expected_identity,member in zip(expected,members):
        require(set(member)=={"pair_id","condition","captures","fresh_capture_collection","cache_reused"},"member schema")
        identity=(member["pair_id"],member["condition"])
        require(identity==expected_identity and identity not in seen_members,"member order/identity")
        seen_members.add(identity)
        require(member["fresh_capture_collection"] is True and member["cache_reused"] is False,"member freshness")
        captures=member["captures"]; require(isinstance(captures,Mapping) and id(captures) not in seen_collections,"shared capture collection")
        seen_collections.add(id(captures)); validate_captures(captures, len(next(iter(captures.values()))) if captures else 0, eligible_layers)
        for layer in layers:
            states=captures[layer]; require(id(states) not in seen_inner,"shared layer collection"); seen_inner.add(id(states))
            for state in states.values(): require(id(state) not in seen_states,"shared captured state"); seen_states.add(id(state))
def _array(value:Any)->np.ndarray:
    x=value.detach().clone().cpu() if hasattr(value,"detach") else np.array(value,copy=True); a=np.asarray(x,dtype="<f4"); require(a.ndim>=1 and a.size>0 and np.isfinite(a).all(),"state metadata"); return np.ascontiguousarray(a)
def state_rows(trajectories:Mapping[tuple[str,str,int],Sequence[Any]],descriptors:Sequence[Mapping[str,Any]])->tuple[list[dict[str,Any]],list[np.ndarray]]:
    desc={int(x["layer_index"]):dict(x) for x in descriptors}; require(list(desc)==sorted(desc) and len(desc)==len(descriptors),"layer descriptors"); rows=[]; vectors=[]
    for pair in PAIR_ORDER:
      for condition in CONDITION_ORDER:
       keys=sorted((k for k in trajectories if k[:2]==(pair,condition)),key=lambda k:k[2]); require([k[2] for k in keys]==list(desc),"all layer/member completeness")
       for key in keys:
        values=trajectories[key]; require(bool(values),"empty trajectory")
        for t,value in enumerate(values):
         a=_array(value); rows.append({"schema_version":SCHEMA_VERSION,"pair_id":pair,"condition":condition,"token_count":len(values),"terminal_index":len(values)-1,"absolute_token_index":t,"layer_index":key[2],"layer_role":desc[key[2]]["layer_role"],"state_source":"native_selective_ssm_recurrent_state","state_timing":"post_consumption_s_t","source_module":MAMBA_MODULE,"source_qualname":CAPTURE_QUALNAME,"source_line":CAPTURE_LINE,"vector_index":len(vectors),"tensor_shape":list(a.shape),"flattened_size":int(a.size),"dtype":"float32","device":"cpu"}); vectors.append(np.ascontiguousarray(a.reshape(-1),dtype="<f4"))
    matrix=np.ascontiguousarray(np.asarray(vectors,dtype="<f4")); validate_state_rows(rows,matrix); return rows,vectors
def validate_state_rows(rows:Sequence[Mapping[str,Any]],vectors:Sequence[Any]|np.ndarray|int|None=None)->None:
    """Validate physical vector position, shape and content as well as rows."""
    if isinstance(vectors,int): vector_count,matrix=vectors,None
    else:
     matrix=None if vectors is None else np.asarray(vectors); vector_count=None if matrix is None else matrix.shape[0]
     if matrix is not None: require(matrix.ndim==2 and matrix.shape[0]>0 and matrix.shape[1]>0 and matrix.dtype==np.dtype("<f4") and matrix.flags.c_contiguous and np.isfinite(matrix).all(),"vector matrix")
    require(bool(rows),"state rows")
    eligible=sorted({r["layer_index"] for r in rows})
    require(eligible and all(type(x) is int for x in eligible),"eligible layers")
    expected=[]; seen=set(); widths=set()
    for n,row in enumerate(rows):
     require(set(row)==set(STATE_KEYS),"state row schema"); require(row["schema_version"]==SCHEMA_VERSION and row["state_source"]=="native_selective_ssm_recurrent_state" and row["state_timing"]=="post_consumption_s_t" and row["source_module"]==MAMBA_MODULE and row["source_qualname"]==CAPTURE_QUALNAME and row["source_line"]==CAPTURE_LINE and row["dtype"]=="float32" and row["device"]=="cpu","state constants")
     shape=row["tensor_shape"]; require(isinstance(shape,list) and shape and all(type(x) is int and x>0 for x in shape),"tensor shape")
     size=1
     for dim in shape: size*=dim
     require(type(row["flattened_size"]) is int and row["flattened_size"]==size,"flattened size")
     require(row["vector_index"]==n and type(row["token_count"]) is int and row["token_count"]>0 and row["terminal_index"]==row["token_count"]-1 and 0<=row["absolute_token_index"]<=row["terminal_index"],"state indexing")
     key=(row["pair_id"],row["condition"],row["layer_index"],row["absolute_token_index"]); require(key not in seen,"duplicate coordinate"); seen.add(key)
     widths.add(row["flattened_size"])
     if matrix is not None: require(row["vector_index"] < matrix.shape[0] and row["flattened_size"]==matrix.shape[1] and np.isfinite(matrix[row["vector_index"]]).all(),"row/vector consistency")
    token_counts={(r["pair_id"],r["condition"],r["layer_index"]):r["token_count"] for r in rows}
    for pair in PAIR_ORDER:
     for condition in CONDITION_ORDER:
      for layer in eligible:
       count=token_counts.get((pair,condition,layer)); require(type(count) is int,"missing eligible layer")
       expected.extend((pair,condition,layer,t) for t in range(count))
    require([ (r["pair_id"],r["condition"],r["layer_index"],r["absolute_token_index"]) for r in rows ]==expected and seen==set(expected),"canonical state row membership/order")
    require(len(widths)<=1,"heterogeneous state widths")
    if vector_count is not None: require(len(rows)==vector_count,"row/vector count")
def deterministic_npz(vectors:Sequence[Any])->bytes:
    a=np.asarray(vectors,dtype="<f4"); require(a.ndim==2 and a.shape[0]>0 and a.shape[1]>0 and a.flags.c_contiguous and np.isfinite(a).all(),"vectors"); payload=io.BytesIO(); np.lib.format.write_array(payload,a,version=(1,0),allow_pickle=False); out=io.BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_STORED,allowZip64=False,compresslevel=None) as z:
     info=zipfile.ZipInfo("vectors.npy",(1980,1,1,0,0,0)); info.create_system=3; info.create_version=20; info.extract_version=20; info.external_attr=0o100644<<16; info.internal_attr=0; info.extra=b""; info.comment=b""; info.flag_bits=0; info.compress_type=zipfile.ZIP_STORED; z.comment=b""; z.writestr(info,payload.getvalue())
    return out.getvalue()
def parse_npz(data:bytes)->np.ndarray:
    try:
     with zipfile.ZipFile(io.BytesIO(data)) as z: require(z.namelist()==["vectors.npy"] and len(z.infolist())==1 and not z.comment,"NPZ members"); i=z.infolist()[0]; require(i.date_time==(1980,1,1,0,0,0) and i.compress_type==zipfile.ZIP_STORED and i.create_system==3 and i.create_version==20 and i.extract_version==20 and i.external_attr==0o100644<<16 and i.internal_attr==0 and i.extra==b"" and i.comment==b"" and i.flag_bits==0,"NPZ metadata")
     with np.load(io.BytesIO(data),allow_pickle=False) as z: require(z.files==["vectors"],"NPZ arrays"); a=z["vectors"]
    except (OSError,ValueError,KeyError,TypeError,AttributeError,zipfile.BadZipFile) as e: raise ContractError("invalid NPZ") from e
    require(isinstance(a,np.ndarray) and a.dtype==np.dtype("<f4") and a.ndim==2 and a.shape[0]>0 and a.shape[1]>0 and a.flags.c_contiguous and np.isfinite(a).all(),"NPZ vectors"); return a
def assert_pre_divergence(reference:Any,member:Any)->None: require(np.allclose(np.asarray(reference),np.asarray(member),rtol=0.0,atol=1e-6),"pre-divergence mismatch")

def _validate_comparison_anchors(comparison_anchors:Any,lookup:Mapping[tuple[str,str,int,int],Mapping[str,Any]],layers:Sequence[int])->None:
    expected={(pair,comparison) for pair in PAIR_ORDER for comparison,_ in COMPARISONS}
    require(isinstance(comparison_anchors,Mapping),"comparison anchors mapping")
    require(set(comparison_anchors)==expected,"comparison anchor keys")
    members=dict(COMPARISONS)
    for key,anchor_map in comparison_anchors.items():
     require(type(key) is tuple and len(key)==2,"comparison anchor key")
     pair,comparison=key; require(pair in PAIR_ORDER and comparison in members,"comparison anchor key")
     require(isinstance(anchor_map,Mapping) and set(anchor_map)==set(ANCHOR_ORDER),"comparison anchor map")
     member=members[comparison]
     for anchor,t in anchor_map.items():
      require(anchor in ANCHOR_ORDER and type(t) is int and t>=0,"comparison anchor coordinate")
      for layer in layers:
       require((pair,"reference_sufficient",layer,t) in lookup and (pair,member,layer,t) in lookup,"comparison anchor coordinate range")

def measurements(rows:Sequence[Mapping[str,Any]],vectors:Sequence[Any],comparison_anchors:Mapping[tuple[str,str],Mapping[str,int]])->list[dict[str,Any]]:
    matrix=np.ascontiguousarray(np.asarray(vectors,dtype="<f4")); validate_state_rows(rows,matrix); lookup={(r["pair_id"],r["condition"],r["layer_index"],r["absolute_token_index"]):r for r in rows}; layers=sorted({r["layer_index"] for r in rows}); _validate_comparison_anchors(comparison_anchors,lookup,layers); out=[]
    for pair in PAIR_ORDER:
     for comparison,member in COMPARISONS:
      anchor_map=comparison_anchors[(pair,comparison)]
      for layer in layers:
       dminus=anchor_map["anchor_pre_minus_1"]
       assert_pre_divergence(matrix[lookup[(pair,"reference_sufficient",layer,dminus)]["vector_index"]],matrix[lookup[(pair,member,layer,dminus)]["vector_index"]])
      for anchor in ANCHOR_ORDER:
       for layer in layers:
        t=anchor_map[anchor]; rr=lookup[(pair,"reference_sufficient",layer,t)]; mr=lookup[(pair,member,layer,t)]; rv=np.asarray(vectors[rr["vector_index"]],dtype="<f4"); mv=np.asarray(vectors[mr["vector_index"]],dtype="<f4"); rp=np.zeros_like(rv) if t==0 else np.asarray(vectors[lookup[(pair,"reference_sufficient",layer,t-1)]["vector_index"]],dtype="<f4"); mp=np.zeros_like(mv) if t==0 else np.asarray(vectors[lookup[(pair,member,layer,t-1)]["vector_index"]],dtype="<f4"); rt,mt=rv-rp,mv-mp; rn,mn,rtn,mtn=map(float,(np.linalg.norm(rv),np.linalg.norm(mv),np.linalg.norm(rt),np.linalg.norm(mt))); require(all(np.isfinite(x) and x>0 for x in (rn,mn,rtn,mtn)),"nonfinite/zero metric")
        out.append({"schema_version":SCHEMA_VERSION,"pair_id":pair,"comparison_id":comparison,"reference_condition":"reference_sufficient","member_condition":member,"anchor_name":anchor,"absolute_token_index":t,"layer_index":layer,"reference_vector_index":rr["vector_index"],"member_vector_index":mr["vector_index"],"reference_previous_vector_index":None if t==0 else lookup[(pair,"reference_sufficient",layer,t-1)]["vector_index"],"member_previous_vector_index":None if t==0 else lookup[(pair,member,layer,t-1)]["vector_index"],"normalized_l2_state_distance":float(np.linalg.norm(rv/rn-mv/mn)),"reference_transition_l2":rtn,"member_transition_l2":mtn,"paired_transition_delta":mtn-rtn,"transition_direction_cosine":float(np.dot(rt,mt)/(rtn*mtn)),"pre_divergence_integrity_status":"PASS"})
    validate_measurements(out,rows,vectors); return out
def validate_measurements(records:Sequence[Mapping[str,Any]],rows:Sequence[Mapping[str,Any]],vectors:Sequence[Any])->None:
    matrix=np.ascontiguousarray(np.asarray(vectors,dtype="<f4")); validate_state_rows(rows,matrix); byindex={r["vector_index"]:r for r in rows}; indexes=set(byindex); pairs=dict(COMPARISONS); seen=[]
    pre={(r["pair_id"],r["comparison_id"],r["layer_index"]):r for r in records if r.get("anchor_name")=="anchor_pre_minus_1"}
    for r in records:
     require(set(r)==set(MEASUREMENT_KEYS) and r["comparison_id"] in pairs and r["member_condition"]==pairs[r["comparison_id"]] and r["reference_condition"]=="reference_sufficient" and r["anchor_name"] in ANCHOR_ORDER,"measurement schema")
     require(r["reference_vector_index"] in indexes and r["member_vector_index"] in indexes,"measurement references"); rr,mr=byindex[r["reference_vector_index"]],byindex[r["member_vector_index"]]
     t=r["absolute_token_index"]; require(type(t) is int and (rr["pair_id"],rr["condition"],rr["layer_index"],rr["absolute_token_index"])==(r["pair_id"],"reference_sufficient",r["layer_index"],t) and (mr["pair_id"],mr["condition"],mr["layer_index"],mr["absolute_token_index"])==(r["pair_id"],r["member_condition"],r["layer_index"],t),"measurement coordinate")
     previous=(r["reference_previous_vector_index"],r["member_previous_vector_index"])
     if t==0: require(previous==(None,None),"t0 predecessor"); rp=np.zeros_like(matrix[rr["vector_index"]]); mp=np.zeros_like(matrix[mr["vector_index"]])
     else:
      require(all(type(x) is int and x in indexes for x in previous),"previous indices"); pr,pm=byindex[previous[0]],byindex[previous[1]]; require((pr["pair_id"],pr["condition"],pr["layer_index"],pr["absolute_token_index"])==(r["pair_id"],"reference_sufficient",r["layer_index"],t-1) and (pm["pair_id"],pm["condition"],pm["layer_index"],pm["absolute_token_index"])==(r["pair_id"],r["member_condition"],r["layer_index"],t-1),"wrong predecessor"); rp,mp=matrix[previous[0]],matrix[previous[1]]
     rv,mv=matrix[rr["vector_index"]],matrix[mr["vector_index"]]; rt,mt=rv-rp,mv-mp; rn,mn,rtn,mtn=map(float,(np.linalg.norm(rv),np.linalg.norm(mv),np.linalg.norm(rt),np.linalg.norm(mt))); require(all(np.isfinite(x) and x>0 for x in (rn,mn,rtn,mtn)),"metric norms")
     expected={"normalized_l2_state_distance":float(np.linalg.norm(rv/rn-mv/mn)),"reference_transition_l2":rtn,"member_transition_l2":mtn,"paired_transition_delta":mtn-rtn,"transition_direction_cosine":float(np.dot(rt,mt)/(rtn*mtn))}
     pre_row=pre.get((r["pair_id"],r["comparison_id"],r["layer_index"])); require(pre_row is not None,"pre-divergence coordinate")
     derived="PASS" if np.allclose(matrix[pre_row["reference_vector_index"]],matrix[pre_row["member_vector_index"]],rtol=0.0,atol=1e-6) else "FAIL"
     require(r["pre_divergence_integrity_status"]==derived=="PASS" and all(r[k]==v for k,v in expected.items()),"measurement reconstruction")
     seen.append((r["pair_id"],r["comparison_id"],r["anchor_name"],r["layer_index"]))
    _require_measurement_coordinates(records,rows,seen)

def _require_measurement_coordinates(records:Sequence[Mapping[str,Any]],rows:Sequence[Mapping[str,Any]],seen:Sequence[tuple[Any,...]]|None=None)->None:
    layers=sorted({r["layer_index"] for r in rows}); expected=[(p,c,a,l) for p in PAIR_ORDER for c,_ in COMPARISONS for a in ANCHOR_ORDER for l in layers]
    actual=list(seen) if seen is not None else [(r.get("pair_id"),r.get("comparison_id"),r.get("anchor_name"),r.get("layer_index")) for r in records]
    require(actual==expected and len(set(actual))==len(actual),"complete measurement coordinates/order")

def build_manifest(fields:Mapping[str,Any])->dict[str,Any]:
    m=dict(fields); require(set(m)==set(MANIFEST_KEYS),"manifest schema")
    fixed={"schema_version":SCHEMA_VERSION,"experiment_name":EXPERIMENT_NAME,"scientific_design_authority_commit":SCIENTIFIC_DESIGN_AUTHORITY_COMMIT,"implementation_authority_commit":IMPLEMENTATION_AUTHORITY_COMMIT,"model_id":MODEL_ID,"tokenizer_id":TOKENIZER_ID,"model_revision":MODEL_REVISION,"tokenizer_revision":TOKENIZER_REVISION,"dataset_path":DATASET_PATH,"dataset_sha256":DATASET_SHA256,"validation_artifact_path":VALIDATION_ARTIFACT_PATH,"validation_artifact_sha256":VALIDATION_ARTIFACT_SHA256,"mamba_source_module":MAMBA_MODULE,"mamba_source_sha256":MAMBA_SHA256,"mamba_source_bytes":MAMBA_BYTES,"cache_source_module":CACHE_MODULE,"cache_source_sha256":CACHE_SHA256,"cache_source_bytes":CACHE_BYTES,"capture_source_qualname":CAPTURE_QUALNAME,"capture_source_line":CAPTURE_LINE,"capture_state_source":"native_selective_ssm_recurrent_state","capture_state_timing":"post_consumption_s_t","serialization_template":"canonical-json-v1/deterministic-npz-v1"}
    require(all(m[k]==v for k,v in fixed.items()),"manifest constants"); require(m["required_artifacts"]==list(REQUIRED_ARTIFACTS) and m["pair_order"]==list(PAIR_ORDER) and m["condition_order"]==list(CONDITION_ORDER) and m["comparison_order"]==[x[0] for x in COMPARISONS] and m["anchor_order"]==list(ANCHOR_ORDER),"manifest orders")
    for key,size in (("observer_implementation_commit",40),("observer_script_sha256",64)): require(type(m[key]) is str and re.fullmatch(rf"[0-9a-f]{{{size}}}",m[key]) is not None,"manifest provenance")
    for key in ("observed_python_version","observed_numpy_version","observed_torch_version","observed_transformers_version","transformers_distribution_root","transformers_import_root","exact_command","run_name"): require(type(m[key]) is str and m[key].strip().lower() not in ("","unknown","n/a"),"manifest required string")
    require(m["observer_script_path"]=="scripts/observe_longterm_o0c_selective_ssm_native_state_dynamics.py" and m["model_trust_remote_code"] is False and m["tokenizer_trust_remote_code"] is False and m["add_special_tokens"] is False and m["device"]=="cpu" and m["dtype"]=="float32","manifest fixed policy")
    require(all(m["expected_"+k+"_version"]==v for k,v in EXPECTED_VERSIONS.items()),"expected versions")
    require(m["source_resolution_classification"]=="PASS_RECONCILED_UNIQUE_TRANSFORMERS_SOURCE" and m["backend_classification"]=="BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN","source classifications")
    for key in ("transformers_distribution_root","transformers_import_root"):
     require(Path(m[key]).is_absolute() and str(Path(m[key]))==m[key],"normalized absolute root")
    require(type(m["layer_descriptors"]) is list and m["layer_descriptors"] and all(isinstance(x,dict) and set(x)=={"layer_index","layer_role"} and type(x["layer_index"]) is int for x in m["layer_descriptors"]),"layer descriptors")
    identity=observer_script_identity()
    require(m["observer_script_path"]==identity["observer_script_path"] and m["observer_script_sha256"]==identity["observer_script_sha256"] and m["observer_script_bytes"]==identity["observer_script_bytes"],"observer script identity")
    require(all(m[key]==value for key,value in SUCCESS_STATUSES.items()) and m["blocker"] is None,"publication status vocabulary"); return m

def observer_script_identity()->dict[str,Any]:
    path=Path(__file__).resolve(); root=path.parent.parent.resolve()
    require(path.is_relative_to(root) and path.relative_to(root).as_posix()=="scripts/observe_longterm_o0c_selective_ssm_native_state_dynamics.py","observer script path")
    data=path.read_bytes(); return {"observer_script_path":path.relative_to(root).as_posix(),"observer_script_sha256":sha256_bytes(data),"observer_script_bytes":len(data)}

def build_summary(records:Sequence[Mapping[str,Any]], rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    _require_measurement_coordinates(records,rows)
    descriptors={r["layer_index"]:r["layer_role"] for r in rows}; result=[]
    for layer,role in sorted(descriptors.items()):
     for anchor in ANCHOR_ORDER:
      for metric in ("normalized_l2_state_distance","paired_transition_delta","transition_direction_cosine"):
       values={cid:[(x["pair_id"],float(x[metric])) for x in records if x["comparison_id"]==cid and x["layer_index"]==layer and x["anchor_name"]==anchor] for cid,_ in COMPARISONS}; item={"layer_index":layer,"layer_role":role,"anchor_name":anchor,"measurement_name":metric}
       for prefix,(cid,_) in zip(("a","b","c"),COMPARISONS):
        ids=[p for p,_ in values[cid]]; vs=[v for _,v in values[cid]]; item.update({f"{prefix}_available_pair_ids":ids,f"{prefix}_available_pair_count":len(ids),f"{prefix}_mean":float(np.mean(vs)) if vs else None,f"{prefix}_median":float(np.median(vs)) if vs else None})
       for prefix,left,right in (("a_gt_b","comparison-A","comparison-B"),("a_gt_c","comparison-A","comparison-C")):
        lm,rm=dict(values[left]),dict(values[right]); ids=[p for p in PAIR_ORDER if p in lm and p in rm]; item.update({f"{prefix}_comparable_pair_ids":ids,f"{prefix}_denominator":len(ids),f"{prefix}_count":sum(lm[p]>rm[p] for p in ids)})
       result.append(item)
    return {"schema_version":SCHEMA_VERSION,"experiment_name":EXPERIMENT_NAME,"measurement_order":["normalized_l2_state_distance","paired_transition_delta","transition_direction_cosine"],"comparison_order":[x[0] for x in COMPARISONS],"anchor_order":list(ANCHOR_ORDER),"rows":result}

def validate_summary(summary:Mapping[str,Any],records:Sequence[Mapping[str,Any]]|None=None,rows:Sequence[Mapping[str,Any]]|None=None)->None:
    require(set(summary)==set(SUMMARY_KEYS) and summary["schema_version"]==SCHEMA_VERSION and summary["experiment_name"]==EXPERIMENT_NAME,"summary schema"); require(summary["measurement_order"]==["normalized_l2_state_distance","paired_transition_delta","transition_direction_cosine"] and summary["comparison_order"]==[x[0] for x in COMPARISONS] and summary["anchor_order"]==list(ANCHOR_ORDER),"summary constants")
    require(bool(summary["rows"]),"summary completeness")
    for row in summary["rows"]: require(set(row)==set(SUMMARY_ROW_KEYS),"summary row schema")
    if records is not None and rows is not None:
     _require_measurement_coordinates(records,rows)
     require(dict(summary)==build_summary(records,rows),"summary reconstruction")
def checksum_text(files:Mapping[str,bytes])->bytes: return b"".join(f"{sha256_bytes(files[n])}  {n}\n".encode("ascii") for n in REQUIRED_ARTIFACTS[:-1])
def validate_checksums(data:bytes,files:Mapping[str,bytes])->None: require(data==checksum_text(files),"checksums")
def render_report(manifest:Mapping[str,Any],measures:Sequence[Mapping[str,Any]],summary:Mapping[str,Any])->bytes:
    return ("# O0c Selective-SSM Native Recurrent-State Dynamics\n\nSCIENTIFIC_CONCLUSION: NONE\n\n"+canonical_json({"manifest":manifest,"measurement_count":len(measures),"summary":summary}).decode()).encode("utf-8")
def validate_bundle(files:Mapping[str,bytes])->None:
    require(tuple(files)==REQUIRED_ARTIFACTS,"artifact set")
    for name,data in files.items(): require(data and (not name.endswith((".json",".jsonl",".txt",".md")) or (b"\r" not in data and data.endswith(b"\n"))),"artifact encoding")
    manifest=json.loads(files["manifest.json"]); require(canonical_json(manifest)==files["manifest.json"],"manifest canonical"); build_manifest(manifest)
    decoded={}
    for name in ("state_rows.jsonl","paired_measurements.jsonl"):
     decoded[name]=[json.loads(line) for line in files[name].splitlines()]; require(all(canonical_json(value)==line+b"\n" for value,line in zip(decoded[name],files[name].splitlines())),"jsonl canonical")
    vectors=parse_npz(files["full_recurrent_states.npz"]); validate_state_rows(decoded["state_rows.jsonl"],vectors); validate_measurements(decoded["paired_measurements.jsonl"],decoded["state_rows.jsonl"],vectors)
    summary=json.loads(files["summary.json"]); require(canonical_json(summary)==files["summary.json"],"summary canonical"); validate_summary(summary,decoded["paired_measurements.jsonl"],decoded["state_rows.jsonl"]); require(files["report.md"]==render_report(manifest,decoded["paired_measurements.jsonl"],summary),"report rendering"); validate_checksums(files["SHA256SUMS.txt"],files)
def build_bundle(manifest:Mapping[str,Any],rows:Sequence[Mapping[str,Any]],vectors:Sequence[Any],measures:Sequence[Mapping[str,Any]],summary:Mapping[str,Any])->dict[str,bytes]:
    manifest=build_manifest(manifest); matrix=np.ascontiguousarray(np.asarray(vectors,dtype="<f4")); validate_state_rows(rows,matrix); parse_npz(deterministic_npz(matrix)); validate_measurements(measures,rows,matrix); validate_summary(summary,measures,rows)
    files={"manifest.json":canonical_json(manifest),"state_rows.jsonl":canonical_jsonl(rows),"full_recurrent_states.npz":deterministic_npz(vectors),"paired_measurements.jsonl":canonical_jsonl(measures),"summary.json":canonical_json(summary)}; files["report.md"]=render_report(manifest,measures,summary); files["SHA256SUMS.txt"]=checksum_text(files); validate_bundle(files); return files
def _atomic_rename_noreplace_directory(src:Path,dst:Path)->None:
    """One native directory move, no replacement fallback on any platform."""
    if os.name=="nt":
     kernel32=ctypes.WinDLL("kernel32",use_last_error=True); move=kernel32.MoveFileExW
     move.argtypes=(ctypes.wintypes.LPCWSTR,ctypes.wintypes.LPCWSTR,ctypes.wintypes.DWORD); move.restype=ctypes.wintypes.BOOL
     ctypes.set_last_error(0)
     if not move(str(src),str(dst),0):
      code=ctypes.get_last_error()
      if code in (80,183): raise ContractError("output collision")
      require(type(code) is int and code!=0,"MoveFileExW failed without native error")
      raise OSError(code,"MoveFileExW failed")
     return
    if sys.platform.startswith("linux"):
     libc=ctypes.CDLL(None,use_errno=True); renameat2=getattr(libc,"renameat2",None)
     if renameat2 is None: raise ContractError("atomic no-replace unsupported")
     renameat2.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint); renameat2.restype=ctypes.c_int
     if renameat2(-100,os.fsencode(src),-100,os.fsencode(dst),1)!=0:
      code=ctypes.get_errno()
      if code in (errno.EEXIST,errno.ENOTEMPTY): raise ContractError("output collision")
      raise OSError(code,"renameat2 failed")
     return
    raise ContractError("atomic no-replace unsupported")
def publish_bundle(output:Path,files:Mapping[str,bytes])->None:
    """Publish through an atomic directory claim; never replace a final path.

    ``mkdir(output)`` is the no-clobber operation.  Unlike check+replace it
    cannot overwrite a directory that appears after staging validation.
    """
    output=Path(output); staging=output.with_name(output.name+".staging"); require(not output.exists(),"output collision"); require(not staging.exists(),"staging collision"); validate_bundle(files); staging.mkdir()
    try:
     for name in REQUIRED_ARTIFACTS: (staging/name).write_bytes(files[name])
     validate_bundle({name:(staging/name).read_bytes() for name in REQUIRED_ARTIFACTS})
     _atomic_rename_noreplace_directory(staging,output)
    except Exception:
     if staging.exists():
      for child in staging.iterdir(): child.unlink()
      staging.rmdir()
     raise
def main(argv:Sequence[str]|None=None)->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir",type=Path); args=parser.parse_args(argv)
    if args.output_dir is not None: raise ContractError("scientific execution requires separate authorization")
if __name__=="__main__": main()
