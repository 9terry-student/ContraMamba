from __future__ import annotations
import argparse, hashlib, json, math, struct, subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch

from scripts import reason_router_gen4_family_subspace_sensitivity_fast_cuda as basis_prior
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter
from scripts import reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda as holdout

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"
AUTHORITY_COMMIT = "0f428d84a5065268c9a4dfaf80855483c4e99c42"
STATIC_COMMIT = "e470f132a37c731754feb4333dcb2e48e29af53b"
AUTHORITY_PATH = "reports/reason_router_gen4_pp3_necessity_implementation_authority.md"
AUTHORITY_BLOB = "b5dba9cf17ac04cfed0eaa19ae37bd284e7686fb"

DATA_ROOT = Path("data/reason_router_gen4_xg1_necessity_v1")
STATIC_ROOT = Path("reports/reason_router_gen4_pp3_necessity_static_preparation_f4419f1")
SOURCE_SHA = "49bec37150630d31bb5f502f49ef23ffc9a75bb93e079127c8c430aae3da6abd"
ROWS_SHA = "e03534599c07201e371eb07938d8492d22a30de39dcbbb3c0c22300a4ff94224"
STRUCT_SHA = "219994e7ec757148ffcae30a7d50357e347530e06cecbb618db619b73bbe8b76"
ANCHOR_SHA = "67d74cd2b3ab8daa8065227b1e85ea125ca88af4de64c872e5d47f60d9c40ec2"
ELIG_SHA = "b2f05232c3e319c22fa171cdd141d337966bc1ed47f15658932f09368097f6c3"

PP3_ROOT = Path("reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89")
PP5_ROOT = Path("reports/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_preparation_0bc49ab")
PLANE_FILES = {
    "pp3_plus": (PP3_ROOT / "pp3_plus.f64le", "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff"),
    "pp3_minus": (PP3_ROOT / "pp3_minus.f64le", "ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7"),
    "pp5_plus": (PP5_ROOT / "pp5_plus.f64le", "7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2"),
    "pp5_minus": (PP5_ROOT / "pp5_minus.f64le", "311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855"),
}
PLAN_SHA = {
    "xg2": "b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c",
    "xg4": "792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f",
}
N, ROWS, DIM, K, EPS = 300, 1800, 395, 5, 0.025
CONDITIONS = ("native", "pp3_neutralized", "pp5_coefficient_control")
DIRECTIONS = tuple([f"xg2_{i}" for i in range(K)] + [f"xg4_{i}" for i in range(K)])
F_SIGNED, F_DIR, F_COND, F_PAIR, F_TOTAL = 2, 4, 40, 120, 36000
TOL = 1e-12

ITEM_FILE = "pp3_necessity_items.jsonl"
SUMMARY_FILE = "pp3_necessity_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
ITEM_SCHEMA = "gen4-pp3-necessity-item-v1"
SUMMARY_SCHEMA = "gen4-pp3-necessity-summary-v1"
MANIFEST_SCHEMA = "gen4-pp3-necessity-manifest-v1"
RESULT_PASS = "PASS_PP3_NECESSITY_RAW_OBSERVATION"

class PP3NecessityError(RuntimeError): pass
def require(x: bool, msg: str) -> None:
    if not x: raise PP3NecessityError(msg)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def sha256_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def git(*a: str) -> str:
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise PP3NecessityError("GIT_FAILURE:" + " ".join(a)) from e

def authenticate_repo(expected_head: str) -> None:
    branch, head = git("branch","--show-current"), git("rev-parse","HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status","--porcelain") == "", "WORKTREE_NOT_CLEAN")
    for c, label in ((STATIC_COMMIT,"STATIC"), (AUTHORITY_COMMIT,"AUTHORITY")):
        rc = subprocess.call(["git","merge-base","--is-ancestor",c,head], cwd=ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        require(rc == 0, f"{label}_NOT_ANCESTOR")
    require(git("rev-parse",f"HEAD:{AUTHORITY_PATH}") == AUTHORITY_BLOB, "AUTHORITY_BLOB_DRIFT")

def expected_pairs() -> tuple[str,...]:
    return tuple(f"xg1_fact_{i:03d}" for i in range(601,901))

def read_jsonl(p: Path) -> list[dict[str,Any]]:
    out=[]
    for n,line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip(): continue
        v=json.loads(line); require(isinstance(v,dict), f"JSONL_OBJECT:{p}:{n}"); out.append(v)
    return out

def validate_static_inputs() -> None:
    checks = {
        DATA_ROOT/"structured_source_facts.jsonl": SOURCE_SHA,
        DATA_ROOT/"synthetic_reason_router_six_cell.jsonl": ROWS_SHA,
        DATA_ROOT/"structural_manifest.json": STRUCT_SHA,
        STATIC_ROOT/"tokenizer_anchor_manifest.jsonl": ANCHOR_SHA,
        STATIC_ROOT/"tokenizer_eligibility_summary.json": ELIG_SHA,
    }
    for p,s in checks.items():
        require((ROOT/p).is_file(), f"MISSING:{p}")
        require(sha256_file(ROOT/p) == s, f"SHA:{p}")
    e=json.loads((ROOT/STATIC_ROOT/"tokenizer_eligibility_summary.json").read_text(encoding="utf-8-sig"))
    require(e["result"]=="PASS_300_OF_300" and e["model_forward_count"]==0
            and e["checkpoint_load_count"]==0 and e["gpu_used"] is False, "ELIGIBILITY_BOUNDARY")

def vector(raw: bytes, label: str) -> torch.Tensor:
    require(len(raw)==DIM*8, f"VECTOR_BYTES:{label}")
    x=torch.tensor(struct.unpack(f"<{DIM}d",raw),dtype=torch.float64)
    require(bool(torch.isfinite(x).all()), f"VECTOR_FINITE:{label}")
    require(abs(float(torch.linalg.vector_norm(x))-1.0)<=TOL, f"VECTOR_NORM:{label}")
    return x.contiguous()

def load_planes() -> dict[str,torch.Tensor]:
    out={}
    for name,(rel,s) in PLANE_FILES.items():
        p=ROOT/rel; require(p.is_file() and sha256_file(p)==s, f"PLANE:{name}")
        out[name]=vector(p.read_bytes(),name)
    require(abs(float(torch.dot(out["pp3_plus"],out["pp3_minus"])))<=TOL,"PP3_ORTH")
    require(abs(float(torch.dot(out["pp5_plus"],out["pp5_minus"])))<=TOL,"PP5_ORTH")
    cross=torch.stack([out["pp3_plus"],out["pp3_minus"]],1).T @ torch.stack([out["pp5_plus"],out["pp5_minus"]],1)
    require(float(torch.max(torch.abs(cross)))<=TOL,"CROSS_ORTH")
    return out

def load_bases() -> dict[str,torch.Tensor]:
    loaded=holdout._load_frozen_bases(); out={}
    for fam in ("xg2","xg4"):
        require(loaded[fam]["plan_sha256"]==PLAN_SHA[fam], f"PLAN_SHA:{fam}")
        b=loaded[fam]["basis"]["basis"].detach().cpu().to(torch.float64).contiguous()
        require(tuple(b.shape)==(DIM,K), f"BASIS_SHAPE:{fam}")
        out[fam]=b
    return out

def pair_order(rows: Sequence[Mapping[str,Any]]) -> tuple[str,...]:
    seen=set(); order=[]
    for r in rows:
        p=str(r["source_pair_id"])
        if p not in seen: seen.add(p); order.append(p)
    require(tuple(order)==expected_pairs(),"PAIR_ORDER")
    return tuple(order)

def load_inputs(tokenizer_snapshot: str|Path|None):
    validate_static_inputs()
    rows=adapter.validate_gen4_rows(read_jsonl(ROOT/DATA_ROOT/"synthetic_reason_router_six_cell.jsonl"),
                                    require_canonical_shape=True)
    pairs=pair_order(rows)
    tok,_=tokenizer_gate.load_canonical_analysis_tokenizer(tokenizer_snapshot)
    enc=adapter.encode_gen4_rows(rows,tok)
    events_rows=read_jsonl(ROOT/STATIC_ROOT/"tokenizer_anchor_manifest.jsonl")
    require(len(rows)==ROWS and len(events_rows)==ROWS,"ROW_COUNTS")
    require(Counter(str(r["anchor_name"]) for r in events_rows)==Counter({"A_IDENTITY":1200,"A_NAME":600}),
            "ANCHOR_COUNTS")
    runtime=holdout.phase1.base.prevalence_eq; parent=runtime.parent
    events=parent.event_lookup(events_rows); parent.validate_transport_event_plan(pairs,events)
    require(list(enc["source_pair_id"])==[str(r["source_pair_id"]) for r in rows],"ENCODED_PAIR_ORDER")
    return rows,enc,events_rows

def condition_correction(h: torch.Tensor, condition: str, planes: Mapping[str,torch.Tensor]) -> dict[str,Any]:
    require(condition in CONDITIONS, f"CONDITION:{condition}")
    h=h.detach().cpu().to(torch.float64).contiguous(); require(tuple(h.shape)==(DIM,),"H_SHAPE")
    a=float(torch.dot(h,planes["pp3_plus"])); b=float(torch.dot(h,planes["pp3_minus"]))
    if condition=="native": d=torch.zeros_like(h)
    elif condition=="pp3_neutralized": d=-a*planes["pp3_plus"]-b*planes["pp3_minus"]
    else: d=-a*planes["pp5_plus"]-b*planes["pp5_minus"]
    post=h+d; rp=float(torch.dot(post,planes["pp3_plus"])); rm=float(torch.dot(post,planes["pp3_minus"]))
    if condition=="pp3_neutralized":
        require(abs(rp)<=TOL and abs(rm)<=TOL, f"NEUTRAL_RESIDUAL:{rp}:{rm}")
    return {"a":a,"b":b,"d":d.contiguous(),"l2":float(torch.linalg.vector_norm(d)),
            "res_plus":rp,"res_minus":rm}

def apply_hook(output: torch.Tensor, *, token_index: int, strong_mask: torch.Tensor,
               condition: str, planes: Mapping[str,torch.Tensor], direction: torch.Tensor,
               orientation: int, branch_sign: int, audit: dict[str,Any]) -> torch.Tensor:
    runtime=holdout.phase1.base.prevalence_eq; core=runtime.core
    require(output.ndim==3 and output.shape[0]==1 and output.shape[-1]==2*core.INTERMEDIATE_SIZE,"INPROJ_SHAPE")
    require(orientation in {-1,1} and branch_sign in {-1,1},"SIGN")
    m=strong_mask.detach().cpu().bool().contiguous()
    require(m.numel()==core.INTERMEDIATE_SIZE and int(m.sum())==DIM,"MASK")
    before=output.detach().clone(); md=m.to(before.device)
    h=before[0,token_index,:core.INTERMEDIATE_SIZE][md].detach().cpu().to(torch.float64).contiguous()
    ci=condition_correction(h,condition,planes)
    v=direction.detach().cpu().to(torch.float64).contiguous()
    require(tuple(v.shape)==(DIM,) and abs(float(torch.linalg.vector_norm(v))-1.0)<=TOL,"DIRECTION")
    probe=v*(float(branch_sign)*float(orientation)*EPS); total=(ci["d"]+probe).contiguous()
    out=output.clone(); intended=total.to(device=out.device,dtype=out.dtype)
    out[0,token_index,:core.INTERMEDIATE_SIZE][md] += intended
    require(torch.equal(out[:,:,core.INTERMEDIATE_SIZE:],before[:,:,core.INTERMEDIATE_SIZE:]),"GATE_CHANGED")
    non=~md
    require(torch.equal(out[:,:,:core.INTERMEDIATE_SIZE][:,:,non],before[:,:,:core.INTERMEDIATE_SIZE][:,:,non]),"NONSTRONG_CHANGED")
    if token_index: require(torch.equal(out[:,:token_index,:],before[:,:token_index,:]),"EARLIER_CHANGED")
    if token_index+1<out.shape[1]: require(torch.equal(out[:,token_index+1:,:],before[:,token_index+1:,:]),"LATER_CHANGED")
    applied=(out[0,token_index,:core.INTERMEDIATE_SIZE][md]-before[0,token_index,:core.INTERMEDIATE_SIZE][md]).detach().cpu().to(torch.float64)
    residual=float(torch.max(torch.abs(applied-intended.detach().cpu().to(torch.float64))))
    require(residual<=runtime.transport_runtime.RUNTIME_CAST_TOL,f"APPLIED_RESIDUAL:{residual}")
    audit.clear(); audit.update({
        "condition":condition,"token_index":int(token_index),"orientation":int(orientation),
        "branch_sign":int(branch_sign),"coefficient_source":"native_pp3_coordinates",
        "native_pp3_a":ci["a"],"native_pp3_b":ci["b"],"condition_correction_l2":ci["l2"],
        "probe_correction_l2":float(torch.linalg.vector_norm(probe)),
        "pp3_post_condition_residual_plus":ci["res_plus"],
        "pp3_post_condition_residual_minus":ci["res_minus"],
        "applied_correction_max_abs_residual":residual,
    })
    return out

def install_hook(mixer17: Any, **kw):
    def hook(_m,_a,out): return apply_hook(out,**kw)
    return mixer17.in_proj.register_forward_hook(hook)

def probe_seed(index: int, pair: str, events):
    require(pair==expected_pairs()[index],f"PAIR:{index}")
    a=holdout.phase1._anchors_for_pair(pair,events)
    return {"family_key":"xg1","source_pair_id":pair,"pair_index":index,
            "target_plus_anchor":int(a["tp"]),"target_minus_anchor":int(a["tm"]),
            "reference_plus_anchor":int(a["rp"]),"reference_minus_anchor":int(a["rm"])}

def input_row(enc,row_index,pair,cell):
    return holdout.phase2._input_row(enc,row_index,pair,cell)

def run_signed(seed,direction,*,condition,orientation,planes,model,runtime_ctx,trace_code,trace_line,
               encoded,row_index,events,budget):
    runtime=holdout.phase1.base.prevalence_eq; parent=runtime.parent; core=runtime.core
    pair=str(seed["source_pair_id"]); cells=holdout.phase1._cells(); anchors=holdout.phase1._anchors_for_pair(pair,events)
    captured={}; audits={}
    for role,sgn in (("tp",1),("tm",-1)):
        audit={}; target=anchors[role]+core.TARGET_OFFSET
        h=install_hook(runtime_ctx["mixer17"],token_index=target,strong_mask=runtime_ctx["strong_mask"],
                       condition=condition,planes=planes,direction=direction,orientation=orientation,
                       branch_sign=sgn,audit=audit)
        try:
            captured[role]=parent.capture_branch(model,runtime_ctx,trace_code=trace_code,trace_line=trace_line,
                input_ids=input_row(encoded,row_index,pair,cells[role]),anchor=anchors[role],
                budget=budget,capture_states=True)
        finally: h.remove()
        require(bool(audit) and captured[role]["intervention_audit"] is None,"HOOK_AUDIT")
        audits[role]=dict(audit)
    pp=float(parent.path_efficiency(captured["tp"])); pm=float(parent.path_efficiency(captured["tm"]))
    return {"condition":condition,"orientation":orientation,"F":pp-pm,"plus_path_efficiency":pp,
            "minus_path_efficiency":pm,"branch_audits":audits,"model_forward_count":F_SIGNED}

def run_direction(seed,direction,*,condition,family,index,**kw):
    pos=run_signed(seed,direction,condition=condition,orientation=1,**kw)
    neg=run_signed(seed,direction,condition=condition,orientation=-1,**kw)
    fp,fm=float(pos["F"]),float(neg["F"]); j=(fp-fm)/(2*EPS)
    return {"direction_key":f"{family}_{index}","basis_family":family,"basis_index":index,
            "F_plus":fp,"F_minus":fm,"J":j,"J_squared":j*j,
            "positive_probe":pos,"negative_probe":neg,"model_forward_count":F_DIR}

def run_condition(seed,*,condition,bases,**kw):
    probes=[]
    for fam in ("xg2","xg4"):
        for i in range(K):
            probes.append(run_direction(seed,bases[fam][:,i],condition=condition,family=fam,index=i,**kw))
    require([p["direction_key"] for p in probes]==list(DIRECTIONS),"DIRECTION_ORDER")
    e2=sum(float(p["J_squared"]) for p in probes[:K])/K
    e4=sum(float(p["J_squared"]) for p in probes[K:])/K
    return {"condition":condition,"direction_order":list(DIRECTIONS),"direction_probes":probes,
            "E_XG2":e2,"E_XG4":e4,"Q":e2-e4,"scientific_model_forward_count":F_COND}

def endpoint(q0: float,q3: float,q5: float) -> dict[str,float]:
    a3=q0-q3; a5=q0-q5; d=a3-a5; require(d==q5-q3,"D_NEC_INTERNAL")
    return {"Q0":q0,"Q3":q3,"Q5":q5,"A3":a3,"A5":a5,"D_NEC":d}

def validate_endpoint(row: Mapping[str,Any]) -> None:
    exp=endpoint(float(row["Q0"]),float(row["Q3"]),float(row["Q5"]))
    for k,v in exp.items(): require(float(row[k])==v,f"ENDPOINT:{k}")

def iter_audits(c):
    for d in c["direction_probes"]:
        for pn in ("positive_probe","negative_probe"):
            for role in ("tp","tm"):
                yield d["direction_key"],pn,role,d[pn]["branch_audits"][role]

def validate_matching(item):
    runtime=holdout.phase1.base.prevalence_eq
    by={c["condition"]:c for c in item["conditions"]}
    a=list(iter_audits(by["pp3_neutralized"])); b=list(iter_audits(by["pp5_coefficient_control"]))
    require(len(a)==len(b)==40,"MATCH_COUNT")
    for x,y in zip(a,b,strict=True):
        require(x[:3]==y[:3],"MATCH_KEY"); ax,ay=x[3],y[3]
        for k in ("native_pp3_a","native_pp3_b","condition_correction_l2"):
            require(abs(float(ax[k])-float(ay[k]))<=runtime.transport_runtime.RUNTIME_CAST_TOL,f"MATCH:{k}")

def run_pair(seed,*,bases,**kw):
    cs=[run_condition(seed,condition=c,bases=bases,**kw) for c in CONDITIONS]
    require([c["condition"] for c in cs]==list(CONDITIONS),"CONDITION_ORDER")
    by={c["condition"]:c for c in cs}; ep=endpoint(float(by["native"]["Q"]),float(by["pp3_neutralized"]["Q"]),
                                                  float(by["pp5_coefficient_control"]["Q"]))
    item={**seed,"schema_version":ITEM_SCHEMA,"implementation_authority_commit":AUTHORITY_COMMIT,
          "static_preparation_freeze_commit":STATIC_COMMIT,"epsilon":EPS,"condition_order":list(CONDITIONS),
          "direction_order":list(DIRECTIONS),"conditions":cs,**ep,
          "baseline_model_forward_count_this_run":0,"scientific_model_forward_count_this_run":F_PAIR}
    validate_matching(item); validate_endpoint(item); return item

def validate_item(item: Mapping[str,Any], expected_pair: str, index: int) -> None:
    require(item["schema_version"]==ITEM_SCHEMA and item["source_pair_id"]==expected_pair and item["pair_index"]==index,
            f"ITEM_ID:{index}")
    require(item.get("family_key")=="xg1",f"ITEM_FAMILY:{index}")
    require(item.get("implementation_authority_commit")==AUTHORITY_COMMIT,f"ITEM_AUTHORITY:{index}")
    require(item.get("static_preparation_freeze_commit")==STATIC_COMMIT,f"ITEM_STATIC_FREEZE:{index}")
    require(float(item["epsilon"])==EPS,f"ITEM_EPSILON:{index}")
    require(item["condition_order"]==list(CONDITIONS) and item["direction_order"]==list(DIRECTIONS),f"ITEM_ORDER:{index}")
    require(item["scientific_model_forward_count_this_run"]==F_PAIR and item["baseline_model_forward_count_this_run"]==0,
            f"ITEM_BUDGET:{index}")
    cs=item["conditions"]; require([c["condition"] for c in cs]==list(CONDITIONS),f"COND_ORDER:{index}")
    for c in cs:
        require(c["direction_order"]==list(DIRECTIONS) and c["scientific_model_forward_count"]==F_COND,"COND_META")
        ps=c["direction_probes"]; require(len(ps)==10,"PROBE_COUNT")
        for key,p in zip(DIRECTIONS,ps,strict=True):
            require(p["direction_key"]==key and p["model_forward_count"]==F_DIR,"DIR_META")
            for orient,pn in ((1,"positive_probe"),(-1,"negative_probe")):
                s=p[pn]; require(s["orientation"]==orient and s["model_forward_count"]==F_SIGNED,"SIGNED_META")
                require(float(s["F"])==float(s["plus_path_efficiency"])-float(s["minus_path_efficiency"]),"F_ID")
                for role,sgn in (("tp",1),("tm",-1)):
                    a=s["branch_audits"][role]
                    require(a["condition"]==c["condition"] and a["orientation"]==orient and a["branch_sign"]==sgn,"AUDIT_COORDINATE")
                    require(a["coefficient_source"]=="native_pp3_coordinates","AUDIT_SOURCE")
                    finite=("native_pp3_a","native_pp3_b","condition_correction_l2","probe_correction_l2",
                            "pp3_post_condition_residual_plus","pp3_post_condition_residual_minus",
                            "applied_correction_max_abs_residual")
                    require(all(math.isfinite(float(a[k])) for k in finite),"AUDIT_FINITE")
                    require(abs(float(a["probe_correction_l2"])-EPS)<=TOL,"AUDIT_PROBE_L2")
                    require(float(a["applied_correction_max_abs_residual"])<=holdout.phase1.base.prevalence_eq.transport_runtime.RUNTIME_CAST_TOL,"AUDIT_RESID")
                    if c["condition"]=="native":
                        require(float(a["condition_correction_l2"])==0.0,"NATIVE_CONDITION_L2")
                    if c["condition"]=="pp3_neutralized":
                        require(abs(float(a["pp3_post_condition_residual_plus"]))<=TOL
                                and abs(float(a["pp3_post_condition_residual_minus"]))<=TOL,"NEUTRAL_AUDIT")
            fp,fm=float(p["F_plus"]),float(p["F_minus"]); j=float(p["J"])
            require(j==(fp-fm)/(2*EPS) and float(p["J_squared"])==j*j,"J_ID")
        e2=sum(float(p["J_squared"]) for p in ps[:K])/K; e4=sum(float(p["J_squared"]) for p in ps[K:])/K
        require(float(c["E_XG2"])==e2 and float(c["E_XG4"])==e4 and float(c["Q"])==e2-e4,"Q_ID")
    validate_matching(item); validate_endpoint(item)

def validate_items(items):
    require(len(items)==N,"ITEM_COUNT")
    for i,(p,x) in enumerate(zip(expected_pairs(),items,strict=True)): validate_item(x,p,i)

def canonical(v): return (json.dumps(dict(v),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode()
def jsonl(rows): return b"".join(canonical(r) for r in rows)

def write_outputs(out: Path, items, summary):
    require(not out.exists(),"OUTPUT_COLLISION"); validate_items(items); out.mkdir(parents=True)
    payloads={ITEM_FILE:jsonl(items),SUMMARY_FILE:canonical(summary)}; hs={}
    for n,b in payloads.items(): (out/n).write_bytes(b); hs[n]=sha256_bytes(b)
    m={"schema_version":MANIFEST_SCHEMA,"files":{n:{"sha256":s,"bytes":(out/n).stat().st_size} for n,s in sorted(hs.items())}}
    mb=canonical(m); (out/MANIFEST_FILE).write_bytes(mb); hs[MANIFEST_FILE]=sha256_bytes(mb)
    (out/CHECKSUM_FILE).write_text("".join(f"{s}  {n}\n" for n,s in sorted(hs.items())),encoding="utf-8",newline="\n")

def validate_artifact(out: Path):
    m=json.loads((out/MANIFEST_FILE).read_text(encoding="utf-8-sig")); require(m["schema_version"]==MANIFEST_SCHEMA,"MANIFEST")
    hs={}
    for n in (ITEM_FILE,SUMMARY_FILE):
        p=out/n; s=sha256_file(p); require(s==m["files"][n]["sha256"] and p.stat().st_size==m["files"][n]["bytes"],f"FILE:{n}"); hs[n]=s
    hs[MANIFEST_FILE]=sha256_file(out/MANIFEST_FILE)
    got={}
    for line in (out/CHECKSUM_FILE).read_text(encoding="utf-8-sig").splitlines():
        if line.strip(): s,n=line.split("  ",1); got[n]=s
    require(got=={n:s for n,s in sorted(hs.items())},"CHECKSUMS")
    items=read_jsonl(out/ITEM_FILE); validate_items(items)
    s=json.loads((out/SUMMARY_FILE).read_text(encoding="utf-8-sig"))
    require(s["schema_version"]==SUMMARY_SCHEMA and s["result"]==RESULT_PASS,"SUMMARY")
    require(s["source_pair_count"]==N and s["pair_id_first"]=="xg1_fact_601" and s["pair_id_last"]=="xg1_fact_900","SUMMARY_POP")
    require(bool(s.get("execution_head")),"SUMMARY_EXECUTION_HEAD")
    require(s.get("implementation_authority_commit")==AUTHORITY_COMMIT,"SUMMARY_AUTHORITY")
    require(s.get("static_preparation_freeze_commit")==STATIC_COMMIT,"SUMMARY_STATIC_FREEZE")
    require(float(s["epsilon"])==EPS,"SUMMARY_EPSILON")
    require(s.get("condition_order")==list(CONDITIONS),"SUMMARY_CONDITION_ORDER")
    require(s.get("direction_order")==list(DIRECTIONS),"SUMMARY_DIRECTION_ORDER")
    require(s.get("model_forwards_per_direction")==F_DIR,"SUMMARY_FORWARD_DIRECTION")
    require(s.get("model_forwards_per_condition")==F_COND,"SUMMARY_FORWARD_CONDITION")
    require(s.get("model_forwards_per_pair")==F_PAIR,"SUMMARY_FORWARD_PAIR")
    require(s.get("representative_checkpoint_sha256")==holdout.phase1.base.prevalence_eq.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,"SUMMARY_CHECKPOINT")
    require(s["scientific_model_forward_count_this_run"]==F_TOTAL and s["baseline_model_forward_count_this_run"]==0,"SUMMARY_BUDGET")
    for k in ("primary_inference_executed","multiplicity_correction_executed","training_executed","backward_executed","task_heads_executed","logits_read"):
        require(s[k] is False,f"BOUNDARY:{k}")
    require(s["scientific_conclusion"] is None,"CONCLUSION")
    return {"items":items,"summary":s,"manifest":m}

def run_observation(*,expected_head,model_snapshot,tokenizer_snapshot,checkpoint_path,output_dir):
    authenticate_repo(expected_head); require(not output_dir.exists(),"OUTPUT_COLLISION")
    planes,bases=load_planes(),load_bases()
    runtime=holdout.phase1.base.prevalence_eq; runtime.backend.runtime_gate()
    with runtime.backend.parent_runtime_rebind():
        rows,encoded,event_rows=load_inputs(tokenizer_snapshot); pairs=pair_order(rows)
        parent=runtime.parent; events=parent.event_lookup(event_rows); row_index=parent.build_row_index(rows)
        trace_code,trace_line=runtime.measurement._resolve_and_validate_runtime_binding()
        kernels=runtime.kernel_compat.load_exact_fast_kernels()
        with runtime.kernel_compat.exact_transformers_kernel_loader(kernels) as calls:
            model,checkpoint_sha=parent.load_representative_model_external(model_snapshot=model_snapshot,checkpoint_path=checkpoint_path)
            require(checkpoint_sha==runtime.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,"CHECKPOINT")
            runtime_ctx=runtime.transport_runtime.validate_runtime_components(model)
        cc=Counter(calls); require(set(cc)=={"causal-conv1d","mamba-ssm"} and cc["causal-conv1d"]>0 and cc["causal-conv1d"]==cc["mamba-ssm"],"KERNEL_CONSTRUCTOR")
        runtime.kernel_compat.validate_transformers_kernel_bindings(kernels)
        model.to(torch.device("cuda:0")); model.eval()
        fast=runtime.backend._make_fast_capture(kernels); original=parent.capture_branch
        budget=parent.ForwardBudget(F_TOTAL); items=[]; parent.capture_branch=fast
        try:
            for i,p in enumerate(pairs):
                items.append(run_pair(probe_seed(i,p,events),bases=bases,planes=planes,model=model,runtime_ctx=runtime_ctx,
                    trace_code=trace_code,trace_line=trace_line,encoded=encoded,row_index=row_index,events=events,budget=budget))
            budget.assert_exact(); torch.cuda.synchronize()
        finally: parent.capture_branch=original
    summary={"schema_version":SUMMARY_SCHEMA,"result":RESULT_PASS,"execution_head":expected_head,
        "implementation_authority_commit":AUTHORITY_COMMIT,"static_preparation_freeze_commit":STATIC_COMMIT,
        "source_pair_count":N,"pair_id_first":items[0]["source_pair_id"],"pair_id_last":items[-1]["source_pair_id"],
        "epsilon":EPS,"condition_order":list(CONDITIONS),"direction_order":list(DIRECTIONS),
        "model_forwards_per_direction":F_DIR,"model_forwards_per_condition":F_COND,"model_forwards_per_pair":F_PAIR,
        "scientific_model_forward_count_this_run":F_TOTAL,"baseline_model_forward_count_this_run":0,
        "primary_endpoint_definition":"D_NEC=(Q0-Q3)-(Q0-Q5)=Q5-Q3",
        "primary_inference_executed":False,"multiplicity_correction_executed":False,"training_executed":False,
        "backward_executed":False,"task_heads_executed":False,"logits_read":False,"scientific_conclusion":None,
        "representative_checkpoint_sha256":checkpoint_sha}
    write_outputs(output_dir,items,summary); validate_artifact(output_dir); return summary

def parse_args(argv: Sequence[str]|None=None):
    p=argparse.ArgumentParser(description="Frozen PP3 necessity raw observation; no statistical inference.")
    p.add_argument("--expected-head",required=True); p.add_argument("--model-snapshot",type=Path,required=True)
    p.add_argument("--tokenizer-snapshot",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); return p.parse_args(argv)

def main(argv: Sequence[str]|None=None):
    a=parse_args(argv); s=run_observation(expected_head=a.expected_head,model_snapshot=a.model_snapshot,
        tokenizer_snapshot=a.tokenizer_snapshot,checkpoint_path=a.checkpoint,output_dir=a.output_dir)
    print("RESULT="+s["result"]); print("SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="+str(s["scientific_model_forward_count_this_run"]))
    print("BASELINE_MODEL_FORWARD_COUNT_THIS_RUN=0"); print("PRIMARY_INFERENCE_EXECUTED=False"); print("SCIENTIFIC_CONCLUSION=None")

if __name__=="__main__": main()
