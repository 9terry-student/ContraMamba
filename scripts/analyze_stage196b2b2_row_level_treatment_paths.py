#!/usr/bin/env python3
"""Stage196-B2-B2 artifact-only row-level paired treatment-path probe."""
from __future__ import annotations

import argparse, csv, io, json, math, os, re, statistics, subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

SEEDS=(183,184,185); POSITIVE=(184,185); EPOCHS=tuple(range(1,21)); TAIL=(18,19,20)
MODES=("joint","frame_local_only"); RUNS=tuple(f"seed{s}_{m}" for s in SEEDS for m in MODES)
ROW_COUNT=720; TOL=1e-6; MARGIN_SOURCE="support_logit - not_entitled_logit"
B2B1_COMMIT="85f1de8f9e0393ccdca5da4bc0725d88d8f427c9"
B2A_COMMIT="833752ec25890e76ee3c2dc3f3bc3c3c1b7428a6"
P0_COMMIT="e9aaff24054f1d409119b70df13b94159a34a8e4"
B1_COMMIT="9835cbbf86d83aca0964821669e63f7f6deb1c59"
FRAMEGATE_COMMIT="5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8"
EXPECTED={184:{"recovery":5,"harm":6},185:{"recovery":2,"harm":3}}
LOCAL=("frame","predicate","sufficiency","polarity","entitlement"); DELTAS=LOCAL+("margin",)
SETS=("stage196a_baseline_recurrent","stage196a_intervention_recurrent","stage196a_common_recurrent","stage196a_universal_all_six")
CLASSES=("FRAME_ENTITLEMENT_GAIN","FRAME_ENTITLEMENT_LOSS","POLARITY_OVERRIDE_DESPITE_FRAME_GAIN","ENTITLEMENT_OPPOSES_FRAME","COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY","MULTI_CHANNEL_CONFLICT","NO_STABLE_DIRECTIONAL_PATH")
DECISIONS=("STAGE196B2B2_FRAME_ENTITLEMENT_PATH_DOMINANT","STAGE196B2B2_POLARITY_OVERRIDE_HARM_SUBTYPE","STAGE196B2B2_COMPOSITION_WITHOUT_STABLE_LOCAL_PRECURSOR","STAGE196B2B2_SEED_SPECIFIC_MULTIPATH_EFFECT","STAGE196B2B2_ANALYSIS_INCOMPLETE")
INCOMPLETE=DECISIONS[-1]
NEXT={DECISIONS[0]:"STAGE196B2B3_FRAME_ENTITLEMENT_MONOTONICITY_CONSTRAINT_DESIGN",DECISIONS[1]:"STAGE196B2B3_FRAME_GAIN_WITH_POLARITY_PRESERVATION_DESIGN",DECISIONS[2]:"STAGE196B2B3_INFERENCE_ONLY_COMPOSITION_ACTIVATION_SWAP_PROBE",DECISIONS[3]:"STAGE196B2B3_NO_PROMOTION_INFERENCE_ONLY_COMPONENT_SWAP_PROBE",INCOMPLETE:"STAGE196B2B2_REPAIR_ANALYSIS_INPUTS"}
AUTHORIZED={DECISIONS[0]:"The treatment's beneficial and harmful row-level effects predominantly follow opposite Frame-entitlement directional paths.",DECISIONS[1]:"Frame improvement can coexist with preservation harm when the treatment degrades polarity, producing a distinct harm subtype.",DECISIONS[2]:"Final SUPPORT-vs-NOT_ENTITLED divergence emerges without a stable fixed-boundary divergence in the observed local channels.",DECISIONS[3]:"The observed treatment effect follows seed-specific or multiple row-level paths without a stable cross-seed path.",INCOMPLETE:"No scientific interpretation is authorized because required analysis closure failed."}
PROHIBITED=["formal causal mediation","mathematical necessity or sufficiency","deployable routing","a safe intervention","unfrozen behavior","external/OOD validity","architectural superiority","authorization of a new intervention, loss, router, or training regime"]
B1_FILES=("stage196b2b1_analysis.json","stage196b2b1_report.md","stage196b2b1_group_summary.csv","stage196b2b1_row_profiles.csv","stage196b2b1_signature_summary.csv","stage196b2b1_cross_seed_transfer.csv","stage196b2b1_intervention_type_summary.csv","stage196b2b1_contract.csv")
A_FILES=("stage196b2a_analysis.json","stage196b2a_report.md","stage196b2a_seed_summary.csv","stage196b2a_support_transition_rows.csv","stage196b2a_channel_transition_summary.csv","stage196b2a_recurrent_position_propagation.csv","stage196b2a_harm_rescue_rows.csv","stage196b2a_epoch_propagation.csv","stage196b2a_contract.csv")
OUTPUTS=("stage196b2b2_analysis.json","stage196b2b2_report.md","stage196b2b2_row_path_summary.csv","stage196b2b2_epoch_paired_paths.csv","stage196b2b2_group_path_summary.csv","stage196b2b2_event_order_summary.csv","stage196b2b2_intervention_type_paths.csv","stage196b2b2_contrast_summary.csv","stage196b2b2_contract.csv")
FIELDS=("id","source_row_id","dev_position","gold_label","prediction","intervention_type","frame_probability","predicate_coverage_probability","sufficiency_probability","polarity_support_margin","entitlement_probability","support_probability","not_entitled_probability","support_logit","not_entitled_logit","epoch","training_seed","frame_downstream_gradient_mode")
NATIVE={"frame":"frame_probability","predicate":"predicate_coverage_probability","sufficiency":"sufficiency_probability","polarity":"polarity_support_margin","entitlement":"entitlement_probability"}
CONTRACT_H=["scope","run","gate","required","observed","passed","blocking_reason"]
EPOCH_H=["seed","stable_row_id","id","source_row_id","dev_position","epoch","transition_role","intervention_type"]+[f"in_{x}" for x in SETS]+["gold_label","joint_prediction","intervention_prediction","joint_support","intervention_support","support_status_disagreement"]+sum(([f"joint_{c}",f"intervention_{c}",f"delta_{c}",f"delta_{c}_sign",f"{c}_pass_transition",f"delta_{c}_sign_agrees_with_final_margin_delta_sign"] for c in LOCAL),[])+["joint_support_vs_ne_margin","intervention_support_vs_ne_margin","delta_support_vs_ne_margin","delta_support_vs_ne_margin_sign","composition_pass_transition"]
ROW_H=["seed","stable_row_id","id","source_row_id","dev_position","transition_role","intervention_type"]+[f"in_{x}" for x in SETS]+["path_class","joint_tail3_status","intervention_tail3_status","tail3_prediction_pattern","margin_direction_concordant","prediction_pattern_concordant","intervention_tail3_stable_support","selected_checkpoint_agrees_with_tail3_role"]+sum(([f"tail3_joint_{c}",f"tail3_intervention_{c}",f"tail3_delta_{c}",f"terminal_delta_{c}_sign",f"first_terminal_sign_stable_epoch_{c}"] for c in DELTAS),[])+sum(([f"tail3_{c}_pass_transition_frequencies",f"first_persistent_boundary_divergence_epoch_{c}"] for c in LOCAL+("composition",)),[])
EVENT_H=["seed","stable_row_id","transition_role","path_class"]+[f"first_terminal_sign_stable_epoch_{c}" for c in DELTAS]+[f"first_persistent_boundary_divergence_epoch_{c}" for c in LOCAL+("composition",)]
GROUP_H=["seed","transition_role","row_count","path_class_counts","path_class_rates","terminal_delta_sign_counts","pass_transition_counts","mean_tail3_deltas","ordered_raw_tail3_deltas","median_first_terminal_sign_stable_epoch","median_first_persistent_boundary_divergence_epoch","margin_direction_concordance_rate","prediction_pattern_concordance_rate","intervention_type_counts","recurrent_set_membership_counts"]
TYPE_H=["seed","intervention_type","transition_role","path_class","count","polarity_flip_remains_harm_only","paraphrase_harm_follows_polarity_override","none_and_paraphrase_recovery_share_a_path","path_stable_across_positive_seeds"]
CONTRAST_H=["seed","epoch","row_count"]+[f"mean_delta_{c}" for c in DELTAS]+["support_status_disagreement_count","support_status_disagreement_rate"]

def args()->argparse.Namespace:
 p=argparse.ArgumentParser(description=__doc__)
 for name,typ in (("repo-root",Path),("stage196b2b1-analysis-json",Path),("stage196b2b1-analyzer-git-commit",str),("stage196b2a-analysis-json",Path),("stage196b2p0-run-root",Path),("current-git-commit",str),("output-dir",Path)): p.add_argument("--"+name,required=True,type=typ)
 return p.parse_args()
def jread(p:Path)->Any:
 with p.open(encoding="utf-8") as h:return json.load(h)
def jlread(p:Path)->list[dict[str,Any]]:
 out=[]
 with p.open(encoding="utf-8") as h:
  for n,line in enumerate(h,1):
   if not line.strip():raise ValueError(f"{p}:{n}: blank row")
   x=json.loads(line)
   if type(x) is not dict:raise ValueError(f"{p}:{n}: non-object")
   out.append(x)
 return out
def cread(p:Path)->list[dict[str,str]]:
 with p.open(encoding="utf-8",newline="") as h:return list(csv.DictReader(h))
def cell(x:Any)->Any:
 if type(x) is not str:return x
 try:return json.loads(x)
 except json.JSONDecodeError:return x
def integer(x:Any,n:str)->int:
 x=cell(x)
 if type(x) is not int:raise ValueError(f"{n}: integer required")
 return x
def num(x:Any,n:str,prob:bool=False)->float:
 x=cell(x)
 if type(x) not in (int,float) or not math.isfinite(float(x)):raise ValueError(f"{n}: finite number required")
 x=float(x)
 if prob and not 0<=x<=1:raise ValueError(f"{n}: probability out of range")
 return x
def boolean(x:Any,n:str)->bool:
 x=cell(x)
 if type(x) is not bool:raise ValueError(f"{n}: boolean required")
 return x
def gate(gs:list[dict[str,Any]],scope:str,run:str,name:str,req:Any,obs:Any,ok:bool,why:str)->None:
 gs.append({"scope":scope,"run":run,"gate":name,"required":req,"observed":obs,"passed":ok,"blocking_reason":"" if ok else why})
 if not ok:raise ValueError(why)
def columns(name:str,rows:list[dict[str,str]],req:Sequence[str])->None:
 if not rows or set(req)-set(rows[0]):raise ValueError(f"{name}: absent rows/columns {sorted(set(req)-(set(rows[0]) if rows else set()))}")
def under(p:Path,r:Path)->bool:
 try:p.relative_to(r);return True
 except ValueError:return False
def mean(xs:Sequence[float])->float:return math.fsum(xs)/len(xs)
def rate(n:int,d:int)->float|None:return n/d if d else None
def sign(x:float)->str:return "negative" if x<0 else "positive" if x>0 else "zero"
def passing(c:str,x:float)->bool:return x >= (0.0 if c in ("polarity","composition") else 0.5)
def transition(a:bool,b:bool)->str:return "FAIL_TO_PASS" if not a and b else "PASS_TO_FAIL" if a and not b else "STABLE_PASS" if a else "STABLE_FAIL"
def status(ps:Sequence[str])->str:
 return "STABLE_SUPPORT" if tuple(ps)==("SUPPORT",)*3 else "PERSISTENT_NOT_ENTITLED" if tuple(ps)==("NOT_ENTITLED",)*3 else "PERSISTENT_REFUTE" if tuple(ps)==("REFUTE",)*3 else "UNSTABLE"
def contract_ok(rows:list[dict[str,str]])->bool:return bool(rows) and all(boolean(r.get("passed",""),"passed") and r.get("blocking_reason","")=="" for r in rows)
def exact_dir(d:Path,names:Sequence[str],label:str,gs:list[dict[str,Any]])->None:
 obs=sorted(x.name for x in d.iterdir()); req=sorted(names)
 gate(gs,"source","",label,req,obs,obs==req and all((d/x).is_file() for x in names),f"{label} exact closure failed")
def exact_sidecars(d:Path,names:Sequence[str],label:str,gs:list[dict[str,Any]])->None:
 expected=set(names); pattern=re.compile(r"^stage196b2p0_epoch_channels_[0-9]{3}\.jsonl$")
 matched=[p for p in d.iterdir() if p.is_file() and pattern.fullmatch(p.name)]
 observed={p.name for p in matched}; resolved={p.resolve() for p in matched}
 evidence={"expected_names":sorted(expected),"observed_names":sorted(observed),
           "missing_names":sorted(expected-observed),"unexpected_names":sorted(observed-expected),
           "matched_file_count":len(matched),"unique_resolved_file_count":len(resolved)}
 ok=len(matched)==20 and len(resolved)==20 and observed==expected
 gate(gs,"source","",label,sorted(expected),evidence,ok,f"{label} exact closure failed")
def cvalue(rows:list[dict[str,str]],name:str,col:str)->Any:
 vs=[cell(r.get(col,"")) for r in rows if r.get("gate")==name]
 if not vs or any(x!=vs[0] for x in vs):raise ValueError(f"contract {name}/{col} missing or nonuniform")
 return vs[0]
def b2b1_analyzer_commit(rows:list[dict[str,str]])->str:
 def passed_values(name:str)->list[Any]:
  matches=[r for r in rows if r.get("scope")=="provenance" and r.get("gate")==name]
  if any(not boolean(r.get("passed",""),f"{name} passed") for r in matches):raise ValueError(f"contract {name} is not passed")
  values=[cell(r.get("observed","")) for r in matches]
  if any(v in (None,"") for v in values) or (values and any(v!=values[0] for v in values)):raise ValueError(f"contract {name}/observed null or nonuniform")
  return values
 primary=passed_values("current_analyzer_commit_equals_head")
 legacy=passed_values("analysis_runtime_commit_equals_head")
 if primary and legacy and primary[0]!=legacy[0]:raise ValueError("conflicting B2-B1 analyzer commit gate aliases")
 if not primary and not legacy:raise ValueError("B2-B1 analyzer commit provenance gate absent")
 value=str((primary or legacy)[0])
 if re.fullmatch(r"[0-9a-f]{40}",value) is None:raise ValueError("B2-B1 analyzer commit is not lowercase 40-hex")
 return value
def cmap(rows:list[dict[str,str]],name:str)->dict[str,Any]:
 for col in ("observed","required"):
  for r in rows:
   x=cell(r.get(col,""))
   if r.get("gate")==name and type(x) is dict:return x
 raise ValueError(f"contract {name}: mapping absent")
def normfield(a:dict[str,Any],names:Sequence[str],v:str,label:str,w:list[str])->None:
 vals=[(n,a.get(n)) for n in names if a.get(n) is not None]
 if any(x!=v for _,x in vals):raise ValueError(f"{label} top-level provenance disagreement")
 if not vals:w.append(f"SOURCE_SCHEMA_WARNING: {label} top-level {'/'.join(names)} null/absent; normalized from passed contract")

def normalize(a:dict[str,Any],c:list[dict[str,str]],b1:bool,w:list[str])->dict[str,str]:
 if b1:
  m=cmap(c,"normalized_source_roles"); current=b2b1_analyzer_commit(c)
  out={"stage196b2b1_analyzer_git_commit":current,"stage196b2a_analyzer_git_commit":str(m.get("stage196b2a_analyzer_git_commit")),"stage196b2p0_runtime_git_commit":str(m.get("stage196b2p0_runtime_git_commit")),"stage196b1_runtime_git_commit":str(m.get("stage196b1_runtime_git_commit")),"framegate_implementation_origin_git_commit":str(m.get("framegate_implementation_origin_git_commit")),"support_vs_not_entitled_margin_source":str(m.get("support_vs_not_entitled_margin_source"))}
 else:
  o=cvalue(c,"framegate_implementation_origin_commit_preserved","required")
  if type(o) is not dict:raise ValueError("B2-A origin mapping absent")
  out={"stage196b2a_analyzer_git_commit":str(cvalue(c,"analysis_runtime_commit_equals_head","observed")),"stage196b2p0_runtime_git_commit":str(cvalue(c,"p0_runtime_commit","required")),"stage196b1_runtime_git_commit":str(cvalue(c,"stage196b1_runtime_commit_format","observed")),"framegate_implementation_origin_git_commit":str(o.get("git_commit")),"support_vs_not_entitled_margin_source":MARGIN_SOURCE}
 exp={"stage196b2a_analyzer_git_commit":B2A_COMMIT,"stage196b2p0_runtime_git_commit":P0_COMMIT,"stage196b1_runtime_git_commit":B1_COMMIT,"framegate_implementation_origin_git_commit":FRAMEGATE_COMMIT,"support_vs_not_entitled_margin_source":MARGIN_SOURCE}
 if b1:exp={"stage196b2b1_analyzer_git_commit":B2B1_COMMIT,**exp}
 if out!=exp:raise ValueError(f"{('B2-B1' if b1 else 'B2-A')} normalized provenance mismatch")
 alias={"stage196b2b1_analyzer_git_commit":("analysis_runtime_commit","analysis_runtime_git_commit","current_analyzer_git_commit"),"stage196b2a_analyzer_git_commit":("stage196b2a_analyzer_git_commit",) if b1 else ("analysis_runtime_commit","analysis_runtime_git_commit","current_analyzer_git_commit"),"stage196b2p0_runtime_git_commit":("stage196b2p0_runtime_commit","stage196b2p0_runtime_git_commit"),"stage196b1_runtime_git_commit":("stage196b1_runtime_git_commit",),"framegate_implementation_origin_git_commit":("framegate_implementation_origin_git_commit",),"support_vs_not_entitled_margin_source":("support_vs_ne_margin_source","support_vs_not_entitled_margin_source","normalized_support_vs_not_entitled_margin_source")}
 for k,n in alias.items():
  if k in out:normfield(a,n,out[k],"B2-B1" if b1 else "B2-A",w)
 return out

def sources(ns:argparse.Namespace,gs:list[dict[str,Any]])->dict[str,Any]:
 repo=ns.repo_root.resolve(); b1d=ns.stage196b2b1_analysis_json.resolve().parent; ad=ns.stage196b2a_analysis_json.resolve().parent; rr=ns.stage196b2p0_run_root.resolve(); out=ns.output_dir.resolve()
 gate(gs,"path","","repo_root","existing directory",str(repo),repo.is_dir(),"invalid repo root")
 gate(gs,"path","","source_names",[B1_FILES[0],A_FILES[0]],[ns.stage196b2b1_analysis_json.name,ns.stage196b2a_analysis_json.name],ns.stage196b2b1_analysis_json.name==B1_FILES[0] and ns.stage196b2a_analysis_json.name==A_FILES[0],"analysis filename mismatch")
 gate(gs,"path","","paths_below_repo",True,all(under(x,repo) for x in (b1d,ad,rr,out)),all(under(x,repo) for x in (b1d,ad,rr,out)),"path escapes repo")
 sep=all(not under(out,x) and not under(x,out) for x in (b1d,ad,rr)); gate(gs,"path","","output_separation",True,sep,sep,"output overlaps source")
 empty=not out.exists() or (out.is_dir() and not any(out.iterdir())); gate(gs,"path","","new_empty_output",True,empty,empty,"output not new/empty")
 head=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 fmt=re.fullmatch(r"[0-9a-f]{40}",ns.current_git_commit or "") is not None; gate(gs,"provenance","","current_commit_format","40-hex",ns.current_git_commit,fmt,"invalid current commit")
 gate(gs,"provenance","","current_commit_equals_head",ns.current_git_commit,head,head==ns.current_git_commit,"current commit differs from HEAD")
 gate(gs,"provenance","","b2b1_commit_argument",B2B1_COMMIT,ns.stage196b2b1_analyzer_git_commit,ns.stage196b2b1_analyzer_git_commit==B2B1_COMMIT,"B2-B1 commit argument mismatch")
 w=[]; exact_dir(b1d,B1_FILES,"exact_eight_file_b2b1_closure",gs); b1a=jread(b1d/B1_FILES[0]); b1c=cread(b1d/B1_FILES[-1])
 req={"decision":"STAGE196B2B1_SEED_SPECIFIC_NO_STABLE_BIFURCATION","recommended_next_stage":"STAGE196B2B2_NO_PROMOTION_ROW_LEVEL_CAUSAL_PROBE","blocking_reasons":[]}; obs={k:b1a.get(k) for k in req}; gate(gs,"source","","b2b1_decision",req,obs,obs==req,"B2-B1 decision closure mismatch")
 gate(gs,"source","","b2b1_23_passed_gates",23,len(b1c),len(b1c)==23 and contract_ok(b1c),"B2-B1 contract is not exactly 23 passed gates")
 n1=normalize(b1a,b1c,True,w)
 gate(gs,"provenance","","b2b1_contract_commit_equals_cli",ns.stage196b2b1_analyzer_git_commit,n1["stage196b2b1_analyzer_git_commit"],n1["stage196b2b1_analyzer_git_commit"]==ns.stage196b2b1_analyzer_git_commit,"B2-B1 contract analyzer commit differs from CLI argument")
 profiles=cread(b1d/"stage196b2b1_row_profiles.csv")
 columns("profiles",profiles,("id","source_row_id","stable_row_id","dev_position","seed","transition_role","intervention_type","trace_joint_tail3_status","trace_intervention_tail3_status","trace_selected_joint_final","trace_selected_intervention_final",*[f"in_{x}" for x in SETS]))
 counts={s:{"recovery":0,"harm":0} for s in POSITIVE}; seen=set()
 for r in profiles:
  s=integer(r["seed"],"seed"); role=r["transition_role"]; pos=integer(r["dev_position"],"position"); ident=r["stable_row_id"]
  if s not in POSITIVE or role not in counts[s] or r["id"]!=ident or r["source_row_id"]!=ident or (s,ident,pos) in seen:raise ValueError("primary profile identity/role mismatch")
  seen.add((s,ident,pos));counts[s][role]+=1
 gate(gs,"population","","exact_16_primary",EXPECTED,counts,len(profiles)==16 and counts==EXPECTED,"primary population mismatch")
 roles_ok=b1a.get("positive_primary_seeds")==list(POSITIVE) and b1a.get("contrast_seeds")==[183]
 gate(gs,"population","","b2b1_seed_roles",{"positive":[184,185],"contrast":[183]},{"positive":b1a.get("positive_primary_seeds"),"contrast":b1a.get("contrast_seeds")},roles_ok,"B2-B1 seed-role closure mismatch")
 ra=b1a.get("rule_a_evaluation",{}); rb=b1a.get("rule_b_evaluation",{}); de=b1a.get("decision_rule_evaluation",{}); cv=b1a.get("composition_augmented_signature_overlap",{})
 ok=ra.get("passes_every_positive_seed") is False and rb.get("passes_every_positive_seed") is False and de.get("local_exact_cross_seed_transfer_stable") is False and cv.get("view_label")=="composition_augmented_non_authorizing"
 gate(gs,"source","","b2b1_scientific_closure",True,ok,ok,"B2-B1 rule/transfer/composition closure mismatch")
 exact_dir(ad,A_FILES,"exact_nine_file_b2a_closure",gs); aa=jread(ad/A_FILES[0]); ac=cread(ad/A_FILES[-1]); ok=aa.get("decision")=="STAGE196B2A_SEED_SPECIFIC_MIXED_PROPAGATION" and aa.get("blocking_reasons")==[]
 gate(gs,"source","","b2a_decision",True,ok,ok,"B2-A decision closure mismatch"); gate(gs,"source","","b2a_contract",True,contract_ok(ac),contract_ok(ac),"B2-A contract failure"); n2=normalize(aa,ac,False,w)
 rc={"seed_summary":len(cread(ad/A_FILES[2])),"epoch_propagation":len(cread(ad/A_FILES[7])),"support_transition":len(cread(ad/A_FILES[3]))}; erc={"seed_summary":3,"epoch_propagation":60,"support_transition":4320}; gate(gs,"source","","b2a_rows",erc,rc,rc==erc,"B2-A row counts mismatch")
 if {k:n1[k] for k in n2}!=n2:raise ValueError("B2-A/B2-B1 normalized provenance disagreement")
 return {"repo":repo,"run_root":rr,"output":out,"profiles":profiles,"b1":b1a,"warnings":w,"normalized":n1}

def load_runs(ctx:dict[str,Any],gs:list[dict[str,Any]])->dict[str,dict[int,dict[str,dict[str,Any]]]]:
 root=ctx["run_root"]; obs=sorted(x.name for x in root.iterdir()); gate(gs,"p0","","exact_six_runs",sorted(RUNS),obs,obs==sorted(RUNS) and all((root/x).is_dir() for x in RUNS),"P0 run closure mismatch")
 runs={}
 for run in RUNS:
  s=int(run[4:7]); mode=run[8:]; d=root/run; names=[f"stage196b2p0_epoch_channels_{e:03d}.jsonl" for e in EPOCHS]; exact_sidecars(d,names,f"{run}_exact_20_sidecars",gs); eps={}; ref=None
  gate(gs,"provenance",run,"p0_runtime_commit",P0_COMMIT,ctx["normalized"]["stage196b2p0_runtime_git_commit"],ctx["normalized"]["stage196b2p0_runtime_git_commit"]==P0_COMMIT,f"{run}: P0 runtime provenance mismatch")
  for e,name in zip(EPOCHS,names):
   raw=jlread(d/name); gate(gs,"p0",run,f"epoch_{e:03d}_720_rows",ROW_COUNT,len(raw),len(raw)==ROW_COUNT,f"{run} epoch row count mismatch")
   by={}; positions=set()
   for x in raw:
    if tuple(x.keys())!=FIELDS and set(x)!=set(FIELDS):raise ValueError(f"{run}:{e}: exact 18-field schema mismatch")
    ident=str(x["id"]); source=str(x["source_row_id"]); pos=integer(x["dev_position"],"dev_position")
    if ident!=source or ident in by or pos in positions:raise ValueError(f"{run}:{e}: duplicate/misaligned identity")
    if x["epoch"]!=e or x["training_seed"]!=s or x["frame_downstream_gradient_mode"]!=mode:raise ValueError(f"{run}:{e}: provenance mismatch")
    if x["gold_label"] not in ("REFUTE","NOT_ENTITLED","SUPPORT") or x["prediction"] not in ("REFUTE","NOT_ENTITLED","SUPPORT"):raise ValueError(f"{run}:{e}: label mismatch")
    z={"id":ident,"source_row_id":source,"position":pos,"gold":x["gold_label"],"prediction":x["prediction"],"intervention_type":x["intervention_type"]}
    for c,k in NATIVE.items():z[c]=num(x[k],k,c!="polarity")
    z["support_logit"]=num(x["support_logit"],"support_logit");z["not_entitled_logit"]=num(x["not_entitled_logit"],"not_entitled_logit");z["margin"]=z["support_logit"]-z["not_entitled_logit"]
    num(x["support_probability"],"support_probability",True);num(x["not_entitled_probability"],"not_entitled_probability",True)
    by[ident]=z;positions.add(pos)
   mapping={(x["id"],x["source_row_id"],x["position"]) for x in by.values()}
   if ref is None:ref=mapping
   elif mapping!=ref:raise ValueError(f"{run}: population drift across epochs")
   eps[e]=by
  runs[run]=eps
 for s in SEEDS:
  a=runs[f"seed{s}_joint"];b=runs[f"seed{s}_frame_local_only"]
  for e in EPOCHS:
   if set(a[e])!=set(b[e]):raise ValueError(f"seed{s}:{e}: arm population mismatch")
   for ident in a[e]:
    x,y=a[e][ident],b[e][ident]
    if (x["source_row_id"],x["position"],x["gold"],x["intervention_type"])!=(y["source_row_id"],y["position"],y["gold"],y["intervention_type"]):raise ValueError(f"seed{s}:{e}:{ident}: arm metadata mismatch")
 gate(gs,"alignment","","six_run_epoch_population_alignment","6 x 20 x 720", "6 x 20 x 720",True,"")
 return runs

def first_sign(series:list[float],terminal:str)->int|None:
 if terminal=="zero":return None
 for i,x in enumerate(series):
  if all(sign(y)==terminal for y in series[i:]):return i+1
 return None
def first_div(series:list[tuple[bool,bool]])->int|None:
 for i,(a,b) in enumerate(series):
  if a!=b and all(x!=y for x,y in series[i:]):return i+1
 return None
def classify(t:dict[str,float],ss:dict[str,int|None],bd:dict[str,int|None],istatus:str)->str:
 m=ss["margin"]
 before=lambda c:m is None or ss[c] is None or ss[c]<=m
 if t["frame"]>0 and t["entitlement"]>0 and t["margin"]>0 and t["polarity"]>=0 and before("frame") and before("entitlement"):return CLASSES[0]
 if t["frame"]<0 and t["entitlement"]<0 and t["margin"]<0 and t["polarity"]<=0 and before("frame") and before("entitlement"):return CLASSES[1]
 if t["frame"]>0 and t["polarity"]<0 and (t["margin"]<=0 or istatus!="STABLE_SUPPORT") and (m is None or ss["polarity"] is None or ss["polarity"]<=m):return CLASSES[2]
 if sign(t["frame"])!=sign(t["entitlement"]) and "zero" not in (sign(t["frame"]),sign(t["entitlement"])):return CLASSES[3]
 comp=bd["composition"]
 if comp is not None and all(bd[c] is None or bd[c]>comp for c in LOCAL):return CLASSES[4]
 signs={sign(t[c]) for c in LOCAL if sign(t[c])!="zero"}
 if "positive" in signs and "negative" in signs:return CLASSES[5]
 return CLASSES[6]

def primary(ctx:dict[str,Any],runs:dict[str,Any],gs:list[dict[str,Any]])->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
 erows=[]; rrows=[]; events=[]
 for p in ctx["profiles"]:
  s=integer(p["seed"],"seed");ident=p["stable_row_id"];pos=integer(p["dev_position"],"position"); j=runs[f"seed{s}_joint"];f=runs[f"seed{s}_frame_local_only"]
  series={c:[] for c in DELTAS}; bounds={c:[] for c in LOCAL+("composition",)}; trans={c:[] for c in LOCAL+("composition",)}; jp=[];fp=[]
  for e in EPOCHS:
   if ident not in j[e] or ident not in f[e]:raise ValueError(f"primary row missing: {s}/{ident}/{e}")
   a,b=j[e][ident],f[e][ident]
   if a["id"]!=p["id"] or a["source_row_id"]!=p["source_row_id"] or a["position"]!=pos or b["position"]!=pos or a["gold"]!="SUPPORT" or b["gold"]!="SUPPORT":raise ValueError(f"primary alignment mismatch: {s}/{ident}/{e}")
   if a["intervention_type"]!=p["intervention_type"] or b["intervention_type"]!=p["intervention_type"]:raise ValueError("intervention type drift")
   d={c:b[c]-a[c] for c in DELTAS}; finalsign=sign(d["margin"]); row={"seed":s,"stable_row_id":ident,"id":a["id"],"source_row_id":a["source_row_id"],"dev_position":pos,"epoch":e,"transition_role":p["transition_role"],"intervention_type":p["intervention_type"],**{f"in_{x}":boolean(p[f"in_{x}"],x) for x in SETS},"gold_label":"SUPPORT","joint_prediction":a["prediction"],"intervention_prediction":b["prediction"],"joint_support":a["prediction"]=="SUPPORT","intervention_support":b["prediction"]=="SUPPORT","support_status_disagreement":(a["prediction"]=="SUPPORT")!=(b["prediction"]=="SUPPORT")}
   for c in LOCAL:
    pa,pb=passing(c,a[c]),passing(c,b[c]);tr=transition(pa,pb);series[c].append(d[c]);bounds[c].append((pa,pb));trans[c].append(tr);row.update({f"joint_{c}":a[c],f"intervention_{c}":b[c],f"delta_{c}":d[c],f"delta_{c}_sign":sign(d[c]),f"{c}_pass_transition":tr,f"delta_{c}_sign_agrees_with_final_margin_delta_sign":sign(d[c])==finalsign})
   ca,cb=passing("composition",a["margin"]),passing("composition",b["margin"]);ctr=transition(ca,cb);series["margin"].append(d["margin"]);bounds["composition"].append((ca,cb));trans["composition"].append(ctr);row.update({"joint_support_vs_ne_margin":a["margin"],"intervention_support_vs_ne_margin":b["margin"],"delta_support_vs_ne_margin":d["margin"],"delta_support_vs_ne_margin_sign":finalsign,"composition_pass_transition":ctr});erows.append(row);jp.append(a["prediction"]);fp.append(b["prediction"])
  t={c:mean([series[c][e-1] for e in TAIL]) for c in DELTAS}; ts={c:sign(t[c]) for c in DELTAS}; ss={c:first_sign(series[c],ts[c]) for c in DELTAS}; bd={c:first_div(bounds[c]) for c in LOCAL+("composition",)}; js=status([jp[e-1] for e in TAIL]);fs=status([fp[e-1] for e in TAIL])
  if js!=p["trace_joint_tail3_status"] or fs!=p["trace_intervention_tail3_status"]:raise ValueError(f"B2-B1 tail-three role disagreement: {s}/{ident}")
  pc=classify(t,ss,bd,fs);role=p["transition_role"]
  md=t["margin"]>0 if role=="recovery" else t["margin"]<0; pred=(js=="PERSISTENT_NOT_ENTITLED" and fs=="STABLE_SUPPORT") if role=="recovery" else (js=="STABLE_SUPPORT" and fs!="STABLE_SUPPORT"); selected=(p["trace_selected_joint_final"]!="SUPPORT" and p["trace_selected_intervention_final"]=="SUPPORT") if role=="recovery" else (p["trace_selected_joint_final"]=="SUPPORT" and p["trace_selected_intervention_final"]!="SUPPORT")
  rr={"seed":s,"stable_row_id":ident,"id":p["id"],"source_row_id":p["source_row_id"],"dev_position":pos,"transition_role":role,"intervention_type":p["intervention_type"],**{f"in_{x}":boolean(p[f"in_{x}"],x) for x in SETS},"path_class":pc,"joint_tail3_status":js,"intervention_tail3_status":fs,"tail3_prediction_pattern":{"joint":[jp[e-1] for e in TAIL],"intervention":[fp[e-1] for e in TAIL]},"margin_direction_concordant":md,"prediction_pattern_concordant":pred,"intervention_tail3_stable_support":fs=="STABLE_SUPPORT","selected_checkpoint_agrees_with_tail3_role":selected==pred}
  for c in DELTAS:rr.update({f"tail3_joint_{c}":mean([j[e][ident][c] for e in TAIL]),f"tail3_intervention_{c}":mean([f[e][ident][c] for e in TAIL]),f"tail3_delta_{c}":t[c],f"terminal_delta_{c}_sign":ts[c],f"first_terminal_sign_stable_epoch_{c}":ss[c]})
  for c in LOCAL+("composition",):rr.update({f"tail3_{c}_pass_transition_frequencies":{x:sum(trans[c][e-1]==x for e in TAIL)/3 for x in ("FAIL_TO_PASS","PASS_TO_FAIL","STABLE_FAIL","STABLE_PASS")},f"first_persistent_boundary_divergence_epoch_{c}":bd[c]})
  rrows.append(rr);events.append({"seed":s,"stable_row_id":ident,"transition_role":role,"path_class":pc,**{f"first_terminal_sign_stable_epoch_{c}":ss[c] for c in DELTAS},**{f"first_persistent_boundary_divergence_epoch_{c}":bd[c] for c in LOCAL+("composition",)}})
 gate(gs,"output","","paired_primary_cardinality","16 x 20",len(erows),len(rrows)==16 and len(erows)==320,"primary paired cardinality mismatch")
 return rrows,erows,events

def groups(rows:list[dict[str,Any]])->list[dict[str,Any]]:
 out=[]
 for s in POSITIVE:
  for role in ("recovery","harm"):
   g=[r for r in rows if r["seed"]==s and r["transition_role"]==role]; n=len(g); pc=Counter(r["path_class"] for r in g)
   out.append({"seed":s,"transition_role":role,"row_count":n,"path_class_counts":dict(pc),"path_class_rates":{k:v/n for k,v in pc.items()},"terminal_delta_sign_counts":{c:dict(Counter(r[f"terminal_delta_{c}_sign"] for r in g)) for c in DELTAS},"pass_transition_counts":{c:dict(Counter(x for r in g for x,v in r[f"tail3_{c}_pass_transition_frequencies"].items() for _ in range(round(v*3)))) for c in LOCAL+("composition",)},"mean_tail3_deltas":{c:mean([r[f"tail3_delta_{c}"] for r in g]) for c in DELTAS},"ordered_raw_tail3_deltas":{c:sorted(r[f"tail3_delta_{c}"] for r in g) for c in DELTAS},"median_first_terminal_sign_stable_epoch":{c:(float(statistics.median(v)) if (v:=[r[f"first_terminal_sign_stable_epoch_{c}"] for r in g if r[f"first_terminal_sign_stable_epoch_{c}"] is not None]) else None) for c in DELTAS},"median_first_persistent_boundary_divergence_epoch":{c:(float(statistics.median(v)) if (v:=[r[f"first_persistent_boundary_divergence_epoch_{c}"] for r in g if r[f"first_persistent_boundary_divergence_epoch_{c}"] is not None]) else None) for c in LOCAL+("composition",)},"margin_direction_concordance_rate":rate(sum(r["margin_direction_concordant"] for r in g),n),"prediction_pattern_concordance_rate":rate(sum(r["prediction_pattern_concordant"] for r in g),n),"intervention_type_counts":dict(Counter(r["intervention_type"] for r in g)),"recurrent_set_membership_counts":{x:sum(r[f"in_{x}"] for r in g) for x in SETS}})
 return out

def type_audit(rows:list[dict[str,Any]])->tuple[list[dict[str,Any]],dict[str,Any]]:
 polarity=[r for r in rows if r["intervention_type"]=="polarity_flip"]; para_h=[r for r in rows if r["intervention_type"]=="paraphrase" and r["transition_role"]=="harm"]; recovery_paths={r["path_class"] for r in rows if r["transition_role"]=="recovery" and r["intervention_type"] in ("none","paraphrase")}
 flags={"polarity_flip_remains_harm_only":bool(polarity) and all(r["transition_role"]=="harm" for r in polarity),"paraphrase_harm_rows_follow_polarity_override":bool(para_h) and all(r["path_class"]==CLASSES[2] for r in para_h),"none_and_paraphrase_recovery_rows_share_a_path":len(recovery_paths)==1 and bool(recovery_paths)}
 stable={p:all(any(r["seed"]==s and r["path_class"]==p for r in rows) for s in POSITIVE) for p in CLASSES}; flags["paths_stable_across_both_positive_seeds"]=[p for p,v in stable.items() if v]
 out=[]
 for key,n in sorted(Counter((r["seed"],r["intervention_type"],r["transition_role"],r["path_class"]) for r in rows).items()):
  s,t,role,p=key;out.append({"seed":s,"intervention_type":t,"transition_role":role,"path_class":p,"count":n,"polarity_flip_remains_harm_only":flags["polarity_flip_remains_harm_only"],"paraphrase_harm_follows_polarity_override":flags["paraphrase_harm_rows_follow_polarity_override"],"none_and_paraphrase_recovery_share_a_path":flags["none_and_paraphrase_recovery_rows_share_a_path"],"path_stable_across_positive_seeds":stable[p]})
 return out,flags

def contrast(runs:dict[str,Any])->list[dict[str,Any]]:
 out=[];j=runs["seed183_joint"];f=runs["seed183_frame_local_only"]
 for e in EPOCHS:
  ids=sorted(j[e]); ds={c:[f[e][i][c]-j[e][i][c] for i in ids] for c in DELTAS}; disagree=sum((j[e][i]["prediction"]=="SUPPORT")!=(f[e][i]["prediction"]=="SUPPORT") for i in ids);out.append({"seed":183,"epoch":e,"row_count":len(ids),**{f"mean_delta_{c}":mean(ds[c]) for c in DELTAS},"support_status_disagreement_count":disagree,"support_status_disagreement_rate":disagree/len(ids)})
 return out

def decide(rows:list[dict[str,Any]])->tuple[str,dict[str,Any],dict[str,Any],dict[str,Any]]:
 def frac(s:int,role:str,p:str)->float:
  g=[r for r in rows if r["seed"]==s and r["transition_role"]==role];return sum(r["path_class"]==p for r in g)/len(g)
 timing=all(r["first_terminal_sign_stable_epoch_margin"] is None or ((r["first_terminal_sign_stable_epoch_frame"] is None or r["first_terminal_sign_stable_epoch_frame"]<=r["first_terminal_sign_stable_epoch_margin"]) and (r["first_terminal_sign_stable_epoch_entitlement"] is None or r["first_terminal_sign_stable_epoch_entitlement"]<=r["first_terminal_sign_stable_epoch_margin"])) for r in rows if r["path_class"] in CLASSES[:2])
 dom=all(frac(s,"recovery",CLASSES[0])>=.75 and frac(s,"harm",CLASSES[1])>=.75 and max([frac(s,"harm",p) for p in CLASSES if p!=CLASSES[1]],default=0)<=.25 for s in POSITIVE) and timing; d={"passed":dom,"per_seed":{str(s):{"recovery_gain_rate":frac(s,"recovery",CLASSES[0]),"harm_loss_rate":frac(s,"harm",CLASSES[1])} for s in POSITIVE},"timing_passed":timing}
 polrows=[r for r in rows if r["path_class"]==CLASSES[2]]; poltim=all(r["first_terminal_sign_stable_epoch_margin"] is None or (r["first_terminal_sign_stable_epoch_polarity"] is not None and r["first_terminal_sign_stable_epoch_polarity"]<=r["first_terminal_sign_stable_epoch_margin"]) for r in polrows); pol=(not dom and all(frac(s,"recovery",CLASSES[0])>=.75 for s in POSITIVE) and any(sum(r["seed"]==s and r["transition_role"]=="harm" and r["path_class"]==CLASSES[2] for r in rows)>=2 for s in POSITIVE) and not any(r["transition_role"]=="recovery" for r in polrows) and poltim); p={"passed":pol,"polarity_override_harm_count":len([r for r in polrows if r["transition_role"]=="harm"]),"recovery_override_count":len([r for r in polrows if r["transition_role"]=="recovery"]),"timing_passed":poltim}
 comp=not dom and not pol and all(frac(s,role,CLASSES[4])>=.75 for s in POSITIVE for role in ("recovery","harm")) and not any(all(frac(s,role,p)>=.5 for s in POSITIVE for role in ("recovery","harm")) for p in CLASSES[:4]); c={"passed":comp,"per_seed_role_rates":{f"{s}_{role}":frac(s,role,CLASSES[4]) for s in POSITIVE for role in ("recovery","harm")}}
 return (DECISIONS[0] if dom else DECISIONS[1] if pol else DECISIONS[2] if comp else DECISIONS[3]),d,p,c

def report(ns:argparse.Namespace,gs:list[dict[str,Any]])->tuple[dict[str,Any],dict[str,list[dict[str,Any]]]]:
 ctx=sources(ns,gs);runs=load_runs(ctx,gs);rows,epochs,events=primary(ctx,runs,gs);gr=groups(rows);types,ta=type_audit(rows);con=contrast(runs);decision,d,p,c=decide(rows);gate(gs,"decision","","fixed_decision_set",list(DECISIONS),decision,decision in DECISIONS[:-1],"decision escaped fixed set")
 counts={str(s):{role:sum(r["seed"]==s and r["transition_role"]==role for r in rows) for role in ("recovery","harm")} for s in POSITIVE}; pcs={str(s):{role:dict(Counter(r["path_class"] for r in rows if r["seed"]==s and r["transition_role"]==role)) for role in ("recovery","harm")} for s in POSITIVE}
 n=ctx["normalized"];a={"stage":"Stage196-B2-B2","decision":decision,"recommended_next_stage":NEXT[decision],"blocking_reasons":[],"current_analyzer_git_commit":ns.current_git_commit,**n,"normalized_historical_provenance_roles":n,"normalized_support_vs_not_entitled_margin_source":MARGIN_SOURCE,"source_schema_warnings":ctx["warnings"],"positive_seeds":list(POSITIVE),"contrast_seeds":[183],"primary_population_counts":counts,"primary_population_total":16,"path_class_counts_by_seed_and_role":pcs,"frame_entitlement_decision_rule_evaluation":d,"polarity_override_decision_rule_evaluation":p,"composition_without_local_precursor_evaluation":c,"intervention_type_path_summary":ta,"event_order_summary":{"row_count":len(events),"tail_epochs":list(TAIL),"by_seed_and_role":{f"{g['seed']}_{g['transition_role']}":{"median_first_terminal_sign_stable_epoch":g["median_first_terminal_sign_stable_epoch"],"median_first_persistent_boundary_divergence_epoch":g["median_first_persistent_boundary_divergence_epoch"]} for g in gr}},"authorized_interpretation":AUTHORIZED[decision],"prohibited_interpretations":PROHIBITED,"output_file_count":9,"training_performed":False,"checkpoint_loaded":False,"model_loaded":False,"external_evaluation_performed":False,"artifact_only_analysis":True,"threshold_search_performed":False,"classifier_fitted":False}
 return a,{"row":rows,"epoch":epochs,"group":gr,"event":events,"type":types,"contrast":con,"contract":gs}

def incomplete(ns:argparse.Namespace,e:BaseException)->dict[str,Any]:
 return {"stage":"Stage196-B2-B2","decision":INCOMPLETE,"recommended_next_stage":NEXT[INCOMPLETE],"blocking_reasons":[f"{type(e).__name__}: {e}"],"current_analyzer_git_commit":ns.current_git_commit,"stage196b2b1_analyzer_git_commit":ns.stage196b2b1_analyzer_git_commit,"stage196b2a_analyzer_git_commit":None,"stage196b2p0_runtime_git_commit":None,"stage196b1_runtime_git_commit":None,"framegate_implementation_origin_git_commit":None,"normalized_historical_provenance_roles":{},"normalized_support_vs_not_entitled_margin_source":None,"source_schema_warnings":[],"positive_seeds":[],"contrast_seeds":[],"primary_population_counts":{},"primary_population_total":0,"path_class_counts_by_seed_and_role":{},"frame_entitlement_decision_rule_evaluation":{"evaluated":False},"polarity_override_decision_rule_evaluation":{"evaluated":False},"composition_without_local_precursor_evaluation":{"evaluated":False},"intervention_type_path_summary":{},"event_order_summary":{},"authorized_interpretation":AUTHORIZED[INCOMPLETE],"prohibited_interpretations":PROHIBITED,"output_file_count":9,"training_performed":False,"checkpoint_loaded":False,"model_loaded":False,"external_evaluation_performed":False,"artifact_only_analysis":True,"threshold_search_performed":False,"classifier_fitted":False}

SECTIONS=("Executive decision","Authorized interpretation","Stage196-B2-B1 source closure","Stage196-B2-A and P0 closure","Provenance normalization","Primary rows and seed roles","Paired trajectory construction","Fixed thresholds and event definitions","Tail-three directional effects","First terminal-sign-stable epochs","Persistent boundary-divergence ordering","Recovery path analysis","Preservation-harm path analysis","Frame-entitlement path rule","Polarity-override harm rule","Composition-without-local-precursor rule","Intervention-type path audit","Seed183 contrast","Decision-rule evaluation","Remaining uncertainty","Prohibited claims","Recommended next stage")
def markdown(a:dict[str,Any])->str:
 bodies=[f"`{a['decision']}`",a["authorized_interpretation"],"Exact eight-file closure, completed decision, empty blockers, 23 passed gates, and the exact 16 rows were required.","Exact nine-file B2-A closure and exact six-run P0/20-sidecar/720-row closure were required.",json.dumps({"roles":a["normalized_historical_provenance_roles"],"margin":a["normalized_support_vs_not_entitled_margin_source"],"warnings":a["source_schema_warnings"]},sort_keys=True),json.dumps({"positive":a["positive_seeds"],"contrast":a["contrast_seeds"],"counts":a["primary_population_counts"]},sort_keys=True),"Each primary identity is aligned between joint and frame_local_only at every epoch; all deltas are intervention minus joint.","Probability boundaries are 0.5; polarity and composition boundaries are zero. Event times use exact signs and persistent fixed-boundary disagreement.",json.dumps(a["path_class_counts_by_seed_and_role"],sort_keys=True),json.dumps(a["event_order_summary"],sort_keys=True),"Individual local and composition boundary event times remain separate.","Recovery requires positive tail-three margin movement for margin-direction concordance; discordant rows remain included.","Harm requires negative tail-three margin movement for margin-direction concordance, while prediction instability is reported separately.",json.dumps(a["frame_entitlement_decision_rule_evaluation"],sort_keys=True),json.dumps(a["polarity_override_decision_rule_evaluation"],sort_keys=True),json.dumps(a["composition_without_local_precursor_evaluation"],sort_keys=True),json.dumps(a["intervention_type_path_summary"],sort_keys=True),"Seed183 is contrast-only and is excluded from every primary denominator and decision rule.",json.dumps({"frame_entitlement":a["frame_entitlement_decision_rule_evaluation"],"polarity_override":a["polarity_override_decision_rule_evaluation"],"composition":a["composition_without_local_precursor_evaluation"]},sort_keys=True),"This artifact-only frozen-Mamba probe is descriptive; event ordering does not establish formal mediation.","\n".join(f"- {x}" for x in a["prohibited_interpretations"]),f"`{a['recommended_next_stage']}`"]
 return "# Stage196-B2-B2 Row-Level Paired Treatment-Path Probe\n\n"+"\n\n".join(f"## {h}\n\n{b}" for h,b in zip(SECTIONS,bodies))+"\n"
def cv(x:Any)->Any:
 return json.dumps(x,sort_keys=True,separators=(",",":")) if isinstance(x,(dict,list,tuple)) else "true" if x is True else "false" if x is False else x
def render_csv(h:list[str],rows:Iterable[dict[str,Any]])->str:
 s=io.StringIO(newline="");w=csv.DictWriter(s,fieldnames=h,extrasaction="raise",lineterminator="\n");w.writeheader()
 for r in rows:
  if set(r)!=set(h):raise ValueError(f"generated CSV schema mismatch: {sorted(set(r)^set(h))}")
  w.writerow({k:cv(r[k]) for k in h})
 return s.getvalue()
def render(a:dict[str,Any],t:dict[str,list[dict[str,Any]]])->dict[str,str]:
 return {OUTPUTS[0]:json.dumps(a,indent=2,sort_keys=True)+"\n",OUTPUTS[1]:markdown(a),OUTPUTS[2]:render_csv(ROW_H,t["row"]),OUTPUTS[3]:render_csv(EPOCH_H,t["epoch"]),OUTPUTS[4]:render_csv(GROUP_H,t["group"]),OUTPUTS[5]:render_csv(EVENT_H,t["event"]),OUTPUTS[6]:render_csv(TYPE_H,t["type"]),OUTPUTS[7]:render_csv(CONTRAST_H,t["contrast"]),OUTPUTS[8]:render_csv(CONTRACT_H,t["contract"])}
def write(out:Path,data:dict[str,str])->None:
 if set(data)!=set(OUTPUTS):raise ValueError("internal nine-output closure mismatch")
 out.mkdir(parents=True,exist_ok=False)
 for n in OUTPUTS:
  fd=os.open(out/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
  with os.fdopen(fd,"w",encoding="utf-8",newline="") as h:h.write(data[n])
 if {x.name for x in out.iterdir()}!=set(OUTPUTS):raise RuntimeError("written output closure mismatch")
def main()->int:
 ns=args();gs=[]
 try:a,t=report(ns,gs);data=render(a,t)
 except Exception as e:
  gs.append({"scope":"analysis","run":"","gate":"analysis_completed","required":True,"observed":False,"passed":False,"blocking_reason":f"{type(e).__name__}: {e}"});a=incomplete(ns,e);empty={"row":[],"epoch":[],"group":[],"event":[],"type":[],"contrast":[],"contract":gs};data=render(a,empty)
 write(ns.output_dir.resolve(),data);return 0 if a["decision"]!=INCOMPLETE else 2
if __name__=="__main__":raise SystemExit(main())
