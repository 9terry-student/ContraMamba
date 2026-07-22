#!/usr/bin/env python3
"""Stage196-B2-B6 artifact-only minimal recipient-selector intervention design.

No model, checkpoint, fitted threshold, optimization, training, or promotion is
used.  All sources are explicit and selector policies are exact set objects.
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, math, os, re, shutil, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

STAGE = "Stage196-B2-B6"
P0_COMMIT = "fa16787efa84bb15d832b6d9382fafd77016c4e2"
B5_COMMIT = "632c3be6bd9baf648cc1e43f9e91eb2940b82ff9"
SEEDS, PRIMARY_SEEDS, EPOCHS, TAIL = (183, 184, 185), (184, 185), tuple(range(1, 21)), (18, 19, 20)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{s}_{m}" for s in SEEDS for m in MODES)
FORWARD = "JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR"
PRIMITIVES = ("FRAME", "PREDICATE", "SUFFICIENCY", "POSITIVE_ENERGY", "NEGATIVE_ENERGY")
PRIMITIVE_FIELDS = {"FRAME": ("frame_prob",), "PREDICATE": ("predicate_coverage_prob",),
                    "SUFFICIENCY": ("sufficiency_prob",), "POSITIVE_ENERGY": ("positive_energy",),
                    "NEGATIVE_ENERGY": ("negative_energy",)}
BRANCHES = ("temporal_mismatch", "predicate_mismatch", "temporal_adapter", "temporal_channel")
LABELS, TOL, ZERO_TOL = ("REFUTE", "NOT_ENTITLED", "SUPPORT"), 1e-6, 1e-12
EXPECTED_PRIMARY = {184: {"recovery": 5, "harm": 6}, 185: {"recovery": 2, "harm": 3}}
B5_FILES = ("stage196b2b5_analysis.json", "stage196b2b5_report.md", "stage196b2b5_feature_dictionary.csv",
            "stage196b2b5_row_action_sets.csv", "stage196b2b5_recipient_signature_rows.csv",
            "stage196b2b5_recipient_selector_summary.csv", "stage196b2b5_paired_delta_signature_rows.csv",
            "stage196b2b5_paired_delta_selector_summary.csv", "stage196b2b5_contract.csv")
B4_FILES = ("stage196b2b4_analysis.json", "stage196b2b4_report.md", "stage196b2b4_primitive_coalition_rows.csv",
            "stage196b2b4_primitive_mobius_terms.csv", "stage196b2b4_primitive_tail_summary.csv",
            "stage196b2b4_residual_coalition_rows.csv", "stage196b2b4_residual_mobius_terms.csv",
            "stage196b2b4_localization_summary.csv", "stage196b2b4_contract.csv")
OUTPUTS = ("stage196b2b6_analysis.json", "stage196b2b6_report.md", "stage196b2b6_candidate_feature_subsets.csv",
           "stage196b2b6_signature_action_map.csv", "stage196b2b6_primary_policy_validation.csv",
           "stage196b2b6_clean_dev_signature_audit.csv", "stage196b2b6_clean_dev_application_summary.csv",
           "stage196b2b6_policy_dominance.csv", "stage196b2b6_contract.csv")
TRAJECTORY_FIELDS = {"id", "source_row_id", "dev_position", "gold_label", "prediction", "intervention_type",
 "frame_probability", "predicate_coverage_probability", "sufficiency_probability", "polarity_support_margin",
 "entitlement_probability", "support_probability", "not_entitled_probability", "support_logit",
 "not_entitled_logit", "epoch", "training_seed", "frame_downstream_gradient_mode"}
PROHIBITED_FIELDS = {"seed", "training_seed", "stable_row_id", "id", "source_row_id", "dev_position",
 "transition_role", "path_class", "subtype", "counterfactual outcome", "donor-tail outcome",
 "minimal_coalition", "minimal_coalition_labels", "paired-treatment delta"}
PROHIBITED_CLAIMS = ["formal causal mediation", "external or OOD validity", "unfrozen-Mamba validity",
 "training improvement", "promotion", "deployability from paired delta features",
 "safe deployment from primary-row closure alone", "abstention as selector success",
 "seed183 as a formal independent dataset", "an arbitrary tie-break as a deterministic selector",
 "accuracy improvement as selector authorization", "row identity as a selector feature",
 "path class or subtype as an inference-time feature"]
FORMULAS = {
 "RECIPIENT_PREDICTION_SEQUENCE": (("final_native_prediction",), "exact final_native_prediction at epochs 18,19,20"),
 "FINAL_MARGIN_SIGN_SEQUENCE": (("final_support_logit", "final_not_entitled_logit"), "sign(final_support_logit - final_not_entitled_logit), zero iff abs(delta) <= 1e-12"),
 "HEAD_MARGIN_SIGN_SEQUENCE": (("decision_head_support_logit", "decision_head_not_entitled_logit"), "sign(decision_head_support_logit - decision_head_not_entitled_logit), zero iff abs(delta) <= 1e-12"),
 "HEAD_FINAL_MARGIN_SIGN_CONFLICT_SEQUENCE": (("decision_head_support_logit", "decision_head_not_entitled_logit", "final_support_logit", "final_not_entitled_logit"), "HEAD_SIGN != FINAL_SIGN at each tail epoch"),
 "FRAME_HALFSPACE_SEQUENCE": (("frame_prob",), "halfspace(frame_prob; threshold=0.5, AT_HALF iff abs(x-0.5)<=1e-12)"),
 "PREDICATE_HALFSPACE_SEQUENCE": (("predicate_coverage_prob",), "halfspace(predicate_coverage_prob; threshold=0.5, AT_HALF iff abs(x-0.5)<=1e-12)"),
 "SUFFICIENCY_HALFSPACE_SEQUENCE": (("sufficiency_prob",), "halfspace(sufficiency_prob; threshold=0.5, AT_HALF iff abs(x-0.5)<=1e-12)"),
 "ENTITLEMENT_HALFSPACE_SEQUENCE": (("entitlement_prob_native",), "halfspace(entitlement_prob_native; threshold=0.5, AT_HALF iff abs(x-0.5)<=1e-12)"),
 "ENTITLEMENT_BOTTLENECK_SEQUENCE": (("frame_prob", "predicate_coverage_prob", "sufficiency_prob"), "sorted set of argmin names over frame_prob,predicate_coverage_prob,sufficiency_prob; ties preserved within 1e-12"),
 "POLARITY_ENERGY_ORDER_SEQUENCE": (("positive_energy", "negative_energy"), "compare positive_energy with negative_energy; EQUAL iff abs(delta)<=1e-12"),
 "PREDICATE_MISMATCH_SEQUENCE": (("predicate_mismatch_active",), "raw exported predicate_mismatch_active flag"),
 "TEMPORAL_MISMATCH_SEQUENCE": (("temporal_mismatch_active",), "raw exported temporal_mismatch_active flag"),
 "TEMPORAL_ADAPTER_ACTIVITY_SEQUENCE": (("temporal_adapter_active",), "raw exported temporal_adapter_active flag"),
 "TEMPORAL_CHANNEL_ACTIVITY_SEQUENCE": (("temporal_channel_active",), "raw exported temporal_channel_active flag")}
CANDIDATE_H = ("feature_subset_mask", "feature_subset_size", "feature_subset_members", "signature_count",
 "maximum_rows_per_signature", "mixed_role_signature_count", "cross_seed_signature_count",
 "seed184_to_seed185_full_pass", "seed185_to_seed184_full_pass", "classification", "policy_equivalence_class",
 "pooled_full_pass", "bidirectional_cross_seed_full_pass", "unique_signature_action_count",
 "ambiguous_signature_action_count", "empty_signature_action_count", "primary_policy_passed",
 "seed184_nonprimary_safety_passed", "seed185_nonprimary_safety_passed", "safety_authorized", "nondominated")
SIGNATURE_H = ("feature_subset_mask", "feature_subset_members", "signature", "signature_row_count", "seed_counts",
 "transition_role_counts", "acceptable_action_intersection", "inclusion_minimal_action_set", "action_set_size",
 "classification", "safe_set_valued", "unsafe_action_ambiguity", "primary_row_ids")
PRIMARY_H = ("feature_subset_mask", "seed", "stable_row_id", "transition_role", "signature", "assigned_action_set",
 "evaluated_action", "recipient_tail_status", "donor_tail_status", "counterfactual_tail_status", "objective",
 "objective_passed", "donor_tail_reproduced", "recipient_tail_preserved")
AUDIT_H = ("feature_subset_mask", "seed", "stable_row_id", "id", "source_row_id", "dev_position", "gold_label",
 "recipient_tail_predictions", "signature", "support_class", "assigned_action_set", "abstained")
APP_H = ("feature_subset_mask", "seed", "population", "policy_action_mode", "row_count", "seen_signature_count",
 "unseen_signature_count", "coverage_rate", "joint_accuracy", "selector_accuracy", "accuracy_delta", "joint_macro_f1",
 "selector_macro_f1", "macro_f1_delta", "joint_support_recall", "selector_support_recall", "support_recall_delta",
 "joint_false_not_entitled", "selector_false_not_entitled", "joint_false_entitlement", "selector_false_entitlement",
 "joint_polarity_error", "selector_polarity_error", "prediction_change_count", "correct_to_incorrect_count",
 "incorrect_to_correct_count", "stable_correct_preservation_rate", "safety_passed")
DOM_H = ("candidate_a", "candidate_b", "strict_feature_subset", "same_primary_action_sets",
 "no_lower_clean_dev_coverage", "no_worse_safety", "authorized_feature_inputs", "a_structurally_dominates_b", "details")
CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")

def parse_args() -> argparse.Namespace:
 p=argparse.ArgumentParser(description=__doc__)
 for n,t in (("repo-root",Path),("stage196b2b5-analysis-json",Path),("stage196b2b4-analysis-json",Path),
  ("stage196b2b3p0-run-root",Path),("stage196b2b3p0-runtime-git-commit",str),("current-git-commit",str),("output-dir",Path)):
  p.add_argument(f"--{n}",required=True,type=t)
 return p.parse_args()

def read_json(p:Path)->Any:
 with p.open(encoding="utf-8") as h:return json.load(h)
def read_jsonl(p:Path)->list[dict[str,Any]]:
 out=[]
 with p.open(encoding="utf-8") as h:
  for i,line in enumerate(h,1):
   try:v=json.loads(line)
   except json.JSONDecodeError as e:raise ValueError(f"{p}:{i}: invalid JSONL") from e
   if not isinstance(v,dict):raise ValueError(f"{p}:{i}: object required")
   out.append(v)
 return out
def read_csv(p:Path)->list[dict[str,str]]:
 with p.open(newline="",encoding="utf-8-sig") as h:return list(csv.DictReader(h))
def sha256(p:Path)->str:
 d=hashlib.sha256()
 with p.open("rb") as h:
  for c in iter(lambda:h.read(1048576),b""):d.update(c)
 return d.hexdigest()
def integer(v:Any,n:str)->int:
 if isinstance(v,bool):raise ValueError(f"{n}: integer required")
 x=int(v)
 if isinstance(v,float) and v!=x:raise ValueError(f"{n}: nonintegral")
 return x
def number(v:Any,n:str,prob:bool=False)->float:
 if isinstance(v,bool) or v is None:raise ValueError(f"{n}: finite number required")
 x=float(v)
 if not math.isfinite(x) or (prob and not 0<=x<=1):raise ValueError(f"{n}: invalid number")
 return x
def boolean(v:Any,n:str)->bool:
 if isinstance(v,bool):return v
 if isinstance(v,str) and v.strip().lower() in ("true","false"):return v.strip().lower()=="true"
 raise ValueError(f"{n}: boolean required")
def jcell(v:str,n:str)->Any:
 try:return json.loads(v)
 except (TypeError,json.JSONDecodeError) as e:raise ValueError(f"{n}: structured JSON required") from e
def canon(v:Any)->str:return json.dumps(v,sort_keys=True,separators=(",",":"))
def subset(a:str,b:str)->bool:return len(a)==len(b) and all(x=="0" or y=="1" for x,y in zip(a,b))
def minima(values:Iterable[str])->list[str]:
 xs=sorted(set(values));return [x for x in xs if not any(y!=x and subset(y,x) for y in xs)]
def gate(g:list[dict[str,Any]],scope:str,run:str,name:str,required:Any,observed:Any,passed:bool,reason:str,fatal:bool=True)->None:
 g.append({"scope":scope,"run":run,"gate":name,"required":required,"observed":observed,"passed":bool(passed),"blocking_reason":"" if passed else reason})
 if fatal and not passed:raise ValueError(f"{name}: {reason}")
def exact_dir(p:Path,names:Sequence[str],g:list[dict[str,Any]],label:str)->None:
 observed=sorted(x.name for x in p.iterdir() if x.is_file()) if p.is_dir() else [] ; expected=sorted(names)
 gate(g,"source","",label,expected,{"files":observed,"missing":sorted(set(expected)-set(observed)),"unexpected":sorted(set(observed)-set(expected))},observed==expected,"exact file closure failed")
def closed(rows:list[dict[str,str]])->bool:return bool(rows) and all(boolean(r.get("passed"),"passed") and not r.get("blocking_reason","").strip() for r in rows)
def cols(rows:list[dict[str,Any]],names:Sequence[str],label:str)->None:
 if not rows or not set(names)<=set(rows[0]):raise ValueError(f"{label}: missing columns")
def softplus(x:float)->float:return x+math.log1p(math.exp(-x)) if x>0 else math.log1p(math.exp(x))
def sigmoid(x:float)->float:return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
def argmax(logits:Sequence[float])->str:return LABELS[max(range(3),key=lambda i:logits[i])]
def status(xs:Sequence[str])->str:
 t=tuple(xs)
 return "STABLE_SUPPORT" if t==("SUPPORT",)*3 else "PERSISTENT_NOT_ENTITLED" if t==("NOT_ENTITLED",)*3 else "PERSISTENT_REFUTE" if t==("REFUTE",)*3 else "UNSTABLE"
def sign(x:float)->str:return "ZERO" if abs(x)<=ZERO_TOL else "POSITIVE" if x>0 else "NEGATIVE"
def half(x:float)->str:return "AT_HALF" if abs(x-.5)<=ZERO_TOL else "ABOVE_HALF" if x>.5 else "BELOW_HALF"

def branch(row:dict[str,Any],name:str)->float:
 available=boolean(row[f"{name}_available"],"available")
 if name in ("temporal_mismatch","predicate_mismatch"):
  return softplus(number(row[f"raw_alpha_{name.split('_')[0]}"],"raw")) if available and boolean(row[f"{name}_condition_input"],"condition") else 0.0
 if name=="temporal_adapter":
  return sigmoid(number(row["temporal_adapter_logit"],"logit"))*number(row["temporal_adapter_final_penalty_scale"],"scale") if available else 0.0
 return sigmoid(number(row["temporal_channel_logit"],"logit"))*(1-number(row["preservation_entitlement_prob"],"preservation",True))*number(row["temporal_channel_gated_penalty_scale"],"scale") if available else 0.0
def reconstruct(row:dict[str,Any],validate:bool=False)->dict[str,Any]:
 e=number(row["frame_prob"],"frame",True)*number(row["predicate_coverage_prob"],"predicate",True)*number(row["sufficiency_prob"],"sufficiency",True)
 a=softplus(number(row["raw_alpha"],"alpha")); d=(e*number(row["negative_energy"],"neg"),number(row["not_entitled_bias"],"bias")+a*(1-e),e*number(row["positive_energy"],"pos"))
 magnitude=sum(branch(row,b) for b in BRANCHES); final=(d[0]-magnitude,d[1]+magnitude,d[2]-magnitude); pred=argmax(final)
 if validate:
  targets=tuple(number(row[f"final_{x}_logit"],"target") for x in ("refute","not_entitled","support"))
  if abs(e-number(row["entitlement_prob_native"],"entitlement"))>TOL or max(abs(x-y) for x,y in zip(final,targets))>TOL or pred!=row["final_native_prediction"]:raise ValueError("native composer reconstruction failed")
 return {"entitlement":e,"final":final,"prediction":pred}
def apply_mask(r:dict[str,Any],d:dict[str,Any],mask:str)->dict[str,Any]:
 if not re.fullmatch(r"[01]{5}",mask):raise ValueError("invalid action mask")
 x=dict(r)
 for bit,p in zip(mask,PRIMITIVES):
  if bit=="1":
   for f in PRIMITIVE_FIELDS[p]:x[f]=d[f]
 return reconstruct(x)

def sources(ns:argparse.Namespace,g:list[dict[str,Any]])->dict[str,Any]:
 root=ns.repo_root.resolve(); b5p=ns.stage196b2b5_analysis_json.resolve();b4p=ns.stage196b2b4_analysis_json.resolve();p0=ns.stage196b2b3p0_run_root.resolve();out=ns.output_dir.resolve()
 raw_paths=(ns.repo_root,ns.stage196b2b5_analysis_json,ns.stage196b2b4_analysis_json,ns.stage196b2b3p0_run_root,ns.output_dir);explicit=all(x.is_absolute() and (x.resolve()==root or root in x.resolve().parents) for x in raw_paths) and all(re.fullmatch(r"[0-9a-f]{40}",x) for x in (ns.stage196b2b3p0_runtime_git_commit,ns.current_git_commit))
 gate(g,"invocation","","explicit_paths",True,{"all_absolute_and_under_repo_root":explicit},root.is_dir() and explicit,"all paths must be explicit and under repo root")
 if b5p.name!=B5_FILES[0] or b4p.name!=B4_FILES[0]:raise ValueError("analysis JSON basenames must be exact")
 exact_dir(b5p.parent,B5_FILES,g,"b2b5_exact_nine_file_closure");exact_dir(b4p.parent,B4_FILES,g,"b2b4_exact_nine_file_closure")
 b5,b4=read_json(b5p),read_json(b4p)
 r5={"decision":"STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED","recommended_next_stage":"STAGE196B2B6_MINIMAL_SELECTOR_INTERVENTION_DESIGN","blocking_reasons":[]}
 r4={"decision":"STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION","recommended_next_stage":"STAGE196B2B5_ROW_SELECTOR_OBSERVABILITY","blocking_reasons":[]}
 gate(g,"b2b5","","b2b5_decision_closure",r5,{k:b5.get(k) for k in r5},{k:b5.get(k) for k in r5}==r5,"B2-B5 decision changed")
 c5=b5.get("current_git_commit",b5.get("current_analyzer_git_commit"));gate(g,"b2b5","","b2b5_runtime_commit",B5_COMMIT,c5,c5==B5_COMMIT,"B2-B5 runtime commit changed")
 gate(g,"b2b4","","b2b4_decision_closure",r4,{k:b4.get(k) for k in r4},{k:b4.get(k) for k in r4}==r4,"B2-B4 decision changed")
 for label,path,names in (("b2b5",b5p,B5_FILES),("b2b4",b4p,B4_FILES)):
  c=read_csv(path.parent/names[-1]);gate(g,label,"",f"{label}_contract_closure",True,{"rows":len(c),"failed":sum(not boolean(r.get("passed"),"passed") for r in c)},closed(c),f"{label} contract failed")
 f=read_csv(b5p.parent/B5_FILES[2]);a=read_csv(b5p.parent/B5_FILES[3]);s=read_csv(b5p.parent/B5_FILES[4]);q=read_csv(b5p.parent/B5_FILES[5]);p=read_csv(b4p.parent/B4_FILES[2]);t=read_csv(b4p.parent/B4_FILES[4])
 cols(f,("feature_family","feature_name","deployment_authorized","diagnostic_only","source_fields","formula","outcome_derived","available"),"dictionary");cols(a,("seed","stable_row_id","transition_role","direction","acceptable_coalitions"),"actions");cols(s,("feature_subset_mask","feature_subset_members","stable_row_id","signature","transfer_status"),"signatures");cols(q,("feature_subset_mask","feature_subset_members","feasible","pooled_full_pass","bidirectional_cross_seed_full_pass"),"selectors")
 keys={(integer(r["seed"],"seed"),r["stable_row_id"]) for r in a};counts={z:Counter() for z in PRIMARY_SEEDS}
 for r in a:counts[integer(r["seed"],"seed")][r["transition_role"]]+=1
 observed={z:dict(counts[z]) for z in PRIMARY_SEEDS};gate(g,"b2b5","","b2b5_cross_seed_selector_authority",{"identities":16,"roles":EXPECTED_PRIMARY,"direction":FORWARD},{"identities":len(keys),"roles":observed,"directions":sorted({r["direction"] for r in a})},len(a)==len(keys)==16 and observed==EXPECTED_PRIMARY and {r["direction"] for r in a}=={FORWARD},"primary authority changed")
 pc={"primitive coalition rows":len(p),"primitive tail summaries":len(t),"primitive coalitions":len({r["coalition_mask"] for r in p}),"primary identities":len({(r["seed"],r["stable_row_id"]) for r in p}),"directional states":len({(r["seed"],r["epoch"],r["stable_row_id"],r["direction"]) for r in p})};expected={"primitive coalition rows":20480,"primitive tail summaries":1024,"primitive coalitions":32,"primary identities":16,"directional states":640}
 gate(g,"b2b4","","b2b4_primitive_action_closure",expected,pc,pc==expected and sorted({r["coalition_mask"] for r in p})==[f"{i:05b}" for i in range(32)],"primitive lattice changed")
 hashes={str(path.parent/name):sha256(path.parent/name) for path,names in ((b5p,B5_FILES),(b4p,B4_FILES)) for name in names}
 return {"b5":b5,"b4":b4,"b5p":b5p,"b4p":b4p,"p0":p0,"features":f,"actions":a,"signatures":s,"selectors":q,"primitive":p,"tails":t,"keys":keys,"hashes":hashes,"b4_counts":pc}

def load_p0(ns:argparse.Namespace,src:dict[str,Any],g:list[dict[str,Any]])->dict[str,Any]:
 root=src["p0"];runs=sorted(x.name for x in root.iterdir() if x.is_dir());gate(g,"p0","","p0_six_run_closure",sorted(RUNS),runs,runs==sorted(RUNS),"six-run closure failed")
 gate(g,"p0","","p0_runtime_commit_agreement",P0_COMMIT,ns.stage196b2b3p0_runtime_git_commit,ns.stage196b2b3p0_runtime_git_commit==P0_COMMIT,"P0 commit mismatch")
 states=defaultdict(dict);hashes={};cr=tr=cs=ts=native=0
 for run in RUNS:
  seed,mode=int(run[4:7]),run[8:];cd=root/run/"composer_inputs";td=root/run/"trajectory";cn=[f"stage196b2b3p0_epoch_composer_inputs_{e:03d}.jsonl" for e in EPOCHS];tn=[f"stage196b2p0_epoch_channels_{e:03d}.jsonl" for e in EPOCHS];mn="stage196b2b3p0_composer_input_manifest.json"
  exact_dir(cd,(mn,*cn),g,f"{run}_composer_namespace");files=sorted(x.name for x in td.iterdir() if x.is_file());pattern=re.compile(r"^stage196b2p0_epoch_channels_[0-9]{3}\.jsonl$");observed=sorted(x for x in files if pattern.fullmatch(x));malformed=sorted(x for x in files if x.startswith("stage196b2p0_epoch_channels_") and x not in tn)
  gate(g,"p0",run,"trajectory_namespace_closure",{"expected":tn,"malformed":[]},{"observed":observed,"malformed":malformed,"unrelated_ignored":sorted(x for x in files if not x.startswith("stage196b2p0_epoch_channels_"))},observed==tn and not malformed,"trajectory namespace failed")
  m=read_json(cd/mn);hashes[str(cd/mn)]=sha256(cd/mn);mok=m.get("completed") is True and m.get("current_git_commit")==P0_COMMIT and m.get("seed")==seed and m.get("gradient_ownership_mode")==mode and m.get("sidecar_files")==cn;gate(g,"p0",run,"composer_manifest",True,{"completed":m.get("completed"),"commit":m.get("current_git_commit"),"seed":m.get("seed"),"mode":m.get("gradient_ownership_mode")},mok,"manifest changed")
  schema=None
  for epoch,cf,tf in zip(EPOCHS,cn,tn):
   cp,tp=cd/cf,td/tf;crows,trows=read_jsonl(cp),read_jsonl(tp);hashes[str(cp)]=sha256(cp);hashes[str(tp)]=sha256(tp)
   if m.get("sidecar_sha256",{}).get(cf)!=hashes[str(cp)] or len(crows)!=720 or len(trows)!=720 or any(set(r)!=TRAJECTORY_FIELDS for r in trows):raise ValueError(f"{run}:{epoch}: sidecar closure failed")
   schema=set(crows[0]) if schema is None else schema;ti={(str(r["id"]),str(r["source_row_id"]),integer(r["dev_position"],"position")):r for r in trows};seen=set()
   if len(ti)!=720:raise ValueError("trajectory identity collision")
   for r in crows:
    ident=(str(r["id"]),str(r["source_row_id"]),integer(r["dev_position"],"position"))
    if set(r)!=schema or ident in seen or ident not in ti:raise ValueError("composer identity/schema failure")
    seen.add(ident)
    if r.get("current_git_commit")!=P0_COMMIT or integer(r.get("seed"),"seed")!=seed or integer(r.get("epoch"),"epoch")!=epoch or r.get("gradient_ownership_mode")!=mode:raise ValueError("composer provenance failure")
    if ti[ident]["prediction"]!=r["final_native_prediction"]:raise ValueError("trajectory prediction mismatch")
    reconstruct(r,True);native+=1
    if epoch in TAIL:
     x=dict(r);x["_trajectory"]=ti[ident];x["_stable"]=str(r.get("stable_row_id",r["id"]));states[(seed,epoch,*ident)][mode]=x
   cr+=720;tr+=720;cs+=1;ts+=1
 counts={"composer sidecars":cs,"trajectory sidecars":ts,"composer rows":cr,"trajectory rows":tr};expected={"composer sidecars":120,"trajectory sidecars":120,"composer rows":86400,"trajectory rows":86400}
 gate(g,"p0","","p0_count_and_recomposition_closure",expected,{**counts,"tail states":len(states),"native recompositions":native},counts==expected and len(states)==6480 and all(set(x)==set(MODES) for x in states.values()) and native==86400,"P0 closure failed")
 return {"states":states,"hashes":hashes,"counts":counts}

def feature(name:str,row:dict[str,Any])->Any:
 if name=="RECIPIENT_PREDICTION_SEQUENCE":return row["final_native_prediction"]
 fm=number(row["final_support_logit"],"final S")-number(row["final_not_entitled_logit"],"final NE");hm=number(row["decision_head_support_logit"],"head S")-number(row["decision_head_not_entitled_logit"],"head NE")
 if name=="FINAL_MARGIN_SIGN_SEQUENCE":return sign(fm)
 if name=="HEAD_MARGIN_SIGN_SEQUENCE":return sign(hm)
 if name=="HEAD_FINAL_MARGIN_SIGN_CONFLICT_SEQUENCE":return sign(hm)!=sign(fm)
 halfs={"FRAME_HALFSPACE_SEQUENCE":"frame_prob","PREDICATE_HALFSPACE_SEQUENCE":"predicate_coverage_prob","SUFFICIENCY_HALFSPACE_SEQUENCE":"sufficiency_prob","ENTITLEMENT_HALFSPACE_SEQUENCE":"entitlement_prob_native"}
 if name in halfs:return half(number(row[halfs[name]],halfs[name],True))
 if name=="ENTITLEMENT_BOTTLENECK_SEQUENCE":
  xs=[("FRAME",number(row["frame_prob"],"frame")),("PREDICATE",number(row["predicate_coverage_prob"],"predicate")),("SUFFICIENCY",number(row["sufficiency_prob"],"sufficiency"))];m=min(v for _,v in xs);return sorted(k for k,v in xs if abs(v-m)<=ZERO_TOL)
 if name=="POLARITY_ENERGY_ORDER_SEQUENCE":
  x=number(row["positive_energy"],"pos")-number(row["negative_energy"],"neg");return "EQUAL" if abs(x)<=ZERO_TOL else "POSITIVE_DOMINANT" if x>0 else "NEGATIVE_DOMINANT"
 fields={"PREDICATE_MISMATCH_SEQUENCE":"predicate_mismatch_active","TEMPORAL_MISMATCH_SEQUENCE":"temporal_mismatch_active","TEMPORAL_ADAPTER_ACTIVITY_SEQUENCE":"temporal_adapter_active","TEMPORAL_CHANNEL_ACTIVITY_SEQUENCE":"temporal_channel_active"}
 if name not in fields:raise ValueError(f"unrecognized frozen feature {name}")
 return boolean(row[fields[name]],fields[name])
def signature(members:Sequence[str],tail:Sequence[dict[str,Any]])->list[Any]:return [[feature(n,r) for r in tail] for n in members]

def candidates(src:dict[str,Any],p0:dict[str,Any],g:list[dict[str,Any]])->tuple[list[dict[str,Any]],dict[Any,Any],dict[Any,Any]]:
 dictionary={r["feature_name"]:r for r in src["features"] if r["feature_family"]=="recipient_local"};eligible=[]
 for r in src["selectors"]:
  if r["feature_family"]=="recipient_local" and boolean(r["feasible"],"feasible") and boolean(r["pooled_full_pass"],"pooled") and boolean(r["bidirectional_cross_seed_full_pass"],"cross"):eligible.append(r)
 if not eligible:raise ValueError("no B2-B5 cross-seed recipient subset")
 parsed=[]
 for r in eligible:
  members=jcell(r["feature_subset_members"],"members")
  if not isinstance(members,list) or len(members)!=integer(r["feature_subset_size"],"size") or any(n not in dictionary for n in members):raise ValueError("candidate members invalid")
  for n in members:
   d=dictionary[n];fields=jcell(d["source_fields"],"source fields")
   if n not in FORMULAS or tuple(fields)!=FORMULAS[n][0] or d["formula"]!=FORMULAS[n][1]:raise ValueError("frozen feature formula drift")
   if not (boolean(d["deployment_authorized"],"authorized") and not boolean(d["diagnostic_only"],"diagnostic") and not boolean(d["outcome_derived"],"outcome") and boolean(d["available"],"available")) or set(fields)&PROHIBITED_FIELDS:raise ValueError("unauthorized selector feature")
  parsed.append((r,set(members)))
 minimal=[r for r,m in parsed if not any(om<m for _,om in parsed)];gate(g,"candidate","","candidate_inclusion_minimal_subset_closure",True,{"eligible":len(eligible),"retained":len(minimal),"masks":sorted(r["feature_subset_mask"] for r in minimal)},bool(minimal),"no minimal candidates")
 meta={(integer(r["seed"],"seed"),r["stable_row_id"]):r for r in src["actions"]};acceptable={}
 tails=defaultdict(list)
 for r in src["tails"]:
  k=(integer(r["seed"],"seed"),r["stable_row_id"])
  if k in meta and r["direction"]==FORWARD:tails[k].append(r)
 for k,rows in tails.items():
  if len(rows)!=32:raise ValueError("B2-B4 forward 32-mask closure failed")
  role=meta[k]["transition_role"];acceptable[k]={r["coalition_mask"] for r in rows if boolean(r["donor_tail_reproduced"] if role=="recovery" else r["recipient_tail_preserved"],"objective")}
  if acceptable[k]!=set(jcell(meta[k]["acceptable_coalitions"],"B2-B5 actions")):raise ValueError("B2-B4/B2-B5 action mismatch")
 if set(acceptable)!=set(meta):raise ValueError("primary action reconstruction incomplete")
 state_by_stable={}
 for k,pair in p0["states"].items():
  seed,epoch=k[:2];joint=pair["joint"];state_by_stable[(seed,epoch,joint["_stable"])]=pair
 maps={};out=[];sigrows=[];stored={(r["feature_subset_mask"],integer(r["seed"],"seed"),r["stable_row_id"]):jcell(r["signature"],"stored signature") for r in src["signatures"] if r["feature_family"]=="recipient_local" and r["transfer_status"]=="POOLED"}
 for r in minimal:
  mask=r["feature_subset_mask"];members=jcell(r["feature_subset_members"],"members");groups=defaultdict(list)
  for key in sorted(meta):
   tail=[state_by_stable[(key[0],e,key[1])]["joint"] for e in TAIL];sig=signature(members,tail)
   if stored.get((mask,*key))!=sig:raise ValueError(f"{mask}:{key}: independent signature disagreement")
   groups[canon(sig)].append(key)
  transfer={}
  for source,target in ((184,185),(185,184)):
   source_map={}
   for sg,ks in groups.items():
    sk=[k for k in ks if k[0]==source]
    if sk:source_map[sg]=set.intersection(*(acceptable[k] for k in sk))
   target_rows=[k for ks in groups.values() for k in ks if k[0]==target];transfer[(source,target)]=all((sg:=next(s for s,ks in groups.items() if k in ks)) in source_map and bool(source_map[sg]&acceptable[k]) for k in target_rows)
  stored_transfer={(184,185):boolean(r["seed184_to_seed185_full_pass"],"transfer"),(185,184):boolean(r["seed185_to_seed184_full_pass"],"transfer")}
  if transfer!=stored_transfer:raise ValueError("independent cross-seed transfer disagreement")
  smap={};unique=ambiguous=empty=unsafe=0
  for sig,keys in sorted(groups.items()):
   inter=set.intersection(*(acceptable[k] for k in keys));actions=minima(inter)
   if not actions:classification="INVALID_EMPTY_INTERSECTION";empty+=1
   elif len(actions)==1:classification="UNIQUE_DETERMINISTIC";unique+=1
   else:classification="SET_VALUED_EXACT";ambiguous+=1
   action_safe=all(a in acceptable[k] for a in actions for k in keys);unsafe_here=bool(actions) and not action_safe;unsafe+=unsafe_here
   smap[sig]=actions;sigrows.append({"feature_subset_mask":mask,"feature_subset_members":members,"signature":json.loads(sig),"signature_row_count":len(keys),"seed_counts":dict(sorted(Counter(k[0] for k in keys).items())),"transition_role_counts":dict(sorted(Counter(meta[k]["transition_role"] for k in keys).items())),"acceptable_action_intersection":sorted(inter),"inclusion_minimal_action_set":actions,"action_set_size":len(actions),"classification":classification,"safe_set_valued":classification=="SET_VALUED_EXACT" and action_safe,"unsafe_action_ambiguity":unsafe_here,"primary_row_ids":[k[1] for k in keys]})
  classification="INVALID_EMPTY_INTERSECTION" if empty else "SET_VALUED_EXACT" if ambiguous else "UNIQUE_DETERMINISTIC";maps[mask]={"members":members,"map":smap,"groups":groups,"classification":classification,"unsafe":unsafe}
  out.append({"feature_subset_mask":mask,"feature_subset_size":len(members),"feature_subset_members":members,"signature_count":len(groups),"maximum_rows_per_signature":max(map(len,groups.values())),"mixed_role_signature_count":sum(len({meta[k]["transition_role"] for k in ks})>1 for ks in groups.values()),"cross_seed_signature_count":sum(len({k[0] for k in ks})>1 for ks in groups.values()),"seed184_to_seed185_full_pass":transfer[(184,185)],"seed185_to_seed184_full_pass":transfer[(185,184)],"classification":classification,"policy_equivalence_class":"","pooled_full_pass":True,"bidirectional_cross_seed_full_pass":True,"unique_signature_action_count":unique,"ambiguous_signature_action_count":ambiguous,"empty_signature_action_count":empty,"primary_policy_passed":False,"seed184_nonprimary_safety_passed":False,"seed185_nonprimary_safety_passed":False,"safety_authorized":False,"nondominated":False})
 gate(g,"signature","","independent_signature_reconstruction",{"primary":16,"candidates":len(out)},{"primary":len(meta),"candidates":len(maps),"disagreements":0},len(meta)==16,"signature closure failed");gate(g,"signature","","signature_action_intersection_closure",True,{"empty":sum(x["empty_signature_action_count"] for x in out)},all(x["empty_signature_action_count"]==0 for x in out),"empty signature intersection")
 return out,maps,{"meta":meta,"acceptable":acceptable,"tails":tails,"signature_rows":sigrows,"states":state_by_stable}

def primary_validation(rows:list[dict[str,Any]],maps:dict[Any,Any],ctx:dict[str,Any],g:list[dict[str,Any]])->list[dict[str,Any]]:
 out=[]
 for c in rows:
  mask=c["feature_subset_mask"];fail=0;recover=harm=0
  for sig,keys in maps[mask]["groups"].items():
   actions=maps[mask]["map"][sig]
   for key in keys:
    role=ctx["meta"][key]["transition_role"]
    for action in actions:
     preds=[]
     for e in TAIL:
      pair=ctx["states"][(key[0],e,key[1])];preds.append(apply_mask(pair["joint"],pair["frame_local_only"],action)["prediction"])
     tail=next(r for r in ctx["tails"][key] if r["coalition_mask"]==action);expected=jcell(tail["coalition_tail_predictions"],"coalition tail")
     if preds!=expected:raise ValueError("P0/B2-B4 counterfactual reconstruction disagreement")
     passed=action in ctx["acceptable"][key];fail+=not passed;recover+=role=="recovery" and passed;harm+=role=="harm" and passed
     out.append({"feature_subset_mask":mask,"seed":key[0],"stable_row_id":key[1],"transition_role":role,"signature":json.loads(sig),"assigned_action_set":actions,"evaluated_action":action,"recipient_tail_status":tail["recipient_tail_status"],"donor_tail_status":tail["donor_tail_status"],"counterfactual_tail_status":status(preds),"objective":"DONOR_TAIL_REPRODUCTION" if role=="recovery" else "RECIPIENT_TAIL_PRESERVATION","objective_passed":passed,"donor_tail_reproduced":boolean(tail["donor_tail_reproduced"],"donor"),"recipient_tail_preserved":boolean(tail["recipient_tail_preserved"],"recipient")})
  c["primary_policy_passed"]=fail==0 and recover>=7 and harm>=9
 observed={"primary":len(ctx["meta"]),"recovery":sum(r["transition_role"]=="recovery" for r in ctx["meta"].values()),"harm":sum(r["transition_role"]=="harm" for r in ctx["meta"].values()),"objective_failures":sum(not r["objective_passed"] for r in out)};gate(g,"primary","","primary_16_row_closure",16,observed["primary"],observed["primary"]==16,"primary count failed");gate(g,"primary","","recovery_7_row_closure",7,observed["recovery"],observed["recovery"]==7,"recovery count failed");gate(g,"primary","","harm_9_row_closure",9,observed["harm"],observed["harm"]==9,"harm count failed");gate(g,"primary","","primary_objective_closure",0,observed["objective_failures"],observed["objective_failures"]==0 and all(c["primary_policy_passed"] for c in rows),"primary policy closure failed")
 return out

def metrics(gold:list[str],base:list[str],sel:list[str])->dict[str,Any]:
 n=len(gold);acc=lambda p:sum(a==b for a,b in zip(gold,p))/n if n else None
 def f1(p:str)->float:
  tp=sum(a==p and b==p for a,b in zip(gold,sel));fp=sum(a!=p and b==p for a,b in zip(gold,sel));fn=sum(a==p and b!=p for a,b in zip(gold,sel));return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
 def bf1(p:str)->float:
  tp=sum(a==p and b==p for a,b in zip(gold,base));fp=sum(a!=p and b==p for a,b in zip(gold,base));fn=sum(a==p and b!=p for a,b in zip(gold,base));return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
 ba,sa=acc(base),acc(sel);bm=sum(bf1(x) for x in LABELS)/3;sm=sum(f1(x) for x in LABELS)/3;support=sum(x=="SUPPORT" for x in gold);br=sum(a==b=="SUPPORT" for a,b in zip(gold,base))/support if support else None;sr=sum(a==b=="SUPPORT" for a,b in zip(gold,sel))/support if support else None;stable=sum(a==b for a,b in zip(gold,base));cti=sum(a==b and c!=a for a,b,c in zip(gold,base,sel))
 def counts(p:list[str])->tuple[int,int,int]:return (sum(a!="NOT_ENTITLED" and b=="NOT_ENTITLED" for a,b in zip(gold,p)),sum(a=="NOT_ENTITLED" and b!="NOT_ENTITLED" for a,b in zip(gold,p)),sum(a in ("SUPPORT","REFUTE") and b in ("SUPPORT","REFUTE") and a!=b for a,b in zip(gold,p)))
 bc,sc=counts(base),counts(sel)
 return {"joint_accuracy":ba,"selector_accuracy":sa,"accuracy_delta":sa-ba,"joint_macro_f1":bm,"selector_macro_f1":sm,"macro_f1_delta":sm-bm,"joint_support_recall":br,"selector_support_recall":sr,"support_recall_delta":None if br is None else sr-br,"joint_false_not_entitled":bc[0],"selector_false_not_entitled":sc[0],"joint_false_entitlement":bc[1],"selector_false_entitlement":sc[1],"joint_polarity_error":bc[2],"selector_polarity_error":sc[2],"prediction_change_count":sum(a!=b for a,b in zip(base,sel)),"correct_to_incorrect_count":cti,"incorrect_to_correct_count":sum(a!=g and b==g for g,a,b in zip(gold,base,sel)),"stable_correct_preservation_rate":1.0 if stable==0 else (stable-cti)/stable}

def clean_dev(candidates:list[dict[str,Any]],maps:dict[Any,Any],ctx:dict[str,Any],p0:dict[str,Any],g:list[dict[str,Any]])->tuple[list[dict[str,Any]],list[dict[str,Any]],dict[str,Any]]:
 by_ident=defaultdict(dict)
 for key,pair in p0["states"].items():by_ident[(key[0],key[2],key[3],key[4])][key[1]]=pair
 if len(by_ident)!=2160 or any(set(v)!=set(TAIL) for v in by_ident.values()):raise ValueError("2,160-state tail closure failed")
 primary_identities={(ctx["meta"][(s,r)]["id"],ctx["meta"][(s,r)]["source_row_id"],integer(ctx["meta"][(s,r)]["dev_position"],"position")) for s,r in ctx["meta"]}
 audit=[];summaries=[];detail={};retention_checks=[]
 for c in candidates:
  mask=c["feature_subset_mask"];members=maps[mask]["members"];records=[]
  for ident,epochs in sorted(by_ident.items()):
   seed,identity,source,pos=ident;tail=[epochs[e]["joint"] for e in TAIL];sig=canon(signature(members,tail));signature_seen=sig in maps[mask]["map"];actions=maps[mask]["map"][sig] if signature_seen else [];is_primary=(identity,source,pos) in primary_identities;support="PRIMARY_SEEN_SIGNATURE" if signature_seen and is_primary else "NONPRIMARY_SEEN_SIGNATURE" if signature_seen else "UNSEEN_SIGNATURE";gold=tail[-1]["_trajectory"]["gold_label"]
   rec={"ident":ident,"stable":tail[-1]["_stable"],"sig":sig,"actions":actions,"seen":signature_seen,"support":support,"gold":gold,"base":tail[-1]["final_native_prediction"],"epochs":epochs,"primary":is_primary};records.append(rec)
   abstained=support=="UNSEEN_SIGNATURE";assigned=actions if signature_seen else ["00000"];audit.append({"feature_subset_mask":mask,"seed":seed,"stable_row_id":rec["stable"],"id":identity,"source_row_id":source,"dev_position":pos,"gold_label":gold,"recipient_tail_predictions":[r["final_native_prediction"] for r in tail],"signature":json.loads(sig),"support_class":support,"assigned_action_set":assigned,"abstained":abstained})
   if abstained:
    fallback_prediction=apply_mask(epochs[20]["joint"],epochs[20]["frame_local_only"],"00000")["prediction"];retention_checks.append({"feature_subset_mask":mask,"seed":seed,"stable_row_id":rec["stable"],"joint_prediction":rec["base"],"selector_prediction":fallback_prediction,"prediction_changed":fallback_prediction!=rec["base"]})
  detail[mask]={}
  for seed in SEEDS:
   sr=[r for r in records if r["ident"][0]==seed];seen=sum(r["seen"] for r in sr);detail[mask][str(seed)]={"total_rows":720,"seen_rows":seen,"unseen_rows":720-seen,"coverage_rate":seen/720,"signature_count":len({r["sig"] for r in sr}),"unseen_signature_count":len({r["sig"] for r in sr if not r["seen"]}),"seen_exact_no_op_policy_rows":sum(r["seen"] and r["actions"]==["00000"] for r in sr)}
   pops=("CONTRAST_PRIMARY_IDENTITIES","NONPRIMARY","ALL_720") if seed==183 else ("PRIMARY_16","NONPRIMARY_704","ALL_720")
   for pop in pops:
    rr=sr if pop=="ALL_720" else [r for r in sr if r["primary"]==(pop in ("PRIMARY_16","CONTRAST_PRIMARY_IDENTITIES"))]
    interpretations=[("UNIQUE_DETERMINISTIC",None,None)] if c["classification"]=="UNIQUE_DETERMINISTIC" else [(f"SET_VALUED_SIGNATURE_ACTION:{hashlib.sha256(sig.encode()).hexdigest()[:12]}:{a}",sig,a) for sig,acts in maps[mask]["map"].items() for a in acts]
    for mode,sig_filter,action_filter in interpretations:
     gold=[r["gold"] for r in rr];base=[r["base"] for r in rr];sel=[]
     for r in rr:
      action=(r["actions"][0] if mode=="UNIQUE_DETERMINISTIC" and r["seen"] else action_filter if r["sig"]==sig_filter else None)
      sel.append(apply_mask(r["epochs"][20]["joint"],r["epochs"][20]["frame_local_only"],action)["prediction"] if action else r["base"])
     m=metrics(gold,base,sel);seen_count=sum(r["seen"] for r in rr);summaries.append({"feature_subset_mask":mask,"seed":seed,"population":pop,"policy_action_mode":mode,"row_count":len(rr),"seen_signature_count":seen_count,"unseen_signature_count":len(rr)-seen_count,"coverage_rate":seen_count/len(rr) if rr else None,**m,"safety_passed":m["correct_to_incorrect_count"]==0 and m["stable_correct_preservation_rate"]==1.0})
  for seed in PRIMARY_SEEDS:
   non=[r for r in summaries if r["feature_subset_mask"]==mask and r["seed"]==seed and r["population"]=="NONPRIMARY"]
   passed=bool(non) and all(r["safety_passed"] for r in non);c[f"seed{seed}_nonprimary_safety_passed"]=passed
  c["safety_authorized"]=c["primary_policy_passed"] and c["seed184_nonprimary_safety_passed"] and c["seed185_nonprimary_safety_passed"] and maps[mask]["unsafe"]==0
 gate(g,"clean_dev","","clean_dev_2160_state_signature_audit",2160,{"states":len(by_ident),"audit_rows":len(audit),"candidates":len(candidates)},len(audit)==2160*len(candidates),"clean-dev audit incomplete")
 unseen=[r for r in audit if r["support_class"]=="UNSEEN_SIGNATURE"];seen=[r for r in audit if r["support_class"] in ("PRIMARY_SEEN_SIGNATURE","NONPRIMARY_SEEN_SIGNATURE")];retention_violations=abs(len(retention_checks)-len(unseen))+sum(r["selector_prediction"]!=r["joint_prediction"] or r["prediction_changed"] for r in retention_checks);abstention_evidence={"total_audit_rows":len(audit),"unseen_signature_rows":len(unseen),"seen_signature_rows":len(seen),"invalid_support_class_violations":len(audit)-len(unseen)-len(seen),"unseen_mapping_presence_violations":sum(canon(r["signature"]) in maps[r["feature_subset_mask"]]["map"] for r in unseen),"unseen_abstention_violations":sum(r["abstained"] is not True for r in unseen),"unseen_fallback_mask_violations":sum(r["assigned_action_set"]!=["00000"] for r in unseen),"seen_marked_abstained_violations":sum(r["abstained"] is not False for r in seen),"seen_empty_action_set_violations":sum(not r["assigned_action_set"] for r in seen),"seen_action_set_mismatch_violations":sum(r["assigned_action_set"]!=maps[r["feature_subset_mask"]]["map"].get(canon(r["signature"])) for r in seen),"seen_exact_no_op_policy_rows":sum(r["assigned_action_set"]==["00000"] for r in seen),"joint_prediction_retention_violations":retention_violations};abstention_required={"unseen_signature":{"abstained":True,"fallback_action":["00000"],"retain_joint_prediction":True},"seen_signature":{"abstained":False,"assigned_action_set_nonempty":True,"assigned_action_set_equals_reconstructed_mapping":True,"exact_no_op_action_allowed":True}};violation_fields=("invalid_support_class_violations","unseen_mapping_presence_violations","unseen_abstention_violations","unseen_fallback_mask_violations","seen_marked_abstained_violations","seen_empty_action_set_violations","seen_action_set_mismatch_violations","joint_prediction_retention_violations");gate(g,"clean_dev","","unseen_signature_abstention_closure",abstention_required,abstention_evidence,all(abstention_evidence[name]==0 for name in violation_fields),"abstention closure failed")
 gate(g,"clean_dev","","paired_component_recomposition_closure",True,{"application_rows":len(summaries),"epoch_used_for_metrics":20,"all_tail_epochs_retained_in_signature":list(TAIL)},bool(summaries),"application audit incomplete")
 return audit,summaries,detail

def equivalence_dominance(candidates:list[dict[str,Any]],maps:dict[Any,Any],support:dict[str,Any],g:list[dict[str,Any]])->tuple[list[dict[str,Any]],dict[str,Any]]:
 fingerprints={c["feature_subset_mask"]:canon({f"{k[0]}:{k[1]}":maps[c["feature_subset_mask"]]["map"][sig] for sig,ks in maps[c["feature_subset_mask"]]["groups"].items() for k in ks}) for c in candidates};classes={};ids={}
 for fp in sorted(set(fingerprints.values())):
  members=sorted(k for k,v in fingerprints.items() if v==fp);cid=f"POLICY_EQUIVALENCE_{len(classes)+1:03d}";classes[cid]={"candidate_masks":members,"structurally_minimal_masks":[m for m in members if not any(set(maps[x]["members"])<set(maps[m]["members"]) for x in members)]}
  for m in members:ids[m]=cid
 for c in candidates:c["policy_equivalence_class"]=ids[c["feature_subset_mask"]]
 rows=[];dominated=set()
 for a in candidates:
  for b in candidates:
   if a is b:continue
   am,bm=a["feature_subset_mask"],b["feature_subset_mask"];strict=set(maps[am]["members"])<set(maps[bm]["members"]);same=fingerprints[am]==fingerprints[bm];coverage=all(support[am][str(s)]["coverage_rate"]>=support[bm][str(s)]["coverage_rate"] for s in SEEDS);safety=(not b["safety_authorized"] or a["safety_authorized"]);authorized=True;dominates=strict and same and coverage and safety and authorized
   if dominates:dominated.add(bm)
   rows.append({"candidate_a":am,"candidate_b":bm,"strict_feature_subset":strict,"same_primary_action_sets":same,"no_lower_clean_dev_coverage":coverage,"no_worse_safety":safety,"authorized_feature_inputs":authorized,"a_structurally_dominates_b":dominates,"details":{"accuracy_not_used":True,"coverage_by_seed":{str(s):[support[am][str(s)]["coverage_rate"],support[bm][str(s)]["coverage_rate"]] for s in SEEDS}}})
 for c in candidates:c["nondominated"]=c["feature_subset_mask"] not in dominated
 gate(g,"policy","","policy_equivalence_closure",True,{"classes":classes},bool(classes),"policy equivalence incomplete");gate(g,"policy","","policy_dominance_completion",True,{"pairs":len(rows),"nondominated":[c["feature_subset_mask"] for c in candidates if c["nondominated"]]},True,"dominance incomplete")
 return rows,classes

def decide(candidates:list[dict[str,Any]],maps:dict[Any,Any],support:dict[str,Any])->tuple[str,str,dict[str,Any]]:
 safe=[c for c in candidates if c["nondominated"] and c["safety_authorized"]];safe_classes={c["policy_equivalence_class"] for c in safe};unique=[c for c in safe if c["classification"]=="UNIQUE_DETERMINISTIC"]
 all_primary=all(c["primary_policy_passed"] for c in candidates);all_unsafe=all(not(c["seed184_nonprimary_safety_passed"] and c["seed185_nonprimary_safety_passed"]) for c in candidates);low=all(support[c["feature_subset_mask"]][str(s)]["seen_rows"]<=support[c["feature_subset_mask"]][str(s)]["unseen_rows"] for c in candidates for s in PRIMARY_SEEDS);amb=all(c["classification"]=="SET_VALUED_EXACT" for c in candidates) and any(maps[c["feature_subset_mask"]]["unsafe"] for c in candidates)
 ev={"ordered_rules":["unique_minimal_safe","multiple_safe_exact","primary_exact_clean_dev_unsafe","signature_support_insufficient","action_ambiguity_unresolved"],"safe_nondominated_candidates":[c["feature_subset_mask"] for c in safe],"safe_policy_classes":sorted(safe_classes),"all_primary_exact":all_primary,"all_clean_dev_unsafe":all_unsafe,"abstention_primary":low,"unsafe_action_ambiguity":amb,"accuracy_used_for_authorization":False}
 if unique and len(safe_classes)==1:return "STAGE196B2B6_UNIQUE_MINIMAL_SAFE_SELECTOR","STAGE196B2B7_SELECTOR_ARCHITECTURE_INTEGRATION_DESIGN",ev
 if len(safe_classes)>1 or safe and not unique:return "STAGE196B2B6_MULTIPLE_SAFE_SELECTOR_POLICIES","STAGE196B2B7_CONTROLLED_SELECTOR_POLICY_ABLATION",ev
 if all_primary and all_unsafe:return "STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE","STAGE196B2B6P0_SELECTOR_SAFETY_STATE_OBSERVABILITY",ev
 if all_primary and low:return "STAGE196B2B6_SIGNATURE_SUPPORT_INSUFFICIENT","STAGE196B2B6P0_SELECTOR_COVERAGE_OBSERVABILITY",ev
 if amb:return "STAGE196B2B6_ACTION_AMBIGUITY_UNRESOLVED","STAGE196B2B6P0_ACTION_DISAMBIGUATION_OBSERVABILITY",ev
 raise ValueError("deterministic decision rules did not resolve")

def analyze(ns:argparse.Namespace,g:list[dict[str,Any]])->tuple[dict[str,Any],dict[str,Any]]:
 src=sources(ns,g);p0=load_p0(ns,src,g);cand,maps,ctx=candidates(src,p0,g);gate(g,"b2b5","","b2b5_outcome_leakage_prohibition",True,{"candidate_features_authorized":True,"prohibited_fields":sorted(PROHIBITED_FIELDS)},True,"outcome leakage detected");primary=primary_validation(cand,maps,ctx,g);audit,apps,support=clean_dev(cand,maps,ctx,p0,g);dom,classes=equivalence_dominance(cand,maps,support,g);decision,next_stage,evaluation=decide(cand,maps,support)
 for name,passed,observed in (("seed184_nonprimary_safety_audit",all(any(r["seed"]==184 and r["population"]=="NONPRIMARY" for r in apps) for _ in [0]),{"completed":True}),("seed185_nonprimary_safety_audit",any(r["seed"]==185 and r["population"]=="NONPRIMARY" for r in apps),{"completed":True}),("seed183_contrast_audit",any(r["seed"]==183 and r["population"]=="CONTRAST_PRIMARY_IDENTITIES" and r["row_count"]==16 for r in apps),{"hard_authorization_condition":False,"contrast_primary_rows":16})):
  gate(g,"safety","",name,True,observed,passed,f"{name} incomplete")
 gate(g,"decision","","decision_evaluation_completion",True,evaluation,True,"decision incomplete");gate(g,"output","","exact_nine_output_closure",list(OUTPUTS),list(OUTPUTS),True,"output plan changed")
 hashes={**src["hashes"],**p0["hashes"]};unique=[c["feature_subset_mask"] for c in cand if c["classification"]=="UNIQUE_DETERMINISTIC"];safe_set=[c["feature_subset_mask"] for c in cand if c["classification"]=="SET_VALUED_EXACT" and c["safety_authorized"]];unsafe=[c["feature_subset_mask"] for c in cand if maps[c["feature_subset_mask"]]["unsafe"]]
 analysis={"stage":STAGE,"decision":decision,"recommended_next_stage":next_stage,"blocking_reasons":[],"current_git_commit":ns.current_git_commit,"stage196b2b3p0_runtime_git_commit":ns.stage196b2b3p0_runtime_git_commit,"source_paths":{"stage196b2b5_analysis_json":str(src["b5p"]),"stage196b2b4_analysis_json":str(src["b4p"]),"stage196b2b3p0_run_root":str(src["p0"])},"source_hashes":hashes,"source_closure":{"b2b5_files":list(B5_FILES),"b2b4_files":list(B4_FILES),"b2b4_counts":src["b4_counts"],"p0_counts":p0["counts"]},"candidate_feature_subsets":cand,"policy_equivalence_classes":classes,"signature_action_maps":ctx["signature_rows"],"unique_deterministic_candidates":unique,"safe_set_valued_candidates":safe_set,"unsafe_ambiguous_candidates":unsafe,"primary_policy_validation":primary,"clean_dev_signature_support":support,"clean_dev_safety":apps,"seed183_contrast":{"contrast_only":True,"formal_independent_dataset":False,"support":{m:support[m]["183"] for m in support}},"nondominated_candidates":[c["feature_subset_mask"] for c in cand if c["nondominated"]],"decision_rule_evaluation":evaluation,"authorized_interpretation":"Within the frozen-Mamba, frozen-composer controlled population, recipient-local exact tail signatures may index the reported exact or set-valued primitive action sets subject to primary closure, clean-dev safety, abstention, and the stated decision.","remaining_uncertainty":["The audit reuses the same controlled dataset and architecture.","Unseen signatures receive no inferred action.","Seed183 is contrast-only, not a formal held-out dataset."],"prohibited_claims":PROHIBITED_CLAIMS,"artifact_only":True,"training_performed":False,"model_loaded":False,"checkpoint_loaded":False,"promotion_authorized":False,"accuracy_used_for_authorization":False}
 return analysis,{"candidates":cand,"signatures":ctx["signature_rows"],"primary":primary,"audit":audit,"apps":apps,"dominance":dom}

SECTIONS=("Executive decision","Authorized interpretation","B2-B5 source result","B2-B4 action authority","Source closure","Candidate recipient feature subsets","Independent signature reconstruction","Signature action intersections","Deterministic versus set-valued policies","Policy equivalence classes","Primary 16-row validation","Seed184 clean-dev safety","Seed185 clean-dev safety","Seed183 contrast audit","Unseen-signature behavior","Policy dominance","Decision-rule evaluation","Remaining uncertainty","Prohibited claims","Recommended next stage")
def report(a:dict[str,Any])->str:
 bodies=[f"`{a['decision']}`",a["authorized_interpretation"],"The exact successful B2-B5 decision, next-stage recommendation, empty blocking reasons, nine-file closure, contract, and analyzer commit are required.","B2-B4 supplies the exact 32-mask primitive order and forward row objectives; no performance ranking changes those actions.",canon(a["source_closure"]),canon(a["candidate_feature_subsets"]),"Every candidate signature is rebuilt from P0 joint recipient state using the frozen B2-B5 formulas and exact natural thresholds.",canon(a["signature_action_maps"]),canon({"unique":a["unique_deterministic_candidates"],"safe_set_valued":a["safe_set_valued_candidates"],"unsafe_ambiguous":a["unsafe_ambiguous_candidates"]}),canon(a["policy_equivalence_classes"]),canon(a["primary_policy_validation"]),canon([x for x in a["clean_dev_safety"] if x["seed"]==184]),canon([x for x in a["clean_dev_safety"] if x["seed"]==185]),canon(a["seed183_contrast"]),canon({"unseen_signature":{"support_class":"UNSEEN_SIGNATURE","fallback_action":["00000"],"retain_joint_prediction":True,"counts_as_seen_coverage":False},"seen_exact_no_op_policy":{"classification":"SEEN_EXACT_NO_OP_POLICY","action_set":["00000"],"abstained":False,"counts_as_seen_coverage":True,"row_count":sum(seed_support.get("seen_exact_no_op_policy_rows",0) for candidate_support in a["clean_dev_signature_support"].values() for seed_support in candidate_support.values())}}),canon(a["nondominated_candidates"]),canon(a["decision_rule_evaluation"]),"\n".join(f"- {x}" for x in a["remaining_uncertainty"]),"\n".join(f"- {x}" for x in a["prohibited_claims"]),f"`{a['recommended_next_stage']}`\n\nNo training or promotion is authorized."]
 return f"# {STAGE}: Minimal Recipient-Selector Intervention\n\n"+"\n\n".join(f"## {h}\n\n{b}" for h,b in zip(SECTIONS,bodies))+"\n"
def cv(v:Any)->Any:
 if isinstance(v,(dict,list,tuple)):return canon(v)
 if v is True:return "true"
 if v is False:return "false"
 if v is None:return ""
 return v
def render_csv(header:Sequence[str],rows:Iterable[dict[str,Any]])->str:
 b=io.StringIO(newline="");w=csv.DictWriter(b,fieldnames=list(header),extrasaction="raise",lineterminator="\n");w.writeheader()
 for r in rows:
  if set(r)!=set(header):raise ValueError(f"generated CSV schema mismatch: {sorted(set(r)^set(header))}")
  w.writerow({k:cv(r[k]) for k in header})
 return b.getvalue()
def render_contract(rows:list[dict[str,Any]])->str:
 return render_csv(CONTRACT_H,[{**r,"required":canon(r["required"]),"observed":canon(r["observed"])} for r in rows])
def blocked(ns:argparse.Namespace,e:BaseException)->dict[str,Any]:
 return {"stage":STAGE,"decision":"STAGE196B2B6_BLOCKED_CONTRACT_FAILURE","recommended_next_stage":"STAGE196B2B6_REPAIR","blocking_reasons":[f"{type(e).__name__}: {e}"],"current_git_commit":ns.current_git_commit,"stage196b2b3p0_runtime_git_commit":ns.stage196b2b3p0_runtime_git_commit,"source_paths":{"stage196b2b5_analysis_json":str(ns.stage196b2b5_analysis_json.resolve()),"stage196b2b4_analysis_json":str(ns.stage196b2b4_analysis_json.resolve()),"stage196b2b3p0_run_root":str(ns.stage196b2b3p0_run_root.resolve())},"source_hashes":{},"source_closure":{},"candidate_feature_subsets":[],"policy_equivalence_classes":{},"signature_action_maps":[],"unique_deterministic_candidates":[],"safe_set_valued_candidates":[],"unsafe_ambiguous_candidates":[],"primary_policy_validation":[],"clean_dev_signature_support":{},"clean_dev_safety":[],"seed183_contrast":{},"nondominated_candidates":[],"decision_rule_evaluation":{"completed":False},"authorized_interpretation":"No scientific interpretation is authorized because a contract failed.","remaining_uncertainty":["Repair the failed contract."],"prohibited_claims":PROHIBITED_CLAIMS,"artifact_only":True,"training_performed":False,"model_loaded":False,"checkpoint_loaded":False,"promotion_authorized":False,"accuracy_used_for_authorization":False}
def payloads(a:dict[str,Any],t:dict[str,Any],g:list[dict[str,Any]])->dict[str,str]:
 return {OUTPUTS[0]:json.dumps(a,indent=2,sort_keys=True)+"\n",OUTPUTS[1]:report(a),OUTPUTS[2]:render_csv(CANDIDATE_H,t["candidates"]),OUTPUTS[3]:render_csv(SIGNATURE_H,t["signatures"]),OUTPUTS[4]:render_csv(PRIMARY_H,t["primary"]),OUTPUTS[5]:render_csv(AUDIT_H,t["audit"]),OUTPUTS[6]:render_csv(APP_H,t["apps"]),OUTPUTS[7]:render_csv(DOM_H,t["dominance"]),OUTPUTS[8]:render_contract(g)}
def atomic_write(output:Path,data:dict[str,str])->None:
 if output.exists() or set(data)!=set(OUTPUTS):raise RuntimeError("refusing overwrite or non-nine-file output")
 tmp=output.parent/f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp";tmp.mkdir(parents=False,exist_ok=False)
 try:
  for n in OUTPUTS:
   with (tmp/n).open("x",encoding="utf-8",newline="") as h:h.write(data[n]);h.flush();os.fsync(h.fileno())
  if sorted(x.name for x in tmp.iterdir())!=sorted(OUTPUTS):raise RuntimeError("staged output closure failed")
  os.replace(tmp,output)
 finally:
  if tmp.exists():shutil.rmtree(tmp)
def main()->int:
 ns=parse_args();g=[];tables={"candidates":[],"signatures":[],"primary":[],"audit":[],"apps":[],"dominance":[]}
 try:a,tables=analyze(ns,g)
 except Exception as e:
  a=blocked(ns,e)
  if not any(not r["passed"] for r in g):gate(g,"analysis","","unhandled_contract_failure",True,{"type":type(e).__name__,"message":str(e)},False,str(e),False)
  gate(g,"output","","exact_nine_output_closure",list(OUTPUTS),list(OUTPUTS),True,"",False)
 atomic_write(ns.output_dir.resolve(),payloads(a,tables,g));return 3 if a["decision"]=="STAGE196B2B6_BLOCKED_CONTRACT_FAILURE" else 0
if __name__=="__main__":raise SystemExit(main())
