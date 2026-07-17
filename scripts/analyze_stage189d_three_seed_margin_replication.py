#!/usr/bin/env python3
"""Analyze Stage189 paired runs and posthoc references; never load a model."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics, sys, traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SEEDS=(174,175,176); ARMS=("baseline","intervention")
LABELS=("REFUTE","NOT_ENTITLED","SUPPORT")
BLOCKED="STAGE189D_THREE_SEED_MARGIN_REPLICATION_BLOCKED"
NEGATIVE="STAGE189D_THREE_SEED_MARGIN_NEGATIVE_OR_REGRESSIVE"
NONSELECTIVE="STAGE189D_THREE_SEED_MARGIN_REPLICATED_BENEFICIAL_BUT_NONSELECTIVE"
PARTIAL="STAGE189D_THREE_SEED_MARGIN_PARTIAL_REPLICATION_NO_ADVANCE"
POSITIVE="STAGE189D_THREE_SEED_MARGIN_POSITIVE_SELECTIVE_REPLICATION"
READY="STAGE189A_THREE_SEED_MARGIN_REPLICATION_AND_POSTHOC_REFERENCE_SPEC_READY"
STATUS_COUNTS={"ELIGIBLE":605,"INELIGIBLE":716,"UNRESOLVED":119}
ALLOWED_DIFFS={"compatible_positive_margin_weight","controlled_integrity_sidecar_path",
 "expected_integrity_sidecar_semantic_sha256","output_json","output_predictions_json",
 "stage115_clean_dev_scalar_output_jsonl"}
AGG_KEYS=frozenset({"configured_weight","configured_margin_logit",
 "compatible_positive_margin_eligible_count","compatible_positive_margin_eligible_observation_count",
 "epoch_metrics","score_source","normalization"})

def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--stage189a-dir",type=Path,required=True)
    for seed in SEEDS:
        for arm in ARMS:
            p.add_argument(f"--seed{seed}-{arm}-run-dir",type=Path,required=True)
            p.add_argument(f"--seed{seed}-{arm}-posthoc",type=Path,required=True)
            p.add_argument(f"--seed{seed}-{arm}-posthoc-report",type=Path,required=True)
    p.add_argument("--stage182b-dir",type=Path,required=True); p.add_argument("--stage185a-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); return p.parse_args()

def read_json(path):
    with path.open("r",encoding="utf-8") as h:return json.load(h)
def read_jsonl(path):
    rows=[]
    with path.open("r",encoding="utf-8") as h:
        for n,line in enumerate(h,1):
            if not line.strip():continue
            value=json.loads(line)
            if not isinstance(value,dict):raise ValueError(f"{path}:{n} is not an object")
            rows.append(value)
    return rows
def read_csv(path):
    with path.open("r",newline="",encoding="utf-8-sig") as h:return list(csv.DictReader(h))
def file_sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()
def semantic_sidecar_sha(rows):
    semantic=[{k:r[k] for k in sorted(r) if k!="created_at"} for r in rows]
    payload=json.dumps(semantic,sort_keys=True,ensure_ascii=False,separators=(",",":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
def write_csv(path,headers,rows):
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=headers,extrasaction="ignore");w.writeheader()
        for row in rows:w.writerow({k:row.get(k) for k in headers})
def rid(row):
    value=row.get("row_id",row.get("id",row.get("stable_id")))
    if value is None and isinstance(row.get("raw_record"),dict):value=row["raw_record"].get("id")
    return str(value) if value is not None else None
def index_rows(rows,name):
    result={}
    for row in rows:
        key=rid(row)
        if key is None or key in result:raise ValueError(f"{name} has missing or duplicate row IDs")
        result[key]=row
    return result
def gold(row):
    for key in ("gold_label","gold_final_label","final_label"):
        if row.get(key) in LABELS:return str(row[key])
    raw=row.get("raw_record");return str(raw.get("final_label")) if isinstance(raw,dict) and raw.get("final_label") in LABELS else None
def prediction(row):
    for key in ("prediction","pred_final_label","pred_label","base_prediction"):
        if row.get(key) in LABELS:return str(row[key])
    return None
def finite(value):return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(float(value))
def flogit(row):return float(row["frame_logit"]) if finite(row.get("frame_logit")) else None
def fprob(row):
    value=row.get("frame_prob")
    return float(value) if finite(value) and 0.0<=float(value)<=1.0 else None
def argv_has(argv,option):
    return isinstance(argv,list) and any(t==option or (isinstance(t,str) and t.startswith(option+"=")) for t in argv)
def is_train_compatible(row):
    value=row.get("frame_compatible_label")
    return row.get("split")=="train" and type(value) is int and value==1
def zero_or_none(value):return value is None or (finite(value) and float(value)==0.0)
def aggregate_contract(aggregate,arm,expected_sidecar_sha):
    if not isinstance(aggregate,dict):return False,{"schema":"not_object"}
    sidecar=aggregate.get("sidecar_contract")
    sidecar=sidecar if isinstance(sidecar,dict) else {}
    observed={
      "enabled":aggregate.get("enabled"),"configured_weight":aggregate.get("configured_weight"),
      "configured_margin_logit":aggregate.get("configured_margin_logit"),
      "eligible_count":aggregate.get("compatible_positive_margin_eligible_count"),
      "eligible_observation_count":aggregate.get("compatible_positive_margin_eligible_observation_count"),
      "active_count":aggregate.get("compatible_positive_margin_active_count"),
      "active_rate":aggregate.get("compatible_positive_margin_active_rate"),
      "raw_loss":aggregate.get("compatible_positive_margin_loss_raw"),
      "weighted_loss":aggregate.get("compatible_positive_margin_loss_weighted"),
      "mean_eligible_frame_logit":aggregate.get("compatible_positive_margin_mean_eligible_frame_logit"),
      "zero_eligible_batch_count":aggregate.get("zero_eligible_batch_count"),
      "score_source":aggregate.get("score_source"),"normalization":aggregate.get("normalization"),
      "checkpoint_selection_unchanged":aggregate.get("checkpoint_selection_unchanged"),
      "epoch_metric_count":len(aggregate.get("epoch_metrics")) if isinstance(aggregate.get("epoch_metrics"),list) else None,
      "sidecar_accessed":sidecar.get("sidecar_accessed",aggregate.get("sidecar_accessed")),
      "sidecar_contract":sidecar}
    common=(observed["configured_margin_logit"]==0.0
      and observed["zero_eligible_batch_count"]==0
      and observed["score_source"]=='output["frame_logit"]'
      and observed["normalization"]=="eligible_row_mean")
    if arm=="baseline":
        passed=(common and observed["enabled"] is False and observed["configured_weight"]==0.0
          and observed["eligible_count"]==0 and observed["eligible_observation_count"]==0
          and observed["active_count"]==0 and zero_or_none(observed["raw_loss"])
          and zero_or_none(observed["weighted_loss"])
          and observed["sidecar_accessed"] in (None,False))
    else:
        passed=(common and observed["enabled"] is True and observed["configured_weight"]==0.05
          and observed["eligible_count"]==605 and observed["eligible_observation_count"]==12100
          and finite(observed["raw_loss"]) and finite(observed["weighted_loss"])
          and finite(observed["active_count"]) and observed["active_count"]>=0
          and finite(observed["active_rate"]) and 0.0<=observed["active_rate"]<=1.0
          and finite(observed["mean_eligible_frame_logit"])
          and observed["checkpoint_selection_unchanged"] is True
          and isinstance(aggregate.get("epoch_metrics"),list) and bool(aggregate["epoch_metrics"]))
        passed=passed and bool(sidecar) and (
          sidecar.get("eligible_rows")==605 and sidecar.get("eligible_pairs")==121
          and sidecar.get("eligible_families")==5
          and sidecar.get("observed_sidecar_semantic_sha256")==expected_sidecar_sha)
    return passed,observed

def metrics(rows):
    pairs=[(gold(r),prediction(r)) for r in rows]
    if not pairs or any(g not in LABELS or p not in LABELS for g,p in pairs):raise ValueError("invalid clean labels")
    c={g:{p:0 for p in LABELS} for g in LABELS}
    for g,p in pairs:c[g][p]+=1
    recalls={};f1=[]
    for label in LABELS:
        tp=c[label][label];fn=sum(c[label][p] for p in LABELS if p!=label);fp=sum(c[g][label] for g in LABELS if g!=label)
        r=tp/(tp+fn) if tp+fn else 0.0;pr=tp/(tp+fp) if tp+fp else 0.0;recalls[label]=r;f1.append(2*pr*r/(pr+r) if pr+r else 0.0)
    return {"accuracy":sum(g==p for g,p in pairs)/len(pairs),"macro_f1":statistics.fmean(f1),
      "support_recall":recalls["SUPPORT"],"refute_recall":recalls["REFUTE"],"not_entitled_recall":recalls["NOT_ENTITLED"],
      "false_not_entitled_total":sum(g!="NOT_ENTITLED" and p=="NOT_ENTITLED" for g,p in pairs),
      "false_entitlement_total":sum(g=="NOT_ENTITLED" and p!="NOT_ENTITLED" for g,p in pairs),
      "polarity_error_total":sum(g in ("REFUTE","SUPPORT") and p in ("REFUTE","SUPPORT") and g!=p for g,p in pairs),
      "prediction_counts":dict(Counter(p for _,p in pairs))}

def find_obj(value,key):
    if isinstance(value,dict):
        if key in value:return value
        for child in value.values():
            found=find_obj(child,key)
            if found is not None:return found
    elif isinstance(value,list):
        for child in value:
            found=find_obj(child,key)
            if found is not None:return found
    return None
def find_aggs(value,path="$"):
    out=[]
    if isinstance(value,dict):
        if AGG_KEYS.issubset(value):out.append((path,value))
        for key,child in value.items():out.extend(find_aggs(child,f"{path}.{key}"))
    elif isinstance(value,list):
        for i,child in enumerate(value):out.extend(find_aggs(child,f"{path}[{i}]"))
    return out
def unique_json(directory,predicate):
    found=[]
    for path in sorted(directory.rglob("*.json")) if directory.is_dir() else []:
        try:value=read_json(path)
        except (OSError,ValueError,json.JSONDecodeError):continue
        if isinstance(value,dict) and predicate(value,path):found.append((path,value))
    return found[0] if len(found)==1 else (None,{})
def load_run(directory):
    pp,prov=unique_json(directory,lambda v,p:isinstance(v.get("parsed_args"),dict) and "source_provenance" in v)
    rp,report=unique_json(directory,lambda v,p:"train" in p.name.lower() and find_obj(v,"best_dev_metrics") is not None)
    dp,preds=unique_json(directory,lambda v,p:"prediction" in p.name.lower() and isinstance(v.get("predictions"),list))
    sp=list(directory.rglob("clean_dev_scalars.jsonl")) if directory.is_dir() else []
    return {"provenance_path":pp,"provenance":prov,"report_path":rp,"report":report,"prediction_path":dp,
      "predictions":preds.get("predictions",[]) if dp else [],"scalar_path":sp[0] if len(sp)==1 else None,
      "scalars":read_jsonl(sp[0]) if len(sp)==1 else [],"aggregates":find_aggs(report)}
def epoch_metrics(rows):
    grouped={}
    for row in (rows if isinstance(rows,list) else []):
        if not isinstance(row,dict) or row.get("enabled") is not True or row.get("epoch") is None:continue
        x=grouped.setdefault(row["epoch"],{"epoch":row["epoch"],"eligible":0,"active":0,"hinge":0.0,"logit":0.0})
        x["eligible"]+=int(row.get("eligible_count") or 0);x["active"]+=int(row.get("active_count") or 0)
        x["hinge"]+=float(row.get("hinge_loss_sum") or 0);x["logit"]+=float(row.get("eligible_frame_logit_sum") or 0)
    out=[]
    for x in grouped.values():
        n=x["eligible"];out.append({"epoch":x["epoch"],"active_rate":x["active"]/n if n else None,
          "mean_frame_logit":x["logit"]/n if n else None,"raw_margin_loss":x["hinge"]/n if n else None,"active_count":x["active"]})
    return out
def agg_summary(values):return {"mean":statistics.fmean(values),"std":statistics.stdev(values),"positive_seed_count":sum(v>0 for v in values)}

def _main_impl():
    args=parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    blockers=[];gates=[];identity_rows=[];clean_rows=[];mechanism_rows=[];integrity_rows=[]
    selectivity_rows=[];cohort_rows=[];transition_rows=[];seed_results={}
    observed_split_contracts=[]
    def gate(name,required,observed,passed,category,reason):
        gates.append({"gate":name,"category":category,"required":json.dumps(required,sort_keys=True),
          "observed":json.dumps(observed,sort_keys=True),"passed":passed,"status":"pass" if passed else "fail",
          "blocking_reason":"" if passed else reason})
        if category=="identity_runtime" and not passed:blockers.append(f"{name}: {reason}")

    manifest=read_json(args.stage189a_dir/"stage189a_manifest_report.json")
    gate("stage189a_ready",READY,manifest.get("decision"),manifest.get("decision")==READY,"identity_runtime","Stage189-A not ready")
    commit=manifest.get("current_git_commit");trainer_sha=manifest.get("trainer_sha256");data_sha=manifest.get("dataset_sha256")
    sidecar_path=args.stage185a_dir/"stage185a_controlled_train_integrity_sidecar.jsonl"
    try:
        sidecar_rows=read_jsonl(sidecar_path);sidecar_index=index_rows(sidecar_rows,"authoritative sidecar")
        observed_sidecar_sha=semantic_sidecar_sha(sidecar_rows)
    except (OSError,ValueError,json.JSONDecodeError) as exc:
        sidecar_rows=[];sidecar_index={};observed_sidecar_sha=None;blockers.append(f"sidecar unavailable: {exc}")
    expected_sidecar_sha=manifest.get("sidecar_semantic_sha256")
    gate("authoritative_sidecar_semantic_sha",expected_sidecar_sha,observed_sidecar_sha,
      observed_sidecar_sha==expected_sidecar_sha,"identity_runtime","Stage185-A sidecar semantic SHA mismatch")
    full_train={i:r for i,r in sidecar_index.items() if r.get("split")=="train"}
    auth_train={i:r for i,r in sidecar_index.items() if is_train_compatible(r)}
    train_incompatible={i:r for i,r in full_train.items() if not is_train_compatible(r)}
    auth_counts=Counter(r.get("integrity_status") for r in auth_train.values())
    auth_eligible=[r for r in auth_train.values() if r.get("integrity_status")=="ELIGIBLE"]
    full_topology={"sidecar_rows":len(sidecar_index),"train_rows":len(full_train),
      "dev_rows":sum(r.get("split")=="dev" for r in sidecar_index.values()),
      "train_compatible_rows":len(auth_train),"train_incompatible_rows":len(train_incompatible)}
    gate("authoritative_full_sidecar_topology",
      {"sidecar_rows":3600,"train_rows":2880,"dev_rows":720,
       "train_compatible_rows":1440,"train_incompatible_rows":1440},
      full_topology,
      full_topology=={"sidecar_rows":3600,"train_rows":2880,"dev_rows":720,
       "train_compatible_rows":1440,"train_incompatible_rows":1440},
      "identity_runtime","Stage185-A full sidecar/split topology mismatch")
    compatible_topology={"rows":len(auth_train),**dict(auth_counts),
      "eligible_pairs":len({r.get("pair_id") for r in auth_eligible}),
      "eligible_families":len({r.get("family_contract_id") for r in auth_eligible})}
    gate("authoritative_train_compatible_topology",
      {"rows":1440,**STATUS_COUNTS,"eligible_pairs":121,"eligible_families":5},
      compatible_topology,
      len(auth_train)==1440 and all(auth_counts.get(k,0)==v for k,v in STATUS_COUNTS.items())
      and compatible_topology["eligible_pairs"]==121
      and compatible_topology["eligible_families"]==5,
      "identity_runtime","Stage185-A authoritative train-compatible topology mismatch")
    source=read_csv(args.stage182b_dir/"stage182b_candidate_localization.csv")
    controls=read_csv(args.stage182b_dir/"stage182b_matched_control_pairs.csv")
    compatible={r["row_id"] for r in source if r.get("native_error_direction")=="compatible_false_negative"}
    incompatible={r["row_id"] for r in source if r.get("native_error_direction")=="incompatible_false_positive"}
    control={r["control_row_id"] for r in controls if r.get("control_row_id")};failures=compatible|incompatible
    gate("stage182b_topology",[13,1,14],[len(compatible),len(incompatible),len(control)],
      len(compatible)==13 and len(incompatible)==1 and len(control)==14,"identity_runtime","Stage182-B topology mismatch")

    for seed in SEEDS:
        seed_blockers=len(blockers);runs={};posthoc={}
        for arm in ARMS:
            run=load_run(getattr(args,f"seed{seed}_{arm}_run_dir"));runs[arm]=run
            ok=all(run[k] is not None for k in ("provenance_path","report_path","prediction_path","scalar_path"))
            gate(f"seed{seed}_{arm}_artifacts",True,ok,ok,"identity_runtime","run artifact missing or ambiguous")
            prov=run["provenance"];parsed=prov.get("parsed_args") or {};run_commit=(prov.get("source_provenance") or {}).get("git_commit")
            resolved_runtime=prov.get("resolved_runtime_config") or {}
            provenance_split_contract=prov.get("split_seed_contract") or {}
            gate(f"seed{seed}_{arm}_run_completed","completed",prov.get("status"),
              prov.get("status")=="completed","identity_runtime","run provenance is not completed")
            run_trainer_sha=(prov.get("source_provenance") or {}).get("trainer_sha256")
            run_data=((prov.get("data_provenance") or {}).get("main_data") or {}).get("sha256")
            arm_manifest=read_json(args.stage189a_dir/f"stage189a_seed{seed}_{arm}_manifest.json")
            manifest_split_ok=(
              arm_manifest.get("training_seed")==seed
              and arm_manifest.get("split_seed")==174
              and arm_manifest.get("split_policy")=="fixed_across_replication_seeds"
              and (arm_manifest.get("parsed_argv_contract") or {}).get("split_seed")=="174")
            gate(f"seed{seed}_{arm}_manifest_split_identity",True,manifest_split_ok,
              manifest_split_ok,"identity_runtime","arm manifest split-seed identity mismatch")
            raw_argv=prov.get("raw_sys_argv") or []
            expected_runtime={"seed":seed,"git_commit":commit,"trainer_sha256":trainer_sha,
              "dataset_sha256":data_sha,"architecture":"v6b_minimal","backbone":"mamba",
              "model_name":"state-spaces/mamba-130m-hf","device":"cuda","epochs":20,
              "configured_split_seed":174,"resolved_split_seed":174,
              "split_seed_explicit":True,
              "split_policy":"fixed_explicit_split_seed",
              "select_metric":"final_macro_f1","margin_logit":0.0,
              "stage174c_mode":"off","stage174c_weight":0.0,"stage175b_mode":"off",
              "stage175b_weight":0.0,"stage177c_mode":"off","stage177c_weight":0.0,
              "selected_checkpoint_enabled":True}
            observed_runtime={"seed":parsed.get("seed"),"git_commit":run_commit,
              "trainer_sha256":run_trainer_sha,"dataset_sha256":run_data,
              "architecture":parsed.get("architecture"),"backbone":parsed.get("backbone"),
              "model_name":parsed.get("model_name"),"device":parsed.get("device"),"epochs":parsed.get("epochs"),
              "configured_split_seed":parsed.get("split_seed"),
              "resolved_split_seed":resolved_runtime.get("resolved_split_seed"),
              "split_seed_explicit":resolved_runtime.get("split_seed_explicit"),
              "split_policy":resolved_runtime.get("split_policy"),
              "select_metric":parsed.get("select_metric"),
              "margin_logit":parsed.get("compatible_positive_margin_logit"),
              "stage174c_mode":parsed.get("stage174c_clean_pairwise_mode"),
              "stage174c_weight":parsed.get("stage174c_clean_pairwise_weight"),
              "stage175b_mode":parsed.get("stage175b_support_anchor_mode"),
              "stage175b_weight":parsed.get("stage175b_support_anchor_weight"),
              "stage177c_mode":parsed.get("stage177c_frame_pairwise_mode"),
              "stage177c_weight":parsed.get("stage177c_frame_pairwise_weight"),
              "selected_checkpoint_enabled":bool(parsed.get("save_selected_checkpoint"))}
            observed_split_contracts.append({
              "seed":seed,"arm":arm,"training_seed":parsed.get("seed"),
              "configured_split_seed":parsed.get("split_seed"),
              "resolved_split_seed":resolved_runtime.get("resolved_split_seed"),
              "split_seed_explicit":resolved_runtime.get("split_seed_explicit"),
              "split_policy":resolved_runtime.get("split_policy"),
              "provenance_training_seed":provenance_split_contract.get("training_seed"),
              "provenance_configured_split_seed":provenance_split_contract.get("configured_split_seed"),
              "provenance_resolved_split_seed":provenance_split_contract.get("resolved_split_seed"),
              "provenance_split_seed_explicit":provenance_split_contract.get("split_seed_explicit"),
              "provenance_split_policy":provenance_split_contract.get("split_policy"),
              "clean_main_train_rows":provenance_split_contract.get("clean_main_train_rows"),
              "clean_main_dev_rows":provenance_split_contract.get("clean_main_dev_rows")})
            identity_ok=observed_runtime==expected_runtime
            gate(f"seed{seed}_{arm}_runtime_identity",expected_runtime,observed_runtime,
              identity_ok,"identity_runtime","run provenance identity/config mismatch")
            provenance_split_ok=(
              provenance_split_contract.get("training_seed")==seed
              and provenance_split_contract.get("configured_split_seed")==174
              and provenance_split_contract.get("resolved_split_seed")==174
              and provenance_split_contract.get("split_seed_explicit") is True
              and provenance_split_contract.get("split_policy")=="fixed_explicit_split_seed"
              and provenance_split_contract.get("clean_main_train_rows")==2880
              and provenance_split_contract.get("clean_main_dev_rows")==720)
            gate(f"seed{seed}_{arm}_provenance_split_contract",True,
              provenance_split_contract,provenance_split_ok,"identity_runtime",
              "run provenance fixed split-seed or 2880/720 row-count contract mismatch")
            sidecar_option=parsed.get("controlled_integrity_sidecar_path")
            sidecar_suffix=str(sidecar_option or "").replace(chr(92),"/")
            arm_ok=(parsed.get("compatible_positive_margin_weight")==(0.0 if arm=="baseline" else 0.05))
            if arm=="baseline":
                arm_ok=arm_ok and sidecar_option is None and parsed.get("expected_integrity_sidecar_semantic_sha256") is None
                arm_ok=arm_ok and not argv_has(raw_argv,"--controlled-integrity-sidecar-path")
                arm_ok=arm_ok and not argv_has(raw_argv,"--expected-integrity-sidecar-semantic-sha256")
            else:
                arm_ok=arm_ok and sidecar_suffix.endswith(
                  "reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/"
                  "stage185a_controlled_train_integrity_sidecar.jsonl")
                arm_ok=arm_ok and parsed.get("expected_integrity_sidecar_semantic_sha256")==expected_sidecar_sha
            gate(f"seed{seed}_{arm}_margin_sidecar_contract",True,arm_ok,arm_ok,
              "identity_runtime","arm margin/sidecar contract mismatch")
            gate(f"seed{seed}_{arm}_aggregate",1,len(run["aggregates"]),len(run["aggregates"])==1,
              "identity_runtime","margin aggregate missing or ambiguous")
            if len(run["aggregates"])==1:
                aggregate_value=run["aggregates"][0][1]
                aggregate_ok,aggregate_observed=aggregate_contract(
                  aggregate_value,arm,expected_sidecar_sha)
                gate(f"seed{seed}_{arm}_aggregate_contract",True,aggregate_observed,
                  aggregate_ok,"identity_runtime",f"{arm} completed-run margin aggregate contract mismatch")
                aggregate_sidecar=aggregate_value.get("sidecar_contract")
                aggregate_sidecar=aggregate_sidecar if isinstance(aggregate_sidecar,dict) else {}
                sidecar_aggregate_ok=(
                  aggregate_sidecar.get("sidecar_accessed") in (None,False)
                  if arm=="baseline" else (
                    aggregate_sidecar.get("eligible_rows")==605
                    and aggregate_sidecar.get("eligible_pairs")==121
                    and aggregate_sidecar.get("eligible_families")==5
                    and aggregate_sidecar.get("observed_sidecar_semantic_sha256")==expected_sidecar_sha))
                gate(f"seed{seed}_{arm}_aggregate_sidecar_contract",True,
                  aggregate_sidecar,sidecar_aggregate_ok,"identity_runtime",
                  f"{arm} aggregate sidecar contract mismatch")
            posthoc_path=getattr(args,f"seed{seed}_{arm}_posthoc").resolve()
            posthoc_report=read_json(getattr(args,f"seed{seed}_{arm}_posthoc_report"))
            rows=read_jsonl(posthoc_path)
            try:posthoc[arm]=index_rows(rows,f"seed{seed} {arm} posthoc")
            except ValueError as exc:posthoc[arm]={};blockers.append(str(exc))
            identity=rows[0].get("checkpoint_identity") if rows else {}; identity=identity or {}
            unique_identity=len({json.dumps(r.get("checkpoint_identity"),sort_keys=True) for r in rows})==1
            provenance_checkpoint=((prov.get("finalization") or {}).get("selected_checkpoint") or {})
            checkpoint_path=Path(provenance_checkpoint.get("path",""))
            actual_checkpoint_sha=file_sha(checkpoint_path) if checkpoint_path.is_file() else None
            checkpoint_artifact_ok=(actual_checkpoint_sha is not None
              and provenance_checkpoint.get("sha256")==actual_checkpoint_sha
              and identity.get("checkpoint_sha256")==actual_checkpoint_sha)
            gate(f"seed{seed}_{arm}_selected_checkpoint_artifact",True,checkpoint_artifact_ok,
              checkpoint_artifact_ok,"identity_runtime","selected checkpoint path/SHA is missing or mismatched")
            report_contract=(posthoc_report.get("decision")=="STAGE189C_POSTHOC_MARGIN_REFERENCE_EXPORTED"
              and posthoc_report.get("blocking_reasons")==[] and posthoc_report.get("row_count")==1440
              and Path(posthoc_report.get("output_jsonl","")).resolve()==posthoc_path
              and posthoc_report.get("output_jsonl_sha256")==file_sha(posthoc_path)
              and posthoc_report.get("seed")==seed and posthoc_report.get("arm")==arm
              and posthoc_report.get("training_seed")==seed
              and posthoc_report.get("split_seed")==174
              and posthoc_report.get("split_policy")=="fixed_explicit_split_seed"
              and posthoc_report.get("selected_epoch")==identity.get("selected_epoch")
              and posthoc_report.get("checkpoint_sha256")==identity.get("checkpoint_sha256")
              and posthoc_report.get("trainer_sha256")==trainer_sha
              and posthoc_report.get("git_commit")==commit
              and posthoc_report.get("dataset_sha256")==data_sha
              and posthoc_report.get("sidecar_semantic_sha256")==expected_sidecar_sha
              and posthoc_report.get("checkpoint_helper_sha256")==manifest.get("checkpoint_helper_sha256")
              and posthoc_report.get("score_source")=='direct output["frame_logit"]'
              and posthoc_report.get("checkpoint_identity")==identity)
            gate(f"seed{seed}_{arm}_stage189c_report",True,report_contract,report_contract,
              "identity_runtime","Stage189-C report/JSONL identity mismatch")
            posthoc_identity=(len(rows)==1440 and unique_identity and rows[0].get("seed")==seed and rows[0].get("arm")==arm
              and identity.get("git_commit")==commit and identity.get("trainer_sha256")==trainer_sha
              and identity.get("dataset_sha256")==data_sha and identity.get("sidecar_semantic_sha256")==expected_sidecar_sha
              and identity.get("architecture")=="v6b_minimal" and identity.get("backbone")=="mamba"
              and identity.get("model_name")=="state-spaces/mamba-130m-hf"
              and identity.get("training_seed")==seed and identity.get("split_seed")==174
              and identity.get("split_policy")=="fixed_explicit_split_seed"
              and all(r.get("selected_epoch")==identity.get("selected_epoch") for r in rows)
              and all(r.get("checkpoint_identity")==identity for r in rows))
            gate(f"seed{seed}_{arm}_posthoc_identity",[seed,arm,commit,trainer_sha],
              [rows[0].get("seed") if rows else None,rows[0].get("arm") if rows else None,identity.get("git_commit"),identity.get("trainer_sha256")],
              posthoc_identity,"identity_runtime","posthoc checkpoint identity mismatch")
            direct=len(rows)==1440 and all(flogit(r) is not None and fprob(r) is not None
              and r.get("score_source")== 'direct output["frame_logit"]' for r in rows)
            gate(f"seed{seed}_{arm}_direct_frame_logit",1440,len(rows),direct,"identity_runtime","direct frame_logit incomplete")

        ba=runs["baseline"]["provenance"].get("parsed_args") or {};ia=runs["intervention"]["provenance"].get("parsed_args") or {}
        forbidden=sorted(k for k in set(ba)|set(ia) if ba.get(k)!=ia.get(k) and k not in ALLOWED_DIFFS)
        gate(f"seed{seed}_paired_argv",[],forbidden,not forbidden,"identity_runtime","common argv mismatch")
        try:
            bp=index_rows(runs["baseline"]["predictions"],"baseline predictions");ip=index_rows(runs["intervention"]["predictions"],"intervention predictions")
            bs=index_rows(runs["baseline"]["scalars"],"baseline scalars");ins=index_rows(runs["intervention"]["scalars"],"intervention scalars")
        except ValueError as exc:bp={};ip={};bs={};ins={};blockers.append(str(exc))
        clean_match=len(bp)==len(ip)==len(bs)==len(ins)==720 and set(bp)==set(ip)==set(bs)==set(ins)
        gate(f"seed{seed}_clean_pairing",{"rows":720,"join":"exact"},[len(bp),len(ip),len(bs),len(ins)],clean_match,"identity_runtime","clean row IDs mismatch")
        gold_match=clean_match and all(gold(bp[i]) in LABELS
          and gold(bp[i])==gold(ip[i])==gold(bs[i])==gold(ins[i]) for i in bp)
        prediction_match=clean_match and all(
          prediction(bp[i]) in LABELS and prediction(ip[i]) in LABELS
          and prediction(bp[i])==prediction(bs[i])
          and prediction(ip[i])==prediction(ins[i]) for i in bp)
        direct_clean=clean_match and all(flogit(bs[i]) is not None and flogit(ins[i]) is not None
          and fprob(bs[i]) is not None and fprob(ins[i]) is not None
          and bs[i].get("score_source")=='direct output["frame_logit"]'
          and ins[i].get("score_source")=='direct output["frame_logit"]' for i in bs)
        gate(f"seed{seed}_clean_gold_identity",True,gold_match,gold_match,"identity_runtime","clean gold identity mismatch")
        gate(f"seed{seed}_clean_prediction_identity",True,prediction_match,prediction_match,
          "identity_runtime","prediction JSON/scalar prediction mismatch")
        gate(f"seed{seed}_clean_direct_frame_logit",len(bs),sum(flogit(r) is not None for r in bs.values()),
          direct_clean,"identity_runtime","clean scalar direct frame_logit/frame_prob contract incomplete")
        post_match=bool(posthoc.get("baseline")) and set(posthoc["baseline"])==set(posthoc.get("intervention",{}))
        metadata_match=post_match and all(posthoc["baseline"][i].get(k)==posthoc["intervention"][i].get(k)
          for i in posthoc["baseline"] for k in ("integrity_status","pair_id","family","gold_label"))
        counts=Counter(r.get("integrity_status") for r in posthoc.get("baseline",{}).values())
        authoritative_match=post_match and set(posthoc["baseline"])==set(auth_train) and all(
          all(posthoc[arm][i].get("integrity_status")==auth_train[i].get("integrity_status")
            and posthoc[arm][i].get("pair_id")==auth_train[i].get("pair_id")
            and posthoc[arm][i].get("family")==auth_train[i].get("family_contract_id")
            for arm in ARMS) for i in auth_train)
        row_contract=post_match and all(
          r.get("gold_label") in LABELS and r.get("prediction") in LABELS
          and flogit(r) is not None and fprob(r) is not None
          and r.get("integrity_status") in STATUS_COUNTS
          and r.get("eligible_boolean")==(r.get("integrity_status")=="ELIGIBLE")
          and isinstance(r.get("selected_epoch"),int)
          and isinstance(r.get("checkpoint_identity"),dict)
          and r.get("selected_epoch")==r["checkpoint_identity"].get("selected_epoch")
          for arm in ARMS for r in posthoc[arm].values())
        eligible_rows=[r for r in posthoc.get("baseline",{}).values() if r.get("integrity_status")=="ELIGIBLE"]
        eligible_pairs=len({r.get("pair_id") for r in eligible_rows})
        eligible_families=len({r.get("family") for r in eligible_rows})
        topology=(post_match and metadata_match and authoritative_match and row_contract
          and len(posthoc["baseline"])==1440 and all(counts.get(k,0)==v for k,v in STATUS_COUNTS.items())
          and eligible_pairs==121 and eligible_families==5)
        gate(f"seed{seed}_posthoc_pairing",{"rows":1440,**STATUS_COUNTS,"eligible_pairs":121,"eligible_families":5},
          {"rows":len(posthoc.get("baseline",{})),**dict(counts),"eligible_pairs":eligible_pairs,"eligible_families":eligible_families},
          topology,"identity_runtime","posthoc IDs/status metadata mismatch")
        identity_rows.append({"seed":seed,"baseline_commit":((runs["baseline"]["provenance"].get("source_provenance") or {}).get("git_commit")),
          "intervention_commit":((runs["intervention"]["provenance"].get("source_provenance") or {}).get("git_commit")),
          "trainer_sha256":trainer_sha,"dataset_sha256":data_sha,"clean_row_ids_match":clean_match,
          "posthoc_row_ids_match":post_match,"integrity_metadata_match":metadata_match})
        if len(blockers)>seed_blockers:
            seed_results[seed]={"incomplete":True};continue

        bm=metrics(list(bp.values()));im=metrics(list(ip.values()))
        keys=("accuracy","macro_f1","support_recall","refute_recall","not_entitled_recall","false_not_entitled_total","false_entitlement_total","polarity_error_total")
        delta={k:im[k]-bm[k] for k in keys};corrected=0;harmed=0
        for i in sorted(bp):
            g=gold(bp[i]);before=prediction(bp[i]);after=prediction(ip[i])
            trans="stable_correct" if before==after==g else "corrected" if before!=g and after==g else "newly_harmed" if before==g and after!=g else "other"
            corrected+=trans=="corrected";harmed+=trans=="newly_harmed"
            transition_rows.append({"seed":seed,"row_id":i,"gold":g,"baseline_prediction":before,"intervention_prediction":after,"transition":trans})
        hard={"macro_f1":delta["macro_f1"]>=-0.01,"support_recall":delta["support_recall"]>=-0.02,
          "false_entitlement":delta["false_entitlement_total"]<=2,"polarity":delta["polarity_error_total"]<=0,
          "no_collapse":all(im["prediction_counts"].get(label,0)>0 for label in LABELS)}
        clean_rows.append({"seed":seed,**{f"baseline_{k}":v for k,v in bm.items() if k!="prediction_counts"},
          **{f"intervention_{k}":v for k,v in im.items() if k!="prediction_counts"},**{f"delta_{k}":v for k,v in delta.items()},
          "baseline_prediction_counts":json.dumps(bm["prediction_counts"],sort_keys=True),
          "intervention_prediction_counts":json.dumps(im["prediction_counts"],sort_keys=True),"corrected":corrected,
          "newly_harmed":harmed,"hard_guardrails_pass":all(hard.values())})
        for name,passed in hard.items():gate(f"seed{seed}_clean_{name}",True,passed,passed,"clean_guardrail","hard guardrail failed")

        aggregate=runs["intervention"]["aggregates"][0][1];epochs=epoch_metrics(aggregate.get("epoch_metrics"))
        report_run=find_obj(runs["intervention"]["report"],"best_dev_metrics") or {};selected_epoch=report_run.get("best_epoch")
        selected=next((r for r in epochs if r.get("epoch")==selected_epoch),{})
        epoch_ok=bool(epochs) and bool(selected) and all(finite(r.get(k)) for r in (epochs[0],selected,epochs[-1]) for k in ("active_rate","mean_frame_logit","raw_margin_loss","active_count"))
        gate(f"seed{seed}_epoch_mechanism_complete",selected_epoch,[r.get("epoch") for r in epochs],epoch_ok,
          "identity_runtime","first/selected/final epoch mechanism metrics incomplete")
        if not epoch_ok:seed_results[seed]={"incomplete":True};continue
        first=epochs[0];final=epochs[-1]
        mechanism_rows.append({"seed":seed,"eligible_observation_count":aggregate.get("compatible_positive_margin_eligible_observation_count"),
          "zero_eligible_batch_count":aggregate.get("zero_eligible_batch_count"),**{f"first_{k}":v for k,v in first.items()},
          **{f"selected_{k}":v for k,v in selected.items()},**{f"final_{k}":v for k,v in final.items()}})

        status_stats={}
        for status,expected in STATUS_COUNTS.items():
            ids=[i for i,r in posthoc["baseline"].items() if r.get("integrity_status")==status]
            values=[flogit(posthoc["intervention"][i])-flogit(posthoc["baseline"][i]) for i in ids]
            base_mean=statistics.fmean(flogit(posthoc["baseline"][i]) for i in ids)
            int_mean=statistics.fmean(flogit(posthoc["intervention"][i]) for i in ids)
            corr=sum(prediction(posthoc["baseline"][i])!=gold(posthoc["baseline"][i]) and prediction(posthoc["intervention"][i])==gold(posthoc["intervention"][i]) for i in ids)
            harm=sum(prediction(posthoc["baseline"][i])==gold(posthoc["baseline"][i]) and prediction(posthoc["intervention"][i])!=gold(posthoc["intervention"][i]) for i in ids)
            status_stats[status]={"mean_delta":statistics.fmean(values),"median_delta":statistics.median(values),
              "positive_fraction":sum(v>0 for v in values)/len(values),"baseline_mean":base_mean,
              "intervention_mean":int_mean,"difference_in_means":int_mean-base_mean,"corrected":corr,"newly_harmed":harm}
            integrity_rows.append({"seed":seed,"integrity_status":status,"row_count":len(ids),**status_stats[status]})
        clean_delta={i:flogit(ins[i])-flogit(bs[i]) for i in bs}
        cohort_stats={}
        for name,ids in (("compatible_fn",compatible),("incompatible_fp",incompatible),("matched_controls",control),("clean_model_failures",failures)):
            values=[clean_delta[i] for i in ids if i in clean_delta]
            corr=sum(prediction(bp[i])!=gold(bp[i]) and prediction(ip[i])==gold(ip[i]) for i in ids if i in bp)
            harm=sum(prediction(bp[i])==gold(bp[i]) and prediction(ip[i])!=gold(ip[i]) for i in ids if i in bp)
            cohort_stats[name]={"mean_delta":statistics.fmean(values) if values else None,
              "median_delta":statistics.median(values) if values else None,"positive_count":sum(v>0 for v in values),
              "corrected":corr,"newly_harmed":harm,"matched_rows":len(values)}
            cohort_rows.append({"seed":seed,"cohort":name,"expected_rows":len(ids),"matched_rows":len(values),
              **cohort_stats[name],"evidence_status":"prior_selected_internal_diagnostic"})
            gate(f"seed{seed}_{name}_exact_rows",len(ids),len(values),len(values)==len(ids),"identity_runtime","cohort row-ID join incomplete")
        cohort_join_ok=all(
          cohort_stats.get(name,{}).get("mean_delta") is not None
          and cohort_stats.get(name,{}).get("median_delta") is not None
          and cohort_stats.get(name,{}).get("matched_rows",len(ids))==len(ids)
          for name,ids in (("compatible_fn",compatible),("incompatible_fp",incompatible),
            ("matched_controls",control),("clean_model_failures",failures)))
        if not cohort_join_ok:
            gate(f"seed{seed}_cohort_statistics_ready",True,False,False,
              "identity_runtime","cohort exact join or finite statistics input is incomplete")
            seed_results[seed]={"incomplete":True}
            continue
        e=status_stats["ELIGIBLE"];ine=status_stats["INELIGIBLE"];ctl=cohort_stats["matched_controls"];cfn=cohort_stats["compatible_fn"]
        sel={"eligible_mean_minus_ineligible_mean":e["mean_delta"]-ine["mean_delta"],
          "eligible_median_minus_ineligible_median":e["median_delta"]-ine["median_delta"],
          "eligible_mean_minus_matched_control_mean":e["mean_delta"]-ctl["mean_delta"],
          "compatible_fn_mean_minus_matched_control_mean":cfn["mean_delta"]-ctl["mean_delta"],
          "compatible_fn_median_minus_matched_control_median":cfn["median_delta"]-ctl["median_delta"]}
        selectivity_rows.append({"seed":seed,**sel,**{f"{k}_positive":v>0 for k,v in sel.items()}})
        mech={"eligible_mean_delta_positive":e["mean_delta"]>0,"eligible_median_delta_positive":e["median_delta"]>0,
          "eligible_positive_fraction_at_least_080":e["positive_fraction"]>=0.80,
          "selected_and_final_active_rate_decreased":finite(first.get("active_rate"))
            and finite(selected.get("active_rate")) and finite(final.get("active_rate"))
            and selected["active_rate"]<first["active_rate"] and final["active_rate"]<first["active_rate"]}
        for name,passed in mech.items():gate(f"seed{seed}_mechanism_{name}",True,passed,passed,"mechanism","mechanism replication failed")
        seed_results[seed]={"clean_deltas":delta,"hard":hard,"mechanism":mech,"selectivity":sel,
          "critical_newly_harmed":cfn["newly_harmed"]+cohort_stats["incompatible_fp"]["newly_harmed"]}

    six_run_split_contract=(
      len(observed_split_contracts)==6
      and {row.get("training_seed") for row in observed_split_contracts}==set(SEEDS)
      and all(row.get("configured_split_seed")==174
        and row.get("resolved_split_seed")==174
        and row.get("split_seed_explicit") is True
        and row.get("split_policy")=="fixed_explicit_split_seed"
        and row.get("provenance_training_seed")==row.get("training_seed")
        and row.get("provenance_configured_split_seed")==174
        and row.get("provenance_resolved_split_seed")==174
        and row.get("provenance_split_seed_explicit") is True
        and row.get("provenance_split_policy")=="fixed_explicit_split_seed"
        and row.get("clean_main_train_rows")==2880
        and row.get("clean_main_dev_rows")==720
        for row in observed_split_contracts)
      and all(
        {row.get("configured_split_seed") for row in observed_split_contracts
          if row.get("seed")==seed}=={174}
        for seed in SEEDS))
    gate("six_run_fixed_split_seed_identity",True,observed_split_contracts,
      six_run_split_contract,"identity_runtime",
      "all six runs must use fixed explicit split seed 174")
    complete=len(seed_results)==3 and all("incomplete" not in seed_results[s] for s in SEEDS)
    aggregate_rows=[];aggregate_clean={};mechanism_pass={};selectivity_counts={};critical_harm=None
    if complete:
        macro=[seed_results[s]["clean_deltas"]["macro_f1"] for s in SEEDS]
        support=[seed_results[s]["clean_deltas"]["support_recall"] for s in SEEDS]
        false_ent=[seed_results[s]["clean_deltas"]["false_entitlement_total"] for s in SEEDS]
        aggregate_clean={"mean_macro_f1_nonnegative":statistics.fmean(macro)>=0,
          "mean_support_recall_nonnegative":statistics.fmean(support)>=0,
          "mean_false_entitlement_nonpositive":statistics.fmean(false_ent)<=0}
        mechanism_pass={s:all(seed_results[s]["mechanism"].values()) for s in SEEDS}
        selectivity_counts={
          "eligible_mean_above_ineligible":sum(seed_results[s]["selectivity"]["eligible_mean_minus_ineligible_mean"]>0 for s in SEEDS),
          "eligible_mean_above_matched_control":sum(seed_results[s]["selectivity"]["eligible_mean_minus_matched_control_mean"]>0 for s in SEEDS),
          "compatible_fn_median_above_matched_control":sum(seed_results[s]["selectivity"]["compatible_fn_median_minus_matched_control_median"]>0 for s in SEEDS)}
        hard_all=all(all(seed_results[s]["hard"].values()) for s in SEEDS);selective=all(v>=2 for v in selectivity_counts.values())
        critical_harm=sum(seed_results[s]["critical_newly_harmed"] for s in SEEDS)
        aggregate_rows=[{"metric":"macro_f1_delta",**agg_summary(macro)},{"metric":"support_recall_delta",**agg_summary(support)},
          {"metric":"false_entitlement_delta",**agg_summary(false_ent)}]
        for key in seed_results[174]["selectivity"]:
            aggregate_rows.append({"metric":key,**agg_summary([seed_results[s]["selectivity"][key] for s in SEEDS])})
    else:hard_all=False;selective=False;blockers.append("incomplete seed prevented aggregate decision")
    mechanism_failures=sum(not v for v in mechanism_pass.values()) if mechanism_pass else 3
    if blockers:decision=BLOCKED
    elif (not hard_all or not all(aggregate_clean.values()) or mechanism_failures>=2
          or critical_harm is None or critical_harm>0):decision=NEGATIVE
    elif mechanism_failures==1:decision=PARTIAL
    elif mechanism_failures==0 and selective:decision=POSITIVE
    else:decision=NONSELECTIVE
    report={"stage":"Stage189-D","decision":decision,"blocking_reasons":blockers,"seed_results":seed_results,
      "aggregate_clean_direction":aggregate_clean,"mechanism_seed_pass":mechanism_pass,
      "mechanism_failure_count":mechanism_failures,
      "six_run_split_seed_contract":observed_split_contracts,
      "fixed_split_seed_identity_passed":six_run_split_contract,
      "selectivity_pass_counts":selectivity_counts,"selectivity_all_pass":selective,
      "critical_stage182b_newly_harmed":critical_harm,"matched_controls_independent_evaluation":False,
      "posthoc_training_rows_generalization_evidence":False}
    write_json(output/"stage189d_three_seed_analysis_report.json",report)
    write_csv(output/"stage189d_identity_audit.csv",["seed","baseline_commit","intervention_commit","trainer_sha256","dataset_sha256","clean_row_ids_match","posthoc_row_ids_match","integrity_metadata_match"],identity_rows)
    write_csv(output/"stage189d_seed_clean_metrics.csv",list(clean_rows[0]) if clean_rows else ["seed"],clean_rows)
    write_csv(output/"stage189d_seed_mechanism_metrics.csv",list(mechanism_rows[0]) if mechanism_rows else ["seed"],mechanism_rows)
    write_csv(output/"stage189d_posthoc_integrity_status_diagnostics.csv",list(integrity_rows[0]) if integrity_rows else ["seed"],integrity_rows)
    write_csv(output/"stage189d_selectivity_diagnostics.csv",list(selectivity_rows[0]) if selectivity_rows else ["seed"],selectivity_rows)
    write_csv(output/"stage189d_stage182b_cohort_diagnostics.csv",list(cohort_rows[0]) if cohort_rows else ["seed"],cohort_rows)
    write_csv(output/"stage189d_prediction_transitions.csv",["seed","row_id","gold","baseline_prediction","intervention_prediction","transition"],transition_rows)
    write_csv(output/"stage189d_precommitted_gate.csv",["gate","category","required","observed","passed","status","blocking_reason"],gates)
    write_csv(output/"stage189d_aggregate_summary.csv",["metric","mean","std","positive_seed_count"],aggregate_rows)
    md=f"""# Stage189-D three-seed paired replication analysis\n\n**Decision:** `{decision}`\n\n- Identity/runtime blockers: {len(blockers)}\n- Hard clean guardrails all seeds: {hard_all}\n- Mechanism replicated seeds: {sum(mechanism_pass.values()) if mechanism_pass else 0}/3\n- Aggregate clean direction: {all(aggregate_clean.values()) if aggregate_clean else False}\n- Selectivity 2/3 tests all pass: {selective}\n- Newly harmed critical Stage182-B rows: {critical_harm}\n\nMatched controls are prior-selected internal diagnostics, not independent evaluation. Posthoc training rows are mechanism diagnostics, not generalization evidence.\n\n## Blocking reasons\n\n{chr(10).join('- '+x for x in blockers) if blockers else '- None.'}\n"""
    (output/"stage189d_three_seed_analysis_report.md").write_text(md,encoding="utf-8")
    return 2 if decision==BLOCKED else 0

def _output_dir_from_argv():
    for index,token in enumerate(sys.argv[1:]):
        if token.startswith("--output-dir="):
            return Path(token.split("=",1)[1]).resolve()
        if token=="--output-dir" and index+2<=len(sys.argv[1:]):
            return Path(sys.argv[1:][index+1]).resolve()
    return None

def main():
    try:
        return _main_impl()
    except BaseException as exc:
        output=_output_dir_from_argv()
        if output is None:
            raise
        output.mkdir(parents=True,exist_ok=True)
        detail=f"{type(exc).__name__}: {exc}; location={traceback.format_exc().strip()}"
        report={"stage":"Stage189-D","decision":BLOCKED,"blocking_reasons":[detail],
          "fail_closed_exception":True,"seed_results":{},"aggregate_clean_direction":{},
          "mechanism_seed_pass":{},"selectivity_pass_counts":{},
          "six_run_split_seed_contract":[],"fixed_split_seed_identity_passed":False,
          "matched_controls_independent_evaluation":False,
          "posthoc_training_rows_generalization_evidence":False}
        write_json(output/"stage189d_three_seed_analysis_report.json",report)
        md=("# Stage189-D three-seed paired replication analysis\n\n"
          f"**Decision:** `{BLOCKED}`\n\n"
          "Fail-closed runtime exception:\n\n"
          f"- {detail}\n")
        (output/"stage189d_three_seed_analysis_report.md").write_text(md,encoding="utf-8")
        write_csv(output/"stage189d_precommitted_gate.csv",
          ["gate","category","required","observed","passed","status","blocking_reason"],
          [{"gate":"top_level_runtime_safety","category":"identity_runtime",
            "required":json.dumps("no exception"),"observed":json.dumps(detail),
            "passed":False,"status":"fail","blocking_reason":detail}])
        return 2

if __name__=="__main__":raise SystemExit(main())
