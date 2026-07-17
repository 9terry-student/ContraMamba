#!/usr/bin/env python3
"""Pair six Stage190-B reports and apply the precommitted Stage190-C taxonomy."""
from __future__ import annotations
import argparse, csv, json, math, traceback
from pathlib import Path
from typing import Any, Iterable

SEEDS=(174,175,176); ARMS=("baseline","intervention")
EXPORTED="STAGE190B_GRADIENT_DIAGNOSTIC_EXPORTED"
BLOCKED="STAGE190C_GRADIENT_DIAGNOSTIC_BLOCKED"
ALIGNED="STAGE190C_GRADIENT_COUPLING_CONFLICT_ALIGNS_WITH_SEED_REGRESSION"
SHARED="STAGE190C_SHARED_GRADIENT_CONFLICT_PRESENT_BUT_NOT_SEED_ALIGNED"
HEAD_LOCAL="STAGE190C_MARGIN_GRADIENT_HEAD_LOCAL_OR_NONCONFLICTING"
INCONCLUSIVE="STAGE190C_GRADIENT_DIAGNOSTIC_INCONCLUSIVE_OR_SEED_UNSTABLE"
NEGATIVE189="STAGE189D_THREE_SEED_MARGIN_NEGATIVE_OR_REGRESSIVE"
EXPECTED_OBJECTIVE_ROW_COUNTS={
 "margin_eligible":605,"ce_eligible":605,"ce_clean_dev_all":720,"ce_clean_dev_support":89,
 "neg_support_vs_not_entitled_margin":89,"neg_support_vs_max_other_margin":89,
 "neg_mean_frame_logit_compatible_fn":13,"neg_mean_frame_logit_matched_controls":14,
 "neg_mean_frame_logit_ineligible":716,"neg_mean_frame_logit_unresolved":119}
METRIC_GROUPS=("frame_head","decision_head","router_and_epistemic_heads","backbone","other_trainable","overall")
EXPECTED_GROUP_ROWS=54
EXPECTED_DIRECTIONAL_ROWS=9
FRACTION_TOLERANCE=1e-12
TARGETS=("ce_eligible","ce_clean_dev_all","ce_clean_dev_support",
 "neg_support_vs_not_entitled_margin","neg_support_vs_max_other_margin",
 "neg_mean_frame_logit_compatible_fn","neg_mean_frame_logit_matched_controls",
 "neg_mean_frame_logit_ineligible","neg_mean_frame_logit_unresolved")
OUTCOMES={
174:{"macro_f1_delta":.0089975,"support_recall_delta":.0224719,"false_entitlement_delta":-3,"polarity_delta":0,"mechanism_eligible_mean_delta":.251300,"mechanism_eligible_median_delta":.279331,"compatible_fn_newly_harmed":0},
175:{"macro_f1_delta":-.00364033,"support_recall_delta":-.0898876,"false_entitlement_delta":-13,"polarity_delta":0,"mechanism_eligible_mean_delta":.131983,"mechanism_eligible_median_delta":.101600,"compatible_fn_newly_harmed":2},
176:{"macro_f1_delta":-.00944096,"support_recall_delta":-.247191,"false_entitlement_delta":-40,"polarity_delta":1,"mechanism_eligible_mean_delta":-.392888,"mechanism_eligible_median_delta":-.395615,"compatible_fn_newly_harmed":7}}

def args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--stage190a-dir",type=Path,required=True);p.add_argument("--stage190b-dir",type=Path,required=True)
 p.add_argument("--stage189d-closure",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
 return p.parse_args()
def read(path):
 with path.open("r",encoding="utf-8") as h:return json.load(h)
def write_json(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
def write_csv(path,headers,rows:Iterable[dict[str,Any]]):
 with path.open("w",newline="",encoding="utf-8") as h:
  w=csv.DictWriter(h,fieldnames=headers,extrasaction="ignore");w.writeheader()
  for row in rows:w.writerow({k:row.get(k) for k in headers})
def finite(v):return not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(float(v))
def locate(root,seed,arm):
 candidates=[root/f"stage190b_seed{seed}_{arm}"/"stage190b_gradient_report.json",root/f"seed{seed}_{arm}"/"stage190b_gradient_report.json"]
 found=[p.resolve() for p in candidates if p.is_file()]
 if len(found)!=1:raise ValueError(f"seed{seed} {arm}: expected one report, found {found}")
 return found[0]
def conflict(report):
 d=report.get("directional_derivatives") or {};a=(d.get("ce_clean_dev_support") or {}).get("projected_target_change");b=(d.get("neg_support_vs_not_entitled_margin") or {}).get("projected_target_change")
 return finite(a) and finite(b) and a>0 and b>0
def nullable_cosine_valid(row):
 cosine=row.get("cosine_similarity")
 return finite(cosine) or (cosine is None and (row.get("zero_source") is True or row.get("zero_target") is True))

def validate(r,m,seed,arm):
 topology={"eligible":605,"ineligible":716,"unresolved":119,"clean_dev_all":720,"clean_dev_support":89,"compatible_fn":13,"incompatible_fp":1,"matched_controls":14,"clean_model_failures":14}
 directional=r.get("directional_derivatives") or {}
 directional_exact=set(directional)==set(TARGETS) and len(directional)==EXPECTED_DIRECTIONAL_ROWS
 directional_rows_valid=directional_exact and all(
  isinstance(row,dict) and row.get("objective")==name
  and row.get("row_count")==EXPECTED_OBJECTIVE_ROW_COUNTS[name]
  and row.get("finite_value_gate") is True
  and all(finite(row.get(key)) for key in ("dot_product","projected_target_change","projected_target_change_per_unit_margin_norm","projected_target_change_at_weight_0_05"))
  and nullable_cosine_valid(row)
  for name,row in directional.items())
 group_rows=r.get("group_gradient_metrics") or []
 group_pairs=[(row.get("objective"),row.get("parameter_group")) for row in group_rows if isinstance(row,dict)]
 expected_pairs={(target,group) for target in TARGETS for group in METRIC_GROUPS}
 group_exact=len(group_rows)==EXPECTED_GROUP_ROWS and len(set(group_pairs))==EXPECTED_GROUP_ROWS and set(group_pairs)==expected_pairs
 group_rows_valid=group_exact and all(
  row.get("finite_value_gate") is True
  and all(finite(row.get(key)) for key in ("source_gradient_squared_norm","target_gradient_squared_norm","dot_product","source_norm_fraction"))
  and (finite(row.get("target_norm_fraction")) or (row.get("target_norm_fraction") is None and row.get("zero_target") is True))
  and nullable_cosine_valid(row)
  for row in group_rows)
 objective_rows=r.get("objectives") or {}
 objective_rows_valid=set(objective_rows)==set(EXPECTED_OBJECTIVE_ROW_COUNTS) and all(
  isinstance(row,dict) and row.get("objective")==name
  and row.get("row_count")==EXPECTED_OBJECTIVE_ROW_COUNTS[name]
  and row.get("finite_gradient_gate") is True
  and finite(row.get("objective_mean")) and finite(row.get("gradient_norm"))
  for name,row in objective_rows.items())
 fractions=[r.get("shared_margin_gradient_fraction"),r.get("frame_head_margin_gradient_fraction"),r.get("decision_head_margin_gradient_fraction")]
 fraction_sum=sum(fractions) if all(finite(value) for value in fractions) else None
 fraction_difference=abs(fraction_sum-1.0) if fraction_sum is not None else None
 fraction_rows_valid=(all(finite(value) and 0.0<=value<=1.0 for value in fractions)
  and finite(r.get("source_fraction_sum")) and abs(r.get("source_fraction_sum")-fraction_sum)<=FRACTION_TOLERANCE
  and finite(r.get("source_fraction_partition_difference")) and abs(r.get("source_fraction_partition_difference")-fraction_difference)<=FRACTION_TOLERANCE
  and r.get("source_fraction_partition_tolerance")==FRACTION_TOLERANCE
  and fraction_difference<=FRACTION_TOLERANCE and r.get("source_fraction_partition_passed") is True)
 checks={"exported":r.get("decision")==EXPORTED,"seed_arm":r.get("training_seed")==seed and r.get("arm")==arm,
 "split":r.get("split_seed")==174,"checkpoint":r.get("selected_checkpoint_sha256")==m.get("checkpoint_sha256"),
 "training_commit":r.get("training_git_commit")==m.get("training_git_commit"),"diagnostic_commit":r.get("diagnostic_git_commit")==m.get("diagnostic_git_commit"),
 "trainer_sha":r.get("trainer_sha256")==m.get("trainer_sha256"),"diagnostic_sha":r.get("diagnostic_script_sha256")==m.get("diagnostic_script_sha256"),
 "helper_sha":r.get("checkpoint_helper_sha256")==m.get("checkpoint_helper_sha256"),"artifact_hashes":r.get("artifact_hashes")==m.get("artifact_hashes"),
 "state_flag":r.get("model_state_unchanged") is True,
 "trainable_state_sha":isinstance(r.get("trainable_state_sha256_before"),str) and bool(r.get("trainable_state_sha256_before")) and r.get("trainable_state_sha256_before")==r.get("trainable_state_sha256_after"),
 "buffer_state_sha":isinstance(r.get("buffer_state_sha256_before"),str) and bool(r.get("buffer_state_sha256_before")) and r.get("buffer_state_sha256_before")==r.get("buffer_state_sha256_after"),
 "evaluation":r.get("evaluation_only") is True and r.get("training_performed") is False,"optimizer":r.get("optimizer_created") is False and r.get("optimizer_step_performed") is False,
 "sources":r.get("source_score")== 'direct output["frame_logit"]' and r.get("classifier_source")== 'output["logits"]',
 "grad_enabled":r.get("model_grad_enabled_during_diagnostic") is True,
 "topology":all((r.get("cohort_topology") or {}).get(k)==v for k,v in topology.items()),
 "support_count":r.get("clean_dev_support_row_count")==89,
 "support_sha":isinstance(r.get("gold_support_clean_dev_row_ids_sha256"),str) and bool(r.get("gold_support_clean_dev_row_ids_sha256")),
 "objective_counts":r.get("objective_row_counts")==EXPECTED_OBJECTIVE_ROW_COUNTS and r.get("expected_objective_row_counts")==EXPECTED_OBJECTIVE_ROW_COUNTS,
 "objective_component_rows":objective_rows_valid,
 "directional_rows":directional_rows_valid and r.get("directional_derivative_expected_rows")==EXPECTED_DIRECTIONAL_ROWS and r.get("directional_derivative_observed_rows")==EXPECTED_DIRECTIONAL_ROWS and r.get("directional_derivative_grid_passed") is True,
 "group_rows":group_rows_valid and r.get("parameter_group_metric_expected_rows")==EXPECTED_GROUP_ROWS and r.get("parameter_group_metric_observed_rows")==EXPECTED_GROUP_ROWS and r.get("parameter_group_metric_grid_passed") is True,
 "parameter_ordering":isinstance(r.get("parameter_ordering_sha256"),str) and bool(r.get("parameter_ordering_sha256")),
 "parameter_counts":type(r.get("trainable_parameter_count")) is int and r.get("trainable_parameter_count")>0 and type(r.get("trainable_parameter_numel")) is int and r.get("trainable_parameter_numel")>0,
 "fraction_partition":fraction_rows_valid}
 return [k for k,v in checks.items() if not v]
def main():
 a=args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);blockers=[];reports={}
 try:
  if read(a.stage189d_closure.resolve()).get("decision")!=NEGATIVE189:blockers.append("Stage189-D negative closure mismatch")
  ma=read(a.stage190a_dir.resolve()/"stage190a_manifest_report.json")
  if ma.get("decision")!="STAGE190A_GRADIENT_CONFLICT_DIAGNOSTIC_MANIFEST_READY":blockers.append("Stage190-A not READY")
  for seed in SEEDS:
   for arm in ARMS:
    m=read(a.stage190a_dir.resolve()/f"stage190a_seed{seed}_{arm}_manifest.json");r=read(locate(a.stage190b_dir.resolve(),seed,arm));fails=validate(r,m,seed,arm)
    if fails:blockers.append(f"seed{seed}_{arm}: "+", ".join(fails))
    reports[seed,arm]=r
  support_hashes={r.get("gold_support_clean_dev_row_ids_sha256") for r in reports.values()}
  if len(support_hashes)!=1 or None in support_hashes:blockers.append("gold-SUPPORT row identity mismatch")
  ordering_hashes={r.get("parameter_ordering_sha256") for r in reports.values()}
  if len(ordering_hashes)!=1 or None in ordering_hashes or "" in ordering_hashes:blockers.append("parameter ordering SHA differs across six reports")
  parameter_counts={(r.get("trainable_parameter_count"),r.get("trainable_parameter_numel")) for r in reports.values()}
  if len(parameter_counts)!=1:blockers.append("trainable parameter count/numel differs across six reports")
  if len({r.get("selected_checkpoint_sha256") for r in reports.values()})!=6:blockers.append("checkpoint identities not distinct")
 except Exception as exc:blockers.append(f"input inspection failed: {type(exc).__name__}: {exc}")
 summary=[];paired=[];groups=[];joins=[]
 if len(reports)==6:
  for seed in SEEDS:
   for arm in ARMS:
    r=reports[seed,arm];row={"seed":seed,"arm":arm,"active_eligible_row_count":r.get("active_eligible_row_count"),"source_margin_loss":r.get("source_margin_loss"),"total_margin_gradient_norm":r.get("total_margin_gradient_norm"),"shared_margin_gradient_fraction":r.get("shared_margin_gradient_fraction"),"frame_head_margin_gradient_fraction":r.get("frame_head_margin_gradient_fraction"),"decision_head_margin_gradient_fraction":r.get("decision_head_margin_gradient_fraction"),"support_conflict_both_signs":conflict(r)}
    for target in TARGETS:
     d=(r.get("directional_derivatives") or {}).get(target) or {};row[target+"__cosine"]=d.get("cosine_similarity");row[target+"__projected_change"]=d.get("projected_target_change")
    summary.append(row)
    for metric in r.get("group_gradient_metrics") or []:groups.append({"seed":seed,"arm":arm,**metric})
   base=next(x for x in summary if x["seed"]==seed and x["arm"]=="baseline");inter=next(x for x in summary if x["seed"]==seed and x["arm"]=="intervention")
   for key in base:
    if key not in ("seed","arm","support_conflict_both_signs") and finite(base[key]) and finite(inter[key]):paired.append({"seed":seed,"metric":key,"baseline":base[key],"intervention":inter[key],"intervention_minus_baseline":inter[key]-base[key]})
   bg={(x.get("objective"),x.get("parameter_group")):x for x in reports[seed,"baseline"].get("group_gradient_metrics") or []}
   ig={(x.get("objective"),x.get("parameter_group")):x for x in reports[seed,"intervention"].get("group_gradient_metrics") or []}
   for group_key in sorted(set(bg)&set(ig)):
    for metric in ("source_gradient_squared_norm","target_gradient_squared_norm","dot_product","cosine_similarity","source_norm_fraction","target_norm_fraction"):
     bv=bg[group_key].get(metric);iv=ig[group_key].get(metric)
     if finite(bv) and finite(iv):paired.append({"seed":seed,"metric":f"group__{group_key[0]}__{group_key[1]}__{metric}","baseline":bv,"intervention":iv,"intervention_minus_baseline":iv-bv})
   joins.append({"seed":seed,"join_role":"descriptive_only_no_causality",**OUTCOMES[seed]})
 conflicts={s:conflict(reports[s,"intervention"]) for s in SEEDS} if len(reports)==6 else {}
 shared={s:reports[s,"intervention"].get("shared_margin_gradient_fraction") for s in SEEDS} if len(reports)==6 else {}
 aligned=len(reports)==6 and conflicts[175] and conflicts[176] and not conflicts[174] and finite(shared.get(175)) and finite(shared.get(176)) and shared[175]>=.05 and shared[176]>=.05
 qualified=[s for s in SEEDS if conflicts.get(s) and finite(shared.get(s)) and shared[s]>=.05]
 local=len(reports)==6 and all(finite(shared.get(s)) and shared[s]<.05 for s in SEEDS);zero=len(reports)==6 and not any(conflicts.values())
 decision=BLOCKED if blockers else ALIGNED if aligned else SHARED if len(qualified)>=2 else HEAD_LOCAL if local or zero else INCONCLUSIVE
 gates=[{"rule":"blocked","satisfied":bool(blockers),"details":" | ".join(blockers)},{"rule":"exact_seed_alignment","satisfied":aligned,"details":json.dumps(conflicts,sort_keys=True)},{"rule":"at_least_two_shared_conflicts","satisfied":len(qualified)>=2,"details":json.dumps(qualified)},{"rule":"all_shared_below_0.05","satisfied":local,"details":json.dumps(shared,sort_keys=True)},{"rule":"zero_support_conflicts","satisfied":zero,"details":json.dumps(conflicts,sort_keys=True)}]
 next_design={ALIGNED:"Consider stop-gradient, a frame-head-local margin, or a detached auxiliary path.",HEAD_LOCAL:"Investigate checkpoint-selection and optimization-trajectory effects.",INCONCLUSIVE:"Require trajectory checkpoints or per-epoch gradient snapshots before architecture change.",SHARED:"No architecture change is authorized because the conflict is not regression-seed aligned.",BLOCKED:"None until failed gates are repaired."}[decision]
 report={"stage":"Stage190-C","decision":decision,"blocking_reasons":blockers,"diagnostic_only":True,"model_advancement_decision":False,"significance_testing_performed":False,"causality_claimed":False,"intervention_support_conflict_by_seed":conflicts,"intervention_shared_margin_gradient_fraction_by_seed":shared,"qualified_shared_conflict_seeds":qualified,"authorized_next_design_class":next_design}
 write_json(out/"stage190c_gradient_conflict_report.json",report);write_csv(out/"stage190c_precommitted_gate.csv",["rule","satisfied","details"],gates)
 write_csv(out/"stage190c_seed_arm_summary.csv",list(summary[0]) if summary else ["seed","arm"],summary);write_csv(out/"stage190c_paired_seed_deltas.csv",["seed","metric","baseline","intervention","intervention_minus_baseline"],paired)
 write_csv(out/"stage190c_group_conflict_summary.csv",list(groups[0]) if groups else ["seed","arm","objective","parameter_group"],groups);write_csv(out/"stage190c_stage189_outcome_join.csv",list(joins[0]) if joins else ["seed"],joins)
 observations=[f"- Seed {s} intervention: active eligible {reports[s,'intervention'].get('active_eligible_row_count')}, source loss {reports[s,'intervention'].get('source_margin_loss')}, shared fraction {shared.get(s)}." for s in SEEDS if (s,"intervention") in reports]
 projections=[f"- Seed {s}: both SUPPORT-conflict signs = {conflicts.get(s)}." for s in SEEDS] if conflicts else ["- Unavailable because inputs were blocked."]
 md=f'''# Stage190-C gradient-conflict analysis

**Decision:** `{decision}`

## Direct observations

{chr(10).join(observations) if observations else '- No valid six-report observation set.'}

## First-order local projections

`delta_theta = -g_margin`; `projected_target_change = -dot(g_target, g_margin)`. Positive worsens a target loss; for negative decision-margin objectives it reduces the SUPPORT margin.

{chr(10).join(projections)}

## Seed-alignment interpretation

The precommitted 175/176-versus-174 signs and shared-fraction rules are applied exactly. Stage189 outcomes are descriptive joins only; no significance test or causal inference is made.

## Limitations

- Selected-checkpoint local diagnostic, not trajectory or 20-epoch causal proof.
- Not generalization evidence, external evaluation, or model-advancement permission.
- Posthoc train rows remain mechanism diagnostics; n=3 is not significance-tested.

## Authorized next design class

{next_design}

Stage191 is not implemented.

## Blocking reasons

{chr(10).join('- '+x for x in blockers) if blockers else '- None.'}
'''
 (out/"stage190c_gradient_conflict_report.md").write_text(md,encoding="utf-8");return 2 if decision==BLOCKED else 0

if __name__=="__main__":
 try:raise SystemExit(main())
 except BaseException:traceback.print_exc();raise
