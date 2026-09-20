#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

RAW_ROOT = Path(
    "reports/reason_router_gen4_pre_emission_forced_decisive_stage_a_generation_runs/"
    "g4k-preemission-forceddecisive-stagea-raw-370m-0d3a094-gpu1"
)
RAW_ROWS = RAW_ROOT / "forced_decisive_stage_a_generation_rows.jsonl"
RAW_SUMMARY = RAW_ROOT / "execution_summary.json"
RAW_SUMS = RAW_ROOT / "SHA256SUMS.txt"
PRIOR_ANALYSIS = Path(
    "reports/reason_router_gen4_pre_emission_forced_decisive_stage_a_analysis_runs/"
    "fabd313-confirmatory/stage_a_analysis.json"
)

RAW_ROWS_BLOB = "1691f8e9582f786760df19bee778f247d53b238b"
RAW_SUMMARY_BLOB = "725e5454a824adc18814d3a20931bb2ab65209f7"
RAW_SUMS_BLOB = "a7c41fb72d08bc8f5083f45d96bf39aca693bd64"
PRIOR_ANALYSIS_BLOB = "f3e70808bfaace2cf8b13566b59e67b2d0d09284"

RAW_ROWS_SHA256 = "5cc15d2626b350108c2c532ab8502d87aacb0d5253f82905ab83dc438aff5fcd"
RAW_SUMMARY_SHA256 = "0dc283213dda376b9684ee34e093de3ad6e09a425ea4fa4884390fa8b14555c5"

N = 462
OFFSETS = (-4,-3,-2,-1)
PARTITIONS = ("calibration","confirmatory")
GROUPS = ("unsupported","supported")
SCHEMA = "gen4-pre-emission-forced-decisive-signed-p3-failure-anatomy-v1"
RESULT = "PASS_PRE_EMISSION_FORCED_DECISIVE_SIGNED_P3_FAILURE_ANATOMY"
JSON_FILE = "signed_p3_failure_anatomy.json"
REPORT_FILE = "signed_p3_failure_anatomy.md"
SUMS_FILE = "SHA256SUMS.txt"

class AnatomyError(RuntimeError): pass
def req(ok: bool,msg: str)->None:
    if not ok: raise AnatomyError(msg)

def git(*args: str)->str:
    try:
        return subprocess.check_output(["git",*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT).strip()
    except (OSError,subprocess.CalledProcessError) as exc:
        raise AnatomyError("GIT_FAILURE:"+" ".join(args)) from exc

def blob(path: Path)->bytes:
    try:
        return subprocess.check_output(["git","cat-file","blob",f"HEAD:{path.as_posix()}"],cwd=ROOT,stderr=subprocess.STDOUT)
    except (OSError,subprocess.CalledProcessError) as exc:
        raise AnatomyError(f"BLOB:{path}") from exc

def sha256_bytes(x: bytes)->str: return hashlib.sha256(x).hexdigest()
def sha256_file(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()

def finite(x: Any,label: str)->float:
    y=float(x); req(math.isfinite(y),f"NONFINITE:{label}"); return y

def quantile(values: Sequence[float],q: float)->float|None:
    if not values:return None
    v=sorted(float(x) for x in values); p=(len(v)-1)*q; lo=math.floor(p); hi=math.ceil(p)
    return v[lo] if lo==hi else v[lo]+(p-lo)*(v[hi]-v[lo])

def desc(values: Sequence[float])->dict[str,Any]:
    v=[finite(x,"DESC") for x in values]; req(v,"DESC_EMPTY")
    return {"n":len(v),"mean":statistics.fmean(v),"median":quantile(v,.5),
            "q25":quantile(v,.25),"q75":quantile(v,.75),"min":min(v),"max":max(v),
            "fraction_positive":sum(x>0 for x in v)/len(v),
            "fraction_negative":sum(x<0 for x in v)/len(v),
            "fraction_zero":sum(x==0 for x in v)/len(v)}

def wrap_angle(x: float)->float:
    return math.atan2(math.sin(x),math.cos(x))

def authenticate(head: str)->None:
    req(git("branch","--show-current") in ("",EXPECTED_BRANCH),"BRANCH")
    req(git("rev-parse","HEAD")==head,"HEAD")
    req(git("status","--porcelain")=="","WORKTREE")
    for p,b in {RAW_ROWS:RAW_ROWS_BLOB,RAW_SUMMARY:RAW_SUMMARY_BLOB,RAW_SUMS:RAW_SUMS_BLOB,PRIOR_ANALYSIS:PRIOR_ANALYSIS_BLOB}.items():
        req(git("rev-parse",f"HEAD:{p.as_posix()}")==b,f"BLOB:{p}")

def load()->tuple[list[dict[str,Any]],dict[str,Any],dict[str,Any]]:
    rr,sr,xr,pr=blob(RAW_ROWS),blob(RAW_SUMMARY),blob(RAW_SUMS),blob(PRIOR_ANALYSIS)
    req(sha256_bytes(rr)==RAW_ROWS_SHA256,"ROWS_SHA")
    req(sha256_bytes(sr)==RAW_SUMMARY_SHA256,"SUMMARY_SHA")
    declared={}
    for line in xr.decode().splitlines():
        if line.strip():
            d,n=line.split("  ",1); declared[n]=d
    req(declared=={"execution_summary.json":RAW_SUMMARY_SHA256,
                   "forced_decisive_stage_a_generation_rows.jsonl":RAW_ROWS_SHA256},"SUMS")
    rows=[json.loads(x) for x in rr.decode().splitlines() if x.strip()]
    summary=json.loads(sr); prior=json.loads(pr)
    req(len(rows)==N,"N")
    req(summary["result"]=="PASS_PRE_EMISSION_FORCED_DECISIVE_STAGE_A_RAW_GENERATION","RAW_RESULT")
    req(summary["generation_protocol"]["observation_offsets"]==list(OFFSETS),"OFFSETS")
    req(summary["p_value_count_added"]==0,"RAW_P")
    req(summary["layer_scan_executed"] is False,"LAYER_SCAN")
    req(prior["result"]=="FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED","PRIOR_RESULT")
    req(prior["p_value_count_added"]==4,"PRIOR_PCOUNT")
    req(prior["significant_offsets"]==[],"PRIOR_SIG")
    for i,row in enumerate(rows):
        req(row["schema_version"]=="gen4-pre-emission-forced-decisive-stage-a-generation-row-v1",f"SCHEMA:{i}")
        req(row["partition"] in PARTITIONS,f"PART:{i}")
        req(row["primary_stage_a_group"] in GROUPS,f"GROUP:{i}")
        obs=row["observations"]; req(len(obs)==4,f"OBS_N:{i}")
        req(tuple(int(o["relative_offset"]) for o in obs)==OFFSETS,f"OBS_ORDER:{i}")
        for o in obs:
            a,b=finite(o["p3_a"],"a"),finite(o["p3_b"],"b")
            l2=finite(o["p3_component_l2"],"l2")
            req(math.isclose(math.hypot(a,b),l2,rel_tol=1e-10,abs_tol=1e-10),f"L2:{i}")
            finite(o["m47_support_minus_refute"],"m47")
    return rows,summary,prior

def summarize_rows(rows: Sequence[Mapping[str,Any]])->dict[str,Any]:
    by_offset={o:{"unsupported":[],"supported":[]} for o in OFFSETS}
    increments={(a,b):{"unsupported":[],"supported":[]} for a,b in zip(OFFSETS[:-1],OFFSETS[1:])}

    for row in rows:
        group=str(row["primary_stage_a_group"])
        obs={int(o["relative_offset"]):o for o in row["observations"]}
        for off in OFFSETS:
            o=obs[off]; a=finite(o["p3_a"],"a"); b=finite(o["p3_b"],"b")
            by_offset[off][group].append({
                "a":a,"b":b,"l2":finite(o["p3_component_l2"],"l2"),
                "theta":math.atan2(b,a),
                "m47":finite(o["m47_support_minus_refute"],"m47"),
            })
        for left,right in zip(OFFSETS[:-1],OFFSETS[1:]):
            lo,ro=obs[left],obs[right]
            a0,b0=finite(lo["p3_a"],"a0"),finite(lo["p3_b"],"b0")
            a1,b1=finite(ro["p3_a"],"a1"),finite(ro["p3_b"],"b1")
            increments[(left,right)][group].append({
                "delta_a":a1-a0,
                "delta_b":b1-b0,
                "delta_l2":finite(ro["p3_component_l2"],"l21")-finite(lo["p3_component_l2"],"l20"),
                "delta_theta_wrapped":wrap_angle(math.atan2(b1,a1)-math.atan2(b0,a0)),
                "delta_m47":finite(ro["m47_support_minus_refute"],"m471")-finite(lo["m47_support_minus_refute"],"m470"),
            })

    offsets_out={}
    for off in OFFSETS:
        entry={}
        for group in GROUPS:
            rs=by_offset[off][group]
            mean_a=statistics.fmean(r["a"] for r in rs); mean_b=statistics.fmean(r["b"] for r in rs)
            entry[group]={
                "n":len(rs),
                "a":desc([r["a"] for r in rs]),
                "b":desc([r["b"] for r in rs]),
                "l2":desc([r["l2"] for r in rs]),
                "theta_radians":desc([r["theta"] for r in rs]),
                "m47_support_minus_refute":desc([r["m47"] for r in rs]),
                "mean_vector":{"a":mean_a,"b":mean_b,
                               "radius":math.hypot(mean_a,mean_b),
                               "theta":math.atan2(mean_b,mean_a)},
                "quadrant_counts":{
                    "a_pos_b_pos":sum(r["a"]>0 and r["b"]>0 for r in rs),
                    "a_pos_b_nonpos":sum(r["a"]>0 and r["b"]<=0 for r in rs),
                    "a_nonpos_b_pos":sum(r["a"]<=0 and r["b"]>0 for r in rs),
                    "a_nonpos_b_nonpos":sum(r["a"]<=0 and r["b"]<=0 for r in rs),
                },
            }
        offsets_out[str(off)]={
            **entry,
            "unsupported_minus_supported_mean":{
                key: entry["unsupported"][key]["mean"]-entry["supported"][key]["mean"]
                for key in ("a","b","l2","m47_support_minus_refute")
            },
            "mean_vector_difference":{
                "a":entry["unsupported"]["mean_vector"]["a"]-entry["supported"]["mean_vector"]["a"],
                "b":entry["unsupported"]["mean_vector"]["b"]-entry["supported"]["mean_vector"]["b"],
            },
        }

    inc_out={}
    for (left,right),groups in increments.items():
        key=f"{left}_to_{right}"
        inc_out[key]={}
        for group in GROUPS:
            rs=groups[group]
            inc_out[key][group]={k:desc([r[k] for r in rs]) for k in
                                 ("delta_a","delta_b","delta_l2","delta_theta_wrapped","delta_m47")}
        inc_out[key]["unsupported_minus_supported_mean"]={
            k:inc_out[key]["unsupported"][k]["mean"]-inc_out[key]["supported"][k]["mean"]
            for k in ("delta_a","delta_b","delta_l2","delta_theta_wrapped","delta_m47")
        }
    return {"offsets":offsets_out,"increments":inc_out}

def analyze(rows: Sequence[Mapping[str,Any]],head: str)->dict[str,Any]:
    calibration=[r for r in rows if r["partition"]=="calibration"]
    confirmatory=[r for r in rows if r["partition"]=="confirmatory"]
    req(len(calibration)==231 and len(confirmatory)==231,"PARTITION_N")
    return {
        "schema_version":SCHEMA,
        "result":RESULT,
        "analysis_head":head,
        "source_result":"FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED",
        "raw_population_n":N,
        "offset_order":list(OFFSETS),
        "all_rows":summarize_rows(rows),
        "calibration":summarize_rows(calibration),
        "confirmatory":summarize_rows(confirmatory),
        "analysis_boundary":{
            "exploratory_representation_anatomy_only":True,
            "new_p_value_count":0,
            "model_forward_count":0,
            "feature_selection_performed":False,
            "offset_selection_performed":False,
            "coordinate_selection_performed":False,
            "precursor_rescue_claimed":False,
            "v2_design_must_be_prospective_and_fresh":True,
            "interpretation":"Tests whether the failed radial observable may have discarded signed/orientational structure. Any descriptive pattern here is hypothesis-generating only and cannot establish a precursor on this dataset."
        }
    }

def render(a: Mapping[str,Any])->str:
    lines=[
        "# Forced-Decisive Precursor — Signed P3 Static Failure Anatomy","",
        f"Result: `{a['result']}`","",
        "The prior confirmatory result remains `FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`. This report adds no p-values and does not rescue it.","",
        "## Question","",
        "Did the failed radial observable `P3_COMPONENT_L2 = sqrt(a^2+b^2)` discard signed/orientational state information that is visible in the already frozen raw trajectories?","",
        "All four offsets and both signed coordinates are reported symmetrically. No coordinate or offset is selected or ranked.","",
    ]
    for off in OFFSETS:
        e=a["confirmatory"]["offsets"][str(off)]
        lines += [f"## Confirmatory offset {off}","",
                  f"- unsupported N: `{e['unsupported']['n']}`; supported N: `{e['supported']['n']}`.",
                  f"- mean a, unsupported/supported: `{e['unsupported']['a']['mean']}` / `{e['supported']['a']['mean']}`.",
                  f"- mean b, unsupported/supported: `{e['unsupported']['b']['mean']}` / `{e['supported']['b']['mean']}`.",
                  f"- mean L2, unsupported/supported: `{e['unsupported']['l2']['mean']}` / `{e['supported']['l2']['mean']}`.",
                  f"- mean-vector theta, unsupported/supported: `{e['unsupported']['mean_vector']['theta']}` / `{e['supported']['mean_vector']['theta']}`.",
                  f"- mean M47 SUPPORT-REFUTE, unsupported/supported: `{e['unsupported']['m47_support_minus_refute']['mean']}` / `{e['supported']['m47_support_minus_refute']['mean']}`.",""]
    lines += ["## Boundary","",
              "These descriptive signed trajectories are hypothesis-generating only. They are not evidence for a recovered precursor, are not used to choose a best coordinate/offset, and do not alter the frozen Stage A null. Precursor v2 must define one dynamic causal susceptibility observable prospectively on a fresh population before response inspection.",""]
    return "\n".join(lines)

def write(out: Path,a: Mapping[str,Any])->None:
    req(not out.exists(),"OUTPUT_COLLISION"); out.mkdir(parents=True)
    jp,rp,sp=out/JSON_FILE,out/REPORT_FILE,out/SUMS_FILE
    jp.write_text(json.dumps(a,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    rp.write_text(render(a),encoding="utf-8",newline="\n")
    hashes={JSON_FILE:sha256_file(jp),REPORT_FILE:sha256_file(rp)}
    sp.write_text("".join(f"{d}  {n}\n" for n,d in sorted(hashes.items())),encoding="utf-8",newline="\n")

def main(argv: Sequence[str]|None=None)->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--expected-head",required=True); ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args(argv)
    authenticate(args.expected_head)
    rows,_,_=load()
    a=analyze(rows,args.expected_head)
    write(args.output_dir,a)
    print("RESULT="+a["result"])
    print("RAW_POPULATION_N=462")
    print("CONFIRMATORY_N=231")
    print("OFFSET_ORDER=-4,-3,-2,-1")
    print("SIGNED_COORDINATES_REPORTED=a,b,theta")
    print("NEW_P_VALUE_COUNT=0")
    print("MODEL_FORWARD_COUNT=0")
    print("FEATURE_SELECTION_PERFORMED=False")
    print("PRECURSOR_RESCUE_CLAIMED=False")

if __name__=="__main__":
    main()
