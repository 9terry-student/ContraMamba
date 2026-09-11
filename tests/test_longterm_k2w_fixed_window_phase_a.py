import hashlib, importlib.util, json, sys, zipfile
from pathlib import Path
import pytest

P=Path(__file__).parents[1]/"scripts"/"longterm_k2w_fixed_window_phase_a.py"; spec=importlib.util.spec_from_file_location("k2w",P); k=importlib.util.module_from_spec(spec);sys.modules["k2w"]=k;spec.loader.exec_module(k)

class Tok:
 is_fast=True
 def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
  # fixed-width characters make deliberate shared continuation prefixes simple
  d={"input_ids":[ord(x) for x in text]}
  if return_offsets_mapping:d["offset_mapping"]=[(i,i+1) for i in range(len(text))]
  return d
def row(pair,kind,claim="c",evidence="t",label="NOT_ENTITLED",failure="sufficiency",polarity="NONE"):
 return {"id":pair+kind,"pair_id":pair,"claim":claim,"evidence":evidence,"final_label":label,"frame_compatible_label":1,"predicate_covered_label":1,"sufficiency_label":0,"polarity_label":polarity,"primary_failure_type":failure,"intervention_type":kind}
def valid_rows(pair="a",claim="claim",c="abcdefghi",n="xyzabcdefghi"):
 return [row(pair,"evidence_truncation",claim,"trunc"),row(pair,"entity_swap",claim,n,"NOT_ENTITLED","frame","NONE"),row(pair,"polarity_flip",claim,c,"REFUTE","none","REFUTE")]
def test_constants_and_no_native_state_boundary():
 s=P.read_text(encoding="utf8")
 assert k.K2W_PREREG_AUTHORITY_COMMIT=="cc5386e730c333209eb070b14025c5368038e247"
 assert k.K2_CLOSURE_COMMIT=="386ef0763a0dd8470c22617581af196a324f222f"
 assert "This is false:" not in s and "A separate event:" not in s
 for bad in ("register_forward_hook","O0c","state instrumentation","enable trace") : assert bad not in s
def test_exact_recipe_and_stable_identity():
 x=k.recipe("a",valid_rows()[0:])
 assert x["prefix_text"]=="Claim: claim\nEvidence: trunc\nAdditional evidence:\n"
 assert x["correction_text"]=="abcdefghi" and x["control_text"]=="xyzabcdefghi"
 core={z:x[z] for z in ("schema_version","pair_id","truncation_source_id","refute_source_id","control_source_id","prefix_text","correction_text","control_text","source_dataset_physical_sha256","source_dataset_semantic_sha256")}
 assert x["stable_item_id"]=="k2w-v1:"+hashlib.sha256(k.canonical_json(core)).hexdigest()
def test_role_failures_and_none_refute_fallback():
 rows=valid_rows();rows[2]["intervention_type"]="none"; assert k.recipe("a",rows)["construction_status"]=="valid"
 rows=valid_rows();rows[1]["evidence"]="trunc";assert k.recipe("a",rows)["construction_failure_label"]=="INVALID_CONTROL_SEMANTICS"
 rows=valid_rows();rows.append(rows[2].copy());assert k.recipe("a",rows)["construction_failure_label"]=="INVALID_SOURCE_MULTIPLICITY"
def test_masks_are_literal_overlap_and_disjoint():
 b=k.tokenize_prefix_bundle(Tok(),"Claim: aa\nEvidence: bb\nAdditional evidence:\n")
 assert sum(b["claim_mask"])==2 and sum(b["evidence_mask"])==2
 assert not any(a and c for a,c in zip(b["claim_mask"],b["evidence_mask"]))
def test_event_first_difference_and_shared_initial_continuation():
 x=k.recipe("a",valid_rows(c="xabcdefgh",n="xyabcdefghi"));z=k.token_contract(x,Tok())
 assert z["ok"] and z["d"]>z["p"] and z["tau"]==z["d"]-1 and z["tau_minus_p"]==1
 assert z["corr_post_tau_available"]>=8 and z["ctrl_post_tau_available"]>=8
def test_event_failures_exact_prefix_divergence_prestate_and_window():
 x=k.recipe("a",valid_rows(c="same",n="same"));assert k.token_contract(x,Tok())["failure"]=="INVALID_EVENT_DIVERGENCE"
 x=k.recipe("a",valid_rows(c="a",n="b"));assert k.token_contract(x,Tok())["failure"]=="INVALID_WINDOW_AVAILABILITY"
 x=k.recipe("a",valid_rows(c="abcdefgh",n="ijklmnop"));q=k.tokenize_prefix_bundle(Tok(),x["prefix_text"]);q["input_ids"][0]=0;assert k.token_contract(x,Tok(),q)["failure"]=="INVALID_EXACT_PREFIX"
def test_unequal_lengths_allowed_and_duplicate_resolution_pre_screen():
 rows=valid_rows("b",claim="same",c="aBBBBBBBBBB",n="aXCCCCCCCCCCCC")+valid_rows("a",claim="same",c="aBBBBBBBBBB",n="aXCCCCCCCCCCCC")
 pool,checked=k.prepare_candidate_pool(rows,Tok())
 assert len(checked)==1
 kept=[x for x in pool if x["construction_status"]=="valid"][0];assert kept["pair_id"]=="a"
 assert any(x["construction_failure_label"]=="DUPLICATE_BASE_CLAIM_EXCLUDED" for x in pool)
def test_final_n_and_deterministic_over64_selection():
 pool={};screen=[]
 for i in range(29):screen.append({"stable_item_id":str(i),"eligible":True});pool[str(i)]={"schema_version":k.CANDIDATE_SCHEMA,"pair_id":str(i),"truncation_source_id":"t","refute_source_id":"q","control_source_id":"e","prefix_text":"p","correction_text":"c","control_text":"n","source_dataset_physical_sha256":"x","source_dataset_semantic_sha256":"y"}
 e,f,v=k.id_lists(screen,pool);assert len(e)==29 and not f and v.startswith("INCONCLUSIVE")
 for i in range(29,70):screen.append({"stable_item_id":str(i),"eligible":True});pool[str(i)]=pool["0"]|{"pair_id":str(i)}
 e,f,v=k.id_lists(screen,pool);assert len(e)==70 and len(f)==64 and v=="PHASE_B_ELIGIBLE"
def test_exact_three_of_three_support_eligibility_ignores_continuous_values():
 pool=[{"stable_item_id":"x","construction_status":"valid","construction_failure_label":None}]
 checked={"x":{"p":1,"d":2,"tau":1,"tau_minus_p":0,"corr_post_tau_available":8,"ctrl_post_tau_available":8}}
 def head(seed,label,confidence,margin):return {"seed":seed,"checkpoint_sha256":"c","predicted_final_label":label,"probabilities":{"REFUTE":.1,"NOT_ENTITLED":.2,"SUPPORT":.7},"confidence":confidence,"margin":margin}
 outputs={"seed180":{"x":head("seed180","SUPPORT",.99,.98)},"seed181":{"x":head("seed181","SUPPORT",.34,.01)},"seed182":{"x":head("seed182","SUPPORT",.51,.02)}}
 assert k.join_screening(pool,outputs,checked)[0]["eligible"] is True
 outputs["seed182"]["x"]=head("seed182","REFUTE",.999,.997)
 assert k.join_screening(pool,outputs,checked)[0]["eligible"] is False
def test_screening_never_forwards_invalid_or_excluded():
 pool=[{"stable_item_id":"x","construction_status":"invalid","construction_failure_label":"INVALID_EVENT_DIVERGENCE"},{"stable_item_id":"y","construction_status":"excluded","construction_failure_label":"DUPLICATE_BASE_CLAIM_EXCLUDED"}]
 rows=k.join_screening(pool,{},{});assert all(not x["eligible"] and x["head_outputs"]==[] for x in rows)
def test_handoff_schema_discovery_and_path_safety():
 cp=b"checkpoint";h=hashlib.sha256(cp).hexdigest(); old=k.EXPECTED_CHECKPOINT_SHA256["seed180"];oldzip=k.EXPECTED_ZIP_SHA256["seed180"]
 try:
  k.EXPECTED_CHECKPOINT_SHA256["seed180"]=h
  m={"schema":k.HANDOFF_SCHEMA,"expected_commit":k.A0_COMMIT,"actual_commit":k.A0_COMMIT,"files":[{"path":"a.pt","sha256":h,"size_bytes":len(cp)}]}
  import io
  b=io.BytesIO()
  with zipfile.ZipFile(b,"w") as q:q.writestr("manifest.json",json.dumps(m));q.writestr("files/a.pt",cp)
  with zipfile.ZipFile(io.BytesIO(b.getvalue())) as q:
   n,found=k.discover_handoff_manifest(q);assert n=="manifest.json" and found==m
 finally:k.EXPECTED_CHECKPOINT_SHA256["seed180"]=old;k.EXPECTED_ZIP_SHA256["seed180"]=oldzip
 assert pytest.raises(ValueError,lambda:k._safe_member("../evil"))
def test_source_format_rejects_bom_cr_missing_lf_and_blank():
 for raw in (b"\xef\xbb\xbf{}\n",b"{}\r\n",b"{}",b"\n"):
  with pytest.raises(ValueError):k.parse_frozen_jsonl(raw)
def test_git_provenance_mode_split_and_tracked_changes_rejected(monkeypatch):
 status=["?? scripts/longterm_k2w_fixed_window_phase_a.py","?? tests/test_longterm_k2w_fixed_window_phase_a.py"]
 def out(args,**kw):
  if args[1]=="status":return "\n".join(status)+("\n" if status else "")
  return "longterm-k-series-native-state-kinematics\n" if args[1]=="branch" else "head\n" if args[1]=="rev-parse" else ""
 monkeypatch.setattr(k.subprocess,"check_output",out);monkeypatch.setattr(k.subprocess,"call",lambda *a,**kw:0)
 assert k.git_provenance(Path("."),integration_preflight=True)["runtime_branch"].startswith("longterm")
 with pytest.raises(ValueError):k.git_provenance(Path("."),integration_preflight=False)
 status[:]=["?? scripts/longterm_k1_native_state_kinematics.py","?? tests/test_longterm_k1_native_state_kinematics.py"]
 assert k.git_provenance(Path("."),integration_preflight=False)["runtime_branch"].startswith("longterm")
 status[:]=[" M scripts/longterm_k2w_fixed_window_phase_a.py"]
 with pytest.raises(ValueError):k.git_provenance(Path("."),integration_preflight=True)
 monkeypatch.setattr(k.subprocess,"call",lambda *a,**kw:1)
 with pytest.raises(ValueError):k.git_provenance(Path("."))
def test_construction_funnel_credits_earlier_stages():
 pool=[
  {"construction_status":"invalid","construction_failure_label":"INVALID_EVENT_DIVERGENCE","tau_minus_p":None,"corr_post_tau_available":None,"ctrl_post_tau_available":None},
  {"construction_status":"invalid","construction_failure_label":"INVALID_WINDOW_AVAILABILITY","tau_minus_p":None,"corr_post_tau_available":None,"ctrl_post_tau_available":None},
 ]
 s=k.construction_summary(pool,{})
 assert (s["N_semantically_valid"],s["N_exact_prefix_valid"],s["N_event_divergence_valid"],s["N_window8_available"],s["N_construction_valid"])==(2,2,1,0,0)
def test_manifest_preserves_observed_per_seed_encoder_provenance():
 encoder={"canonical_digest":"canonical","raw_concat_digest":"raw","tensor_count":2,"total_numel":3,"total_raw_bytes":12,"dtypes":["torch.float32"]}
 handoffs={s:{"seed":s,"checkpoint_sha256":s,"encoder":encoder|{"canonical_digest":"canonical-"+s},"strict_load":"PASS","metadata_training_args":"PASS"} for s in k.EXPECTED_ZIP_SHA256}
 hf={"resolved_hf_revision":k.HF_REVISION,"tokenizer_files":[],"tokenizer_class":"Tok","tokenizer_backend_class":"Backend"}
 pool=[];screen=[];summary={"N_total_attempts":0,"N_semantically_valid":0,"N_exact_prefix_valid":0,"N_event_divergence_valid":0,"N_window8_available":0,"N_construction_valid":0,"N_duplicate_excluded":0,"tau_minus_p_distribution":{},"corr_post_tau_available_distribution":{},"ctrl_post_tau_available_distribution":{}}
 blobs={"candidate_pool.jsonl":b"","screening.jsonl":b"","eligible_ids.jsonl":b"","final_confirmatory_ids.jsonl":b""}
 m=k.phase_a_manifest(handoffs,hf,{},summary,blobs,[],[],"INCONCLUSIVE")
 for s in k.EXPECTED_ZIP_SHA256:
  assert m["handoffs"][s]["encoder"]==handoffs[s]["encoder"]
  assert m["handoffs"][s]["strict_load"]=="PASS" and m["handoffs"][s]["metadata_training_args"]=="PASS"
