import json
from pathlib import Path
from collections import Counter, defaultdict

root = Path(__file__).resolve().parents[1]
src = root / "data/controlled_v5_v3.jsonl"
out = root / "data/stage30c1_temporal_safety_diagnostic_from_controlled_v5_v3.jsonl"

assert src.exists(), f"missing source: {src}"

rows = []
with src.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

wanted = {"none", "paraphrase", "time_swap"}
buckets = defaultdict(list)

for r in rows:
    t = str(r.get("intervention_type", ""))
    if t in wanted:
        buckets[t].append(r)

print("bucket sizes:", {k: len(v) for k, v in buckets.items()})

missing = [k for k in wanted if len(buckets[k]) == 0]
assert not missing, f"missing buckets: {missing}"

n = min(300, *(len(buckets[k]) for k in wanted))

diagnostic = []
for t in ["none", "paraphrase", "time_swap"]:
    for r in buckets[t][:n]:
        rr = dict(r)
        rr["temporal_intervention_type"] = t
        rr["temporal_safety_label"] = 0 if t == "time_swap" else 1
        rr["temporal_mismatch_label"] = 1 if t == "time_swap" else 0
        diagnostic.append(rr)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in diagnostic:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("wrote:", out)
print("rows:", len(diagnostic))
print("interventions:", Counter(r["temporal_intervention_type"] for r in diagnostic))
print("temporal_safety_label:", Counter(r["temporal_safety_label"] for r in diagnostic))
print("temporal_mismatch_label:", Counter(r["temporal_mismatch_label"] for r in diagnostic))
print("size:", out.stat().st_size)