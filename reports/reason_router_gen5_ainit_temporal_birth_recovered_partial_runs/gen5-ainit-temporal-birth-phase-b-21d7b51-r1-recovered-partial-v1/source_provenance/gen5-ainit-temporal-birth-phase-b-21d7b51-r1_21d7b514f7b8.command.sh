set -euo pipefail
cd /kaggle/working/ContraMamba

EXPECTED_HEAD="21d7b514f7b88617cff3b679fb97ecaf6c5411ec"
SNAPSHOT="/root/.cache/huggingface/hub/models--state-spaces--mamba-130m-hf/snapshots/40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
CHECKPOINT="/kaggle/working/gen5_phase2_seed180_G3_GROUP_D_HALF_selected_checkpoint.pt"
EXPECTED_CHECKPOINT_SHA256="1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
OUTPUT_ROOT="reports/reason_router_gen5_ainit_temporal_birth_analysis_runs/gen5-ainit-temporal-birth-phase-b-21d7b51-r1"

test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
test -d "$SNAPSHOT"
test -f "$CHECKPOINT"
test ! -e "$OUTPUT_ROOT"

ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
test "$ACTUAL_CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256"

PYTHONPATH="$PWD:$PWD/src" python - <<'PY'
import torch
from scripts import audit_reason_router_gen5_ainit_temporal_birth as audit

assert torch.cuda.is_available(), "CUDA_NOT_AVAILABLE"
assert torch.cuda.device_count() == 2, f"EXPECTED_2_GPUS observed={torch.cuda.device_count()}"
names = [torch.cuda.get_device_name(i) for i in range(2)]
assert all("T4" in name for name in names), f"EXPECTED_2XT4 observed={names}"
assert audit.PHASE_B_PROJECTOR_BACKEND == "float64_cpu"
print("PHASE_B_SCIENCE_ENV_AUTH_PASS")
print("GPU_TOPOLOGY=2x_TESLA_T4")
print(f"PHASE_B_PROJECTOR_BACKEND={audit.PHASE_B_PROJECTOR_BACKEND}")
PY

PYTHONPATH="$PWD:$PWD/src" python scripts/audit_reason_router_gen5_ainit_temporal_birth.py   --run-phase-b   --expected-head "$EXPECTED_HEAD"   --implementation-freeze-commit "$EXPECTED_HEAD"   --model-snapshot "$SNAPSHOT"   --tokenizer-snapshot "$SNAPSHOT"   --checkpoint "$CHECKPOINT"   --output-root "$OUTPUT_ROOT"