# Stage 13 OOD validation Kaggle commands

Run from a Kaggle notebook after cloning or resetting the repository.

## Setup

```bash
%%bash
cd /kaggle/working/ContraMamba
git fetch origin
git reset --hard origin/main
pip install -e .
```

## Local smoke equivalent

```bash
%%bash
cd /kaggle/working/ContraMamba
python tools/run_stage13_ood_v5_vs_v6a.py \
  --epochs 2 \
  --seeds 1 \
  --backbone dummy \
  --device cpu \
  --smoke
```

## Full dummy-backbone Stage 13 run

This is the main controlled OOD validation command. It trains v5 clean retrain
and v6A residual on controlled_v5_v3 with `time_swap` excluded before pair_id
splitting, then evaluates both on the configured OOD probe.

```bash
%%bash
cd /kaggle/working/ContraMamba
python tools/run_stage13_ood_v5_vs_v6a.py \
  --epochs 200 \
  --seeds 1 2 3 \
  --backbone dummy \
  --device cpu
```

## Optional GPU/Mamba run

Use only when the Kaggle environment has the required Mamba/HF dependencies and
GPU memory available.

```bash
%%bash
cd /kaggle/working/ContraMamba
python tools/run_stage13_ood_v5_vs_v6a.py \
  --epochs 20 \
  --seeds 1 2 3 \
  --backbone mamba \
  --device cuda
```

## Optional Stage10C probe path

If a Stage10C surface/temporality probe file is available, pass it explicitly:

```bash
%%bash
cd /kaggle/working/ContraMamba
python tools/run_stage13_ood_v5_vs_v6a.py \
  --epochs 200 \
  --seeds 1 2 3 \
  --backbone dummy \
  --device cpu \
  --ood-data data/stage10c_surface_temporality_probe.jsonl
```
