# LoRA vs full fine-tuning: 48-hour classification study

Three scripts:

- `train_cls.py` runs one experiment (linear probe, LoRA or full fine-tuning) and appends its results to a CSV.
- `run_grid.py` runs whole stages of the grid and skips finished runs, so it can be restarted after a timeout.
- `make_report.py` builds the summary table and plots from the CSV.

## Setup

```bash
pip install torch torchvision timm peft pandas matplotlib
```

Datasets download automatically into `--data-root` (default `./data`) through torchvision.
For forgetting, download **ImageNetV2** (matched-frequency, 10,000 images) or use the ImageNet validation set,
arranged as one folder per class. ImageNetV2's numeric folder names (`0` … `999`) are handled correctly.

## Order of work

```bash
# 0. Code check on fake data (no downloads, ~minutes on CPU): every method, both model types
python run_grid.py --stage lr --dry-run --models vit_tiny_patch16_224 resnet18 --results-dir results_dry

# 1. Sanity checks on real data (see "Checks" below)
python train_cls.py --model vit_base_patch16_224 --dataset flowers102 --method lora --lr 1e-3 \
    --overfit 100 --out results/sanity.csv --label sanity

# 2. Learning-rate mini-grid (seed 0), writes results/best_lrs.json   (~45 min per model on a T4)
python run_grid.py --stage lr --forget-root /path/to/imagenetv2

# 3. Main grid: 3 methods x (Flowers, EuroSAT 1%, EuroSAT 100%) x 3 seeds   (~2.5 h per model on a T4)
python run_grid.py --stage main --forget-root /path/to/imagenetv2

# 4. If time allows: ResNet-50 (run its lr stage first), then the rank sweep
python run_grid.py --stage lr   --models resnet50 --forget-root /path/to/imagenetv2
python run_grid.py --stage main --models resnet50 --forget-root /path/to/imagenetv2
python run_grid.py --stage rank --forget-root /path/to/imagenetv2

# 5. Tables and plots
python make_report.py
```

## Checks before trusting any result

- **Parameter counts:** ViT-B/16 with LoRA r=16 on `qkv` and `proj` trains about 0.96M parameters on
  Flowers-102 (about 1.1%). Check the `trainable_params` column.
- **LoRA starts as the pretrained model:** `lora_init_maxdiff` must be 0 (or ~1e-3 under fp16).
- **Overfitting:** with `--overfit 100`, every method should reach close to 100% `test_acc`
  (it trains and tests on the same 100 images).
- **Forgetting pipeline:** the linear probe must show `forgetting` = 0, and LoRA's `forget_adapters_off`
  must equal `forget_before`.

## Notes

- Results land in `results/results.csv` one row per run, with the full configuration and seed.
- Failed runs are listed in `results/failures.txt`; rerun the same stage command to retry them.
- Mixed precision is on by default (fp16 on T4, bf16 on A100/RTX 30xx+).
- Frozen BatchNorm layers are kept in eval mode, so the linear probe and LoRA on ResNet show no false forgetting.
- The linear probe trains on cached features (no augmentation), the standard way to run it fast.
