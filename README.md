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

# Detection: RT-DETR on Pascal VOC 2007

Two more scripts:

- `train_det.py` runs one detection experiment (head only, LoRA or full fine-tuning) and appends to `results/det_results.csv`.
- `run_det_grid.py` runs the learning-rate stage and the main stage, skipping finished runs.

`make_report.py` adds a detection table and `det_forgetting_vs_map.png` when `results/det_results.csv` exists.

## Setup

```powershell
uv add transformers pycocotools
```

Data layout (the RT-DETR-R18 weights, `PekingU/rtdetr_r18vd`, download automatically):

```
data/VOCdevkit/VOC2007/Annotations, JPEGImages, ImageSets   (VOCtrainval_06-Nov-2007.tar + VOCtest_06-Nov-2007.tar)
data/coco/val2017/*.jpg                                     (val2017.zip)
data/coco/annotations/instances_val2017.json                (annotations_trainval2017.zip)
```

## What is compared

| Method | Trained | Frozen |
|---|---|---|
| Head only | New 20-class heads (encoder score head, decoder class heads, denoising embedding) | Everything else |
| LoRA (r=16) | Adapters on every linear layer of the transformer encoder/decoder and box MLPs, plus the new heads | Convolutions (backbone, hybrid encoder) |
| Full fine-tuning | Everything; backbone at 0.1 x the learning rate (standard RT-DETR recipe) | Backbone BatchNorm (RT-DETR default) |

- **Target:** VOC 2007 trainval split 90/10 into train/val; VOC 2007 test for the final score. Difficult objects are ignored.
- **Metric:** COCO-style mAP@[.5:.95] and mAP@0.5 with pycocotools (101-point interpolation, not VOC's 11-point).
- **Forgetting:** COCO val2017 mAP@[.5:.95] before minus after, with the original COCO class heads put back.

## Order of work

```powershell
# 0. Code check on fake data, no downloads (~5 min on GPU)
uv run python scripts/run_det_grid.py --stage lr --dry-run --results-dir results_dry

# 1. Timing and sanity: one short real run on 10% of the data, with the LoRA forgetting check
uv run python scripts/train_det.py --method lora --lr 3e-4 --fraction 0.1 --coco-root data/coco --adapters-off-check --out results/det_sanity.csv --label sanity

# 2. Learning-rate stage (9 runs), then main stage (12 runs)
uv run python scripts/run_det_grid.py --stage lr   --coco-root data/coco
uv run python scripts/run_det_grid.py --stage main --coco-root data/coco

# 3. Report
uv run python scripts/make_report.py
```

Checks for step 1, in `results/det_sanity.csv`:

- `forget_before` is around 0.45 (RT-DETR-R18 on COCO val2017). Near 0 means a data or label-mapping problem.
- `forget_adapters_off` equals `forget_before` (within about 0.001).
- `test_map50` is clearly above 0 after training.
- `epoch_time_s` x epochs gives the run time; full-data runs are about 10x longer per epoch.

Faster option: add `--precision bf16`. It is tested on the code path but not on your GPU; if the loss becomes
non-finite, go back to the default (`tf32`). Use the same precision for every run in a stage.
