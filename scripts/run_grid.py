#!/usr/bin/env python
"""Runs the 48-hour experiment grid, calling train_cls.py once per configuration.

Stages, in order:
    lr    learning-rate mini-grid (seed 0) on Flowers-102 and EuroSAT 1%; writes best_lrs.json
    main  core grid with the tuned learning rates: Flowers-102, EuroSAT 1% and 100%, 3 seeds
    rank  LoRA rank sweep on Flowers-102 (ViT only)

Finished runs are skipped, so after a timeout just start the same command again.

Example:
    python run_grid.py --stage lr   --forget-root /data/imagenetv2
    python run_grid.py --stage main --forget-root /data/imagenetv2
    python run_grid.py --stage main --models resnet50 --forget-root /data/imagenetv2
"""
import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_cls.py")

LR_GRID = {
    "linear": [1e-3, 3e-3, 1e-2, 3e-2],
    "lora": [3e-4, 1e-3, 3e-3, 1e-2],
    "full_vit": [3e-6, 1e-5, 3e-5, 1e-4],
    "full_cnn": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
}
TUNE_SETTINGS = [("flowers102", 1.0), ("eurosat", 0.01)]
MAIN_SETTINGS = [("flowers102", 1.0), ("eurosat", 0.01), ("eurosat", 1.0)]
TUNED_AT = {("eurosat", 1.0): ("eurosat", 0.01)}  # EuroSAT 100% reuses the rate tuned at 1%


def lr_values(method, model):
    if method == "full":
        return LR_GRID["full_vit" if any(k in model for k in ("vit", "deit", "swin")) else "full_cnn"]
    return LR_GRID[method]


def done_ids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def make_id(label, model, dataset, fraction, method, rank, lr, seed):
    return f"{label}|{model}|{dataset}|f{fraction}|{method}|r{rank}|lr{lr}|s{seed}"


def run(cfg, a, out, label, done):
    rid = make_id(label, cfg["model"], cfg["dataset"], cfg["fraction"], cfg["method"],
                  cfg.get("rank", 16), cfg["lr"], cfg["seed"])
    if rid in done:
        print(f"skip (done): {rid}", flush=True)
        return
    cmd = [sys.executable, TRAIN, "--label", label, "--out", out,
           "--data-root", a.data_root, "--workers", str(a.workers), "--batch-size", str(a.batch_size)]
    for k, v in cfg.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    if a.forget_root:
        cmd += ["--forget-root", a.forget_root, "--forget-max-images", str(a.forget_max_images)]
    if a.dry_run:
        cmd += ["--dry-run", "--epochs", "1", "--min-steps", "1", "--linear-epochs", "2"]
    print("\n>>>", rid, flush=True)
    if subprocess.run(cmd).returncode != 0:
        with open(os.path.join(a.results_dir, "failures.txt"), "a") as f:
            f.write(rid + "\n")
        print(f"FAILED: {rid} (logged to failures.txt)", flush=True)


def pick_best_lrs(lr_csv, best_path):
    best = json.load(open(best_path)) if os.path.exists(best_path) else {}
    scores = {}
    with open(lr_csv) as f:
        for r in csv.DictReader(f):
            key = f"{r['model']}|{r['dataset']}|{float(r['fraction'])}|{r['method']}"
            v = float(r["val_acc_best"])
            if key not in scores or v > scores[key][0]:
                scores[key] = (v, float(r["lr"]))
    for key, (v, lr) in scores.items():
        best[key] = lr
        print(f"best lr  {key:60s} {lr:g}  (val acc {v:.4f})")
    json.dump(best, open(best_path, "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["lr", "main", "rank"])
    p.add_argument("--models", nargs="+", default=["vit_base_patch16_224"])
    p.add_argument("--methods", nargs="+", default=["linear", "lora", "full"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 4, 16, 64])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--forget-root", default=None)
    p.add_argument("--forget-max-images", type=int, default=10000)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    os.makedirs(a.results_dir, exist_ok=True)
    lr_csv = os.path.join(a.results_dir, "lr_grid.csv")
    main_csv = os.path.join(a.results_dir, "results.csv")
    best_path = os.path.join(a.results_dir, "best_lrs.json")

    if a.stage == "lr":
        done = done_ids(lr_csv)
        for model in a.models:
            for dataset, frac in TUNE_SETTINGS:
                for method in a.methods:
                    for lr in lr_values(method, model):
                        run(dict(model=model, dataset=dataset, fraction=frac, method=method, lr=lr, seed=0),
                            a, lr_csv, "lr_grid", done)
        pick_best_lrs(lr_csv, best_path)
        return

    if not os.path.exists(best_path):
        sys.exit("best_lrs.json not found: run --stage lr first")
    best = json.load(open(best_path))

    def tuned_lr(model, dataset, frac, method):
        d, f = TUNED_AT.get((dataset, frac), (dataset, frac))
        key = f"{model}|{d}|{float(f)}|{method}"
        if key not in best:
            sys.exit(f"no tuned learning rate for {key}: run --stage lr for this model first")
        return best[key]

    done = done_ids(main_csv)
    if a.stage == "main":
        for model in a.models:
            for dataset, frac in MAIN_SETTINGS:
                for method in a.methods:
                    lr = tuned_lr(model, dataset, frac, method)
                    for seed in a.seeds:
                        run(dict(model=model, dataset=dataset, fraction=frac, method=method, lr=lr, seed=seed),
                            a, main_csv, "main", done)
    elif a.stage == "rank":
        for model in a.models:
            lr = tuned_lr(model, "flowers102", 1.0, "lora")
            for rank in a.ranks:
                for seed in a.seeds:
                    run(dict(model=model, dataset="flowers102", fraction=1.0, method="lora", lr=lr,
                             seed=seed, rank=rank, alpha=2 * rank), a, main_csv, "rank", done)


if __name__ == "__main__":
    main()
