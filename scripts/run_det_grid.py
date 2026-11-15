#!/usr/bin/env python
"""Runs the detection grid, calling train_det.py once per configuration.

Stages, in order:
    lr    learning-rate tuning on 10% of VOC07 train (seed 0); writes det_best_lrs.json
    main  head / LoRA / full at 10% and 100% of the data, 2 seeds, with the tuned learning rates

Finished runs are skipped, so after a stop or a crash just start the same command again.

Example:
    python run_det_grid.py --stage lr   --coco-root data/coco
    python run_det_grid.py --stage main --coco-root data/coco
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_det.py")

LR_GRID = {
    "head": [3e-4, 1e-3, 3e-3, 1e-2],
    "lora": [1e-4, 3e-4, 1e-3, 3e-3],
    "full": [3e-5, 1e-4, 3e-4],  # backbone trains at 0.1 x this
}
TUNE_FRACTION = 0.1
MAIN_FRACTIONS = [0.1, 1.0]


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def run(cfg, a, out, label, done):
    rid = f"{label}|{a.model}|voc07|f{cfg['fraction']}|{cfg['method']}|r16|lr{cfg['lr']}|s{cfg['seed']}"
    if rid in done:
        print(f"skip (done): {rid}", flush=True)
        return
    cmd = [sys.executable, TRAIN, "--label", label, "--out", out, "--model", a.model,
           "--voc-root", a.voc_root, "--workers", str(a.workers), "--batch-size", str(a.batch_size),
           "--precision", a.precision, "--coco-max-images", str(a.coco_max_images)]
    for k, v in cfg.items():
        cmd += [f"--{k}", str(v)]
    if a.coco_root:
        cmd += ["--coco-root", a.coco_root]
    if a.dry_run:
        cmd += ["--dry-run", "--epochs", "1", "--min-steps", "1", "--evals", "1", "--img-size", "256"]
    print("\n>>>", rid, flush=True)
    import subprocess
    if subprocess.run(cmd).returncode != 0:
        with open(os.path.join(a.results_dir, "det_failures.txt"), "a", encoding="utf-8") as f:
            f.write(rid + "\n")
        print(f"FAILED: {rid} (logged to det_failures.txt)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["lr", "main"])
    p.add_argument("--model", default="PekingU/rtdetr_r18vd")
    p.add_argument("--methods", nargs="+", default=["head", "lora", "full"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--voc-root", default="./data")
    p.add_argument("--coco-root", default=None)
    p.add_argument("--coco-max-images", type=int, default=5000)
    p.add_argument("--precision", default="tf32", choices=["tf32", "bf16", "fp32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    os.makedirs(a.results_dir, exist_ok=True)
    lr_csv = os.path.join(a.results_dir, "det_lr_grid.csv")
    main_csv = os.path.join(a.results_dir, "det_results.csv")
    best_path = os.path.join(a.results_dir, "det_best_lrs.json")

    if a.stage == "lr":
        done = done_ids(lr_csv)
        for method in a.methods:
            for lr in LR_GRID[method]:
                run(dict(method=method, lr=lr, fraction=TUNE_FRACTION, seed=0), a, lr_csv, "det_lr", done)
        best = json.load(open(best_path)) if os.path.exists(best_path) else {}
        scores = {}
        with open(lr_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["model"] != a.model:
                    continue
                v = float(r["val_map_best"])
                if r["method"] not in scores or v > scores[r["method"]][0]:
                    scores[r["method"]] = (v, float(r["lr"]))
        for m, (v, lr) in sorted(scores.items()):
            best[f"{a.model}|{m}"] = lr
            edge = " (at the edge of the grid)" if lr in (min(LR_GRID[m]), max(LR_GRID[m])) else ""
            print(f"best lr  {a.model}|{m:5s} {lr:g}  (val mAP {v:.4f}){edge}")
        json.dump(best, open(best_path, "w"), indent=1)
        return

    if not os.path.exists(best_path):
        sys.exit("det_best_lrs.json not found: run --stage lr first")
    best = json.load(open(best_path))
    done = done_ids(main_csv)
    for frac in MAIN_FRACTIONS:
        for method in a.methods:
            key = f"{a.model}|{method}"
            if key not in best:
                sys.exit(f"no tuned learning rate for {key}: run --stage lr first")
            for seed in a.seeds:
                run(dict(method=method, lr=best[key], fraction=frac, seed=seed), a, main_csv, "main", done)


if __name__ == "__main__":
    main()
