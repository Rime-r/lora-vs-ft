#!/usr/bin/env python
"""Turns results/results.csv into a summary table (Markdown) and three plots.

    python make_report.py --results results/results.csv --out-dir results/report

Outputs: summary.md, forgetting_vs_accuracy_<model>.png, eurosat_data_size.png, rank_sweep.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["linear", "lora", "full"]
METHOD_NAME = {"linear": "Linear probe", "lora": "LoRA", "full": "Full fine-tuning"}
COLORS = {"linear": "#8c8c8c", "lora": "#1f77b4", "full": "#d62728"}


def pm(mean, std, scale=1.0, digits=1):
    if pd.isna(mean):
        return "–"
    s = f"{mean * scale:.{digits}f}"
    return s + (f" ± {std * scale:.{digits}f}" if not pd.isna(std) else "")


def summarize(df):
    keys = ["label", "model", "dataset", "fraction", "method", "rank"]
    metrics = ["test_acc", "test_macro_f1", "forgetting", "forgetting_bn_restored",
               "trainable_pct", "peak_mem_gb", "epoch_time_s"]
    df = df.copy()
    if "forgetting_bn_restored" not in df:   # results from before this column existed
        df["forgetting_bn_restored"] = np.nan
    df["rank"] = df["rank"].fillna(0).astype(int)
    agg = df.groupby(keys)[metrics].agg(["mean", "std"])
    agg["seeds"] = df.groupby(keys).size()
    return agg.reset_index()


def to_markdown(s):
    lines = ["| Stage | Model | Dataset | Data | Method | Rank | Test acc (%) | Macro-F1 (%) | Forgetting (pts) "
             "| Forgetting, BN stats restored (pts) | Trainable (%) | Peak mem (GB) | Time/epoch (s) | Seeds |",
             "|" + "---|" * 14]
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    s = s.sort_values(by=[("label", ""), ("model", ""), ("dataset", ""), ("fraction", ""), ("method", ""), ("rank", "")],
                      key=lambda c: c.map(order) if c.name == ("method", "") else c)
    for _, r in s.iterrows():
        lines.append("| " + " | ".join([
            str(r[("label", "")]), str(r[("model", "")]), str(r[("dataset", "")]),
            f"{r[('fraction', '')] * 100:g}%", METHOD_NAME.get(r[("method", "")], r[("method", "")]),
            str(r[("rank", "")]) if r[("method", "")] == "lora" else "–",
            pm(r[("test_acc", "mean")], r[("test_acc", "std")], 100),
            pm(r[("test_macro_f1", "mean")], r[("test_macro_f1", "std")], 100),
            pm(r[("forgetting", "mean")], r[("forgetting", "std")], 100, 2),
            pm(r[("forgetting_bn_restored", "mean")], r[("forgetting_bn_restored", "std")], 100, 2),
            pm(r[("trainable_pct", "mean")], np.nan, 1, 2),
            pm(r[("peak_mem_gb", "mean")], np.nan, 1, 2),
            pm(r[("epoch_time_s", "mean")], np.nan, 1, 1),
            str(int(r[("seeds", "")])),
        ]) + " |")
    return "\n".join(lines)


def plot_forgetting(df, out_dir):
    d = df[(df["label"] == "main") & df["forgetting"].notna()]
    for model, dm in d.groupby("model"):
        settings = sorted(dm.groupby(["dataset", "fraction"]).groups.keys())
        fig, axes = plt.subplots(1, len(settings), figsize=(4.2 * len(settings), 3.8), squeeze=False)
        for ax, (ds, fr) in zip(axes[0], settings):
            for m in METHOD_ORDER:
                x = dm[(dm.dataset == ds) & (dm.fraction == fr) & (dm.method == m)]
                if x.empty:
                    continue
                ax.errorbar(x.test_acc.mean() * 100, x.forgetting.mean() * 100,
                            xerr=x.test_acc.std() * 100 if len(x) > 1 else None,
                            yerr=x.forgetting.std() * 100 if len(x) > 1 else None,
                            fmt="o", color=COLORS[m], label=METHOD_NAME[m], capsize=3, ms=7)
            ax.axhline(0, color="#cccccc", lw=1)
            ax.set_title(f"{ds}, {fr * 100:g}% data")
            ax.set_xlabel("Target test accuracy (%)")
            ax.set_ylabel("Forgetting on ImageNet (pts)")
        axes[0][0].legend(frameon=False)
        fig.suptitle(f"{model}: higher accuracy, at what forgetting cost?")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"forgetting_vs_accuracy_{model}.png"), dpi=150)
        plt.close(fig)


def plot_data_size(df, out_dir):
    d = df[(df["label"] == "main") & (df["dataset"] == "eurosat")]
    if d.empty:
        return
    models = sorted(d.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.8), squeeze=False)
    for ax, model in zip(axes[0], models):
        dm = d[d.model == model]
        fracs = sorted(dm.fraction.unique())
        w = 0.8 / len(METHOD_ORDER)
        for i, m in enumerate(METHOD_ORDER):
            means = [dm[(dm.fraction == f) & (dm.method == m)].test_acc.mean() * 100 for f in fracs]
            stds = [dm[(dm.fraction == f) & (dm.method == m)].test_acc.std() * 100 for f in fracs]
            ax.bar(np.arange(len(fracs)) + (i - 1) * w, means, w, yerr=stds, capsize=3,
                   color=COLORS[m], label=METHOD_NAME[m])
        ax.set_xticks(np.arange(len(fracs)), [f"{f * 100:g}% of training data" for f in fracs])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(model)
    axes[0][0].legend(frameon=False, loc="lower right")
    fig.suptitle("EuroSAT: does the gap between methods change with data size?")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eurosat_data_size.png"), dpi=150)
    plt.close(fig)


def plot_rank(df, out_dir):
    d = df[df["label"] == "rank"]
    if d.empty:
        return
    g = d.groupby("rank").agg(acc=("test_acc", "mean"), acc_sd=("test_acc", "std"),
                              fg=("forgetting", "mean"), fg_sd=("forgetting", "std")).reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.errorbar(g["rank"], g.acc * 100, yerr=g.acc_sd * 100, fmt="o-", capsize=3, color=COLORS["lora"])
    a1.set_xscale("log", base=2)
    a1.set_xlabel("LoRA rank r")
    a1.set_ylabel("Test accuracy (%)")
    a2.errorbar(g["rank"], g.fg * 100, yerr=g.fg_sd * 100, fmt="o-", capsize=3, color=COLORS["lora"])
    a2.set_xscale("log", base=2)
    a2.set_xlabel("LoRA rank r")
    a2.set_ylabel("Forgetting on ImageNet (pts)")
    fig.suptitle("Flowers-102: effect of LoRA rank")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rank_sweep.png"), dpi=150)
    plt.close(fig)


DET_METHOD_ORDER = ["head", "lora", "full"]
DET_METHOD_NAME = {"head": "Head only", "lora": "LoRA", "full": "Full fine-tuning"}
DET_COLORS = {"head": COLORS["linear"], "lora": COLORS["lora"], "full": COLORS["full"]}


def det_markdown(df):
    keys = ["model", "fraction", "method"]
    metrics = ["test_map", "test_map50", "forgetting", "trainable_pct", "peak_mem_gb", "epoch_time_s"]
    g = df.groupby(keys)[metrics].agg(["mean", "std"])
    g["seeds"] = df.groupby(keys).size()
    g = g.reset_index()
    order = {m: i for i, m in enumerate(DET_METHOD_ORDER)}
    g = g.sort_values(by=[("model", ""), ("fraction", ""), ("method", "")],
                      key=lambda c: c.map(order) if c.name == ("method", "") else c)
    lines = ["| Model | Data | Method | VOC mAP (%) | VOC mAP@0.5 (%) | COCO forgetting (mAP pts) "
             "| Trainable (%) | Peak mem (GB) | Time/epoch (s) | Seeds |", "|" + "---|" * 10]
    for _, r in g.iterrows():
        lines.append("| " + " | ".join([
            str(r[("model", "")]), f"{r[('fraction', '')] * 100:g}%",
            DET_METHOD_NAME.get(r[("method", "")], r[("method", "")]),
            pm(r[("test_map", "mean")], r[("test_map", "std")], 100),
            pm(r[("test_map50", "mean")], r[("test_map50", "std")], 100),
            pm(r[("forgetting", "mean")], r[("forgetting", "std")], 100, 2),
            pm(r[("trainable_pct", "mean")], np.nan, 1, 2),
            pm(r[("peak_mem_gb", "mean")], np.nan, 1, 2),
            pm(r[("epoch_time_s", "mean")], np.nan, 1, 1),
            str(int(r[("seeds", "")])),
        ]) + " |")
    return "\n".join(lines)


def plot_det(df, out_dir):
    d = df[df["forgetting"].notna()]
    if d.empty:
        return
    fracs = sorted(d.fraction.unique())
    fig, axes = plt.subplots(1, len(fracs), figsize=(4.4 * len(fracs), 3.8), squeeze=False)
    for ax, fr in zip(axes[0], fracs):
        for m in DET_METHOD_ORDER:
            x = d[(d.fraction == fr) & (d.method == m)]
            if x.empty:
                continue
            ax.errorbar(x.test_map.mean() * 100, x.forgetting.mean() * 100,
                        xerr=x.test_map.std() * 100 if len(x) > 1 else None,
                        yerr=x.forgetting.std() * 100 if len(x) > 1 else None,
                        fmt="o", color=DET_COLORS[m], label=DET_METHOD_NAME[m], capsize=3, ms=7)
        ax.axhline(0, color="#cccccc", lw=1)
        ax.set_title(f"VOC 2007, {fr * 100:g}% data")
        ax.set_xlabel("VOC test mAP@[.5:.95] (%)")
        ax.set_ylabel("COCO forgetting (mAP pts)")
    axes[0][0].legend(frameon=False)
    fig.suptitle("RT-DETR: detection accuracy vs forgetting")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "det_forgetting_vs_map.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/results.csv")
    p.add_argument("--out-dir", default="results/report")
    p.add_argument("--det-results", default="results/det_results.csv")
    p.add_argument("--include-dry-run", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.results)
    if not a.include_dry_run:
        df = df[df["dry_run"].astype(str) != "True"]
    if df.empty:
        raise SystemExit("no (non-dry-run) results yet")
    md = "## Classification\n\n" + to_markdown(summarize(df))
    det = None
    if os.path.exists(a.det_results):
        det = pd.read_csv(a.det_results)
        if not a.include_dry_run:
            det = det[det["dry_run"].astype(str) != "True"]
        if not det.empty:
            md += "\n\n## Detection\n\n" + det_markdown(det)
    with open(os.path.join(a.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    plot_forgetting(df, a.out_dir)
    plot_data_size(df, a.out_dir)
    plot_rank(df, a.out_dir)
    if det is not None and not det.empty:
        plot_det(det, a.out_dir)
    print(f"\nplots and summary.md written to {a.out_dir}")


if __name__ == "__main__":
    main()
