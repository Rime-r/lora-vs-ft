#!/usr/bin/env python
"""Combines every result into one file, as an Excel workbook (one sheet per table) and as Markdown.

    python combine_results.py --results-dir results

Writes results/report/all_results.xlsx and results/report/all_results.md with four tables:
    1. Main results    classification and detection together (mean and std over seeds)
    2. Rank sweeps     ViT on Flowers-102, RT-DETR on 10% of VOC
    3. LR tuning       every learning-rate tuning run, with the selected rate marked
    4. BatchNorm check ResNet-50 full fine-tuning with BatchNorm statistics restored

Scores are in percent and forgetting in percentage points. A table whose CSV files are
missing is skipped. The Excel file shows means only and leaves out the BatchNorm check
(add --xlsx-std and/or --xlsx-bn to include them); the Markdown file always has everything.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

MODEL = {"vit_base_patch16_224": "ViT-B/16", "resnet50": "ResNet-50", "PekingU/rtdetr_r18vd": "RT-DETR-R18"}
DATASET = {"flowers102": "Flowers-102", "eurosat": "EuroSAT", "pets": "Oxford Pets", "voc07": "Pascal VOC 2007"}
METHOD = {"linear": "Linear probe", "head": "Head only", "lora": "LoRA", "full": "Full fine-tuning"}
METHOD_ORDER = {"linear": 0, "head": 0, "lora": 1, "full": 2}
TASK = {
    "Classification": dict(score="test_acc", score2="test_macro_f1", val="val_acc_best",
                           m1="Top-1 accuracy", m2="Macro-F1", forget_on="ImageNetV2 top-1"),
    "Detection": dict(score="test_map", score2="test_map50", val="val_map_best",
                      m1="mAP@[.5:.95]", m2="mAP@0.5", forget_on="COCO val2017 mAP@[.5:.95]"),
}


def read(path, include_dry):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, encoding="utf-8")
    if not include_dry and "dry_run" in d:
        d = d[d["dry_run"].astype(str) != "True"]
    return d if not d.empty else None


def col(d, name):
    return d[name].astype(float) if name in d else pd.Series(np.nan, index=d.index)


def unify(d, task):
    """One row per run, with the same columns for classification and detection."""
    t = TASK[task]
    method = d["method"].astype(str)
    rank = col(d, "rank").where(method == "lora")
    return pd.DataFrame({
        "Task": task,
        "Model": d["model"].map(lambda m: MODEL.get(m, m)),
        "Dataset": d["dataset"].map(lambda x: DATASET.get(x, x)),
        "Data (%)": col(d, "fraction") * 100,
        "Method": method.map(lambda m: METHOD.get(m, m)),
        "LoRA rank": rank,
        "Score metric": t["m1"],
        "Second metric": t["m2"],
        "Forgetting measured on": t["forget_on"],
        "seed": d["seed"],
        "lr": col(d, "lr"),
        "val": col(d, t["val"]) * 100,
        "score": col(d, t["score"]) * 100,
        "score2": col(d, t["score2"]) * 100,
        "forgetting": col(d, "forgetting") * 100,
        "forgetting_bn": col(d, "forgetting_bn_restored") * 100,
        "Trainable (%)": col(d, "trainable_pct"),
        "Peak memory (GB)": col(d, "peak_mem_gb"),
        "Time per epoch (s)": col(d, "epoch_time_s"),
        "Training time per run (min)": col(d, "train_time_s") / 60,
        "_order": method.map(lambda m: METHOD_ORDER.get(m, 9)),
    })


def aggregate(u, keys, stats):
    """Mean and std over seeds; stats maps internal column -> display name."""
    g = u.groupby(keys, dropna=False, sort=False)
    out = g.size().rename("Seeds").reset_index()
    for c, name in stats.items():
        m = g[c].mean().reset_index(drop=True)
        s = g[c].std().reset_index(drop=True)
        out[f"{name} mean"] = m.values
        out[f"{name} std"] = s.values
    for c in ("Trainable (%)", "Peak memory (GB)", "Time per epoch (s)", "Training time per run (min)"):
        if c in u:
            out[c] = g[c].mean().reset_index(drop=True).values
    seeds = out.pop("Seeds")
    out["Seeds"] = seeds
    return out.dropna(axis=1, how="all")


def main_table(cls, det):
    parts = []
    if cls is not None:
        parts.append(unify(cls[cls["label"] == "main"], "Classification"))
    if det is not None:
        parts.append(unify(det[det["label"] == "main"], "Detection"))
    if not parts:
        return None
    u = pd.concat(parts, ignore_index=True)
    keys = ["Task", "Model", "Dataset", "Data (%)", "_order", "Method", "LoRA rank",
            "Score metric", "Second metric", "Forgetting measured on"]
    t = aggregate(u, keys, {"score": "Score (%)", "score2": "Second metric (%)",
                            "forgetting": "Forgetting (pts)", "forgetting_bn": "Forgetting, BN stats restored (pts)"})
    t = t.sort_values(["Task", "Model", "Dataset", "Data (%)", "_order"]).drop(columns="_order")
    return t.reset_index(drop=True)


def rank_table(cls, det, det_rank):
    parts = []
    if cls is not None and (cls["label"] == "rank").any():
        parts.append(unify(cls[cls["label"] == "rank"], "Classification"))
    if det_rank is not None:
        r = det_rank.copy()
        if det is not None:  # rank 16 comes from the main runs, same data size and seeds
            seeds, fracs = set(r["seed"]), set(r["fraction"].astype(float))
            m = det[(det["label"] == "main") & (det["method"] == "lora") & det["seed"].isin(seeds)
                    & det["fraction"].astype(float).isin(fracs) & ~det["rank"].isin(set(r["rank"]))]
            r = pd.concat([r, m], ignore_index=True)
        parts.append(unify(r, "Detection"))
    if not parts:
        return None
    u = pd.concat(parts, ignore_index=True)
    keys = ["Task", "Model", "Dataset", "Data (%)", "LoRA rank", "Score metric", "Forgetting measured on"]
    t = aggregate(u, keys, {"score": "Score (%)", "forgetting": "Forgetting (pts)"})
    return t.sort_values(["Task", "Model", "LoRA rank"]).reset_index(drop=True)


def lr_table(cls_lr, det_lr, best_cls, best_det):
    parts = []
    for d, task, best in ((cls_lr, "Classification", best_cls), (det_lr, "Detection", best_det)):
        if d is None:
            continue
        u = unify(d, task)
        if task == "Classification":
            keys = d["model"] + "|" + d["dataset"] + "|" + d["fraction"].astype(float).astype(str) + "|" + d["method"]
        else:
            keys = d["model"] + "|" + d["method"]
        chosen = [k in best and math.isclose(float(best[k]), float(lr), rel_tol=1e-9)
                  for k, lr in zip(keys, d["lr"])]
        u["Selected"] = np.where(chosen, "yes", "")
        parts.append(u)
    if not parts:
        return None
    u = pd.concat(parts, ignore_index=True)
    t = u[["Task", "Model", "Dataset", "Data (%)", "Method", "lr", "val", "score", "forgetting", "Selected",
           "Time per epoch (s)", "Training time per run (min)", "Score metric", "_order"]].rename(columns={
        "lr": "Learning rate", "val": "Validation score (%)", "score": "Test score (%)",
        "forgetting": "Forgetting (pts)"})
    t = t.sort_values(["Task", "Model", "Dataset", "Data (%)", "_order", "Learning rate"]).drop(columns="_order")
    return t.reset_index(drop=True)


def bn_table(bn):
    if bn is None or "forgetting_bn_restored" not in bn:
        return None
    t = pd.DataFrame({
        "Model": bn["model"].map(lambda m: MODEL.get(m, m)),
        "Dataset": bn["dataset"].map(lambda x: DATASET.get(x, x)),
        "Data (%)": bn["fraction"].astype(float) * 100,
        "Learning rate": bn["lr"],
        "Seed": bn["seed"].astype(str),
        "Test accuracy (%)": bn["test_acc"] * 100,
        "Forgetting (pts)": bn["forgetting"] * 100,
        "Forgetting, BN stats restored (pts)": bn["forgetting_bn_restored"] * 100,
    })
    t["Share caused by BN statistics (%)"] = 100 * (t["Forgetting (pts)"] - t["Forgetting, BN stats restored (pts)"]) \
        / t["Forgetting (pts)"]
    num = t.select_dtypes("number").columns.drop(["Data (%)", "Learning rate"])
    summary = []
    for name, fn in (("mean", "mean"), ("std", "std")):
        row = {c: getattr(t[c], fn)() for c in num}
        row.update({"Model": "", "Dataset": "", "Seed": name})
        summary.append(row)
    return pd.concat([t, pd.DataFrame(summary)], ignore_index=True)


# ---------------------------------------------------------------- output

def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:g}" if (abs(v) < 1e-3 and v != 0) else f"{v:.{digits}f}"
    return str(v)


def to_markdown(t):
    """Merges 'X mean' and 'X std' columns into 'X' shown as mean ± std."""
    cols, rows = [], []
    names = list(t.columns)
    skip = set()
    for c in names:
        if c in skip:
            continue
        if c.endswith(" mean") and c[:-5] + " std" in names:
            cols.append((c[:-5], c, c[:-5] + " std"))
            skip.add(c[:-5] + " std")
        else:
            cols.append((c, c, None))
    lines = ["| " + " | ".join(h for h, _, _ in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        cells = []
        for h, m, s in cols:
            if s is None:
                v = r[m]
                if h == "Learning rate" and isinstance(v, float):
                    cells.append(f"{v:g}")
                elif h in ("Data (%)", "LoRA rank", "Seeds") and isinstance(v, float) and not math.isnan(v):
                    cells.append(f"{v:g}")
                else:
                    cells.append(fmt(v))
            else:
                mv, sv = r[m], r[s]
                cells.append(fmt(mv) if (isinstance(sv, float) and math.isnan(sv)) or isinstance(mv, str)
                             else f"{fmt(mv)} ± {fmt(sv)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


NOTE_COLS = ("Score metric", "Second metric", "Forgetting measured on")
RENAME = [("Validation score (%)", "Validation accuracy or mAP (%)"), ("Test score (%)", "Test accuracy or mAP (%)"),
          ("Score (%)", "Accuracy or mAP (%)"), ("Second metric (%)", "Macro-F1 or mAP@0.5 (%)")]
NOTES = [
    ("Accuracy or mAP (%)", "Top-1 accuracy on the test set", "mAP@[.5:.95] on the VOC 2007 test set"),
    ("Macro-F1 or mAP@0.5 (%)", "Macro-F1: F1 per class, averaged with equal weight", "mAP at IoU threshold 0.5"),
    ("Forgetting (pts)", "Drop in ImageNetV2 top-1 accuracy, original head put back",
     "Drop in COCO val2017 mAP@[.5:.95], original class heads put back"),
    ("Data (%)", "Share of the training split used", "Share of the VOC train split used"),
    ("LoRA rank", "Rank r of the LoRA update (LoRA rows only)", "Same as classification"),
    ("Trainable (%)", "Share of the model's parameters updated by training", "Same as classification"),
    ("Peak memory (GB)", "Most GPU memory allocated during training", "Same as classification"),
    ("Time per epoch (s)", "One pass over the training data; empty for the linear probe (cached features)", "Same as classification"),
    ("Training time per run (min)", "Training loop incl. validation, for one seed; excludes the final test and "
     "forgetting evaluations", "Same as classification"),
    ("Seeds", "Number of runs averaged", "Same as classification"),
    ("Selected (LR tuning)", "The learning rate used in the main runs", "Same as classification"),
    ("Values", "Means over seeds. Standard deviations are in all_results.md", "Same as classification"),
]
GREY, GREEN, RED, LINE = "E8EAED", "E6F4EA", "F4A6A6", "9AA0A6"
SHEETS = {  # sheet -> (columns that define a group, column whose best value is bold, frozen cell)
    "Main results": (["Task", "Model", "Dataset", "Data (%)"], "Accuracy or mAP (%)", "F2"),
    "Rank sweeps": (["Task", "Model"], "Accuracy or mAP (%)", "F2"),
    "LR tuning": (["Task", "Model", "Dataset", "Data (%)", "Method"], None, "G2"),
}


def tidy(t):
    """Drops columns that repeat the same text on every row (they go to the Notes sheet) and
    gives the score columns self-explaining names."""
    t = t.drop(columns=[c for c in NOTE_COLS if c in t])
    names = {}
    for c in t.columns:
        for old, new in RENAME:
            if c.startswith(old):
                names[c] = new + c[len(old):]
                break
    return t.rename(columns=names)


def number_format(name):
    if name.startswith("Learning rate") or name in ("Data (%)", "LoRA rank", "Seeds", "Seed"):
        return "General"
    if name.startswith("Trainable") or "(pts)" in name:
        return "0.00"
    if "(%)" in name or name.startswith(("Peak memory", "Time per epoch", "Training time")):
        return "0.0"
    return "General"


def style_sheet(ws, t, name):
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    n, cols = len(t), list(t.columns)
    letter = {c: get_column_letter(i) for i, c in enumerate(cols, 1)}
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor=GREY)
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[1].height = 48
    for c in cols:
        fmt_ = number_format(c)
        values = [fmt(v) for v in t[c].head(300)]
        width = max([len(w) for w in str(c).split()] + [len(v) for v in values])
        ws.column_dimensions[letter[c]].width = min(30, max(10, width + 2))
        for r in range(2, n + 2):
            ws[f"{letter[c]}{r}"].number_format = fmt_
    for c in ("Data (%)", "LoRA rank", "Seeds", "Seed", "Selected", "Learning rate"):
        if c in letter:
            for r in range(2, n + 2):
                ws[f"{letter[c]}{r}"].alignment = Alignment(horizontal="center")
    ws.page_setup.orientation = "landscape"  # whole table on one page width when printed or exported
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    group, best, freeze = SHEETS.get(name, ([], None, "A2"))
    ws.freeze_panes = freeze
    if n:
        ws.auto_filter.ref = f"A1:{letter[cols[-1]]}{n + 1}"
    # a line above each new group (model, dataset, data size...)
    if group:
        keys = t[group].astype(str).agg("|".join, axis=1).tolist()
        line = Side(style="medium", color=LINE)
        for i in range(1, n):
            if keys[i] != keys[i - 1]:
                for c in cols:
                    ws[f"{letter[c]}{i + 2}"].border = Border(top=line)
        # best score in each group in bold
        if best in t:
            vals = t[best].tolist()
            for k in set(keys):
                idx = [i for i in range(n) if keys[i] == k and vals[i] == vals[i]]
                if idx:
                    top = max(vals[i] for i in idx)
                    for i in idx:
                        if vals[i] == top:
                            ws[f"{letter[best]}{i + 2}"].font = Font(bold=True)
    # forgetting: white (none) to red (most)
    for c in cols:
        if c.startswith("Forgetting (pts)") and not c.endswith(" std") and n:
            ws.conditional_formatting.add(f"{letter[c]}2:{letter[c]}{n + 1}", ColorScaleRule(
                start_type="num", start_value=0, start_color="FFFFFF", end_type="max", end_color=RED))
    # selected learning rates highlighted
    if "Selected" in t:
        for i, v in enumerate(t["Selected"].tolist()):
            if v == "yes":
                for c in cols:
                    ws[f"{letter[c]}{i + 2}"].fill = PatternFill("solid", fgColor=GREEN)
                ws[f"{letter['Selected']}{i + 2}"].font = Font(bold=True)


def write_xlsx(path, tables):
    from openpyxl.styles import Alignment, Font, PatternFill
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, t in tables:
            tt = t.copy()
            for c in tt.select_dtypes("number").columns:
                if not c.startswith("Learning rate"):  # 3e-05 would round to 0
                    tt[c] = tt[c].round(3)
            tt.to_excel(xw, sheet_name=name, index=False)
            style_sheet(xw.sheets[name], tt, name)
        notes = pd.DataFrame(NOTES, columns=["Column", "Classification", "Detection"])
        notes.to_excel(xw, sheet_name="Notes", index=False)
        ws = xw.sheets["Notes"]
        for col, w in zip("ABC", (28, 60, 60)):
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor=GREY)
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True


def simplify_for_excel(t, keep_std, keep_bn):
    """By default the Excel file shows means only ('Score (%) mean' becomes 'Score (%)')
    and no BatchNorm columns; the Markdown file keeps everything."""
    t = t.copy()
    if not keep_bn:
        t = t.drop(columns=[c for c in t.columns if c.startswith("Forgetting, BN stats restored")])
    if not keep_std:
        t = t.drop(columns=[c for c in t.columns if c.endswith(" std")])
        t = t.rename(columns={c: c[:-5] for c in t.columns if c.endswith(" mean")})
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default=None, help="default: <results-dir>/report")
    p.add_argument("--include-dry-run", action="store_true")
    p.add_argument("--xlsx-std", action="store_true", help="also put standard deviations in the Excel file")
    p.add_argument("--xlsx-bn", action="store_true", help="also put the BatchNorm check in the Excel file")
    a = p.parse_args()
    R, out = a.results_dir, a.out_dir or os.path.join(a.results_dir, "report")
    os.makedirs(out, exist_ok=True)

    def rd(name):
        return read(os.path.join(R, name), a.include_dry_run)

    def js(name):
        f = os.path.join(R, name)
        return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}

    cls, det = rd("results.csv"), rd("det_results.csv")
    tables = [
        ("Main results", main_table(cls, det)),
        ("Rank sweeps", rank_table(cls, det, rd("det_rank.csv"))),
        ("LR tuning", lr_table(rd("lr_grid.csv"), rd("det_lr_grid.csv"), js("best_lrs.json"), js("det_best_lrs.json"))),
        ("BatchNorm check", bn_table(rd("bn_check.csv"))),
    ]
    missing = [n for n, t in tables if t is None]
    tables = [(n, tidy(t)) for n, t in tables if t is not None]
    if not tables:
        raise SystemExit(f"no result files found in {R}")

    xlsx = os.path.join(out, "all_results.xlsx")
    write_xlsx(xlsx, [(n, simplify_for_excel(t, a.xlsx_std, a.xlsx_bn)) for n, t in tables
                      if a.xlsx_bn or n != "BatchNorm check"])
    md = ["# All results", "",
          "Mean \u00b1 standard deviation over seeds. Accuracy or mAP: top-1 accuracy (classification) or "
          "mAP@[.5:.95] on VOC 2007 (detection). Macro-F1 or mAP@0.5: the second metric of each task. "
          "Forgetting, in points: drop in ImageNetV2 top-1 accuracy (classification) or COCO val2017 "
          "mAP@[.5:.95] (detection), with the original heads put back."]
    for n, t in tables:
        md += ["", f"## {n}", "", to_markdown(t)]
    with open(os.path.join(out, "all_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    for n, t in tables:
        print(f"{n}: {len(t)} rows")
    if missing:
        print("skipped (no data): " + ", ".join(missing))
    print(f"written: {xlsx} and {os.path.join(out, 'all_results.md')}")


if __name__ == "__main__":
    main()
