#!/usr/bin/env python
"""LoRA vs full fine-tuning vs linear probe for image classification.

One call = one run (model, dataset, data fraction, method, seed, learning rate).
The result is appended to a CSV as soon as the run finishes, so a crash or a
session timeout loses only the run in progress.

Example:
    python train_cls.py --model vit_base_patch16_224 --dataset flowers102 \
        --method lora --lr 1e-3 --seed 0 --forget-root /data/imagenetv2

Forgetting is measured by putting the original ImageNet classifier back on the
(fine-tuned) backbone and testing it on --forget-root, an ImageFolder of
ImageNet-class images (ImageNetV2 or the ImageNet validation set).
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from torch.utils.data import DataLoader, Subset

DEFAULT_EPOCHS = {"flowers102": 30, "pets": 20, "eurosat": 10}

FIELDS = [
    "run_id", "label", "timestamp", "model", "dataset", "fraction", "method", "seed",
    "lr", "weight_decay", "epochs", "steps", "batch_size", "rank", "alpha", "lora_dropout",
    "n_train", "n_val", "n_test", "val_acc_best", "test_acc", "test_macro_f1",
    "forget_before", "forget_after", "forget_adapters_off", "forgetting", "rel_forgetting",
    "forget_after_bn_restored", "forgetting_bn_restored",
    "trainable_params", "total_params", "trainable_pct", "trainable_mb",
    "peak_mem_gb", "epoch_time_s", "train_time_s", "lora_init_maxdiff", "dry_run",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_base_patch16_224", help="any timm model name")
    p.add_argument("--dataset", default="flowers102", choices=["flowers102", "pets", "eurosat"])
    p.add_argument("--method", default="lora", choices=["linear", "lora", "full"])
    p.add_argument("--fraction", type=float, default=1.0, help="share of the training split to use")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=None, help="default depends on the dataset")
    p.add_argument("--min-steps", type=int, default=300, help="raise epochs until at least this many steps")
    p.add_argument("--patience", type=int, default=0, help="early-stopping patience in epochs; 0 = off")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=None, help="LoRA alpha; default 2 x rank")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default=None, help="regex of module names; default chosen per model")
    p.add_argument("--linear-epochs", type=int, default=100, help="epochs for the head on cached features")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--forget-root", default=None, help="ImageFolder of ImageNet-class images; omit to skip")
    p.add_argument("--forget-max-images", type=int, default=10000)
    p.add_argument("--split-seed", type=int, default=0, help="fixes train/val/test splits across runs")
    p.add_argument("--out", default="results/results.csv")
    p.add_argument("--label", default="main", help="free tag stored with the run, e.g. lr_grid / main / rank")
    p.add_argument("--save-dir", default=None, help="save LoRA adapters and heads here")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--overfit", type=int, default=0, help="sanity check: train and test on N training images")
    p.add_argument("--dry-run", action="store_true", help="fake data, random weights: tests the code path only")
    return p.parse_args()


# ---------------------------------------------------------------- utilities

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_id(a):
    return f"{a.label}|{a.model}|{a.dataset}|f{a.fraction}|{a.method}|r{a.rank}|lr{a.lr}|s{a.seed}"


def append_row(path, row):
    """Append one result row. A CSV written by an older version of this script (fewer columns)
    is rewritten once with the current header, keeping all its rows, so columns never misalign."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            old_rows = list(reader) if header != FIELDS else None
        if old_rows is not None:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                for r in old_rows:
                    w.writerow({k: r.get(k, "") or "" for k in FIELDS})
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def count_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def feats(base, x):
    """Pooled features just before the classifier (works for timm ViTs, ResNets, ConvNeXts)."""
    return base.forward_head(base.forward_features(x), pre_logits=True)


def classifier_name(model):
    name = getattr(model, "pretrained_cfg", {}).get("classifier")
    if isinstance(name, str) and "." not in name and hasattr(model, name):
        return name
    for n in ("head", "fc", "classifier"):
        if hasattr(model, n):
            return n
    raise ValueError("cannot find the classifier layer; pass a model with .head or .fc")


def default_lora_targets(model_name):
    m = model_name.lower()
    if any(k in m for k in ("vit", "deit", "beit", "swin", "eva")):
        return r".*\.attn\.(qkv|proj)"          # attention projections only; not patch_embed.proj
    if "resnet" in m:
        return r"layer\d\.\d+\.conv[123]"         # every conv in the residual blocks
    if "convnext" in m:
        return r"stages\.\d\.blocks\.\d+\.mlp\.fc[12]"
    raise ValueError(f"no default LoRA targets for {model_name}; pass --lora-targets")


def set_frozen_bn_eval(module):
    """Frozen BatchNorm must stay in eval mode, or its running stats drift and fake 'forgetting'."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()


def snapshot_bn_stats(module):
    """Copies of every BatchNorm layer's running mean and variance (empty list for models without BN)."""
    snap = []
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
            snap.append((m, m.running_mean.detach().clone(), m.running_var.detach().clone()))
    return snap


@torch.no_grad()
def swap_bn_stats(snap):
    """Exchange each BatchNorm layer's current running stats with the saved copy.
    Calling it twice puts everything back as it was."""
    for m, mean, var in snap:
        cur_mean, cur_var = m.running_mean.clone(), m.running_var.clone()
        m.running_mean.copy_(mean)
        m.running_var.copy_(var)
        mean.copy_(cur_mean)
        var.copy_(cur_var)


def metrics_from_preds(y, pred, num_classes):
    idx = y.long().cpu() * num_classes + pred.long().cpu()
    conf = torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)
    acc = conf.diag().sum().item() / max(1, conf.sum().item())
    tp = conf.diag().float()
    prec = tp / conf.sum(0).float().clamp(min=1)
    rec = tp / conf.sum(1).float().clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    present = conf.sum(1) > 0
    return acc, f1[present].mean().item()


@torch.no_grad()
def evaluate(fn, loader, device, num_classes, amp, amp_dtype):
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            logits = fn(x)
        ps.append(logits.float().argmax(1).cpu())
        ys.append(torch.as_tensor(y))
    return metrics_from_preds(torch.cat(ys), torch.cat(ps), num_classes)


@torch.no_grad()
def extract(base, loader, device, amp, amp_dtype):
    fs, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            fs.append(feats(base, x.to(device, non_blocking=True)).float().cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(fs), torch.cat(ys)


def make_loader(ds, bs, shuffle, workers):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      drop_last=shuffle and len(ds) > bs)


def cosine_with_warmup(opt, total_steps, warmup_frac=0.05):
    warmup = max(1, int(warmup_frac * total_steps))

    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ---------------------------------------------------------------- data

def get_targets(ds):
    for attr in ("targets", "_labels"):
        if hasattr(ds, attr):
            return list(getattr(ds, attr))
    raise AttributeError(f"{type(ds).__name__} has no targets")


def stratified_split(targets, fractions, seed):
    """Split indices into parts with the given fractions, keeping class proportions."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    parts = [[] for _ in fractions]
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        cuts = np.round(np.cumsum(fractions) * len(idx)).astype(int)
        start = 0
        for i, end in enumerate(cuts):
            parts[i].extend(idx[start:end].tolist())
            start = end
    return [sorted(p) for p in parts]


def stratified_subsample(indices, targets, fraction, seed):
    if fraction >= 1.0:
        return list(indices)
    rng = np.random.RandomState(seed)
    targets, indices = np.asarray(targets), np.asarray(indices)
    out = []
    for c in np.unique(targets[indices]):
        idx = indices[targets[indices] == c]
        k = max(1, int(round(fraction * len(idx))))
        out.extend(rng.choice(idx, k, replace=False).tolist())
    return sorted(out)


def build_datasets(args, train_tf, eval_tf):
    """Returns train base (augmented), train base (eval transform), train indices and targets, val, test, classes."""
    root = args.data_root
    if args.dry_run:
        def fake(n, tf, offset):
            return torchvision.datasets.FakeData(size=n, image_size=(3, 224, 224), num_classes=10,
                                                 transform=tf, random_offset=offset)
        tr, tr_eval = fake(64, train_tf, 0), fake(64, eval_tf, 0)
        return tr, tr_eval, list(range(64)), None, fake(32, eval_tf, 1000), fake(32, eval_tf, 2000), 10

    if args.dataset == "flowers102":
        D = torchvision.datasets.Flowers102
        tr = D(root, split="train", transform=train_tf, download=True)
        tr_eval = D(root, split="train", transform=eval_tf, download=True)
        val = D(root, split="val", transform=eval_tf, download=True)
        test = D(root, split="test", transform=eval_tf, download=True)
        return tr, tr_eval, list(range(len(tr))), get_targets(tr), val, test, 102

    if args.dataset == "pets":
        D = torchvision.datasets.OxfordIIITPet
        tr = D(root, split="trainval", target_types="category", transform=train_tf, download=True)
        tr_eval = D(root, split="trainval", target_types="category", transform=eval_tf, download=True)
        test = D(root, split="test", target_types="category", transform=eval_tf, download=True)
        targets = get_targets(tr)
        tr_idx, va_idx = stratified_split(targets, [0.85, 0.15], args.split_seed)
        return tr, tr_eval, tr_idx, targets, Subset(tr_eval, va_idx), test, 37

    if args.dataset == "eurosat":
        D = torchvision.datasets.EuroSAT
        tr = D(root, transform=train_tf, download=True)
        tr_eval = D(root, transform=eval_tf, download=True)
        targets = get_targets(tr)
        tr_idx, va_idx, te_idx = stratified_split(targets, [0.7, 0.1, 0.2], args.split_seed)
        return tr, tr_eval, tr_idx, targets, Subset(tr_eval, va_idx), Subset(tr_eval, te_idx), 10

    raise ValueError(args.dataset)


def build_forget_loader(args, eval_tf):
    if args.dry_run:
        ds = torchvision.datasets.FakeData(size=32, image_size=(3, 224, 224), num_classes=1000,
                                           transform=eval_tf, random_offset=5000)
        return make_loader(ds, args.batch_size, False, args.workers)
    ds = torchvision.datasets.ImageFolder(args.forget_root, transform=eval_tf)
    if all(c.isdigit() for c in ds.classes):
        # ImageNetV2 layout: folder name is the class index ("0".."999"); ImageFolder sorts
        # names as text ("0", "1", "10", ...), so map each folder back to its integer index.
        mapping = [int(c) for c in ds.classes]
        ds.samples = [(path, mapping[t]) for path, t in ds.samples]
        ds.targets = [t for _, t in ds.samples]
    elif len(ds.classes) != 1000:
        print(f"WARNING: forget set has {len(ds.classes)} classes; expected 1000 ImageNet classes")
    if len(ds) > args.forget_max_images:
        idx = np.random.RandomState(0).choice(len(ds), args.forget_max_images, replace=False)
        ds = Subset(ds, sorted(idx.tolist()))  # same fixed subset for every run
    return make_loader(ds, args.batch_size, False, args.workers)


def forget_baseline(args, model, orig_head, loader, device, amp, amp_dtype):
    """Original-task accuracy of the pretrained model, cached per model and forget set."""
    path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "forget_baselines.json")
    key = f"{args.model}|{os.path.abspath(args.forget_root or 'fake')}|{args.forget_max_images}|dry{args.dry_run}"
    cache = json.load(open(path)) if os.path.exists(path) else {}
    if key in cache:
        return cache[key]
    model.eval()
    acc, _ = evaluate(lambda x: orig_head(feats(model, x)), loader, device, 1000, amp, amp_dtype)
    cache = json.load(open(path)) if os.path.exists(path) else {}
    cache[key] = acc
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(cache, open(path, "w"), indent=1)
    return acc


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    if args.alpha is None:
        args.alpha = 2 * args.rank
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = use_cuda and not args.no_amp
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)

    # Model and its original ImageNet classifier (kept aside for the forgetting test).
    model = timm.create_model(args.model, pretrained=not args.dry_run)
    data_cfg = timm.data.resolve_data_config({}, model=model)
    train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    eval_tf = timm.data.create_transform(**data_cfg)
    orig_head = copy.deepcopy(model.get_classifier()).to(device).eval()
    for p in orig_head.parameters():
        p.requires_grad = False
    head_name = classifier_name(model)

    # Data.
    tr, tr_eval, tr_idx, tr_targets, val, test, C = build_datasets(args, train_tf, eval_tf)
    sub = tr_idx if args.dry_run else stratified_subsample(tr_idx, tr_targets, args.fraction, args.seed)
    if args.overfit:
        sub = sub[:args.overfit]
        val = test = Subset(tr_eval, sub)
    train_set, train_eval_set = Subset(tr, sub), Subset(tr_eval, sub)
    train_loader = make_loader(train_set, args.batch_size, True, args.workers)
    val_loader = make_loader(val, args.batch_size, False, args.workers)
    test_loader = make_loader(test, args.batch_size, False, args.workers)

    # Forgetting baseline, measured on the untouched pretrained model.
    model.to(device)
    forget_loader = build_forget_loader(args, eval_tf) if (args.forget_root or args.dry_run) else None
    before = forget_baseline(args, model, orig_head, forget_loader, device, amp, amp_dtype) if forget_loader else None
    # Pretrained BatchNorm statistics, kept to split full fine-tuning's forgetting into
    # "BN statistics adapted to the new data" and "weights changed" (empty for models without BN).
    bn_snap = snapshot_bn_stats(model) if (forget_loader is not None and args.method == "full") else []

    model.reset_classifier(C)
    model.to(device)
    total_params = count_params(model)

    row = dict(run_id=run_id(args), label=args.label, timestamp=datetime.now().isoformat(timespec="seconds"),
               model=args.model, dataset=args.dataset, fraction=args.fraction, method=args.method,
               seed=args.seed, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
               n_train=len(train_set), n_val=len(val), n_test=len(test), total_params=total_params,
               dry_run=args.dry_run)
    if args.method == "lora":
        row.update(rank=args.rank, alpha=args.alpha, lora_dropout=args.lora_dropout)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    if args.method == "linear":
        # Frozen backbone: extract features once (no augmentation), then train only the head.
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        base, net = model, None
        ftr, ytr = extract(model, make_loader(train_eval_set, args.batch_size, False, args.workers), device, amp, amp_dtype)
        fva, yva = extract(model, val_loader, device, amp, amp_dtype)
        fte, yte = extract(model, test_loader, device, amp, amp_dtype)
        ftr, ytr, fva, fte = ftr.to(device), ytr.to(device), fva.to(device), fte.to(device)
        head = nn.Linear(ftr.shape[1], C).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bs = 256
        steps = args.linear_epochs * math.ceil(len(ftr) / bs)
        sched = cosine_with_warmup(opt, steps)
        best_val, best_state = -1.0, None
        for _ in range(args.linear_epochs):
            perm = torch.randperm(len(ftr), device=device)
            for i in range(0, len(ftr), bs):
                j = perm[i:i + bs]
                loss = F.cross_entropy(head(ftr[j]), ytr[j])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            with torch.no_grad():
                v, _ = metrics_from_preds(yva, head(fva).argmax(1), C)
            if v > best_val:
                best_val, best_state = v, copy.deepcopy(head.state_dict())
        head.load_state_dict(best_state)
        with torch.no_grad():
            test_acc, test_f1 = metrics_from_preds(yte, head(fte).argmax(1), C)
        trainable = count_params(head)
        row.update(epochs=args.linear_epochs, steps=steps, val_acc_best=best_val)
        train_time = time.time() - t_start
        epoch_times = []
    else:
        if args.method == "lora":
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.lora_dropout, bias="none",
                             target_modules=args.lora_targets or default_lora_targets(args.model),
                             modules_to_save=[head_name])
            net = get_peft_model(model, cfg).to(device)
            base = net.get_base_model()
            # B starts at zero, so the adapted backbone must match the pretrained one exactly.
            net.eval()
            x0 = next(iter(val_loader))[0][:8].to(device)
            with torch.no_grad():
                a = feats(base, x0)
                with net.disable_adapter():
                    b = feats(base, x0)
            row["lora_init_maxdiff"] = (a - b).abs().max().item()
        else:
            for p in model.parameters():
                p.requires_grad = True
            net = base = model

        params = [p for p in net.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        epochs = args.epochs or DEFAULT_EPOCHS[args.dataset]
        epochs = max(epochs, math.ceil(args.min_steps / max(1, len(train_loader))))
        steps = epochs * len(train_loader)
        sched = cosine_with_warmup(opt, steps)

        best_val, best_state, bad, epoch_times = -1.0, None, 0, []
        for ep in range(epochs):
            net.train()
            set_frozen_bn_eval(net)
            t0 = time.time()
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), torch.as_tensor(y).to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                    loss = F.cross_entropy(net(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            if use_cuda:
                torch.cuda.synchronize()
            epoch_times.append(time.time() - t0)
            net.eval()
            v, _ = evaluate(net, val_loader, device, C, amp, amp_dtype)
            print(f"epoch {ep + 1}/{epochs}  loss {loss.item():.4f}  val_acc {v:.4f}  {epoch_times[-1]:.1f}s", flush=True)
            if v > best_val:
                best_val, bad = v, 0
                if args.method == "lora":   # only adapters and head change
                    best_state = {n: p.detach().cpu().clone() for n, p in net.named_parameters() if p.requires_grad}
                else:
                    best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if args.patience and bad >= args.patience:
                    print("early stop", flush=True)
                    break
        net.load_state_dict(best_state, strict=args.method != "lora")
        train_time = time.time() - t_start
        net.eval()
        test_acc, test_f1 = evaluate(net, test_loader, device, C, amp, amp_dtype)
        row.update(epochs=len(epoch_times), steps=len(epoch_times) * len(train_loader), val_acc_best=best_val)

    peak = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
    row.update(test_acc=test_acc, test_macro_f1=test_f1, trainable_params=trainable,
               trainable_pct=100.0 * trainable / total_params, trainable_mb=trainable * 4 / 1e6,
               peak_mem_gb=peak, train_time_s=train_time,
               epoch_time_s=float(np.mean(epoch_times)) if epoch_times else "")

    # Forgetting: original ImageNet head on the (possibly changed) backbone.
    if forget_loader is not None:
        base.eval()
        fn = lambda x: orig_head(feats(base, x))
        after, _ = evaluate(fn, forget_loader, device, 1000, amp, amp_dtype)
        row.update(forget_before=before, forget_after=after, forgetting=before - after,
                   rel_forgetting=100.0 * (before - after) / max(before, 1e-12))
        if args.method == "lora":
            with net.disable_adapter():
                off, _ = evaluate(fn, forget_loader, device, 1000, amp, amp_dtype)
            row["forget_adapters_off"] = off
        if bn_snap:
            # Fine-tuned weights, but the pretrained BatchNorm statistics put back.
            swap_bn_stats(bn_snap)
            restored, _ = evaluate(fn, forget_loader, device, 1000, amp, amp_dtype)
            swap_bn_stats(bn_snap)  # back to the fine-tuned statistics
            row.update(forget_after_bn_restored=restored, forgetting_bn_restored=before - restored)

    if args.save_dir:
        d = os.path.join(args.save_dir, run_id(args).replace("|", "_"))
        os.makedirs(d, exist_ok=True)
        if args.method == "lora":
            net.save_pretrained(d)
        elif args.method == "linear":
            torch.save(head.state_dict(), os.path.join(d, "head.pt"))

    append_row(args.out, row)
    print(json.dumps({k: row.get(k) for k in ("run_id", "val_acc_best", "test_acc", "test_macro_f1",
                                              "forgetting", "trainable_pct", "peak_mem_gb", "train_time_s")},
                     default=str), flush=True)


if __name__ == "__main__":
    main()
