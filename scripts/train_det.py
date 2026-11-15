#!/usr/bin/env python
"""LoRA vs full fine-tuning vs head-only training for object detection.

Model:  RT-DETR (Hugging Face transformers), pretrained on COCO (80 classes).
Target: Pascal VOC 2007 (20 classes). VOC 2007 trainval is split into train (90%) and val (10%);
        VOC 2007 test is the test set. Difficult objects are ignored in evaluation.
Methods:
    head  only the class heads are trained (a new 20-class head); everything else frozen
    lora  LoRA adapters on every linear layer of the transformer encoder/decoder and the
          box-prediction MLPs, plus the new class heads; convolutions frozen
    full  every weight trained; backbone at --backbone-lr-mult x lr (the standard RT-DETR recipe)
Forgetting: COCO val2017 mAP@[.5:.95] with the original COCO class heads put back.

One call = one run; the result is appended to a CSV as soon as the run finishes.

Example:
    python train_det.py --method lora --lr 3e-4 --fraction 0.1 --seed 0 \
        --voc-root data --coco-root data/coco
"""
import argparse
import contextlib
import csv
import io
import json
import math
import os
import random
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]

FIELDS = [
    "run_id", "label", "timestamp", "model", "dataset", "fraction", "method", "seed",
    "lr", "backbone_lr_mult", "weight_decay", "epochs", "steps", "batch_size", "img_size", "precision",
    "rank", "alpha", "lora_dropout", "n_train", "n_val", "n_test",
    "val_map_best", "test_map", "test_map50", "test_map_small", "test_map_medium", "test_map_large",
    "forget_before", "forget_after", "forget_adapters_off", "forgetting", "rel_forgetting",
    "trainable_params", "total_params", "trainable_pct", "trainable_mb",
    "peak_mem_gb", "epoch_time_s", "train_time_s", "dry_run",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="PekingU/rtdetr_r18vd", help="Hugging Face RT-DETR checkpoint (COCO)")
    p.add_argument("--method", default="lora", choices=["head", "lora", "full"])
    p.add_argument("--fraction", type=float, default=1.0, help="share of the VOC train split to use")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1, help="full fine-tuning only")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--min-steps", type=int, default=2000, help="raise epochs until at least this many steps")
    p.add_argument("--evals", type=int, default=8, help="validation evaluations spread over training")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--precision", default="tf32", choices=["tf32", "bf16", "fp32"],
                   help="tf32: fp32 weights with TF32 matmuls (safe default); bf16: faster mixed precision")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=None, help="LoRA alpha; default 2 x rank")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--voc-root", default="./data", help="folder that contains VOCdevkit/VOC2007")
    p.add_argument("--coco-root", default=None, help="folder with val2017/ and annotations/; omit to skip forgetting")
    p.add_argument("--coco-max-images", type=int, default=5000)
    p.add_argument("--val-frac", type=float, default=0.1, help="share of VOC07 trainval held out as validation")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--out", default="results/det_results.csv")
    p.add_argument("--label", default="main")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--adapters-off-check", action="store_true",
                   help="LoRA: also measure COCO mAP with adapters disabled (must equal forget_before)")
    p.add_argument("--overfit", type=int, default=0, help="sanity check: train and test on N training images")
    p.add_argument("--dry-run", action="store_true", help="fake data, small random model: tests the code path only")
    return p.parse_args()


# ---------------------------------------------------------------- utilities

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_id(a):
    return f"{a.label}|{a.model}|voc07|f{a.fraction}|{a.method}|r{a.rank}|lr{a.lr}|s{a.seed}"


def append_row(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):  # rewrite once if the file has an older header
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old = list(reader) if (reader.fieldnames or []) != FIELDS else None
        if old is not None:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                for r in old:
                    w.writerow({k: r.get(k, "") or "" for k in FIELDS})
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def set_frozen_bn_eval(module):
    """Frozen BatchNorm must stay in eval mode, or its running stats drift and fake 'forgetting'."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()


def patch_matcher_for_bf16():
    """RT-DETR's Hungarian matcher hands a torch tensor to scipy; scipy cannot read bfloat16."""
    try:
        import transformers.loss.loss_rt_detr as L
    except Exception:
        return
    orig = L.linear_sum_assignment
    if getattr(orig, "_float_patch", False):
        return

    def lsa(c):
        return orig(c.float() if torch.is_tensor(c) else c)

    lsa._float_patch = True
    L.linear_sum_assignment = lsa


def cosine_with_warmup(opt, total_steps, warmup_frac=0.05):
    warmup = max(1, int(warmup_frac * total_steps))

    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ---------------------------------------------------------------- data

def parse_voc(root, image_set):
    base = os.path.join(root, "VOCdevkit", "VOC2007")
    ids_file = os.path.join(base, "ImageSets", "Main", f"{image_set}.txt")
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"{ids_file} not found: --voc-root must contain VOCdevkit/VOC2007 "
                                f"(extract VOCtrainval_06-Nov-2007.tar and VOCtest_06-Nov-2007.tar there)")
    recs = []
    for k, name in enumerate(open(ids_file).read().split()):
        tree = ET.parse(os.path.join(base, "Annotations", f"{name}.xml")).getroot()
        size = tree.find("size")
        w, h = int(size.find("width").text), int(size.find("height").text)
        boxes, labels, diff = [], [], []
        for o in tree.findall("object"):
            bb = o.find("bndbox")
            boxes.append([float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")])
            boxes[-1][0] -= 1  # VOC pixel coordinates are 1-based
            boxes[-1][1] -= 1
            labels.append(VOC_CLASSES.index(o.find("name").text.strip()))
            d = o.find("difficult")
            diff.append(int(d.text) if d is not None else 0)
        recs.append(dict(id=k, path=os.path.join(base, "JPEGImages", f"{name}.jpg"), w=w, h=h,
                         boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                         labels=np.array(labels, dtype=np.int64), difficult=np.array(diff, dtype=np.int64)))
    return recs


def fake_records(n, num_classes, offset):
    rng = np.random.RandomState(offset)
    recs = []
    for k in range(n):
        w, h = int(rng.randint(200, 400)), int(rng.randint(200, 400))
        m = int(rng.randint(1, 4))
        x1, y1 = rng.uniform(0, w * 0.6, m), rng.uniform(0, h * 0.6, m)
        bw, bh = rng.uniform(20, w * 0.4, m), rng.uniform(20, h * 0.4, m)
        boxes = np.stack([x1, y1, x1 + bw, y1 + bh], 1).astype(np.float32)
        recs.append(dict(id=offset + k, path=None, w=w, h=h, boxes=boxes,
                         labels=rng.randint(0, num_classes, m).astype(np.int64),
                         difficult=np.zeros(m, dtype=np.int64)))
    return recs


class DetDataset(Dataset):
    """Images resized to img_size x img_size; training targets in normalized cx, cy, w, h."""

    def __init__(self, recs, img_size, train):
        self.recs, self.size, self.train = recs, img_size, train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        if r["path"] is None:  # dry run
            g = torch.Generator().manual_seed(r["id"])
            img = TF.to_pil_image(torch.rand(3, r["h"], r["w"], generator=g))
        else:
            img = Image.open(r["path"]).convert("RGB")
        w, h = img.size
        boxes = torch.as_tensor(r.get("boxes", np.zeros((0, 4), np.float32))).clone()
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        x = TF.to_tensor(TF.resize(img, [self.size, self.size]))
        meta = {"image_id": r["id"], "orig_size": (h, w)}
        if self.train:
            keep = torch.as_tensor(r["difficult"]) == 0
            b = boxes[keep]
            b[:, [0, 2]] /= w
            b[:, [1, 3]] /= h
            b = b.clamp(0, 1)
            cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                                  b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            meta["class_labels"] = torch.as_tensor(r["labels"])[keep]
            meta["boxes"] = cxcywh
        return x, meta


def collate(batch):
    return torch.stack([x for x, _ in batch]), [m for _, m in batch]


def make_loader(ds, bs, shuffle, workers):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      drop_last=shuffle and len(ds) > bs)


# ---------------------------------------------------------------- evaluation (pycocotools)

def coco_from_records(recs, class_names):
    """COCO ground truth from records; difficult objects become iscrowd, so they are ignored."""
    from pycocotools.coco import COCO
    images, anns, k = [], [], 1
    for r in recs:
        images.append({"id": int(r["id"]), "width": r["w"], "height": r["h"]})
        for b, lab, d in zip(r["boxes"], r["labels"], r["difficult"]):
            bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
            anns.append({"id": k, "image_id": int(r["id"]), "category_id": int(lab) + 1,
                         "bbox": [float(b[0]), float(b[1]), bw, bh], "area": bw * bh, "iscrowd": int(d)})
            k += 1
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


@torch.no_grad()
def predict(fn, loader, device, autocast, topk=100):
    """RT-DETR post-processing: sigmoid scores, top-k over all (query, class) pairs."""
    dets = []
    for x, metas in loader:
        with autocast():
            out = fn(x.to(device, non_blocking=True))
        prob, boxes = out.logits.float().sigmoid(), out.pred_boxes.float()
        B, Q, C = prob.shape
        scores, idx = prob.flatten(1).topk(min(topk, Q * C), dim=1)
        labels, q = idx % C, idx // C
        b = boxes.gather(1, q.unsqueeze(-1).expand(-1, -1, 4))
        xyxy = torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1)
        for i, m in enumerate(metas):
            h, w = m["orig_size"]
            bb = (xyxy[i] * torch.tensor([w, h, w, h], device=xyxy.device)).cpu().numpy()
            dets.append((m["image_id"], bb, scores[i].cpu().numpy(), labels[i].cpu().numpy()))
    return dets


def coco_map(gt, dets, cat_ids, img_ids):
    """Returns [mAP, mAP50, mAP75, small, medium, large] (fractions); cat_ids maps label index -> category id."""
    from pycocotools.cocoeval import COCOeval
    res = []
    for image_id, bb, sc, lab in dets:
        for b, s, l in zip(bb, sc, lab):
            res.append({"image_id": int(image_id), "category_id": int(cat_ids[int(l)]),
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                        "score": float(s)})
    if not res:
        return [0.0] * 6
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = sorted(int(i) for i in img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return [max(0.0, float(s)) for s in ev.stats[:6]]


# ---------------------------------------------------------------- model and class heads

def load_model(args):
    from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrResNetConfig
    if args.dry_run:  # small random RT-DETR-R18-like model, nothing downloaded
        bb = RTDetrResNetConfig(depths=[2, 2, 2, 2], layer_type="basic", hidden_sizes=[64, 128, 256, 512],
                                out_features=["stage2", "stage3", "stage4"])
        id2label = {i: f"class{i}" for i in range(80)}
        cfg = RTDetrConfig(backbone_config=bb, encoder_in_channels=[128, 256, 512], decoder_layers=3,
                           id2label=id2label, label2id={v: k for k, v in id2label.items()})
        return RTDetrForObjectDetection(cfg)
    return RTDetrForObjectDetection.from_pretrained(args.model)


def find_class_heads(model, num_labels):
    """All class-prediction modules: Linear(d, num_labels) and the denoising Embedding(num_labels + 1, d).
    remove_duplicate=False also returns every second path to a shared module."""
    heads = {}
    for name, mod in model.named_modules(remove_duplicate=False):
        if not any(k in name for k in ("class", "score")):
            continue
        if isinstance(mod, nn.Linear) and mod.out_features == num_labels:
            heads[name] = mod
        elif isinstance(mod, nn.Embedding) and mod.num_embeddings == num_labels + 1:
            heads[name] = mod
    if not any(isinstance(m, nn.Linear) for m in heads.values()):
        raise RuntimeError(f"no class heads with {num_labels} outputs found")
    return heads


def new_heads_like(old_heads, num_new, prior_prob=0.01):
    """Fresh heads for num_new classes; modules shared between paths stay shared."""
    bias = -math.log((1 - prior_prob) / prior_prob)  # focal-loss prior, as RT-DETR initializes its heads
    made, out = {}, {}
    for name, mod in old_heads.items():
        if id(mod) not in made:
            if isinstance(mod, nn.Linear):
                m = nn.Linear(mod.in_features, num_new)
                nn.init.constant_(m.bias, bias)
            else:
                m = nn.Embedding(num_new + 1, mod.embedding_dim, padding_idx=num_new)
                nn.init.xavier_uniform_(m.weight)
                with torch.no_grad():
                    m.weight[num_new].zero_()
            made[id(mod)] = m.to(mod.weight.device)
        out[name] = made[id(mod)]
    return out


def set_module(root, name, new):
    parent_name, _, attr = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attr, new)


class HeadSwitch:
    """Puts either the COCO heads or the target-task heads into the model."""

    def __init__(self, model, coco_heads, target_heads, coco_id2label, target_id2label):
        self.model = model
        self.heads = {"coco": coco_heads, "target": target_heads}
        self.id2label = {"coco": dict(coco_id2label), "target": dict(target_id2label)}

    def use(self, which):
        for name, mod in self.heads[which].items():
            set_module(self.model, name, mod)
        self.model.config.id2label = self.id2label[which]
        self.model.config.label2id = {v: k for k, v in self.id2label[which].items()}


def coco_category_ids(model_id2label, coco_gt):
    """Maps the model's label index (0..79) to COCO category ids, by class name."""
    name2id = {c["name"]: c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    ids = [name2id.get(model_id2label[i]) for i in range(len(model_id2label))]
    if None in ids:
        print("WARNING: COCO class names did not match the model's labels; using sorted category ids")
        ids = sorted(coco_gt.getCatIds())
    return ids


# ---------------------------------------------------------------- COCO forgetting set

def build_forget(args, img_size):
    """Returns (COCO gt, loader, image ids) or None."""
    if args.dry_run:
        recs = fake_records(8, 80, 90000)
        gt = coco_from_records(recs, [f"class{i}" for i in range(80)])
        return gt, make_loader(DetDataset(recs, img_size, False), args.batch_size, False, args.workers), \
            [r["id"] for r in recs]
    if not args.coco_root:
        return None
    from pycocotools.coco import COCO
    ann = os.path.join(args.coco_root, "annotations", "instances_val2017.json")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann)
    ids = sorted(gt.getImgIds())[:args.coco_max_images]  # same fixed subset for every run
    recs = [dict(id=i, path=os.path.join(args.coco_root, "val2017", gt.imgs[i]["file_name"]),
                 w=gt.imgs[i]["width"], h=gt.imgs[i]["height"]) for i in ids]
    return gt, make_loader(DetDataset(recs, img_size, False), args.batch_size, False, args.workers), ids


def forget_baseline(args, fn, forget, cat_ids, device, autocast):
    path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "det_forget_baselines.json")
    key = (f"{args.model}|{os.path.abspath(args.coco_root or 'fake')}|{args.coco_max_images}|"
           f"{args.img_size}|{args.precision}|dry{args.dry_run}")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    if key in cache:
        return cache[key]
    gt, loader, ids = forget
    val = coco_map(gt, predict(fn, loader, device, autocast), cat_ids, ids)[0]
    cache = json.load(open(path)) if os.path.exists(path) else {}
    cache[key] = val
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(cache, open(path, "w"), indent=1)
    return val


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    if args.alpha is None:
        args.alpha = 2 * args.rank
    seed_all(args.seed)
    patch_matcher_for_bf16()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = args.precision != "fp32"
        torch.backends.cudnn.allow_tf32 = args.precision != "fp32"
    amp = args.precision == "bf16"

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp)

    # ---- data
    if args.dry_run:
        trainval, test_recs = fake_records(24, 20, 0), fake_records(8, 20, 50000)
    else:
        trainval, test_recs = parse_voc(args.voc_root, "trainval"), parse_voc(args.voc_root, "test")
    rng = np.random.RandomState(args.split_seed)
    order = rng.permutation(len(trainval))
    n_val = int(round(args.val_frac * len(trainval)))
    val_recs = [trainval[i] for i in sorted(order[:n_val])]
    # images whose objects are all "difficult" carry no training signal
    train_all = [trainval[i] for i in sorted(order[n_val:]) if (trainval[i]["difficult"] == 0).any()]
    if args.fraction < 1.0:
        k = max(1, int(round(args.fraction * len(train_all))))
        pick = np.random.RandomState(args.seed).choice(len(train_all), k, replace=False)
        train_recs = [train_all[i] for i in sorted(pick)]
    else:
        train_recs = train_all
    if args.overfit:
        train_recs = train_recs[:args.overfit]
        val_recs = test_recs = train_recs
    S = args.img_size
    train_loader = make_loader(DetDataset(train_recs, S, True), args.batch_size, True, args.workers)
    val_loader = make_loader(DetDataset(val_recs, S, False), args.batch_size, False, args.workers)
    test_loader = make_loader(DetDataset(test_recs, S, False), args.batch_size, False, args.workers)
    val_gt, test_gt = coco_from_records(val_recs, VOC_CLASSES), coco_from_records(test_recs, VOC_CLASSES)
    voc_cat_ids = list(range(1, len(VOC_CLASSES) + 1))

    # ---- model, original COCO heads, forgetting baseline
    model = load_model(args).to(device)
    coco_id2label = {int(k): v for k, v in model.config.id2label.items()}
    coco_heads = find_class_heads(model, len(coco_id2label))
    forget = build_forget(args, S)
    before = None
    if forget is not None:
        coco_cat_ids = coco_category_ids(coco_id2label, forget[0])
        model.eval()
        before = forget_baseline(args, lambda x: model(pixel_values=x), forget, coco_cat_ids, device, autocast)

    # ---- swap in new 20-class heads
    target_heads = new_heads_like(coco_heads, len(VOC_CLASSES))
    switch = HeadSwitch(model, coco_heads, target_heads, coco_id2label, dict(enumerate(VOC_CLASSES)))
    switch.use("target")
    total_params = sum(p.numel() for p in model.parameters())
    head_params = {id(p) for m in set(target_heads.values()) for p in m.parameters()}

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model
        head_names = set(target_heads)
        targets = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n not in head_names]
        cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.lora_dropout, bias="none",
                         target_modules=targets)
        net = get_peft_model(model, cfg)
        base = net.get_base_model()
        for m in set(target_heads.values()):
            for p in m.parameters():
                p.requires_grad = True
    else:
        net = base = model
        for p in model.parameters():
            p.requires_grad = args.method == "full" or id(p) in head_params

    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for _, p in named)
    if args.method == "full":
        bb = [p for n, p in named if "backbone" in n]
        rest = [p for n, p in named if "backbone" not in n]
        groups = [{"params": rest, "lr": args.lr}, {"params": bb, "lr": args.lr * args.backbone_lr_mult}]
    else:
        groups = [{"params": [p for _, p in named], "lr": args.lr}]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    epochs = max(args.epochs, math.ceil(args.min_steps / max(1, len(train_loader))))
    steps = epochs * len(train_loader)
    sched = cosine_with_warmup(opt, steps)
    n_ev = max(1, min(args.evals, epochs))
    eval_epochs = set(np.linspace(epochs / n_ev, epochs, n_ev).round().astype(int).clip(1, epochs).tolist())

    row = dict(run_id=run_id(args), label=args.label, timestamp=datetime.now().isoformat(timespec="seconds"),
               model=args.model, dataset="voc07", fraction=args.fraction, method=args.method, seed=args.seed,
               lr=args.lr, backbone_lr_mult=args.backbone_lr_mult if args.method == "full" else "",
               weight_decay=args.weight_decay, epochs=epochs, steps=steps, batch_size=args.batch_size,
               img_size=S, precision=args.precision, n_train=len(train_recs), n_val=len(val_recs),
               n_test=len(test_recs), total_params=total_params, trainable_params=trainable,
               trainable_pct=100.0 * trainable / total_params, trainable_mb=trainable * 4 / 1e6,
               dry_run=args.dry_run)
    if args.method == "lora":
        row.update(rank=args.rank, alpha=args.alpha, lora_dropout=args.lora_dropout)
    print(f"{run_id(args)}: {len(train_recs)} train images, {epochs} epochs, {steps} steps, "
          f"trainable {trainable:,} ({row['trainable_pct']:.2f}%)", flush=True)

    # ---- train
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    best_val, best_state, epoch_times = -1.0, None, []
    for ep in range(1, epochs + 1):
        net.train()
        if args.method != "full":
            set_frozen_bn_eval(net)
        t0 = time.time()
        for x, metas in train_loader:
            labels = [{"class_labels": m["class_labels"].to(device), "boxes": m["boxes"].to(device)}
                      for m in metas]
            with autocast():
                loss = net(pixel_values=x.to(device, non_blocking=True), labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {loss.item()} at epoch {ep}; try a lower --lr "
                                   f"or --precision tf32")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in named], 0.1)
            opt.step()
            sched.step()
        if use_cuda:
            torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        msg = f"epoch {ep}/{epochs}  loss {loss.item():.3f}  {epoch_times[-1]:.1f}s"
        if ep in eval_epochs:
            net.eval()
            v = coco_map(val_gt, predict(lambda x: net(pixel_values=x), val_loader, device, autocast),
                         voc_cat_ids, [r["id"] for r in val_recs])[0]
            msg += f"  val mAP {v:.4f}"
            if v > best_val:
                best_val = v
                if args.method == "full":
                    best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
                else:
                    best_state = {n: p.detach().cpu().clone() for n, p in named}
        print(msg, flush=True)
    if args.method == "full":
        net.load_state_dict(best_state)
    else:
        with torch.no_grad():
            for n, p in named:
                p.copy_(best_state[n].to(p.device))
    train_time = time.time() - t_start

    # ---- test
    net.eval()
    stats = coco_map(test_gt, predict(lambda x: net(pixel_values=x), test_loader, device, autocast),
                     voc_cat_ids, [r["id"] for r in test_recs])
    row.update(val_map_best=best_val, test_map=stats[0], test_map50=stats[1], test_map_small=stats[3],
               test_map_medium=stats[4], test_map_large=stats[5], train_time_s=train_time,
               epoch_time_s=float(np.mean(epoch_times)),
               peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0)

    # ---- forgetting: original COCO heads back on the trained model
    if forget is not None:
        switch.use("coco")
        gt, loader, ids = forget
        after = coco_map(gt, predict(lambda x: net(pixel_values=x), loader, device, autocast), coco_cat_ids, ids)[0]
        row.update(forget_before=before, forget_after=after, forgetting=before - after,
                   rel_forgetting=100.0 * (before - after) / max(before, 1e-12))
        if args.method == "lora" and args.adapters_off_check:
            with net.disable_adapter():
                off = coco_map(gt, predict(lambda x: net(pixel_values=x), loader, device, autocast),
                               coco_cat_ids, ids)[0]
            row["forget_adapters_off"] = off
        switch.use("target")

    append_row(args.out, row)
    print(json.dumps({k: row.get(k) for k in ("run_id", "val_map_best", "test_map", "test_map50", "forget_before",
                                              "forgetting", "trainable_pct", "peak_mem_gb", "epoch_time_s")},
                     default=str), flush=True)


if __name__ == "__main__":
    main()
