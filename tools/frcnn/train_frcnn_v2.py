#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster R-CNN v2（torchvision の fasterrcnn_resnet50_fpn_v2。今の Faster R-CNN の改良版）を、
YOLO 形式のデータセット（convert_to_yolo.py → sam3_augment.py の出力）で学習する。

今の検出器（fasterrcnn_resnet50_fpn）との違い
  - バックボーン・FPN・RPN・検出ヘッドに BatchNorm / GroupNorm が入り、COCO の mAP が 37.0 → 46.7
  - 出力するラベル名・形式は同じ（wod-c_classes.json の順。0 は background）

使い方（/workspace/.yolo_env の torch 2.4 / torchvision 0.19 を使う。GPU は検出ノードと共有）
  PYTHONPATH=/workspace/.yolo_env /usr/bin/python3 /workspace/tools/frcnn/train_frcnn_v2.py \
      /workspace/datasets/wrs_yolo --epochs 24 --out /workspace/runs/frcnn_v2
  → <out>/best.pth（val の mAP@0.5 が一番よかったもの）と <out>/report.txt（クラスごとの結果）
  できた best.pth を src/wrs_detector/checkpoints/frcnn_v2_wrs.pth に置くと DETECTOR=frcnn_v2 で使われる。
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

CLASS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src",
                          "wrs_detector", "config", "wod-c_classes.json")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def build_model(num_classes, pretrained=True):
    """62401738 いしい
    COCO で学習済みの Faster R-CNN v2 の、最後のクラス分類だけを num_classes に差し替えたモデル。
    """
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None,
                                       min_size=480, max_size=640)   # カメラ画像は 640x480
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class YoloDataset(torch.utils.data.Dataset):
    """62401738 いしい
    YOLO 形式（images/<split>, labels/<split>）を Faster R-CNN の入力（画像, {"boxes", "labels"}）にする。
    YOLO のクラス番号 k は wod-c の k + 1（0 は background）。
    """

    def __init__(self, root, split, train):
        self.images = sorted(p for p in glob.glob(os.path.join(root, "images", split, "*"))
                             if p.lower().endswith(IMAGE_EXTS) and os.path.exists(p))
        self.label_dir = os.path.join(root, "labels", split)
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        boxes, labels = [], []
        stem = os.path.splitext(os.path.basename(path))[0]
        label_path = os.path.join(self.label_dir, stem + ".txt")
        if os.path.exists(label_path):
            with open(label_path) as label_file:
                for line in label_file:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls, c_x, c_y, b_w, b_h = int(parts[0]), *map(float, parts[1:])
                    x_0, y_0 = (c_x - b_w / 2) * width, (c_y - b_h / 2) * height
                    x_1, y_1 = (c_x + b_w / 2) * width, (c_y + b_h / 2) * height
                    if x_1 - x_0 >= 2 and y_1 - y_0 >= 2:
                        boxes.append([x_0, y_0, x_1, y_1])
                        labels.append(cls + 1)
        if self.train:
            # 色だけ変える（形・向きは sam3_augment.py の水増しで変えてある。左右反転は文字の向きが変わるので使わない）
            image = F.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = F.adjust_contrast(image, random.uniform(0.85, 1.15))
            image = F.adjust_saturation(image, random.uniform(0.85, 1.15))
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(labels, dtype=torch.int64)}
        return F.to_tensor(image), target, path


def collate(batch):
    """62401738 いしい
    画像の大きさがそろっていなくてもよいよう、リストのまま返す
    """
    return tuple(zip(*batch))


def box_iou(box_a, box_b):
    """62401738 いしい
    (x0, y0, x1, y1) 形式の2つの枠の IoU
    """
    inter_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    inter_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    return inter / ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                    + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - inter)


def average_precision(hits, n_truth):
    """62401738 いしい
    スコア順に並べた [(スコア, 当たりか)] から AP（VOC の全点補間）を求める
    """
    if n_truth == 0:
        return None
    hits = sorted(hits, key=lambda h: -h[0])
    t_p = np.cumsum([h[1] for h in hits]) if hits else np.zeros(0)
    f_p = np.cumsum([1 - h[1] for h in hits]) if hits else np.zeros(0)
    recall = np.concatenate([[0.0], t_p / n_truth, [1.0]])
    precision = np.concatenate([[1.0], t_p / np.maximum(t_p + f_p, 1e-9), [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def evaluate(predict, dataset, names, report_threshold=0.5):
    """62401738 いしい
    predict(画像のパス) → [(ラベル番号, スコア, 枠)] で val を評価し、
    (mAP@0.5, クラスごとの AP, 取り違えの回数) を返す。取り違えはスコア report_threshold 以上で数える。
    """
    hits = collections.defaultdict(list)
    n_truth = collections.Counter()
    confusion = collections.Counter()
    for idx in range(len(dataset)):
        _, target, path = dataset[idx]
        truths = list(zip(target["labels"].tolist(), target["boxes"].tolist()))
        preds = sorted(predict(path), key=lambda p: -p[1])
        for label, _ in truths:
            n_truth[label] += 1
        used = set()
        for label, score, box in preds:
            best, best_k = 0.0, None
            for k, (t_label, t_box) in enumerate(truths):
                if t_label == label and k not in used and box_iou(box, t_box) > best:
                    best, best_k = box_iou(box, t_box), k
            hit = best >= 0.5
            if hit:
                used.add(best_k)
            hits[label].append((score, int(hit)))
        for t_label, t_box in truths:
            strong = [p for p in preds if p[1] >= report_threshold and box_iou(p[2], t_box) >= 0.5]
            answer = max(strong, key=lambda p: p[1])[0] if strong else None
            confusion[(t_label, answer)] += 1
    per_class = {names[c]: average_precision(hits[c], n_truth[c]) for c in n_truth}
    values = [v for v in per_class.values() if v is not None]
    return (sum(values) / len(values) if values else 0.0), per_class, confusion


def write_report(path, title, m_ap, per_class, confusion, names):
    """62401738 いしい
    評価結果（mAP、クラスごとの AP、取り違え）をテキストに書く
    """
    lines = ["%s  mAP@0.5 = %.3f" % (title, m_ap), "", "クラスごとの AP@0.5:"]
    for name, value in sorted(per_class.items(), key=lambda kv: (kv[1] is None, kv[1])):
        lines.append("  %-22s %s" % (name, "-" if value is None else "%.3f" % value))
    lines += ["", "取り違え・見落とし（正解 -> 答え : 回数、スコア 0.5 以上）:"]
    for (truth, answer), count in sorted(confusion.items(), key=lambda kv: -kv[1]):
        if truth != answer:
            lines.append("  %s -> %s : %d" % (names[truth], "（なし）" if answer is None else names[answer],
                                               count))
    with open(path, "a", encoding="utf-8") as text_file:
        text_file.write("\n".join(lines) + "\n\n")
    print("\n".join(lines[:1]))


def main():
    """62401738 いしい
    学習し、エポックごとに val の mAP@0.5 を測って一番よい重みを残す
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="YOLO 形式のデータセット（images/train, images/val）")
    parser.add_argument("--out", default="/workspace/runs/frcnn_v2")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compare-yolo", help="同じ val で比べる YOLO の重み（任意）")
    parser.add_argument("--compare-v1", help="dump_frcnn_v1.py が書いた今の Faster R-CNN の検出結果（任意）")
    parser.add_argument("--eval-only", help="学習せず、この重み（best.pth）と比較相手だけ評価する")
    parser.add_argument("--limit", type=int, default=0, help="試すとき用: train をこの枚数に減らす")
    args = parser.parse_args()

    with open(CLASS_JSON, encoding="utf-8") as json_file:
        names = [c["name"] for c in json.load(json_file)["classes"]]   # 0 は background
    os.makedirs(args.out, exist_ok=True)
    report = os.path.join(args.out, "report.txt")
    open(report, "w").close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)

    train_set = YoloDataset(args.dataset, "train", train=True)
    if args.limit:
        train_set.images = train_set.images[:args.limit]
    val_set = YoloDataset(args.dataset, "val", train=False)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, collate_fn=collate)
    print("train %d images, val %d images" % (len(train_set), len(val_set)))

    model = build_model(len(names), pretrained=not args.eval_only).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    warmup = min(300, len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.cuda.amp.GradScaler()

    def predict(path):
        image = F.to_tensor(Image.open(path).convert("RGB")).to(device)
        with torch.no_grad():
            out = model([image])[0]
        return list(zip(out["labels"].tolist(), out["scores"].tolist(), out["boxes"].tolist()))

    best = -1.0
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location="cpu")["model"])
        model.eval()
        model.roi_heads.score_thresh = 0.01
        args.epochs = 0
        best_result = (0,) + evaluate(predict, val_set, names)
    for epoch in range(args.epochs):
        model.train()
        started, total = time.time(), 0.0
        for images, targets, _ in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss)
        model.eval()
        model.roi_heads.score_thresh = 0.01
        m_ap, per_class, confusion = evaluate(predict, val_set, names)
        print("epoch %d/%d loss %.3f val mAP@0.5 %.3f (%.0f s)"
              % (epoch + 1, args.epochs, total / max(1, len(loader)), m_ap, time.time() - started),
              flush=True)
        if m_ap > best:
            best = m_ap
            torch.save({"arch": "fasterrcnn_resnet50_fpn_v2", "classes": names,
                        "model": model.state_dict()}, os.path.join(args.out, "best.pth"))
            best_result = (epoch + 1, m_ap, per_class, confusion)
    write_report(report, "Faster R-CNN v2 (best epoch %d)" % best_result[0], *best_result[1:], names)

    if args.compare_v1:
        with open(args.compare_v1) as json_file:
            dumped = json.load(json_file)
        index = {name: k for k, name in enumerate(names)}
        write_report(report, "Faster R-CNN v1 (%s)" % os.path.basename(args.compare_v1),
                     *evaluate(lambda path: [(index[l], s, b) for l, s, b in dumped[os.path.realpath(path)]],
                               val_set, names), names)

    if args.compare_yolo:
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
        yolo = YOLO(args.compare_yolo)
        index = {name: k for k, name in enumerate(names)}

        def predict_yolo(path):
            res = yolo.predict(path, conf=0.01, verbose=False)[0]
            return [(index[res.names[int(c)]], float(s), b) for c, s, b in
                    zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())]
        write_report(report, "YOLOv8 (%s)" % os.path.basename(args.compare_yolo),
                     *evaluate(predict_yolo, val_set, names), names)
    print("-> %s, %s" % (os.path.join(args.out, "best.pth"), report))


if __name__ == "__main__":
    main()
