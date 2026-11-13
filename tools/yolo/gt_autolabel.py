#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_gt.py で撮った画像に、YOLO 形式のラベルを付ける。
クラス名は Gazebo の正解（モデル名）を使い、枠だけを SAM 2 で作る（物体の位置を点で指定）。
SAM 3 のように名前から探さないので、似た物体（softball と mini_soccer_ball など）でもクラスを間違えない。

使い方（GPU をシミュレータと取り合う。SAM 2 の重み 154MB は初回に自動で落ちる）
  PYTHONPATH=/workspace/.yolo_env /usr/bin/python3 /workspace/tools/yolo/gt_autolabel.py \
      /workspace/datasets/capture_gt --out /workspace/datasets/wrs_gt --preview \
      [--compare /workspace/src/wrs_detector/checkpoints/yolo_wrs.pt]
  → <out>/images/train と <out>/labels/train。--compare を付けると、今の検出器がどのクラスを
    何と間違えているかを <out>/confusion.txt に書く。

1つでも枠を作れなかった「見えている物体」がある画像は、ラベル漏れになるので捨てる。
"""
import argparse
import collections
import glob
import json
import os
import shutil

from PIL import Image, ImageDraw

CLASS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src",
                          "wrs_detector", "config", "wod-c_classes.json")
FOCAL = 554.0          # 頭部カメラの焦点距離 [画素]（640x480）
OCCLUDED_GAP = 0.12    # 深度が物体までの距離よりこれ以上近ければ、手前の物に隠れている [m]
MAX_SIZE = 0.35        # 枠の一辺がこれより大きければ、棚や机を拾っている [m]
MIN_PIXELS = 6         # 枠の一辺がこれより小さい物体は学習に使わない
TASK2A_LABEL = "object"   # Task 2a の障害物はモデル名に種類がないので、この名前にする


def judge(obj, box, score, min_score):
    """62401738 いしい
    SAM 2 が返した枠 box を採用するか決める。採用なら "ok"、だめなら理由。
    """
    width, height = box[2] - box[0], box[3] - box[1]
    limit = MAX_SIZE * FOCAL / obj["dist"]
    if score < min_score:
        return "low score %.2f" % score
    if width > limit or height > limit:
        return "too large"
    if min(width, height) < MIN_PIXELS:
        return "too small"
    # 点は物体の底に打っているので、枠の下の辺の近くにあるはず
    if not (box[0] - 3 <= obj["u"] <= box[2] + 3 and box[1] <= obj["v"] <= box[3] + 0.5 * height + 4):
        return "point outside"
    return "ok"


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


def main():
    """62401738 いしい
    フォルダ内の NNNN_image.png と NNNN_gt.json から YOLO 形式のラベルを書き出す。
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", help="capture_gt.py の出力フォルダ")
    parser.add_argument("--out", default="/workspace/datasets/wrs_gt")
    parser.add_argument("--min-score", type=float, default=0.85, help="採用する SAM 2 の最低スコア")
    parser.add_argument("--sam", default="sam2.1_b.pt")
    parser.add_argument("--preview", action="store_true", help="枠を描いた確認用画像も出す")
    parser.add_argument("--compare", help="今の検出器の重み（クラスの間違いを数える）")
    args = parser.parse_args()

    from ultralytics import SAM, YOLO  # pylint: disable=import-outside-toplevel

    with open(CLASS_JSON) as json_file:
        names = [c["name"] for c in json.load(json_file)["classes"] if c["name"] != "background"]
    sam = SAM(args.sam)
    detector = YOLO(args.compare) if args.compare else None
    confusion = collections.Counter()
    missed = collections.Counter()

    img_dir = os.path.join(args.out, "images", "train")
    lbl_dir = os.path.join(args.out, "labels", "train")
    prev_dir = os.path.join(args.out, "preview")
    for folder in (img_dir, lbl_dir) + ((prev_dir,) if args.preview else ()):
        os.makedirs(folder, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.images, "**", "*_image.png"), recursive=True))
    kept = dropped = n_boxes = 0
    for path in paths:
        gt_path = path.replace("_image.png", "_gt.json")
        if not os.path.exists(gt_path):
            continue
        with open(gt_path) as json_file:
            objects = json.load(json_file)["objects"]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        boxes, reasons = [], []
        for obj in objects:
            if obj["depth"] is None or obj["depth"] < obj["dist"] - OCCLUDED_GAP:
                continue   # 隠れている（または深度がない）物体にはラベルを付けない
            label = obj["label"] or TASK2A_LABEL
            lift = max(2, int(FOCAL * 0.012 / obj["dist"]))   # 底から 1.2cm 上を指す
            result = sam.predict(path, points=[[obj["u"], obj["v"] - lift]], labels=[1],
                                 verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                reasons.append("%s: no mask" % label)
                continue
            box = result.boxes.xyxy[0].tolist()
            verdict = judge(obj, box, float(result.boxes.conf[0]), args.min_score)
            if verdict != "ok":
                reasons.append("%s: %s" % (label, verdict))
                continue
            boxes.append((label, box))
        name = os.path.relpath(path, args.images).replace(os.sep, "_")
        if reasons or not boxes:
            dropped += 1
            print("drop %s (%s)" % (name, "; ".join(reasons) or "no visible object"))
            continue
        shutil.copy(path, os.path.join(img_dir, name))
        with open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), "w") as label_file:
            for label, box in boxes:
                label_file.write("%d %.6f %.6f %.6f %.6f\n" % (
                    names.index(label), (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height,
                    (box[2] - box[0]) / width, (box[3] - box[1]) / height))
        if detector is not None:
            found = detector.predict(path, conf=0.2, verbose=False)[0]
            preds = [(found.names[int(c)], b) for c, b in
                     zip(found.boxes.cls.tolist(), found.boxes.xyxy.tolist())]
            for label, box in boxes:
                best = max(preds, key=lambda p, b=box: box_iou(p[1], b), default=None)
                if best is None or box_iou(best[1], box) < 0.4:
                    missed[label] += 1
                else:
                    confusion[(label, best[0])] += 1
        if args.preview:
            draw = ImageDraw.Draw(image)
            for label, box in boxes:
                draw.rectangle(box, outline=(255, 0, 0), width=1)
                draw.text((box[0], max(0, box[1] - 11)), label, fill=(255, 0, 0))
            image.save(os.path.join(prev_dir, name))
        kept += 1
        n_boxes += len(boxes)
    print("images: %d kept, %d dropped, boxes: %d -> %s" % (kept, dropped, n_boxes, args.out))

    if detector is not None:
        lines = ["正解 -> 今の検出器の答え : 回数（conf 0.2 以上、IoU 0.4 以上で対応付け）"]
        for (truth, pred), count in sorted(confusion.items(), key=lambda kv: (kv[0][0], -kv[1])):
            lines.append("%s %s -> %s : %d" % ("  " if truth == pred else "NG", truth, pred, count))
        for truth, count in sorted(missed.items()):
            lines.append("NG %s -> （検出なし） : %d" % (truth, count))
        with open(os.path.join(args.out, "confusion.txt"), "w") as text_file:
            text_file.write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
