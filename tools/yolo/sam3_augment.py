#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ラベル付きの YOLO 形式データセットの train を、SAM 3 のマスクを使って水増しする（傾け + 弱いノイズ）。

augment_dataset.py は BBox の4隅を回して囲み直すので、傾けるほど枠が物体より大きくなる。
ここでは SAM 3 に「そのクラスの名前」を渡して物体のマスクを取り、既にあるラベルの枠と重なるマスクを
その物体のものとして、画像と一緒にマスクを回してから枠を取り直す（枠が物体にぴったり付く）。
クラスは元のラベルのまま使う（SAM 3 にクラスを決めさせない）。

ついでに、SAM 3 が「その名前の物体」を枠の位置に見つけられなかったラベルを
<データセット>/sam3_label_check.txt に書く（ラベルの付け間違い・枠ずれの候補。目で確認する）。

使い方（SAM 3 の環境で。HF_TOKEN は facebook/sam3 の利用申請が通ったトークン。ファイルには書かない）
  HF_TOKEN=hf_xxx /workspace/.sam3_env/bin/python /workspace/tools/yolo/sam3_augment.py \
      /workspace/datasets/wrs_yolo --copies 2 [--max-angle 10] [--noise 3] [--preview 20]
もう一度実行すると、前回作ったコピー（ファイル名に _aug が入るもの）を消してから作り直す。
"""
import argparse
import glob
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

from augment_dataset import AUG_TAG, IMAGE_EXTS, read_labels, rotate_box

CLASS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src",
                          "wrs_detector", "config", "wod-c_classes.json")
MATCH_IOU = 0.5     # SAM 3 の枠と元のラベルの枠がこれ以上重なれば、同じ物体とみなす
KEEP_RATIO = 0.4    # 傾けた後に画像の中に残るマスクがこれ未満なら、その物体のラベルは捨てる
# 名前だけでは SAM 3 が見つけにくいクラスの言い換え（データセットの名前の代わりに渡す）
PROMPT_OVERRIDES = {
    "padlock": "padlock", "clamp": "clamp", "nine_hole_peg_test": "wooden peg board",
    "cup": "plastic cup", "marble": "marble", "colored_wood_block": "wooden block",
    "bleach_cleanser": "bleach bottle", "sponge": "sponge", "pitcher_base": "pitcher",
    "black_t_shirt": "black t-shirt", "credit_card_blank": "credit card",
}


def load_prompts():
    """62401738 いしい
    クラス番号順（background を除く）に (クラス名, SAM 3 に渡す文) のリストを返す。
    """
    with open(CLASS_JSON, encoding="utf-8") as json_file:
        classes = [c for c in json.load(json_file)["classes"] if c["name"] != "background"]
    return [(c["name"], PROMPT_OVERRIDES.get(c["name"], c["train_dataset_name"].replace("_", " ").lower()))
            for c in classes]


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


def find_masks(processor, image, labels, prompts):
    """62401738 いしい
    labels（YOLO の相対座標）の各物体について、SAM 3 のマスク（高さ×幅の bool 配列）か None のリストと、
    「クラスの名前で見つかったか」のリストを返す。
    画像の特徴は1回だけ計算し、画像にあるクラスの名前だけを順に渡す。
    """
    width, height = image.size
    state = processor.set_image(image)
    masks = [None] * len(labels)
    for cls in sorted({label[0] for label in labels}):
        state = processor.set_text_prompt(prompt=prompts[cls][1], state=state)
        boxes = state["boxes"].tolist()
        found = state["masks"].squeeze(1).cpu().numpy() if boxes else []
        used = set()
        for idx, (l_cls, c_x, c_y, b_w, b_h) in enumerate(labels):
            if l_cls != cls:
                continue
            truth = ((c_x - b_w / 2) * width, (c_y - b_h / 2) * height,
                     (c_x + b_w / 2) * width, (c_y + b_h / 2) * height)
            ranked = sorted(((box_iou(truth, box), k) for k, box in enumerate(boxes) if k not in used),
                            reverse=True)
            if ranked and ranked[0][0] >= MATCH_IOU:
                used.add(ranked[0][1])
                masks[idx] = found[ranked[0][1]].astype(bool)
    named = [mask is not None for mask in masks]
    # 名前で見つからなかった物体は、元の枠そのものを SAM 3 に渡してマスクを取る（クラスの確認にはならない）
    for idx, (_, c_x, c_y, b_w, b_h) in enumerate(labels):
        if masks[idx] is not None:
            continue
        processor.reset_all_prompts(state)
        state = processor.add_geometric_prompt([c_x, c_y, b_w, b_h], True, state)
        boxes = state["boxes"].tolist()
        if not boxes:
            continue
        truth = ((c_x - b_w / 2) * width, (c_y - b_h / 2) * height,
                 (c_x + b_w / 2) * width, (c_y + b_h / 2) * height)
        iou, k = max((box_iou(truth, box), k) for k, box in enumerate(boxes))
        if iou >= MATCH_IOU:
            masks[idx] = state["masks"].squeeze(1)[k].cpu().numpy().astype(bool)
    return masks, named


def mask_box(mask):
    """62401738 いしい
    マスクを囲む枠 (x0, y0, x1, y1) と画素数。空なら (None, 0)。
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None, 0
    return (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1), int(mask.sum())


def augment(image, labels, masks, max_angle, noise, rng):
    """62401738 いしい
    画像を傾け、明るさを少し変え、弱いノイズを足したコピーと、その YOLO ラベルを返す。
    マスクがある物体は回したマスクから枠を取り、ない物体は枠の4隅を回して囲み直す。
    """
    angle = rng.uniform(-max_angle, max_angle)
    width, height = image.size
    mean_color = tuple(int(v) for v in np.asarray(image).reshape(-1, 3).mean(axis=0))
    rotated = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=mean_color)
    pixels = np.asarray(rotated).astype(np.float32) * rng.uniform(0.9, 1.1)
    if noise > 0:
        pixels += np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, noise, pixels.shape)
    out = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    new_labels = []
    for label, mask in zip(labels, masks):
        if mask is None:
            box = rotate_box(label, angle, width, height)
            if box is not None:
                new_labels.append(box)
            continue
        turned = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).rotate(
            angle, resample=Image.Resampling.NEAREST, fillcolor=0)) > 127
        box, count = mask_box(turned)
        if box is None or count < KEEP_RATIO * mask.sum():
            continue
        new_labels.append((label[0], (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height,
                           (box[2] - box[0]) / width, (box[3] - box[1]) / height))
    return out, new_labels


def main():
    """62401738 いしい
    train の画像を SAM 3 のマスクで水増しし、見つからなかったラベルの一覧も書く。
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="convert_to_yolo.py の出力フォルダ（data.yaml がある所）")
    parser.add_argument("--copies", type=int, default=2, help="1枚から作るコピーの数")
    parser.add_argument("--max-angle", type=float, default=10.0, help="傾ける最大角度 [度]")
    parser.add_argument("--noise", type=float, default=3.0, help="ノイズの標準偏差（画素値。弱め）")
    parser.add_argument("--min-score", type=float, default=0.3, help="SAM 3 の最低スコア")
    parser.add_argument("--preview", type=int, default=0, help="枠を描いた確認用画像を出す枚数")
    parser.add_argument("--limit", type=int, default=0, help="試すとき用: 先頭のこの枚数だけ処理")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    img_dir = os.path.join(args.dataset, "images", "train")
    lbl_dir = os.path.join(args.dataset, "labels", "train")
    if not os.path.isdir(img_dir):
        raise SystemExit(img_dir + " がありません。先に convert_to_yolo.py を実行")
    for old in glob.glob(os.path.join(img_dir, "*" + AUG_TAG + "*")) + \
            glob.glob(os.path.join(lbl_dir, "*" + AUG_TAG + "*")):
        os.remove(old)
    prev_dir = os.path.join(args.dataset, "preview_aug")
    if args.preview:
        os.makedirs(prev_dir, exist_ok=True)

    # SAM 3 はインストールした環境でだけ読み込めるので、ここで import する
    import torch  # pylint: disable=import-outside-toplevel
    from sam3.model_builder import build_sam3_image_model  # pylint: disable=import-outside-toplevel
    from sam3.model.sam3_image_processor import Sam3Processor  # pylint: disable=import-outside-toplevel

    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=args.min_score)
    prompts = load_prompts()
    rng = random.Random(args.seed)
    sources = sorted(p for p in glob.glob(os.path.join(img_dir, "*"))
                     if p.lower().endswith(IMAGE_EXTS) and os.path.exists(p))
    if args.limit:
        sources = sources[:args.limit]
    n_made = n_labels = n_masked = n_named = 0
    unconfirmed = []
    for number, path in enumerate(sources):
        stem, ext = os.path.splitext(os.path.basename(path))
        labels = read_labels(os.path.join(lbl_dir, stem + ".txt"))
        image = Image.open(path).convert("RGB")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, named = find_masks(processor, image, labels, prompts) if labels else ([], [])
        n_labels += len(labels)
        n_masked += len([m for m in masks if m is not None])
        n_named += sum(named)
        for label, found_by_name in zip(labels, named):
            if not found_by_name:
                unconfirmed.append("%s (%s) %s  cx=%.3f cy=%.3f w=%.3f h=%.3f" % (
                    stem, os.path.realpath(path), prompts[label[0]][0], *label[1:]))
        for k in range(args.copies):
            out, new_labels = augment(image, labels, masks, args.max_angle, args.noise, rng)
            name = "{}{}{}".format(stem, AUG_TAG, k)
            out.save(os.path.join(img_dir, name + ext))
            with open(os.path.join(lbl_dir, name + ".txt"), "w") as label_file:
                for box in new_labels:
                    label_file.write("%d %.6f %.6f %.6f %.6f\n" % box)
            n_made += 1
            if number < args.preview and k == 0:
                draw = ImageDraw.Draw(out)
                for cls, c_x, c_y, b_w, b_h in new_labels:
                    box = ((c_x - b_w / 2) * out.width, (c_y - b_h / 2) * out.height,
                           (c_x + b_w / 2) * out.width, (c_y + b_h / 2) * out.height)
                    draw.rectangle(box, outline=(255, 0, 0), width=1)
                    draw.text((box[0], max(0, box[1] - 11)), prompts[cls][0], fill=(255, 0, 0))
                out.save(os.path.join(prev_dir, name + ".png"))
        if number % 50 == 0:
            print("%d / %d" % (number, len(sources)), flush=True)
    with open(os.path.join(args.dataset, "sam3_label_check.txt"), "w", encoding="utf-8") as text_file:
        text_file.write("SAM 3 がその名前で枠の位置に見つけられなかったラベル（%d / %d）\n" %
                        (len(unconfirmed), n_labels))
        text_file.write("\n".join(unconfirmed) + "\n")
    print("source images: %d -> made %d copies; labels: %d, with a SAM 3 mask: %d (found by name: %d)"
          % (len(sources), n_made, n_labels, n_masked, n_named))


if __name__ == "__main__":
    main()
