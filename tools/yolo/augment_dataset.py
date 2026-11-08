#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 形式のデータセットの train を水増しする（複製 + ノイズ + 傾け）。

convert_to_yolo.py が作った <データセット>/images/train と labels/train の各画像から、
--copies 枚のコピーを作って同じフォルダに書き足す。val は評価用なので触らない。
  - 傾け: -max_angle〜+max_angle 度でランダムに回転。BBox は4隅を回して囲み直す
  - ノイズ: 標準偏差 --noise（0〜255 の画素値）のガウスノイズ
  - 明るさ: 0.8〜1.2 倍
回転で画像の外に大きくはみ出した BBox（元の面積の 40% 未満しか残らない）は捨てる。

使い方:
  /usr/bin/python3 tools/yolo/augment_dataset.py /workspace/datasets/wrs_yolo --copies 2
  そのあと tools/yolo/train_yolo.sh で学習する。
もう一度実行すると、前回作ったコピー（ファイル名に _aug が入るもの）を消してから作り直す。
"""
import argparse
import glob
import math
import os
import random

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
AUG_TAG = "_aug"


def read_labels(path):
    """62401738 いしい
    YOLO のラベルファイルを [(クラス, cx, cy, w, h), ...]（0〜1 の相対座標）で読む
    """
    labels = []
    if not os.path.exists(path):
        return labels
    with open(path) as label_file:
        for line in label_file:
            parts = line.split()
            if len(parts) == 5:
                labels.append((int(parts[0]),) + tuple(float(v) for v in parts[1:]))
    return labels


def rotate_box(box, angle_deg, width, height):
    """62401738 いしい
    画像を中心まわりに angle_deg 度（反時計回り。PIL の rotate と同じ向き）回したときの BBox を返す。
    4隅を回して外接矩形を取り、画像の中に切り詰める。残った面積が元の 40% 未満なら None。
    """
    cls, c_x, c_y, b_w, b_h = box
    x_0, x_1 = (c_x - b_w / 2) * width, (c_x + b_w / 2) * width
    y_0, y_1 = (c_y - b_h / 2) * height, (c_y + b_h / 2) * height
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    o_x, o_y = width / 2.0, height / 2.0
    xs, ys = [], []
    for p_x, p_y in ((x_0, y_0), (x_1, y_0), (x_0, y_1), (x_1, y_1)):
        d_x, d_y = p_x - o_x, p_y - o_y
        # 画像の y は下向きなので、見た目の反時計回りは次の式になる
        xs.append(o_x + d_x * cos_t + d_y * sin_t)
        ys.append(o_y - d_x * sin_t + d_y * cos_t)
    full_w, full_h = max(xs) - min(xs), max(ys) - min(ys)
    n_x0, n_x1 = max(0.0, min(xs)), min(float(width), max(xs))
    n_y0, n_y1 = max(0.0, min(ys)), min(float(height), max(ys))
    if n_x1 <= n_x0 or n_y1 <= n_y0:
        return None
    if (n_x1 - n_x0) * (n_y1 - n_y0) < 0.4 * full_w * full_h:
        return None
    return (cls, (n_x0 + n_x1) / 2 / width, (n_y0 + n_y1) / 2 / height,
            (n_x1 - n_x0) / width, (n_y1 - n_y0) / height)


def augment_image(image, labels, max_angle, noise, rng):
    """62401738 いしい
    1枚の画像とラベルから、傾け・明るさ・ノイズをかけたコピーを1つ作る
    """
    angle = rng.uniform(-max_angle, max_angle)
    width, height = image.size
    mean_color = tuple(int(v) for v in np.asarray(image).reshape(-1, 3).mean(axis=0))
    bilinear = getattr(Image, "Resampling", Image).BILINEAR  # Pillow 10 以降は Resampling の下
    rotated = image.rotate(angle, resample=bilinear, fillcolor=mean_color)
    pixels = np.asarray(rotated).astype(np.float32) * rng.uniform(0.8, 1.2)
    if noise > 0:
        pixels += np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, noise, pixels.shape)
    out = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    new_labels = [rotate_box(box, angle, width, height) for box in labels]
    return out, [box for box in new_labels if box is not None]


def main():
    """62401738 いしい
    train の画像を水増しして、同じフォルダに書き足す
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="convert_to_yolo.py の出力フォルダ（data.yaml がある所）")
    parser.add_argument("--copies", type=int, default=2, help="1枚から作るコピーの数")
    parser.add_argument("--max-angle", type=float, default=10.0, help="傾ける最大角度 [度]")
    parser.add_argument("--noise", type=float, default=8.0, help="ノイズの標準偏差（画素値）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    img_dir = os.path.join(args.dataset, "images", "train")
    lbl_dir = os.path.join(args.dataset, "labels", "train")
    if not os.path.isdir(img_dir):
        raise SystemExit(img_dir + " がありません。先に convert_to_yolo.py を実行")

    for old in glob.glob(os.path.join(img_dir, "*" + AUG_TAG + "*")) + \
            glob.glob(os.path.join(lbl_dir, "*" + AUG_TAG + "*")):
        os.remove(old)

    rng = random.Random(args.seed)
    sources = sorted(p for p in glob.glob(os.path.join(img_dir, "*"))
                     if p.lower().endswith(IMAGE_EXTS))
    n_made = n_dropped = 0
    for path in sources:
        stem, ext = os.path.splitext(os.path.basename(path))
        labels = read_labels(os.path.join(lbl_dir, stem + ".txt"))
        image = Image.open(path).convert("RGB")
        for k in range(args.copies):
            out, new_labels = augment_image(image, labels, args.max_angle, args.noise, rng)
            n_dropped += len(labels) - len(new_labels)
            name = "{}{}{}".format(stem, AUG_TAG, k)
            out.save(os.path.join(img_dir, name + ext))
            with open(os.path.join(lbl_dir, name + ".txt"), "w") as label_file:
                for box in new_labels:
                    label_file.write("%d %.6f %.6f %.6f %.6f\n" % box)
            n_made += 1
    print("source images: %d -> made %d copies (dropped boxes: %d)"
          % (len(sources), n_made, n_dropped))


if __name__ == "__main__":
    main()
