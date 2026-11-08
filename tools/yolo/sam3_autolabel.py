#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シミュレータで撮った画像に、SAM 3 で自動的に YOLO 形式のラベルを付ける（理論上の手順。未実行）。

SAM 3 は「物体の名前（テキスト）」を渡すと、その物体の BBox とマスクを返す。
検出器の56クラスの名前を1つずつ渡し、見つかった BBox をそのクラスのラベルとして書き出す。
評価に使う Seed 25 / 86 の部屋の画像を撮ってラベルを付ければ、本番と同じ見え方のデータが増える。

必要なもの（SAM 3 の README より。GPU サーバの環境では動かないので、自分の PC か Colab で実行）
  - Python 3.12 以上、PyTorch 2.7 以上、CUDA 12.6 以上の GPU
  - https://github.com/facebookresearch/sam3 をインストール（pip install -e .）
  - Hugging Face で facebook/sam3 の利用申請が通っていて、`hf auth login` 済み

使い方:
  python tools/yolo/sam3_autolabel.py <画像フォルダ> --out datasets/wrs_sam3 [--min-score 0.5]
  → <out>/images/train と <out>/labels/train に書き出す。
    convert_to_yolo.py の出力（datasets/wrs_yolo）の train に足してから学習する。
  付いたラベルは必ず何枚か目で確認すること（--preview で BBox を描いた画像も出せる）。
"""
import argparse
import glob
import json
import os
import shutil

from PIL import Image, ImageDraw

CLASS_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "src", "wrs_detector",
                          "config", "wod-c_classes.json")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_prompts():
    """62401738 いしい
    検出器のクラス順（background を除く）に (クラス名, SAM 3 に渡す文) のリストを返す。
    文は授業データセットの名前（"Cheez-it cracker box" など）を使う。ラベル名より物体を言い当てやすい。
    """
    with open(CLASS_JSON) as json_file:
        classes = [c for c in json.load(json_file)["classes"] if c["name"] != "background"]
    return [(c["name"], c["train_dataset_name"].replace("_", " ").lower()) for c in classes]


def box_iou(box_a, box_b):
    """62401738 いしい
    (x0, y0, x1, y1) 形式の2つの BBox の IoU
    """
    inter_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    inter_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def resolve_overlaps(found, iou_thr):
    """62401738 いしい
    別々の名前で同じ物体が見つかった場合（例: "tennis ball" と "racquetball"）、
    重なりが iou_thr 以上のものはスコアが高い方だけ残す。found は (クラス番号, スコア, box) のリスト。
    """
    kept = []
    for item in sorted(found, key=lambda f: f[1], reverse=True):
        if all(box_iou(item[2], other[2]) < iou_thr for other in kept):
            kept.append(item)
    return kept


def label_image(processor, image, prompts, min_score, iou_thr):
    """62401738 いしい
    1枚の画像に全クラスの名前を順に渡し、見つかった (クラス番号, スコア, box) を返す。
    画像の特徴は set_image で1回だけ計算し、文だけを差し替える（set_text_prompt は前の文の結果を上書きする）。
    """
    state = processor.set_image(image)
    found = []
    for idx, (_, prompt) in enumerate(prompts):
        state = processor.set_text_prompt(prompt=prompt, state=state)
        for box, score in zip(state["boxes"].tolist(), state["scores"].tolist()):
            if score >= min_score:
                found.append((idx, float(score), box))  # box は画素単位の [x0, y0, x1, y1]
    return resolve_overlaps(found, iou_thr)


def to_yolo_line(idx, box, width, height):
    """62401738 いしい
    画素単位の [x0, y0, x1, y1] を YOLO の1行（クラス cx cy w h、0〜1）にする
    """
    x_0, y_0 = max(0.0, box[0]), max(0.0, box[1])
    x_1, y_1 = min(float(width), box[2]), min(float(height), box[3])
    return "%d %.6f %.6f %.6f %.6f\n" % (
        idx, (x_0 + x_1) / 2 / width, (y_0 + y_1) / 2 / height,
        (x_1 - x_0) / width, (y_1 - y_0) / height)


def main():
    """62401738 いしい
    フォルダ内の画像にラベルを付けて YOLO 形式で書き出す
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", help="シミュレータで撮った画像のフォルダ")
    parser.add_argument("--out", default="datasets/wrs_sam3")
    parser.add_argument("--min-score", type=float, default=0.5, help="採用する最低スコア")
    parser.add_argument("--iou", type=float, default=0.7, help="別クラスの重複とみなす IoU")
    parser.add_argument("--preview", action="store_true", help="BBox を描いた確認用画像も出す")
    args = parser.parse_args()

    # SAM 3 はインストールした環境でだけ読み込めるので、ここで import する
    from sam3.model_builder import build_sam3_image_model  # pylint: disable=import-outside-toplevel
    from sam3.model.sam3_image_processor import Sam3Processor  # pylint: disable=import-outside-toplevel

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.min_score)
    prompts = load_prompts()

    img_dir = os.path.join(args.out, "images", "train")
    lbl_dir = os.path.join(args.out, "labels", "train")
    prev_dir = os.path.join(args.out, "preview")
    for folder in (img_dir, lbl_dir) + ((prev_dir,) if args.preview else ()):
        os.makedirs(folder, exist_ok=True)

    paths = sorted(p for p in glob.glob(os.path.join(args.images, "**", "*"), recursive=True)
                   if p.lower().endswith(IMAGE_EXTS))
    n_boxes = 0
    for path in paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        found = label_image(processor, image, prompts, args.min_score, args.iou)
        # seed86/0000_image.png と seed25/0000_image.png がぶつからないよう、フォルダ名を頭に付ける
        name = os.path.relpath(path, args.images).replace(os.sep, "_")
        stem = os.path.splitext(name)[0]
        shutil.copy(path, os.path.join(img_dir, name))
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as label_file:
            for idx, _, box in found:
                label_file.write(to_yolo_line(idx, box, width, height))
        if args.preview:
            draw = ImageDraw.Draw(image)
            for idx, score, box in found:
                draw.rectangle(box, outline=(255, 0, 0), width=2)
                draw.text((box[0], max(0, box[1] - 12)),
                          "%s %.2f" % (prompts[idx][0], score), fill=(255, 0, 0))
            image.save(os.path.join(prev_dir, name))
        n_boxes += len(found)
        print("%s: %d objects" % (name, len(found)))
    print("images: %d, boxes: %d -> %s" % (len(paths), n_boxes, args.out))


if __name__ == "__main__":
    main()
