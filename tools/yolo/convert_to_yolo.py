#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授業の独自データセット（Faster R-CNN の学習に使ったもの）を YOLO 形式に変換する。

COCO 形式（annotations の .json）と Pascal VOC 形式（画像ごとの .xml）に対応。
クラスは wrs_detector/config/wod-c_classes.json の順番（background を除く）にそろえるので、
学習した YOLO は今の検出器と同じラベル名を出す。

使い方:
  /usr/bin/python3 tools/yolo/convert_to_yolo.py <データセットのフォルダ> [--out datasets/wrs_yolo]
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET

CLASS_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "src", "wrs_detector",
                          "config", "wod-c_classes.json")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def normalize(name):
    """62401738 いしい
    クラス名を比べやすくする（"Plastic apple" → "plasticapple"）
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_classes():
    """62401738 いしい
    wod-c のクラス名リストと、いろいろな書き方 → クラス番号の辞書を返す
    """
    with open(CLASS_JSON) as json_file:
        classes = json.load(json_file)["classes"]
    names = [c["name"] for c in classes if c["name"] != "background"]
    lookup = {}
    for idx, info in enumerate([c for c in classes if c["name"] != "background"]):
        lookup[normalize(info["name"])] = idx
        lookup[normalize(info["train_dataset_name"])] = idx
    return names, lookup


def find_image(root, file_name):
    """62401738 いしい
    アノテーションに書かれた画像ファイル名から、実際のパスを探す
    """
    cand = os.path.join(root, file_name)
    if os.path.exists(cand):
        return cand
    base = os.path.basename(file_name)
    hits = glob.glob(os.path.join(root, "**", base), recursive=True)
    return hits[0] if hits else None


def read_coco(root, json_path):
    """62401738 いしい
    COCO 形式を読み、[(画像パス, 幅, 高さ, [(クラス名, x, y, w, h), ...]), ...] を返す
    """
    with open(json_path) as json_file:
        coco = json.load(json_file)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    anns = {}
    for ann in coco["annotations"]:
        anns.setdefault(ann["image_id"], []).append(ann)
    samples = []
    for img in coco["images"]:
        path = find_image(root, img["file_name"])
        if path is None:
            print("[warn] image not found:", img["file_name"], file=sys.stderr)
            continue
        boxes = [(cat_names[a["category_id"]],) + tuple(a["bbox"]) for a in anns.get(img["id"], [])]
        samples.append((path, img["width"], img["height"], boxes))
    return samples


def read_voc(root, xml_paths):
    """62401738 いしい
    Pascal VOC 形式を読み、read_coco と同じ形で返す
    """
    samples = []
    for xml_path in xml_paths:
        tree = ET.parse(xml_path).getroot()
        file_name = tree.findtext("filename")
        path = find_image(os.path.dirname(xml_path), file_name) or find_image(root, file_name)
        if path is None:
            print("[warn] image not found:", file_name, file=sys.stderr)
            continue
        width = float(tree.find("size").findtext("width"))
        height = float(tree.find("size").findtext("height"))
        boxes = []
        for obj in tree.findall("object"):
            bnd = obj.find("bndbox")
            x_1, y_1 = float(bnd.findtext("xmin")), float(bnd.findtext("ymin"))
            x_2, y_2 = float(bnd.findtext("xmax")), float(bnd.findtext("ymax"))
            boxes.append((obj.findtext("name"), x_1, y_1, x_2 - x_1, y_2 - y_1))
        samples.append((path, width, height, boxes))
    return samples


def read_labelme(json_paths):
    """62401738 いしい
    labelme 形式（画像ごとの .json に矩形とラベル名）を読み、read_coco と同じ形で返す。
    授業のデータセット（hsr_sim_obj_1000）はこの形式。
    """
    samples = []
    for json_path in json_paths:
        with open(json_path) as json_file:
            ann = json.load(json_file)
        path = find_image(os.path.dirname(json_path), ann["imagePath"])
        if path is None:
            print("[warn] image not found:", json_path, file=sys.stderr)
            continue
        boxes = []
        for shape in ann["shapes"]:
            if shape.get("shape_type", "rectangle") != "rectangle":
                continue
            (x_1, y_1), (x_2, y_2) = shape["points"][:2]
            boxes.append((shape["label"], min(x_1, x_2), min(y_1, y_2),
                          abs(x_2 - x_1), abs(y_2 - y_1)))
        samples.append((path, ann["imageWidth"], ann["imageHeight"], boxes))
    return samples


def split_by_folder(samples, val_ratio):
    """62401738 いしい
    同じフォルダ（同じシード＝同じ部屋の配置）の画像が train と val に分かれないように、
    フォルダ単位で val を選ぶ。フォルダが少ないときは画像単位でランダムに分ける。
    """
    folders = sorted(set([os.path.dirname(s[0]) for s in samples]))
    if len(folders) < 5:
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * val_ratio))
        return samples[n_val:], samples[:n_val]
    random.shuffle(folders)
    val_folders, n_val = set(), 0
    for folder in folders:
        if n_val >= len(samples) * val_ratio:
            break
        val_folders.add(folder)
        n_val += len([s for s in samples if os.path.dirname(s[0]) == folder])
    print("val folders:", sorted(val_folders))
    train = [s for s in samples if os.path.dirname(s[0]) not in val_folders]
    val = [s for s in samples if os.path.dirname(s[0]) in val_folders]
    return train, val


def main():
    """62401738 いしい
    データセットを探して読み、train / val に分けて YOLO 形式で書き出す
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("--out", default="datasets/wrs_yolo")
    parser.add_argument("--val", type=float, default=0.1, help="検証用に回す割合")
    args = parser.parse_args()

    names, lookup = load_classes()
    all_jsons = glob.glob(os.path.join(args.src, "**", "*.json"), recursive=True)
    coco_jsons, labelme_jsons = [], []
    for path in all_jsons:
        with open(path) as json_file:
            head = json_file.read(4096)
        if '"shapes"' in head:
            labelme_jsons.append(path)
        elif '"annotations"' in head or '"images"' in head:
            coco_jsons.append(path)
    xmls = glob.glob(os.path.join(args.src, "**", "*.xml"), recursive=True)
    samples = []
    for json_path in coco_jsons:
        print("COCO:", json_path)
        samples += read_coco(args.src, json_path)
    if labelme_jsons:
        print("labelme: %d json files" % len(labelme_jsons))
        samples += read_labelme(labelme_jsons)
    if not samples and xmls:
        print("VOC: %d xml files" % len(xmls))
        samples = read_voc(args.src, xmls)
    if not samples:
        sys.exit("アノテーション（COCO / labelme の json か VOC の xml）が見つかりません: " + args.src)

    random.seed(0)
    train, val = split_by_folder(samples, args.val)
    n_val = len(val)
    unknown, counts = {}, [0] * len(names)
    for split, part in (("val", val), ("train", train)):
        img_dir = os.path.join(args.out, "images", split)
        lbl_dir = os.path.join(args.out, "labels", split)
        # 前回の出力（古いリンク・ラベル・水増し画像）を消してから作る。
        # 残すと、番号がずれて画像とラベルが食い違ったり、消した画像へのリンクが残ったりする
        for old_dir in (img_dir, lbl_dir):
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i, (path, width, height, boxes) in enumerate(part):
            stem = "%s_%06d" % (split, i)
            ext = os.path.splitext(path)[1].lower()
            link = os.path.join(img_dir, stem + ext)
            os.symlink(os.path.abspath(path), link)  # 画像はコピーせずリンク
            lines = []
            for name, b_x, b_y, b_w, b_h in boxes:
                idx = lookup.get(normalize(str(name)))
                if idx is None:
                    unknown[name] = unknown.get(name, 0) + 1
                    continue
                counts[idx] += 1
                c_x, c_y = (b_x + b_w / 2) / width, (b_y + b_h / 2) / height
                lines.append("%d %.6f %.6f %.6f %.6f" % (idx, c_x, c_y, b_w / width, b_h / height))
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as txt:
                txt.write("\n".join(lines) + ("\n" if lines else ""))

    with open(os.path.join(args.out, "data.yaml"), "w") as yaml:
        yaml.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n"
                   % os.path.abspath(args.out))
        for i, name in enumerate(names):
            yaml.write("  %d: %s\n" % (i, name))

    print("images: train %d / val %d" % (len(samples) - n_val, n_val))
    print("boxes per class:", {names[i]: c for i, c in enumerate(counts) if c})
    if unknown:
        print("[warn] wod-c に無いクラス名（無視した）:", unknown)
    print("->", os.path.join(args.out, "data.yaml"))


if __name__ == "__main__":
    main()
