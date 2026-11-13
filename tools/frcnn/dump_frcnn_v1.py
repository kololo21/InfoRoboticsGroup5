#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
今の Faster R-CNN（id9_model_e50.pth、ROS の Python / torch 1.7 で動く）の検出結果を、
YOLO 形式のデータセットの val について JSON に書き出す。train_frcnn_v2.py --compare-v1 で比べるのに使う。
検出ノードと同じく、画像は BGR で渡す。

使い方:
  source /opt/ros/noetic/setup.bash && source /workspace/devel/setup.bash
  python3 /workspace/tools/frcnn/dump_frcnn_v1.py /workspace/datasets/wrs_yolo /workspace/runs/id9_val.json
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src",
                                "wrs_detector", "scripts"))
from wrs_detector.frcnn_detector import FasterRcnnDetector  # noqa: E402  pylint: disable=wrong-import-position

DETECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "wrs_detector")


def main():
    """62401738 いしい
    val の各画像を検出し、{画像のパス: [[ラベル名, スコア, [x0, y0, x1, y1]], ...]} を書く
    """
    dataset, out = sys.argv[1], sys.argv[2]
    checkpoint = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DETECTOR_DIR, "checkpoints",
                                                                     "id9_model_e50.pth")
    detector = FasterRcnnDetector(os.path.join(DETECTOR_DIR, "config", "wod-c_classes.json"), checkpoint)
    result = {}
    for path in sorted(glob.glob(os.path.join(dataset, "images", "val", "*"))):
        bgr = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
        result[os.path.realpath(path)] = [
            [d["label"], float(d["score"]),
             [int(d["bbox"]["x"]), int(d["bbox"]["y"]),
              int(d["bbox"]["x"] + d["bbox"]["w"]), int(d["bbox"]["y"] + d["bbox"]["h"])]]
            for d in detector.predict(bgr, threshold=0.01)]
    with open(out, "w") as json_file:
        json.dump(result, json_file)
    print("%d images -> %s" % (len(result), out))


if __name__ == "__main__":
    main()
