#!/bin/bash -eu
# YOLOv8 を学習し、できた重みを wrs_detector/checkpoints/yolo_wrs.pt に置く。
# GPU をシミュレータと取り合うので、誰もシミュレーションしていない時間に実行すること。
#
# 使い方: tools/yolo/train_yolo.sh [モデル(yolov8s.pt)] [エポック数(100)] [バッチ(16)]
MODEL=${1:-yolov8s.pt}
EPOCHS=${2:-100}
BATCH=${3:-16}
DATA=/workspace/datasets/wrs_yolo/data.yaml
YOLO_ENV=/workspace/.yolo_env

if [ ! -f "${DATA}" ]; then
  echo "${DATA} がありません。先に tools/yolo/convert_to_yolo.py を実行" >&2
  exit 1
fi

unset PYTHONPATH
export PYTHONPATH="${YOLO_ENV}"
cd /workspace
/usr/bin/python3 - <<EOF
from ultralytics import YOLO
model = YOLO("${MODEL}")  # COCO で事前学習した重みから学習を始める
model.train(data="${DATA}", epochs=${EPOCHS}, batch=${BATCH}, imgsz=640, device=0,
            project="/workspace/runs", name="wrs", exist_ok=True,
            fliplr=0.0)  # 左右反転は文字・形の向きが変わるので使わない
EOF

cp /workspace/runs/wrs/weights/best.pt /workspace/src/wrs_detector/checkpoints/yolo_wrs.pt
echo "-> src/wrs_detector/checkpoints/yolo_wrs.pt"
