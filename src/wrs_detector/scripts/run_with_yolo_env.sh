#!/bin/bash
# YOLO 用: Python 3.8（ROS noetic 本体）と /workspace/.yolo_env のライブラリでノードを起動する
# 使い方（launch-prefix から呼ばれる）: run_with_yolo_env.sh <script.py> [args...]
YOLO_ENV=${YOLO_ENV:-/workspace/.yolo_env}
if [ ! -d "${YOLO_ENV}/ultralytics" ]; then
  echo "[run_with_yolo_env] ${YOLO_ENV} に ultralytics がありません。README の YOLO の章を参照" >&2
  exit 1
fi
export PYTHONPATH="${YOLO_ENV}:${PYTHONPATH}"
exec /usr/bin/python3 "$@"
