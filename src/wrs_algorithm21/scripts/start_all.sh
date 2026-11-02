#!/bin/bash -eu
# 使い方: [DETECTOR=yolo] start_all.sh [a|b|c]
#   引数なし → wrs_main_node.py（統合版）、a/b/c → wrs_main_node_a.py などを実行
#   DETECTOR=yolo / frcnn → 物体検出器を指定する
#   省略時: 統合版（引数なし）と c は YOLOv8、a・b は今までの Faster R-CNN
#   YOLO のライブラリ（/workspace/.yolo_env）か重みがなければ Faster R-CNN に戻す

if [ $# -ge 1 ]; then
  case "$1" in
    a|b|c) NODE="wrs_main_node_$1.py" ;;
    *) echo "usage: $0 [a|b|c]" >&2; exit 1 ;;
  esac
else
  NODE="wrs_main_node.py"
fi
if [ -z "${DETECTOR:-}" ]; then
  case "${NODE}" in
    wrs_main_node.py|wrs_main_node_c.py) DETECTOR=yolo ;;
    *) DETECTOR=frcnn ;;
  esac
fi
if [ "${DETECTOR}" = "yolo" ]; then
  if [ ! -d /workspace/.yolo_env/ultralytics ] || \
     [ ! -f /workspace/src/wrs_detector/checkpoints/yolo_wrs.pt ]; then
    echo "[warn] YOLO のライブラリか重みがないので Faster R-CNN で実行します（README の YOLO の章）" >&2
    DETECTOR=frcnn
  fi
fi
case "${DETECTOR}" in
  frcnn) DETECTOR_LAUNCH="frcnn_finetuned.launch" ;;
  yolo)  DETECTOR_LAUNCH="yolo.launch" ;;
  *) echo "DETECTOR must be frcnn or yolo" >&2; exit 1 ;;
esac
echo "run: ${NODE} / detector: ${DETECTOR}"

unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash
roslaunch hsr_perception pcd_to_tf.launch &
roslaunch wrs_detector "${DETECTOR_LAUNCH}" &
roslaunch wrs_algorithm start_task.launch node:="${NODE}" &

wait
