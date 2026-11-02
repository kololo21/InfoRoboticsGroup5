#!/bin/bash -eu
# 使い方: [DETECTOR=yolo] start_all.sh [a|b|c]
#   引数なし → wrs_main_node.py（統合版）、a/b/c → wrs_main_node_a.py などを実行
#   DETECTOR=yolo → 物体検出に YOLOv8 を使う（省略時は今までの Faster R-CNN）

if [ $# -ge 1 ]; then
  case "$1" in
    a|b|c) NODE="wrs_main_node_$1.py" ;;
    *) echo "usage: $0 [a|b|c]" >&2; exit 1 ;;
  esac
else
  NODE="wrs_main_node.py"
fi
DETECTOR=${DETECTOR:-frcnn}
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
