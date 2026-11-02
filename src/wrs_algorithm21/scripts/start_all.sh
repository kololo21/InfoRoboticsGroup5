#!/bin/bash -eu
# 使い方: start_all.sh [a|b|c]
#   引数なし → wrs_main_node.py（統合版）、a/b/c → wrs_main_node_a.py などを実行

if [ $# -ge 1 ]; then
  case "$1" in
    a|b|c) NODE="wrs_main_node_$1.py" ;;
    *) echo "usage: $0 [a|b|c]" >&2; exit 1 ;;
  esac
else
  NODE="wrs_main_node.py"
fi
echo "run: ${NODE}"

unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash
roslaunch hsr_perception pcd_to_tf.launch &
roslaunch wrs_detector frcnn_finetuned.launch &
roslaunch wrs_algorithm start_task.launch node:="${NODE}" &

wait
