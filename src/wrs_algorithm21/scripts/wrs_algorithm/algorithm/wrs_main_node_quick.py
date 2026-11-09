#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
短縮確認ノード: 1 物体だけ「探す → つかむ → 置く」を、本番と同じ関数で実行し、結果を 1 行で記録する。

目的: GPU のない PC(RTF 約 0.3)や、実験室での素早い確認。Task 1 を通すと実時間で数十分かかるが、
これは 1 物体だけなので、数分で「つかめるか」「運ぶ途中で落とさないか」「置けるか」を見られる。

使い方(tools/experiment/run_experiment.sh -Q が、これを呼ぶ):
  QUICK_BASE_NODE=wrs_main_node_c.py QUICK_DRAWERS=0 start_all.sh quick
  QUICK_BASE_NODE  読み込むノードのファイル名(同じフォルダ)。既定 wrs_main_node_c.py
  QUICK_DRAWERS    1 なら、物体を取る前に、本番と同じく引き出しを開ける(既定 0)

読み込むノードのファイルは、変更しない(クラス WrsMainController を、ファイルから読み込むだけ)。
本番の execute_task1 と同じ pick_here → place_until_scored の経路を通るので、結果は本番の動きを反映する。
提出するコードではなく、確認用のツール。
"""
import importlib.util
import json
import os
import traceback

import rospy
from std_msgs.msg import Float32

from wrs_algorithm.util import gripper

# 本番の execute_task1 と同じ探索地点(順に探して、最初につかめた物体で止める)
SEARCH_POINTS = [
    ("tall_table", "look_at_tall_table"),
    ("near_long_table_l", "look_at_near_floor"),
    ("long_table_r", "look_at_tall_table"),
]
REQUIRED_METHODS = (
    "wait_for_controllers", "change_pose", "goto_name", "pick_here", "finger_gap",
    "normalize_placement_label", "place_until_scored",
)


def load_controller_class(path):
    """ノードのファイルから WrsMainController を読み込む(ファイルは変更しない)"""
    spec = importlib.util.spec_from_file_location("quick_base_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WrsMainController


def read_score():
    """今のスコア(/score)。取れなければ None"""
    try:
        return rospy.wait_for_message("/score", Float32, timeout=3.0).data
    except rospy.ROSException:
        return None


def run_quick(ctrl, drawers):
    """1 物体だけ 探す → つかむ → 置く を実行して、結果の辞書を返す"""
    result = {"mode": "quick", "drawers": drawers, "error": None}
    start = rospy.Time.now()
    if not ctrl.wait_for_controllers():
        result["error"] = "controllers did not start"
        return result
    ctrl.run_start = rospy.Time.now()   # 本番の run() と同じ(時間配分の基準)
    ctrl.change_pose("all_neutral")
    if drawers:
        ctrl.prescan_containers()
        ctrl.open_drawers()

    label = None
    for plc, pose in SEARCH_POINTS:
        ctrl.goto_name(plc)
        ctrl.change_pose(pose)
        gripper.command(1)
        label, status = ctrl.pick_here(plc, pose)
        result["pick_place"], result["pick_status"] = plc, status
        if label is not None:
            break
    if label is None:
        result["error"] = "no object was grasped"
        result["elapsed_sim_sec"] = round((rospy.Time.now() - start).to_sec(), 1)
        return result

    result["label"] = label
    gap = ctrl.finger_gap()   # つかんだ直後の指の間隔(0.00 付近なら、何も持っていない)
    result["finger_gap_after_pick"] = None if gap is None else round(gap, 4)
    label = ctrl.normalize_placement_label(label)
    result["placed"] = bool(ctrl.place_until_scored(label))
    result["last_placement"] = getattr(ctrl, "last_placement_result", None)
    result["score"] = read_score()
    result["elapsed_sim_sec"] = round((rospy.Time.now() - start).to_sec(), 1)
    return result


def main():
    """ノードを起動して、短縮確認を 1 回実行する"""
    rospy.init_node("quick_check")
    base = os.environ.get("QUICK_BASE_NODE", "wrs_main_node_c.py")
    drawers = os.environ.get("QUICK_DRAWERS", "0") == "1"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), base)
    result = {"mode": "quick", "base_node": base, "drawers": drawers, "error": None}
    try:
        controller_class = load_controller_class(path)
        missing = [m for m in REQUIRED_METHODS if not hasattr(controller_class, m)]
        if missing:
            result["error"] = "{} has no {} (quick check needs a c.py-family node)".format(
                base, ", ".join(missing))
        else:
            ctrl = controller_class()
            rospy.loginfo("node initialized [%s]", rospy.get_name())
            rospy.loginfo("#### start QUICK check (%s, drawers=%s) ####", base, drawers)
            result.update(run_quick(ctrl, drawers))
            result["base_node"] = base
    except rospy.ROSInterruptException:
        result["error"] = "interrupted"
    except Exception as error:  # pylint: disable=broad-except
        result["error"] = "{}: {}".format(type(error).__name__, error)
        rospy.logerr("quick check failed:\n%s", traceback.format_exc())
    rospy.loginfo("QUICK RESULT: %s", json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
