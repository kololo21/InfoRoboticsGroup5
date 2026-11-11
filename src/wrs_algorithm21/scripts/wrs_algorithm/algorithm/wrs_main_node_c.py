#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import json
import os
import traceback
import time
import rospy
import rospkg
import tf2_ros
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetPlan
from actionlib_msgs.msg import GoalStatus
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper
import math
import copy
import heapq
import collections
import threading


DIAG = None  # 原因切り分けの記録（WrsMainController.__init__ で作る）


class Diagnostics(object):
    """62401738 いしい
    腕の指令が通らない原因の切り分け用の記録。ロボットの動きは変えない（記録するだけ）。
    node.log に "[DIAG] {json}" の行を出す。集計は tools/experiment/diag_verdict.py。

      仮説1 CPU 環境で計画が間に合わない
        → MoveIt の呼び出しごとに、かかった時間・move_group の状態（PLANNING のまま終わったか、
          MONITOR＝実行に入ったか）・CPU 使用率・シミュレータの速さ（RTF）・手の動きを記録する。
      仮説2 計画の負荷でコントローラが止まる
        → 腕の指令が通らなかった瞬間に、コントローラの状態と、action の status が届いていたか
          （常駐の購読で見る）・joint_states が来ていたかを記録する。5 秒ごとに受信の頻度も記録する。
      仮説3 move_joints が毎回新しい接続を作り、5 秒では間に合わない
        → 接続にかかった時間を毎回記録し、失敗したときは 15 秒まで待って繋がるかを見る。

    CPU 使用率・負荷は、ワークスペースのコンテナから見える PC 全体の値（シミュレータのプロセス単体ではない）。
    ~diagnose:=false で無効にできる。
    """
    ARM_ACTION = "/hsrb/arm_trajectory_controller/follow_joint_trajectory"
    HEAD_ACTION = "/hsrb/head_trajectory_controller/follow_joint_trajectory"

    def __init__(self, lookup_xyz=None):
        self.lookup_xyz = lookup_xyz
        self.stamps = dict((key, collections.deque(maxlen=400))
                           for key in ("arm", "head", "joint_states"))
        self.mg_states = []
        self.mg_result = None
        self.mg_ready = False
        self.last_moveit = None
        self.last_tick = None
        self.retry_sec = float(rospy.get_param("~diagnose_retry_sec", 15.0))
        self.tick_sec = float(rospy.get_param("~diagnose_tick_sec", 5.0))

    # ---- 受信と計測 ----
    def start(self):
        """購読と、5 秒ごとの記録を始める"""
        from actionlib_msgs.msg import GoalStatusArray
        from sensor_msgs.msg import JointState
        for key, action in (("arm", self.ARM_ACTION), ("head", self.HEAD_ACTION)):
            rospy.Subscriber(action + "/status", GoalStatusArray, self._on_stamp,
                             callback_args=key, queue_size=1)
        rospy.Subscriber("/hsrb/joint_states", JointState, self._on_stamp,
                         callback_args="joint_states", queue_size=1)
        self.find_move_group()
        thread = threading.Thread(target=self._loop)
        thread.daemon = True
        thread.start()

    def _on_stamp(self, _msg, key):
        self.stamps[key].append(time.time())

    def find_move_group(self):
        """move_group の feedback / result の購読（見つかるまで毎回探す）"""
        if self.mg_ready:
            return
        try:
            from moveit_msgs.msg import MoveGroupActionFeedback, MoveGroupActionResult
            topics = dict((name, kind) for name, kind in rospy.get_published_topics())
            for name, kind in topics.items():
                if kind == "moveit_msgs/MoveGroupActionFeedback":
                    rospy.Subscriber(name, MoveGroupActionFeedback, self._on_mg_feedback,
                                     queue_size=10)
                    self.mg_ready = True
                elif kind == "moveit_msgs/MoveGroupActionResult":
                    rospy.Subscriber(name, MoveGroupActionResult, self._on_mg_result,
                                     queue_size=10)
        except Exception:  # pylint: disable=broad-except
            pass

    def _on_mg_feedback(self, msg):
        state = str(msg.feedback.state)
        if not self.mg_states or self.mg_states[-1] != state:
            self.mg_states.append(state)

    def _on_mg_result(self, msg):
        self.mg_result = {"wall": time.time(), "code": int(msg.result.error_code.val),
                          "text": str(msg.status.text)}

    @staticmethod
    def read_cpu():
        """(止まっている時間, 全体の時間)。/proc/stat は PC 全体の値"""
        try:
            with open("/proc/stat") as stat:
                fields = [float(v) for v in stat.readline().split()[1:]]
            return fields[3] + fields[4], sum(fields[:8])
        except Exception:  # pylint: disable=broad-except
            return None

    @staticmethod
    def cpu_util(before, after):
        """2 回の read_cpu の間の CPU 使用率（0〜1）"""
        if before is None or after is None or after[1] <= before[1]:
            return None
        return round(1.0 - (after[0] - before[0]) / (after[1] - before[1]), 3)

    @staticmethod
    def load1():
        try:
            return round(os.getloadavg()[0], 2)
        except Exception:  # pylint: disable=broad-except
            return None

    def age(self, key):
        """最後に受信してからの秒数。一度も届いていなければ None"""
        stamps = self.stamps[key]
        return round(time.time() - stamps[-1], 2) if stamps else None

    def hz(self, key, window):
        now = time.time()
        return round(sum(1 for t in list(self.stamps[key]) if now - t <= window) / window, 1)

    def palm(self):
        return self.lookup_xyz("hand_palm_link") if self.lookup_xyz else None

    @staticmethod
    def dist(a, b):
        return round(math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b))), 3)

    def emit(self, rec):
        rospy.loginfo("[DIAG] %s", json.dumps(rec, sort_keys=True, default=str))

    def _loop(self):
        """5 秒ごとに、シミュレータの速さ・CPU・受信の頻度を記録する"""
        prev = (time.time(), rospy.Time.now().to_sec(), self.read_cpu())
        while not rospy.is_shutdown():
            time.sleep(self.tick_sec)
            try:
                cur = (time.time(), rospy.Time.now().to_sec(), self.read_cpu())
                wall = cur[0] - prev[0]
                rec = {"kind": "tick", "t_sim": round(cur[1], 1),
                       "rtf": round((cur[1] - prev[1]) / wall, 2) if wall > 0 else None,
                       "cpu_util": self.cpu_util(prev[2], cur[2]), "load1": self.load1(),
                       "cores": os.cpu_count(),
                       "arm_hz": self.hz("arm", self.tick_sec),
                       "head_hz": self.hz("head", self.tick_sec),
                       "js_hz": self.hz("joint_states", self.tick_sec)}
                self.last_tick = rec
                self.emit(rec)
                prev = cur
            except Exception:  # pylint: disable=broad-except
                pass

    # ---- MoveIt の呼び出しの記録 ----
    def install(self, modules):
        """whole_body / gripper の関数を、記録つきの関数に差し替える（戻り値と例外はそのまま）"""
        whole_body_module, gripper_module = modules
        targets = (
            (whole_body_module, "move_end_effector_pose", "ee_pose"),
            (whole_body_module, "move_to_neutral", "neutral"),
            (whole_body_module, "move_to_go", "to_go"),
            (whole_body_module, "move_head_tilt", "head_tilt"),
            (whole_body_module, "move_to_joint_positions", "joint_positions"),
            (gripper_module, "command", "gripper"),
        )
        newly = False
        for module, name, label in targets:
            func = getattr(module, name, None)
            if func is not None and not getattr(func, "diag_wrapped", False):
                setattr(module, name, self.wrap(label, func))
                newly = True
        if not newly:
            return
        info = {"kind": "start", "cores": os.cpu_count(), "load1": self.load1(),
                "retry_sec": self.retry_sec, "tick_sec": self.tick_sec}
        try:
            command = whole_body_module.WHOLE_BODY_CMD
            info["planning_time"] = command.get_planning_time()
            info["planning_attempts"] = command.get_num_planning_attempts()
        except Exception:  # pylint: disable=broad-except
            pass
        self.emit(info)

    def wrap(self, label, func):
        diag = self

        def wrapper(*args, **kwargs):
            ctx = None
            try:
                ctx = diag._begin(label, args)
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                result = func(*args, **kwargs)
            except Exception as error:
                diag._finish(ctx, "exception: {!r}".format(error))
                raise
            diag._finish(ctx, result)
            return result

        wrapper.__name__ = getattr(func, "__name__", label)
        wrapper.diag_wrapped = True
        return wrapper

    def _begin(self, label, args):
        self.find_move_group()
        self.mg_states = []
        self.mg_result = None
        ctx = {"label": label, "wall": time.time(), "sim": rospy.Time.now().to_sec(),
               "cpu": self.read_cpu(),
               "palm": self.palm() if label not in ("gripper", "head_tilt") else None}
        if label == "ee_pose" and len(args) >= 3:
            ctx["target"] = tuple(float(v) for v in args[:3])
        if label == "gripper" and args:
            ctx["value"] = args[0]
        return ctx

    def _finish(self, ctx, result):
        if ctx is None:
            return
        try:
            if result is not True:
                time.sleep(0.1)  # move_group の result が少し遅れて届くことがある
            wall = time.time() - ctx["wall"]
            sim = rospy.Time.now().to_sec() - ctx["sim"]
            rec = {"kind": "moveit", "label": ctx["label"], "t_sim": round(ctx["sim"], 1),
                   "ok": result if isinstance(result, str) else bool(result),
                   "wall_s": round(wall, 2), "sim_s": round(sim, 2),
                   "rtf": round(sim / wall, 2) if wall > 0 else None,
                   "cpu_util": self.cpu_util(ctx["cpu"], self.read_cpu()),
                   "load1": self.load1(), "cores": os.cpu_count(),
                   "states": list(self.mg_states), "mg_code": None, "mg_text": None}
            mg_result = self.mg_result
            if mg_result is not None and mg_result["wall"] >= ctx["wall"]:
                rec["mg_code"], rec["mg_text"] = mg_result["code"], mg_result["text"]
            if "value" in ctx:
                rec["value"] = ctx["value"]
            end = self.palm() if ctx["palm"] is not None else None
            if end is not None:
                rec["palm_moved_m"] = self.dist(ctx["palm"], end)
                if "target" in ctx:
                    rec["start_dist_m"] = self.dist(ctx["palm"], ctx["target"])
                    rec["end_dist_m"] = self.dist(end, ctx["target"])
            self.last_moveit = {"label": rec["label"], "ok": rec["ok"], "wall_s": rec["wall_s"],
                                "t_sim": rec["t_sim"]}
            self.emit(rec)
        except Exception:  # pylint: disable=broad-except
            pass

    # ---- move_joints の接続の記録 ----
    def joints_done(self, connect_log, success, total_wall):
        """move_joints 1 回ごとの、接続にかかった時間と結果"""
        try:
            self.emit({"kind": "joints", "t_sim": round(rospy.Time.now().to_sec(), 1),
                       "connect_wall_s": connect_log, "ok": bool(success),
                       "total_wall_s": round(total_wall, 2)})
        except Exception:  # pylint: disable=broad-except
            pass

    @staticmethod
    def list_controllers(timeout=3.0):
        """コントローラの状態。応答がなければそのことを返す（止まっていること自体が手がかり）"""
        out = {}

        def call():
            try:
                from controller_manager_msgs.srv import ListControllers
                name = "/hsrb/controller_manager/list_controllers"
                rospy.wait_for_service(name, timeout=2.0)
                for controller in rospy.ServiceProxy(name, ListControllers)().controller:
                    out[controller.name] = controller.state
            except Exception as error:  # pylint: disable=broad-except
                out["error"] = repr(error)

        thread = threading.Thread(target=call)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            out["error"] = "no response in {:.0f}s".format(timeout)
        return dict(out)

    def controller_unavailable(self, client, action_name, first_wait):
        """腕・頭の指令が 5 秒待っても繋がらなかった瞬間の記録。
        その後 retry_sec まで待って、繋がれば True を返す（繋がれば指令は続けて出される）。"""
        key = "arm" if "arm_" in action_name else "head"
        rec = {"kind": "controller_unavailable", "action": key,
               "t_sim": round(rospy.Time.now().to_sec(), 1), "first_wait": first_wait,
               "status_age_s": self.age(key), "status_hz": self.hz(key, 5.0),
               "arm_status_age_s": self.age("arm"), "head_status_age_s": self.age("head"),
               "js_age_s": self.age("joint_states"), "js_hz": self.hz("joint_states", 5.0),
               "load1": self.load1(), "cores": os.cpu_count(), "last_tick": self.last_tick,
               "last_moveit": self.last_moveit}
        try:
            rec["controllers"] = self.list_controllers()
        except Exception:  # pylint: disable=broad-except
            pass
        wall0, sim0 = time.time(), rospy.Time.now().to_sec()
        connected = bool(client.wait_for_server(rospy.Duration(self.retry_sec)))
        rec["retry"] = {"connected": connected, "limit_s": self.retry_sec,
                        "wall_s": round(time.time() - wall0, 2),
                        "sim_s": round(rospy.Time.now().to_sec() - sim0, 2)}
        rec["status_age_after_retry_s"] = self.age(key)
        self.emit(rec)
        return connected


class WrsMainController(object):
    """
    WRSのシミュレーション環境内でタスクを実行するクラス
    """
    IGNORE_LIST = ["small_marker", "large_marker", "lego_duplo", "spatula", "nine_hole_peg_test"]
    GRASP_TF_NAME = "object_grasping"
    GRASP_BACK_SAFE = {"z": 0.05, "xy": 0.3}
    GRASP_BACK = {"z": 0.05, "xy": 0.1}
    HAND_PALM_OFFSET = 0.05  # hand_palm_linkは指の付け根なので、把持のために少しずらす必要がある
    HAND_PALM_Z_OFFSET = 0.075
    DETECT_CNT = 1
    TROFAST_Y_OFFSET = 0.2

    def __init__(self):
        """サンプルのまま（担当なし）
        変数・設定ファイル・ROS通信を初期化する
        """
        # 変数の初期化
        self.instruction_list = []
        self.detection_list = []

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses = self.load_json(self.get_path(["config", "poses.json"]))
        self.load_placement_overrides()
        self.coordinates["positions"].setdefault("container_scan", list(self.CONTAINER_SCAN_POINT))

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.instruction_sub = rospy.Subscriber(
            "/message", String, self.instruction_cb, queue_size=10)
        self.detection_sub = rospy.Subscriber(
            "/detect_msg", String, self.detection_cb, queue_size=10)
        # シミュレータの得点（画面左上の数字）。置いた後に増えたかで、置けたかを確かめる
        self.score = None
        self.score_sub = rospy.Subscriber("/score", Float32, self.score_cb, queue_size=10)

        # 腕の指令が通らない原因の切り分け用の記録（動きは変えない。~diagnose:=false で無効）
        global DIAG
        if DIAG is None and rospy.get_param("~diagnose", True):
            try:
                DIAG = Diagnostics(self.lookup_xyz)
                DIAG.start()
                DIAG.install((whole_body, gripper))
            except Exception as error:  # pylint: disable=broad-except
                DIAG = None
                rospy.logwarn("diagnostics could not start: %s", error)

    @staticmethod
    def get_path(pathes, package="wrs_algorithm"):
        """サンプルのまま（担当なし）
        ROSパッケージ名とファイルまでのパスを指定して、ファイルのパスを取得する
        """
        if not pathes:  # check if the list is empty
            rospy.logerr("Can NOT resolve file path.")
            raise ValueError("You must specify the path to file.")
        pkg_path = rospkg.RosPack().get_path(package)
        path = os.path.join(*pathes)
        return os.path.join(pkg_path, path)

    @staticmethod
    def load_json(path):
        """サンプルのまま（担当なし）
        jsonファイルを辞書型で読み込む
        """
        with open(path, "r") as json_file:
            return json.load(json_file)

    def instruction_cb(self, msg):
        """サンプルのまま（担当なし）
        指示文を受信する
        """
        rospy.loginfo("instruction received. [%s]", msg.data)
        self.instruction_list.append(msg.data)

    def score_cb(self, msg):
        """62401738 いしい
        /score（シミュレータの合計得点）を覚えておく。
        """
        self.score = float(msg.data)

    def detection_cb(self, msg):
        """サンプルのまま（担当なし）
        検出結果を受信する
        """
        rospy.loginfo("received [Collision detected with %s]", msg.data)
        self.detection_list.append(msg.data)

    def get_relative_coordinate(self, parent, child):
        """サンプルのまま（担当なし）
        tfで相対座標を取得する
        """
        try:
            # 4秒待機して各tfが存在すれば相対関係をセット
            trans = self.tf_buffer.lookup_transform(
                parent, child, rospy.Time.now(), rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = "failed to get transform between [{}] and [{}]\n".format(parent, child)
            log_str += traceback.format_exc()
            rospy.logerr(log_str)
            return None

    def goto_name(self, name):
        """サンプルのまま（担当なし）
        waypoint名で指定された場所に移動する
        """
        if name in self.coordinates["positions"].keys():
            pos = self.coordinates["positions"][name]
            rospy.loginfo("go to [%s](%.2f, %.2f, %.2f)", name, pos[0], pos[1], pos[2])
            return omni_base.go_abs(pos[0], pos[1], pos[2])

        rospy.logerr("unknown waypoint name [%s]", name)
        return False

    def goto_pos(self, pos):
        """サンプルのまま（担当なし）
        waypoint名で指定された場所に移動する
        """
        rospy.loginfo("go to [raw_pos](%.2f, %.2f, %.2f)", pos[0], pos[1], pos[2])
        return omni_base.go_abs(pos[0], pos[1], pos[2])

    def change_pose(self, name):
        """サンプルのまま（担当なし）
        指定された姿勢名に遷移する
        （B版の MoveIt で計画する方式は、腕が遠回りして物に当たるので week1 の直接指令に戻した）
        """
        if name in self.poses.keys():
            rospy.loginfo("change pose to [%s]", name)
            return self.move_joints(self.poses[name])

        rospy.logerr("unknown pose name [%s]", name)
        return False

    @staticmethod
    def move_joints(joints, timeout=10.0):
        """62401738 いしい
        whole_body.move_to_joint_positions と同じ直接指令（腕・頭の trajectory controller）で動かす。
        whole_body 側は get_result()（結果のメッセージ）を GoalStatus.SUCCEEDED（整数）と比べているので、
        腕が動けても必ず False を返す。そのため上段の引き出しで「cannot take the drawer grasp pose」になり、
        配置でも transport / place / retreat の段階で失敗扱いになって物を置かずに中断していた（11/9）。
        ここでは get_state() で成功を判定する。
        """
        import actionlib
        from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        groups = (
            ("/hsrb/arm_trajectory_controller/follow_joint_trajectory",
             ("arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
              "wrist_flex_joint", "wrist_roll_joint")),
            ("/hsrb/head_trajectory_controller/follow_joint_trajectory",
             ("head_pan_joint", "head_tilt_joint")),
        )
        clients = []
        started, connect_log = time.time(), {}
        for action_name, names in groups:
            point = JointTrajectoryPoint()
            trajectory = JointTrajectory()
            for name in names:
                if name in joints:
                    trajectory.joint_names.append(name)
                    point.positions.append(joints[name])
                    point.velocities.append(0.0)
            if not trajectory.joint_names:
                continue
            point.time_from_start = rospy.Duration(1.0)
            trajectory.points = [point]
            client = actionlib.SimpleActionClient(action_name, FollowJointTrajectoryAction)
            wall0, sim0 = time.time(), rospy.Time.now().to_sec()
            connected = client.wait_for_server(rospy.Duration(5.0))
            first_wait = {"wall_s": round(time.time() - wall0, 2),
                          "sim_s": round(rospy.Time.now().to_sec() - sim0, 2)}
            connect_log["arm" if "arm_" in action_name else "head"] = first_wait["wall_s"]
            if not connected:
                rospy.logerr("joint controller unavailable: %s", action_name)
                if DIAG is not None:
                    # 原因の切り分け: 状態を記録し、15 秒まで待って繋がれば指令を続ける
                    try:
                        connected = DIAG.controller_unavailable(client, action_name, first_wait)
                    except Exception:  # pylint: disable=broad-except
                        connected = False
                if not connected:
                    return False
            client.send_goal(FollowJointTrajectoryGoal(trajectory=trajectory))
            clients.append(client)
        success = True
        for client in clients:
            if not client.wait_for_result(rospy.Duration(timeout)):
                client.cancel_goal()
                success = False
            elif client.get_state() != GoalStatus.SUCCEEDED:
                success = False
        if DIAG is not None:
            DIAG.joints_done(connect_log, success, time.time() - started)
        return success

    def check_positions(self):
        """サンプルのまま（担当なし）
        読み込んだ座標ファイルの座標を巡回する
        """
        whole_body.move_to_go()
        for wp_name in self.coordinates["routes"]["test"]:
            self.goto_name(wp_name)
            rospy.sleep(1)

    def get_latest_detection(self):
        """サンプルのまま（担当なし）
        最新の認識結果が到着するまで待つ
        """
        res = self.detection_clt(GetObjectDetectionRequest())
        return res.bboxes

    def get_grasp_coordinate(self, bbox):
        """62418450 向井悠人（B版から取り込み）
        BBox情報から把持座標を取得する
        """
        # B版から取り込み: tf が取れないときは None を返す（呼び出し側で再試行）
        return self.get_bbox_position(bbox)

    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """サンプルのまま（担当なし）
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None:
            return None
        return obj["bbox"]

    @classmethod
    def get_most_graspable_obj(cls, obj_list):
        """62408993 Masami Sakuma
        把持しやすさのスコアが最も高い物体を返す。
        """
        task1_scoreable_labels = {
            "small_marker", "large_marker", "lego_duplo",
            "spatula", "nine_hole_peg_test", "fork", "spoon"
        }

        best_obj = None
        best_score = -1.0

        for obj in obj_list:
            if obj.w <= 0 or obj.h <= 0 or obj.score <= 0:
                continue

            if (obj.label in cls.IGNORE_LIST
                    and obj.label not in task1_scoreable_labels):
                continue

            score = cls.calc_score_bbox(obj)

            if score > best_score:
                best_score = score
                best_obj = obj

        if best_obj is None:
            rospy.logwarn("No graspable object was detected.")
            return None

        rospy.loginfo(
            "Selected object: %s, score=%.3f, confidence=%.2f",
            best_obj.label, best_score, best_obj.score
        )

        return {
            "bbox": best_obj,
            "score": best_score,
            "label": best_obj.label
        }

    @classmethod
    def calc_score_bbox(cls, bbox):
        """62408993 Masami Sakuma
        """
        # Masami Sakuma
        """BBoxの検出信頼度・画像中央からの距離・大きさからスコアを計算する。"""
        center_x = bbox.x + bbox.w / 2.0
        center_y = bbox.y + bbox.h / 2.0

        # 画像中央に近いほど高得点
        dx = abs(320.0 - center_x) / 320.0
        dy = abs(360.0 - center_y) / 240.0
        center_score = max(0.0, 1.0 - (dx + dy) / 2.0)

        confidence_score = max(0.0, min(1.0, float(bbox.score)))

        # 極端に大きすぎる・小さすぎる検出をペナルティ化
        bbox_area = float(bbox.w * bbox.h)
        ideal_area = 10000.0
        size_score = max(0.0, 1.0 - abs(bbox_area - ideal_area) / ideal_area)

        return (
            confidence_score * 0.4
            + center_score * 0.4
            + size_score * 0.2
        )

    @classmethod
    def get_most_graspable_bboxes_by_label(cls, obj_list, label):
        """サンプルのまま（担当なし）
        label名が一致するオブジェクトの中から最も把持すべき物体のbboxを返す
        """
        match_objs = [obj for obj in obj_list if obj.label in label]
        if not match_objs:
            rospy.logwarn("Cannot find a object which labeled with similar name.")
            return None
        return cls.get_most_graspable_bbox(match_objs)

    def grasp_from_side(
        self,
        pos_x, pos_y, pos_z,
        yaw, pitch, roll,
        preliminary="-y"
    ):
        """62408993 Masami Sakuma
        安全退避位置から物体を把持し、空振り判定を行って退避する。
        """
        if preliminary not in ["+y", "-y", "+x", "-x"]:
            raise RuntimeError(
                "unknown grasp preliminary type [{}]".format(preliminary)
            )

        # 1. 軸ずれが大きい場合はロボット台車を水平移動させてIK（逆運動学）解の安定性を確保
        # 統合時に無効化: omni_base.go_rel は util に存在せず（呼ぶとノードが止まる）、
        # pos_x は map 座標の絶対値なので「台車をずらして pos_x=0」にすると手が map の x=0 へ向かう。
        # if abs(pos_x) > 0.12:
        #     rospy.loginfo("Adjusting base position X by %.2f for optimal IK pose.", pos_x)
        #     omni_base.go_rel(pos_x, 0.0, 0.0, timeout=3.0)
        #     pos_x = 0.0

        grasp_back_safe = {
            "x": pos_x,
            "y": pos_y,
            "z": pos_z + self.GRASP_BACK_SAFE["z"]
        }

        sign = 1 if "+" in preliminary else -1
        if "x" in preliminary:
            grasp_back_safe["x"] += sign * self.GRASP_BACK_SAFE["xy"]
        elif "y" in preliminary:
            grasp_back_safe["y"] += sign * self.GRASP_BACK_SAFE["xy"]

        rospy.loginfo("Fast grasp: target=(%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        # グリッパー事前開放
        gripper.command(1)

        # アプローチ姿勢へ移動
        whole_body.move_end_effector_pose(
            grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"],
            yaw, pitch, roll
        )

        # 把持目標位置へ直接移動
        whole_body.move_end_effector_pose(
            pos_x, pos_y, pos_z,
            yaw, pitch, roll
        )

        # 把持実行
        gripper.command(0)
        rospy.sleep(0.3)

        # 把持成功判定（空振りのチェック）
        # ※グリッパー開口幅が0（完全に閉じきっている）場合は物体を掴めていないと判断
        get_state = getattr(gripper, "get_state", None)  # util に無いので今は判定されない
        if get_state is not None:
            gripper_pos = get_state()  # pylint: disable=not-callable
            if gripper_pos is not None and gripper_pos < 0.01:
                rospy.logwarn("Grasp verification failed: Empty grasp detected.")
                # 退出動作を行って失敗を返す
                whole_body.move_end_effector_pose(
                    grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"],
                    yaw, pitch, roll
                )
                return False

        # 退避位置へ戻る
        whole_body.move_end_effector_pose(
            grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"],
            yaw, pitch, roll
        )
        return True

    def grasp_from_front_side(self, grasp_pos):
        """62408993 Masami Sakuma
        正面把持を行う。
        """
        pos_x = grasp_pos.x
        pos_y = grasp_pos.y - self.HAND_PALM_OFFSET
        pos_z = grasp_pos.z

        rospy.loginfo("grasp_from_front_side (%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        return self.grasp_from_side(
            pos_x, pos_y, pos_z,
            -90, -100, 0,
            "-y"
        )

    def grasp_from_upper_side(self, grasp_pos):
        """62408993 Masami Sakuma
        上面把持を行う。
        """
        pos_x = grasp_pos.x
        pos_y = grasp_pos.y
        pos_z = grasp_pos.z + self.HAND_PALM_Z_OFFSET

        rospy.loginfo("grasp_from_upper_side (%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        return self.grasp_from_side(
            pos_x, pos_y, pos_z,
            -90, -160, 0,
            "-y"
        )

    def exec_graspable_method(self, grasp_pos, label=""):
        """62408993 Masami Sakuma
        物体の位置と種類に応じて把持方法を選択する。
        """
        graspable_y = 1.85
        desk_y = 1.5
        desk_z = 0.35

        if graspable_y < grasp_pos.y and grasp_pos.z < desk_z:
            rospy.logwarn(
                "Object is out of graspable area: y=%.2f, z=%.2f",
                grasp_pos.y, grasp_pos.z
            )
            return False

        label = label.lower().replace(" ", "_").replace("-", "_")

        # 上面把持を優先する物体（110 点の版 49344b3 に戻した。小物・薄い物まで上からつかむと、
        # まっすぐ下ろす手が机や隣の物に当たっていた。統合時に いしい が変更）
        upper_grasp_labels = {"cup", "cups", "frisbee", "bowl"}

        if label in upper_grasp_labels:
            return self.grasp_from_upper_side(grasp_pos)
        elif desk_y < grasp_pos.y and grasp_pos.z < desk_z:
            return self.grasp_from_front_side(grasp_pos)
        else:
            return self.grasp_from_upper_side(grasp_pos)

    @staticmethod
    def normalize_placement_label(label):
        """62418450 向井悠人
        YOLO/A側のラベルを配置用に統一し、空のラベルを拒否する。
        """
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Placement requires a nonempty object label")
        label = "_".join(label.strip().lower().replace("-", "_").split())
        return label

    def load_placement_overrides(self):
        """62418450 向井悠人
        実測した配置座標・姿勢をROSパラメータから読み、B側だけに適用する。
        """
        positions = rospy.get_param("~placement_positions", {})
        poses = rospy.get_param("~placement_poses", {})
        targets = rospy.get_param("~placement_targets", {})
        if any(not isinstance(value, dict) for value in (positions, poses, targets)):
            raise ValueError("Placement settings must be dictionaries")
        joint_names = {"arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                       "wrist_flex_joint", "wrist_roll_joint",
                       "head_pan_joint", "head_tilt_joint"}
        for name, values in positions.items():
            if (not isinstance(values, (list, tuple)) or len(values) != 3
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in values)):
                raise ValueError("Invalid placement position: {}".format(name))
        for name, joints in poses.items():
            if (not isinstance(joints, dict) or not joints or set(joints) - joint_names
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in joints.values())):
                raise ValueError("Invalid placement pose: {}".format(name))
        # map座標の手先開放点。センサ推定値または利用を認められた実測値を指定する。
        for name, target in targets.items():
            if (not isinstance(target, dict) or set(target) != {"x", "y", "z", "hover"}
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in target.values())
                    or target["hover"] <= 0
                    or name not in set(self.coordinates["positions"]) | set(positions)):
                raise ValueError("Invalid measured placement target: {}".format(name))
        self.coordinates["positions"].update(positions)
        self.poses.update(poses)
        self.placement_targets = {name: dict(target) for name, target in targets.items()}

    def read_placement_cloud(self, after):
        """62418450 向井悠人
        頭部RGB-Dの新しい深度点群を受信し、撮影時刻のTFでmap座標へ変換する。
        """
        import numpy as np
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs import point_cloud2
        from tf.transformations import quaternion_matrix

        topic = rospy.get_param(
            "~placement_cloud_topic", "/hsrb/head_rgbd_sensor/depth_registered/rectified_points")
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            cloud = rospy.wait_for_message(topic, PointCloud2,
                                           timeout=max(0.01, deadline - time.monotonic()))
            if cloud.header.stamp <= after:
                continue
            if not cloud.header.frame_id or cloud.height <= 1:
                raise ValueError("Placement requires an organized, stamped depth cloud")
            transform = self.tf_buffer.lookup_transform(
                "map", cloud.header.frame_id, cloud.header.stamp, rospy.Duration(2.0)).transform
            pixels = [(u, v) for v in range(0, cloud.height, 3)
                      for u in range(0, cloud.width, 3)]
            points = np.asarray(list(point_cloud2.read_points(
                cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=pixels)))
            if points.ndim != 2 or len(points) < 100:
                raise ValueError("Not enough depth measurements for placement")
            points = points[np.isfinite(points).all(axis=1)]
            q = transform.rotation
            matrix = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
            t = transform.translation
            return points.dot(matrix.T) + np.array([t.x, t.y, t.z]), cloud.header.stamp
        raise ValueError("No new depth cloud after the observation pose")

    @staticmethod
    def detect_container_rims(points):
        """62418450 向井悠人
        深度点群から水平な机面と矩形の縁を抽出し、観測寸法で容器を分類する。
        座標・高さは点群だけから求める。閾値は形状認識用で実環境で要調整。
        """
        import cv2
        import numpy as np

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected an Nx3 map-frame cloud")
        points = points[np.isfinite(points).all(axis=1)]
        points = points[(points[:, 2] > 0.25) & (points[:, 2] < 1.1)]
        if len(points) < 100:
            return []
        # 水平な面はmap座標の同じ高さに多数の点を持つ。
        bins = np.floor(points[:, 2] / 0.008).astype(int)
        levels, counts = np.unique(bins, return_counts=True)
        plane_bin = levels[np.argmax(counts)]
        surface = points[bins == plane_bin]
        if len(surface) < 100 or np.min(np.ptp(surface[:, :2], axis=0)) < 0.25:
            return []
        table_z = float(np.median(surface[:, 2]))
        raised = points[(points[:, 2] > table_z + 0.007)
                        & (points[:, 2] < table_z + 0.18)]
        if len(raised) < 40:
            return []
        resolution = 0.006
        origin = raised[:, :2].min(axis=0)
        cells = np.floor((raised[:, :2] - origin) / resolution).astype(int)
        cells += 2
        shape = cells.max(axis=0) + 3
        if np.any(shape > 1200):
            return []
        mask = np.zeros((shape[1], shape[0]), dtype=np.uint8)
        mask[cells[:, 1], cells[:, 0]] = 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        assignments = labels[cells[:, 1], cells[:, 0]]
        detections = []
        for component in np.unique(assignments):
            if component == 0:
                continue
            cluster = raised[assignments == component]
            if len(cluster) < 40:
                continue
            rect = cv2.minAreaRect(cluster[:, :2].astype(np.float32))
            width, length = sorted(rect[1])
            height = float(np.percentile(cluster[:, 2], 95) - table_z)
            if 0.24 <= width <= 0.42 and 0.28 <= length <= 0.48 and height < 0.065:
                kind = "tray"
            elif 0.14 <= width <= 0.32 and 0.17 <= length <= 0.35 and height >= 0.065:
                kind = "container_a"
            elif 0.07 <= width <= 0.14 and 0.07 <= length <= 0.16 and height >= 0.065:
                kind = "container_b"
            else:
                continue
            corners = cv2.boxPoints(rect)
            # 丸い物体や一辺しか見えていない物体を矩形容器と取り違えない。
            tolerance = min(0.02, width * 0.12)
            corner_support = sum(np.min(np.linalg.norm(cluster[:, :2] - corner, axis=1))
                                 < tolerance for corner in corners)
            if corner_support < 3:
                continue
            center = np.asarray(rect[0])
            angle = math.radians(rect[2])
            axes = np.array([[math.cos(angle), math.sin(angle)],
                             [-math.sin(angle), math.cos(angle)]])
            local = (cluster[:, :2] - center).dot(axes.T)
            half = np.asarray(rect[1]) / 2.0
            edge = np.any(np.abs(local) >= half - tolerance, axis=1)
            if np.mean(edge) < 0.65:
                continue
            detections.append({"kind": kind, "x": float(center[0]), "y": float(center[1]),
                               "rim_z": table_z + height, "width": float(width),
                               "length": float(length)})
        return detections

    @staticmethod
    def merge_container_observations(frames):
        """62418450 向井悠人
        独立した複数フレームで分類と中心位置が一致した容器だけを採用する。
        """
        groups = []
        for frame_id, detections in enumerate(frames):
            for detection in detections:
                matches = [group for group in groups
                           if group[0][1]["kind"] == detection["kind"]
                           and all(math.hypot(item["x"] - detection["x"],
                                              item["y"] - detection["y"]) < 0.04
                                   for _, item in group)]
                if len(matches) > 1:
                    continue
                group = matches[0] if matches else []
                if not matches:
                    groups.append(group)
                if any(index == frame_id for index, _ in group):
                    continue
                group.append((frame_id, detection))
        results = []
        for group in groups:
            if len(group) < 2:
                continue
            count = float(len(group))
            result = {key: sum(item[key] for _, item in group) / count
                      for key in ("x", "y", "rim_z", "width", "length")}
            result["kind"] = group[0][1]["kind"]
            results.append(result)
        return results

    def observe_placement_containers(self):
        """62418450 向井悠人
        既存の観測地点から頭部を振り、深度点群で容器を探す。把持は維持する。
        """
        # 既定は長机の前（container_scan、いしい追加）。引き出しの前（drawer）からは 1.1〜1.8m 先で、
        # 容器が見つからない・2 つに見えることが多かった（11/9）
        waypoint = rospy.get_param("~placement_scan_waypoint", "container_scan")
        if waypoint not in self.coordinates["positions"]:
            raise ValueError("Unknown placement observation waypoint: {}".format(waypoint))
        if (not self.move_placement_pose("all_neutral")
                or not self.navigate_for_placement(waypoint)):
            raise ValueError("Cannot reach the placement observation waypoint")
        frames = []
        for pan in self.CONTAINER_SCAN_PANS:
            head = whole_body.HEAD_CMD
            head.set_joint_value_target({"head_pan_joint": pan,
                                         "head_tilt_joint": self.CONTAINER_SCAN_TILT})
            try:
                if not head.go(wait=True):
                    raise ValueError("Cannot point the depth camera at the deposit area")
            finally:
                head.stop()
            stamp = rospy.Time.now()
            for _ in range(2):
                points, stamp = self.read_placement_cloud(stamp)
                frames.append(self.detect_container_rims(points))
        return self.merge_container_observations(frames)

    def ensure_sensor_placements(self, category, destinations):
        """62418450 向井悠人
        配置設定がない容器をセンサで特定し、接近点と手先開放点を登録する。
        """
        if category not in ("tray", "container_a", "container_b"):
            return
        targets = getattr(self, "placement_targets", {})
        generated = getattr(self, "sensor_placement_names", set())
        if any(place not in generated and place in self.coordinates["positions"]
               and (self.poses.get(pose) or place in targets) for place, pose in destinations):
            return
        # 過去のセンサ推定値は、今回の観測が失敗したときに再使用しない。
        for place in generated:
            targets.pop(place, None)
            self.coordinates["positions"].pop(place, None)
        self.sensor_placement_names = set()
        maximum = 2 if category == "tray" else 1
        # 最初（引き出しを開ける前）に見ておいた結果があれば、それを使う。
        # 引き出しを開けた後に見ると、開いた引き出しが写り込んで容器を見分けられなくなる。
        scan = getattr(self, "container_scan", None)
        found = [item for item in scan["observations"] if item["kind"] == category] if scan else []
        if found and len(found) <= maximum:
            base = scan["base"]
        else:
            observations = self.observe_placement_containers()
            found = [item for item in observations if item["kind"] == category]
            if not found or len(found) > maximum:
                raise ValueError("Missing or ambiguous sensor observations for {}".format(category))
            base = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rospy.Time(0), rospy.Duration(2.0)).transform.translation
        distance = float(rospy.get_param("~placement_standoff", 0.65))
        clearance = float(rospy.get_param("~placement_release_clearance",
                                          self.PLACE_RIM_CLEARANCE))
        hover = float(rospy.get_param("~placement_hover_distance", 0.12))
        if any(not math.isfinite(value) or value <= 0 for value in
               (distance, clearance, hover)):
            raise ValueError("Placement distances must be positive and finite")
        self.placement_targets = targets
        for (place, _), item in zip(destinations, sorted(found, key=lambda value: value["x"])):
            dx, dy = base.x - item["x"], base.y - item["y"]
            norm = math.hypot(dx, dy)
            if norm < 0.1:
                raise ValueError("Observation viewpoint is too close to container center")
            # 真正面（-y 向き）から、put_in_drawer_high の指先が容器の中心に来る位置に立つ
            self.poses.update(self.DRAWER_RELEASE_POSES)
            self.coordinates["positions"][place] = [
                item["x"] - self.DRAWER_HAND_LEFT, item["y"] + self.DRAWER_HAND_FORWARD, -90.0]
            targets[place] = {"x": item["x"], "y": item["y"], "rim_z": item["rim_z"],
                              "z": item["rim_z"] + clearance, "hover": hover}
            # 真正面に行けないとき用の、観測した位置の方から近づく位置（week1 の計算）
            if not hasattr(self, "placement_alternates"):
                self.placement_alternates = {}
            self.placement_alternates[place] = [
                item["x"] + distance * dx / norm, item["y"] + distance * dy / norm,
                math.degrees(math.atan2(-dy, -dx))]
            self.sensor_placement_names.add(place)
            rospy.loginfo("Sensor placement %s: %s", place, json.dumps(item, sort_keys=True))

    # 容器・トレイを探すときに立つ所（長机の容器の 0.7m 手前、-y 向き）と首の向き。
    # これはロボットが立つ場所で、容器の位置ではない（容器の位置は毎回センサで測る）。
    CONTAINER_SCAN_POINT = [1.5, 0.1, -90]
    CONTAINER_SCAN_PANS = (-0.6, 0.0, 0.6)
    CONTAINER_SCAN_TILT = -0.7
    PLACE_RIM_CLEARANCE = 0.18   # 容器・トレイの縁からどれだけ上で離すか（指先の高さ）

    def prescan_containers(self):
        """62401738 いしい
        引き出しを開ける前（手が空で、引き出しが写り込まないうち）に、長机の前から容器・トレイを見ておく。
        結果は ensure_sensor_placements が使う。失敗しても、置くときに今までどおり見直す。
        """
        self.coordinates["positions"].setdefault("container_scan", list(self.CONTAINER_SCAN_POINT))
        try:
            observations = self.observe_placement_containers()
            base = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rospy.Time(0), rospy.Duration(2.0)).transform.translation
        except Exception as error:  # 見られなくても Task 1 は続ける
            rospy.logwarn("container prescan failed: %s", error)
            return False
        self.container_scan = {"observations": observations, "base": base}
        rospy.loginfo("container prescan: %s",
                      ", ".join(sorted(item["kind"] for item in observations)) or "nothing")
        return True

    def move_over_container(self, place, hover=False, linear=False):
        """62418450 向井悠人
        手先の向きを保って中心上へ移動し、下降・退避は直線軌道を使う。
        直線軌道が最後まで計画できなければ実行しない。
        """
        target = self.placement_targets[place]
        transform = self.tf_buffer.lookup_transform(
            "map", "hand_palm_link", rospy.Time(0), rospy.Duration(2.0))
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = target["x"]
        pose.pose.position.y = target["y"]
        pose.pose.position.z = target["z"] + (target["hover"] if hover else 0)
        pose.pose.orientation = transform.transform.rotation
        commander = whole_body.WHOLE_BODY_CMD
        if not linear:
            commander.set_pose_target(pose)
            try:
                return bool(commander.go(wait=True))
            finally:
                commander.stop()
                commander.clear_pose_targets()
        previous_frame = commander.get_pose_reference_frame()
        try:
            commander.set_pose_reference_frame("map")
            trajectory, fraction = commander.compute_cartesian_path(
                [pose.pose], 0.01, 0.0, avoid_collisions=True)
            if (not math.isfinite(fraction) or fraction < 1.0
                    or not trajectory.joint_trajectory.points):
                rospy.logerr("Incomplete placement trajectory: %.3f", fraction)
                return False
            return bool(commander.execute(trajectory, wait=True))
        finally:
            commander.stop()
            commander.set_pose_reference_frame(previous_frame)

    def select_place_for_label(self, label):
        """62418450 向井悠人
        正しいカテゴリの配置先だけを選ぶ。設定不足時は誤配送せず停止する。
        """
        label = self.normalize_placement_label(label)
        categories = {
            "tray": (
                "cracker_box", "sugar_box", "pudding_box", "gelatin_box",
                "potted_meat_can", "master_chef_can", "tuna_fish_can", "chips_can",
                "mustard_bottle", "tomato_soup_can", "banana", "strawberry", "apple",
                "lemon", "peach", "pear", "orange", "plum",
            ),
            "container_a": (
                "windex_bottle", "bleach_cleanser", "sponge", "pitcher_base",
                "pitcher_lid", "plate", "bowl", "spatula", "wine_glass", "mug",
            ),
            "container_b": ("fork", "spoon", "large_marker", "small_marker"),
            "drawer_tools": ("padlock", "bolt_and_nut", "clamp"),
            "drawer_shape": (
                "credit_card_blank", "mini_soccer_ball", "softball", "baseball",
                "tennis_ball", "racquetball", "golf_ball", "marble", "cup",
                "foam_brick", "dice", "rope", "chain",
            ),
            "bin_a": (
                "rubiks_cube", "colored_wood_block", "nine_hole_peg_test",
                "toy_airplane", "lego_duplo", "magazine", "black_t_shirt", "timer",
            ),
            "bin_b": ("unknown", "object"),
        }
        destinations = {
            "tray": (("tray_a_place", "put_in_tray"),
                     ("tray_b_place", "put_in_tray")),
            "container_a": (("container_a_place", "put_in_container_a"),),
            "container_b": (("container_b_place", "put_in_container_b"),),
            "bin_a": (("bin_a_place", "put_in_bin"),),
            "bin_b": (("bin_b_place", "put_in_bin"),),
        }
        category = next((name for name, labels in categories.items() if label in labels), None)
        if category is None:
            raise ValueError("Unrecognized placement label: {}".format(label))
        if category in ("drawer_tools", "drawer_shape"):
            return self.select_drawer_place(label, category)
        self.ensure_sensor_placements(category, destinations[category])
        targets = getattr(self, "placement_targets", {})
        candidates = [(place, pose) for place, pose in destinations[category]
                      if place in self.coordinates["positions"]
                      and (self.poses.get(pose) or place in targets)]
        if candidates:
            selected = self.choose_shortest_placement(candidates)
            rospy.loginfo("place %s (%s) at %s", label, category, selected[0])
            return selected

        required = "; ".join("{} + {}".format(place, pose)
                             for place, pose in destinations[category])
        raise ValueError("Missing placement configuration for {} ({}): {}. "
                         "Set measured placement_positions / placement_poses".format(
                             label, category, required))

    def placement_path_length(self, place):
        """62418450 向井悠人
        move_baseの経路長を取得する。未到達は無限大、計画不能はNone。
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rospy.Time(0), rospy.Duration(1.0))
            start = PoseStamped()
            start.header.frame_id = "map"
            start.pose.position.x = transform.transform.translation.x
            start.pose.position.y = transform.transform.translation.y
            start.pose.orientation = transform.transform.rotation
            goal = PoseStamped()
            goal.header.frame_id = "map"
            x_pos, y_pos, yaw = self.coordinates["positions"][place]
            goal.pose.position.x = x_pos
            goal.pose.position.y = y_pos
            goal.pose.orientation.z = math.sin(math.radians(yaw) / 2.0)
            goal.pose.orientation.w = math.cos(math.radians(yaw) / 2.0)
            service = rospy.get_param("~placement_plan_service", "/move_base/make_plan")
            rospy.wait_for_service(service, timeout=1.0)
            path = rospy.ServiceProxy(service, GetPlan)(start, goal, 0.0).plan.poses
            if not path:
                return float("inf")
            points = [start] + list(path) + [goal]
            return sum(math.hypot(b.pose.position.x - a.pose.position.x,
                                  b.pose.position.y - a.pose.position.y)
                       for a, b in zip(points, points[1:]))
        except (rospy.ROSException, rospy.ServiceException,
                tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn("Placement path unavailable: %s", error)
            return None

    def choose_shortest_placement(self, candidates):
        """62418450 向井悠人
        同カテゴリの候補を計画経路長で比較する。計画未取得なら設定順を使う。
        """
        if len(candidates) == 1:
            return candidates[0]
        lengths = [self.placement_path_length(place) for place, _ in candidates]
        if any(length is None for length in lengths):
            return candidates[0]
        index = min(range(len(candidates)), key=lambda item: lengths[item])
        if math.isinf(lengths[index]):
            raise ValueError("No reachable placement destination")
        return candidates[index]

    def move_placement_pose(self, name):
        """62418450 向井悠人
        配置用姿勢をMoveItで計画・実行し、各グループの成功を確認する。
        """
        joints = self.poses.get(name)
        if not joints:
            return False
        arm_names = {"arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                     "wrist_flex_joint", "wrist_roll_joint"}
        head_names = {"head_pan_joint", "head_tilt_joint"}
        if set(joints) - (arm_names | head_names):
            rospy.logerr("Unsupported joints in placement pose %s", name)
            return False
        for commander, names in ((whole_body.ARM_CMD, arm_names),
                                 (whole_body.HEAD_CMD, head_names)):
            targets = {joint: value for joint, value in joints.items() if joint in names}
            if targets:
                commander.set_joint_value_target(targets)
                try:
                    success = commander.go(wait=True)
                finally:
                    commander.stop()
                if not success:
                    return False
        return True

    def navigate_for_placement(self, place):
        """62418450 向井悠人
        台車の移動に時間制限を設け、失敗時は残ったゴールを取り消す。
        """
        from move_base_msgs.msg import MoveBaseGoal
        from wrs_algorithm.util.mathematics import quaternion_from_euler

        timeout = float(rospy.get_param("~placement_navigation_timeout", 60.0))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("placement_navigation_timeout must be positive and finite")
        client = omni_base.NAVCLIENT
        if not client.wait_for_server(rospy.Duration(2.0)):
            return False
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        x_pos, y_pos, yaw = self.coordinates["positions"][place]
        goal.target_pose.pose.position.x = x_pos
        goal.target_pose.pose.position.y = y_pos
        goal.target_pose.pose.orientation = quaternion_from_euler(yaw, 0, 0)
        client.send_goal(goal)
        if not client.wait_for_result(rospy.Duration(timeout)):
            client.cancel_goal()
            return False
        if client.get_state() != GoalStatus.SUCCEEDED:
            client.cancel_goal()
            return False
        return True

    def navigate_with_alternate(self, place):
        """62401738 いしい
        place へ移動する。行けなければ（開いた引き出しが邪魔など）容器の真正面の位置から近づく。
        """
        alternate = getattr(self, "placement_alternates", {}).get(place)
        use_front = getattr(self, "placement_use_front", set())
        if alternate is not None and place in use_front:
            # 前に行けなかった容器は、最初から真正面の位置へ行く（失敗する移動で時間を使わない）
            self.coordinates["positions"][place] = list(alternate)
            return self.navigate_for_placement(place)
        if self.navigate_for_placement(place):
            return True
        if alternate is None:
            return False
        rospy.logwarn("cannot reach %s; approach from the front %s", place, alternate)
        self.placement_use_front = use_front | {place}
        self.coordinates["positions"][place] = list(alternate)
        return self.navigate_for_placement(place)

    def put_object_by_label(self, label):
        """62418450 向井悠人
        A側で把持成功を確認した後、検出器に依存せずラベルを受け取る。
        """
        if not isinstance(label, str) or not label.strip():
            rospy.logerr("Placement requires a nonempty object label")
            return False
        label = self.normalize_placement_label(label)
        try:
            place, pose = self.select_place_for_label(label)
            return self.put_in_place(place, pose, label=label)
        except Exception as error:  # ロボット動作失敗をTask 1へ返す。
            self.last_placement_result = {
                "label": label, "stage": "select_destination", "released": False,
                "success": False, "error": str(error)}
            rospy.logerr("Placement result: %s",
                         json.dumps(self.last_placement_result, sort_keys=True))
            return False

    def put_in_place(self, place, into_pose, label=None):
        """62418450 向井悠人
        運搬、移動、配置、開放、復帰の順に実行し工程と結果を記録する。
        """
        started = time.monotonic()
        record = {"label": label, "place": place, "pose": into_pose, "stage": "validate",
                  "released": False, "success": False}
        self.last_placement_result = record
        try:
            use_target = place in getattr(self, "placement_targets", {})
            if (place not in self.coordinates["positions"] or not self.poses.get("all_neutral")
                    or (not use_target and not self.poses.get(into_pose))):
                return False
            # 姿勢変更は MoveIt で計画し、計画できなければ直接指令で動かす（drawer_pose）。
            # 11/9 06:32 の実行では、windex_bottle を持ったまま all_neutral を MoveIt で計画できず、
            # どこにも置けないまま次の物を取りに行って物を蹴っていた。
            stages = [("transport", lambda: self.drawer_pose("all_neutral")),
                      ("navigate", lambda: self.navigate_with_alternate(place)),
                      ("holding", lambda: self.check_holding(label, record))]
            if use_target:
                # 容器・トレイ: 全身 IK（move_over_container）は手が届かず失敗していた（06:32 container_a）。
                # 引き出しと同じく、決まった腕の姿勢にしてから台車で手を容器の真上に合わせて離す。
                stages.extend([
                    ("locate", lambda: self.locate_place_target(place)),
                    ("place", lambda: self.lift_then_pose(self.release_pose_for(place, label))),
                    ("align", lambda: self.align_hand_over(
                        place, self.placement_targets[place]["x"],
                        self.placement_targets[place]["y"])),
                    ("lower", lambda: self.lower_tall_object(place, label)),
                    ("holding", lambda: self.check_holding(label, record)),
                    ("release", self.release_object)])
            else:
                stages.append(("place", lambda: self.lift_then_pose(into_pose)))
                if place in getattr(self, "drawer_places", {}):
                    # 引き出し: move_base も自己位置も 5cm ほどずれるので、置く直前に開いた取っ手を見て
                    # 落とす位置を決め直し、台車を少しずつ動かして手をその真上に合わせる
                    stages.insert(-1, ("locate", lambda: self.locate_drawer_drop(place)))
                    stages.append(("align", lambda: self.align_hand_over_drawer(place)))
                stages.append(("holding", lambda: self.check_holding(label, record)))
                stages.append(("release", self.release_object))
            stages.append(("retreat", lambda: self.drawer_pose("all_neutral")))
            for stage, action in stages:
                record["stage"] = stage
                rospy.loginfo("Placement %s: %s", place, stage)
                if not action():
                    return False
                if stage == "release":
                    record["released"] = True
            record["success"] = True
            return True
        except Exception as error:  # 開放後の失敗もログに残し、自動再試行はしない。
            rospy.logerr("Placement error at %s: %s", record["stage"], error)
            return False
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            rospy.loginfo("Placement result: %s", json.dumps(record, sort_keys=True))
            self.log_drawer_poses("placing at {}".format(place))

    def pull_out_trofast(self, x, y, z, yaw, pitch, roll):
        """サンプルのまま（担当なし）
        pull_out_trofast
        """
        # trofastの引き出しを引き出す
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        a = True  # TODO 不要な変数
        gripper.command(1)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(x, y, z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw,  pitch, roll)
        gripper.command(1)
        self.change_pose("all_neutral")

    def push_in_trofast(self, pos_x, pos_y, pos_z, yaw, pitch, roll):
        """サンプルのまま（担当なし）
        trofastの引き出しを戻す
        NOTE:サンプル
            self.push_in_trofast(0.178, -0.29, 0.75, -90, 100, 0)
        """
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        pos_y += self.HAND_PALM_OFFSET

        # 予備動作-押し込む
        whole_body.move_end_effector_pose(
            pos_x, pos_y + self.TROFAST_Y_OFFSET * 1.5, pos_z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(
            pos_x, pos_y + self.TROFAST_Y_OFFSET, pos_z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(pos_x, pos_y, pos_z, yaw, pitch, roll)

        self.change_pose("all_neutral")

    # ------------------------------------------------------------------
    # 引き出し（3 つとも開ける）
    # 座標はシミュレータ（Gazebo のモデル位置・trofast_knob モデル・得点判定のパラメータ）から取った値。
    #   ロボット初期位置 world(-2.1, 1.2, yaw -90°) が map 原点なので
    #   map_x = 1.2 - world_y,  map_y = world_x + 2.1
    #   trofast_1（横の 1 段だけ）world(-2.7, 0.67, z 0.10) → 取っ手 map(0.53, 前面 -0.355, z 0.28)
    #   trofast_2（2 段の下）    world(-2.7, 1.00, z 0.10) → 取っ手 map(0.20, 前面 -0.355, z 0.28)
    #   trofast_3（2 段の上）    world(-2.7, 1.00, z 0.36) → 取っ手 map(0.20, 前面 -0.355, z 0.54)
    #   取っ手（T 字のつまみ）はモデル原点から前へ 0.24・上へ 0.18、板の裏の軸は前面から 1〜3cm 奥
    # 得点判定（*_detector の box_pose / box_size）は各引き出しの箱に付いていて、箱と一緒に動く
    #   中心 (0, 0, 0.125)・大きさ 0.35 x 0.21 x 0.25
    #   tools_in_drawertop = trofast_2, tools_in_drawerbottom = trofast_3, shapeitems_in_drawerleft = trofast_1
    # 引き具合: 上段を半分（0.20）、下段を全開（0.30）、横を 0.25。
    #   両方の判定に重ねるダブルスコア（下段 0.30・上段 0.36 で上段を傾ける）は、上段を引きすぎて
    #   抜け落ちたり斜めに引いて失敗したりしたのでやめた（11/9）。
    # 11/9 のロボットでの確認で分かったこと
    #   - 手のひら y=-0.29 では指先が板の前面に当たるだけ → 指先を板の裏の軸まで入れる
    #   - 自己位置が数 cm ずれる → 頭の深度カメラで取っ手を見つけ、その位置へ手を出す
    #     （点群も手の目標も同じ map 座標なので、自己位置のずれが打ち消される）
    #   - gripper.command(0)（MoveIt）は指が取っ手に触れていると閉じない → 力で握る grasp アクションを使う
    # ------------------------------------------------------------------
    USE_DRAWER = True                       # False にすると引き出しを使わない（今までどおり bin_b へ）
    DRAWER_STANDPOINTS = {                  # 引き出しの正面に立つ位置
        "drawer_standpoint": [0.2, 0.28, -90],
        "drawer_left_standpoint": [0.53, 0.28, -90],
    }
    # 11/10 判明した競技ルール: 3 つの引き出しを引き切っておけば、どれか 1 つに入った物は「引き出しに入れた」になる。
    # そこで 3 つとも DRAWER_FULL_PULL まで引く（~drawer_pull で変更。DRAWER_PULL_MAX より大きくはしない）。
    # 引き切りの長さ（tmc_wrs_gazebo の trofast_knob と wrc_stair_like_drawer のモデルから）:
    #   引き出しの箱は奥行き 0.424（重心は箱の中心）。棚のレールは長さ 0.42 で、箱の縁を下のレールに載せ、
    #   上のレールで押さえている。引き出しはレールに載っているだけで、止める物はない。
    #   0.21 を越えると重心がレールの先より前に出て、前が下がる（上のレールが後ろの縁を押さえるので傾くだけ）。
    #   0.42 で後ろの縁が上のレールから外れて抜け落ちる。
    #   実測: 下段 0.30 は全開で問題なし（11/9 04:26 に 3 つとも開いた回）。上段 0.36 は傾いて抜け落ちた（11/9）。
    #   → 0.30 にする。0.34 を上限とし、それより大きい値を指定されても使わない（散らかさない）。
    DRAWER_FULL_PULL = 0.30
    DRAWER_PULL_MAX = 0.34
    DRAWERS = {
        # x: 取っ手の x、z: 取っ手の高さ、pull: 引く距離、tilt: 取っ手を見るときの頭の傾き [rad]
        "bottom": {"x": 0.20, "z": 0.28, "pull": DRAWER_FULL_PULL, "tilt": -0.85,
                   "stand": "drawer_standpoint"},
        "top": {"x": 0.20, "z": 0.54, "pull": DRAWER_FULL_PULL, "tilt": -0.6, "stand": "drawer_standpoint"},
        "left": {"x": 0.53, "z": 0.28, "pull": DRAWER_FULL_PULL, "tilt": -0.85,
                 "stand": "drawer_left_standpoint"},
    }
    DRAWER_KNOB_FACE_Y = -0.355             # 取っ手の板の前面の y（3 つとも同じ）
    DRAWER_FINGER_REACH = 0.063             # 手のひら(hand_palm_link)から指先までの前後距離（実測）
    # 指先を板の前面からどれだけ奥へ入れるか。取っ手は「板（幅 6.6cm・厚さ 1cm）→ 軸（太さ 3cm・長さ 2cm）
    # → 根元の円板（直径 6.8cm）」の順に並ぶ。0.022 では指の腹が板の両端をはさみ（閉じた後の間隔 0.06 前後）、
    # 摩擦だけで引くので途中で外れた。指先を根元の円板の手前（3cm）まで入れ、指の腹で軸をはさんで
    # 板の裏に引っかける（閉じた後の間隔は軸の太さ 0.03 前後になるはず）。
    DRAWER_GRASP_DEPTH = 0.030
    DRAWER_APPROACH = 0.15                  # 差し込む前に手を止める、取っ手の手前の距離
    DRAWER_HAND_RPY = (-90, 100, 0)         # 取っ手をつかむ手の向き（シミュレータで到達を確認済み）
    DRAWER_Z_RETRY = (0.0, 0.012, -0.012)   # つかめなかったときに高さを少しずらして試す
    DRAWER_GRASP_EFFORT = -1.0              # grasp アクションで握る強さ（負で閉じる。-0.5 では途中で外れた）
    DRAWER_GRASP_EFFORT_SAFE = -0.5         # -1.0 で閉じなかったときに使う強さ（04:26 に 3 つとも開いた値）
    DRAWER_PULL_TRIES = 3                   # 途中で外れたときに、つかみ直して引く回数
    DRAWER_PULL_SPEED = 0.04                # 台車で引くときの速さ [m/s]（ゆっくり引いて外れにくくする）
    # 引いている途中で指の間隔が「つかんだ直後の 6 割」かつ 0.025 未満になったら外れた（すぐ止める）。
    # 11/9 の記録: 外れた回は 0.005〜0.016 まで閉じた。外れなかった回は 0.035 以上を保った。
    DRAWER_SLIP_GAP = 0.025
    DRAWER_SLIP_RATIO = 0.6
    # 閉じた後の指先の間隔: これ未満なら空振り、これより大きければ閉じていない（取っ手は 3〜6.6cm）
    DRAWER_FINGER_GAP_RANGE = (0.015, 0.09)
    # 物体を落とす位置（指先の map 座標 x, y）と、そのときの腕の姿勢
    # 引き出しの近くでは全身 IK（move_over_container）を使わない。
    #   11/9 のシミュレータ確認で、全身 IK は台車を前に出して開いた引き出しにぶつかり、
    #   手が目標から 20cm ずれた（自己位置も 20cm ずれた）。
    # 代わりに week1 の bin と同じ「決まった腕の姿勢 + 台車の位置合わせ」で置く。
    #   put_in_bin（腕を前に水平）の指先は、台車から前へ 0.632・左へ 0.078・高さ 0.51（MoveIt の FK で計算）
    #   put_in_drawer_high は arm_lift を 0.2 上げた同じ姿勢で、指先の高さ 0.71
    DRAWER_HAND_FORWARD = 0.632
    DRAWER_HAND_LEFT = 0.078
    DRAWER_RELEASE_POSES = {
        "put_in_drawer_high": {"arm_lift_joint": 0.2, "arm_flex_joint": -1.0, "arm_roll_joint": 0.0,
                               "wrist_flex_joint": -0.57, "wrist_roll_joint": 0.0,
                               "head_tilt_joint": 0.0, "head_pan_joint": 0.0},
    }
    # 物体を落とす位置の x と、離すときの腕の姿勢（縁の高さ: 上段 0.59、下段・横 0.33）。
    # y は引いた距離から drawer_drop_point で決める。
    DRAWER_DROP = {
        "top": (0.2, "put_in_drawer_high"),
        "bottom": (0.2, "put_in_bin"),
        "left": (0.53, "put_in_bin"),
    }
    DRAWER_SHELF_FRONT_Y = -0.39   # 棚（レールの先）の前面の y。これより奥は棚の中で、上から落とせない
    # 取っ手の板の前面から、得点判定の箱の前の端まで（判定の箱は引き出しの中心から前後 0.175。中心は板の前面の 0.245 奥）
    DRAWER_JUDGE_FRONT = 0.07
    # 同じ引き出しに入れる物は、落とす位置を 1 個ごとにずらす（同じ所に積むと崩れて外へ転がり出る）。
    # drawer_drop_point からのずれ (dx, dy) [m]。奥の列 → 手前の列の順に使い、6 個を越えたら最初に戻る。
    #   x: 箱の内幅は約 0.28（中心 ±0.14）。物の半径 5cm を引いて ±0.07 なら壁から 2cm 以上離れる。
    #   y: 0.30 引いたとき、落とせる範囲は棚の前面 -0.39 〜 判定の箱の前の端 -0.125（真ん中 -0.258）。
    #      奥の列 -0.303（棚の前面まで 8.7cm）、手前の列 -0.218（判定の箱の端まで 9.3cm）。
    DRAWER_DROP_SLOTS = ((0.0, -0.045), (-0.07, -0.045), (0.07, -0.045),
                         (0.0, 0.04), (-0.07, 0.04), (0.07, 0.04))
    DRAWER_PLACE_NAME = "drawer_top_place"      # 2 段の引き出し（開いた段のうち上の方に入れる。名前は前のまま）
    # 開ける引き出しの順番。下段を先に開ける（上段を先に開けると、下段へ腕を下ろすときに上段に引っかかる）。
    # 上段は傾いて抜けることがあったので最後にする（失敗しても下段と横は開いている）。
    # 上段を引き切ると下段は上段の下に隠れるので、2 段の方は上段に入れる。
    DRAWER_LEVELS = ("bottom", "left", "top")
    DRAWER_YAW = -90.0          # 引き出しを引くときの台車の向き [deg]（棚の正面に垂直）
    DRAWER_YAW_TOL = 1.0        # 向きを合わせる許容誤差 [deg]
    DRAWER_YAW_GAIN = 1.5       # 向きの誤差 [deg] → 回転速度 [deg/s]
    DRAWER_YAW_MAX = 15.0       # 向きを合わせる最大の回転速度 [deg/s]
    DRAWER_LEFT_PLACE_NAME = "drawer_left_place"  # 形のもの（横の引き出し）
    def lookup_xyz(self, child, frame="map"):
        """62401738 いしい
        frame（既定は map）座標での child フレームの位置 (x, y, z) を返す。取れなければ None。
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                frame, child, rospy.Time(0), rospy.Duration(1.0)).transform.translation
            return trans.x, trans.y, trans.z
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    def check_holding(self, label, record=None):
        """62401738 いしい
        まだ物を持っているかを指の間隔で確かめる。落としていたら False（record に lost を付ける）。
        つかんだ直後だけでなく、運んだ後・離す直前にも確かめる（11/9: 途中で落として何も持たずに
        置き場所まで行き、「置いた」ことになっていた）。
        薄い物（THIN_LABELS）と、指の間隔が取れないときは確かめられないので True。
        """
        if label in self.THIN_LABELS:
            return True
        gap = self.finger_gap()
        if gap is None or gap >= self.GRASP_EMPTY_GAP:
            return True
        rospy.logwarn("lost %s on the way (finger gap %.3f)", label, gap)
        if record is not None:
            record["lost"] = True
        gripper.command(1)
        return False

    RELEASE_SHAKE_ROLL = 0.35   # 離した後に手首をひねる量 [rad]（指に引っかかった物を振り落とす）

    def release_object(self):
        """62401738 いしい
        ハンドを開いて物を離す。11/10: 物が指に引っかかって離せないことがあった。
        gripper.command(1) は MoveIt で計画するので、物が指に触れていると開かないことがある。
        開いたか指の間隔で確かめ、開いていなければ力の指令で開く。その後、手首を左右にひねって振り落とす。
        """
        gripper.command(1)
        gap = self.finger_gap()
        if gap is not None and gap < self.DRAWER_OPEN_GAP:
            rospy.logwarn("gripper did not open to release (gap %.3f); open with effort", gap)
            try:
                import actionlib
                from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortGoal
                client = actionlib.SimpleActionClient(
                    "/hsrb/gripper_controller/grasp", GripperApplyEffortAction)
                if client.wait_for_server(rospy.Duration(2.0)):
                    client.send_goal(GripperApplyEffortGoal(effort=0.5))
                    client.wait_for_result(rospy.Duration(3.0))
            except Exception as error:
                rospy.logwarn("effort open failed: %s", error)
            gripper.command(1)
        rospy.sleep(0.3)
        self.shake_wrist()
        return True  # 離せなくても次へ進む（持ったまま止まらない）

    def shake_wrist(self):
        """62401738 いしい
        今の手首の向きから左右に RELEASE_SHAKE_ROLL ひねって戻す（引っかかった物を落とす）。
        """
        try:
            from sensor_msgs.msg import JointState as _JointState
            msg = rospy.wait_for_message("/hsrb/joint_states", _JointState, timeout=1.0)
            # 腕の 5 関節をそろえて送る（一部の関節だけの指令はコントローラが受け付けないことがある）
            arm = {name: msg.position[msg.name.index(name)] for name in (
                "arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                "wrist_flex_joint", "wrist_roll_joint")}
        except Exception as error:
            rospy.logwarn("cannot read the arm joints; skip shaking: %s", error)
            return
        roll = arm["wrist_roll_joint"]
        for delta in (self.RELEASE_SHAKE_ROLL, -self.RELEASE_SHAKE_ROLL, 0.0):
            self.move_joints(dict(arm, wrist_roll_joint=roll + delta), timeout=3.0)

    def lift_then_pose(self, name):
        """62401738 いしい
        先に arm_lift だけを目標の高さまで上げてから、腕を前に出して name の姿勢にする。
        一度に動かすと、腕が前に出る途中で持っている物の下の端が容器の縁に当たっていた（11/9 青いボトル）。
        """
        joints = self.poses.get(name, {})
        if "arm_lift_joint" in joints:
            self.move_joints({"arm_lift_joint": joints["arm_lift_joint"]})
        return self.drawer_pose(name)

    def finger_gap(self):
        """62401738 いしい
        左右の指先の間隔 [m]。閉じても間隔が残っていれば何かをつかんでいる。取れなければ None。
        """
        left = self.lookup_xyz("hand_l_finger_tip_frame")
        right = self.lookup_xyz("hand_r_finger_tip_frame")
        if left is None or right is None:
            return None
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def base_yaw(self, frame="map"):
        """62401738 いしい
        frame（既定は map）座標での台車の向き [rad]。取れなければ None。
        """
        try:
            rot = self.tf_buffer.lookup_transform(
                frame, "base_footprint", rospy.Time(0), rospy.Duration(1.0)).transform.rotation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        return math.atan2(2 * (rot.w * rot.z + rot.x * rot.y),
                          1 - 2 * (rot.y * rot.y + rot.z * rot.z))

    def map_to_odom(self, x, y):
        """62401738 いしい
        map 座標の点 (x, y) を、今の自己位置推定で odom 座標に変換する。取れなければ None。
        odom は台車の車輪の回転から計算するので、AMCL の自己位置のように急に飛ばない。
        11/9 の実行では、引き出しや長机のそば（レーザーに見える物が少ない）で自己位置が 16cm ずれ、
        しかも位置合わせの途中で推定が変わって、置く位置がずれた。
        置く直前に見つけた目標を odom に直し、odom の中で手を合わせる。
        """
        try:
            t = self.tf_buffer.lookup_transform(
                "odom", "map", rospy.Time(0), rospy.Duration(1.0)).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        q = t.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return (t.translation.x + math.cos(yaw) * x - math.sin(yaw) * y,
                t.translation.y + math.sin(yaw) * x + math.cos(yaw) * y)

    # 棚に向かう台車の動きの安全柵（map の y）。手のひらがこれより棚側（-y）へ行く指令は出さない。
    # 棚の前面 -0.39、閉じた取っ手の前面 -0.355、つかむときの手のひら y は約 -0.32。押し付けて車体が倒れた対策
    DRAWER_PALM_Y_MIN = -0.34
    DRAWER_DROP_Y_MIN = -0.36    # 落とす指先の y の下限（棚の前面 + 3cm）。これより奥へは寄らない

    def move_hand_by_base(self, target_y, keep_x, timeout, watch_grip=False, x_tol=None,
                          frame="map", hold_yaw=None, line=None, shelf_guard=False):
        """62401738 いしい
        腕は止めたまま台車だけを動かし、手のひらを map の y = target_y まで真っすぐ運ぶ。
        全身 IK で引くと台車が回ってしまうことがあるので、引き出しは台車でまっすぐ引く。
        台車の向きが -90° でなくても、map の +y 方向へ動くように速度を変換する。
        watch_grip=True のときは指の間隔を見張り、取っ手から外れたらその場で止まる
        （外れたまま下がり続けない。self.drawer_slipped が True になる）。
        x_tol を指定したとき（物を置く位置合わせ）は、x も keep_x から x_tol 以内に入るまで動く。
        指定しないとき（引き出しを引く）は y が着いたら止まる（11/9: 置くときに x が 6cm ずれたまま離していた）。
        hold_yaw [rad] を指定したとき（引き出しを引く）は、台車の向きをその値に保つ（斜めに引かない）。
        line = (始点, 向き, 長さ) を指定したとき（引き出しを引く）は、target_y / keep_x の代わりに
        始点から「向き」へ「長さ」だけ、横にずれないようにまっすぐ動く。
        """
        self.drawer_slipped = False
        self.base_stalled = False
        rate = rospy.Rate(20)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        window = []   # 直近の (時刻, 手の位置, 指令した速さ)。進んでいないかを見る
        try:
            while not rospy.is_shutdown() and rospy.Time.now() < deadline:
                hand = self.lookup_xyz("hand_palm_link", frame)
                yaw = self.base_yaw(frame)
                if hand is None or yaw is None:
                    return False
                if line is None:
                    remain = target_y - hand[1]
                    remain_x = keep_x - hand[0]
                else:
                    (o_x, o_y), (d_x, d_y), length = line
                    along = (hand[0] - o_x) * d_x + (hand[1] - o_y) * d_y
                    across = -(hand[0] - o_x) * d_y + (hand[1] - o_y) * d_x
                    remain = length - along      # 向きに沿って残っている距離
                    remain_x = -across           # 直線から横にずれた分を戻す
                if abs(remain) < 0.005 and (x_tol is None or abs(remain_x) < x_tol):
                    return True
                if shelf_guard:
                    palm_map = self.lookup_xyz("hand_palm_link", "map")
                    if palm_map is not None and palm_map[1] < self.DRAWER_PALM_Y_MIN:
                        # 棚に近づきすぎ。押し続けると倒れるので、ここで止める（後ろへ戻る指令は下で通す）
                        goes_in = (remain < 0) if frame == "map" else None
                        if goes_in is None:
                            goes_in = True
                        if goes_in:
                            rospy.logwarn("hand is too close to the shelf (map y %.3f < %.2f); stop the base",
                                          palm_map[1], self.DRAWER_PALM_Y_MIN)
                            self.base_stalled = True
                            return False
                if self.is_base_stalled(window, hand):
                    # 手が何かに当たって進まない（取っ手に押し付けている・引き出しが止まった）。
                    # 押し続けると腕を伸ばしたまま車体が後ろに傾く（11/9 07:59 上段）ので、ここで止める
                    rospy.logwarn("hand is not moving (%.3f m left); stop the base", remain)
                    self.base_stalled = True
                    return False
                if watch_grip:
                    gap = self.finger_gap()
                    if gap is not None and gap < self.slip_threshold():
                        rospy.logwarn("slipped off the drawer handle (gap %.3f, %.3f m left)",
                                      gap, remain)
                        self.drawer_slipped = True
                        return False
                speed = self.DRAWER_PULL_SPEED
                vy_map = max(-speed, min(speed, 2.0 * remain))
                # 引き出しを引くとき（x_tol なし）は横の補正をごく小さくして、まっすぐ引く
                vx_max = 0.01 if x_tol is None else speed
                vx_map = max(-vx_max, min(vx_max, 2.0 * remain_x))
                if line is not None:
                    # 直線の座標（沿う向き・横）の速度 → frame 座標の速度
                    (_, _), (d_x, d_y), _ = line
                    vx_map, vy_map = (vy_map * d_x - vx_map * d_y, vy_map * d_y + vx_map * d_x)
                # map 座標の速度 → 台車座標の速度
                vel_x = math.cos(yaw) * vx_map + math.sin(yaw) * vy_map
                vel_y = -math.sin(yaw) * vx_map + math.cos(yaw) * vy_map
                yaw_rate = 0.0
                if hold_yaw is not None:
                    err = math.degrees(math.atan2(math.sin(hold_yaw - yaw), math.cos(hold_yaw - yaw)))
                    yaw_rate = max(-5.0, min(5.0, self.DRAWER_YAW_GAIN * err))
                omni_base.cmd_vel(vel_x, vel_y, yaw_rate)
                window.append((rospy.Time.now().to_sec(), hand, math.hypot(vx_map, vy_map)))
                rate.sleep()
            return False
        finally:
            for _ in range(3):
                omni_base.cmd_vel(0, 0, 0)
                rospy.sleep(0.05)

    # 引いた後に手を抜くとき下がる距離。指先は板の前面から DRAWER_GRASP_DEPTH 奥にあるので、
    # それより 4cm 多く下がれば板から抜ける（以前は 0.12。台車の後ろの床の物に近づきすぎていた）
    DRAWER_RETREAT = 0.07
    DRAWER_OPEN_GAP = 0.08   # 開いたとみなす指先の間隔（取っ手の板の幅 6.6cm より広い）

    def open_gripper_checked(self):
        """62401738 いしい
        ハンドを開き、指先の間隔が取っ手の板より広くなったか確かめる。開かなければもう一度開く。
        指の間隔が取れないとき（tf がない）は、開いたものとして True を返す。
        """
        for _ in range(2):
            gripper.command(1)
            gap = self.finger_gap()
            if gap is None or gap >= self.DRAWER_OPEN_GAP:
                return True
            rospy.logwarn("gripper still closed after open command (gap %.3f)", gap)
        return False

    STALL_WINDOW = 1.5     # 進んでいるかを見る時間 [s]
    STALL_MIN_CMD = 0.02   # この時間に指令した距離がこれ以上のときだけ判定する [m]
    STALL_RATIO = 0.3      # 実際に動いた距離が指令の 3 割未満なら、当たって進んでいない

    def is_base_stalled(self, window, hand):
        """62401738 いしい
        window（直近の 時刻・手の位置・指令した速さ）から、台車に指令しているのに手が進んでいないかを判定する。
        """
        now = rospy.Time.now().to_sec()
        while window and now - window[0][0] > self.STALL_WINDOW:
            window.pop(0)
        if len(window) < 2 or now - window[0][0] < self.STALL_WINDOW * 0.8:
            return False
        commanded = sum(speed * (b[0] - a[0]) for a, b, speed in
                        ((window[i], window[i + 1], window[i][2]) for i in range(len(window) - 1)))
        moved = math.hypot(hand[0] - window[0][1][0], hand[1] - window[0][1][1])
        return commanded > self.STALL_MIN_CMD and moved < commanded * self.STALL_RATIO

    def slip_threshold(self):
        """62401738 いしい
        指の間隔がこれより小さくなったら取っ手から外れたとみなす値（つかんだ直後の間隔から決める）。
        """
        grip = getattr(self, "drawer_grip_gap", None) or self.DRAWER_SLIP_GAP
        return min(self.DRAWER_SLIP_GAP, grip * self.DRAWER_SLIP_RATIO)

    def close_gripper_on_handle(self):
        """62401738 いしい
        力で握る grasp アクションでハンドを閉じる。
        gripper.command(0) は MoveIt で計画するので、指が取っ手に触れていると「衝突」とみなされて閉じない。
        アクションが使えなければ gripper.command(0) で閉じる。
        """
        import actionlib
        from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortGoal

        client = actionlib.SimpleActionClient(
            "/hsrb/gripper_controller/grasp", GripperApplyEffortAction)
        if not client.wait_for_server(rospy.Duration(3.0)):
            rospy.logwarn("grasp action is not available; close with MoveIt")
            return gripper.command(0)
        client.send_goal(GripperApplyEffortGoal(effort=self.DRAWER_GRASP_EFFORT))
        client.wait_for_result(rospy.Duration(5.0))
        rospy.sleep(0.5)
        gap = self.finger_gap()
        if gap is not None and gap > self.DRAWER_FINGER_GAP_RANGE[1]:
            # 強く握る指令が受け付けられず開いたままなら、04:26 と同じ強さでもう一度閉じる
            rospy.logwarn("gripper did not close with effort %.1f; retry with %.1f",
                          self.DRAWER_GRASP_EFFORT, self.DRAWER_GRASP_EFFORT_SAFE)
            client.send_goal(GripperApplyEffortGoal(effort=self.DRAWER_GRASP_EFFORT_SAFE))
            client.wait_for_result(rospy.Duration(5.0))
            rospy.sleep(0.5)
        return True

    def look_at_drawer(self, level, fold_arm=True):
        """62401738 いしい
        腕を下ろして（カメラの前からどけて）、頭を level の取っ手の方へ向ける。
        物を持っているとき（fold_arm=False）は腕はそのまま（all_neutral で上に上げてある）。
        """
        if fold_arm:
            whole_body.move_to_go()
        head = whole_body.HEAD_CMD
        head.set_joint_value_target(
            {"head_pan_joint": 0.0, "head_tilt_joint": self.DRAWERS[level]["tilt"]})
        try:
            ok = bool(head.go(wait=True))
        finally:
            head.stop()
        rospy.sleep(0.5)
        return ok

    def detect_drawer_knob(self, level, face_y, z_tol=0.04, y_tol=0.10):
        """62401738 いしい
        頭の深度カメラの点群から level の取っ手を探し、(x, 板の前面の y, z) を返す。
        予想位置のまわりで、引き出しの前面より 1.5〜7cm 手前に飛び出した点のかたまりを取っ手とみなす。
        見つからなければ None。
        """
        import numpy as np

        x0, z0 = self.DRAWERS[level]["x"], self.DRAWERS[level]["z"]
        found = []
        stamp = rospy.Time.now()
        for _ in range(2):
            try:
                points, stamp = self.read_placement_cloud(stamp)
            except Exception as error:  # 点群が来なければ決め打ち座標を使う
                rospy.logwarn("no depth cloud for the drawer: %s", error)
                return None
            box = points[(np.abs(points[:, 0] - x0) < 0.10)
                         & (np.abs(points[:, 2] - z0) < z_tol)
                         & (np.abs(points[:, 1] - face_y) < y_tol + 0.05)]
            if len(box) < 30:
                continue
            # 一番点が多い y（1cm 刻み）を引き出しの前面とみなし、そこから手前に飛び出した点を取る
            bins = np.round(box[:, 1] / 0.01).astype(int)
            values, counts = np.unique(bins, return_counts=True)
            panel_y = values[np.argmax(counts)] * 0.01
            knob = box[(box[:, 1] > panel_y + 0.015) & (box[:, 1] < panel_y + 0.07)]
            if len(knob) < 10:
                continue
            found.append((float(np.median(knob[:, 0])), float(np.percentile(knob[:, 1], 90)),
                          float(np.median(knob[:, 2]))))
        if not found:
            return None
        x, y, z = [float(np.mean(v)) for v in zip(*found)]
        rospy.loginfo("%s drawer knob seen at (%.3f, %.3f, %.3f)", level, x, y, z)
        if abs(x - x0) > 0.08 or abs(z - z0) > z_tol or abs(y - face_y) > y_tol:
            rospy.logwarn("knob position is too far from the expected one; ignore it")
            return None
        return x, y, z

    # 上段の取っ手を、全身 IK を使わずにつかむための腕の姿勢（MoveIt の FK で計算）。
    #   arm_lift 0.1・arm_flex -1.20・wrist_flex -0.37（手先が水平に前を向く）で、
    #   手のひらは台車から前へ 0.603・左へ 0.075・高さ 0.548（指は左右に開く）。
    #   arm_lift を 1 増やすと手のひらの高さも 1 上がる。
    # 下段を全開にすると台車が近づけず、全身 IK では上段に手が届かなかった（05:43）。
    # この姿勢なら台車の前端は棚から 0.06 で、全開の下段（前端 -0.055）に当たらない。
    DRAWER_POSE_GRASP = {"arm_flex_joint": -1.20, "arm_roll_joint": 0.0,
                         "wrist_flex_joint": -0.37, "wrist_roll_joint": 0.0}
    DRAWER_POSE_LIFT0 = 0.1      # このときの arm_lift
    DRAWER_POSE_PALM_Z = 0.548   # そのときの手のひらの高さ
    DRAWER_POSE_FORWARD = 0.603  # 手のひらの前への距離
    DRAWER_POSE_LEFT = 0.075     # 手のひらの左への距離

    def grasp_drawer_handle_by_pose(self, level, knob):
        """62401738 いしい
        腕を決まった姿勢（手先が水平に前）にし、台車を前に進めて手を取っ手に差し込んでつかむ。
        全身 IK を使わないので、下段が開いていても上段に手が届く。つかめたら (手の x, 手の y, 手の z)。
        """
        gap_min, gap_max = self.DRAWER_FINGER_GAP_RANGE
        pos_x, face_y, knob_z = knob
        grasp_y = face_y - self.DRAWER_GRASP_DEPTH + self.DRAWER_FINGER_REACH
        for dz in self.DRAWER_Z_RETRY:
            lift = self.DRAWER_POSE_LIFT0 + (knob_z + dz - self.DRAWER_POSE_PALM_Z)
            lift = max(0.0, min(0.6, lift))
            rospy.loginfo("grasp %s drawer handle by pose (hand x=%.3f y=%.3f, lift %.3f)",
                          level, pos_x, grasp_y, lift)
            gripper.command(1)
            # 取っ手の手前 DRAWER_APPROACH の所に手が来る位置まで、腕をたたんだまま台車で下がる/進む
            self.drawer_pose("all_neutral")
            self.coordinates["positions"]["drawer_pose_approach"] = [
                pos_x - self.DRAWER_POSE_LEFT,
                grasp_y + self.DRAWER_APPROACH + self.DRAWER_POSE_FORWARD, -90.0]
            if not self.navigate_for_placement("drawer_pose_approach"):
                # 腕の姿勢が固定なので、立ち位置がずれたまま手を出すと棚にぶつかる
                rospy.logwarn("cannot reach the approach point for the %s drawer; do not grasp", level)
                return None
            self.square_to_drawer()
            joints = dict(self.DRAWER_POSE_GRASP, arm_lift_joint=lift)
            if not self.move_joints(joints):
                rospy.logwarn("cannot take the drawer grasp pose")
                continue
            # 台車を前に進めて、手を取っ手に差し込む
            if not self.move_hand_by_base(grasp_y, pos_x, timeout=10, shelf_guard=True):
                # 指が取っ手に触れて止まった場合もあるので、閉じて指の間隔で判定する
                rospy.logwarn("base insertion did not finish for the %s drawer; close and check", level)
            self.close_gripper_on_handle()
            gap = self.finger_gap()
            rospy.loginfo("finger gap after closing: %s", gap)
            hand = self.lookup_xyz("hand_palm_link")
            if gap is not None and gap_min <= gap <= gap_max and hand is not None:
                self.drawer_grip_gap = gap
                return pos_x, grasp_y, hand[2]
            rospy.logwarn("did not grasp the %s drawer handle (gap %s); retry", level, gap)
            gripper.command(1)
            self.move_hand_by_base(grasp_y + self.DRAWER_APPROACH, pos_x, timeout=8)
        return None

    def grasp_drawer_handle(self, level, knob):
        """62401738 いしい
        knob（x, 板の前面の y, z）の取っ手をつかむ。つかめたら (手の x, 手の y, 手の z) を返す。
        つかめなければ高さを少しずつずらして試し、だめなら None。
        """
        yaw, pitch, roll = self.DRAWER_HAND_RPY
        gap_min, gap_max = self.DRAWER_FINGER_GAP_RANGE
        pos_x, face_y, knob_z = knob
        grasp_y = face_y - self.DRAWER_GRASP_DEPTH + self.DRAWER_FINGER_REACH
        self.drawer_pose("grasp_on_table")
        self.square_to_drawer()
        for dz in self.DRAWER_Z_RETRY:
            pos_z = knob_z + dz
            rospy.loginfo("grasp %s drawer handle (hand x=%.3f y=%.3f z=%.3f)",
                          level, pos_x, grasp_y, pos_z)
            gripper.command(1)
            if not whole_body.move_end_effector_pose(
                    pos_x, grasp_y + self.DRAWER_APPROACH, pos_z, yaw, pitch, roll):
                rospy.logwarn("cannot reach the front of the %s drawer handle", level)
                continue
            if not whole_body.move_end_effector_pose(pos_x, grasp_y, pos_z, yaw, pitch, roll):
                rospy.logwarn("cannot insert the hand to the %s drawer handle", level)
                continue
            self.close_gripper_on_handle()
            gap = self.finger_gap()
            rospy.loginfo("finger gap after closing: %s", gap)
            if gap is not None and gap_min <= gap <= gap_max:
                self.drawer_grip_gap = gap
                return pos_x, grasp_y, pos_z
            rospy.logwarn("did not grasp the %s drawer handle (gap %s); retry", level, gap)
            gripper.command(1)
            whole_body.move_end_effector_pose(
                pos_x, grasp_y + self.DRAWER_APPROACH, pos_z, yaw, pitch, roll)
        return None

    def pull_drawer(self, level, pull=None, total=None):
        """62401738 いしい
        level（"bottom" / "top" / "left"）の引き出しを pull [m] だけ引き出す。
        取っ手を深度カメラで見つける → つかむ → 台車でまっすぐ下がって引く → 離す → 手を抜く
        → もう一度取っ手を見て、どこまで開いたか確かめる。
        11/9 の実行では、つかみ方しだいで引いている途中に指が外れることがあった
        （成功した回は引き終わりまで指の間隔が 0.06 前後のまま、失敗した回は 0.01 まで閉じていた）。
        そこで、足りなければ取っ手の今の位置からもう一度つかんで、残りを引く（DRAWER_PULL_TRIES 回まで）。
        """
        if pull is None:
            pull = self.DRAWERS[level]["pull"]
        self.drawer_face0 = getattr(self, "drawer_face0", {})
        self.drawer_opened = getattr(self, "drawer_opened", {})
        second = total is not None and level in self.drawer_face0
        # 2 回目は、取っ手は 1 回目に開けた分だけ手前にある
        expected = (self.drawer_face0[level] + self.drawer_opened.get(level, 0.0) if second
                    else self.DRAWER_KNOB_FACE_Y)
        self.look_at_drawer(level)
        knob = (self.detect_drawer_knob(level, expected, z_tol=0.10, y_tol=0.15) if second
                else self.detect_drawer_knob(level, expected))
        knob_seen = knob is not None
        if knob is None:
            rospy.logwarn("%s drawer knob not seen; use the expected position", level)
            knob = (self.DRAWERS[level]["x"], expected, self.DRAWERS[level]["z"])
        face0 = knob[1]                 # 閉じているときの取っ手の位置
        if second:
            # 2 回目: 閉じていたときの位置から合計 total 引いた所までにする
            face0 = self.drawer_face0[level]
            pull = total
        else:
            self.drawer_face0[level] = face0
            # 実際に見えた位置か（ログ・調査用）
            self.drawer_face0_seen = getattr(self, "drawer_face0_seen", {})
            self.drawer_face0_seen[level] = knob_seen
        goal = face0 + pull             # 引き終わりの取っ手の位置
        # 手を引く先は「最初の取っ手の位置 + pull」で固定する（つかみ直しても、これより先へは引かない）。
        # 上段は傾くと取っ手が下がって見え、開いた量を少なく測る。測った量で引き足すと
        # 引き出しが抜けて落ちた（11/9 06:00）ので、引く先は測った量に頼らない。
        hand_goal_y = face0 - self.DRAWER_GRASP_DEPTH + self.DRAWER_FINGER_REACH + pull
        tries = self.DRAWER_PULL_TRIES   # 上段も半分だけなので傾かない。外れたらつかみ直す
        opened = 0.0
        for attempt in range(tries):
            remain = goal - knob[1]
            rospy.loginfo("pull %s drawer: try %d, %.3f m left", level, attempt + 1, remain)
            if level == "top":
                # 上段は下段を開けた後なので、全身 IK ではなく決まった姿勢 + 台車でつかむ
                hand = self.grasp_drawer_handle_by_pose(level, knob)
            else:
                hand = self.grasp_drawer_handle(level, knob)
            if hand is None:
                break
            pos_x, grasp_y, _ = hand
            remain = hand_goal_y - grasp_y   # 実際に引く量（決めた所より先へは引かない）
            if remain < 0.03:
                gripper.command(1)
                break
            # 引く先・手を抜く先を、つかんだ時点で odom 座標に直して固定する。
            # map（AMCL）のまま引くと、引き出しのそばで自己位置が飛んだ分（最大 16cm）だけ
            # 余計に引いてしまい、引きすぎ・引きずりになっていた。odom は車輪の回転なので飛ばない。
            frame = "map"
            pull_goal = (pos_x, hand_goal_y)
            back_goal = (pos_x, hand_goal_y + self.DRAWER_RETREAT)
            odom_pull = self.map_to_odom(*pull_goal)
            odom_back = self.map_to_odom(*back_goal)
            if odom_pull is not None and odom_back is not None:
                frame, pull_goal, back_goal = "odom", odom_pull, odom_back
            # 棚の正面に垂直（map の +y）に、つかんだ所から remain だけまっすぐ引く。
            # 以前は取っ手の予想 x へ寄せながら（横に最大 0.03m/s）引いていたので、手の x がずれていると
            # 斜めに引いて引っかかっていた（11/9）。odom で引くときは、map の +y を odom の向きに直す。
            line = self.drawer_pull_line(frame, pos_x, grasp_y, remain)
            hold_yaw = self.base_yaw(frame)   # 引いている間、つかんだ時の向きを保つ
            if (not self.move_hand_by_base(
                    pull_goal[1], pull_goal[0], timeout=remain / self.DRAWER_PULL_SPEED * 3 + 5,
                    watch_grip=True, frame=frame, hold_yaw=hold_yaw, line=line)
                    and not self.drawer_slipped):
                # 以前はここで全身 IK で引き直していたが、IK は台車を回したり余計に動かしたりして
                # 引き出しを引きずる・引きすぎる原因になるので、引けた所で止める
                rospy.logwarn("base pull did not finish; stop pulling here")
            slipped = self.drawer_slipped   # 手を抜く移動で上書きされるので、先に覚えておく
            gap_end = self.finger_gap()
            if not self.open_gripper_checked():
                # 指が開かないまま下がると引き出しごと引きずるので、下がらずに諦める
                rospy.logerr("gripper did not open on the %s drawer handle; stop", level)
                break
            # 手を抜く（指先が取っ手の板の手前に出るだけ下がる。下がりすぎると後ろの床の物を蹴る）
            self.move_hand_by_base(back_goal[1], back_goal[0], timeout=8, frame=frame)
            rospy.loginfo("%s drawer: finger gap at end of pull %s", level, gap_end)
            # 腕を上げてから（開いた引き出しの前で腕を下ろさない）、取っ手がどこまで来たかをカメラで確かめる
            self.drawer_pose("all_neutral")
            self.look_at_drawer(level)
            # 途中で外れたときは、引き出しはほとんど動いていないので元の位置のまわりを探す
            # （11/9 08:26 下段: 引けた先 -0.065 ± 0.25 を探し、元の -0.33 にある取っ手を見落とした）
            expect_y = knob[1] if slipped else knob[1] + remain
            seen = self.detect_drawer_knob(level, expect_y, z_tol=0.10, y_tol=0.25)
            if seen is not None:
                knob = seen
                opened = seen[1] - face0
            elif gap_end is not None and gap_end >= self.slip_threshold():
                # 取っ手は見えなかったが、最後までつかんでいた → 引けたとみなす
                opened = pull
            rospy.loginfo("%s drawer opened %.3f m (goal %.2f)", level, opened, pull)
            if opened >= pull * 0.8:
                break
            if seen is None and not slipped:
                break  # 取っ手の位置が分からないので、もう一度つかみに行けない
            # 外れて取っ手も見えないときは、最後に分かっている位置（knob のまま）でつかみ直す
            self.go_to_drawer(level)  # 立ち位置に戻ってつかみ直す
        if opened >= pull * 0.5:
            self.drawer_opened = getattr(self, "drawer_opened", {})
            self.drawer_opened[level] = opened
            return True
        rospy.logwarn("%s drawer did not open (%.3f m)", level, opened)
        return False

    def drawer_pose(self, name):
        """62401738 いしい
        引き出しのそばでの姿勢変更。開いた引き出しに腕をぶつけないよう MoveIt で計画して動かし、
        計画できなければ直接指令で動かす（引き出しがうまく開いた 04:26 の実行と同じ動かし方）。
        Task 1 の他の場所では week1 と同じ直接指令（change_pose）を使う。
        """
        if self.move_placement_pose(name):
            return True
        return self.change_pose(name)

    def go_to_drawer(self, level):
        """62401738 いしい
        腕をたたんで level の引き出しの正面へ移動する。戻れなくても全身 IK が台車を動かして手を伸ばす。
        """
        self.drawer_pose("all_neutral")
        stand = self.DRAWERS[level]["stand"]
        if not self.navigate_for_placement(stand):
            rospy.logwarn("cannot reach %s; try the %s drawer from here", stand, level)
            return False
        self.square_to_drawer()
        return True

    def drawer_pull_line(self, frame, x, y, length):
        """62401738 いしい
        引き出しをまっすぐ引く直線 (始点, 向き, 長さ) を frame 座標で返す。
        始点は今の手のひら、向きは map の +y（棚の正面に垂直）。取れなければ None（従来どおり y だけ見る）。
        """
        palm = self.lookup_xyz("hand_palm_link", frame)
        if palm is None:
            return None
        if frame == "map":
            return (palm[0], palm[1]), (0.0, 1.0), length
        p_0, p_1 = self.map_to_odom(x, y), self.map_to_odom(x, y + 1.0)
        if p_0 is None or p_1 is None:
            return None
        norm = math.hypot(p_1[0] - p_0[0], p_1[1] - p_0[1])
        return (palm[0], palm[1]), ((p_1[0] - p_0[0]) / norm, (p_1[1] - p_0[1]) / norm), length

    def square_to_drawer(self, timeout=6.0):
        """62401738 いしい
        台車をその場で回して、棚の正面に垂直（map で DRAWER_YAW）に向ける。
        move_base で着いた向きは数度ずれていて、そのまま引くと斜めに引いて引っかかっていた（11/9）。
        """
        end_time = rospy.Time.now() + rospy.Duration(timeout)
        try:
            while not rospy.is_shutdown() and rospy.Time.now() < end_time:
                yaw = self.base_yaw()
                if yaw is None:
                    return False
                err = math.degrees(math.atan2(math.sin(math.radians(self.DRAWER_YAW) - yaw),
                                              math.cos(math.radians(self.DRAWER_YAW) - yaw)))
                if abs(err) < self.DRAWER_YAW_TOL:
                    return True
                rate = max(-self.DRAWER_YAW_MAX, min(self.DRAWER_YAW_MAX, self.DRAWER_YAW_GAIN * err))
                if abs(rate) < 2.0:
                    rate = math.copysign(2.0, rate)   # 遅すぎると台車が動かない
                omni_base.cmd_vel(0, 0, rate)
                rospy.sleep(0.05)
            rospy.logwarn("could not square the base to the drawer")
            return False
        finally:
            omni_base.cmd_vel(0, 0, 0)

    def drawer_full_pull(self):
        """62401738 いしい
        引き切る距離 [m]（~drawer_pull、既定 DRAWER_FULL_PULL）。DRAWER_PULL_MAX を越える値は使わない（抜け落ちて散らかる）。
        """
        pull = float(rospy.get_param("~drawer_pull", self.DRAWER_FULL_PULL))
        if pull > self.DRAWER_PULL_MAX:
            rospy.logwarn("drawer pull %.3f m is too far; use %.3f m", pull, self.DRAWER_PULL_MAX)
            pull = self.DRAWER_PULL_MAX
        return pull

    def drawer_drop_point(self, level, pull):
        """62401738 いしい
        level の引き出しを pull [m] 引いたときに、物体を落とす位置 (x, y, 腕の姿勢) を返す。
        y は「棚の前面」と「得点判定の箱の前の端」の真ん中（上から落とせて、判定の箱に入る所）。
          pull 0.30: 判定の箱の前の端 -0.355 + 0.30 - 0.07 = -0.125 → 真ん中 -0.258
        """
        x, pose = self.DRAWER_DROP[level]
        judge_front = self.DRAWER_KNOB_FACE_Y + pull - self.DRAWER_JUDGE_FRONT
        return x, (self.DRAWER_SHELF_FRONT_Y + judge_front) / 2.0, pose

    def open_drawers(self):
        """62401738 いしい
        DRAWER_LEVELS の順（下段 → 横 → 上段）に、3 つとも引き切る（drawer_full_pull）。
        開いた引き出しに物体を落とす位置を登録する（2 段の方: DRAWER_PLACE_NAME、横: DRAWER_LEFT_PLACE_NAME）。
        """
        if getattr(self, "drawers_attempted", False):
            return getattr(self, "drawers_open", False)
        self.drawers_attempted = True
        self.drawers_open = False
        if not rospy.get_param("~use_drawer", self.USE_DRAWER):
            return False
        rospy.loginfo("#### open drawers ####")
        gripper.command(1)  # 手を空にしてから始める（物を持ったまま引かない）
        for name, pos in self.DRAWER_STANDPOINTS.items():
            self.coordinates["positions"][name] = list(pos)
        opened = {}
        pull = self.drawer_full_pull()
        try:
            # 下段が出ていると全身 IK では上段に届かないので、上段は決まった姿勢 + 台車でつかむ
            # （台車の前端は全開の下段に当たらない位置。grasp_drawer_handle_by_pose）。
            end = float(rospy.get_param("~drawer_end_sec", self.DRAWER_END_SEC))
            for level in self.DRAWER_LEVELS:
                if self.elapsed_since_start() > end:
                    # 引き出しに時間を使いすぎると Task 1・2 の時間がなくなるので、残りは開けない
                    rospy.logwarn("drawer time is up; skip the %s drawer", level)
                    continue
                try:
                    self.go_to_drawer(level)
                    opened[level] = self.pull_drawer(level, pull=pull)
                    self.log_drawer_poses("pulling the {} drawer".format(level))
                except Exception as error:
                    # 1 つで失敗しても、残りの引き出しは試す
                    opened[level] = False
                    rospy.logerr("could not open the %s drawer: %s", level, error)
                    gripper.command(1)
                    self.drawer_pose("all_neutral")
            # 3 つ引いた後、引き切れていない（失敗・途中まで）引き出しを、時間があれば引き足す。
            # 3 つとも引き切らないと点にならない。取っ手の今の位置から合計 pull になるまで引く（total）。
            retry_end = float(rospy.get_param("~drawer_retry_end_sec", self.DRAWER_RETRY_END_SEC))
            for level in self.DRAWER_LEVELS:
                done = getattr(self, "drawer_opened", {}).get(level, 0.0)
                if done >= pull * self.DRAWER_DONE_RATIO or self.elapsed_since_start() > retry_end:
                    continue
                rospy.logwarn("%s drawer opened only %.3f of %.2f m; pull it again", level, done, pull)
                try:
                    self.go_to_drawer(level)
                    if self.pull_drawer(level, pull=pull, total=pull):
                        opened[level] = True
                except Exception as error:
                    rospy.logerr("could not pull the %s drawer again: %s", level, error)
                    gripper.command(1)
                    self.drawer_pose("all_neutral")
        except Exception as error:  # 引き出しで失敗しても Task 1 は続ける
            rospy.logerr("drawer error: %s", error)
        finally:
            gripper.command(1)
            self.drawer_pose("all_neutral")
        rospy.loginfo("drawer result: %s  opened [m]: %s", json.dumps(opened, sort_keys=True),
                      json.dumps({k: round(v, 3) for k, v in getattr(self, "drawer_opened", {}).items()}))

        # 2 段の引き出しの置き場所: 上段が開いていれば上段（下段は上段の下に隠れている）、なければ下段
        level = "top" if opened.get("top") else ("bottom" if opened.get("bottom") else None)
        if level is not None:
            self.register_drawer_place(self.DRAWER_PLACE_NAME, self.drawer_drop_point(level, pull), level)
        # 横の引き出しの置き場所
        if opened.get("left"):
            self.register_drawer_place(self.DRAWER_LEFT_PLACE_NAME, self.drawer_drop_point("left", pull),
                                       "left")
        self.drawers_open = any(opened.values())
        if not self.drawers_open:
            rospy.logwarn("could not open the drawers. use bin_b instead.")
        return self.drawers_open

    DRAWER_MODELS = {"trofast_1": "left", "trofast_2": "bottom", "trofast_3": "top"}
    DRAWER_CLOSED_X = -2.7   # 閉じているときの引き出しの world x（前へ引くと増える）

    def log_drawer_poses(self, when):
        """62401738 いしい
        デバッグ用: Gazebo の引き出しの位置（どれだけ出ているか）をログに出すだけ。動作の判断には使わない。
        11/9 08:26〜08:32 の実行で下段が棚から抜けて床に落ちていたが、どの動作で抜けたかがログから分からなかった。
        ~log_drawer_poses を False にすると出さない。
        """
        if not rospy.get_param("~log_drawer_poses", True):
            return
        from gazebo_msgs.msg import ModelStates
        try:
            states = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=2.0)
        except rospy.ROSException:
            return
        parts = []
        for name, level in sorted(self.DRAWER_MODELS.items()):
            if name in states.name:
                pos = states.pose[states.name.index(name)].position
                parts.append("{} {:+.3f} (y {:.2f}, z {:.2f})".format(
                    level, pos.x - self.DRAWER_CLOSED_X, pos.y, pos.z))
        rospy.loginfo("[debug] drawers after %s: %s", when, ", ".join(parts))

    def register_drawer_place(self, place, drop, level):
        """62401738 いしい
        指先を drop（x, y）に持っていける台車の位置（引き出しの方を向く）と、離すときの腕の姿勢を登録する。
        put_in_place は bin と同じ「移動 → 姿勢 → 離す」で置く（placement_targets には入れない）。
        """
        x, y, pose = drop
        # ロボットは -y 向き（yaw -90°）: 前は map の -y、左は map の +x
        self.coordinates["positions"][place] = [
            x - self.DRAWER_HAND_LEFT, y + self.DRAWER_HAND_FORWARD, -90]
        self.poses.update(self.DRAWER_RELEASE_POSES)
        self.placement_targets.pop(place, None)
        if not hasattr(self, "drawer_places"):
            self.drawer_places = {}
        self.drawer_places[place] = pose
        if not hasattr(self, "drawer_drops"):
            self.drawer_drops = {}
        self.drawer_drops[place] = (x, y)
        # ずらす前の落とす位置（取っ手を見て決め直したら、その位置に更新する）
        if not hasattr(self, "drawer_drop_base"):
            self.drawer_drop_base = {}
        self.drawer_drop_base[place] = (x, y)
        # 落とす位置を「開いた取っ手からのずれ」としても覚えておく（置く直前に取っ手を見て合わせる）
        opened = getattr(self, "drawer_opened", {}).get(level, self.DRAWERS[level]["pull"])
        knob = (self.DRAWERS[level]["x"], self.DRAWER_KNOB_FACE_Y + opened)
        if not hasattr(self, "drawer_drop_refs"):
            self.drawer_drop_refs = {}
        self.drawer_drop_refs[place] = (level, opened, x - knob[0], y - knob[1])
        rospy.loginfo("drawer place %s: stand at (%.3f, %.3f), release with %s",
                      place, self.coordinates["positions"][place][0],
                      self.coordinates["positions"][place][1], pose)

    def locate_drawer_drop(self, place):
        """62401738 いしい
        開いた引き出しの取っ手を深度カメラで見て、落とす位置 drawer_drops[place] を取っ手の位置から決め直す。
        取っ手も手も同じ（ずれた）map 座標で測るので、自己位置のずれが打ち消される。
        見えなければ決めてある位置のまま。物を持っているので腕はたたまない。
        """
        level, opened, off_x, off_y = self.drawer_drop_refs[place]
        self.look_at_drawer(level, fold_arm=False)
        knob = self.detect_drawer_knob(level, self.DRAWER_KNOB_FACE_Y + opened, z_tol=0.10,
                                       y_tol=0.15)
        dx, dy = getattr(self, "drawer_slot", {}).get(place, (0.0, 0.0))
        if knob is None:
            rospy.logwarn("opened %s drawer knob not seen; use the planned drop point", level)
            base = self.drawer_drop_base[place]
            self.drawer_drops[place] = (base[0] + dx, max(base[1] + dy, self.DRAWER_DROP_Y_MIN))
            self.lock_target(place, *self.drawer_drops[place])
            return True
        old = self.drawer_drops[place]
        self.drawer_drop_base[place] = (knob[0] + off_x, max(knob[1] + off_y, self.DRAWER_DROP_Y_MIN + 0.05))
        self.drawer_drops[place] = (knob[0] + off_x + dx,
                                    max(knob[1] + off_y + dy, self.DRAWER_DROP_Y_MIN))
        self.lock_target(place, *self.drawer_drops[place])
        rospy.loginfo("drop point for %s: (%.3f, %.3f) -> (%.3f, %.3f) from the knob",
                      place, old[0], old[1], self.drawer_drops[place][0], self.drawer_drops[place][1])
        return True

    DRAWER_PALM_TO_TIPS = 0.058   # 腕を前に水平に出した姿勢での、手のひらから指先までの距離（FK）

    RELEASE_TIPS_Z0 = 0.51   # put_in_bin（arm_lift 0）の指先の高さ。arm_lift を上げた分だけ上がる
    # 容器・トレイの位置は決め打ちにしない（11/9 夜: Gazebo のモデル位置から作った決め打ち座標はやめる）。
    # 最初に長机の前（container_scan）で深度カメラで探し（prescan_containers）、
    # 置くたびに容器の前でもう一度見て中心を決め直す（locate_place_target）。

    # 背の高い物の高さ [m]（YCB の寸法）。真ん中をつかんでいるので、底は指先より高さの半分だけ下にある。
    # 容器の上 0.71 で離すと、底が縁（0.51）より上から落ちて倒れ、外へこぼれていた（11/9 windex_bottle）。
    TALL_OBJECTS = {
        "windex_bottle": 0.27, "bleach_cleanser": 0.25, "chips_can": 0.25, "pitcher_base": 0.24,
        "cracker_box": 0.21, "mustard_bottle": 0.19, "sugar_box": 0.18, "master_chef_can": 0.14,
    }
    # 細長い物の長さ [m]。上からつかむと指の間で回って縦にぶら下がり、高い所から離すと的を外していた（11/10）。
    LONG_OBJECTS = {
        "large_marker": 0.12, "small_marker": 0.10, "spatula": 0.33, "fork": 0.20, "spoon": 0.20,
    }
    RELEASE_CLEARANCE = 0.02   # 背の高い物・細長い物の底を、容器の縁からどれだけ上で離すか
    # 運ぶ高さの余裕: 物がつかんだ所から丸ごと下にぶら下がっても、底が縁よりこれだけ上を通る。
    # 11/10: 青いボトル（上からつかんで全長が指の下）を指先 0.71 で運び、底（約 0.44）が縁（0.53）に当たった
    CARRY_CLEARANCE = 0.05
    GENERIC_RELEASE_TIPS = 0.12   # それ以外の物を離す指先の高さ（縁から）。運ぶときは 0.18 上
    ARM_LIFT_MAX = 0.6

    def object_length(self, label):
        """62401738 いしい
        背の高い物・細長い物の長さ [m]。どちらでもなければ None。
        """
        return self.TALL_OBJECTS.get(label) or self.LONG_OBJECTS.get(label)

    def lower_tall_object(self, place, label):
        """62401738 いしい
        容器・トレイの真上で位置を合わせた後、腕を少し下げてから離す（高い所から離すと的を外す）。
        背の高い物・細長い物は、真ん中をつかんでいるとして底が縁の RELEASE_CLEARANCE 上に来る高さ、
        それ以外の物は指先が縁の GENERIC_RELEASE_TIPS 上に来る高さまで下げる。下げるだけで上げない。
        """
        target = self.placement_targets[place]
        rim_z = target.get("rim_z", target["z"] - self.PLACE_RIM_CLEARANCE)
        length = self.object_length(label)
        if length is not None:
            tips_z = rim_z + self.RELEASE_CLEARANCE + length / 2.0
        else:
            tips_z = rim_z + self.GENERIC_RELEASE_TIPS
        joints = dict(self.poses[self.release_pose_for(place, label)])
        lift = max(0.0, min(joints["arm_lift_joint"], tips_z - self.RELEASE_TIPS_Z0))
        if lift >= joints["arm_lift_joint"] - 0.01:
            return True
        rospy.loginfo("lower %s into %s: arm_lift %.3f -> %.3f",
                      label, place, joints["arm_lift_joint"], lift)
        joints["arm_lift_joint"] = lift
        self.move_joints(joints)
        return True  # 下ろせなくても、その高さで離す（持ったまま次へ行かない）

    def release_pose_for(self, place, label=None):
        """62401738 いしい
        place の上へ運ぶときの腕の姿勢名。指先の高さが目標の z（縁 + 0.18）になるよう arm_lift を決める。
        背の高い物・細長い物は、全長がぶら下がっても底が縁より CARRY_CLEARANCE 上を通る高さまで上げる
        （上に運んでから lower_tall_object で少し下げて離す）。
        """
        target = self.placement_targets[place]
        z = target["z"]
        length = self.object_length(label) if label else None
        if length is not None:
            rim_z = target.get("rim_z", z - self.PLACE_RIM_CLEARANCE)
            z = max(z, rim_z + length + self.CARRY_CLEARANCE)
        lift = max(0.0, min(self.ARM_LIFT_MAX, z - self.RELEASE_TIPS_Z0))
        name = "release_{}".format(place)
        self.poses[name] = {"arm_lift_joint": lift, "arm_flex_joint": -1.0, "arm_roll_joint": 0.0,
                            "wrist_flex_joint": -0.57, "wrist_roll_joint": 0.0,
                            "head_tilt_joint": 0.0, "head_pan_joint": 0.0}
        return name

    # 置く直前にカメラで探す容器・トレイの種類（detect_container_rims の kind）
    PLACE_KINDS = {"container_a_place": "container_a", "container_b_place": "container_b",
                   "tray_a_place": "tray", "tray_b_place": "tray"}
    PLACE_SEARCH_RADIUS = 0.25   # 予想位置からこれ以上離れた容器は別の物とみなす（自己位置のずれは最大 0.2）

    def lock_target(self, place, x, y):
        """62401738 いしい
        map 座標の目標 (x, y) を今の odom 座標に直して覚える（位置合わせは odom の中で行う）。
        """
        self.locked_targets = getattr(self, "locked_targets", {})
        odom = self.map_to_odom(x, y)
        if odom is None:
            self.locked_targets.pop(place, None)
        else:
            self.locked_targets[place] = odom

    def locate_place_target(self, place):
        """62401738 いしい
        容器・トレイの前に着いたら頭のカメラで見て、予想位置の近くにある同じ種類の容器の中心を目標にする。
        カメラの点群も手の位置も同じ（ずれた）自己位置で測るので、自己位置のずれが打ち消される。
        見つからなければ決めてある位置のまま。物を持っているので腕は動かさない。
        """
        import numpy as np

        target = self.placement_targets[place]
        kind = self.PLACE_KINDS.get(place)
        x, y = target["x"], target["y"]
        if kind is not None:
            head = whole_body.HEAD_CMD
            head.set_joint_value_target({"head_pan_joint": 0.0, "head_tilt_joint": -0.8})
            try:
                head.go(wait=True)
            finally:
                head.stop()
            rospy.sleep(0.5)
            found = []
            stamp = rospy.Time.now()
            for _ in range(2):
                try:
                    points, stamp = self.read_placement_cloud(stamp)
                except Exception as error:  # 点群が来なければ決めてある位置を使う
                    rospy.logwarn("no depth cloud at %s: %s", place, error)
                    break
                near = points[(np.abs(points[:, 0] - x) < 0.40) & (np.abs(points[:, 1] - y) < 0.35)]
                for item in self.detect_container_rims(near):
                    dist = math.hypot(item["x"] - x, item["y"] - y)
                    if item["kind"] == kind and dist < self.PLACE_SEARCH_RADIUS:
                        found.append((dist, item["x"], item["y"]))
            if found:
                _, best_x, best_y = min(found)
                same = [(fx, fy) for _, fx, fy in found
                        if math.hypot(fx - best_x, fy - best_y) < 0.05]
                seen_x = sum(v[0] for v in same) / len(same)
                seen_y = sum(v[1] for v in same) / len(same)
                rospy.loginfo("%s seen at (%.3f, %.3f); planned (%.3f, %.3f)",
                              place, seen_x, seen_y, x, y)
                x, y = seen_x, seen_y
                # 次に置くときの予想位置も、今見た中心にする（毎回センサで決め直す）
                target["x"], target["y"] = x, y
            else:
                rospy.logwarn("%s not seen; use the planned position (%.3f, %.3f)", place, x, y)
        self.lock_target(place, x, y)
        return True

    def align_tips_over(self, place, x, y, shelf_guard=False):
        """62401738 いしい
        腕の姿勢はそのまま台車だけ動かし、指先を目標の真上に合わせる（台車の向きによらない）。
        目標は locate で odom 座標に直したもの（なければ map の (x, y) を今 odom に直す）。
        odom の中で合わせるので、途中で自己位置の推定が変わっても手の行き先はずれない。
        合わせきれなくても、その場で離す。
        """
        locked = getattr(self, "locked_targets", {}).get(place)
        if locked is None:
            self.lock_target(place, x, y)
            locked = getattr(self, "locked_targets", {}).get(place)
        frame = "odom" if locked is not None else "map"
        tips = locked if locked is not None else (x, y)
        yaw = self.base_yaw(frame)
        if yaw is None:
            return True
        palm_x = tips[0] - self.DRAWER_PALM_TO_TIPS * math.cos(yaw)
        palm_y = tips[1] - self.DRAWER_PALM_TO_TIPS * math.sin(yaw)
        if not self.move_hand_by_base(palm_y, palm_x, timeout=12, x_tol=0.01, frame=frame,
                                      shelf_guard=shelf_guard):
            rospy.logwarn("could not align the hand over %s; release here", place)
        hand = self.lookup_xyz("hand_palm_link", frame)
        if hand is not None:
            rospy.loginfo("release over %s (%s): palm (%.3f, %.3f, %.3f), target tips (%.3f, %.3f)",
                          place, frame, hand[0], hand[1], hand[2], tips[0], tips[1])
        getattr(self, "locked_targets", {}).pop(place, None)
        return True

    def align_hand_over(self, place, x, y):
        """62401738 いしい
        容器・トレイの上に指先を合わせる（align_tips_over）。
        """
        return self.align_tips_over(place, x, y)

    def align_hand_over_drawer(self, place):
        """62401738 いしい
        引き出しの落とす位置（drawer_drops[place]）の真上に指先を合わせる（align_tips_over）。
        """
        x, y = self.drawer_drops[place]
        return self.align_tips_over(place, x, y, shelf_guard=True)

    def set_drawer_slot(self, place):
        """62401738 いしい
        place の引き出しに今度落とす位置を、DRAWER_DROP_SLOTS の次のずれにする（1 個ごとにずらして積まない）。
        台車の立つ位置も同じだけずらす（置く直前の位置合わせの移動を小さくする）。
        """
        self.drawer_slot_count = getattr(self, "drawer_slot_count", {})
        index = self.drawer_slot_count.get(place, 0)
        self.drawer_slot_count[place] = index + 1
        dx, dy = self.DRAWER_DROP_SLOTS[index % len(self.DRAWER_DROP_SLOTS)]
        self.drawer_slot = getattr(self, "drawer_slot", {})
        self.drawer_slot[place] = (dx, dy)
        x, y = self.drawer_drop_base[place]
        y = max(y + dy, self.DRAWER_DROP_Y_MIN) - dy
        self.drawer_drops[place] = (x + dx, y + dy)
        self.coordinates["positions"][place] = [
            x + dx - self.DRAWER_HAND_LEFT, y + dy + self.DRAWER_HAND_FORWARD, -90]
        rospy.loginfo("drawer %s: item %d, drop at (%.3f, %.3f)", place, index + 1, x + dx, y + dy)

    def select_drawer_place(self, label, category):
        """62401738 いしい
        引き出し行きの物体の置き場所を返す。~drawer_any（既定 True）なら、横の 1 段の引き出し
        （開いていなければ 2 段の方）。False なら以前どおり、工具は 2 段の引き出し、形のもの（Shape items）は横の引き出し。
        入れられる引き出しが開いていなければ例外にして、Task 1 側の代わりの置き場所（put_in_fallback）に任せる。
        """
        places = getattr(self, "drawer_places", {})
        if rospy.get_param("~drawer_any", True):
            # 11/10 判明したルール: 引き切った引き出しのどれかに入れば「引き出しに入れた」になる。
            # 引き出し行きの物（工具も形のものも）は基本的に横の 1 段の引き出しに入れる。
            # 横が開いていないときだけ 2 段の方に入れる。
            order = (self.DRAWER_LEFT_PLACE_NAME, self.DRAWER_PLACE_NAME)
        elif category == "drawer_tools":
            order = (self.DRAWER_PLACE_NAME,)
        else:
            order = (self.DRAWER_LEFT_PLACE_NAME,)
        candidates = [name for name in order if name in places]
        if not candidates:
            raise ValueError("Drawer for {} ({}) is not open".format(label, category))
        place = candidates[0]
        self.set_drawer_slot(place)
        rospy.loginfo("place %s (%s) at %s", label, category, place)
        return place, places[place]

    def put_in_fallback(self, label):
        """62401738 いしい
        正しい置き場所に置けなかった物体を、week1 と同じく bin_b に入れる（持ったまま次の物を取りに行かない）。
        以前は先に引き出しへ入れていたが、食べ物・台所用品が工具の引き出しに入って「置き場所が違う」状態になり、
        しかも引き出しは位置合わせが難しく、開いた引き出しのそばを何度も通ることになった。
        引き出しに入れても bin_b に入れても、カテゴリ違いの点（10 点）は同じなので、確実な bin_b にする。
        """
        return self.put_in_place("bin_b_place", "put_in_bin", label=label)

    # 時間配分（実行開始 run_start からの秒数。rospy の時刻なのでシミュレータの時間）。
    # Task 1 は物がなくなるまで続ける（全部取る）。それでも終わらないときだけ、ここで打ち切って Task 2 へ行く。
    # 11/9 08:30 は 10 分で打ち切っていたが、シミュレータの時間は実時間の約 2 倍の速さで進んでいて、
    # 実時間 5 分・3 個で Task 2a へ行ってしまった。Task 2a・2b はシミュレータの時間で 3〜4 分かかる。
    TASK1_END_SEC = 16 * 60      # これを過ぎたら Task 1 をやめる（~task1_end_sec で変更）
    TASK1_MIN_CYCLE_SEC = 60     # 物体 1 個（探す・つかむ・置く）にかかる時間の目安
    TASK1_RESCANS = 1            # 全部の場所で「物なし」になった後に、見直す回数
    TASK1_RESCAN_MIN_LEFT = 240  # 見直すのに必要な Task 1 の残り時間 [s]
    DRAWER_RETRY_END_SEC = 7 * 60   # 開ききらなかった引き出しをやり直すのはここまで（~drawer_retry_end_sec）
    DRAWER_DONE_RATIO = 0.9      # 引く量のこの割合以上開いていれば、引き切ったとみなす
    DRAWER_END_SEC = 6 * 60      # 引き出しを開けるのはここまで（~drawer_end_sec で変更。3 つで約 4 分）

    def elapsed_since_start(self):
        """62401738 いしい
        run() を始めてからの秒数。run() を通さずに呼ばれたときは、初めて呼ばれた時から数える。
        """
        if getattr(self, "run_start", None) is None:
            self.run_start = rospy.Time.now()
        return (rospy.Time.now() - self.run_start).to_sec()

    def task1_time_left(self):
        """62401738 いしい
        Task 1 に使える残りの秒数。
        """
        end = float(rospy.get_param("~task1_end_sec", self.TASK1_END_SEC))
        return end - self.elapsed_since_start()

    # ------------------------------------------------------------------
    # 点数が入るまで置き直す（/score が増えたかで確かめる）
    #   正しい置き場所に PLACE_TRIES 回 → だめならボックス（Bin B / Bin A）に BIN_TRIES 回まで。
    #   離したのに点が増えなければ、足元に落ちた同じ物を探して拾い直してから次を試す。
    # ------------------------------------------------------------------
    PLACE_TRIES = 2
    BIN_TRIES = 3
    BIN_PLACES = ("bin_b_place", "bin_a_place")
    SCORE_WAIT = 5.0        # 離してから点が入るまで待つ時間 [s]（物が落ち着くまで）
    SCORE_MIN_GAIN = 5.0    # これ以上増えたら入ったとみなす（1 個 10 点。衝突の減点 ×0.5 でも 5 点）
    REPICK_HEAD_TILT = -0.9  # 落とした物を探すときの頭の下向き [rad]
    # 拾い直しは置き場所の前（ロボットは -y 向き）で行う。手の向きは引き出しの取っ手と同じ（-y 向きで確認済み）
    REPICK_HAND_RPY = (-90, 100, 0)

    def score_available(self):
        """62401738 いしい
        /score を使って点数を確かめられるか（~use_score_feedback が False なら使わない）。
        """
        return bool(rospy.get_param("~use_score_feedback", True)) and self.score is not None

    def wait_for_score(self, before):
        """62401738 いしい
        得点が before から SCORE_MIN_GAIN 以上増えるのを SCORE_WAIT 秒まで待つ。増えたら True。
        """
        deadline = rospy.Time.now() + rospy.Duration(self.SCORE_WAIT)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.score is not None and self.score - before >= self.SCORE_MIN_GAIN:
                rospy.loginfo("score %.1f -> %.1f", before, self.score)
                return True
            rospy.sleep(0.2)
        rospy.logwarn("score did not increase (%.1f -> %s)", before, self.score)
        return False

    def try_place(self, label, place=None):
        """62401738 いしい
        持っている物を 1 回置き、点が入ったか確かめる。
        place が None なら正しい置き場所（put_object_by_label）、そうでなければその場所（ボックス）。
        戻り値は (点が入ったか, まだ持っているか)。
        """
        before = self.score
        if place is None:
            self.put_object_by_label(label)
        else:
            self.put_in_place(place, "put_in_bin", label=label)
        released = getattr(self, "last_placement_result", {}).get("released", False)
        if not released:
            return False, True
        return self.wait_for_score(before), False

    def place_until_scored(self, label):
        """62401738 いしい
        点数が入るまで置き直す。正しい置き場所に PLACE_TRIES 回、だめならボックスに BIN_TRIES 回。
        落としたときは拾い直す（拾えなければあきらめる）。点が入ったら True。
        """
        if not self.score_available():
            # 得点が分からないときは今までどおり（置けなければ bin_b）
            if self.put_object_by_label(label):
                return True
            result = getattr(self, "last_placement_result", {})
            if result.get("lost"):
                return False
            if not result.get("released", False):
                return self.put_in_fallback(label)
            return False
        plan = [None] * self.PLACE_TRIES + [
            self.BIN_PLACES[i % len(self.BIN_PLACES)] for i in range(self.BIN_TRIES)]
        holding = True
        for attempt, place in enumerate(plan):
            if self.task1_time_left() < -60 and not holding:
                break  # Task 2 の時間に食い込んでいるので、落ちた物はあきらめる
            if not holding and not self.repick_dropped(label):
                rospy.logwarn("cannot find dropped %s; give up", label)
                return False
            rospy.loginfo("place %s: try %d/%d at %s", label, attempt + 1, len(plan),
                          place or "its category place")
            scored, holding = self.try_place(label, place)
            if scored:
                return True
            result = getattr(self, "last_placement_result", {})
            if result.get("lost"):
                # 途中で落とした。置き場所の前には無いので拾い直さない（探索地点で見つかれば取り直す）
                rospy.logwarn("%s was dropped on the way; go back to searching", label)
                return False
            if place is None and result.get("stage") == "select_destination":
                # 置き場所が決まらない（容器が見つからない・引き出しが開いていない）。何度試しても同じなので箱へ
                plan[attempt + 1:self.PLACE_TRIES] = [self.BIN_PLACES[0]] * max(
                    0, self.PLACE_TRIES - attempt - 1)
        if holding:
            # 全部だめでも持ったまま次へ行かない（次の探索地点で落として物を蹴る）
            self.put_in_place(self.BIN_PLACES[0], "put_in_bin", label=label)
        return False

    def repick_dropped(self, label):
        """62401738 いしい
        置き場所の前で下を見て、落とした label の物を探して拾う。拾えたら True。
        ロボットは置き場所の方（-y）を向いている前提。向きが違えば拾わない。
        """
        yaw = self.base_yaw()
        if yaw is None or abs(math.degrees(yaw) + 90) > 30:
            rospy.logwarn("not facing -y (yaw %s); cannot re-pick here", yaw)
            return False
        self.drawer_pose("all_neutral")
        self.move_joints({"head_pan_joint": 0.0, "head_tilt_joint": self.REPICK_HEAD_TILT})
        rospy.sleep(0.5)
        objs = [obj for obj in self.get_latest_detection().bboxes
                if self.normalize_placement_label(obj.label) == label]
        best = self.get_most_graspable_obj(objs)
        if best is None:
            return False
        pos = self.get_grasp_coordinate(best["bbox"])
        if pos is None:
            return False
        rospy.loginfo("re-pick %s at (%.2f, %.2f, %.2f)", label, pos.x, pos.y, pos.z)
        yaw_h, pitch, roll = self.REPICK_HAND_RPY
        # 手前（+y、ロボット側）から差し込む。手のひらは物の中心より HAND_PALM_OFFSET 手前
        self.grasp_from_side(pos.x, pos.y + self.HAND_PALM_OFFSET, pos.z,
                             yaw_h, pitch, roll, "+y")
        gap = self.finger_gap()
        self.drawer_pose("all_neutral")
        if gap is not None and gap < 0.01:
            rospy.logwarn("re-pick of %s missed (gap %.3f)", label, gap)
            gripper.command(1)
            return False
        return True

    # ------------------------------------------------------------------
    # 取り直し（検出 → 把持を、その場で GRASP_TRIES 回まで）
    # ------------------------------------------------------------------
    GRASP_TRIES = 3
    GRASP_EMPTY_GAP = 0.005    # 閉じた後の指の間隔がこれ未満なら、何もつかんでいない（空で閉じると 0.001 未満）
    # 薄くて、つかめていても指の間隔がほぼ 0 になる物は、空振りの判定をしない
    THIN_LABELS = {"credit_card_blank", "fork", "spoon", "plate", "large_marker", "small_marker"}
    GRASP_AVOID_DIST = 0.08    # 家具に当たった物体の位置から、これ以内の物は同じ物とみなして避ける [m]

    def furniture_hits(self):
        """62401738 いしい
        これまでに受け取った、家具（wrc_*。Task 2a の障害物以外）との衝突メッセージの数。
        """
        return len([msg for msg in self.detection_list
                    if "Collision" in msg and "wrc_" in msg and "task2a" not in msg])

    def pick_here(self, plc, pose):
        """62401738 いしい
        今いる探索地点で物体を検出してつかむ。失敗したら検出し直して GRASP_TRIES 回まで取り直す。
        - 空振り（閉じた指の間隔が GRASP_EMPTY_GAP 未満）: 物が動いたかもしれないので検出し直す
        - つかみに行く途中で机などに当たった物: その位置を覚え、次からは別の物を選ぶ
          （同じ物に何度も手を伸ばして机にぶつけ続けない）
        戻り値は (つかんだ物のラベル or None, 状態)。状態は "ok" / "none"（物がない）/ "failed"。
        """
        self.grasp_avoid = getattr(self, "grasp_avoid", [])
        for attempt in range(self.GRASP_TRIES):
            if attempt > 0:
                self.change_pose(pose)
                gripper.command(1)
            objs = list(self.get_latest_detection().bboxes)
            target, grasp_pos = None, None
            while objs:
                obj = self.get_most_graspable_obj(objs)
                if obj is None:
                    break
                objs.remove(obj["bbox"])
                pos = self.get_grasp_coordinate(obj["bbox"])
                if pos is None:
                    continue
                if any(math.hypot(pos.x - a_x, pos.y - a_y) < self.GRASP_AVOID_DIST
                       for a_x, a_y in self.grasp_avoid):
                    rospy.loginfo("skip %s at (%.2f, %.2f): hit furniture before",
                                  obj["label"], pos.x, pos.y)
                    continue
                target, grasp_pos = obj, pos
                break
            if target is None:
                return None, ("none" if attempt == 0 else "failed")
            label = target["label"]
            rospy.loginfo("pick %s at %s: try %d/%d", label, plc, attempt + 1, self.GRASP_TRIES)
            hits_before = self.furniture_hits()
            # 110 点の版（49344b3）と同じく、つかむ前に腕を机の上の姿勢にする。
            # これがないと探索の姿勢から低い所を通って手を伸ばし、周りの物を蹴っていた（11/9）
            self.change_pose("grasp_on_table")
            if not self.exec_graspable_method(grasp_pos, label):
                # 手の届かない所の物（つかみに行っていない）。次からは選ばない
                self.grasp_avoid.append((grasp_pos.x, grasp_pos.y))
                self.change_pose("all_neutral")
                continue
            gap = self.finger_gap()
            hit = self.furniture_hits() > hits_before
            self.change_pose("all_neutral")
            # 腕を上げる途中で落とすこともあるので、上げた後にもう一度確かめる
            gap_up = self.finger_gap()
            self.log_drawer_poses("picking {} at {}".format(label, plc))
            empty = label not in self.THIN_LABELS and any(
                g is not None and g < self.GRASP_EMPTY_GAP for g in (gap, gap_up))
            if empty and gap_up is not None:
                gap = min(g for g in (gap, gap_up) if g is not None)
            if not empty:
                if hit:
                    rospy.logwarn("hit furniture while picking %s, but holding it", label)
                return label, "ok"
            gripper.command(1)
            if hit:
                # 机などに当たって取れなかった物は、次から選ばない
                self.grasp_avoid.append((grasp_pos.x, grasp_pos.y))
                rospy.logwarn("hit furniture and missed %s; avoid it from now on", label)
            else:
                rospy.logwarn("missed %s (finger gap %.3f); detect again", label, gap)
        return None, "failed"

    def execute_task1(self):
        """62408993 Masami Sakuma
        Task1を実行する。
        """
        rospy.loginfo("#### start Task 1 ####")

        hsr_position = [
            ("tall_table", "look_at_tall_table"),
            ("near_long_table_l", "look_at_near_floor"),
            ("long_table_r", "look_at_tall_table")
        ]

        # 置き場所の判断は B（put_object_by_label）に任せる（統合時に一本化）
        placed = []
        empty_count = [0] * len(hsr_position)
        max_attempts_per_place = 12
        attempt_count = [0] * len(hsr_position)

        # 物を持つ前に引き出しを開けておく（run() で開けていれば何もしない）
        self.open_drawers()

        rospy.loginfo("Start searching objects.")

        time_up = False
        rescans = 0
        while not time_up:
            if all(count >= 2 for count in empty_count):
                # 3 か所とも「物がない」と見たが、見落としで終わらないよう、時間が残っていれば 1 回だけ見直す
                if rescans < self.TASK1_RESCANS and self.task1_time_left() > self.TASK1_RESCAN_MIN_LEFT:
                    rescans += 1
                    rospy.logwarn("No object seen at all places, %.0f s left; scan once more (%d/%d)",
                                  self.task1_time_left(), rescans, self.TASK1_RESCANS)
                    empty_count = [0] * len(hsr_position)
                    attempt_count = [0] * len(hsr_position)
                else:
                    break
            for i, (plc, pose) in enumerate(hsr_position):
                # 次の物体を取りに行く前に、Task 2 の時間が残るか確かめる（1 個の途中では止めない）
                if self.task1_time_left() < self.TASK1_MIN_CYCLE_SEC:
                    rospy.logwarn("Task 1 time is up (%.0f s left). Go to Task 2.",
                                  self.task1_time_left())
                    time_up = True
                    break
                if empty_count[i] >= 2 or attempt_count[i] >= max_attempts_per_place:
                    empty_count[i] = 2
                    continue

                self.goto_name(plc)
                self.change_pose(pose)
                gripper.command(1)

                attempt_count[i] += 1
                # 検出 → 把持。失敗（空振り・家具に当たった）したら、その場で検出し直して取り直す（pick_here）
                label, status = self.pick_here(plc, pose)
                if status == "none":
                    empty_count[i] += 1
                    rospy.loginfo("No object at %s (%d/2)", plc, empty_count[i])
                    continue
                empty_count[i] = 0
                if label is None:
                    rospy.logwarn("Could not grasp any object at %s. Try other places.", plc)
                    continue

                # 点数が入るまで置き直す（正しい置き場所に 2 回 → だめならボックス）。
                # 置き場所の決め方・置く動作は B の put_object_by_label / put_in_place を使う
                label = self.normalize_placement_label(label)
                if self.place_until_scored(label):
                    placed.append("{}({})".format(
                        label, getattr(self, "last_placement_result", {}).get("place")))

        rospy.loginfo("#### finish Task 1 ####")
        self.log_drawer_poses("Task 1")
        rospy.loginfo("Placed objects (%d): %s", len(placed), ", ".join(placed))

    # ------------------------------------------------------------------
    # Task 2a / 2b（担当 C）
    # ------------------------------------------------------------------
    TASK2A_LANES_X = [1.7, 2.15, 2.6]   # 通過するレーンの x 座標（サンプルの xa, xb, xc）
    TASK2A_STEPS_Y = [2.5, 2.9, 3.3]    # 各ステップで進む y 座標
    TASK2A_START = (2.5, 1.85)          # standby_2a の座標
    TASK2A_YAW = 90                     # 進行方向（+y）を向いたまま移動し、カメラで前方を見る
    TASK2A_AREA = {"x": (1.2, 3.2), "y": (1.9, 3.9), "z_max": 0.3}  # 床の障害物とみなす範囲
    # ロボット中心の経路と障害物中心の最小距離 [m]
    # = HSR 半径 0.22 + 物体の半径（最大 0.1 程度）+ 自己位置・追従のずれの余裕 0.13
    TASK2A_SAFE_DIST = 0.45
    # A* で試す安全距離（広い順）。0.32 = 半径 0.22 + 物体 0.05 + 余裕 0.05（> TASK2A_DANGER_DIST）
    # 最後の 0.28 は、ほかに道がないときだけ使う（接触は 0.26〜0.27。11/9 08:31 は lego と foam_brick の
    # 間が 0.58m しかなく、0.32 では道がないので壁の角を削って回り込み、枠に当たった）
    # 0.28 の段はやめた（11/9 夜）。接触は 0.26〜0.27 で余裕が 1〜2cm しかなく、壁や枠に当たると減点される
    TASK2A_SAFE_DISTS = (0.45, 0.38, 0.32)
    TASK2A_NARROW_DIST = 0.32         # これより狭い安全距離では、格子を細かくし、見張りの距離も詰める
    TASK2A_GRID_FINE = 0.025          # 狭い所を探すときの格子 [m]（5cm では隙間の真ん中が格子に乗らない）
    TASK2A_DANGER_MARGIN = 0.015      # 狭い道のときの見張りの距離 = 安全距離 - これ
    TASK2A_MERGE_DIST = 0.15  # これより近い検出は同じ障害物とみなす [m]
    TASK2B_WAIT_SEC = 10.0    # 棚の前で指示文を待つ最大時間 [s]
    TASK2B_DETECT_RETRY = 3   # 対象物体が見つからないときに検出をやり直す回数

    # A* による経路計画（Task 2a）
    TASK2A_GOAL = (2.0, 3.5)          # go_throw_2a の座標
    TASK2A_PLAN_X = (1.55, 2.75)      # 経路を探す範囲 x [m]（サンプルのレーン 1.7〜2.6 の少し外まで）
    TASK2A_PLAN_Y = (1.8, 3.6)        # 経路を探す範囲 y [m]
    TASK2A_GRID = 0.05                # 格子の大きさ [m]
    TASK2A_SOFT_DIST = 0.6            # 障害物にこれより近い格子はコストを上乗せする [m]
    TASK2A_CLEAR_WEIGHT = 2.0         # 上乗せの重み
    TASK2A_STEP_MAX = 0.4             # 1回に進む最大距離 [m]（進むたびに検出し直す）
    TASK2A_MAX_MOVES = 15             # 移動回数の上限（脱出・計画し直しを含む）

    # 速度制御での直進（move_base を使わず、A* の線分どおりに動く）
    ROBOT_RADIUS = 0.22               # HSR の台車の半径 [m]
    TASK2A_DANGER_DIST = 0.30         # 移動中、障害物の中心にこれより近づこうとしたら止まる [m]
    TASK2A_MAX_SPEED = 0.15           # 最大速度 [m/s]
    TASK2A_GAIN = 1.0                 # 位置の比例ゲイン [1/s]
    TASK2A_YAW_GAIN = 1.5             # 向きの比例ゲイン [1/s]
    TASK2A_GOAL_TOL = 0.03            # 到着とみなす距離 [m]
    TASK2A_CONTROL_DT = 0.1           # 制御周期 [s]

    # 首を振って見渡す（前を向いたまま横にも動くので、横の物体を見落とさないため）
    TASK2A_HEAD_TILT = -1.0           # look_at_near_floor と同じ下向き [rad]
    TASK2A_NEAR_TILT = -1.3           # 足元近くを見るときの下向き [rad]（開始時に1回）
    TASK2A_SCAN_PAN = 0.7             # 左右を見るときの首の角度 [rad]（正が左）
    TASK2A_SIDE_MOVE = 0.1            # これ以上横に動くときは、先にその方向を見る [m]

    # 踏んでしまったときの脱出
    ESCAPE_BACK_DIST = 0.15           # 2回目以降に追加で下がる距離 [m]
    ESCAPE_MAX_TRY = 3                # 下がる回数の上限
    ESCAPE_WAIT = 1.0                 # 下がった後、衝突が止まったかを見る時間 [s]
    HIT_MERGE_DIST = 0.5              # この距離以内で再び踏んだら同じ物体とみなし、推定位置を平均する [m]

    # 検出結果の安定化（複数フレームの多数決）
    DETECT_FRAMES = 3     # 何回検出するか
    DETECT_MIN_HITS = 2   # 何回以上見えたものを採用するか
    DETECT_IOU = 0.5      # 同じ物体とみなす BBox の重なり（IoU）
    DETECT_FRAMES_YOLO = 5     # YOLO のときの検出回数（速いので多めに）
    DETECT_MIN_HITS_YOLO = 3   # YOLO のときに採用する最低回数

    # 指示文の人物 → 配達先 waypoint
    # 人物の正面から見て左が person_a（y=2.8）、右が person_b（y=3.9）と仮定。
    # 実際の指示文で確認すること（README の実行記録に書く）
    PERSON_WAYPOINTS = {
        "left": "person_a", "right": "person_b",
        "a": "person_a", "b": "person_b",
    }
    DEFAULT_PERSON = "right"

    # 検出ラベル（wrs_detector/config/wod-c_classes.json）
    DETECTOR_LABELS = [
        "cracker_box", "sugar_box", "pudding_box", "gelatin_box", "potted_meat_can",
        "master_chef_can", "tuna_fish_can", "chips_can", "mustard_bottle", "tomato_soup_can",
        "banana", "strawberry", "apple", "lemon", "peach", "pear", "orange", "plum",
        "windex_bottle", "bleach_cleanser", "sponge", "pitcher_base", "pitcher_lid", "plate",
        "bowl", "fork", "spoon", "spatula", "wine_glass", "mug", "large_marker", "small_marker",
        "padlock", "bolt_and_nut", "clamp", "credit_card_blank", "mini_soccer_ball", "softball",
        "baseball", "tennis_ball", "racquetball", "golf_ball", "marble", "cup", "foam_brick",
        "dice", "rope", "chain", "rubiks_cube", "colored_wood_block", "nine_hole_peg_test",
        "toy_airplane", "lego_duplo", "magazine", "black_t_shirt", "timer",
    ]
    # ラベル名以外の言い方（指示文で使われそうなもの）
    LABEL_ALIASES = {
        "cheez it": "cracker_box", "cracker": "cracker_box", "crackers": "cracker_box",
        "sugar": "sugar_box", "pudding": "pudding_box", "jello": "gelatin_box",
        "gelatin": "gelatin_box", "spam": "potted_meat_can", "potted meat": "potted_meat_can",
        "coffee can": "master_chef_can", "master chef": "master_chef_can",
        "tuna": "tuna_fish_can", "tuna can": "tuna_fish_can", "pringles": "chips_can",
        "chips": "chips_can", "mustard": "mustard_bottle", "tomato soup": "tomato_soup_can",
        "soup can": "tomato_soup_can", "windex": "windex_bottle",
        "spray bottle": "windex_bottle", "cleanser": "bleach_cleanser",
        "bleach": "bleach_cleanser", "pitcher": "pitcher_base", "wine glass": "wine_glass",
        "marker": "large_marker", "keys": "padlock", "key": "padlock",
        "bolt": "bolt_and_nut", "nut": "bolt_and_nut", "credit card": "credit_card_blank",
        "soccer ball": "mini_soccer_ball", "soft ball": "softball", "marbles": "marble",
        "cups": "cup", "clamps": "clamp", "rubik s cube": "rubiks_cube",
        "rubiks cube": "rubiks_cube", "rubik cube": "rubiks_cube",
        "wood block": "colored_wood_block", "wooden block": "colored_wood_block",
        "peg test": "nine_hole_peg_test", "9 peg hole test": "nine_hole_peg_test",
        "airplane": "toy_airplane", "lego": "lego_duplo", "t shirt": "black_t_shirt",
        "tshirt": "black_t_shirt",
    }

    @staticmethod
    def normalize_text(text):
        """62401738 いしい
        文字列を小文字・英数字と空白だけにする（"Tuna_Fish-Can." → "tuna fish can"）
        """
        text = text.lower()
        text = "".join([c if c.isalnum() else " " for c in text])
        return " ".join(text.split())

    @classmethod
    def extract_target_obj_and_person(cls, instruction):
        """62401738 いしい
        指示文から対象の物体ラベルと人物（"left" / "right" / "a" / "b"）を取り出す。
        見つからない場合はそれぞれ None を返す。
        """
        rospy.loginfo("[extract_target_obj_and_person] instruction: %s", instruction)
        words = " " + cls.normalize_text(instruction) + " "

        # 物体: ラベル名と別名のうち、指示文に含まれる最も長いものを採用する
        candidates = [(cls.normalize_text(label), label) for label in cls.DETECTOR_LABELS]
        candidates += list(cls.LABEL_ALIASES.items())
        target_obj = None
        best_len = 0
        for phrase, label in candidates:
            for form in (phrase, phrase + "s", phrase + "es"):
                if " " + form + " " in words and len(phrase) > best_len:
                    target_obj = label
                    best_len = len(phrase)

        # 人物: left / right / person a / person b
        target_person = None
        if " left " in words:
            target_person = "left"
        elif " right " in words:
            target_person = "right"
        else:
            for key in ("a", "b"):
                if " person " + key + " " in words:
                    target_person = key

        rospy.loginfo("[extract_target_obj_and_person] target_obj=%s target_person=%s",
                      target_obj, target_person)
        return target_obj, target_person

    def wait_for_instruction(self, timeout):
        """62401738 いしい
        指示文が届くまで最大 timeout 秒待ち、最新の指示文を返す。届かなければ None。
        """
        end_time = rospy.Time.now() + rospy.Duration(timeout)
        while not self.instruction_list and rospy.Time.now() < end_time:
            rospy.sleep(0.5)
        if not self.instruction_list:
            return None
        return self.instruction_list[-1]

    def get_bbox_position(self, bbox):
        """62401738 いしい
        BBox の map 座標を返す。tf が取れなければ None（get_grasp_coordinate の None 対応版）
        """
        self.tf_from_bbox_clt.call(SetTransformFromBBoxRequest(bbox=bbox, frame=self.GRASP_TF_NAME))
        rospy.sleep(1.0)  # tfが安定するのを待つ
        trans = self.get_relative_coordinate("map", self.GRASP_TF_NAME)
        if trans is None:
            return None
        return trans.translation

    def deliver_to_target(self, target_obj, target_person):
        """62401738 いしい
        棚で target_obj を取り、target_person に渡す。
        target_obj が見つからなければ、棚で見えている物体を代わりに渡す（誤った物体でも20点）。
        """
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")
        self.change_pose("look_at_shelf")
        rospy.loginfo("target_obj: %s  target_person: %s", target_obj, target_person)

        # 物体検出結果（複数フレームの多数決）から、把持するbboxを決定（見つからなければやり直す）
        grasp_bbox = None
        bboxes = []
        for _ in range(self.TASK2B_DETECT_RETRY):
            bboxes = self.get_stable_detection()
            matches = [bbox for bbox in bboxes if bbox.label == target_obj]
            if matches:
                grasp_bbox = max(matches, key=lambda bbox: bbox.score)
                break
            rospy.sleep(1.0)

        if grasp_bbox is None:
            rospy.logwarn("Cannot find [%s]. deliver another object instead.", target_obj)
            if bboxes:
                grasp_bbox = max(bboxes, key=lambda bbox: bbox.score)
        if grasp_bbox is None:
            rospy.logwarn("Cannot find any object on the shelf. task2b is aborted.")
            return
        rospy.loginfo("grasp [%s] (%.2f)", grasp_bbox.label, grasp_bbox.score)

        # BBoxの3次元座標を取得して、その座標で把持する
        grasp_pos = self.get_bbox_position(grasp_bbox)
        if grasp_pos is None:
            rospy.logwarn("Cannot get the position of the object. task2b is aborted.")
            return
        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos)
        self.change_pose("all_neutral")

        # target_personの前に持っていく（week1 に戻した。スライドして渡す方式はうまくいかなかった）
        if target_person not in self.PERSON_WAYPOINTS:
            rospy.logwarn("unknown person [%s]. use default [%s].", target_person,
                          self.DEFAULT_PERSON)
            target_person = self.DEFAULT_PERSON
        self.change_pose("look_at_near_floor")
        self.goto_name(self.PERSON_WAYPOINTS[target_person])
        self.change_pose("deliver_to_human")
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")

    @classmethod
    def is_task2a_obstacle(cls, pos):
        """62401738 いしい
        検出した物体が Task 2a の床の障害物か（範囲内・床の高さ）を判定する
        """
        area = cls.TASK2A_AREA
        return (area["x"][0] <= pos.x <= area["x"][1]
                and area["y"][0] <= pos.y <= area["y"][1]
                and pos.z <= area["z_max"])

    @classmethod
    def add_obstacle(cls, obstacles, new_xy):
        """62401738 いしい
        障害物リストに追加する。既にある障害物の近くなら、位置を平均して1つにまとめる。
        """
        for i, (obs_x, obs_y) in enumerate(obstacles):
            if math.hypot(obs_x - new_xy[0], obs_y - new_xy[1]) < cls.TASK2A_MERGE_DIST:
                obstacles[i] = ((obs_x + new_xy[0]) / 2, (obs_y + new_xy[1]) / 2)
                return
        obstacles.append(new_xy)

    @staticmethod
    def dist_point_to_segment(point, seg_start, seg_end):
        """62401738 いしい
        点 point と線分 seg_start-seg_end の距離を返す（座標はすべて (x, y)）
        """
        seg_x = seg_end[0] - seg_start[0]
        seg_y = seg_end[1] - seg_start[1]
        seg_len2 = seg_x ** 2 + seg_y ** 2
        if seg_len2 == 0:
            ratio = 0.0
        else:
            dot = (point[0] - seg_start[0]) * seg_x + (point[1] - seg_start[1]) * seg_y
            ratio = dot / seg_len2
            ratio = max(0.0, min(1.0, ratio))
        near_x = seg_start[0] + ratio * seg_x
        near_y = seg_start[1] + ratio * seg_y
        return math.hypot(point[0] - near_x, point[1] - near_y)

    @staticmethod
    def bbox_iou(box_a, box_b):
        """62401738 いしい
        2つの BBox の IoU（重なった面積 / 合わせた面積）を返す
        """
        inter_w = min(box_a.x + box_a.w, box_b.x + box_b.w) - max(box_a.x, box_b.x)
        inter_h = min(box_a.y + box_a.h, box_b.y + box_b.h) - max(box_a.y, box_b.y)
        if inter_w <= 0 or inter_h <= 0:
            return 0.0
        inter = inter_w * inter_h
        union = box_a.w * box_a.h + box_b.w * box_b.h - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def merge_detections(cls, frames, min_hits):
        """62401738 いしい
        複数フレームの検出結果を、同じラベルで IoU が DETECT_IOU 以上のものどうしでまとめる。
        min_hits 回以上のフレームで見えた物体だけを残し（1回だけの誤検出を消す）、
        BBox の位置は各座標の中央値、score はフレーム数で割った平均（見えなかった回は0）にする。
        """
        clusters = []
        for frame_id, frame in enumerate(frames):
            for bbox in frame:
                for cluster in clusters:
                    if (cluster["label"] == bbox.label and frame_id not in cluster["frames"]
                            and cls.bbox_iou(cluster["boxes"][0], bbox) >= cls.DETECT_IOU):
                        cluster["boxes"].append(bbox)
                        cluster["frames"].add(frame_id)
                        break
                else:
                    clusters.append({"label": bbox.label, "boxes": [bbox], "frames": {frame_id}})

        def median(values):
            """62401738 いしい
            値のリストの中央値を返す
            """
            values = sorted(values)
            return values[len(values) // 2]

        merged = []
        for cluster in clusters:
            if len(cluster["frames"]) < min_hits:
                continue
            boxes = cluster["boxes"]
            rep = copy.copy(max(boxes, key=lambda box: box.score))
            rep.x = median([box.x for box in boxes])
            rep.y = median([box.y for box in boxes])
            rep.w = median([box.w for box in boxes])
            rep.h = median([box.h for box in boxes])
            rep.score = sum([box.score for box in boxes]) / len(frames)
            merged.append(rep)
        return sorted(merged, key=lambda box: box.score, reverse=True)

    @staticmethod
    def get_detector_type():
        """62401738 いしい
        検出ノードが使っている検出器を返す（"yolo" / "frcnn_finetuned"。分からなければ "unknown"）。
        start_all.sh c では YOLO が標準。
        """
        return rospy.get_param("/frcnn_detector/detector", "unknown")

    def get_detect_frames(self):
        """62401738 いしい
        安定化で何回検出し、何回以上見えたものを残すかを返す。
        YOLO は1回の検出が速い（約3ms）ので回数を増やし、多数決をより確かにする。
        """
        if self.get_detector_type() == "yolo":
            return self.DETECT_FRAMES_YOLO, self.DETECT_MIN_HITS_YOLO
        return self.DETECT_FRAMES, self.DETECT_MIN_HITS

    def get_stable_detection(self):
        """62401738 いしい
        複数回検出し、merge_detections で多数決した結果を返す
        （回数は get_detect_frames で検出器に合わせて決める）
        """
        n_frames, min_hits = self.get_detect_frames()
        frames = []
        for i in range(n_frames):
            frames.append(list(self.get_latest_detection().bboxes))
            if i < n_frames - 1:
                rospy.sleep(0.3)
        merged = self.merge_detections(frames, min_hits)
        rospy.loginfo("stable detection: %s",
                      ", ".join(["{}({:.2f})".format(box.label, box.score) for box in merged]))
        return merged

    def update_task2a_obstacles(self, obstacles):
        """62401738 いしい
        安定化した検出結果から床の障害物を取り出し、obstacles（map 座標）に追加する
        """
        for bbox in self.get_stable_detection():
            pos = self.get_bbox_position(bbox)
            if pos is None:
                continue
            if not self.is_task2a_obstacle(pos):
                rospy.loginfo("ignore %s (%.2f, %.2f, %.2f)", bbox.label, pos.x, pos.y, pos.z)
                continue
            rospy.loginfo("obstacle %s (%.2f, %.2f, %.2f)", bbox.label, pos.x, pos.y, pos.z)
            self.add_obstacle(obstacles, (pos.x, pos.y))

    @classmethod
    def plan_path_astar(cls, start, goal, obstacles, safe_dist, grid=None):
        """62401738 いしい
        TASK2A_GRID 間隔の格子の上で、start から goal までの経路を A* で探す。
        - 障害物から safe_dist 未満の格子は通れない（start の格子だけは例外）
        - 障害物から TASK2A_SOFT_DIST 未満の格子は、近いほどコストを上乗せする
          （ぎりぎりを通らず、余裕のある経路を選ぶため）
        - 8近傍で移動し、ヒューリスティックはゴールまでの直線距離
        経路（(x, y) のリスト）を返す。見つからなければ None。
        """
        res = grid or cls.TASK2A_GRID
        x_min, x_max = cls.TASK2A_PLAN_X
        y_min, y_max = cls.TASK2A_PLAN_Y
        n_x = int(round((x_max - x_min) / res)) + 1
        n_y = int(round((y_max - y_min) / res)) + 1

        def to_cell(point):
            """62401738 いしい
            map 座標 (x, y) を格子の番号に変換する（範囲外は端の格子にする）
            """
            c_x = int(round((point[0] - x_min) / res))
            c_y = int(round((point[1] - y_min) / res))
            return (max(0, min(n_x - 1, c_x)), max(0, min(n_y - 1, c_y)))

        def to_xy(cell):
            """62401738 いしい
            格子の番号を map 座標 (x, y) に変換する
            """
            return (x_min + cell[0] * res, y_min + cell[1] * res)

        clearance_cache = {}

        def clearance(cell):
            """62401738 いしい
            格子の中心から一番近い障害物までの距離を返す（計算結果は覚えておく）
            """
            if cell not in clearance_cache:
                p_x, p_y = to_xy(cell)
                dists = [math.hypot(p_x - o_x, p_y - o_y) for o_x, o_y in obstacles]
                clearance_cache[cell] = min(dists) if dists else float("inf")
            return clearance_cache[cell]

        def near_wall(cell):
            """62401738 いしい
            格子が地図の壁から TASK2A_WALL_SAFE 未満か（台車が壁・扉の枠に当たる）
            """
            return cls.wall_clearance(to_xy(cell)) < cls.TASK2A_WALL_SAFE

        start_cell, goal_cell = to_cell(start), to_cell(goal)
        if clearance(goal_cell) < safe_dist:
            return None

        def heuristic(cell):
            """62401738 いしい
            A* のヒューリスティック: 格子からゴールまでの直線距離 [m]
            """
            return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1]) * res

        cost = {start_cell: 0.0}
        parent = {start_cell: None}
        open_list = [(heuristic(start_cell), start_cell)]
        closed = set()
        while open_list:
            _, cell = heapq.heappop(open_list)
            if cell in closed:
                continue
            if cell == goal_cell:
                path = []
                while cell is not None:
                    path.append(to_xy(cell))
                    cell = parent[cell]
                path.reverse()
                path[0] = (start[0], start[1])
                path[-1] = (goal[0], goal[1])
                return path
            closed.add(cell)
            for d_x in (-1, 0, 1):
                for d_y in (-1, 0, 1):
                    nxt = (cell[0] + d_x, cell[1] + d_y)
                    if (d_x == 0 and d_y == 0) or nxt in closed:
                        continue
                    if not (0 <= nxt[0] < n_x and 0 <= nxt[1] < n_y):
                        continue
                    clr = clearance(nxt)
                    if clr < safe_dist or (nxt != goal_cell and near_wall(nxt)):
                        continue
                    step = math.hypot(d_x, d_y) * res
                    penalty = cls.TASK2A_CLEAR_WEIGHT * max(0.0, cls.TASK2A_SOFT_DIST - clr)
                    new_cost = cost[cell] + step * (1.0 + penalty)
                    if new_cost < cost.get(nxt, float("inf")):
                        cost[nxt] = new_cost
                        parent[nxt] = cell
                        heapq.heappush(open_list, (new_cost + heuristic(nxt), nxt))
        return None

    @classmethod
    def is_segment_free(cls, seg_start, seg_end, obstacles, safe_dist):
        """62401738 いしい
        線分の上をロボットが通ったとき、すべての障害物から safe_dist 以上離れているか
        """
        if not all([cls.dist_point_to_segment(obs, seg_start, seg_end) >= safe_dist
                    for obs in obstacles]):
            return False
        # 地図の壁（部屋の境目の枠など）にも近づかない
        return all(cls.dist_point_to_segment(wall, seg_start, seg_end) >= cls.TASK2A_WALL_SAFE
                   for wall in cls.task2a_walls)

    # 地図（/static_map）の壁。Task 2a の範囲の占有セルの中心を入れておく（load_task2a_walls）。
    # 11/9 08:31、A* が部屋の境目の壁の端（x 2.05, y 2.1〜2.2）をかすめる斜めの経路を選び、
    # 台車が wrc_frame に当たった。床の物だけでなく壁も避ける。
    task2a_walls = []
    TASK2A_WALL_SAFE = 0.32   # 壁の中心から台車の中心まで（半径 0.22 + セルの半分 0.025 + 余裕）

    @classmethod
    def wall_clearance(cls, point):
        """62401738 いしい
        point から一番近い壁（地図の占有セル）までの距離。壁が読めていなければ無限大。
        """
        if not cls.task2a_walls:
            return float("inf")
        return min(math.hypot(point[0] - w_x, point[1] - w_y) for w_x, w_y in cls.task2a_walls)

    def load_task2a_walls(self):
        """62401738 いしい
        /static_map から Task 2a の経路を探す範囲（の少し外まで）の壁を読み込む。読めなければ壁なしで続ける。
        """
        from nav_msgs.srv import GetMap
        try:
            rospy.wait_for_service("/static_map", timeout=5.0)
            grid = rospy.ServiceProxy("/static_map", GetMap)().map
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logwarn("cannot read /static_map: %s (walls are not avoided)", error)
            return
        info = grid.info
        margin = self.TASK2A_WALL_SAFE + 0.1
        x_lo, x_hi = self.TASK2A_PLAN_X[0] - margin, self.TASK2A_PLAN_X[1] + margin
        y_lo, y_hi = self.TASK2A_PLAN_Y[0] - margin, self.TASK2A_PLAN_Y[1] + margin
        walls = []
        for j in range(info.height):
            y = info.origin.position.y + (j + 0.5) * info.resolution
            if not y_lo <= y <= y_hi:
                continue
            row = j * info.width
            for i in range(info.width):
                x = info.origin.position.x + (i + 0.5) * info.resolution
                if x_lo <= x <= x_hi and grid.data[row + i] > 50:
                    walls.append((x, y))
        type(self).task2a_walls = walls
        rospy.loginfo("task2a: %d wall cells from the map", len(walls))

    @classmethod
    def simplify_path(cls, path, obstacles, safe_dist):
        """62401738 いしい
        A* の格子経路から、障害物に近づかずに直線で結べる点を飛ばして、曲がり角だけにする
        """
        if len(path) <= 2:
            return list(path)
        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not cls.is_segment_free(path[i], path[j], obstacles, safe_dist):
                j -= 1
            result.append(path[j])
            i = j
        return result

    @classmethod
    def plan_next_point(cls, current, goal, obstacles):
        """62401738 いしい
        A* で current から goal までの経路を求め、次に進む点（最大 TASK2A_STEP_MAX 先）を返す。
        安全距離は TASK2A_SAFE_DISTS の順（広い→狭い）に試し、広く避けられるときは広く、
        障害物の間が狭いときだけ詰めて通る。どれでも経路がなければ None。
        """
        for safe_dist in cls.TASK2A_SAFE_DISTS:
            narrow = safe_dist < cls.TASK2A_NARROW_DIST
            path = cls.plan_path_astar(current, goal, obstacles, safe_dist,
                                       cls.TASK2A_GRID_FINE if narrow else None)
            if path is None:
                rospy.logwarn("A*: no path with safe_dist=%.2f", safe_dist)
                continue
            # 狭い道を選んだときは、移動中に止まる距離も詰める（0.30 のままだと隙間の手前で止まる）
            cls.task2a_danger = (safe_dist - cls.TASK2A_DANGER_MARGIN if narrow
                                 else cls.TASK2A_DANGER_DIST)
            corners = cls.simplify_path(path, obstacles, safe_dist)
            rospy.loginfo("A* path: %s",
                          " -> ".join(["({:.2f}, {:.2f})".format(p[0], p[1]) for p in corners]))
            # すぐ隣の格子だと到着判定にすぐ引っかかって進まないので、10cm 以上先の角を選ぶ
            far = [c for c in corners[1:] if math.hypot(c[0] - current[0], c[1] - current[1]) > 0.1]
            target = far[0] if far else corners[-1]
            dist = math.hypot(target[0] - current[0], target[1] - current[1])
            if dist > cls.TASK2A_STEP_MAX:
                ratio = cls.TASK2A_STEP_MAX / dist
                target = (current[0] + (target[0] - current[0]) * ratio,
                          current[1] + (target[1] - current[1]) * ratio)
            return target
        return None

    def scan_task2a_obstacles(self, obstacles, pans):
        """62401738 いしい
        首を pans の各角度 [rad]（正が左）に向けて検出し、障害物を obstacles に追加する。
        最後は正面（pan=0）に戻す。
        """
        for pan in pans:
            whole_body.move_to_joint_positions(
                {"head_pan_joint": pan, "head_tilt_joint": self.TASK2A_HEAD_TILT})
            self.update_task2a_obstacles(obstacles)
        if pans and pans[-1] != 0.0:
            whole_body.move_to_joint_positions(
                {"head_pan_joint": 0.0, "head_tilt_joint": self.TASK2A_HEAD_TILT})

    @classmethod
    def side_to_look(cls, pose, target):
        """62401738 いしい
        pose (x, y, yaw) から target へ動くとき、ロボットから見て横に TASK2A_SIDE_MOVE 以上動くなら
        その方向を見る首の角度（左 +TASK2A_SCAN_PAN / 右 -TASK2A_SCAN_PAN）を返す。横に動かなければ None。
        """
        d_x, d_y = target[0] - pose[0], target[1] - pose[1]
        lateral = -math.sin(pose[2]) * d_x + math.cos(pose[2]) * d_y  # ロボット座標の y（左が正）
        if lateral > cls.TASK2A_SIDE_MOVE:
            return cls.TASK2A_SCAN_PAN
        if lateral < -cls.TASK2A_SIDE_MOVE:
            return -cls.TASK2A_SCAN_PAN
        return None

    def get_robot_pose(self):
        """62401738 いしい
        tf からロボットの今の位置と向きを (x, y, yaw[rad]) で返す。取れなければ None。
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rospy.Time(0), rospy.Duration(1.0)).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            rospy.logwarn("cannot get robot pose from tf")
            return None
        rot = trans.rotation
        yaw = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                         1.0 - 2.0 * (rot.y ** 2 + rot.z ** 2))
        return (trans.translation.x, trans.translation.y, yaw)

    def count_task2a_collisions(self):
        """62401738 いしい
        これまでに受け取った Task 2a の障害物との衝突メッセージの数を返す
        """
        return len([msg for msg in self.detection_list if "task2a" in msg])

    @staticmethod
    def stop_base():
        """62401738 いしい
        台車を止める
        """
        omni_base.cmd_vel(0.0, 0.0, 0.0)

    @classmethod
    def calc_base_velocity(cls, pose, target, yaw_deg=None, max_speed=None):
        """62401738 いしい
        今の姿勢 pose (x, y, yaw) から target (x, y) へ直進するための速度指令を計算する。
        - 位置: map 座標での誤差に比例した速度（最大 max_speed、省略時 TASK2A_MAX_SPEED）で、目標へ一直線に向かう
        - 向き: yaw_deg（省略時 TASK2A_YAW）を保つ（全方向台車なので、向きを変えずに横にも進める）
        map 座標の速度をロボット座標に回転して (vx, vy, w[deg/s]) を返す。
        """
        if yaw_deg is None:
            yaw_deg = cls.TASK2A_YAW
        if max_speed is None:
            max_speed = cls.TASK2A_MAX_SPEED
        err_x, err_y = target[0] - pose[0], target[1] - pose[1]
        dist = math.hypot(err_x, err_y)
        speed = min(max_speed, cls.TASK2A_GAIN * dist)
        map_vx = err_x / dist * speed if dist > 0 else 0.0
        map_vy = err_y / dist * speed if dist > 0 else 0.0
        cos_y, sin_y = math.cos(pose[2]), math.sin(pose[2])
        robot_vx = cos_y * map_vx + sin_y * map_vy
        robot_vy = -sin_y * map_vx + cos_y * map_vy
        yaw_err = math.atan2(math.sin(math.radians(yaw_deg) - pose[2]),
                             math.cos(math.radians(yaw_deg) - pose[2]))
        yaw_rate = max(-30.0, min(30.0, math.degrees(cls.TASK2A_YAW_GAIN * yaw_err)))
        return robot_vx, robot_vy, yaw_rate

    @classmethod
    def is_approaching_obstacle(cls, pose, target, obstacles):
        """62401738 いしい
        TASK2A_DANGER_DIST より近い障害物があり、さらに近づく方向へ進もうとしているかを判定する。
        （すでに近すぎるときでも、離れる方向へは動けるようにするため、距離が縮むかどうかを見る）
        """
        err_x, err_y = target[0] - pose[0], target[1] - pose[1]
        dist = math.hypot(err_x, err_y)
        if dist == 0:
            return False
        step = min(0.05, dist)
        next_x, next_y = pose[0] + err_x / dist * step, pose[1] + err_y / dist * step
        for obs_x, obs_y in obstacles:
            now = math.hypot(obs_x - pose[0], obs_y - pose[1])
            nxt = math.hypot(obs_x - next_x, obs_y - next_y)
            if nxt < getattr(cls, "task2a_danger", cls.TASK2A_DANGER_DIST) and nxt < now:
                return True
        return False

    def move_straight_to(self, target, obstacles, watch=True):
        """62401738 いしい
        tf で今の位置を見ながら、速度指令で target (x, y) まで直進する。
        watch=True のときは毎周期、次を監視して止まる。
          - Task 2a の衝突メッセージが新しく来た → "collision"
          - 覚えている障害物に TASK2A_DANGER_DIST より近づこうとした → "too_close"
        戻り値: "reached" / "collision" / "too_close" / "timeout" / "no_tf"
        """
        collisions = self.count_task2a_collisions()
        pose = self.get_robot_pose()
        if pose is None:
            return "no_tf"
        dist0 = math.hypot(target[0] - pose[0], target[1] - pose[1])
        end_time = rospy.Time.now() + rospy.Duration(dist0 / self.TASK2A_MAX_SPEED * 3.0 + 5.0)
        while not rospy.is_shutdown():
            pose = self.get_robot_pose()
            if pose is None:
                self.stop_base()
                return "no_tf"
            if watch and self.count_task2a_collisions() > collisions:
                self.stop_base()
                return "collision"
            if math.hypot(target[0] - pose[0], target[1] - pose[1]) < self.TASK2A_GOAL_TOL:
                self.stop_base()
                return "reached"
            if watch and self.is_approaching_obstacle(pose, target, obstacles):
                self.stop_base()
                return "too_close"
            if rospy.Time.now() > end_time:
                self.stop_base()
                return "timeout"
            omni_base.cmd_vel(*self.calc_base_velocity(pose, target))
            rospy.sleep(self.TASK2A_CONTROL_DT)
        self.stop_base()
        return "timeout"

    @classmethod
    def estimate_hit_position(cls, pose, direction):
        """62401738 いしい
        衝突した物体の位置を推定する。進行方向 direction（単位ベクトル）の前方、
        台車の縁（ROBOT_RADIUS）の少し先にあると考える。
        """
        dist = cls.ROBOT_RADIUS + 0.05
        return (pose[0] + direction[0] * dist, pose[1] + direction[1] * dist)

    @classmethod
    def add_hit_obstacle(cls, obstacles, hits, new_hit):
        """62401738 いしい
        踏んだ物体の推定位置を障害物に加える。推定は進行方向の前方と仮定しているのでずれるが、
        別の方向から同じ物体を踏むと、ずれ方が違う2つの推定が得られる。そこで HIT_MERGE_DIST 以内の
        推定どうしは平均して1つにまとめ、本当の位置に近づける（実在しない障害物で通路をふさがないため）。
        hits は推定位置だけのリスト。まとめた後の位置を返す。
        """
        for i, old in enumerate(hits):
            if math.hypot(old[0] - new_hit[0], old[1] - new_hit[1]) < cls.HIT_MERGE_DIST:
                merged = ((old[0] + new_hit[0]) / 2, (old[1] + new_hit[1]) / 2)
                hits[i] = merged
                if old in obstacles:
                    obstacles[obstacles.index(old)] = merged
                else:
                    obstacles.append(merged)
                return merged
        hits.append(new_hit)
        obstacles.append(new_hit)
        return new_hit

    def escape_from_obstacle(self, obstacles, safe_pos, direction, hits):
        """62401738 いしい
        障害物に触れてしまったときに抜け出す。
        1. 止まる（move_straight_to で止まっている）
        2. 触れた物体の位置を進行方向の前方と推定し、障害物として覚える（次の計画で避ける）
        3. 直前の安全だった位置 safe_pos へ下がる（2回目以降は進行方向と逆にさらに下がる）
        4. ESCAPE_WAIT 秒待って新しい衝突メッセージが来なければ脱出できたとみなす
        脱出できたら True を返す。
        """
        pose = self.get_robot_pose()
        if pose is None:
            return False
        hit = self.add_hit_obstacle(obstacles, hits, self.estimate_hit_position(pose, direction))
        rospy.logwarn("task2a: hit an obstacle near (%.2f, %.2f). escape.", hit[0], hit[1])
        back = safe_pos
        for attempt in range(self.ESCAPE_MAX_TRY):
            self.move_straight_to(back, obstacles, watch=False)
            before = self.count_task2a_collisions()
            rospy.sleep(self.ESCAPE_WAIT)
            if self.count_task2a_collisions() == before:
                rospy.loginfo("task2a: escaped (try %d)", attempt + 1)
                return True
            pose = self.get_robot_pose() or (back[0], back[1], 0.0)
            back = (pose[0] - direction[0] * self.ESCAPE_BACK_DIST,
                    pose[1] - direction[1] * self.ESCAPE_BACK_DIST)
        rospy.logwarn("task2a: could not escape")
        return False

    def execute_avoid_blocks(self):
        """62401738 いしい
        床の障害物を避けながら go_throw_2a まで進む。
        1回進むごとに検出し直し、見つけた障害物は map 座標で覚えておいて（近づいて見えなくなっても
        避ける）、tf で取った今の位置から A* で経路を計画し直す。
        移動は move_base を使わず、速度指令で A* の線分どおりに直進する（move_straight_to）。
          - 障害物に近づきすぎそうになったら止まって計画し直す
          - 触れてしまったら escape_from_obstacle で下がり、触れた位置を避けて計画し直す
        A* で経路が見つからないときは、3レーン方式（select_next_waypoint）で次の段へ進む。
        tf が取れないときは、今までどおり move_base（goto_pos）で動く。
        カメラは前しか見えないので、最初に首を左・正面・右に振って見渡し、
        横に動く前にはその方向を見てから計画し直す（横の物体の見落とし対策）。
        ゴール（ふさがれていれば find_free_goal で選んだ近くの点）に着いたら True を返す。
        """
        obstacles = []
        hits = []  # 踏んだ物体の推定位置
        current = self.TASK2A_START
        # 開始時: 足元（スタート直前の物体はふつうの下向きでは視野の手前に外れる）→ 左・右・正面
        whole_body.move_to_joint_positions(
            {"head_pan_joint": 0.0, "head_tilt_joint": self.TASK2A_NEAR_TILT})
        self.update_task2a_obstacles(obstacles)
        self.scan_task2a_obstacles(
            obstacles, (self.TASK2A_SCAN_PAN, -self.TASK2A_SCAN_PAN, 0.0))
        for move in range(self.TASK2A_MAX_MOVES):
            if move > 0:
                self.update_task2a_obstacles(obstacles)
            pose = self.get_robot_pose()
            if pose is not None:
                current = (pose[0], pose[1])
            goal = self.find_free_goal(obstacles)
            if math.hypot(current[0] - goal[0], current[1] - goal[1]) < 0.1:
                return True
            target = self.plan_next_point(current, goal, obstacles)
            pan = self.side_to_look(pose, target) if (pose and target) else None
            if pan is not None:
                self.scan_task2a_obstacles(obstacles, (pan, 0.0))
                goal = self.find_free_goal(obstacles)
                target = self.plan_next_point(current, goal, obstacles)
            if target is None:
                steps = [i for i, s_y in enumerate(self.TASK2A_STEPS_Y) if s_y > current[1] + 0.05]
                if not steps:
                    break
                target = tuple(self.select_next_waypoint(steps[0], obstacles, current)[:2])

            result = self.move_straight_to(target, obstacles) if pose is not None else "no_tf"
            rospy.loginfo("task2a move to (%.2f, %.2f): %s", target[0], target[1], result)
            if result == "collision":
                length = math.hypot(target[0] - current[0], target[1] - current[1]) or 1.0
                direction = ((target[0] - current[0]) / length, (target[1] - current[1]) / length)
                self.escape_from_obstacle(obstacles, current, direction, hits)
            elif result == "no_tf":
                self.goto_pos([target[0], target[1], self.TASK2A_YAW])
                current = target
        return False

    @classmethod
    def find_free_goal(cls, obstacles):
        """62401738 いしい
        Task 2a のゴール（go_throw_2a）が障害物の安全距離の内側にあると A* が経路を見つけられない。
        そのときはゴールと同じ y で左右に 0.1m ずつずらした点（最大 0.5m）から、空いている点を選ぶ。
        安全距離は広い順に試し、どれでも空いていなければ元のゴールを返す。
        """
        g_x, g_y = cls.TASK2A_GOAL
        offsets = [0.0] + [sign * k * 0.1 for k in range(1, 6) for sign in (-1, 1)]
        for safe_dist in cls.TASK2A_SAFE_DISTS:
            for off in offsets:
                cand = (g_x + off, g_y)
                if not cls.TASK2A_PLAN_X[0] <= cand[0] <= cls.TASK2A_PLAN_X[1]:
                    continue
                if all(math.hypot(cand[0] - o_x, cand[1] - o_y) >= safe_dist
                       for o_x, o_y in obstacles):
                    return cand
        return cls.TASK2A_GOAL

    @classmethod
    def select_next_waypoint(cls, current_stp, obstacles, current_pos):
        """62401738 いしい
        次の段（y = TASK2A_STEPS_Y[current_stp]）のどのレーンへ進むかを決める。
        現在地から各レーンへの移動経路（線分）と障害物の距離を計算し、
          1. 距離が TASK2A_SAFE_DIST 以上のレーンのうち、横移動が最も少ないもの
          2. 安全なレーンがなければ、障害物から最も離れられるレーン
        を選ぶ。obstacles は (x, y) のリスト、戻り値は [x, y, yaw]。
        """
        target_y = cls.TASK2A_STEPS_Y[current_stp]
        lanes = []
        for lane_x in cls.TASK2A_LANES_X:
            goal = (lane_x, target_y)
            dists = [cls.dist_point_to_segment(obs, current_pos, goal) for obs in obstacles]
            clearance = min(dists) if dists else float("inf")
            lanes.append({"x": lane_x, "clearance": clearance,
                          "lateral": abs(lane_x - current_pos[0])})
            rospy.loginfo("step %d lane x=%.2f clearance=%.2f", current_stp, lane_x, clearance)

        safe = [lane for lane in lanes if lane["clearance"] >= cls.TASK2A_SAFE_DIST]
        if safe:
            best = min(safe, key=lambda lane: (lane["lateral"], -lane["clearance"]))
        else:
            rospy.logwarn("no safe lane at step %d. choose the widest one.", current_stp)
            best = max(lanes, key=lambda lane: lane["clearance"])
        rospy.loginfo("select next waypoint x=%.2f y=%.2f", best["x"], target_y)
        return [best["x"], target_y, cls.TASK2A_YAW]

    # 障害物を持って退かす（Task 2a）。把持している物体との接触は衝突にならない
    # week1 と同じ "avoid" に戻した（"grasp" は 11/9 の実行で wrc_frame にぶつかった）
    TASK2A_MODE = "avoid"                 # "grasp": 持って退かす / "avoid": A* で避ける（week1）
    TASK2A_GRASP_STANDOFF = 0.45          # 障害物の手前で止まる距離 [m]（> TASK2A_DANGER_DIST）
    TASK2A_GRASP_MAX = 5                  # 退かす障害物の数の上限
    TASK2A_MATCH_DIST = 0.25              # 近づいて見直したとき、同じ障害物とみなす距離 [m]
    TASK2A_DROP_POS = [2.5, 1.45, -90]    # 退かした障害物を置く場所（standby_2a の後ろ。後で通らない）

    def find_nearest_obstacle(self, pos, max_dist):
        """62401738 いしい
        近くで検出し直し、pos (x, y) から max_dist 以内で一番近い床の物体の (位置, ラベル) を返す。
        """
        best, best_dist = None, max_dist
        for bbox in self.get_stable_detection():
            obj = self.get_bbox_position(bbox)
            if obj is None or obj.z > self.TASK2A_AREA["z_max"]:
                continue
            dist = math.hypot(obj.x - pos[0], obj.y - pos[1])
            if dist < best_dist:
                best, best_dist = (obj, bbox.label), dist
        return best

    def remove_obstacle(self, obstacles):
        """62401738 いしい
        obstacles の中で一番手前（y が小さい）の障害物に近づいて持ち上げ、TASK2A_DROP_POS に置く。
        持ち上げたら obstacles から消して True を返す。
        """
        goal_y = self.TASK2A_GOAL[1]
        ahead = [obs for obs in obstacles if obs[1] < goal_y + 0.3]
        if not ahead:
            return False
        target = min(ahead, key=lambda obs: obs[1])
        others = [obs for obs in obstacles if obs != target]
        standoff = (target[0], target[1] - self.TASK2A_GRASP_STANDOFF)
        result = self.move_straight_to(standoff, others)
        rospy.loginfo("task2a: approach obstacle (%.2f, %.2f): %s", target[0], target[1], result)
        if result not in ("reached", "no_tf"):
            return False

        self.change_pose("look_at_near_floor")
        found = self.find_nearest_obstacle(target, self.TASK2A_MATCH_DIST)
        if found is None:
            rospy.logwarn("task2a: obstacle disappeared at (%.2f, %.2f)", target[0], target[1])
            obstacles.remove(target)
            return True
        grasp_pos, label = found
        rospy.loginfo("task2a: grasp %s (%.2f, %.2f, %.2f)",
                      label, grasp_pos.x, grasp_pos.y, grasp_pos.z)
        if not self.grasp_from_upper_side(grasp_pos):
            return False
        self.change_pose("all_neutral")

        # 後ろ（もう通らない場所）に置いてから、元の位置に戻る
        self.goto_pos(self.TASK2A_DROP_POS)
        self.change_pose("put_in_bin")
        gripper.command(1)
        self.change_pose("all_neutral")
        obstacles.remove(target)
        self.change_pose("look_at_near_floor")
        self.goto_name("standby_2a")
        return True

    def execute_clear_blocks(self):
        """62401738 いしい
        床の障害物を手前から順に持ち上げて退かし、何もなくなったら go_throw_2a まで直進する。
        退かせなかったときは False を返す（execute_task2a が A* の回避に切り替える）。
        """
        obstacles = []
        whole_body.move_to_joint_positions(
            {"head_pan_joint": 0.0, "head_tilt_joint": self.TASK2A_NEAR_TILT})
        self.update_task2a_obstacles(obstacles)
        self.scan_task2a_obstacles(
            obstacles, (self.TASK2A_SCAN_PAN, -self.TASK2A_SCAN_PAN, 0.0))
        for _ in range(self.TASK2A_GRASP_MAX):
            if not obstacles:
                break
            if not self.remove_obstacle(obstacles):
                return False
            # 退かした後に見え方が変わるので見渡し直す
            obstacles = []
            self.scan_task2a_obstacles(
                obstacles, (self.TASK2A_SCAN_PAN, -self.TASK2A_SCAN_PAN, 0.0))
        if obstacles:
            return False
        result = self.move_straight_to(self.TASK2A_GOAL, obstacles)
        rospy.loginfo("task2a: go to goal: %s", result)
        return result == "reached"

    def execute_task2a(self):
        """62401738 いしい
        task2aを実行する。TASK2A_MODE が "grasp" なら障害物を持って退かし、
        退かせなかったときだけ A* で避ける（execute_avoid_blocks）。
        """
        rospy.loginfo("#### start Task 2a ####  detector: %s", self.get_detector_type())
        self.load_task2a_walls()
        self.change_pose("look_at_near_floor")
        gripper.command(0)
        self.goto_name("standby_2a")
        log_start = len(self.detection_list)

        reached = False
        if self.TASK2A_MODE == "grasp":
            reached = self.execute_clear_blocks()
            if not reached:
                rospy.logwarn("task2a: could not clear the obstacles. avoid them with A*.")
                self.change_pose("look_at_near_floor")
        if not reached:
            # 落ちているブロックを避けて移動
            reached = self.execute_avoid_blocks()

        # 回避の移動でゴールに着けなかったときだけ move_base でゴールへ向かう
        # （着いているのに move_base を使うと、床の物体が見えない move_base が近くの物体に寄ることがある）
        if not reached:
            if self.goto_name("go_throw_2a"):
                rospy.loginfo("task2a: reached the goal with the move_base fallback")
            else:
                rospy.logerr("task2a: both the avoidance and the move_base fallback failed")
        whole_body.move_to_go()

        # 障害物に触れたかを確認する（触れると 2a は0点）
        hits = [msg for msg in self.detection_list[log_start:] if "task2a" in msg]
        if hits:
            rospy.logwarn("task2a collision: %s", ", ".join(hits))
        else:
            rospy.loginfo("task2a: no collision")

    def execute_task2b(self):
        """62401738 いしい
        task2bを実行する。指示文は棚の前に行くと届くので、棚へ移動してから待つ。
        """
        rospy.loginfo("#### start Task 2b ####")
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")

        # 命令文を取得して解釈
        instruction = self.wait_for_instruction(self.TASK2B_WAIT_SEC)
        if instruction is None:
            rospy.logwarn("no instruction received. deliver any object to the default person.")
            target_obj, target_person = None, None
        else:
            rospy.loginfo("recieved instruction: %s", instruction)
            target_obj, target_person = self.extract_target_obj_and_person(instruction)

        # 指定したオブジェクトを指定した配達先へ（見つからなくても何かを届けて20点を狙う）
        self.deliver_to_target(target_obj, target_person)

    REQUIRED_CONTROLLERS = ("arm_trajectory_controller", "head_trajectory_controller")

    def wait_for_controllers(self, timeout=120.0):
        """62401738 いしい
        シミュレータのコントローラ（腕・頭）が running になるまで待つ。
        起動直後や起動に失敗したシミュレータでは、指令を出しても腕も台車も動かない
        （MoveIt は CONTROL_FAILED、move_base は計画できても動かない）ので、先に確認する。
        """
        from controller_manager_msgs.srv import ListControllers

        name = "/hsrb/controller_manager/list_controllers"
        deadline = time.monotonic() + timeout
        missing = list(self.REQUIRED_CONTROLLERS)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                rospy.wait_for_service(name, timeout=5.0)
                states = {c.name: c.state for c in rospy.ServiceProxy(name, ListControllers)().controller}
                missing = [c for c in self.REQUIRED_CONTROLLERS if states.get(c) != "running"]
                if not missing:
                    if states.get("omni_base_controller", "running") != "running":
                        rospy.logwarn("omni_base_controller is %s: the base may not move",
                                      states.get("omni_base_controller"))
                    rospy.loginfo("controllers are running")
                    return True
                rospy.loginfo("waiting for controllers: %s",
                              ", ".join("{}={}".format(c, states.get(c, "none")) for c in missing))
            except (rospy.ROSException, rospy.ServiceException) as error:
                rospy.logwarn("cannot list controllers: %s", error)
            time.sleep(3.0)
        rospy.logerr("controllers did not start (%s). The robot cannot move: "
                     "restart the simulator.", ", ".join(missing))
        return False

    def run(self):
        """62418450 向井悠人
        全てのタスクを実行する（Task 1 → Task 2a → Task 2b）。
        Task 2b の指示文は棚の前に行くと届くので、2a を先に実行する。
        """
        if not self.wait_for_controllers():
            return  # 腕も台車も動かないので、何もせずに終わる（シミュレータを再起動する）
        self.run_start = rospy.Time.now()  # 時間配分（引き出し・Task 1 の打ち切り）はここから数える
        self.change_pose("all_neutral")
        self.prescan_containers()  # 引き出しを開ける前に容器を見ておく（開けた後だと写り込む）
        self.open_drawers()  # 手が空のうちに引き出しを開ける
        self.execute_task1()
        self.execute_task2a()
        self.execute_task2b()


def main():
    """
    WRS環境内でタスクを実行するためのメインノードを起動する
    """
    rospy.init_node('main_controller')
    try:
        ctrl = WrsMainController()
        rospy.loginfo("node initialized [%s]", rospy.get_name())

        # タスクの実行モードを確認する
        if rospy.get_param("~test_mode", default=False) is True:
            rospy.loginfo("#### start with TEST mode. ####")
            ctrl.check_positions()
        else:
            rospy.loginfo("#### start with NORMAL mode. ####")
            ctrl.run()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
