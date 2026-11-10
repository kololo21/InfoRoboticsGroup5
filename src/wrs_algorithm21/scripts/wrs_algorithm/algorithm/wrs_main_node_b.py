#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
from collections import deque
import inspect
import json
import os
import re
import struct
import traceback
import time
import threading
import actionlib
import numpy as np
import rospy
import rosgraph
import rospkg
import tf2_ros
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseGoal
from nav_msgs.srv import GetMap, GetPlan
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from tf.transformations import quaternion_matrix
from actionlib_msgs.msg import GoalStatus
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper
from wrs_algorithm.util.mathematics import quaternion_from_euler
import math
import copy
import heapq


class NavigationUnconfirmedError(RuntimeError):
    """move_baseのゴール終了を確認できず、台車制御を引き継げない。"""


class ManipulationUnconfirmedError(RuntimeError):
    """把持指令を中止できたか確認できない。"""


class SimulationClockUnavailableError(RuntimeError):
    """シミュレーション時刻が停止または巻き戻ったため、待機後の動作へ進めない。"""


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
    GRASP_MAX_GAP = 0.12
    floor_z_offset = 0.0
    DETECT_CNT = 1
    TROFAST_Y_OFFSET = 0.2
    NAVIGATION_TERMINAL_STATES = (
        GoalStatus.PREEMPTED, GoalStatus.SUCCEEDED, GoalStatus.ABORTED,
        GoalStatus.REJECTED, GoalStatus.RECALLED)

    def __init__(self):
        """62418450 向井悠人
        設定、深度の履歴、把持状態と台車制御を初期化する。
        """
        # 変数の初期化
        self.instruction_list = []
        self.instruction_cursor = 0
        self.detection_list = []
        self.navigation_pending = False
        self.navigation_needs_recovery = False
        self.navigation_checkpoint = None
        self.query_jobs = {}
        self.depth_clouds = deque(maxlen=8)
        self.hand_state = "empty"
        self.arm_ready = True
        self.held_geometry = None
        self.distinct_pick_pairs = set()
        self.transport_loss = None
        self.transport_empty_stamp = None
        self.transport_unknown_since = None
        self.recovered_position = None
        self.floor_obstacles = set()
        self.floor_clearance = {}
        self.wall_map = None

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses = self.load_json(self.get_path(["config", "poses.json"]))
        self.load_placement_overrides()
        self.validate_runtime_settings()

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.depth_sub = rospy.Subscriber(rospy.get_param(
            "~placement_cloud_topic", "/hsrb/head_rgbd_sensor/depth_registered/rectified_points"),
            PointCloud2, self.depth_clouds.append, queue_size=1)

        self.instruction_sub = rospy.Subscriber(
            "/message", String, self.instruction_cb, queue_size=10)
        self.detection_sub = rospy.Subscriber(
            "/detect_msg", String, self.detection_cb, queue_size=10)

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

    def detection_cb(self, msg):
        """サンプルのまま（担当なし）
        検出結果を受信する
        """
        rospy.loginfo("received [Collision detected with %s]", msg.data)
        self.detection_list.append(msg.data)

    def lookup_transform(self, target, source, stamp, timeout):
        """62418450 向井悠人
        シミュレーション時刻が停止しても、TFの取得を実時間で打ち切る。
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.tf_buffer.lookup_transform(target, source, stamp, rospy.Duration(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or rospy.is_shutdown():
                    raise
                time.sleep(min(0.03, remaining))

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
        """62418450 向井悠人
        登録された地点へ台車を移動し、到達できたかを返す。
        """
        pos = self.coordinates["positions"].get(name)
        if pos is None:
            rospy.logerr("unknown waypoint name [%s]", name)
            return False
        return self.goto_pos(pos, name)

    def goto_pos(self, pos, name="raw_pos"):
        """62418450 向井悠人
        地図上の座標と向きへ、通常移動用の制限時間で台車を動かす。
        """
        return self.navigate_to_pose(pos, name, "~navigation_timeout")

    def bounded_query(self, key, operation, timeout, retry_readonly=False, endpoint=None):
        """62418450 向井悠人
        実時間で待機を区切り、読み取りに限り未応答の要求を1件残して再接続できる。
        """
        pending = [job for job in self.query_jobs.get(key, ()) if not job["done"].is_set()]
        if pending and (not retry_readonly or len(pending) >= 2 or not pending[0]["timed_out"]):
            raise TimeoutError("{} is still waiting for a previous response".format(key))
        job = {"done": threading.Event(), "timed_out": False, "endpoint": endpoint}
        self.query_jobs[key] = pending + [job]
        threading.Thread(target=self.run_query, args=(job, operation), daemon=True).start()
        if not job["done"].wait(timeout):
            job["timed_out"] = True
            raise TimeoutError("{} did not respond within {:.1f}s".format(key, timeout))
        if "error" in job:
            raise job["error"]
        return job["result"]

    @staticmethod
    def run_query(job, operation):
        """62418450 向井悠人
        問い合わせ結果または例外を記録し、待機中の呼び出し元へ知らせる。
        """
        try:
            job["result"] = operation()
        except Exception as error:
            job["error"] = error
        finally:
            job["done"].set()

    @classmethod
    def has_action_result(cls, client, state):
        """62418450 向井悠人
        終端の状態通知と実際の結果メッセージがそろっているかを判定する。
        """
        return state in cls.NAVIGATION_TERMINAL_STATES and client.get_result() is not None

    @classmethod
    def wait_action_terminal(cls, client, timeout):
        """62418450 向井悠人
        実時間の上限内でアクションの終端状態と結果メッセージの両方を確認する。
        """
        deadline = time.monotonic() + timeout
        while True:
            state = client.get_state()
            if cls.has_action_result(client, state):
                return state
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.05, remaining))

    def cancel_navigation(self, name):
        """62418450 向井悠人
        例外経路を含めて台車ゴールを取り消し、終了未確定なら動作を中断する。
        """
        try:
            omni_base.NAVCLIENT.cancel_goal()
        except Exception as error:
            rospy.logerr("Could not request cancellation for %s: %s", name, error)
        self.confirm_navigation_finished()

    def confirm_navigation_finished(self):
        """62418450 向井悠人
        前の台車ゴールの終了を実時間で確認し、未確定なら次の指令を禁止する。
        """
        if not self.navigation_pending:
            return
        client = omni_base.NAVCLIENT
        try:
            state = self.wait_action_terminal(client, 2.0)
        except Exception as error:
            raise NavigationUnconfirmedError("Cannot read navigation result: {}".format(error)) from error
        if state not in self.NAVIGATION_TERMINAL_STATES:
            raise NavigationUnconfirmedError("Previous navigation still unresolved: state={}".format(state))
        self.navigation_pending = False
        rospy.loginfo("Previous navigation ended: state=%s", state)

    def navigate_to_pose(self, pos, name, timeout_param, corridor=None):
        """62418450 向井悠人
        台車を動かし、走行監視の失敗や例外でも取り消しを確認してから制御を返す。
        """
        timeout = self.runtime_number(timeout_param, 60.0, positive=True)
        self.confirm_navigation_finished()
        client = omni_base.NAVCLIENT
        try:
            ready = self.bounded_query("move_base server", lambda: client.wait_for_server(
                rospy.Duration(2.0)), 2.0)
        except (TimeoutError, rospy.ROSException) as error:
            rospy.logwarn("Navigation server unavailable for %s: %s", name, error)
            return False
        if not ready:
            rospy.logerr("Navigation server unavailable for %s", name)
            return False

        x_pos, y_pos, yaw = pos
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x_pos
        goal.target_pose.pose.position.y = y_pos
        goal.target_pose.pose.orientation = quaternion_from_euler(yaw, 0, 0)
        rospy.loginfo("Navigation %s: x=%.2f y=%.2f yaw=%.1f", name, x_pos, y_pos, yaw)

        self.navigation_pending = True
        try:
            client.send_goal(goal)
            result = self.await_base_result(timeout, corridor)
            state = client.get_state() if result == "done" else None
            if not self.has_action_result(client, state):
                state = None
        except Exception as error:
            self.cancel_navigation(name)
            self.navigation_needs_recovery = self.transport_loss is None
            rospy.logerr("Navigation %s aborted after %s: %s", name, type(error).__name__, error)
            return False

        if state not in self.NAVIGATION_TERMINAL_STATES:
            self.cancel_navigation(name)
            self.navigation_needs_recovery = self.transport_loss is None
            rospy.logwarn("Navigation %s interrupted: %s (state=%s)", name, result, state)
            return False
        self.navigation_pending = False
        if state != GoalStatus.SUCCEEDED:
            self.navigation_needs_recovery = True
            rospy.logerr("Navigation %s failed: state=%s", name, state)
            return False
        rospy.loginfo("Navigation reached %s", name)
        return True

    def change_pose(self, name):
        """サンプルのまま（担当なし）
        指定された姿勢名に遷移する
        """
        if name in self.poses.keys():
            rospy.loginfo("change pose to [%s]", name)
            return whole_body.move_to_joint_positions(self.poses[name])

        rospy.logerr("unknown pose name [%s]", name)
        return False

    def check_positions(self):
        """サンプルのまま（担当なし）
        読み込んだ座標ファイルの座標を巡回する
        """
        whole_body.move_to_go()
        for wp_name in self.coordinates["routes"]["test"]:
            self.goto_name(wp_name)
            rospy.sleep(1)

    @staticmethod
    def request_detection(client):
        """62418450 向井悠人
        渡された新規接続を使ってRGB画像の検出を要求する。
        """
        return client(GetObjectDetectionRequest())

    def execute_moveit(self, commander, operation, name):
        """62418450 向井悠人
        MoveItの実行を実時間で監視し、停止後に応答を確認できなければ後続動作を禁止する。
        """
        self.confirm_navigation_finished()
        timeout = self.runtime_number("~motion_timeout", 35.0, positive=True)
        job = {"done": threading.Event()}
        threading.Thread(target=self.run_query, args=(job, operation), daemon=True).start()
        finished = job["done"].wait(timeout)
        try:
            self.bounded_query("MoveIt stop", commander.stop, 2.0)
        except Exception as error:
            raise ManipulationUnconfirmedError("Cannot stop {}: {}".format(name, error)) from error
        if not job["done"].wait(2.0) or "error" in job:
            detail = job.get("error", "no terminal response")
            raise ManipulationUnconfirmedError("{}: {}".format(name, detail))
        if not finished:
            rospy.logwarn("%s exceeded %.1fs and stopped", name, timeout)
            return False
        return bool(job["result"])

    def wait_for_simulation(self, seconds, name):
        """62418450 向井悠人
        指定のシミュレーション時間を待ち、時計の停止や巻き戻りを検出する。
        """
        previous = rospy.Time.now()
        target = previous + rospy.Duration(seconds)
        stalled_since = time.monotonic()
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            if stamp >= target:
                return
            if stamp < previous:
                raise SimulationClockUnavailableError("{}: clock moved backwards".format(name))
            if stamp > previous:
                previous = stamp
                stalled_since = time.monotonic()
            if time.monotonic() - stalled_since >= 20.0:
                raise SimulationClockUnavailableError("{}: clock did not advance for 20s".format(name))
            time.sleep(0.1)
        raise rospy.ROSInterruptException("Interrupted while waiting for {}".format(name))

    def open_gripper(self):
        """62418450 向井悠人
        指を開くMoveIt指令を監視し、実行結果を確認する。
        """
        commander = gripper.GRIPPER_CMD
        try:
            commander.set_joint_value_target("hand_motor_joint", 1)
        except Exception as error:
            rospy.logwarn("Cannot set gripper opening: %s", error)
            return False
        return self.execute_moveit(commander, lambda: commander.go(wait=True), "open gripper")

    def point_head(self, pan, tilt):
        """62418450 向井悠人
        首の目標を設定し、実行の結果を確認してから観測へ進む。
        """
        head = whole_body.HEAD_CMD
        try:
            head.set_joint_value_target({"head_pan_joint": pan, "head_tilt_joint": tilt})
        except Exception as error:
            rospy.logwarn("Cannot set head target: %s", error)
            return False
        return self.execute_moveit(head, lambda: head.go(wait=True), "head motion")

    def measure_object(self, bbox, detection, floor=False, min_height=None):
        """62418450 向井悠人
        RGBと深度を照合し、床・机・保持物の高さ条件に合う点群を求める。
        """
        deadline = time.monotonic() + 0.5
        stamp = detection.header.stamp
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            clouds = self.depth_clouds.copy()
            if clouds:
                cloud = min(clouds, key=lambda item: abs((item.header.stamp - stamp).to_sec()))
                if abs((cloud.header.stamp - stamp).to_sec()) <= 0.10:
                    break
            time.sleep(0.03)
        else:
            return "measurement_failed", None
        if not cloud.header.frame_id or cloud.height <= 1:
            return "measurement_failed", None

        pixels = np.asarray([
            (u, v) for v in range(max(0, bbox.y), min(cloud.height, bbox.y + bbox.h), 2)
            for u in range(max(0, bbox.x), min(cloud.width, bbox.x + bbox.w), 2)
        ])
        if not len(pixels):
            return "measurement_failed", None

        other_boxes = [other for other in detection.bboxes
                       if other is not bbox and self.bbox_iou(bbox, other) > 0
                       and (other.label != bbox.label or self.bbox_iou(bbox, other) < 0.85)]
        keep = np.ones(len(pixels), dtype=bool)
        u, v = pixels.T
        for other in other_boxes:
            keep &= ~((other.x <= u) & (u < other.x + other.w)
                      & (other.y <= v) & (v < other.y + other.h))
        pixels = pixels[keep]
        if len(pixels) < 15:
            return "measurement_failed", None

        uvs = [tuple(pixel) for pixel in pixels.tolist()]
        try:
            points = self.decode_depth_points(cloud, uvs, skip_nans=False)
        except ValueError as error:
            rospy.logwarn("Cannot measure %s: %s", bbox.label, error)
            return "measurement_failed", None
        if points.shape != (len(pixels), 3):
            return "measurement_failed", None
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        points, pixels = points[valid], pixels[valid]
        if len(points) < 15:
            return "measurement_failed", None

        try:
            transform = self.lookup_transform(
                "map", cloud.header.frame_id, cloud.header.stamp, 1.0).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn("Cannot measure %s: %s", bbox.label, error)
            return "measurement_failed", None
        rotation, translation = self.transform_arrays(transform)
        world = points.dot(rotation.T) + translation
        height = world[:, 2]
        if min_height is not None:
            valid = height >= min_height
            other_height = height < min_height
        else:
            height = self.floor_relative_height(world)
            valid = ((height > 0.012) & (height < 0.35)) if floor else height >= 0.32
            other_height = height >= 0.35 if floor else (height > 0.012) & (height < 0.32)
        if np.all(other_height):
            return "outside_height", None
        world, depths, pixels = world[valid], points[valid, 2], pixels[valid]
        if len(world) < 15:
            return "measurement_failed", None

        middle = np.array([bbox.x + bbox.w / 2, bbox.y + bbox.h / 2])
        half_core = 0.28 * np.array([bbox.w, bbox.h])
        core = np.all(np.abs(pixels - middle) <= half_core, axis=1)
        order = np.argsort(depths)
        groups = np.split(order, np.flatnonzero(np.diff(depths[order]) > 0.035) + 1)
        ranked = sorted(
            ((int(np.count_nonzero(core[group])), len(group), index)
             for index, group in enumerate(groups)
             if len(group) >= max(15, len(world) * 0.12)), reverse=True)
        if (not ranked or ranked[0][0] < 4
                or (len(ranked) > 1 and ranked[1][0] >= 0.65 * ranked[0][0])):
            rospy.logwarn("Cannot associate depth confidently with %s", bbox.label)
            return "measurement_failed", None

        points = world[groups[ranked[0][2]]]
        low, high = np.percentile(points, [3, 97], axis=0)
        return "measured", {"center": (low + high) / 2,
                            "size": np.maximum(high - low, 0.015), "points": points}

    @staticmethod
    def transform_arrays(transform):
        """62418450 向井悠人
        TFの回転と平行移動を点群計算に使う配列に変換する。
        """
        q, t = transform.rotation, transform.translation
        return quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3], np.array([t.x, t.y, t.z])

    @staticmethod
    def fresh_tf_stamp(stamp):
        """62418450 向井悠人
        最新値を要求したTFが、現在のシミュレーション時刻に十分近いか判定する。
        """
        age = (rospy.Time.now() - stamp).to_sec()
        return math.isfinite(age) and abs(age) <= 1.0

    def finger_gap(self, timeout=1.0, stamped=False):
        """62418450 向井悠人
        指先の距離をTFから求め、搬送中の確認には計測時刻も返す。
        """
        try:
            transform = self.lookup_transform(
                "hand_l_finger_tip_frame", "hand_r_finger_tip_frame", rospy.Time(0), timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        if not self.fresh_tf_stamp(transform.header.stamp):
            return None
        pos = transform.transform.translation
        gap = math.sqrt(pos.x ** 2 + pos.y ** 2 + pos.z ** 2)
        if not math.isfinite(gap):
            return None
        return (gap, transform.header.stamp) if stamped else gap

    def verify_holding(self, after=None):
        """62418450 向井悠人
        指先間隔から保持を判定し、動作後の確認では指定時刻より新しいTFを待つ。
        """
        empty_gap = self.runtime_number("~empty_finger_gap", 0.015)
        if after is None:
            gap = self.finger_gap()
        else:
            gap = None
            deadline = time.monotonic() + 1.0
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                sample = self.finger_gap(timeout=0.0, stamped=True)
                if sample is not None and sample[1] > after:
                    gap = sample[0]
                    if empty_gap < gap <= self.GRASP_MAX_GAP:
                        break
                time.sleep(0.03)
        holding = gap is not None and empty_gap < gap <= self.GRASP_MAX_GAP
        rospy.loginfo("Grasp gap: %s m, holding=%s", gap, holding)
        return holding

    def close_for_grasp(self):
        """62418450 向井悠人
        把持の完了を実時間で待ち、未完了時は取り消しと終端状態を確認する。
        """
        from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortGoal

        client = actionlib.SimpleActionClient(
            "/hsrb/gripper_controller/grasp", GripperApplyEffortAction)
        try:
            ready = self.bounded_query("grasp server", lambda: client.wait_for_server(
                rospy.Duration(3.0)), 3.0)
        except (TimeoutError, rospy.ROSException) as error:
            rospy.logwarn("Gripper effort controller is unavailable: %s", error)
            return False
        if not ready:
            rospy.logwarn("Gripper effort controller is unavailable")
            return False
        self.hand_state = "unknown"
        try:
            client.send_goal(GripperApplyEffortGoal(effort=-0.5))
            state = self.wait_action_terminal(client, 5.0)
            if state in self.NAVIGATION_TERMINAL_STATES:
                return state == GoalStatus.SUCCEEDED
        except Exception as error:
            rospy.logwarn("Grasp action failed: %s", error)
        try:
            client.cancel_goal()
            if self.wait_action_terminal(client, 2.0) in self.NAVIGATION_TERMINAL_STATES:
                return False
        except Exception as error:
            raise ManipulationUnconfirmedError("Cannot confirm gripper cancellation") from error
        raise ManipulationUnconfirmedError("Cannot confirm gripper cancellation")

    def hand_pose(self, timeout=1.0):
        """62418450 向井悠人
        指定した時間内で新しい手のひらの地図座標と向きを取得する。
        """
        transform = self.lookup_transform("map", "hand_palm_link", rospy.Time(0), timeout)
        if not self.fresh_tf_stamp(transform.header.stamp):
            raise tf2_ros.ExtrapolationException("Hand TF is out of date")
        hand = transform.transform
        t, q = hand.translation, hand.rotation
        if (not all(math.isfinite(value) for value in (t.x, t.y, t.z))
                or not 0.5 <= math.hypot(q.x, q.y, q.z, q.w) <= 1.5):
            raise tf2_ros.ExtrapolationException("Hand TF contains an invalid pose")
        return hand

    @staticmethod
    def hand_path_keeps_base_still(trajectory):
        """62418450 向井悠人
        全身の計画に台車の移動が含まれていれば実行を拒否する。
        """
        arm_joints = set(whole_body.ARM_CMD.get_active_joints())
        joints = trajectory.joint_trajectory
        if not joints.points:
            return False
        start = joints.points[0].positions
        for index, name in enumerate(joints.joint_names):
            if name in arm_joints:
                continue
            if any(abs(point.positions[index] - start[index]) > 0.025 for point in joints.points):
                rospy.logwarn("Hand trajectory would move %s", name)
                return False
        base_path = trajectory.multi_dof_joint_trajectory.points
        if not base_path:
            return True
        initial = base_path[0].transforms
        for point in base_path:
            for first, current in zip(initial, point.transforms):
                a, b = first.translation, current.translation
                p, q = first.rotation, current.rotation
                dot = abs(p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w)
                if math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) > 0.025 or dot < 0.9998:
                    rospy.logwarn("Hand trajectory would move the floating base")
                    return False
        return True

    def plan_hand_motion(self, commander, pose, linear):
        """62418450 向井悠人
        手先の軌道を計画し、終了後にMoveItの座標系と目標設定を元に戻す。
        """
        previous_frame = None
        try:
            if linear:
                previous_frame = commander.get_pose_reference_frame()
                commander.set_pose_reference_frame("map")
                cartesian = commander.compute_cartesian_path
                kwargs = {"avoid_collisions": True}
                if "jump_threshold" in inspect.signature(cartesian).parameters:
                    kwargs["jump_threshold"] = 0.0
                trajectory, fraction = cartesian([pose.pose], 0.01, **kwargs)
                if not math.isfinite(fraction) or fraction < 1.0:
                    rospy.logwarn("Incomplete hand path: %.3f", fraction)
                    return None
            else:
                commander.set_pose_target(pose)
                plan = commander.plan()
                ok, trajectory = plan[:2] if isinstance(plan, tuple) else (True, plan)
                if not ok:
                    return None
            return trajectory if self.hand_path_keeps_base_still(trajectory) else None
        finally:
            if previous_frame is not None:
                commander.set_pose_reference_frame(previous_frame)
            if not linear:
                commander.clear_pose_targets()

    def move_hand(self, xyz, orientation, linear=False):
        """62418450 向井悠人
        台車が動かない手先軌道を時間内に計画し、実行終了を確認する。
        """
        self.confirm_navigation_finished()
        commander = whole_body.WHOLE_BODY_CMD
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = xyz
        pose.pose.orientation = orientation
        timeout = self.runtime_number("~motion_timeout", 35.0, positive=True)
        try:
            trajectory = self.bounded_query("hand planner", lambda: self.plan_hand_motion(
                commander, pose, linear), timeout)
        except Exception as error:
            rospy.logwarn("Cannot plan hand motion: %s", error)
            return False
        if trajectory is None:
            return False
        return self.execute_moveit(
            commander, lambda: commander.execute(trajectory, wait=True), "hand motion")

    def return_grasp_to_surface(self, contact, approach, orientation):
        """62418450 向井悠人
        保持が不明な把持を元の支持面へ戻し、開放してから腕を引く。
        """
        if self.hand_state != "empty":
            if not self.move_hand(contact, orientation, linear=True):
                return False
            if not self.release_hand():
                return False
        return self.move_hand(approach, orientation, linear=True)

    def release_hand(self):
        """62418450 向井悠人
        開放指令が終わった時点で手を空と記録し、その後に安定を待つ。
        """
        self.hand_state = "unknown"
        if not self.open_gripper():
            return False
        self.hand_state = "empty"
        self.held_geometry = None
        self.wait_for_simulation(6.0, "gripper settling after release")
        return True

    @staticmethod
    def grasp_planar_axis(points):
        """62418450 向井悠人
        XY点群が細長く、向きが十分に安定する場合だけ長手方向を返す。
        """
        xy = points[:, :2]
        centered = xy - np.mean(xy, axis=0)
        values, axes = np.linalg.eigh(centered.T.dot(centered) / len(xy))
        if values[1] < 2.5 * max(values[0], 1e-8):
            return None
        return axes[:, 1]

    @staticmethod
    def utensil_handle(points, axis):
        """62418450 向井悠人
        細長い器具の両端寄りを比べ、点数が十分な細い側を柄の候補にする。
        """
        along = points[:, :2].dot(axis)
        across = np.array([-axis[1], axis[0]])
        low, high = np.percentile(along, [8, 92])
        length = high - low
        if length < 0.055:
            return None
        sections = []
        for fraction in (0.27, 0.73):
            band = points[np.abs(along - (low + fraction * length)) <= 0.13 * length]
            if len(band) < 6:
                return None
            width = float(np.ptp(np.percentile(band[:, :2].dot(across), [10, 90])))
            sections.append((width, np.median(band, axis=0)))
        sections.sort(key=lambda section: section[0])
        if sections[0][0] <= 0.8 * sections[1][0]:
            return sections[0][1]
        return None

    @classmethod
    def grasp_body_height(cls, points, direction, preferred=None):
        """62418450 向井悠人
        観測幅が指の上限に収まる胴の高さと、全候補を測れたかを返す。
        """
        low, high = np.percentile(points[:, 2], [5, 95])
        height = high - low
        if height < 0.05:
            return None, False
        across = np.array([-direction[1], direction[0]])
        sections = []
        levels = [low + fraction * height for fraction in (0.45, 0.35, 0.60)]
        if preferred is not None:
            levels.insert(0, preferred)
        for level in levels:
            band = points[np.abs(points[:, 2] - level) <= 0.10 * height]
            if len(band) >= 6:
                width = float(np.ptp(np.percentile(band[:, :2].dot(across), [10, 90])))
                if level == preferred and width <= cls.GRASP_MAX_GAP:
                    return preferred, True
                sections.append((width, float(np.median(band[:, 2]))))
        usable = [section for section in sections if section[0] <= cls.GRASP_MAX_GAP]
        return (max(usable, key=lambda section: section[0])[1] if usable else None,
                len(sections) == len(levels))

    def shape_grasp_geometry(self, item, direction, approach_from_above):
        """62418450 向井悠人
        初回は確かな形状情報から把持点や向きを調整し、再試行は従来の中心を使う。
        """
        center = item["center"]
        points = item["points"]
        if len(points) < 24:
            return center, direction
        retry = item.get("grasp_attempts", 1) > 1
        label = self.normalize_placement_label(item["label"])
        utensils = ("fork", "spoon", "spatula")
        boxes = ("cracker_box", "sugar_box", "pudding_box", "gelatin_box")
        round_items = ("master_chef_can", "tuna_fish_can", "chips_can", "tomato_soup_can",
                       "potted_meat_can", "mustard_bottle", "windex_bottle", "bleach_cleanser",
                       "mini_soccer_ball", "softball", "baseball", "tennis_ball",
                       "racquetball", "golf_ball")
        if label in utensils + boxes and not retry:
            axis = self.grasp_planar_axis(points)
            if axis is None:
                return center, direction
            if label in utensils:
                handle = self.utensil_handle(points, axis)
                if handle is not None:
                    handle[2] = max(handle[2], center[2])
                    return handle, direction
            else:
                faces = (axis, -axis, np.array([-axis[1], axis[0]]),
                         np.array([axis[1], -axis[0]]))
                face = max(faces, key=lambda candidate: float(np.dot(direction, candidate)))
                if np.dot(direction, face) >= 0.87:
                    return center, face
        elif label in round_items and not approach_from_above:
            height, fully_observed = self.grasp_body_height(
                points, direction, preferred=center[2] if retry else None)
            if height is not None:
                return np.array([center[0], center[1], height]), direction
            if fully_observed:
                rospy.logwarn("All observed side-grasp sections are too wide: %s", label)
                return None, direction
        return center, direction

    def grasp_measured_object(self, item, approach_from_above, shelf=False):
        """62418450 向井悠人
        形状から選んだ位置へ指を合わせ、持ち上げと保持を確認して腕を引く。
        """
        base = self.get_robot_pose()
        if base is None or self.hand_state != "empty":
            return False
        self.arm_ready = False
        center = item["center"]
        toward_base = np.array(base[:2]) - center[:2]
        distance = np.linalg.norm(toward_base)
        if distance < 0.25 or distance > 0.95:
            rospy.logwarn("Grasp outside the provisional reach window: %.2fm", distance)
            self.arm_ready = True
            return False
        toward_base /= distance
        anchor, direction = self.shape_grasp_geometry(item, toward_base, approach_from_above)
        if anchor is None:
            self.arm_ready = True
            return False
        if not 0.25 <= math.dist(base[:2], anchor[:2]) <= 0.95:
            anchor, direction = center, toward_base
        yaw = math.degrees(math.atan2(direction[1], direction[0]))
        offset = np.array([0.0, 0.0, self.HAND_PALM_Z_OFFSET] if approach_from_above
                          else [*(direction * self.HAND_PALM_OFFSET), 0.0])
        contact = anchor + offset
        approach = contact + np.array([0.0, 0.0, 0.10] if approach_from_above
                                      else [*(direction * 0.15), 0.0])
        orientation = quaternion_from_euler(yaw, -160 if approach_from_above else -100, 0)
        if not self.open_gripper():
            self.arm_ready = self.move_placement_pose("all_neutral")
            return False
        self.wait_for_simulation(6.0, "gripper settling before grasp")
        if not self.move_hand(approach, orientation):
            self.arm_ready = self.move_placement_pose("all_neutral")
            return False
        if not self.move_hand(contact, orientation, linear=True):
            self.arm_ready = (self.move_hand(approach, orientation, linear=True)
                              and self.move_placement_pose("all_neutral"))
            return False
        try:
            rotation, translation = self.transform_arrays(self.hand_pose())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn("Cannot locate the hand before closing: %s", error)
            self.arm_ready = (self.return_grasp_to_surface(contact, approach, orientation)
                              and self.move_placement_pose("all_neutral"))
            return False
        local = (item["points"] - translation).dot(rotation)
        low, high = np.percentile(local, [3, 97], axis=0)
        self.held_geometry = {"offset": (low + high) / 2,
                              "size": np.maximum(high - low, 0.015) + 0.006}
        if not self.close_for_grasp() or not self.verify_holding(after=rospy.Time.now()):
            self.arm_ready = (self.return_grasp_to_surface(contact, approach, orientation)
                              and self.move_placement_pose("all_neutral"))
            if self.hand_state == "empty":
                self.held_geometry = None
            return False

        lift = np.array([0.0, 0.0, 0.03 if shelf else 0.07])
        if (not self.move_hand(contact + lift, orientation, linear=True)
                or not self.verify_holding(after=rospy.Time.now())):
            self.arm_ready = (self.return_grasp_to_surface(contact, approach, orientation)
                              and self.move_placement_pose("all_neutral"))
            return False
        retreat = approach if approach_from_above else approach + lift
        if not self.move_hand(retreat, orientation, linear=True):
            self.arm_ready = (self.move_hand(contact + lift, orientation, linear=True)
                              and self.return_grasp_to_surface(contact, approach, orientation)
                              and self.move_placement_pose("all_neutral"))
            return False
        self.hand_state = "held"
        self.transport_empty_stamp = None
        self.transport_unknown_since = None
        self.arm_ready = self.move_placement_pose("all_neutral")
        return self.arm_ready

    def get_latest_detection(self):
        """62418450 向井悠人
        新しいRGB検出を時間内に取得し、応答しない接続は一度置き換えられるようにする。
        """
        key = "object detection"
        try:
            client = rospy.ServiceProxy("detection/get_object_detection", GetObjectDetection)
            pending = [job for job in self.query_jobs.get(key, ()) if not job["done"].is_set()]
            retry = False
            if pending:
                # 検出器は画像要求の状態を共有するため、同じ接続先に重複要求を出さない。
                endpoints = [job["endpoint"]() for job in pending]
                if len(pending) < 2 and all(endpoints):
                    current = self.bounded_query("detector provider", lambda:
                        rosgraph.Master(rospy.get_name()).lookupService(client.resolved_name), 1.0)
                    retry = current not in endpoints
            response = self.bounded_query(key, lambda: self.request_detection(client), 5.0,
                                          retry_readonly=retry, endpoint=lambda: client.uri)
        except Exception as error:
            raise rospy.ServiceException(str(error)) from error
        if not response.successed or response.bboxes.header.stamp == rospy.Time(0):
            raise rospy.ServiceException(response.error_string or "No stamped detection result")
        return response.bboxes

    def get_grasp_coordinate(self, bbox):
        """サンプルのまま（担当なし）
        BBox情報から把持座標を取得する
        """
        # BBox情報からtfを生成して、座標を取得
        self.tf_from_bbox_clt.call(
            SetTransformFromBBoxRequest(bbox=bbox, frame=self.GRASP_TF_NAME))
        rospy.sleep(1.0)  # tfが安定するのを待つ
        return self.get_relative_coordinate("map", self.GRASP_TF_NAME).translation

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

        # 机面衝突防止のためのZ軸最小値ガード
        MIN_SAFE_Z = 0.05
        pos_z = max(pos_z, MIN_SAFE_Z)

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

        # 上面把持を適用する形状の拡張（小物体や薄い物体を追加）
        upper_grasp_labels = {
            "cup", "cups", "frisbee", "bowl", "plate",
            "small_marker", "large_marker", "fork", "spoon",
            "lego_duplo", "rubiks_cube", "rubik_cube"
        }

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
        if (not isinstance(self.coordinates, dict)
                or not isinstance(self.coordinates.get("positions"), dict)
                or not isinstance(self.poses, dict)):
            raise ValueError("Invalid structure in coordinates.json or poses.json")
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

    @staticmethod
    def runtime_number(name, default, positive=False):
        """62418450 向井悠人
        ROSパラメータを有限の数値として読み、無効な値には項目名を付ける。
        """
        value = rospy.get_param(name, default)
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("Invalid parameter {}: {!r}".format(name, value)) from error
        if isinstance(value, bool) or not math.isfinite(number) or (positive and number <= 0):
            raise ValueError("Invalid parameter {}: {!r}".format(name, value))
        return number

    def validate_runtime_settings(self):
        """62418450 向井悠人
        通常運転で必要な座標、姿勢、数値設定、依存関係を移動前に確認する。
        """
        import cv2
        from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortGoal

        if not callable(cv2.distanceTransform) or not all(
                (GripperApplyEffortAction, GripperApplyEffortGoal)):
            raise ValueError("Missing map or gripper support")
        scan = rospy.get_param("~placement_scan_waypoint", "drawer")
        if not isinstance(scan, str) or not scan:
            raise ValueError("Invalid ~placement_scan_waypoint")
        for name in ("near_long_table_l", "tall_table", "long_table_r", "bin_a_place",
                     "bin_b_place", "standby_2a", "go_throw_2a", "shelf", scan,
                     *self.PERSON_WAYPOINTS.values()):
            position = self.coordinates["positions"].get(name)
            if (not isinstance(position, (tuple, list)) or len(position) != 3
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in position)):
                raise ValueError("Missing or invalid waypoint: {}".format(name))
        for name in ("all_neutral", "deliver_to_human", "look_at_near_floor",
                     "look_at_tall_table", "look_at_shelf"):
            pose = self.poses.get(name)
            if not isinstance(pose, dict) or not pose:
                raise ValueError("Missing or invalid pose: {}".format(name))
            if name.startswith("look_at") and "head_tilt_joint" not in pose:
                raise ValueError("Missing head_tilt_joint in {}".format(name))
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in pose.values()):
                raise ValueError("Invalid joint value in {}".format(name))
        for name, default in (("~navigation_timeout", 60.0), ("~placement_navigation_timeout", 60.0),
                              ("~placement_standoff", 0.65), ("~placement_hover_distance", 0.12),
                              ("~motion_timeout", 35.0)):
            self.runtime_number(name, default, positive=True)
        self.runtime_number("~task1_time_limit", 14 * 60)
        if not 0 <= self.runtime_number("~empty_finger_gap", 0.015) < self.GRASP_MAX_GAP:
            raise ValueError("~empty_finger_gap must be in [0, 0.12)")
        self.floor_z_offset = self.runtime_number("~floor_z_offset", 0.0)
        if abs(self.floor_z_offset) > 0.015:
            raise ValueError("~floor_z_offset must be between -0.015 and 0.015 m")

    def read_placement_cloud(self, after, stride=3):
        """62418450 向井悠人
        指定時刻より新しい深度点群を取り、地図座標の点群と時刻を返す。
        """
        deadline = time.monotonic() + 4.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            clouds = self.depth_clouds.copy()
            if clouds and clouds[-1].header.stamp > after:
                cloud = clouds[-1]
                return self.cloud_in_map(cloud, stride), cloud.header.stamp
            time.sleep(0.03)
        raise ValueError("No new depth cloud after the observation pose")

    @staticmethod
    def detect_container_rims(points, floor_bin=False):
        """62418450 向井悠人
        深度点群から容器・トレイ・床のBinの矩形の縁を抽出する。
        """
        import cv2
        import numpy as np

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected an Nx3 map-frame cloud")
        points = points[np.isfinite(points).all(axis=1)]
        if floor_bin:
            table_z = 0.0
            raised = points[(points[:, 2] > 0.10) & (points[:, 2] < 0.48)]
        else:
            points = points[(points[:, 2] > 0.25) & (points[:, 2] < 1.1)]
            if len(points) < 100:
                return []
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
            if floor_bin:
                if not (0.18 <= width <= 0.42 and 0.22 <= length <= 0.50
                        and 0.16 <= height <= 0.46):
                    continue
                kind = "bin"
            elif 0.24 <= width <= 0.42 and 0.28 <= length <= 0.48 and height < 0.065:
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
                               "rim_z": table_z + height, "support_z": table_z,
                               "width": float(width),
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

    def observe_placement_containers(self, category):
        """62418450 向井悠人
        把持後に収納側を向いて観測し、足りなければ既知の観測地点で見直す。
        """
        waypoint = rospy.get_param("~placement_scan_waypoint", "drawer")
        if not isinstance(waypoint, str) or waypoint not in self.coordinates["positions"]:
            rospy.logwarn("Invalid placement scan waypoint: %s", waypoint)
            return []
        pos = self.coordinates["positions"][waypoint]
        for attempt in range(2):
            arrived = (self.turn_base_toward(pos[2]) if attempt == 0 else
                       self.navigate_checked(pos, waypoint, "~placement_navigation_timeout"))
            if self.transport_loss is not None:
                return []
            if not arrived:
                continue
            frames = []
            matches = []
            for pan in (0.0, -0.7, 0.7):
                if not self.point_head(pan, -0.55):
                    continue
                stamp = rospy.Time.now()
                try:
                    for _ in range(2):
                        points, stamp = self.read_placement_cloud(stamp)
                        frames.append(self.detect_container_rims(points))
                except (ValueError, rospy.ROSException, tf2_ros.LookupException,
                        tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as error:
                    rospy.logwarn("Deposit view failed: %s", error)
                matches = [item for item in self.merge_container_observations(frames)
                           if item["kind"] == category]
                if matches and (category != "tray" or len(matches) >= 2):
                    return matches
            if matches:
                return matches
        return []

    def ensure_sensor_placements(self, category, destinations):
        """62418450 向井悠人
        配置設定がない容器を把持後に観測し、接近点と開口の候補を登録する。
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
        found = self.observe_placement_containers(category)
        maximum = 2 if category == "tray" else 1
        if not found or len(found) > maximum:
            raise ValueError("Missing or ambiguous sensor observations for {}".format(category))
        base = self.lookup_transform(
            "map", "base_footprint", rospy.Time(0), 2.0).transform.translation
        distance = self.runtime_number("~placement_standoff", 0.65, positive=True)
        hover = self.runtime_number("~placement_hover_distance", 0.12, positive=True)
        self.placement_targets = targets
        for (place, _), item in zip(destinations, sorted(found, key=lambda value: value["x"])):
            dx, dy = base.x - item["x"], base.y - item["y"]
            norm = math.hypot(dx, dy)
            if norm < 0.1:
                raise ValueError("Observation viewpoint is too close to container center")
            self.coordinates["positions"][place] = [
                item["x"] + distance * dx / norm, item["y"] + distance * dy / norm,
                math.degrees(math.atan2(-dy, -dx))]
            targets[place] = dict(item, hover=hover)
            self.sensor_placement_names.add(place)
            rospy.loginfo("Sensor placement %s: %s", place, json.dumps(item, sort_keys=True))

    @staticmethod
    def opening_bottom_height(place, kind, center, half, frames, rim_z):
        """62418450 向井悠人
        実測した深さか、支持面付近に見える底面から収納先の内底の高さを決める。
        """
        depths = rospy.get_param("~placement_inner_depths", {})
        depth = depths.get(place, depths.get(kind)) if isinstance(depths, dict) else None
        if depth is not None:
            if (not isinstance(depth, (float, int)) or isinstance(depth, bool)
                    or not math.isfinite(depth) or not 0.004 <= depth <= 0.45):
                rospy.logwarn("Invalid measured opening depth for %s", place)
                return None
            return rim_z - depth

        bottoms = []
        for cloud, rim in frames:
            support = rim.get("support_z")
            if support is None:
                return None
            inside = np.all(np.abs(cloud[:, :2] - center) < half, axis=1)
            heights = cloud[inside, 2]
            # 台や床のすぐ上で、縁より低い面だけを底面の候補にする。
            base = heights[(heights >= support - 0.005)
                           & (heights <= min(support + 0.025, rim["rim_z"] - 0.004))]
            if len(base) < 8:
                return None
            bottoms.append(float(np.median(base)))
        if abs(bottoms[0] - bottoms[1]) > 0.010:
            return None
        return sum(bottoms) / len(bottoms)

    @staticmethod
    def opening_patch_is_clear(cloud, xy, extent, bottom, rim_z):
        """62418450 向井悠人
        置く範囲の四隅側で底面を確認し、底より高い物体がある候補を除外する。
        """
        relative = cloud[:, :2] - xy
        inside = np.all(np.abs(relative) <= extent + 0.008, axis=1)
        heights = cloud[inside, 2]
        tolerance = min(0.009, max(0.003, (rim_z - bottom) * 0.28))
        raised = (heights > bottom + tolerance) & (heights < rim_z + 0.25)
        if np.count_nonzero(raised) >= 2:
            return False
        bottom_points = relative[inside][np.abs(heights - bottom) <= tolerance]
        quadrants = {(x >= 0, y >= 0) for x, y in bottom_points}
        return len(quadrants) == 4

    def measure_container_opening(self, place, pan, tilt, skip=None):
        """62418450 向井悠人
        到着後の2フレームで縁と底面を確かめ、保持物が入る空きを探す。
        """
        target = self.placement_targets[place]
        kind = target.get("kind", place.rsplit("_place", 1)[0])
        if kind.startswith("tray"):
            kind = "tray"
        floor_bin = kind == "bin"
        base = self.get_robot_pose() if floor_bin else None
        if floor_bin and base is None:
            return False
        if not self.point_head(pan, tilt - 0.28 if floor_bin else tilt):
            return False
        frames = []
        stamp = rospy.Time.now()
        for _ in range(2):
            try:
                cloud, stamp = self.read_placement_cloud(stamp, stride=2)
            except (ValueError, rospy.ROSException, tf2_ros.LookupException,
                    tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as error:
                rospy.logwarn("Opening observation failed for %s: %s", place, error)
                return False
            if floor_bin:
                forward = np.array([math.cos(base[2]), math.sin(base[2])])
                lateral = np.array([-forward[1], forward[0]])
                delta = cloud[:, :2] - np.asarray(base[:2])
                front, side = delta.dot(forward), delta.dot(lateral)
                scan = cloud[(front > 0.15) & (front < 0.80) & (np.abs(side) < 0.24)]
                rims = self.detect_container_rims(scan, floor_bin=True)
            else:
                rims = [rim for rim in self.detect_container_rims(cloud)
                        if rim["kind"] == kind and math.hypot(
                            rim["x"] - target["x"], rim["y"] - target["y"]) <= 0.14]
            if len(rims) != 1:
                return False
            frames.append((cloud, rims[0]))
        first, second = (frame[1] for frame in frames)
        if math.hypot(first["x"] - second["x"], first["y"] - second["y"]) > 0.025:
            return False
        if abs(first["rim_z"] - second["rim_z"]) > 0.025:
            return False
        center = np.array([(first[axis] + second[axis]) / 2 for axis in ("x", "y")])
        rim_z = max(first["rim_z"], second["rim_z"])
        half = min(first["width"], second["width"]) / (2 * math.sqrt(2)) - 0.008
        bottom = self.opening_bottom_height(place, kind, center, half, frames, rim_z)
        if bottom is None or bottom >= rim_z:
            rospy.logwarn("Cannot confirm the bottom of %s", place)
            return False
        rotation, _ = self.transform_arrays(self.hand_pose())
        extent = np.abs(rotation).dot(self.held_geometry["size"]) / 2
        slack = half - extent[:2] - 0.012
        if np.any(slack <= 0):
            return False
        for shift in ((0, 0), (-0.65, 0), (0.65, 0), (0, -0.65), (0, 0.65)):
            xy = center + np.asarray(shift) * slack
            if skip is not None and math.dist(xy, skip) < 0.012:
                continue
            if all(self.opening_patch_is_clear(cloud, xy, extent[:2], bottom, rim_z)
                   for cloud, _ in frames):
                target.update(rim_z=rim_z, drop_x=float(xy[0]), drop_y=float(xy[1]),
                              opening_x=float(center[0]), opening_y=float(center[1]),
                              opening_half=half)
                rospy.loginfo("Opening confirmed for %s: x=%.3f y=%.3f rim=%.3f",
                              place, xy[0], xy[1], rim_z)
                return True
        return False

    def refresh_held_geometry(self, label, rim_z):
        """62418450 向井悠人
        容器の上で新しいRGB-Dから保持物を特定し、手の中での位置を測り直す。
        """
        if not label or self.held_geometry is None:
            return False
        after = rospy.Time.now()
        try:
            detection = self.get_latest_detection()
            stamp = detection.header.stamp
            if stamp <= after:
                return False
            hand = self.lookup_transform("map", "hand_palm_link", stamp, 1.0).transform
        except (rospy.ROSException, rospy.ServiceException, tf2_ros.LookupException,
                tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as error:
            rospy.logwarn("Held-object observation failed: %s", error)
            return False

        rotation, translation = self.transform_arrays(hand)
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            rospy.logwarn("Cannot update payload geometry from an invalid hand transform")
            return False
        prior = self.held_geometry
        expected = translation + rotation.dot(prior["offset"])
        matches = []
        for bbox in detection.bboxes:
            if (not bbox.label.strip() or bbox.w <= 0 or bbox.h <= 0 or bbox.score <= 0
                    or self.normalize_placement_label(bbox.label) != self.normalize_placement_label(label)):
                continue
            _, item = self.measure_object(bbox, detection, min_height=rim_z + 0.003)
            if item is None or math.dist(item["center"], expected) > 0.10 or len(item["points"]) < 24:
                continue
            local = (item["points"] - translation).dot(rotation)
            if not np.isfinite(local).all():
                continue
            low, high = np.percentile(local, [3, 97], axis=0)
            span = high - low
            if np.all((prior["size"] <= 0.025) | (span >= 0.55 * (prior["size"] - 0.006))):
                matches.append({"offset": (low + high) / 2,
                                "size": np.maximum(prior["size"], span + 0.006)})
        if len(matches) != 1:
            return False
        self.held_geometry = matches[0]
        return True

    @staticmethod
    def payload_has_opening_clearance(target, center, extent):
        """62418450 向井悠人
        開口の中心からのずれ、物体の幅、保持位置の観測余裕を合わせて確認する。
        """
        opening = np.array([target["opening_x"], target["opening_y"]])
        margin = target.get("payload_margin", 0.045)
        return bool(np.all(np.abs(center[:2] - opening) + extent[:2] + margin
                           < target["opening_half"]))

    def held_object_fits(self, target, after=None):
        """62418450 向井悠人
        開放直前の保持と、再観測の確かさに応じた横・縦の余裕を確認する。
        """
        if not self.verify_holding(after=after):
            self.hand_state = "unknown"
            return False
        rotation, translation = self.transform_arrays(self.hand_pose())
        center = translation + rotation.dot(self.held_geometry["offset"])
        extent = np.abs(rotation).dot(self.held_geometry["size"]) / 2
        drop = np.array([target["drop_x"], target["drop_y"]])
        bottom = center[2] - extent[2]
        vertical_uncertainty = 0.035 if target.get("payload_margin", 0.045) > 0.010 else 0.0
        return (self.payload_has_opening_clearance(target, center, extent)
                and np.linalg.norm(center[:2] - drop) <= 0.015
                and target["rim_z"] + 0.003 + vertical_uncertainty <= bottom <= target["rim_z"] + 0.055)

    def raise_hand_for_placement(self, height):
        """62418450 向井悠人
        開口付近の現在の手先位置から、横へずらさずに垂直に上げる。
        """
        hand = self.hand_pose()
        _, position = self.transform_arrays(hand)
        return self.move_hand(position + np.array([0.0, 0.0, height]), hand.rotation, linear=True)

    def align_for_release(self, place, record):
        """62418450 向井悠人
        開口を測って置く位置へ移動し、必要なら位置を変えて一度試し直す。
        """
        target = self.placement_targets[place]
        previous = None
        for pan, tilt in ((0.0, -0.72), (0.20, -0.58)):
            record["stage"] = "measure_opening"
            if not self.measure_container_opening(place, pan, tilt, previous):
                continue
            previous = (target["drop_x"], target["drop_y"])
            record["stage"] = "above_container"
            if not self.move_over_container(place, hover=True):
                continue

            record["stage"] = "observe_payload"
            observed = self.refresh_held_geometry(record.get("label"), target["rim_z"])
            target["payload_margin"] = 0.010 if observed else 0.045
            rotation, _ = self.transform_arrays(self.hand_pose())
            extent = np.abs(rotation).dot(self.held_geometry["size"]) / 2
            drop = np.array([target["drop_x"], target["drop_y"]])
            if self.payload_has_opening_clearance(target, drop, extent):
                if not observed or self.move_over_container(place, hover=True, linear=True):
                    record["stage"] = "lower"
                    if self.move_over_container(place, linear=True):
                        after = rospy.Time.now()
                        record["stage"] = "verify_opening"
                        confirmed = self.refresh_held_geometry(record.get("label"), target["rim_z"])
                        target["payload_margin"] = 0.010 if confirmed else 0.045
                        if (observed and not confirmed
                                and self.payload_has_opening_clearance(target, drop, extent)):
                            if not self.raise_hand_for_placement(0.025):
                                return False
                            after = rospy.Time.now()
                        if self.held_object_fits(target, after=after):
                            return True
                    if not self.raise_hand_for_placement(target["hover"]):
                        return False
            else:
                rospy.logwarn("Opening clearance is too small for the payload estimate: %s", place)
            if self.hand_state != "held" or not self.move_placement_pose("all_neutral"):
                return False
        return False

    def move_over_container(self, place, hover=False, linear=False):
        """62418450 向井悠人
        保持物体の中心を開口に合わせ、手先とのずれを補正して動かす。
        """
        target = self.placement_targets[place]
        hand = self.hand_pose()
        rotation, _ = self.transform_arrays(hand)
        offset = rotation.dot(self.held_geometry["offset"])
        extent = np.abs(rotation).dot(self.held_geometry["size"]) / 2
        center = np.array([target["drop_x"], target["drop_y"],
                           target["rim_z"] + extent[2] + 0.024])
        if hover:
            center[2] += target["hover"]
        elif target.get("payload_margin", 0.045) > 0.010:
            center[2] += 0.025
        return self.move_hand(center - offset, hand.rotation, linear)

    @classmethod
    def placement_category(cls, label):
        """62418450 向井悠人
        検出ラベルから本来の収納先の分類だけを判定する。
        """
        label = cls.normalize_placement_label(label)
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
        return next((category for category, labels in categories.items() if label in labels), None)

    def planned_placement(self, label):
        """62418450 向井悠人
        把持前に本来の収納先と、引き出しを使わない初版の搬送先を決める。
        """
        intended = self.placement_category(label)
        return intended, "bin_b" if intended in ("drawer_tools", "drawer_shape") else intended

    def placement_candidates(self, category):
        """62418450 向井悠人
        収納先の分類に合う候補を観測し、到達しやすい順に返す。
        """
        destinations = {
            "tray": (("tray_a_place", "put_in_tray"), ("tray_b_place", "put_in_tray")),
            "container_a": (("container_a_place", "put_in_container_a"),),
            "container_b": (("container_b_place", "put_in_container_b"),),
            "bin_a": (("bin_a_place", "put_in_bin"),),
            "bin_b": (("bin_b_place", "put_in_bin"),),
        }
        if category not in destinations:
            return []
        options = destinations[category]
        self.ensure_sensor_placements(category, options)
        candidates = [(place, pose) for place, pose in options
                      if place in self.coordinates["positions"]
                      and (self.poses.get(pose) or place in self.placement_targets)]
        if len(candidates) <= 1:
            return candidates
        ranked = [(self.placement_path_length(place), (place, pose))
                  for place, pose in candidates]
        return [candidate for length, candidate in sorted(
            ranked, key=lambda item: math.inf if item[0] is None else item[0])
                if length != math.inf]

    @staticmethod
    def request_service(name, service_type, *args):
        """62418450 向井悠人
        読み取りサービスの応答を実時間で監視する呼び出し元へ返す。
        """
        rospy.wait_for_service(name, timeout=1.0)
        return rospy.ServiceProxy(name, service_type)(*args)

    def plan_base_path(self, pos):
        """62418450 向井悠人
        move_baseに計画を問い合わせ、経路なしと問い合わせ失敗を区別する。
        """
        current = self.get_robot_pose()
        if current is None:
            return None
        start = PoseStamped()
        start.header.frame_id = "map"
        start.pose.position.x, start.pose.position.y = current[:2]
        start.pose.orientation = quaternion_from_euler(math.degrees(current[2]), 0, 0)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x, goal.pose.position.y = pos[:2]
        goal.pose.orientation = quaternion_from_euler(pos[2], 0, 0)
        service = rospy.get_param("~placement_plan_service", "/move_base/make_plan")
        if not isinstance(service, str) or not service:
            rospy.logwarn("Invalid placement plan service: %s", service)
            return None
        try:
            path = self.bounded_query(service, lambda: self.request_service(
                service, GetPlan, start, goal, 0.0), 2.5, retry_readonly=True).plan.poses
        except (TimeoutError, rospy.ROSException, rospy.ServiceException) as error:
            rospy.logwarn("Base path unavailable: %s", error)
            return None
        if not path:
            return []
        points = [(p.pose.position.x, p.pose.position.y) for p in path]
        if not all(math.isfinite(value) for point in points for value in point):
            rospy.logwarn("Base path contains nonfinite coordinates")
            return None
        return [current[:2]] + points

    @staticmethod
    def decode_depth_points(cloud, pixels, skip_nans):
        """62418450 向井悠人
        XYZ点群を読み、壊れた形式は観測失敗として扱える例外に統一する。
        """
        try:
            points = np.asarray(list(point_cloud2.read_points(
                cloud, field_names=("x", "y", "z"), skip_nans=skip_nans, uvs=pixels)), dtype=float)
        except (AssertionError, IndexError, TypeError, ValueError, OverflowError, struct.error) as error:
            raise ValueError("Cannot decode XYZ depth cloud: {}".format(error)) from error
        if not points.size:
            return np.empty((0, 3))
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected three coordinates per depth point")
        return points

    def cloud_in_map(self, cloud, stride):
        """62418450 向井悠人
        深度点群を間引き、撮影時刻のTFで地図座標に変換する。
        """
        if not cloud.header.frame_id or cloud.height <= 1:
            raise ValueError("Expected registered, organized depth")
        transform = self.lookup_transform(
            "map", cloud.header.frame_id, cloud.header.stamp, 1.0)
        pixels = [(u, v) for v in range(0, cloud.height, stride)
                  for u in range(0, cloud.width, stride)]
        points = self.decode_depth_points(cloud, pixels, skip_nans=True)
        if len(points) < 30:
            raise ValueError("Not enough depth measurements")
        points = points[np.isfinite(points).all(axis=1)]
        rotation, translation = self.transform_arrays(transform.transform)
        world = points.dot(rotation.T) + translation
        if not np.isfinite(world).all():
            raise ValueError("Nonfinite map coordinates in depth cloud")
        return world

    def floor_relative_height(self, points):
        """62418450 向井悠人
        事前に確認した床の高さを基準に、点群の床面からの高さを求める。
        """
        return points[:, 2] - self.floor_z_offset

    def floor_surface_points(self, points):
        """62418450 向井悠人
        薄い障害物の可能性がある高さを、確認済みの床面から除外する。
        """
        height = self.floor_relative_height(points)
        return points[(height >= -0.015) & (height <= 0.010), :2]

    def floor_obstacle_cells(self, points):
        """62418450 向井悠人
        通常の障害物と、床からわずかに高い複数点のまとまりを障害物セルにする。
        """
        height = self.floor_relative_height(points)
        raised = (height > 0.010) & (height < 0.40)
        cells, indices, counts = np.unique(
            np.floor(points[raised, :2] / 0.05), axis=0, return_inverse=True, return_counts=True)
        strong = np.bincount(indices, weights=(height[raised] > 0.015), minlength=len(cells))
        return (cells[(strong >= 2) | (counts >= 4)] + 0.5) * 0.05

    def transport_payload_present(self):
        """62418450 向井悠人
        搬送中に指が空の間隔まで閉じたことを再確認し、落下の観測位置を記録する。
        """
        if self.hand_state != "held":
            return True
        empty_gap = self.runtime_number("~empty_finger_gap", 0.015)
        sample = self.finger_gap(timeout=0.0, stamped=True)
        if sample is None:
            self.transport_empty_stamp = None
            if self.transport_unknown_since is None:
                self.transport_unknown_since = time.monotonic()
            return time.monotonic() - self.transport_unknown_since < 0.5
        self.transport_unknown_since = None
        if sample[0] > empty_gap:
            self.transport_empty_stamp = None
            return True
        _, stamp = sample
        if self.transport_empty_stamp is None:
            self.transport_empty_stamp = stamp
            return True
        if stamp <= self.transport_empty_stamp:
            return True

        position = None
        if self.held_geometry is not None:
            try:
                rotation, translation = self.transform_arrays(self.hand_pose(timeout=0.0))
                center = translation + rotation.dot(self.held_geometry["offset"])
                if np.isfinite(center).all():
                    position = tuple(center[:2])
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                rospy.logwarn("Cannot locate the lost payload: %s", error)
        self.transport_loss = {"xy": position}
        self.hand_state = "empty"
        self.held_geometry = None
        rospy.logwarn("Payload lost during transport: estimated floor position=%s", position)
        return False

    def latest_floor_cloud(self):
        """62418450 向井悠人
        走行や旋回の監視に使える時刻の点群を地図座標で返す。
        """
        clouds = self.depth_clouds.copy()
        if not clouds or abs((rospy.Time.now() - clouds[-1].header.stamp).to_sec()) > 0.65:
            return None
        try:
            return self.cloud_in_map(clouds[-1], 5)
        except (ValueError, rospy.ROSException, tf2_ros.LookupException,
                tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def corridor_still_clear(self, corridor, separating=None):
        """62418450 向井悠人
        走行中の逸脱と障害物を監視し、後退では残りの床の見え方も確認する。
        """
        start, target = corridor
        pose = self.get_robot_pose()
        if pose is None or self.dist_point_to_segment(pose[:2], start, target) > 0.08:
            return False
        points = self.latest_floor_cloud()
        if points is None:
            return False
        obstacles = self.floor_obstacle_cells(points)
        if separating is not None:
            floor = self.floor_surface_points(points)
            fresh = [point for point in obstacles if tuple(point) not in separating]
            return len(floor) >= 12 and self.observed_segment_clear(
                pose[:2], target, (floor, fresh))
        radius = self.base_radius_with_payload(pose)
        return (radius is not None and all(
            self.dist_point_to_segment(point, pose[:2], target) >= radius + 0.12
            for point in obstacles))

    def await_base_result(self, timeout, corridor):
        """62418450 向井悠人
        実時間で台車の終了を監視し、床や保持が不確かになれば中止理由を返す。
        """
        client = omni_base.NAVCLIENT
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            state = client.get_state()
            if corridor is not None and not self.transport_payload_present():
                return "lost payload" if self.transport_loss is not None else "payload unconfirmed"
            if self.has_action_result(client, state):
                return "done"
            if state == GoalStatus.LOST:
                return "lost action state"
            if corridor is not None and not self.corridor_still_clear(corridor):
                return "left confirmed corridor or floor no longer clear"
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        return "interrupted" if rospy.is_shutdown() else "timed out"

    def observe_floor(self, pan=0.0, tilt=-1.15):
        """62418450 向井悠人
        床面と床から出ている物を、新しい深度点群から分けて返す。
        """
        if not self.point_head(pan, tilt):
            return None
        try:
            points, _ = self.read_placement_cloud(rospy.Time.now(), stride=5)
        except (ValueError, rospy.ROSException, tf2_ros.LookupException,
                tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as error:
            rospy.logwarn("Floor observation failed: %s", error)
            return None
        floor = self.floor_surface_points(points)
        if len(floor) < 12:
            rospy.logwarn("Not enough visible floor to check a base move")
            return None
        return floor, self.floor_obstacle_cells(points)

    @classmethod
    def observed_segment_clear(cls, start, goal, observation, margin=0.09):
        """62418450 向井悠人
        台車が次に通る帯に障害物がなく、外側まで床が見えているか調べる。
        """
        floor, obstacles = observation
        direction = np.asarray(goal) - np.asarray(start)
        distance = float(np.linalg.norm(direction))
        if distance <= 1e-6:
            return True
        direction /= distance
        side = np.array([-direction[1], direction[0]])
        radius = cls.ROBOT_RADIUS + margin
        if any(cls.dist_point_to_segment(point, start, goal) < radius + 0.035
               for point in obstacles):
            return False
        # 台車が今占めている床は視界に入らないので、進行先の外周を確認する。
        stations = np.linspace(radius, distance + radius + 0.07,
                               max(2, int(math.ceil(distance / 0.08)) + 1))
        for along in stations:
            for lateral in (-radius + 0.06, 0.0, radius - 0.06):
                sample = np.asarray(start) + direction * along + side * lateral
                if np.min(np.linalg.norm(floor - sample, axis=1)) > 0.13:
                    return False
        return True

    def base_radius_with_payload(self, pose):
        """62418450 向井悠人
        台車中心から保持物の外側まで含めた旋回・通行用の半径を見積もる。
        """
        if self.hand_state == "empty" or self.held_geometry is None:
            return self.ROBOT_RADIUS
        try:
            rotation, translation = self.transform_arrays(self.hand_pose())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn("Cannot locate the payload for base clearance: %s", error)
            return None
        center = translation + rotation.dot(self.held_geometry["offset"])
        extent = np.abs(rotation).dot(self.held_geometry["size"]) / 2
        radius = math.dist(center[:2], pose[:2]) + float(np.linalg.norm(extent[:2]))
        return max(self.ROBOT_RADIUS, radius)

    @staticmethod
    def floor_pan_toward(pose, target):
        """62418450 向井悠人
        HSRの首の可動域内で、床の観測点に最も近い向きを選ぶ。
        """
        bearing = math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        pans = (bearing, bearing - 2 * math.pi, bearing + 2 * math.pi)
        return min((max(-3.8, min(1.7, angle)) for angle in pans),
                   key=lambda angle: abs(math.atan2(math.sin(angle - bearing),
                                                     math.cos(angle - bearing))))

    def floor_covers_turn(self, pose, swept, views):
        """62418450 向井悠人
        旋回半径の外周を8方向に分け、各方向に観測済みの床があるか確認する。
        """
        origin = np.asarray(pose[:2])
        floor = np.vstack(views)
        distance = np.linalg.norm(floor - origin, axis=1)
        ring = floor[(distance >= self.ROBOT_RADIUS + 0.015)
                     & (np.abs(distance - swept - 0.02) <= 0.075)]
        if not len(ring):
            return False
        for angle in np.arange(8) * (math.pi / 4):
            sample = origin + (swept + 0.02) * np.array([math.cos(angle), math.sin(angle)])
            if np.min(np.linalg.norm(ring - sample, axis=1)) > 0.075:
                return False
        return True

    def turn_base_toward(self, yaw_deg):
        """62418450 向井悠人
        腕を移動姿勢にした台車を、現在地で目標の向きまで回す。
        """
        self.confirm_navigation_finished()
        if (not self.arm_ready or self.hand_state not in ("empty", "held")
                or not self.transport_payload_present()):
            return False
        pose = self.get_robot_pose()
        if pose is None:
            return False
        desired = math.radians(yaw_deg)
        error = math.atan2(math.sin(desired - pose[2]), math.cos(desired - pose[2]))
        if abs(error) <= math.radians(5):
            return True
        observation = self.observe_floor()
        if observation is None:
            return False
        radius = self.base_radius_with_payload(pose)
        if radius is None:
            return False
        swept = radius + 0.12
        views = [observation[0]]
        self.update_floor_obstacles(observation, self.floor_obstacles, self.floor_clearance)
        nearby = {point for point in self.floor_obstacles if math.dist(point, pose[:2]) < swept}
        if nearby and not nearby.intersection(map(tuple, observation[1])):
            # 現在の視点から見えない記録は、別の角度で床を確認できた場合だけ消す。
            point = min(nearby, key=lambda cell: math.dist(cell, pose[:2]))
            pan = self.floor_pan_toward(pose, point)
            for tilt in (-1.15, -0.95):
                refreshed = self.observe_floor(pan, tilt)
                if refreshed is not None:
                    views.append(refreshed[0])
                    self.update_floor_obstacles(refreshed, self.floor_obstacles, self.floor_clearance)
        if any(math.dist(point, pose[:2]) < swept for point in self.floor_obstacles):
            rospy.logwarn("Cannot turn: floor obstacle in the swept footprint")
            return False
        if self.wall_map is not None and not self.wall_segment_clear(pose[:2], pose[:2], swept):
            rospy.logwarn("Cannot turn: wall in the swept footprint")
            return False
        for pan in (1.45, -1.45, -math.pi, -3.72):
            if self.floor_covers_turn(pose, swept, views):
                break
            view = self.observe_floor(pan, -1.05 if swept > 0.48 else -1.25)
            if view is None:
                continue
            views.append(view[0])
            self.update_floor_obstacles(view, self.floor_obstacles, self.floor_clearance)
            if any(math.dist(point, pose[:2]) < swept for point in self.floor_obstacles):
                rospy.logwarn("Cannot turn: obstacle in a newly observed sector")
                return False
        if not self.floor_covers_turn(pose, swept, views):
            rospy.logwarn("Cannot turn: floor around the swept footprint is not fully visible")
            return False

        deadline = time.monotonic() + 15.0
        next_scan = time.monotonic()
        try:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                pose = self.get_robot_pose()
                if pose is None:
                    return False
                error = math.atan2(math.sin(desired - pose[2]), math.cos(desired - pose[2]))
                if not self.transport_payload_present():
                    return False
                if abs(error) <= math.radians(5):
                    return True
                if time.monotonic() >= next_scan:
                    cloud = self.latest_floor_cloud()
                    if cloud is None:
                        rospy.logwarn("Stopped turn: floor cloud unavailable")
                        return False
                    blocked = {tuple(point) for point in self.floor_obstacle_cells(cloud)
                               if math.dist(point, pose[:2]) < swept}
                    if blocked:
                        self.floor_obstacles.update(blocked)
                        for point in blocked:
                            self.floor_clearance.pop(point, None)
                        rospy.logwarn("Stopped turn: new obstacle in the swept footprint")
                        return False
                    next_scan = time.monotonic() + 0.25
                omega = max(-28.0, min(28.0, math.degrees(error) * 1.5))
                omni_base.cmd_vel(0, 0, omega)
                time.sleep(0.1)
            return False
        except Exception as error:
            rospy.logwarn("Base rotation failed: %s", error)
        finally:
            self.stop_base()
        self.confirm_manual_stop()
        self.navigation_needs_recovery = self.transport_loss is None
        return False

    @staticmethod
    def update_floor_obstacles(observation, obstacles, cleared):
        """62418450 向井悠人
        障害物を登録し、同じセルの床面を2回十分に観測した場合だけ消す。
        """
        floor, raised = observation
        raised = {tuple(point) for point in raised}
        observed_subcells = {}
        for x, y in np.unique(np.floor(floor / 0.025).astype(int), axis=0):
            cell = (x // 2, y // 2)
            observed_subcells.setdefault(cell, set()).add((x % 2, y % 2))
        for point in obstacles - raised:
            cell = tuple(math.floor(value / 0.05) for value in point)
            if len(observed_subcells.get(cell, ())) == 4:
                cleared[point] = cleared.get(point, 0) + 1
                if cleared[point] >= 2:
                    obstacles.remove(point)
                    cleared.pop(point, None)
            else:
                cleared.pop(point, None)
        for point in raised:
            cleared.pop(point, None)
        obstacles.update(raised)

    def route_step_target(self, current, path, obstacles, radius):
        """62418450 向井悠人
        計画線に沿う短い移動点を選び、壁と記憶した床の物体を横切る候補を除く。
        """
        if not path:
            return None
        origin = np.asarray(current)
        selected = None
        for point in path[1:]:
            delta = np.asarray(point) - origin
            length = float(np.linalg.norm(delta))
            if length < 0.045:
                continue
            step = tuple(origin + delta * min(1.0, 0.30 / length))
            wall_clear = self.wall_map is None or self.wall_segment_clear(
                current, step, radius + 0.07)
            if not wall_clear or any(self.dist_point_to_segment(obstacle, current, step)
                                     < radius + 0.125 for obstacle in obstacles):
                break
            selected = step
            if length >= 0.30:
                break
        return selected

    def plan_floor_detour(self, current, path, obstacles, radius):
        """62418450 向井悠人
        壁と床の物体を避け、元の経路の先へ合流する局所A*経路を返す。
        """
        forward = [point for point in path if 0.60 <= math.dist(current, point) <= 1.6]
        if math.dist(current, path[-1]) < 0.60:
            forward = [path[-1]]
        if not forward:
            return None
        joins = dict.fromkeys(min(forward, key=lambda point: abs(math.dist(current, point) - distance))
                             for distance in (1.0, 1.5, 0.65))
        for goal in joins:
            bounds = tuple((min(a, b) - 0.65, max(a, b) + 0.65)
                           for a, b in zip(current, goal))
            candidate = self.plan_path_astar(
                current, goal, obstacles, radius + 0.125, bounds=bounds, wall_radius=radius + 0.07)
            if candidate and self.route_step_target(current, candidate, obstacles, radius):
                return candidate
        return None

    def drive_observed_step(self, target, name, timeout_param, obstacles, cleared):
        """62418450 向井悠人
        床を観測してから短い目標へ進み、失敗時の床障害物も次の計画に残す。
        """
        current = self.get_robot_pose()
        if current is None:
            return False
        heading = math.degrees(math.atan2(target[1] - current[1], target[0] - current[0]))
        if not self.turn_base_toward(heading):
            observation = self.observe_floor()
            if observation is not None:
                self.update_floor_obstacles(observation, obstacles, cleared)
            return False
        current = self.get_robot_pose()
        if current is None:
            return False
        radius = self.base_radius_with_payload(current)
        if radius is None:
            return False
        if self.wall_map is not None and not self.wall_segment_clear(
                current[:2], target, radius + 0.07):
            return False
        for tilt in (-1.15, -0.85):
            observation = self.observe_floor(tilt=tilt)
            if observation is None:
                continue
            self.update_floor_obstacles(observation, obstacles, cleared)
            floor, _ = observation
            occupied = np.asarray(list(obstacles), dtype=float).reshape(-1, 2)
            margin = radius - self.ROBOT_RADIUS + 0.09
            if self.observed_segment_clear(current[:2], target, (floor, occupied), margin):
                break
        else:
            rospy.logwarn("Floor not clear on the next segment to %s", name)
            return False
        if self.navigate_to_pose([*target, heading], name, timeout_param,
                                 corridor=(current[:2], target)):
            reached = self.get_robot_pose()
            return reached is not None and math.dist(current[:2], reached[:2]) >= 0.035
        observation = self.observe_floor()
        if observation is not None:
            self.update_floor_obstacles(observation, obstacles, cleared)
        return False

    def remember_navigation_checkpoint(self):
        """62418450 向井悠人
        停止と移動姿勢を確認した地点を次の移動の復帰候補にする。
        """
        if self.navigation_pending or not self.arm_ready or self.hand_state not in ("empty", "held"):
            return
        pose = self.get_robot_pose()
        if pose is not None:
            self.navigation_checkpoint = (*pose[:2], math.degrees(pose[2]))

    def navigate_checked(self, pos, name, timeout_param="~navigation_timeout"):
        """62418450 向井悠人
        確認付き移動に失敗したら、現在地、最後の停止地点の順に限られた回数で再試行する。
        """
        self.navigation_needs_recovery = False
        if self.navigate_checked_route(pos, name, timeout_param):
            self.remember_navigation_checkpoint()
            return True
        if not self.navigation_needs_recovery or not self.can_resume_navigation():
            return False

        rospy.logwarn("Navigation %s: replanning from the current position", name)
        if self.navigate_checked_route(pos, name + "_retry", timeout_param):
            self.remember_navigation_checkpoint()
            return True
        if not self.can_resume_navigation() or self.navigation_checkpoint is None:
            return False

        checkpoint = self.navigation_checkpoint
        current = self.get_robot_pose()
        if current is None or math.dist(current[:2], checkpoint[:2]) <= 0.10:
            return False
        rospy.logwarn("Navigation %s: trying recovery from %s", name, checkpoint)
        if not self.navigate_checked_route(checkpoint, "navigation_checkpoint", timeout_param):
            return False
        if not self.can_resume_navigation():
            return False
        if self.navigate_checked_route(pos, name + "_from_checkpoint", timeout_param):
            self.remember_navigation_checkpoint()
            return True
        rospy.logwarn("Navigation %s: recovery attempts exhausted", name)
        return False

    def can_resume_navigation(self):
        """62418450 向井悠人
        台車が停止し、自己位置、腕の移動姿勢、手の状態が確かな場合に再開を許す。
        """
        self.confirm_navigation_finished()
        return (not rospy.is_shutdown() and self.transport_loss is None and self.arm_ready
                and self.hand_state in ("empty", "held") and self.retry_robot_pose() is not None
                and self.transport_payload_present()
                and (self.hand_state == "empty" or self.verify_holding()))

    def navigate_checked_route(self, pos, name, timeout_param):
        """62418450 向井悠人
        観測した床障害物を保持して経路を引き直し、確認済みの短い区間を進む。
        """
        self.confirm_navigation_finished()
        if (not self.arm_ready or self.hand_state not in ("empty", "held")
                or self.transport_loss is not None):
            return False
        obstacles = self.floor_obstacles
        cleared = self.floor_clearance
        for _ in range(32):
            current = self.get_robot_pose()
            if current is None or rospy.is_shutdown() or not self.transport_payload_present():
                return False
            if math.dist(current[:2], pos[:2]) <= 0.10:
                return self.turn_base_toward(pos[2])
            path = self.plan_base_path(pos)
            if path is None:
                self.navigation_needs_recovery = True
            if not path and (not obstacles or self.wall_map is None
                             or math.dist(current[:2], pos[:2]) > 1.6):
                return False
            radius = self.base_radius_with_payload(current)
            if radius is None:
                return False
            radius = float(radius)
            step = self.route_step_target(current[:2], path, obstacles, radius)
            if step is None and path and obstacles:
                # 以前の占有セルが進路をふさぐ場合は、拾い上げた後の床も再確認する。
                for tilt in (-1.30, -1.0):
                    observation = self.observe_floor(tilt=tilt)
                    if observation is not None:
                        self.update_floor_obstacles(observation, obstacles, cleared)
                step = self.route_step_target(current[:2], path, obstacles, radius)
            if step is not None and self.drive_observed_step(
                    step, name, timeout_param, obstacles, cleared):
                continue

            current = self.get_robot_pose()
            if (current is None or self.transport_loss is not None
                    or not obstacles or self.wall_map is None):
                return False
            # 同じ直線を再送せず、床の占有セルを足した地図で先の経路へ合流する。
            route = path or [current[:2], tuple(pos[:2])]
            detour = self.plan_floor_detour(current[:2], route, obstacles, radius)
            step = self.route_step_target(current[:2], detour, obstacles, radius)
            if step is None or not self.drive_observed_step(
                    step, name + "_detour", timeout_param, obstacles, cleared):
                rospy.logwarn("No checked detour to %s", name)
                return False
            rospy.loginfo("Used checked detour towards %s via %s", name, step)
        rospy.logwarn("Checked navigation used its segment limit for %s", name)
        return False

    @staticmethod
    def floor_supports_release(floor, xy, extent):
        """62418450 向井悠人
        置く物体の四隅と中心の付近に床が見えるか調べる。
        """
        local = floor - xy
        half = extent + 0.035
        near = local[np.all(np.abs(local) <= half + 0.02, axis=1)]
        if len({(x >= 0, y >= 0) for x, y in near}) != 4:
            return False
        samples = [(x, y) for x in (-half[0], 0.0, half[0])
                   for y in (-half[1], 0.0, half[1])]
        return all(np.min(np.linalg.norm(near - sample, axis=1)) <= 0.045 for sample in samples)

    def recover_after_transport(self):
        """62418450 向井悠人
        手を安全な状態に戻し、物体を床へ預けた場合は位置を再試行用に記録する。
        """
        self.confirm_navigation_finished()
        self.recovered_position = None
        if self.hand_state == "empty":
            self.arm_ready = self.move_placement_pose("all_neutral")
            return self.arm_ready
        if self.held_geometry is None:
            return False
        base = self.get_robot_pose()
        if base is None:
            return False
        views = []
        for tilt in (-1.15, -0.95):
            observation = self.observe_floor(tilt=tilt)
            if observation is not None:
                self.update_floor_obstacles(observation, self.floor_obstacles, self.floor_clearance)
                views.append(observation[0])
        if not views:
            return False
        for _ in range(2):
            try:
                hand = self.hand_pose()
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                rospy.logwarn("Cannot locate the hand during recovery: %s", error)
        else:
            return False
        rotation, _ = self.transform_arrays(hand)
        offset = rotation.dot(self.held_geometry["offset"])
        extent = np.abs(rotation).dot(self.held_geometry["size"]) / 2
        direction = np.array([math.cos(base[2]), math.sin(base[2])])
        side = np.array([-direction[1], direction[0]])
        radius = float(np.linalg.norm(extent[:2])) + 0.08
        for lateral in (0.0, -0.16, 0.16):
            xy = np.asarray(base[:2]) + direction * 0.52 + side * lateral
            if not any(self.floor_supports_release(floor, xy, extent[:2]) for floor in views):
                continue
            if any(math.dist(point, xy) < radius for point in self.floor_obstacles):
                continue
            if self.wall_map is not None and not self.wall_segment_clear(xy, xy, radius):
                continue
            target = np.array([*xy, extent[2] + 0.015]) - offset
            above = target + np.array([0.0, 0.0, 0.12])
            self.arm_ready = False
            if not self.move_hand(above, hand.rotation):
                continue
            if not self.move_hand(target, hand.rotation, linear=True):
                continue
            if not self.release_hand():
                return False
            self.recovered_position = xy
            self.arm_ready = (self.move_hand(above, hand.rotation, linear=True)
                              and self.move_placement_pose("all_neutral"))
            return self.arm_ready
        return False

    def placement_path_length(self, place):
        """62418450 向井悠人
        現在地から収納の停止候補までの計画経路長を求める。
        """
        path = self.plan_base_path(self.coordinates["positions"][place])
        if path is None:
            return None
        if not path:
            return math.inf
        return sum(math.dist(a, b) for a, b in zip(path, path[1:]))

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
                try:
                    commander.set_joint_value_target(targets)
                except Exception as error:
                    rospy.logwarn("Cannot set pose %s: %s", name, error)
                    return False
                if not self.execute_moveit(commander, lambda: commander.go(wait=True), name):
                    return False
        return True

    def navigate_for_placement(self, place):
        """62418450 向井悠人
        床を確認して収納先へ進み、机上の収納先なら横へずらした停止位置も試す。
        """
        pos = self.coordinates["positions"][place]
        if self.navigate_checked(pos, place, "~placement_navigation_timeout"):
            return True
        if self.transport_loss is not None or place not in self.placement_targets:
            return False
        yaw = math.radians(pos[2])
        for offset in (0.15, -0.15):
            candidate = [pos[0] - math.sin(yaw) * offset,
                         pos[1] + math.cos(yaw) * offset, pos[2]]
            if self.plan_base_path(candidate) and self.navigate_checked(
                    candidate, place + "_alternate", "~placement_navigation_timeout"):
                return True
            if self.transport_loss is not None:
                return False
        return False

    def put_object_by_label(self, label, plan=None):
        """62418450 向井悠人
        正規の収納候補を順に試し、安全に保持できていれば Bin B に切り替える。
        """
        intended, destination = plan if plan is not None else self.planned_placement(label)
        rospy.loginfo("Placement intent: label=%s intended=%s destination=%s",
                      label, intended, destination)
        self.last_placement_result = {"released": False}
        try:
            candidates = self.placement_candidates(destination)
        except (ValueError, rospy.ROSException, rospy.ServiceException,
                tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn("Cannot observe placement for %s: %s", label, error)
            candidates = []
        if destination != "bin_b":
            candidates.extend(self.placement_candidates("bin_b"))
        for place, pose in candidates:
            if self.hand_state != "held" or not self.arm_ready:
                break
            if self.put_in_place(place, pose, label=label):
                return True
            if self.last_placement_result["released"]:
                break
        return False

    def put_in_place(self, place, into_pose, label=None):
        """62418450 向井悠人
        収納先に到達し、保持と開口を確認して開放・退避の結果を記録する。
        """
        started = time.monotonic()
        record = {"label": label, "place": place, "pose": into_pose, "stage": "validate",
                  "released": False, "success": False}
        self.last_placement_result = record
        floor_bin = place in ("bin_a_place", "bin_b_place")
        original = self.placement_targets.get(place) if floor_bin else None
        try:
            measured = floor_bin or place in self.placement_targets
            if (self.hand_state != "held" or not self.arm_ready
                    or place not in self.coordinates["positions"]
                    or (not measured and not self.poses.get(into_pose))):
                return False
            record["stage"] = "navigate"
            if not self.navigate_for_placement(place):
                return False
            record["stage"] = "holding"
            if not self.verify_holding():
                if self.transport_payload_present():
                    self.hand_state = "unknown"
                return False
            self.arm_ready = False
            if floor_bin:
                self.placement_targets[place] = {"kind": "bin", "hover": 0.12}
            if measured:
                aligned = self.align_for_release(place, record)
            else:
                record["stage"] = "place"
                aligned = self.move_placement_pose(into_pose)
                if aligned:
                    record["stage"] = "holding_before_release"
                    if not self.verify_holding(after=rospy.Time.now()):
                        self.hand_state = "unknown"
                        return False
            if not aligned:
                self.arm_ready = self.move_placement_pose("all_neutral")
                return False
            record["stage"] = "release"
            if not self.release_hand():
                return False
            record["released"] = True
            record["stage"] = "retreat"
            if measured:
                hand = self.hand_pose()
                _, current = self.transform_arrays(hand)
                if not self.move_hand(current + np.array([0.0, 0.0, 0.12]),
                                      hand.rotation, linear=True):
                    return False
            self.arm_ready = self.move_placement_pose("all_neutral")
            record["success"] = self.arm_ready
            return self.arm_ready
        except (ValueError, rospy.ROSException, rospy.ServiceException,
                tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            record["error"] = str(error)
            rospy.logwarn("Placement failed at %s: %s", record["stage"], error)
            return False
        finally:
            if floor_bin:
                if original is None:
                    self.placement_targets.pop(place, None)
                else:
                    self.placement_targets[place] = original
            if record["stage"] == "release" and self.hand_state == "empty":
                record["released"] = True
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["holding_state"] = self.hand_state
            rospy.loginfo("Placement result: %s", json.dumps(record, sort_keys=True))

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

    def item_on_shelf_side(self, center, side):
        """62418450 向井悠人
        棚前の基準点と向きから見た左右を地図座標で判定する。
        """
        x, y, yaw = self.coordinates["positions"]["shelf"]
        yaw = math.radians(yaw)
        lateral = -math.sin(yaw) * (center[0] - x) + math.cos(yaw) * (center[1] - y)
        return lateral > 0.04 if side == "left" else lateral < -0.04

    def remember_distinct_pick_pairs(self, observations):
        """62418450 向井悠人
        同時に別々の枠と深度で確認した近接する同ラベル個体の位置を覚える。
        """
        for index, (bbox, item) in enumerate(observations):
            first = tuple(round(float(value), 2) for value in item["center"])
            for other, measured in observations[index + 1:]:
                if other.label != bbox.label or self.bbox_iou(bbox, other) >= 0.5:
                    continue
                center = measured["center"]
                if 0.045 <= math.dist(item["center"], center) < 0.16:
                    second = tuple(round(float(value), 2) for value in center)
                    self.distinct_pick_pairs.add((bbox.label, *sorted((first, second))))

    def known_distinct_pick(self, label, center, attempted):
        """62418450 向井悠人
        近接候補が試行済みの個体と別々に観測された証拠を探す。
        """
        dimensions = len(attempted)
        position = center[:dimensions]
        if math.dist(position, attempted) < 0.035:
            return False
        for kind, first, second in self.distinct_pick_pairs:
            if kind != label:
                continue
            for known, neighbor in ((first, second), (second, first)):
                if (math.dist(attempted, known[:dimensions]) <= 0.035
                        and math.dist(position, neighbor[:dimensions]) <= 0.035):
                    return True
        return False

    def attempted_observation_indices(self, observations, attempted, floor):
        """62418450 向井悠人
        試行済みの位置を照合し、以前に確認した近くの別個体は候補に残す。
        """
        blocked = set()
        for label, location in attempted:
            dimensions = len(location)
            if dimensions == 2 and not floor:
                continue
            distance, index = min(
                ((math.dist(item["center"][:dimensions], location), index)
                 for index, (bbox, item) in enumerate(observations)
                 if bbox.label == label and not self.known_distinct_pick(label, item["center"], location)),
                default=(math.inf, None))
            if distance < 0.12:
                center = observations[index][1]["center"]
                blocked.update(candidate for candidate, (bbox, item) in enumerate(observations)
                               if bbox.label == label and math.dist(item["center"], center) < 0.025)
        return blocked

    def observe_pick_candidate(self, pose, floor=False, attempted=(), label=None,
                               previous=None, shelf=False, allow_approach=False,
                               near_xy=None, shelf_side=None):
        """62418450 向井悠人
        同じ視点の検出を照合し、深度や棚の左右が指定に合う候補を選ぶ。
        """
        tilt = self.poses[pose]["head_tilt_joint"]
        observed = False
        unresolved = False
        pans = (0.0, 0.45 if previous is None else -0.45)
        if shelf and shelf_side is not None:
            pans = (0.0, 0.45 if shelf_side == "left" else -0.45)
        if near_xy is not None:
            base = self.get_robot_pose()
            if base is None:
                return "observation_failed", None
            bearing = math.atan2(near_xy[1] - base[1], near_xy[0] - base[0]) - base[2]
            pan = max(-0.75, min(0.75, math.atan2(math.sin(bearing), math.cos(bearing))))
            pans = tuple(dict.fromkeys(max(-0.9, min(0.9, pan + offset))
                                       for offset in (0.0, -0.45, 0.45)))
        for pan in pans:
            if not self.point_head(pan, tilt):
                continue
            prior_boxes = []
            prior_stamp = None
            for frame in range(3):
                detection = self.observe_detections()
                if detection is None or (prior_stamp is not None
                                         and detection.header.stamp <= prior_stamp):
                    if not prior_boxes:
                        break
                    continue
                prior_stamp = detection.header.stamp
                observed = True
                boxes = sorted((bbox for bbox in detection.bboxes
                                if bbox.w > 0 and bbox.h > 0 and bbox.score > 0
                                and isinstance(bbox.label, str) and bbox.label.strip()
                                and (label is None or bbox.label == label)
                                and (shelf or self.placement_category(bbox.label) is not None)),
                               key=self.calc_score_bbox, reverse=True)
                if frame == 0 and not boxes:
                    break
                stable = [bbox for bbox in boxes if any(
                    other.label == bbox.label and self.bbox_iou(other, bbox) >= 0.5
                    for other in prior_boxes)]
                prior_boxes = boxes
                if not stable:
                    if frame > 0:
                        unresolved = True
                    continue
                base = self.get_robot_pose()
                if base is None:
                    return "observation_failed", None
                observations = []
                depth_failed = False
                for bbox in stable:
                    measurement, item = self.measure_object(bbox, detection, floor)
                    if measurement == "measurement_failed":
                        depth_failed = True
                    if item is not None:
                        observations.append((bbox, item))
                self.remember_distinct_pick_pairs(observations)
                blocked = self.attempted_observation_indices(observations, attempted, floor)
                candidates = []
                for index, (bbox, item) in enumerate(observations):
                    if index in blocked:
                        continue
                    center = item["center"]
                    if shelf and shelf_side is not None and not self.item_on_shelf_side(
                            center, shelf_side):
                        continue
                    reach = math.dist(base[:2], center[:2])
                    if (near_xy is not None and math.dist(center[:2], near_xy) > 0.28
                            or near_xy is None and (reach < 0.25
                                                    or (reach > 0.95 and not allow_approach))
                            or previous is not None and (math.dist(center, previous) > 0.18
                                or self.known_distinct_pick(bbox.label, center, previous))):
                        continue
                    item["label"] = bbox.label
                    item["placement"] = self.planned_placement(bbox.label) if not shelf else None
                    lateral = abs(-math.sin(base[2]) * (center[0] - base[0])
                                  + math.cos(base[2]) * (center[1] - base[1]))
                    if near_xy is not None:
                        priority = (math.dist(center[:2], near_xy),)
                    elif previous is not None:
                        priority = (math.dist(center, previous),)
                    else:
                        priority = (lateral > 0.32, reach) if floor else (reach > 0.95,)
                    candidates.append((priority, item))
                if candidates:
                    return "found", min(candidates, key=lambda entry: entry[0])[1]
                # 深度不足なら首を動かさず、次のRGB画像と対応する深度で測り直す。
                if not depth_failed and len(stable) == len(boxes):
                    break
                unresolved = True
            else:
                unresolved = unresolved or bool(prior_boxes)
        return ("empty" if observed and not unresolved else "observation_failed"), None

    def approach_pick_candidate(self, item, pose, floor):
        """62418450 向井悠人
        遠い物体へ床を確認しながら接近し、停止後の再計測で同じ個体を選ぶ。
        """
        center = np.asarray(item["center"], dtype=float)
        base = self.get_robot_pose()
        if base is None:
            return None
        direction = np.asarray(base[:2]) - center[:2]
        distance = float(np.linalg.norm(direction))
        if not math.isfinite(distance) or distance < 0.25:
            return None
        if distance <= 0.95:
            # すでに把持範囲に入った場合も、現在地から新しく測る。
            _, fresh = self.observe_pick_candidate(
                pose, floor, label=item["label"], previous=center)
            return fresh

        direction /= distance
        side = np.array([-direction[1], direction[0]])
        for offset in (0.0, -0.15, 0.15):
            stop = center[:2] + 0.64 * direction + offset * side
            heading = math.degrees(math.atan2(center[1] - stop[1], center[0] - stop[0]))
            if self.navigate_checked([*stop, heading], "grasp_approach"):
                _, fresh = self.observe_pick_candidate(
                    pose, floor, label=item["label"], previous=center)
                if fresh is not None:
                    return fresh
        return None

    def observe_detections(self):
        """62418450 向井悠人
        検出の通信失敗を一度だけ再試行し、空の結果と区別する。
        """
        for _ in range(2):
            try:
                return self.get_latest_detection()
            except (rospy.ROSException, rospy.ServiceException) as error:
                rospy.logwarn("Object observation failed: %s", error)
        return None

    @staticmethod
    def record_attempted_item(attempted, item):
        """62418450 向井悠人
        観測した個体の位置と、安全な床へ下ろした場合の平面位置を試行履歴に残す。
        """
        for location in (tuple(item["center"]), item.get("recovered_xy")):
            entry = (item["label"], location)
            if location is not None and entry not in attempted:
                attempted.append(entry)

    def try_pick_object(self, item, pose, floor, view=None, shelf=False, shelf_side=None):
        """62418450 向井悠人
        個体ごとの把持を最大2回に制限し、安全に復帰した後は再観測して試す。
        """
        for attempt in range(item.get("grasp_attempts", 0) + 1, 3):
            above = False if shelf else floor == (attempt == 1)
            item["grasp_attempts"] = attempt
            rospy.loginfo("Grasp attempt %d: %s from %s",
                          attempt, item["label"], "above" if above else "side")
            if self.grasp_measured_object(item, above, shelf):
                return True, item
            relocated = None
            if self.hand_state != "empty" or not self.arm_ready:
                if not self.recover_after_transport():
                    return False, item
                relocated = self.recovered_position
            if relocated is not None:
                item["recovered_xy"] = tuple(relocated)
            if attempt == 2:
                break
            rospy.loginfo("Reobserving %s after grasp failure (view=%s)", item["label"], view)
            if relocated is None:
                _, fresh = self.observe_pick_candidate(
                    pose, floor, label=item["label"], previous=item["center"], shelf=shelf,
                    shelf_side=shelf_side)
            else:
                pose, floor = "look_at_near_floor", True
                _, fresh = self.observe_pick_candidate(
                    pose, floor, label=item["label"], near_xy=relocated, shelf=shelf)
                shelf = False
                base = self.get_robot_pose()
                if fresh is not None and (base is None
                        or not 0.25 <= math.dist(base[:2], fresh["center"][:2]) <= 0.95):
                    fresh = None
            if fresh is None:
                break
            fresh["grasp_attempts"] = attempt
            item = fresh
        return False, item

    def reobserve_dropped_object(self, item):
        """62418450 向井悠人
        停止後に落下地点の床を観測し、物体を障害物として記録して再把持候補を返す。
        """
        self.confirm_navigation_finished()
        position = self.transport_loss["xy"]
        self.transport_loss = None
        observation = self.observe_floor()
        if observation is not None:
            self.update_floor_obstacles(observation, self.floor_obstacles, self.floor_clearance)
        if position is None:
            return None
        _, dropped = self.observe_pick_candidate(
            "look_at_near_floor", True, label=item["label"], near_xy=position,
            shelf=item.get("placement") is None)
        if dropped is None:
            rospy.logwarn("Task 1: dropped %s could not be confirmed", item["label"])
            return None
        occupied = self.floor_obstacle_cells(dropped["points"])
        if not len(occupied):
            occupied = ((np.floor(np.asarray(dropped["center"][:2]) / 0.05) + 0.5) * 0.05,)
        for point in occupied:
            cell = tuple(point)
            self.floor_obstacles.add(cell)
            self.floor_clearance.pop(cell, None)
        dropped["grasp_attempts"] = item["grasp_attempts"]
        rospy.loginfo("Task 1: reobserved dropped %s at %s", item["label"], dropped["center"])
        return dropped

    def execute_task1(self):
        """62418450 向井悠人
        床、机の順に片付け、観測失敗、動作失敗、探索終了を別々に返す。
        """
        rospy.loginfo("#### start Task 1 ####")
        phases = ((True, ("near_long_table_l",), "look_at_near_floor"),
                  (False, ("tall_table", "long_table_r"), "look_at_tall_table"))
        attempted = []
        self.distinct_pick_pairs.clear()
        placed = 0
        incomplete = False
        started = rospy.Time.now()
        limit = self.runtime_number("~task1_time_limit", 14 * 60)
        for floor, places, search_pose in phases:
            views = list(places)
            last_view = None
            status = "empty"
            for _ in range(12 * len(places)):
                if rospy.is_shutdown():
                    return "motion_failed"
                if limit > 0 and (rospy.Time.now() - started).to_sec() >= limit:
                    rospy.loginfo("Task 1 search budget used; placements=%d", placed)
                    return "time_budget"
                current = self.retry_robot_pose()
                if current is None:
                    return "motion_failed"
                status, item = "observation_failed", None
                if self.turn_base_toward(self.coordinates["positions"][places[0]][2]):
                    status, item = self.observe_pick_candidate(
                        search_pose, floor, attempted, allow_approach=True)
                if item is None:
                    while views:
                        name = min(views, key=lambda view: math.dist(
                            current[:2], self.coordinates["positions"][view][:2]))
                        views.remove(name)
                        if self.navigate_checked(self.coordinates["positions"][name], name):
                            last_view = name
                            break
                        rospy.logwarn("Task 1: observation waypoint not reached: %s", name)
                        incomplete = True
                        current = self.retry_robot_pose()
                        if current is None:
                            return "motion_failed"
                    else:
                        break
                    continue

                attempted.append((item["label"], tuple(item["center"])))
                if math.dist(current[:2], item["center"][:2]) > 0.95:
                    item = self.approach_pick_candidate(item, search_pose, floor)
                    if item is None:
                        rospy.logwarn("Task 1: could not safely approach or reobserve %s",
                                      attempted[-1][0])
                        incomplete = True
                        continue
                if limit > 0 and (rospy.Time.now() - started).to_sec() >= limit:
                    rospy.loginfo("Task 1 search budget used; placements=%d", placed)
                    return "time_budget"
                grasped, item = self.try_pick_object(item, search_pose, floor, last_view)
                self.record_attempted_item(attempted, item)
                while grasped:
                    completed = self.put_object_by_label(item["label"], item["placement"])
                    if completed or self.last_placement_result.get("released", False):
                        placed += 1
                    if self.transport_loss is None:
                        break
                    estimated_xy = self.transport_loss["xy"]
                    dropped = self.reobserve_dropped_object(item)
                    if dropped is None:
                        if estimated_xy is not None:
                            attempted.append((item["label"], tuple(estimated_xy)))
                        incomplete = True
                        break
                    attempted.append((dropped["label"], tuple(dropped["center"])))
                    base = self.get_robot_pose()
                    if (dropped["grasp_attempts"] >= 2 or base is None
                            or not 0.25 <= math.dist(base[:2], dropped["center"][:2]) <= 0.95):
                        break
                    if limit > 0 and (rospy.Time.now() - started).to_sec() >= limit:
                        break
                    grasped, item = self.try_pick_object(dropped, "look_at_near_floor", True)
                    self.record_attempted_item(attempted, item)
                if not grasped and (self.hand_state != "empty" or not self.arm_ready):
                    return "motion_failed"
                if self.hand_state != "empty" or not self.arm_ready:
                    if not self.recover_after_transport():
                        rospy.logerr("Cannot resume Task 1 after placement failure")
                        return "motion_failed"
                    if self.recovered_position is not None:
                        attempted.append((item["label"], tuple(self.recovered_position)))

            if status == "observation_failed":
                rospy.logwarn("Task 1: no usable %s observation after trying available views",
                              "floor" if floor else "table")
                incomplete = True

        result = "observation_failed" if incomplete else "completed"
        rospy.loginfo("#### finish Task 1 #### outcome=%s placement motions completed=%d",
                      result, placed)
        return result

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
    # A* で試す安全距離（広い順）。最小 0.32 = 半径 0.22 + 物体 0.05 + 余裕 0.05（> TASK2A_DANGER_DIST）
    TASK2A_SAFE_DISTS = (0.45, 0.38, 0.32)
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
        """62418450 向井悠人
        指示文から物体のラベルと、人物を指している方向またはIDを取り出す。
        """
        text = cls.normalize_text(instruction)
        words = " " + text + " "
        names = [(cls.normalize_text(label), label) for label in cls.DETECTOR_LABELS]
        names.extend(cls.LABEL_ALIASES.items())
        matches = [(len(phrase), label) for phrase, label in names
                   if any(" " + variant + " " in words
                          for variant in (phrase, phrase + "s", phrase + "es"))]
        longest = max((length for length, _ in matches), default=0)
        labels = {label for length, label in matches if length == longest}
        target_obj = next(iter(labels)) if len(labels) == 1 else None

        patterns = (
            r"\b(left|right) (?:person|human)\b",
            r"\b(?:person|human) (?:(?:on|to|at) (?:(?:the|your|my) )?)?(left|right|a|b)\b",
            r"\b(?:to|for) (?:the )?(left|right)(?: side)?$",
        )
        people = {key for pattern in patterns for key in re.findall(pattern, text)}
        if not people and "shelf" not in text.split():
            people = {word for word in text.split() if word in ("left", "right")}
        person = next(iter(people)) if len(people) == 1 else None
        rospy.loginfo("Task 2b request: object=%s person=%s", target_obj, person)
        return target_obj, person

    @classmethod
    def extract_shelf_side(cls, instruction):
        """62418450 向井悠人
        棚にかかる左右の指定を、受け渡す相手の左右と分けて抽出する。
        """
        text = cls.normalize_text(instruction)
        patterns = (
            r"\b(left|right)(?: (?:side|half))?(?: of(?: the)?)? shelf\b",
            r"\bshelf (?:s|on) (?:the )?(left|right)\b",
        )
        sides = {side for pattern in patterns for side in re.findall(pattern, text)}
        return next(iter(sides)) if len(sides) == 1 else None

    def wait_for_instruction(self, timeout):
        """62418450 向井悠人
        今回の実行で受け取った未処理の指示を、時間を区切って待つ。
        """
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            count = len(self.instruction_list)
            if count > self.instruction_cursor:
                self.instruction_cursor = count
                return self.instruction_list[count - 1]
            time.sleep(0.1)
        return None

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

    def deliver_to_target(self, target_obj, target_person, shelf_side=None):
        """62418450 向井悠人
        棚から指定物か代替品を取り、保持と到達を確認して指定の人物に渡す。
        """
        waypoint = self.PERSON_WAYPOINTS.get(target_person)
        if waypoint is None:
            return False
        if not self.move_placement_pose("look_at_shelf"):
            return False
        attempted = []
        self.distinct_pick_pairs.clear()
        for label in ((target_obj, None) if target_obj is not None else (None,)):
            side = shelf_side if label is not None else None
            status, item = self.observe_pick_candidate(
                "look_at_shelf", label=label, attempted=attempted, shelf=True, shelf_side=side)
            if status == "observation_failed":
                rospy.logwarn("Task 2b: shelf observation failed (requested label=%s)", label)
            if item is None:
                continue
            attempted.append((item["label"], tuple(item["center"])))
            grasped, item = self.try_pick_object(
                item, "look_at_shelf", False, shelf=True, shelf_side=side)
            if grasped:
                break
            self.record_attempted_item(attempted, item)
            if self.hand_state != "empty" or not self.arm_ready:
                return False
        else:
            rospy.logwarn("Task 2b: no deliverable object")
            return False

        rospy.loginfo("Task 2b carrying: requested=%s shelf_side=%s selected=%s person=%s fallback=%s",
                      target_obj, shelf_side, item["label"], target_person, label is None)
        if not self.navigate_checked(self.coordinates["positions"][waypoint], waypoint):
            if self.transport_loss is not None:
                self.reobserve_dropped_object(item)
            else:
                self.recover_after_transport()
            return False
        self.arm_ready = False
        if not self.move_placement_pose("deliver_to_human") or not self.verify_holding():
            self.recover_after_transport()
            return False
        self.wait_for_simulation(10.0, "Task 2b handoff")
        if not self.verify_holding() or not self.release_hand():
            self.recover_after_transport()
            return False
        self.arm_ready = self.move_placement_pose("all_neutral")
        if not self.arm_ready:
            self.recover_after_transport()
        rospy.loginfo("Task 2b handoff: requested=%s delivered=%s person=%s retreated=%s",
                      target_obj, item["label"], target_person, self.arm_ready)
        return self.arm_ready

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

    @staticmethod
    def repeats_unusable_edge(start, end, excluded):
        """62418450 向井悠人
        観測できなかった出発点の近くで、同じ方向へ進む線分を除外する。
        """
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return False
        for origin, blocked in excluded:
            if math.dist(start, origin) > 0.085:
                continue
            bx, by = blocked[0] - origin[0], blocked[1] - origin[1]
            if dx * bx + dy * by > 0.93 * length * math.hypot(bx, by):
                return True
        return False

    def plan_path_astar(self, start, goal, obstacles, safe_dist, bounds=None,
                        wall_radius=None, excluded=()):
        """62418450 向井悠人
        TASK2A_GRID 間隔の格子の上で、start から goal までの経路を A* で探す。
        - 障害物から safe_dist 未満の格子は通れない（start の格子だけは例外）
        - 障害物から TASK2A_SOFT_DIST 未満の格子は、近いほどコストを上乗せする
          （ぎりぎりを通らず、余裕のある経路を選ぶため）
        - 8近傍で移動し、ヒューリスティックはゴールまでの直線距離
        Task 2aまたは指定の探索範囲で壁と床の物体を避ける経路を返す。
        """
        cls = type(self)
        res = cls.TASK2A_GRID
        (x_min, x_max), (y_min, y_max) = bounds or (cls.TASK2A_PLAN_X, cls.TASK2A_PLAN_Y)
        if not all(x_min <= x <= x_max and y_min <= y <= y_max for x, y in (start, goal)):
            return None
        n_x = int(round((x_max - x_min) / res)) + 1
        n_y = int(round((y_max - y_min) / res)) + 1

        def to_cell(point):
            """62418450 向井悠人
            map 座標 (x, y) を格子の番号に変換する（範囲外は端の格子にする）
            """
            c_x = int(round((point[0] - x_min) / res))
            c_y = int(round((point[1] - y_min) / res))
            return (max(0, min(n_x - 1, c_x)), max(0, min(n_y - 1, c_y)))

        def to_xy(cell):
            """62418450 向井悠人
            格子の番号を map 座標 (x, y) に変換する
            """
            return (x_min + cell[0] * res, y_min + cell[1] * res)

        clearance_cache = {}

        def clearance(cell):
            """62418450 向井悠人
            格子の中心から一番近い障害物までの距離を返す（計算結果は覚えておく）
            """
            if cell not in clearance_cache:
                p_x, p_y = to_xy(cell)
                dists = [math.hypot(p_x - o_x, p_y - o_y) for o_x, o_y in obstacles]
                clearance_cache[cell] = min(dists) if dists else float("inf")
            return clearance_cache[cell]

        start_cell, goal_cell = to_cell(start), to_cell(goal)
        if clearance(goal_cell) < safe_dist or not self.wall_segment_clear(goal, goal, wall_radius):
            return None

        def heuristic(cell):
            """62418450 向井悠人
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
                    seg_start = start if cell == start_cell else to_xy(cell)
                    seg_end = goal if nxt == goal_cell else to_xy(nxt)
                    if (clr < safe_dist or self.repeats_unusable_edge(seg_start, seg_end, excluded)
                            or not self.is_segment_free(
                                seg_start, seg_end, obstacles, safe_dist, wall_radius)):
                        continue
                    step = math.hypot(d_x, d_y) * res
                    penalty = cls.TASK2A_CLEAR_WEIGHT * max(0.0, cls.TASK2A_SOFT_DIST - clr)
                    new_cost = cost[cell] + step * (1.0 + penalty)
                    if new_cost < cost.get(nxt, float("inf")):
                        cost[nxt] = new_cost
                        parent[nxt] = cell
                        heapq.heappush(open_list, (new_cost + heuristic(nxt), nxt))
        return None

    def load_wall_map(self):
        """62418450 向井悠人
        ROSの静的地図から壁・未知領域までの距離を計算する。
        """
        import cv2
        from nav_msgs.msg import OccupancyGrid

        try:
            grid = self.bounded_query("/static_map", lambda: self.request_service(
                "/static_map", GetMap), 2.5, retry_readonly=True).map
        except (TimeoutError, rospy.ROSException, rospy.ServiceException):
            try:
                grid = self.bounded_query("/map", lambda: rospy.wait_for_message(
                    "/map", OccupancyGrid, timeout=2.0), 2.5, retry_readonly=True)
            except (TimeoutError, rospy.ROSException) as error:
                rospy.logwarn("Static wall map unavailable: %s", error)
                return False
        info = grid.info
        if (grid.header.frame_id.lstrip("/") != "map"
                or not math.isfinite(info.resolution) or info.resolution <= 0
                or info.width <= 0 or info.height <= 0
                or len(grid.data) != info.width * info.height):
            rospy.logwarn("Static wall map has invalid coordinates")
            return False
        occupancy = np.asarray(grid.data).reshape(info.height, info.width)
        free = ((occupancy >= 0) & (occupancy < 50)).astype(np.uint8)
        try:
            distance = cv2.distanceTransform(np.pad(free, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        except cv2.error as error:
            rospy.logwarn("Cannot process static wall map: %s", error)
            return False
        q = info.origin.orientation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:2, :2]
        origin = np.array([info.origin.position.x, info.origin.position.y])
        if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
            rospy.logwarn("Static wall map has a nonfinite origin")
            return False
        self.wall_map = (distance * info.resolution, rotation, origin, info.resolution)
        return True

    def wall_segment_clear(self, start, end, radius=None):
        """62418450 向井悠人
        台車の幅を含め、線分が地図上の壁・扉枠・未知セルを避けるか判定する。
        """
        if self.wall_map is None:
            return False
        distances, rotation, origin, resolution = self.wall_map
        margin = self.ROBOT_RADIUS + 0.07 if radius is None else radius
        samples = np.linspace(start, end, max(2, math.ceil(math.dist(start, end) /
                                                           (resolution / 2)) + 1))
        indices = np.floor((samples - origin).dot(rotation) / resolution).astype(int)
        height, width = distances.shape
        return all(0 <= x < width and 0 <= y < height
                   and distances[y, x] >= margin + resolution * math.sqrt(2)
                   for x, y in indices)

    def is_segment_free(self, seg_start, seg_end, obstacles, safe_dist, wall_radius=None):
        """62418450 向井悠人
        台車や保持物の幅を含め、線分全体が壁と床上の障害物を避けるか調べる。
        """
        return (self.wall_segment_clear(seg_start, seg_end, wall_radius)
                and all(self.dist_point_to_segment(obs, seg_start, seg_end) >= safe_dist
                        for obs in obstacles))

    def simplify_path(self, path, obstacles, safe_dist, excluded=()):
        """62418450 向井悠人
        壁、物体、観測できなかった方向を避けて経路を単純化する。
        """
        result = [path[0]]
        index = 0
        while index < len(path) - 1:
            following = next((i for i in range(len(path) - 1, index, -1)
                              if not self.repeats_unusable_edge(path[index], path[i], excluded)
                              and self.is_segment_free(path[index], path[i], obstacles, safe_dist)),
                             None)
            if following is None:
                return []
            result.append(path[following])
            index = following
        return result

    def plan_next_point(self, current, goal, obstacles, excluded=()):
        """62418450 向井悠人
        壁と進めなかった方向を除外してA*を引き直し、次の短い移動点を返す。
        """
        for safe_dist in self.TASK2A_SAFE_DISTS:
            path = self.plan_path_astar(current, goal, obstacles, safe_dist, excluded=excluded)
            if not path:
                continue
            corners = self.simplify_path(path, obstacles, safe_dist, excluded)
            if len(corners) < 2:
                continue
            distance = math.dist(current, corners[1])
            if distance < 0.035:
                continue
            ratio = min(1.0, self.TASK2A_STEP_MAX / distance)
            return tuple(a + (b - a) * ratio for a, b in zip(current, corners[1]))
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

    def retry_robot_pose(self):
        """62418450 向井悠人
        復帰や探索の区切りで自己位置を取得できなければ一度だけ取り直す。
        """
        pose = self.get_robot_pose()
        return self.get_robot_pose() if pose is None and not rospy.is_shutdown() else pose

    def get_robot_pose(self, stamped=False):
        """62418450 向井悠人
        有限の自己位置を返し、速度制御の停止確認にはTFの時刻も返す。
        """
        try:
            transform = self.lookup_transform(
                "map", "base_footprint", rospy.Time(0), 1.0)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            rospy.logwarn("cannot get robot pose from tf")
            return None
        stamp = transform.header.stamp
        if not self.fresh_tf_stamp(stamp):
            rospy.logwarn("Robot pose TF is out of date")
            return None
        trans = transform.transform
        rot = trans.rotation
        yaw = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                         1.0 - 2.0 * (rot.y ** 2 + rot.z ** 2))
        pose = (trans.translation.x, trans.translation.y, yaw)
        if not all(math.isfinite(value) for value in pose):
            rospy.logwarn("Robot pose contains nonfinite coordinates")
            return None
        return (pose, stamp) if stamped else pose

    def count_task2a_collisions(self):
        """62401738 いしい
        これまでに受け取った Task 2a の障害物との衝突メッセージの数を返す
        """
        return len([msg for msg in self.detection_list if "task2a" in msg])

    @staticmethod
    def stop_base():
        """62418450 向井悠人
        台車へ停止を指令し、指令を送れなければ後続動作を禁止する。
        """
        try:
            omni_base.cmd_vel(0.0, 0.0, 0.0)
        except Exception as error:
            raise NavigationUnconfirmedError("Cannot send a base stop command") from error

    def confirm_manual_stop(self):
        """62418450 向井悠人
        速度制御に例外が出た後、停止指令と自己位置から静止を確認する。
        """
        first = self.get_robot_pose(stamped=True)
        time.sleep(0.25)
        second = self.get_robot_pose(stamped=True)
        if first is None or second is None or second[1] <= first[1]:
            raise NavigationUnconfirmedError("Cannot confirm the base stopped after a velocity error")
        before, after = first[0], second[0]
        yaw_delta = after[2] - before[2]
        rotation = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
        if math.dist(before[:2], after[:2]) > 0.01 or rotation > math.radians(2):
            raise NavigationUnconfirmedError("Base still moving after a velocity error")

    @classmethod
    def calc_base_velocity(cls, pose, target):
        """62401738 いしい
        今の姿勢 pose (x, y, yaw) から target (x, y) へ直進するための速度指令を計算する。
        - 位置: map 座標での誤差に比例した速度（最大 TASK2A_MAX_SPEED）で、目標へ一直線に向かう
        - 向き: TASK2A_YAW を保つ（全方向台車なので、向きを変えずに横にも進める）
        map 座標の速度をロボット座標に回転して (vx, vy, w[deg/s]) を返す。
        """
        err_x, err_y = target[0] - pose[0], target[1] - pose[1]
        dist = math.hypot(err_x, err_y)
        speed = min(cls.TASK2A_MAX_SPEED, cls.TASK2A_GAIN * dist)
        map_vx = err_x / dist * speed if dist > 0 else 0.0
        map_vy = err_y / dist * speed if dist > 0 else 0.0
        cos_y, sin_y = math.cos(pose[2]), math.sin(pose[2])
        robot_vx = cos_y * map_vx + sin_y * map_vy
        robot_vy = -sin_y * map_vx + cos_y * map_vy
        yaw_err = math.atan2(math.sin(math.radians(cls.TASK2A_YAW) - pose[2]),
                             math.cos(math.radians(cls.TASK2A_YAW) - pose[2]))
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
            if nxt < cls.TASK2A_DANGER_DIST and nxt < now:
                return True
        return False

    def retreat_from_collision(self, target, obstacles, contact):
        """62418450 向井悠人
        進む方向の床を確認し、近接または接触した物体から離れる区間を低速で移動する。
        """
        pose = self.get_robot_pose()
        if pose is None:
            return "no_tf"
        start = pose[:2]
        if math.dist(start, target) < self.TASK2A_GOAL_TOL:
            return "reached"
        if not self.wall_segment_clear(start, target):
            return "wall"
        dx, dy = target[0] - start[0], target[1] - start[1]
        pan = self.floor_pan_toward(pose, target)
        observation = self.observe_floor(pan, self.TASK2A_HEAD_TILT)
        if observation is None:
            return "unobserved"
        self.update_floor_obstacles(observation, self.floor_obstacles, self.floor_clearance)
        known = self.floor_obstacles | {tuple(point) for point in obstacles}
        separating = {point for point in known
                      if math.dist(point, start) < self.ROBOT_RADIUS + 0.125
                      and (point[0] - start[0]) * dx + (point[1] - start[1]) * dy <= 0}
        if not self.observed_segment_clear(start, target, (observation[0], known - separating)):
            return "too_close"
        return self.move_straight_to(target, known, separating=separating, contact=contact)

    def move_straight_to(self, target, obstacles, separating=None, contact=None):
        """62418450 向井悠人
        壁、床、接触を監視して短く移動し、低速後退でも監視を維持する。
        """
        self.confirm_navigation_finished()
        collisions = self.count_task2a_collisions()
        collision_cursor = len(self.detection_list)
        pose = self.get_robot_pose()
        if pose is None:
            return "no_tf"
        start = pose[:2]
        distance = math.dist(target, start)
        retreating = separating is not None
        hazards = obstacles if not retreating else {tuple(point) for point in obstacles} - separating
        max_speed = 0.08 if retreating else self.TASK2A_MAX_SPEED
        deadline = time.monotonic() + distance / max_speed * 3.0 + 5.0
        try:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                pose = self.get_robot_pose()
                if pose is None:
                    return "no_tf"
                if retreating:
                    contacts = self.detection_list[collision_cursor:]
                    collision_cursor += len(contacts)
                    if any(msg != contact or math.dist(pose[:2], start) >= 0.08 for msg in contacts):
                        return "collision"
                elif self.count_task2a_collisions() > collisions:
                    return "collision"
                if math.dist(target, pose[:2]) < self.TASK2A_GOAL_TOL:
                    return "reached"
                if not self.wall_segment_clear(pose[:2], target):
                    return "wall"
                if self.is_approaching_obstacle(pose, target, hazards):
                    return "too_close"
                corridor = (start, target)
                clear = (self.corridor_still_clear(corridor, separating) if retreating
                         else self.corridor_still_clear(corridor))
                if not clear:
                    return "too_close"
                vx, vy, omega = self.calc_base_velocity(pose, target)
                if retreating:
                    scale = min(1.0, max_speed / max(math.hypot(vx, vy), 1e-6))
                    vx, vy, omega = vx * scale, vy * scale, 0.0
                omni_base.cmd_vel(vx, vy, omega)
                time.sleep(self.TASK2A_CONTROL_DT)
            return "timeout"
        except Exception as error:
            rospy.logwarn("Task 2a velocity control failed: %s", error)
        finally:
            self.stop_base()
        self.confirm_manual_stop()
        return "motion_error"

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
        """62418450 向井悠人
        接触位置を記録し、戻る方向を観測して安全に離れられたか返す。
        """
        pose = self.get_robot_pose()
        if pose is None:
            return False
        hit = self.add_hit_obstacle(obstacles, hits, self.estimate_hit_position(pose, direction))
        rospy.logwarn("task2a: hit an obstacle near (%.2f, %.2f). escape.", hit[0], hit[1])
        back = safe_pos
        contact = next((msg for msg in reversed(self.detection_list) if "task2a" in msg), None)
        for attempt in range(self.ESCAPE_MAX_TRY):
            result = self.retreat_from_collision(back, obstacles, contact)
            before = self.count_task2a_collisions()
            self.wait_for_simulation(self.ESCAPE_WAIT, "Task 2a retreat")
            if result == "reached" and self.count_task2a_collisions() == before:
                rospy.loginfo("task2a: escaped (try %d)", attempt + 1)
                return True
            pose = self.get_robot_pose() or (back[0], back[1], 0.0)
            back = (pose[0] - direction[0] * self.ESCAPE_BACK_DIST,
                    pose[1] - direction[1] * self.ESCAPE_BACK_DIST)
        rospy.logwarn("task2a: could not escape")
        return False

    def leave_obstacle_clearance(self, current, obstacles, excluded):
        """62418450 向井悠人
        通常の安全距離内にいるとき、近い物体すべてから離れる観測済みの一歩を試す。
        """
        clearance = self.ROBOT_RADIUS + 0.125
        near = {point for point in obstacles if math.dist(point, current) < clearance}
        if not near:
            return None, None
        candidates = []
        for angle in np.arange(8) * (math.pi / 4):
            dx, dy = 0.30 * math.cos(angle), 0.30 * math.sin(angle)
            target = (current[0] + dx, current[1] + dy)
            if (not self.TASK2A_PLAN_X[0] <= target[0] <= self.TASK2A_PLAN_X[1]
                    or not self.TASK2A_PLAN_Y[0] <= target[1] <= self.TASK2A_PLAN_Y[1]
                    or self.repeats_unusable_edge(current, target, excluded)):
                continue
            if any((point[0] - current[0]) * dx + (point[1] - current[1]) * dy > 1e-9
                   for point in near):
                continue
            if self.is_segment_free(current, target, obstacles - near, clearance):
                space = min(math.dist(point, target) for point in near)
                if space >= clearance:
                    candidates.append((space, target))
        for _, target in sorted(candidates, reverse=True)[:3]:
            result = self.retreat_from_collision(target, obstacles, contact=None)
            pose = self.get_robot_pose()
            if (result not in ("unobserved", "too_close", "wall")
                    or pose is None or math.dist(current, pose[:2]) > 0.025):
                return target, result
        return None, None

    def execute_avoid_blocks(self):
        """62418450 向井悠人
        障害物の記録を使って短く進み、観測不足の区間は再計測してから判断する。
        """
        history = []
        attempted = set()
        excluded_edges = {}
        observation_misses = {}
        obstacles = self.floor_obstacles
        cleared = self.floor_clearance
        for iteration in range(self.TASK2A_MAX_MOVES * 2):
            pose = self.get_robot_pose()
            if pose is None or rospy.is_shutdown():
                return False
            current = pose[:2]
            for pan in ((0.0, -0.7, 0.7) if iteration == 0 else (0.0,)):
                observation = self.observe_floor(pan, -1.0)
                if observation is not None:
                    self.update_floor_obstacles(observation, obstacles, cleared)
            excluded = tuple(excluded_edges.values())
            goal, primary = self.find_free_goal(current, obstacles, excluded)
            if goal is not None and primary is None:
                return True
            escape_target, escape_result = self.leave_obstacle_clearance(current, obstacles, excluded)
            if escape_target is not None:
                rospy.logwarn("Task 2a separation from nearby obstacles: %s", escape_result)
                history.clear()
                if escape_result == "reached":
                    self.remember_navigation_checkpoint()
                    observation_misses.clear()
                else:
                    edge = tuple(round(value, 1) for value in (*current, *escape_target))
                    excluded_edges[edge] = (current, escape_target)
                continue
            alternatives = [(current[0] + delta, current[1]) for delta in (-0.19, 0.19)]
            if history:
                alternatives.append(history[-1])
            moved = False
            needs_replan = False
            for target in [primary] + alternatives:
                if target is None:
                    continue
                edge = tuple(round(value, 1) for value in (*current, *target))
                reversing = bool(history) and target == history[-1]
                if edge in attempted and edge not in excluded_edges:
                    excluded_edges[edge] = (current, target)
                    needs_replan = True
                    continue
                if (edge in excluded_edges or self.repeats_unusable_edge(current, target, excluded)
                        or not self.is_segment_free(
                            current, target, obstacles, self.TASK2A_SAFE_DISTS[-1])):
                    continue
                pan = self.side_to_look(pose, target) or 0.0
                observation = self.observe_floor(pan, -1.0)
                if observation is not None:
                    self.update_floor_obstacles(observation, obstacles, cleared)
                if not self.is_segment_free(current, target, obstacles,
                                            self.TASK2A_SAFE_DISTS[-1]):
                    needs_replan = True
                    continue
                if not reversing and not (
                        observation is not None
                        and self.observed_segment_clear(current, target, observation)):
                    misses = observation_misses.get(edge, 0) + 1
                    observation_misses[edge] = misses
                    if misses >= 2:
                        excluded_edges[edge] = (current, target)
                    needs_replan = True
                    continue
                attempted.add(edge)
                result = self.move_straight_to(target, obstacles)
                if result != "reached":
                    excluded_edges[edge] = (current, target)
                rospy.loginfo("Task 2a segment %s: %s", target, result)
                moved = True
                if result == "reached":
                    self.remember_navigation_checkpoint()
                    observation_misses.clear()
                    if reversing:
                        history.pop()
                    else:
                        history.append(current)
                elif result == "collision":
                    contact = next((msg for msg in reversed(self.detection_list)
                                    if "task2a" in msg), None)
                    recovery = self.retreat_from_collision(current, obstacles, contact)
                    rospy.logwarn("Task 2a retreat after contact: %s", recovery)
                    if recovery == "reached":
                        self.remember_navigation_checkpoint()
                break
            if not moved and not needs_replan:
                rospy.logwarn("Task 2a: alternate views and known retreat exhausted")
                return False
        return False

    def find_free_goal(self, current, obstacles, excluded=()):
        """62418450 向井悠人
        到達済みの候補を先に確認し、未到達なら安全な候補への最初の一歩を選ぶ。
        """
        gx, gy = self.TASK2A_GOAL
        candidates = []
        checked = set()
        for distance in self.TASK2A_SAFE_DISTS:
            for offset in (0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4):
                point = (gx + offset, gy)
                if (point in checked or not self.TASK2A_PLAN_X[0] <= point[0] <= self.TASK2A_PLAN_X[1]
                        or not self.is_segment_free(point, point, obstacles, distance)):
                    continue
                checked.add(point)
                if (math.dist(current, point) <= 0.09
                        and self.is_segment_free(current, current, obstacles,
                                                 self.ROBOT_RADIUS + 0.125)):
                    return point, None
                candidates.append(point)
        for point in candidates:
            step = self.plan_next_point(current, point, obstacles, excluded)
            if step is not None:
                return point, step
        return None, None

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

    def execute_task2a(self):
        """62418450 向井悠人
        障害物回避を実行し、ゴール到達と接触を別々に記録する。
        """
        start = len(self.detection_list)
        reached = False
        standby = self.coordinates["positions"]["standby_2a"]
        if self.navigate_checked(standby, "standby_2a"):
            if self.wall_map is None:
                self.load_wall_map()
            if self.wall_map is not None:
                reached = self.execute_avoid_blocks()
            else:
                rospy.logwarn("Task 2a: cannot use A* without a map of the walls")
                reached = self.navigate_checked(
                    self.coordinates["positions"]["go_throw_2a"], "go_throw_2a")
        hits = self.detection_list[start:]
        rospy.loginfo("Task 2a result: reached=%s contacts=%s", reached, json.dumps(hits))
        return reached

    def execute_task2b(self):
        """62418450 向井悠人
        棚へ安全に到達して新しい指示を受け、人物が確定した場合に配達する。
        """
        rospy.loginfo("#### start Task 2b ####")
        if not self.navigate_checked(self.coordinates["positions"]["shelf"], "shelf"):
            rospy.logwarn("Task 2b: shelf not reached")
            return False
        for _ in range(2):
            instruction = self.wait_for_instruction(self.TASK2B_WAIT_SEC)
            if instruction is None:
                rospy.logwarn("Task 2b: no instruction at the shelf")
                continue
            rospy.loginfo("Task 2b instruction: %s", instruction)
            label, person = self.extract_target_obj_and_person(instruction)
            if person is not None:
                side = self.extract_shelf_side(instruction)
                return self.deliver_to_target(label, person, shelf_side=side)
            rospy.logwarn("Task 2b: could not confirm the intended person")
        return False

    def run(self):
        """62418450 向井悠人
        センサと初期姿勢を確認し、Task 1から配達まで順に実行する。
        """
        self.confirm_navigation_finished()
        if self.hand_state != "empty":
            return
        try:
            self.read_placement_cloud(rospy.Time.now(), stride=8)
        except (ValueError, rospy.ROSException, tf2_ros.LookupException,
                tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as error:
            rospy.logerr("Cannot start without depth: %s", error)
            return
        if self.retry_robot_pose() is None:
            rospy.logerr("Cannot start without localization")
            return
        try:
            self.bounded_query("detector server", lambda: rospy.wait_for_service(
                "detection/get_object_detection", timeout=30.0), 30.0)
        except (TimeoutError, rospy.ROSException) as error:
            rospy.logwarn("RGB detector unavailable at startup: %s", error)
        self.arm_ready = self.move_placement_pose("all_neutral")
        if not self.arm_ready:
            rospy.logerr("Cannot reach initial pose")
            return
        self.load_wall_map()
        self.remember_navigation_checkpoint()
        task1_result = self.execute_task1()
        if task1_result == "motion_failed" or self.hand_state != "empty" or not self.arm_ready:
            rospy.logerr("Task 1 could not finish with an empty hand and safe arm pose")
            return
        if task1_result == "observation_failed":
            rospy.logwarn("Continuing to Task 2 with incomplete Task 1 observations")
        self.instruction_cursor = len(self.instruction_list)
        self.execute_task2a()
        self.execute_task2b()


def main():
    """62418450 向井悠人
    WRSノードを起動し、台車や把持の終了が未確定なら後続動作を中断する。
    """
    rospy.init_node('main_controller')
    try:
        ctrl = WrsMainController()
    except (ImportError, OSError, ValueError, TypeError, KeyError,
            rospkg.ResourceNotFound, rospy.ROSException) as error:
        rospy.logerr("Cannot initialize node B: %s", error)
        return
    try:
        rospy.loginfo("node initialized [%s]", rospy.get_name())

        # タスクの実行モードを確認する
        if rospy.get_param("~test_mode", default=False) is True:
            rospy.loginfo("#### start with TEST mode. ####")
            ctrl.check_positions()
        else:
            rospy.loginfo("#### start with NORMAL mode. ####")
            ctrl.run()

    except (NavigationUnconfirmedError, ManipulationUnconfirmedError) as error:
        rospy.logerr("Cannot continue with unresolved robot control: %s", error)
    except SimulationClockUnavailableError as error:
        rospy.logerr("Cannot continue without a progressing simulation clock: %s", error)
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
