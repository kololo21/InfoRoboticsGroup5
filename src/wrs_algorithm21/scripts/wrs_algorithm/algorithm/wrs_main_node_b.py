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
import traceback
import time
import threading
import actionlib
import numpy as np
import rospy
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
        self.detection_job = None
        self.depth_clouds = deque(maxlen=8)
        self.hand_state = "empty"
        self.arm_ready = True
        self.held_geometry = None
        self.wall_map = None

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses = self.load_json(self.get_path(["config", "poses.json"]))
        self.load_placement_overrides()

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

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

    def confirm_navigation_finished(self):
        """62418450 向井悠人
        前の台車ゴールの終了を確認し、未確定なら例外で後続動作を防ぐ。
        """
        if not self.navigation_pending:
            return
        client = omni_base.NAVCLIENT
        finished = client.wait_for_result(rospy.Duration(2.0))
        state = client.get_state()
        if not finished or state not in self.NAVIGATION_TERMINAL_STATES:
            raise NavigationUnconfirmedError(
                "Previous navigation still unresolved: state={} text={}".format(
                    state, client.get_goal_status_text()))
        self.navigation_pending = False
        rospy.loginfo("Previous navigation ended: state=%s text=%s",
                      state, client.get_goal_status_text())

    def navigate_to_pose(self, pos, name, timeout_param, corridor=None):
        """62418450 向井悠人
        目標へ台車を動かし、移動結果や床監視による中止理由を記録する。
        """
        timeout = float(rospy.get_param(timeout_param, 60.0))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("{} must be positive and finite".format(timeout_param))
        self.confirm_navigation_finished()

        client = omni_base.NAVCLIENT
        if not client.wait_for_server(rospy.Duration(2.0)):
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
            if result != "done":
                reason = "interrupted" if rospy.is_shutdown() else result
                rospy.logerr("Navigation %s %s: state=%s text=%s",
                             name, reason, client.get_state(), client.get_goal_status_text())
                client.cancel_goal()
                self.confirm_navigation_finished()
                return False
            state = client.get_state()
        except rospy.ROSException as error:
            raise NavigationUnconfirmedError(
                "Cannot confirm navigation to {}: {}".format(name, error)) from error

        if state not in self.NAVIGATION_TERMINAL_STATES:
            raise NavigationUnconfirmedError(
                "Navigation {} ended without a confirmed result: state={} text={}".format(
                    name, state, client.get_goal_status_text()))
        self.navigation_pending = False
        if state != GoalStatus.SUCCEEDED:
            rospy.logerr("Navigation %s failed: state=%s text=%s",
                         name, state, client.get_goal_status_text())
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

    def request_detection(self, job):
        """62418450 向井悠人
        検出サービスの応答を、呼び出し側が時間を区切って待てる形で返す。
        """
        try:
            job["response"] = self.detection_clt(GetObjectDetectionRequest())
        except Exception as error:
            job["error"] = error
        finally:
            job["done"].set()

    def point_head(self, pan, tilt):
        """62418450 向井悠人
        腕の姿勢を保ったまま頭部カメラを指定方向へ向ける。
        """
        head = whole_body.HEAD_CMD
        head.set_joint_value_target({"head_pan_joint": pan, "head_tilt_joint": tilt})
        try:
            return bool(head.go(wait=True))
        finally:
            head.stop()

    def measure_object(self, bbox, detection, floor=False):
        """62418450 向井悠人
        RGBと深度を照合し、計測結果、対象外の高さ、深度不足を区別する。
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
        points = np.asarray(list(point_cloud2.read_points(
            cloud, field_names=("x", "y", "z"), skip_nans=False, uvs=uvs)), dtype=float)
        if points.shape != (len(pixels), 3):
            return "measurement_failed", None
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        points, pixels = points[valid], pixels[valid]
        if len(points) < 15:
            return "measurement_failed", None

        try:
            transform = self.tf_buffer.lookup_transform(
                "map", cloud.header.frame_id, cloud.header.stamp, rospy.Duration(1.0)).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn("Cannot measure %s: %s", bbox.label, error)
            return "measurement_failed", None
        rotation, translation = self.transform_arrays(transform)
        world = points.dot(rotation.T) + translation
        height = world[:, 2]
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

    def finger_gap(self):
        """62418450 向井悠人
        左右の指先の距離を同じTFで求め、取得できないときは不明を返す。
        """
        try:
            pos = self.tf_buffer.lookup_transform(
                "hand_l_finger_tip_frame", "hand_r_finger_tip_frame",
                rospy.Time(0), rospy.Duration(1.0)).transform.translation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        return math.sqrt(pos.x ** 2 + pos.y ** 2 + pos.z ** 2)

    def verify_holding(self):
        """62418450 向井悠人
        指先間隔から保持の手がかりを確認し、結果をログに残す。
        """
        gap = self.finger_gap()
        empty_gap = float(rospy.get_param("~empty_finger_gap", 0.015))
        holding = gap is not None and empty_gap < gap <= 0.12
        rospy.loginfo("Grasp gap: %s m, holding=%s", gap, holding)
        return holding

    def close_for_grasp(self):
        """62418450 向井悠人
        力制御で指を閉じ、時間切れでは指令の中止を確認する。
        """
        from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortGoal

        client = actionlib.SimpleActionClient(
            "/hsrb/gripper_controller/grasp", GripperApplyEffortAction)
        if not client.wait_for_server(rospy.Duration(3.0)):
            rospy.logwarn("Gripper effort controller is unavailable")
            return False
        self.hand_state = "unknown"
        client.send_goal(GripperApplyEffortGoal(effort=-0.5))
        if not client.wait_for_result(rospy.Duration(5.0)):
            client.cancel_goal()
            if (not client.wait_for_result(rospy.Duration(2.0))
                    or client.get_state() not in self.NAVIGATION_TERMINAL_STATES):
                raise ManipulationUnconfirmedError("Cannot confirm gripper cancellation")
            return False
        state = client.get_state()
        if state not in self.NAVIGATION_TERMINAL_STATES:
            raise ManipulationUnconfirmedError("Unconfirmed gripper state {}".format(state))
        return state == GoalStatus.SUCCEEDED

    def hand_pose(self):
        """62418450 向井悠人
        現在の手のひらの地図座標と向きを取得する。
        """
        return self.tf_buffer.lookup_transform(
            "map", "hand_palm_link", rospy.Time(0), rospy.Duration(1.0)).transform

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

    def move_hand(self, xyz, orientation, linear=False):
        """62418450 向井悠人
        手先を地図座標へ動かし、直線移動では衝突のない完全な軌道だけを実行する。
        """
        self.confirm_navigation_finished()
        commander = whole_body.WHOLE_BODY_CMD
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = xyz
        pose.pose.orientation = orientation
        if not linear:
            try:
                commander.set_pose_target(pose)
                plan = commander.plan()
                ok, trajectory = plan[:2] if isinstance(plan, tuple) else (True, plan)
                return (bool(ok) and self.hand_path_keeps_base_still(trajectory)
                        and bool(commander.execute(trajectory, wait=True)))
            finally:
                commander.stop()
                commander.clear_pose_targets()

        previous_frame = commander.get_pose_reference_frame()
        try:
            commander.set_pose_reference_frame("map")
            cartesian = commander.compute_cartesian_path
            kwargs = {"avoid_collisions": True}
            if "jump_threshold" in inspect.signature(cartesian).parameters:
                kwargs["jump_threshold"] = 0.0
            trajectory, fraction = cartesian([pose.pose], 0.01, **kwargs)
            if (not math.isfinite(fraction) or fraction < 1.0
                    or not trajectory.joint_trajectory.points):
                rospy.logwarn("Incomplete hand path: %.3f", fraction)
                return False
            return (self.hand_path_keeps_base_still(trajectory)
                    and bool(commander.execute(trajectory, wait=True)))
        finally:
            commander.stop()
            commander.set_pose_reference_frame(previous_frame)

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
        指の開放が成功したときに保持状態と物体寸法の記録を解消する。
        """
        self.hand_state = "unknown"
        if not gripper.command(1):
            return False
        self.hand_state = "empty"
        self.held_geometry = None
        return True

    def grasp_measured_object(self, item, approach_from_above, shelf=False):
        """62418450 向井悠人
        測った中心に指を合わせ、少し持ち上げて保持を確認し、安全に腕を引く。
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
        yaw = math.degrees(math.atan2(toward_base[1], toward_base[0]))
        offset = np.array([0.0, 0.0, self.HAND_PALM_Z_OFFSET] if approach_from_above
                          else [*(toward_base * self.HAND_PALM_OFFSET), 0.0])
        contact = center + offset
        approach = contact + np.array([0.0, 0.0, 0.10] if approach_from_above
                                      else [*(toward_base * 0.15), 0.0])
        orientation = quaternion_from_euler(yaw, -160 if approach_from_above else -100, 0)
        if not gripper.command(1) or not self.move_hand(approach, orientation):
            self.arm_ready = self.move_placement_pose("all_neutral")
            return False
        if not self.move_hand(contact, orientation, linear=True):
            self.arm_ready = (self.move_hand(approach, orientation, linear=True)
                              and self.move_placement_pose("all_neutral"))
            return False
        if not self.close_for_grasp() or not self.verify_holding():
            self.arm_ready = (self.return_grasp_to_surface(contact, approach, orientation)
                              and self.move_placement_pose("all_neutral"))
            return False

        rotation, translation = self.transform_arrays(self.hand_pose())
        local = (item["points"] - translation).dot(rotation)
        low, high = np.percentile(local, [3, 97], axis=0)
        self.held_geometry = {"offset": (low + high) / 2,
                              "size": np.maximum(high - low, 0.015) + 0.006}
        lift = np.array([0.0, 0.0, 0.03 if shelf else 0.07])
        if not self.move_hand(contact + lift, orientation, linear=True) or (
                not self.verify_holding() and not self.verify_holding()):
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
        self.arm_ready = self.move_placement_pose("all_neutral")
        return self.arm_ready

    def get_latest_detection(self):
        """62418450 向井悠人
        新しいRGB画像の検出を要求し、通信失敗と空の検出結果を区別する。
        """
        if self.detection_job is not None and not self.detection_job["done"].is_set():
            raise rospy.ServiceException("Previous detection request is still running")
        job = {"done": threading.Event()}
        self.detection_job = job
        threading.Thread(target=self.request_detection, args=(job,), daemon=True).start()
        if not job["done"].wait(5.0):
            raise rospy.ServiceException("Object detection timed out")
        if "error" in job:
            raise rospy.ServiceException(str(job["error"]))
        response = job["response"]
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
        pos = self.coordinates["positions"][waypoint]
        for attempt in range(2):
            arrived = (self.turn_base_toward(pos[2]) if attempt == 0 else
                       self.navigate_checked(pos, waypoint, "~placement_navigation_timeout"))
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
        base = self.tf_buffer.lookup_transform(
            "map", "base_footprint", rospy.Time(0), rospy.Duration(2.0)).transform.translation
        distance = float(rospy.get_param("~placement_standoff", 0.65))
        hover = float(rospy.get_param("~placement_hover_distance", 0.12))
        if any(not math.isfinite(value) or value <= 0 for value in
               (distance, hover)):
            raise ValueError("Placement distances must be positive and finite")
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

    def held_object_fits(self, target):
        """62418450 向井悠人
        開放直前に保持物の位置を計算し、縁との余裕と下降高さを確認する。
        """
        if not self.verify_holding():
            self.hand_state = "unknown"
            return False
        rotation, translation = self.transform_arrays(self.hand_pose())
        center = translation + rotation.dot(self.held_geometry["offset"])
        extent = np.abs(rotation).dot(self.held_geometry["size"]) / 2
        opening = np.array([target["opening_x"], target["opening_y"]])
        drop = np.array([target["drop_x"], target["drop_y"]])
        bottom = center[2] - extent[2]
        return (np.all(np.abs(center[:2] - opening) + extent[:2] + 0.010
                       < target["opening_half"])
                and np.linalg.norm(center[:2] - drop) <= 0.015
                and target["rim_z"] + 0.003 <= bottom <= target["rim_z"] + 0.055)

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
            record["stage"] = "lower"
            if self.move_over_container(place, linear=True):
                record["stage"] = "verify_opening"
                if self.held_object_fits(target):
                    return True
            if not self.move_over_container(place, hover=True, linear=True):
                return False
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
        try:
            rospy.wait_for_service(service, timeout=1.0)
            path = rospy.ServiceProxy(service, GetPlan)(start, goal, 0.0).plan.poses
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logwarn("Base path unavailable: %s", error)
            return None
        if not path:
            return []
        return [current[:2]] + [(p.pose.position.x, p.pose.position.y) for p in path]

    def cloud_in_map(self, cloud, stride):
        """62418450 向井悠人
        深度点群を間引き、撮影時刻のTFで地図座標に変換する。
        """
        if not cloud.header.frame_id or cloud.height <= 1:
            raise ValueError("Expected registered, organized depth")
        transform = self.tf_buffer.lookup_transform(
            "map", cloud.header.frame_id, cloud.header.stamp, rospy.Duration(1.0))
        pixels = [(u, v) for v in range(0, cloud.height, stride)
                  for u in range(0, cloud.width, stride)]
        points = np.asarray(list(point_cloud2.read_points(
            cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=pixels)), dtype=float)
        if len(points) < 30:
            raise ValueError("Not enough depth measurements")
        points = points[np.isfinite(points).all(axis=1)]
        rotation, translation = self.transform_arrays(transform.transform)
        return points.dot(rotation.T) + translation

    @staticmethod
    def floor_obstacle_cells(points):
        """62418450 向井悠人
        床の上で複数の深度点が支持する障害物セルを求める。
        """
        raised = points[(points[:, 2] > 0.015) & (points[:, 2] < 0.40), :2]
        cells, counts = np.unique(np.floor(raised / 0.05), axis=0, return_counts=True)
        return (cells[counts >= 2] + 0.5) * 0.05

    def corridor_still_clear(self, corridor):
        """62418450 向井悠人
        走行中に確認済みの線分からの逸脱と、新しい床上の障害物を監視する。
        """
        start, target = corridor
        pose = self.get_robot_pose()
        if pose is None or self.dist_point_to_segment(pose[:2], start, target) > 0.08:
            return False
        clouds = self.depth_clouds.copy()
        if not clouds or abs((rospy.Time.now() - clouds[-1].header.stamp).to_sec()) > 0.65:
            return False
        try:
            obstacles = self.floor_obstacle_cells(self.cloud_in_map(clouds[-1], 5))
        except (ValueError, rospy.ROSException, tf2_ros.LookupException,
                tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return False
        radius = self.base_radius_with_payload(pose)
        return (radius is not None and all(
            self.dist_point_to_segment(point, pose[:2], target) >= radius + 0.12
            for point in obstacles))

    def await_base_result(self, timeout, corridor):
        """62418450 向井悠人
        台車の応答を待ち、監視対象の区間を走行中なら定期的に床を確認する。
        """
        client = omni_base.NAVCLIENT
        if corridor is None:
            return "done" if client.wait_for_result(rospy.Duration(timeout)) else "timed out"
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown():
            remaining = (deadline - rospy.Time.now()).to_sec()
            if remaining <= 0:
                return "timed out"
            if client.wait_for_result(rospy.Duration(min(0.25, remaining))):
                return "done"
            if not self.corridor_still_clear(corridor):
                return "left confirmed corridor or floor no longer clear"
        return "interrupted"

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
        floor = points[np.abs(points[:, 2]) <= 0.015, :2]
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

    def turn_base_toward(self, yaw_deg):
        """62418450 向井悠人
        腕を移動姿勢にした台車を、現在地で目標の向きまで回す。
        """
        self.confirm_navigation_finished()
        if not self.arm_ready:
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
        if any(math.dist(point, pose[:2]) < swept for point in observation[1]):
            rospy.logwarn("Cannot turn: floor obstacle in the swept footprint")
            return False
        if self.wall_map is not None and not self.wall_segment_clear(pose[:2], pose[:2], swept):
            rospy.logwarn("Cannot turn: wall in the swept footprint")
            return False

        deadline = time.monotonic() + 15.0
        try:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                pose = self.get_robot_pose()
                if pose is None:
                    return False
                error = math.atan2(math.sin(desired - pose[2]), math.cos(desired - pose[2]))
                if abs(error) <= math.radians(5):
                    return True
                omega = max(-28.0, min(28.0, math.degrees(error) * 1.5))
                omni_base.cmd_vel(0, 0, omega)
                time.sleep(0.1)
        finally:
            self.stop_base()
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

    def navigate_checked(self, pos, name, timeout_param="~navigation_timeout"):
        """62418450 向井悠人
        観測した床障害物を保持して経路を引き直し、確認済みの短い区間を進む。
        """
        self.confirm_navigation_finished()
        if not self.arm_ready:
            return False
        obstacles = set()
        cleared = {}
        for _ in range(32):
            current = self.get_robot_pose()
            if current is None or rospy.is_shutdown():
                return False
            if math.dist(current[:2], pos[:2]) <= 0.10:
                return self.turn_base_toward(pos[2])
            path = self.plan_base_path(pos)
            if not path and (not obstacles or self.wall_map is None
                             or math.dist(current[:2], pos[:2]) > 1.6):
                return False
            radius = self.base_radius_with_payload(current)
            if radius is None:
                return False
            radius = float(radius)
            step = self.route_step_target(current[:2], path, obstacles, radius)
            if step is not None and self.drive_observed_step(
                    step, name, timeout_param, obstacles, cleared):
                continue

            current = self.get_robot_pose()
            if current is None or not obstacles or self.wall_map is None:
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

    def recover_after_transport(self):
        """62418450 向井悠人
        正規の収納やBin Bが使えないとき、見えている近くの床へ戻す。
        """
        self.confirm_navigation_finished()
        if self.hand_state == "empty":
            self.arm_ready = self.move_placement_pose("all_neutral")
            return self.arm_ready
        if self.held_geometry is None:
            return False
        base = self.get_robot_pose()
        observation = self.observe_floor()
        if base is None or observation is None:
            return False
        floor, obstacles = observation
        hand = self.hand_pose()
        rotation, _ = self.transform_arrays(hand)
        offset = rotation.dot(self.held_geometry["offset"])
        extent = np.abs(rotation).dot(self.held_geometry["size"]) / 2
        direction = np.array([math.cos(base[2]), math.sin(base[2])])
        side = np.array([-direction[1], direction[0]])
        radius = float(np.linalg.norm(extent[:2])) + 0.08
        for lateral in (0.0, -0.16, 0.16):
            xy = np.asarray(base[:2]) + direction * 0.52 + side * lateral
            if np.min(np.linalg.norm(floor - xy, axis=1)) > 0.05:
                continue
            if any(math.dist(point, xy) < radius for point in obstacles):
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
        床を確認して収納先へ進み、机上の収納先なら横へずらした停止位置も試す。
        """
        pos = self.coordinates["positions"][place]
        if self.navigate_checked(pos, place, "~placement_navigation_timeout"):
            return True
        if place not in self.placement_targets:
            return False
        yaw = math.radians(pos[2])
        for offset in (0.15, -0.15):
            candidate = [pos[0] - math.sin(yaw) * offset,
                         pos[1] + math.cos(yaw) * offset, pos[2]]
            if self.plan_base_path(candidate):
                return self.navigate_checked(candidate, place + "_alternate",
                                             "~placement_navigation_timeout")
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

    def observe_pick_candidate(self, pose, floor=False, attempted=(), label=None,
                               previous=None, shelf=False, allow_approach=False):
        """62418450 向井悠人
        検出と深度から候補を選び、観測失敗と候補なしを区別する。
        """
        tilt = self.poses[pose]["head_tilt_joint"]
        observed = False
        unmeasured = False
        for pan in (0.0, 0.45 if previous is None else -0.45):
            if not self.point_head(pan, tilt):
                continue
            detection = self.observe_detections()
            if detection is None:
                continue
            base = self.get_robot_pose()
            if base is None:
                return "observation_failed", None
            observed = True
            ranked = sorted(detection.bboxes, key=self.calc_score_bbox, reverse=True)
            candidates = []
            for bbox in ranked:
                if (bbox.w <= 0 or bbox.h <= 0 or bbox.score <= 0
                        or (label is not None and bbox.label != label)
                        or (not shelf and self.placement_category(bbox.label) is None)):
                    continue
                measurement, item = self.measure_object(bbox, detection, floor)
                if measurement == "measurement_failed":
                    unmeasured = True
                if item is None:
                    continue
                center = item["center"]
                reach = math.hypot(center[0] - base[0], center[1] - base[1])
                if (reach < 0.25 or (reach > 0.95 and not allow_approach)
                        or (previous is not None and math.dist(center, previous) > 0.18)
                        or any(bbox.label == kind and math.dist(center, location) < 0.12
                               for kind, location in attempted)):
                    continue
                item["label"] = bbox.label
                item["placement"] = self.planned_placement(bbox.label) if not shelf else None
                lateral = abs(-math.sin(base[2]) * (center[0] - base[0])
                              + math.cos(base[2]) * (center[1] - base[1]))
                # 床では進行方向を優先し、机と棚では検出スコア順を保つ。
                priority = (lateral > 0.32, reach) if floor else (reach > 0.95,)
                candidates.append((priority, item))
            if candidates:
                return "found", min(candidates, key=lambda entry: entry[0])[1]
        return ("empty" if observed and not unmeasured else "observation_failed"), None

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

    def try_pick_object(self, item, pose, floor, view=None, shelf=False):
        """62418450 向井悠人
        個体を最大2回つかみ、2回目は再計測して把持方法を変える。
        """
        for attempt in (1, 2):
            above = False if shelf else floor == (attempt == 1)
            rospy.loginfo("Grasp attempt %d: %s from %s",
                          attempt, item["label"], "above" if above else "side")
            if self.grasp_measured_object(item, above, shelf):
                return True, item
            if self.hand_state != "empty" or not self.arm_ready or attempt == 2:
                break
            rospy.loginfo("Reobserving %s after grasp failure (view=%s)", item["label"], view)
            _, fresh = self.observe_pick_candidate(
                pose, floor, label=item["label"], previous=item["center"], shelf=shelf)
            if fresh is None:
                break
            item = fresh
        return False, item

    def execute_task1(self):
        """62418450 向井悠人
        床、机の順に片付け、観測失敗、動作失敗、探索終了を別々に返す。
        """
        rospy.loginfo("#### start Task 1 ####")
        phases = ((True, ("near_long_table_l",), "look_at_near_floor"),
                  (False, ("tall_table", "long_table_r"), "look_at_tall_table"))
        attempted = []
        placed = 0
        incomplete = False
        started = rospy.Time.now()
        limit = float(rospy.get_param("~task1_time_limit", 14 * 60))
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
                current = self.get_robot_pose()
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
                        current = self.get_robot_pose()
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
                identity = (item["label"], tuple(item["center"]))
                if identity != attempted[-1]:
                    attempted.append(identity)
                if not grasped:
                    if self.hand_state != "empty" or not self.arm_ready:
                        if not self.recover_after_transport():
                            return "motion_failed"
                    continue

                completed = self.put_object_by_label(item["label"], item["placement"])
                if completed or self.last_placement_result.get("released", False):
                    placed += 1
                if self.hand_state != "empty" or not self.arm_ready:
                    if not self.recover_after_transport():
                        rospy.logerr("Cannot resume Task 1 after placement failure")
                        return "motion_failed"

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

    def deliver_to_target(self, target_obj, target_person):
        """62418450 向井悠人
        棚から指定物か代替品を取り、保持と到達を確認して指定の人物に渡す。
        """
        waypoint = self.PERSON_WAYPOINTS.get(target_person)
        if waypoint is None:
            return False
        if not self.move_placement_pose("look_at_shelf"):
            return False
        attempted = []
        for label in ((target_obj, None) if target_obj is not None else (None,)):
            status, item = self.observe_pick_candidate(
                "look_at_shelf", label=label, attempted=attempted, shelf=True)
            if status == "observation_failed":
                rospy.logwarn("Task 2b: shelf observation failed (requested label=%s)", label)
            if item is None:
                continue
            attempted.append((item["label"], tuple(item["center"])))
            grasped, item = self.try_pick_object(item, "look_at_shelf", False, shelf=True)
            if grasped:
                break
            identity = (item["label"], tuple(item["center"]))
            if identity != attempted[-1]:
                attempted.append(identity)
            if self.hand_state != "empty" or not self.arm_ready:
                if not self.recover_after_transport():
                    return False
                if not self.move_placement_pose("look_at_shelf"):
                    return False
        else:
            rospy.logwarn("Task 2b: no deliverable object")
            return False

        rospy.loginfo("Task 2b carrying: requested=%s selected=%s person=%s",
                      target_obj, item["label"], target_person)
        if not self.navigate_checked(self.coordinates["positions"][waypoint], waypoint):
            self.recover_after_transport()
            return False
        self.arm_ready = False
        if not self.move_placement_pose("deliver_to_human") or not self.verify_holding():
            self.recover_after_transport()
            return False
        rospy.sleep(10.0)
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

    def plan_path_astar(self, start, goal, obstacles, safe_dist, bounds=None, wall_radius=None):
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
                    if clr < safe_dist or not self.is_segment_free(
                            to_xy(cell), to_xy(nxt), obstacles, safe_dist, wall_radius):
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
            rospy.wait_for_service("/static_map", timeout=2.0)
            grid = rospy.ServiceProxy("/static_map", GetMap)().map
        except (rospy.ROSException, rospy.ServiceException):
            try:
                grid = rospy.wait_for_message("/map", OccupancyGrid, timeout=2.0)
            except rospy.ROSException as error:
                rospy.logwarn("Static wall map unavailable: %s", error)
                return False
        info = grid.info
        if (grid.header.frame_id.lstrip("/") != "map"
                or info.resolution <= 0 or info.width <= 0 or info.height <= 0
                or len(grid.data) != info.width * info.height):
            rospy.logwarn("Static wall map has invalid coordinates")
            return False
        occupancy = np.asarray(grid.data).reshape(info.height, info.width)
        free = ((occupancy >= 0) & (occupancy < 50)).astype(np.uint8)
        distance = cv2.distanceTransform(np.pad(free, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        q = info.origin.orientation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:2, :2]
        origin = np.array([info.origin.position.x, info.origin.position.y])
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

    def simplify_path(self, path, obstacles, safe_dist):
        """62418450 向井悠人
        壁と物体を避けられる直線区間だけを残して経路を単純化する。
        """
        result = [path[0]]
        index = 0
        while index < len(path) - 1:
            following = next((i for i in range(len(path) - 1, index, -1)
                              if self.is_segment_free(path[index], path[i], obstacles, safe_dist)),
                             None)
            if following is None:
                return []
            result.append(path[following])
            index = following
        return result

    def plan_next_point(self, current, goal, obstacles):
        """62418450 向井悠人
        壁を含めてA*で計画し、検査済みの最初の線分から次の短い移動点を返す。
        """
        for safe_dist in self.TASK2A_SAFE_DISTS:
            path = self.plan_path_astar(current, goal, obstacles, safe_dist)
            if not path:
                continue
            corners = self.simplify_path(path, obstacles, safe_dist)
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

    def move_straight_to(self, target, obstacles, watch=True):
        """62418450 向井悠人
        壁と床の物体を監視して短距離を進み、結果にかかわらず停止指令を出す。
        """
        self.confirm_navigation_finished()
        collisions = self.count_task2a_collisions()
        pose = self.get_robot_pose()
        if pose is None:
            return "no_tf"
        start = pose[:2]
        distance = math.dist(target, start)
        deadline = time.monotonic() + distance / self.TASK2A_MAX_SPEED * 3.0 + 5.0
        try:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                pose = self.get_robot_pose()
                if pose is None:
                    return "no_tf"
                if watch and self.count_task2a_collisions() > collisions:
                    return "collision"
                if math.dist(target, pose[:2]) < self.TASK2A_GOAL_TOL:
                    return "reached"
                if not self.wall_segment_clear(pose[:2], target):
                    return "wall"
                if watch and (self.is_approaching_obstacle(pose, target, obstacles)
                              or not self.corridor_still_clear((start, target))):
                    return "too_close"
                omni_base.cmd_vel(*self.calc_base_velocity(pose, target))
                time.sleep(self.TASK2A_CONTROL_DT)
            return "timeout"
        finally:
            self.stop_base()

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
        """62418450 向井悠人
        障害物の記録を使って短く進み、観測不足の区間は再計測してから判断する。
        """
        history = []
        attempted = set()
        observation_misses = {}
        obstacles = set()
        cleared = {}
        for iteration in range(self.TASK2A_MAX_MOVES * 2):
            pose = self.get_robot_pose()
            if pose is None or rospy.is_shutdown():
                return False
            current = pose[:2]
            for pan in ((0.0, -0.7, 0.7) if iteration == 0 else (0.0,)):
                observation = self.observe_floor(pan, -1.0)
                if observation is not None:
                    self.update_floor_obstacles(observation, obstacles, cleared)
            goal = self.find_free_goal(obstacles)
            if goal is not None and math.dist(current, goal) <= 0.09:
                return True
            primary = self.plan_next_point(current, goal, obstacles) if goal is not None else None
            alternatives = [(current[0] + delta, current[1]) for delta in (-0.19, 0.19)]
            if history:
                alternatives.append(history[-1])
            moved = False
            needs_reobserve = False
            for target in [primary] + alternatives:
                if target is None:
                    continue
                edge = tuple(round(value, 1) for value in (*current, *target))
                reversing = bool(history) and target == history[-1]
                if (edge in attempted or (not reversing and observation_misses.get(edge, 0) >= 2)
                        or not self.is_segment_free(
                            current, target, obstacles, self.TASK2A_SAFE_DISTS[-1])):
                    continue
                pan = self.side_to_look(pose, target) or 0.0
                observation = self.observe_floor(pan, -1.0)
                if observation is not None:
                    self.update_floor_obstacles(observation, obstacles, cleared)
                if not self.is_segment_free(current, target, obstacles,
                                            self.TASK2A_SAFE_DISTS[-1]):
                    continue
                if not reversing and not (
                        observation is not None
                        and self.observed_segment_clear(current, target, observation)):
                    misses = observation_misses.get(edge, 0) + 1
                    observation_misses[edge] = misses
                    needs_reobserve = needs_reobserve or misses < 2
                    continue
                attempted.add(edge)
                result = self.move_straight_to(target, obstacles)
                rospy.loginfo("Task 2a segment %s: %s", target, result)
                moved = True
                if result == "reached":
                    observation_misses.clear()
                    if reversing:
                        history.pop()
                    else:
                        history.append(current)
                elif result == "collision":
                    hit_pose = self.get_robot_pose()
                    if hit_pose is not None and self.wall_segment_clear(hit_pose[:2], current):
                        # 直前に通った同じ線分だけを戻り、新しい方向は次の観測で決める。
                        self.move_straight_to(current, obstacles, watch=False)
                break
            if not moved and not needs_reobserve:
                rospy.logwarn("Task 2a: alternate views and known retreat exhausted")
                return False
        return False

    def find_free_goal(self, obstacles):
        """62418450 向井悠人
        ゴール領域の中で壁と床の物体から距離を取れる到達候補を選ぶ。
        """
        gx, gy = self.TASK2A_GOAL
        for distance in self.TASK2A_SAFE_DISTS:
            for offset in (0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4):
                point = (gx + offset, gy)
                if (self.TASK2A_PLAN_X[0] <= point[0] <= self.TASK2A_PLAN_X[1]
                        and self.is_segment_free(point, point, obstacles, distance)):
                    return point
        return None

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
                return self.deliver_to_target(label, person)
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
        if self.get_robot_pose() is None:
            rospy.logerr("Cannot start without localization")
            return
        try:
            rospy.wait_for_service("detection/get_object_detection", timeout=30.0)
        except rospy.ROSException as error:
            rospy.logwarn("RGB detector unavailable at startup: %s", error)
        self.arm_ready = self.move_placement_pose("all_neutral")
        if not self.arm_ready:
            rospy.logerr("Cannot reach initial pose")
            return
        self.load_wall_map()
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
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
