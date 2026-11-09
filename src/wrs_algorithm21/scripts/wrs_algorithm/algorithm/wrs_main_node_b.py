#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
week1 (4d2378b) を基準に、現在のC版の引き出し操作とB版の把持・配置確認を統合。
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import json
import os
import traceback
import time
import rospy
import rospkg
import tf2_ros
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.srv import GetPlan
from actionlib_msgs.msg import GoalStatus
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper
import math
import copy
import heapq


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
        """62418450 向井悠人
        姿勢変更をB側のMoveIt実行へ接続し、成否を正しく返す。
        """
        return self.move_placement_pose(name)

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
        """62418450 向井悠人
        背景と他物体を除いた深度点群の幾何中心を把持位置として返す。
        """
        self.current_grasp_geometry = None
        try:
            self.read_placement_cloud(rospy.Time.now())
            others = [item for item in getattr(self, "task1_detected_bboxes", [])
                      if item is not bbox]
            points = self.object_points_from_bbox(bbox, others)
            if points is None:
                rospy.logwarn("Target depth could not be separated from surrounding objects")
                return None
            self.update_grasp_scene(points, others)
            center = self.current_grasp_geometry["center"]
            return Point(*center)
        except Exception as error:
            rospy.logerr("Grasp geometry unavailable: %s", error)
            return None

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

    def grasp_from_side(self, pos_x, pos_y, pos_z, yaw, pitch, roll, preliminary="-y"):
        """62418450 向井悠人
        把持後は10cm持ち上げてから後退し、各制御結果を確認して失敗時に停止する。
        成功は動作の完了を表し、物体を保持したことのセンサ検証とは区別する。
        """
        target = [pos_x, pos_y, max(pos_z, 0.05)]

        # 指定方向から物体に接近する
        approach = self.grasp_approach_position(target, preliminary)

        rospy.loginfo(
            "Grasp approach from %s: pregrasp=%s target=%s",
            preliminary, approach, target
        )
        record = {"stage": "open", "success": False, "closed": False,
                  "holding_verified": False}
        self.last_grasp_result = record
        self.grasp_motion_fault = False
        self.grasp_holding_possible = False
        try:
            if not gripper.command(1):
                self.grasp_motion_fault = True
                return False
            record["stage"] = "approach"
            if not self.move_grasp_hand(approach, yaw, pitch, roll):
                self.grasp_motion_fault = True
                return False
            record["stage"] = "contact"
            if not self.move_grasp_hand(target, yaw, pitch, roll, linear=True):
                self.grasp_motion_fault = True
                return False
            record["stage"] = "close"
            # 閉鎖の成否が不明な場合も、保持している可能性を残して停止する。
            self.grasp_holding_possible = True
            if not self.close_object_gripper():
                self.grasp_motion_fault = True
                return False
            gap = self.finger_gap()
            if gap is None or not 0.015 <= gap <= 0.12:
                if gripper.command(1) and self.move_grasp_hand(approach, yaw, pitch, roll):
                    self.grasp_holding_possible = False
                else:
                    self.grasp_motion_fault = True
                record["stage"] = "empty_grasp"
                return False
            record["holding_verified"] = True
            self.remember_held_object()
            record["closed"] = True
            # 支持面から先に離し、物体の底を床や台へ擦らずに後退する。
            lifted = [target[0], target[1], target[2] + 0.10]
            record["stage"] = "lift"
            rospy.loginfo("Grasp lift target: %s", lifted)
            if not self.move_grasp_hand(lifted, yaw, pitch, roll, linear=True):
                self.grasp_motion_fault = True
                return False
            retreat = [approach[0], approach[1], max(approach[2], lifted[2])]
            record["stage"] = "retreat"
            rospy.loginfo("Grasp retreat target: %s", retreat)
            if not self.move_grasp_hand(retreat, yaw, pitch, roll, linear=True):
                self.grasp_motion_fault = True
                return False
            record["success"] = True
            return True
        except Exception as error:
            self.grasp_motion_fault = True
            record["error"] = str(error)
            return False
        finally:
            rospy.loginfo("Grasp result: %s", json.dumps(record, sort_keys=True))

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
        頭部RGB-Dの新しい深度点群を受信し、撮影時刻のTFでmap座標へ変換する。
        """
        import numpy as np
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs import point_cloud2
        from tf.transformations import quaternion_matrix

        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("Depth cloud stride must be a positive integer")
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
            pixels = [(u, v) for v in range(0, cloud.height, stride)
                      for u in range(0, cloud.width, stride)]
            points = np.asarray(list(point_cloud2.read_points(
                cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=pixels)))
            if points.ndim != 2 or len(points) < 100:
                raise ValueError("Not enough depth measurements for placement")
            points = points[np.isfinite(points).all(axis=1)]
            q = transform.rotation
            matrix = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
            t = transform.translation
            self.last_depth_cloud = cloud
            self.last_depth_transform = (matrix, np.array([t.x, t.y, t.z]))
            self.placement_camera_frame = cloud.header.frame_id
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
        configured = rospy.get_param("~placement_scan_waypoint", None)
        waypoints = ([configured] if configured is not None else rospy.get_param(
            "~placement_scan_waypoints", ["stair_like_drawer", "drawer"]))
        if (not isinstance(waypoints, (list, tuple)) or not waypoints
                or any(not isinstance(name, str) or name not in self.coordinates["positions"]
                       for name in waypoints)):
            raise ValueError("Placement observation waypoints must be known waypoint names")
        if not self.move_placement_pose("all_neutral"):
            raise ValueError("Cannot reach the placement observation posture")
        for waypoint in waypoints:
            if self.navigate_for_placement(waypoint):
                break
            if getattr(self, "placement_nav_cancel_pending", False):
                raise ValueError("Observation navigation cancellation has not completed")
        else:
            raise ValueError("Cannot reach placement observation waypoints: {}".format(waypoints))
        frames = []
        for pan in (-0.8, 0.0, 0.8):
            head = whole_body.HEAD_CMD
            head.set_joint_value_target({"head_pan_joint": pan, "head_tilt_joint": -0.55})
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
        map座標候補を接近位置の初期値に使い、開放の可否は到着後の開口確認で決める。
        """
        if category not in ("tray", "container_a", "container_b"):
            return
        # A/Bはルール上の名前。シミュレータ内のモデル名やx順による入れ替えは行わない。
        centers = {"container_a_place": (1.10, -0.60), "container_b_place": (1.40, -0.60),
                   "tray_a_place": (1.95, -0.60), "tray_b_place": (1.65, -0.60)}
        self.placement_candidates = getattr(self, "placement_candidates", {})
        for place, _ in destinations:
            x, y = centers[place]
            self.placement_candidates[place] = {"x": x, "y": y, "kind": category}
            self.placement_targets.setdefault(place, {"x": x, "y": y, "hover": .12})
            self.coordinates["positions"].setdefault(place, [x, y + .65, -90])
            rospy.loginfo("Unverified map placement candidate %s: (%.2f, %.2f)", place, x, y)

    def move_over_container(self, place, hover=False, linear=False):
        """62418450 向井悠人
        信頼できる開口と保持物体を確認し、衝突確認付きの計画を実行して失敗理由を記録する。
        """
        target = self.placement_targets[place]
        if not self.placement_target_is_trusted(place, target):
            rospy.logerr("Untrusted destination geometry: %s", place)
            return False
        if not hasattr(self, "held_object"):
            raise ValueError("Measured held object is required")
        transform = self.tf_buffer.lookup_transform(
            "map", "hand_palm_link", rospy.Time(0), rospy.Duration(2.0))
        _, offset, extent = self.held_object_world(transform.transform)
        margin = .01 + target["uncertainty"]
        if (extent[0] + margin >= target["half_x"]
                or extent[1] + margin >= target["half_y"]):
            rospy.logerr("Object too large for %s: extent=%s margin=%.3f opening=(%.3f,%.3f)",
                         place, extent, margin, target["half_x"], target["half_y"])
            return False
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = target["x"] - offset[0]
        pose.pose.position.y = target["y"] - offset[1]
        # 最下部を縁より誤差分だけ上へ残す。開放前の上限5cmにも収める。
        pose.pose.position.z = (target["rim_z"] + extent[2] + .02 + target["uncertainty"]
                                - offset[2] + (target["hover"] if hover else 0))
        pose.pose.orientation = transform.transform.rotation
        commander = whole_body.WHOLE_BODY_CMD
        frame = commander.get_pose_reference_frame()
        try:
            commander.set_start_state_to_current_state()
            if linear:
                commander.set_pose_reference_frame("map")
                trajectory, fraction = commander.compute_cartesian_path(
                    [pose.pose], .005, 0.0, avoid_collisions=True)
                if not math.isfinite(fraction) or fraction < 1.0:
                    rospy.logerr("Placement %s Cartesian planning incomplete: fraction=%s",
                                 place, fraction)
                    return False
            else:
                commander.set_pose_target(pose)
                success, trajectory, planning_time, error = commander.plan()
                if not success:
                    rospy.logerr("Placement %s planning failed: code=%s time=%s target=%s",
                                 place, error.val, planning_time, pose)
                    return False
            if not trajectory.joint_trajectory.points:
                rospy.logerr("Placement %s planning produced an empty trajectory", place)
                return False
            success = bool(commander.execute(trajectory, wait=True))
            if not success:
                rospy.logerr("Placement %s trajectory execution failed: linear=%s", place, linear)
            return success
        finally:
            commander.stop()
            commander.clear_pose_targets()
            commander.set_pose_reference_frame(frame)

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
            # 検出器のpadlockは南京錠の鍵（wod-c_classes.json）。ルール上はTools。
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
        C版と同じ関節軌道指令を使い、衝突確認・到達判定とコントローラ結果を確認する。
        """
        import actionlib
        from sensor_msgs.msg import JointState
        from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from moveit_msgs.msg import RobotState
        from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest

        targets = self.poses.get(name)
        arm_names = {"arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                     "wrist_flex_joint", "wrist_roll_joint"}
        head_names = {"head_pan_joint", "head_tilt_joint"}
        if (not targets or set(targets) - (arm_names | head_names)
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in targets.values())):
            rospy.logerr("Invalid joint pose: %s", name)
            return False
        clients = []
        try:
            # 腕・頭・指の情報が別メッセージで届く場合も、必要な関節がそろうまで集める。
            deadline = time.monotonic() + 3.0
            current = {}
            while not set(targets).issubset(current):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = sorted(set(targets) - set(current))
                    raise ValueError("Joint feedback timed out; missing joints: {}".format(missing))
                state = rospy.wait_for_message(
                    "/hsrb/joint_states", JointState, timeout=remaining)
                current.update((joint, value) for joint, value in zip(state.name, state.position)
                               if math.isfinite(value))
            # 衝突判定にも統合済みの関節状態を渡す。速度・力は今回の補間判定では使わない。
            state = copy.deepcopy(state)
            # 指・ばね・mimic関節を別時刻の受信値で上書きすると、架空の自己衝突が生じる。
            # 変更する腕・頭のみを差分として渡し、その他はMoveItの現在状態を使う。
            state.name = sorted(targets)
            state.position = [current[joint] for joint in state.name]
            state.velocity = []
            state.effort = []
            if all(abs(current[joint] - value) < .01 for joint, value in targets.items()):
                rospy.loginfo("Joint pose %s already reached", name)
                return True
            # 直接補間する経路も、観測済みの障害物・保持物体との衝突を確認する。
            rospy.wait_for_service("/check_state_validity", timeout=2.0)
            validity = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
            largest_delta = max(abs(current[joint] - value) for joint, value in targets.items())
            steps = max(2, int(math.ceil(largest_delta / .03)))
            for step in range(steps + 1):
                sample = copy.deepcopy(state)
                sample.position = list(sample.position)
                ratio = float(step) / steps
                for index, joint in enumerate(sample.name):
                    if joint in targets:
                        delta = targets[joint] - current[joint]
                        sample.position[index] = current[joint] + ratio * delta
                request = GetStateValidityRequest()
                request.robot_state = RobotState(joint_state=sample, is_diff=True)
                result = validity(request)
                if not result.valid:
                    contacts = [(item.contact_body_1, item.contact_body_2)
                                for item in getattr(result, "contacts", [])]
                    rospy.logerr("Joint pose %s path collides at sample %d/%d: contacts=%s",
                                 name, step, steps, contacts)
                    return False
            duration = max(2.0, max(abs(current[joint] - value) /
                           (.08 if joint == "arm_lift_joint" else .4)
                           for joint, value in targets.items()))
            goals = []
            for group, names in (("arm", arm_names), ("head", head_names)):
                joints = sorted(set(targets) & names)
                if not joints:
                    continue
                client = actionlib.SimpleActionClient(
                    "/hsrb/{}_trajectory_controller/follow_joint_trajectory".format(group),
                    FollowJointTrajectoryAction)
                if not client.wait_for_server(rospy.Duration(3.0)):
                    raise ValueError("Joint controller unavailable: {}".format(group))
                trajectory = JointTrajectory()
                trajectory.joint_names = joints
                point = JointTrajectoryPoint()
                point.positions = [targets[joint] for joint in joints]
                point.velocities = [0.0] * len(joints)
                point.time_from_start = rospy.Duration(duration)
                trajectory.points = [point]
                goals.append((group, client, FollowJointTrajectoryGoal(trajectory=trajectory)))
            for _, client, goal in goals:
                client.send_goal(goal)
                clients.append(client)
            for group, client, _ in goals:
                if not client.wait_for_result(rospy.Duration(duration + 5.0)):
                    raise ValueError("Joint pose timed out: {} / {}".format(name, group))
                result = client.get_result()
                if (client.get_state() != GoalStatus.SUCCEEDED or result is None
                        or result.error_code != 0):
                    raise ValueError("Joint pose failed: {} / {} state={} result={}".format(
                        name, group, client.get_state(), result))
            rospy.loginfo("Joint pose %s reached", name)
            return True
        except Exception as error:
            for client in clients:
                try:
                    if client.get_state() in (GoalStatus.PENDING, GoalStatus.ACTIVE,
                                              GoalStatus.PREEMPTING, GoalStatus.RECALLING):
                        client.cancel_goal()
                        client.wait_for_result(rospy.Duration(1.0))
                except Exception as cancellation_error:
                    rospy.logerr("Joint pose cancellation failed: %s", cancellation_error)
            rospy.logerr("Joint pose %s failed: %s", name, error)
            return False

    def navigate_for_placement(self, place):
        """62418450 向井悠人
        移動結果を記録し、中止応答を確認するまで次のゴールを送らない。
        """
        from move_base_msgs.msg import MoveBaseGoal
        from wrs_algorithm.util.mathematics import quaternion_from_euler

        timeout = float(rospy.get_param("~placement_navigation_timeout", 60.0))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("placement_navigation_timeout must be positive and finite")
        client = omni_base.NAVCLIENT
        if not client.wait_for_server(rospy.Duration(2.0)):
            rospy.logerr("Navigation server unavailable for %s", place)
            return False
        if getattr(self, "placement_nav_cancel_pending", False):
            if not client.wait_for_result(rospy.Duration(2.0)):
                rospy.logerr("Previous navigation cancellation still pending")
                return False
            self.placement_nav_cancel_pending = False
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        x_pos, y_pos, yaw = self.coordinates["positions"][place]
        goal.target_pose.pose.position.x = x_pos
        goal.target_pose.pose.position.y = y_pos
        goal.target_pose.pose.orientation = quaternion_from_euler(yaw, 0, 0)
        rospy.loginfo("Placement navigation %s: x=%.2f y=%.2f yaw=%.1f", place,
                      x_pos, y_pos, yaw)
        client.send_goal(goal)
        if not client.wait_for_result(rospy.Duration(timeout)):
            rospy.logerr("Navigation timed out at %s; state=%s text=%s", place,
                         client.get_state(), client.get_goal_status_text())
            client.cancel_goal()
            self.placement_nav_cancel_pending = not client.wait_for_result(rospy.Duration(2.0))
            return False
        state = client.get_state()
        if state != GoalStatus.SUCCEEDED:
            rospy.logerr("Navigation failed at %s; state=%s text=%s", place,
                         state, client.get_goal_status_text())
            return False
        rospy.loginfo("Placement navigation reached %s", place)
        return True

    def put_object_by_label(self, label):
        """62418450 向井悠人
        把持した物体を分類して収納する。明示的なunknown/objectだけをBin Bへ退避する。
        """
        self.last_placement_result = {
            "stage": "select_destination", "released": False, "success": False,
            "deposit_verified": False, "holding_state": "unknown",
            "outcome": "pre_release_validation_failed"}
        if label is None or (isinstance(label, str) and not label.strip()):
            label = "unknown"
        if not isinstance(label, str):
            rospy.logerr("Invalid placement label: %s", label)
            return False
        label = self.normalize_placement_label(label)
        try:
            if label in ("unknown", "object"):
                return self.put_in_fallback(label, "Object category is unknown")
            place, pose = self.select_place_for_label(label)
            return self.put_in_place(place, pose, label=label)
        except Exception as error:  # ロボット動作失敗をTask 1へ返す。
            self.last_placement_result.update(label=label, error=str(error))
            rospy.logerr("Placement result: %s",
                         json.dumps(self.last_placement_result, sort_keys=True))
            return False

    def put_in_place(self, place, into_pose, label=None, fallback_reason=None):
        """62418450 向井悠人
        搬送・到達・開放検証・退避・収納確認を実行し、失敗工程と保持状態を分離する。
        """
        started = time.monotonic()
        record = {"label": label, "place": place, "pose": into_pose, "stage": "validate",
                  "released": False, "success": False, "deposit_verified": False,
                  "holding_state": "unknown", "fallback": fallback_reason is not None,
                  "fallback_reason": fallback_reason}
        self.last_placement_result = record
        try:
            if (place not in self.coordinates["positions"] or not self.poses.get("all_neutral")
                    or not hasattr(self, "held_object") or self.held_object.get("released", False)):
                raise ValueError("Placement requires a destination and an unreleased held object")
            record["holding_state"] = "holding"
            stages = [("transport", lambda: self.move_placement_pose("all_neutral")),
                      ("navigate", lambda: self.navigate_for_placement(place)),
                      ("measure_opening", lambda: self.observe_destination_opening(place)),
                      ("above_container", lambda: self.move_over_container(place, hover=True)),
                      ("lower", lambda: self.move_over_container(place, linear=True)),
                      ("release", lambda: self.release_into_destination(place)),
                      ("lift", lambda: self.move_over_container(place, hover=True, linear=True)),
                      ("retreat", lambda: self.move_placement_pose("all_neutral")),
                      ("verify_deposit", lambda: self.verify_destination_contents(place))]
            for stage, action in stages:
                record["stage"] = stage
                rospy.loginfo("Placement %s: %s", place, stage)
                if not action():
                    return False
            record.update(success=True, deposit_verified=True)
            return True
        except Exception as error:
            record["error"] = str(error)
            rospy.logerr("Placement error at %s: %s", record["stage"], error)
            return False
        finally:
            if record["success"]:
                record["outcome"] = "completed"
            elif record["released"]:
                record["outcome"] = ("released_deposit_unconfirmed" if record["stage"] ==
                                     "verify_deposit" else "released_retreat_failed")
            elif record.get("release_commanded"):
                record["outcome"] = "release_failed_holding_unknown"
            elif record["stage"] in ("transport", "navigate"):
                record["outcome"] = "destination_navigation_failed"
            elif record["stage"] in ("above_container", "lower"):
                record["outcome"] = "placement_pose_failed"
            else:
                record["outcome"] = "pre_release_validation_failed"
            self.task1_holding_object = not record["released"]
            self.task1_motion_fault = not record["success"]
            if getattr(self, "held_object", {}).get("released", False):
                del self.held_object
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
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

    def execute_task1(self):
        """62408993 Masami Sakuma
        62418450 向井悠人
        Task1を実行する。
        """
        rospy.loginfo("#### start Task 1 ####")
        if (getattr(self, "task1_holding_object", False)
                or getattr(self, "task1_motion_fault", False)):
            rospy.logerr("Task 1 cannot restart with an unresolved placement/motion state")
            return False

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

        start_time = rospy.Time.now()
        time_limit = 14 * 60

        # 開始時だけ開ける。開閉中に失敗した場合は接触状態のまま探索へ進まない。
        self.open_drawers()
        if getattr(self, "drawer_motion_fault", False):
            return False

        rospy.loginfo("Start searching objects.")

        while not all(count >= 2 for count in empty_count):
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed >= time_limit:
                rospy.logwarn("Task1 time limit is approaching. Stop searching.")
                break

            for i, (plc, pose) in enumerate(hsr_position):
                if empty_count[i] >= 2 or attempt_count[i] >= max_attempts_per_place:
                    empty_count[i] = 2
                    continue

                if (not self.navigate_task1_view(plc) or not self.change_pose(pose)
                        or not gripper.command(1)):
                    rospy.logerr("Cannot prepare the Task 1 observation at %s", plc)
                    return False

                detected_objs = self.get_latest_detection()
                self.task1_detected_bboxes = list(detected_objs.bboxes)
                graspable_obj = self.get_most_graspable_obj(detected_objs.bboxes)
                attempt_count[i] += 1

                if graspable_obj is None:
                    empty_count[i] += 1
                    rospy.loginfo("No object at %s (%d/2)", plc, empty_count[i])
                    continue

                empty_count[i] = 0
                label = graspable_obj["label"]
                grasp_bbox = graspable_obj["bbox"]

                rospy.loginfo("Selected object: %s", label)

                grasp_pos = self.get_grasp_coordinate(grasp_bbox)
                if grasp_pos is None:
                    rospy.logwarn("Failed to obtain grasp coordinate.")
                    continue

                # ----------------------------------------------------
                # 改善点: 中間姿勢 change_pose("grasp_on_table") を削除
                # grasp_from_side 内のアプローチ移動で直接把持動作を行う
                # ----------------------------------------------------
                grasp_success = self.exec_graspable_method(grasp_pos, label)
                self.task1_holding_object = getattr(self, "grasp_holding_possible", False)

                if not grasp_success:
                    self.last_task1_result = {
                        "label": label, "outcome": "grasp_failed",
                        "holding_state": "unknown" if self.task1_holding_object else "empty"}
                if getattr(self, "grasp_motion_fault", False):
                    rospy.logerr("Grasp controller failed for %s; stop before transport", label)
                    return False
                # 把持成功時の搬送姿勢変更はBの収納工程で記録する。
                if not grasp_success:
                    if not self.change_pose("all_neutral"):
                        rospy.logerr("Cannot retreat after failed grasp of %s", label)
                        return False
                    rospy.logwarn("Could not grasp object: %s. Retrying loop.", label)
                    continue

                # B側で保持状態を管理し、誤配送せず配置失敗時は探索を止める。
                if not self.complete_task1_placement(label):
                    return False
                placed.append(label)

        rospy.loginfo("#### finish Task 1 ####")
        rospy.loginfo("Placed objects (%d): %s", len(placed), ", ".join(placed))
        return True

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

        # target_personの前に持っていく
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
    def plan_path_astar(cls, start, goal, obstacles, safe_dist):
        """62401738 いしい
        TASK2A_GRID 間隔の格子の上で、start から goal までの経路を A* で探す。
        - 障害物から safe_dist 未満の格子は通れない（start の格子だけは例外）
        - 障害物から TASK2A_SOFT_DIST 未満の格子は、近いほどコストを上乗せする
          （ぎりぎりを通らず、余裕のある経路を選ぶため）
        - 8近傍で移動し、ヒューリスティックはゴールまでの直線距離
        経路（(x, y) のリスト）を返す。見つからなければ None。
        """
        res = cls.TASK2A_GRID
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
                    if clr < safe_dist:
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
        return all([cls.dist_point_to_segment(obs, seg_start, seg_end) >= safe_dist
                    for obs in obstacles])

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
            path = cls.plan_path_astar(current, goal, obstacles, safe_dist)
            if path is None:
                rospy.logwarn("A*: no path with safe_dist=%.2f", safe_dist)
                continue
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

    def execute_task2a(self):
        """62401738 いしい
        task2aを実行する
        """
        rospy.loginfo("#### start Task 2a ####  detector: %s", self.get_detector_type())
        self.change_pose("look_at_near_floor")
        gripper.command(0)
        self.goto_name("standby_2a")
        log_start = len(self.detection_list)

        # 落ちているブロックを避けて移動
        reached = self.execute_avoid_blocks()

        # 回避の移動でゴールに着けなかったときだけ move_base でゴールへ向かう
        # （着いているのに move_base を使うと、床の物体が見えない move_base が近くの物体に寄ることがある）
        if not reached:
            self.goto_name("go_throw_2a")
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

    def wait_for_task1_controllers(self):
        """62418450 向井悠人
        ListControllersで腕・頭・指のrunningを確認し、サービス停滞も壁時計の期限で打ち切る。
        """
        import queue
        import threading
        from controller_manager_msgs.srv import ListControllers

        timeout = float(rospy.get_param("~controller_start_timeout", 10.0))
        if not math.isfinite(timeout) or timeout <= 0:
            rospy.logerr("Invalid controller_start_timeout: %s", timeout)
            return False
        service = rospy.get_param("~controller_manager_service",
                                  "/hsrb/controller_manager/list_controllers")
        required = {"arm_trajectory_controller", "head_trajectory_controller", "gripper_controller"}
        deadline = time.monotonic() + timeout
        results = queue.Queue()
        status = {"reason": "controller manager has not responded"}

        def poll():
            while time.monotonic() < deadline and not rospy.is_shutdown():
                try:
                    rospy.wait_for_service(service, timeout=min(.5, max(.01,
                                           deadline - time.monotonic())))
                    response = rospy.ServiceProxy(service, ListControllers)()
                    states = {item.name.rsplit("/", 1)[-1]: item.state
                              for item in response.controller}
                    missing = {name: states.get(name, "missing") for name in required
                               if states.get(name) != "running"}
                    status["reason"] = "controllers not running: {}".format(missing)
                    if not missing:
                        results.put(True)
                        return
                except Exception as error:
                    status["reason"] = "{}: {}".format(type(error).__name__, error)
                time.sleep(.1)
            results.put(False)

        worker = threading.Thread(target=poll)
        worker.daemon = True
        worker.start()
        try:
            ready = results.get(timeout=timeout)
        except queue.Empty:
            ready = False
        if not ready:
            rospy.logerr("Controller startup failed after %.1fs at %s: %s",
                         timeout, service, status["reason"])
        else:
            rospy.loginfo("Task 1 arm/head/gripper controllers are running")
        return ready

    def placement_target_is_trusted(self, place, target, require_same_view=False):
        """62418450 向井悠人
        mapの新しい観測と有限な開口寸法だけを信頼し、座標候補単独での開放を防ぐ。
        """
        required = ("x", "y", "rim_z", "half_x", "half_y", "hover", "uncertainty", "measured_at")
        if (not isinstance(target, dict) or target.get("frame") != "map"
                or target.get("source") != "depth"
                or any(isinstance(target.get(key), bool)
                       or not isinstance(target.get(key), (int, float))
                       or not math.isfinite(target[key]) for key in required)):
            return False
        age = time.monotonic() - target["measured_at"]
        if (not 0 <= age <= (10.0 if require_same_view else 60.0)
                or not 0 <= target["uncertainty"] <= .02 or target["hover"] <= 0
                or min(target["half_x"], target["half_y"]) <= target["uncertainty"] + .01
                or not .1 < target["rim_z"] < 1.0):
            return False
        candidate = getattr(self, "placement_candidates", {}).get(place)
        if candidate and math.hypot(target["x"] - candidate["x"],
                                    target["y"] - candidate["y"]) > .12:
            return False
        if require_same_view:
            base = self.get_robot_pose()
            old = target.get("observer_pose", [])
            if base is None or len(old) != 3:
                return False
            yaw_error = math.atan2(math.sin(base[2] - old[2]), math.cos(base[2] - old[2]))
            if math.hypot(base[0] - old[0], base[1] - old[1]) > .05 or abs(yaw_error) > .1:
                return False
        return True

    def verify_destination_contents(self, place):
        """62418450 向井悠人
        開放前後の点群を比較し、開口内・縁以下に新たな物体表面が2回見えることを確認する。
        """
        import numpy as np

        target = self.placement_targets[place]
        before = getattr(self, "placement_opening_clouds", {}).get(place)
        if before is None or not self.point_drawer_camera(-.8):
            rospy.logerr("No baseline/camera for deposit verification: %s", place)
            return False
        # 1.5cm格子の周囲まで既存面とみなす。深度ノイズを収納物の出現と数えない。
        voxels = {tuple(row) for row in np.floor(before / .015).astype(int)}
        stamp = rospy.Time.now()
        for _ in range(2):
            points, stamp = self.read_placement_cloud(stamp)
            inside = points[(np.abs(points[:, 0] - target["x"]) < target["half_x"] - .01)
                            & (np.abs(points[:, 1] - target["y"]) < target["half_y"] - .01)
                            & (points[:, 2] < target["rim_z"] + .015)
                            & (points[:, 2] > max(.02, target["rim_z"] - .30))]
            added = []
            for point in inside:
                cell = np.floor(point / .015).astype(int)
                if not any((cell[0] + x, cell[1] + y, cell[2] + z) in voxels
                           for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)):
                    added.append(point)
            if len(added) < 15 or np.max(np.ptp(added, axis=0)) < .015:
                rospy.logerr("Released object not confirmed inside %s", place)
                return False
        rospy.loginfo("Deposit surface change confirmed inside %s", place)
        return True

    def run(self):
        """62418450 向井悠人
        B上でC版を統合して実行し、開閉・配置の失敗時は後続タスクへ進まない。
        """
        if (getattr(self, "task1_holding_object", False)
                or getattr(self, "drawer_motion_fault", False)
                or getattr(self, "grasp_motion_fault", False)
                or getattr(self, "task1_motion_fault", False)):
            rospy.logerr("Resolve the previous holding/contact state before restarting")
            return False
        if not self.wait_for_task1_controllers():
            return False
        if not self.move_placement_pose("all_neutral"):
            rospy.logerr("Cannot reach initial pose")
            return False
        self.open_drawers()
        if rospy.get_param("~drawer_test_only", False):
            rospy.loginfo("Drawer test completed: opened=%s result=%s",
                          getattr(self, "drawers_open", False),
                          json.dumps(getattr(self, "last_drawer_result", {}), sort_keys=True))
            return getattr(self, "drawers_open", False)
        if getattr(self, "drawer_motion_fault", False):
            rospy.logerr("Drawer motion failed; stop before Task 1")
            return False
        rospy.loginfo("Drawer stage complete; starting Task 1 object search")
        if self.execute_task1() is False or getattr(self, "task1_holding_object", False):
            rospy.logerr("Task 1 failed; stop before Task 2")
            return False
        self.execute_task2a()
        self.execute_task2b()
        return True

    @staticmethod
    def separate_object_depth(points):
        """62418450 向井悠人
        BBox内の深度を分離し、十分な支持点がある最前面の物体を選ぶ。
        """
        import numpy as np

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            return None
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
        if len(points) < 20:
            return None
        ordered = points[np.argsort(points[:, 2])]
        groups = np.split(ordered, np.flatnonzero(np.diff(ordered[:, 2]) > 0.04) + 1)
        candidates = [group for group in groups if len(group) >= max(20, len(points) * 0.10)]
        return candidates[0] if candidates else None

    def object_points_from_bbox(self, bbox, exclusions=()):
        """62418450 向井悠人
        同時取得したRGB対応深度から他の検出枠を除いて対象点群をmapへ変換する。
        """
        import numpy as np
        from sensor_msgs import point_cloud2

        cloud = self.last_depth_cloud
        pixels = [(u, v) for v in range(max(0, int(bbox.y)),
                                        min(cloud.height, int(bbox.y + bbox.h)), 2)
                  for u in range(max(0, int(bbox.x)), min(cloud.width, int(bbox.x + bbox.w)), 2)
                  if not any(other.x <= u < other.x + other.w
                             and other.y <= v < other.y + other.h for other in exclusions)]
        points = self.separate_object_depth(list(point_cloud2.read_points(
            cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=pixels))) if pixels else None
        if points is None:
            return None
        rotation, translation = self.last_depth_transform
        return np.asarray(points).dot(rotation.T) + translation

    def update_grasp_scene(self, target, other_boxes):
        """62418450 向井悠人
        深度で見えた他物体を衝突形状として登録し、対象の幾何中心と寸法を推定する。
        """
        import numpy as np
        import moveit_commander

        if not hasattr(self, "manipulation_scene"):
            self.manipulation_scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        for name in getattr(self, "grasp_obstacle_names", []):
            self.manipulation_scene.remove_world_object(name)
        self.grasp_obstacle_names = []
        for index, bbox in enumerate(other_boxes):
            points = self.object_points_from_bbox(bbox)
            if points is None:
                raise ValueError("Cannot measure a nearby object for collision checking")
            low, high = np.percentile(points, [5, 95], axis=0)
            center = (low + high) / 2
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = center
            pose.pose.orientation.w = 1.0
            name = "b_grasp_obstacle_{}".format(index)
            self.manipulation_scene.add_box(
                name, pose, size=tuple(np.maximum(high - low, .01) + .04))
            self.grasp_obstacle_names.append(name)
        bottom, top = np.percentile(target[:, 2], [5, 95])
        if top - bottom > .025:
            surface = target[:, 2] < bottom + .008
            if np.mean(surface) > .25 and np.sum(~surface) >= 20:
                target = target[~surface]
        low, high = np.percentile(target, [5, 95], axis=0)
        self.current_grasp_geometry = {"center": (low + high) / 2,
                                       "low": low, "high": high, "points": target}

    def remember_held_object(self):
        """62418450 向井悠人
        把持時の物体と手先の位置関係を保存し、搬送中の物体も衝突形状へ追加する。
        """
        import numpy as np
        from tf.transformations import quaternion_matrix

        geometry = getattr(self, "current_grasp_geometry", None)
        if geometry is None:
            return
        hand = self.tf_buffer.lookup_transform(
            "map", "hand_palm_link", rospy.Time(0), rospy.Duration(2.0)).transform
        q, t = hand.rotation, hand.translation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        local = (geometry["points"] - np.array([t.x, t.y, t.z])).dot(rotation)
        low, high = np.percentile(local, [5, 95], axis=0)
        offset = (low + high) / 2
        self.held_object = {"offset": offset, "size": np.maximum(high - low, .01)}
        pose = PoseStamped()
        pose.header.frame_id = "hand_palm_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = offset
        pose.pose.orientation.w = 1.0
        self.manipulation_scene.attach_box(
            "hand_palm_link", "b_held_object", pose, size=tuple(self.held_object["size"]),
            touch_links=["hand_palm_link", "hand_l_proximal_link", "hand_l_distal_link",
                         "hand_r_proximal_link", "hand_r_distal_link"])

    def held_object_world(self, hand):
        """62418450 向井悠人
        保持物体の中心と投影寸法を現在の手先姿勢から計算する。
        """
        import numpy as np
        from tf.transformations import quaternion_matrix

        q, t = hand.rotation, hand.translation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        offset = rotation.dot(self.held_object["offset"])
        extent = np.abs(rotation).dot(self.held_object["size"]) / 2
        return np.array([t.x, t.y, t.z]) + offset, offset, extent

    @staticmethod
    def estimate_opening(points):
        """62418450 向井悠人
        収納先の縁を高さごとに探し、中心が空いた十分な広がりのある開口を測定する。
        """
        import numpy as np

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 40:
            return None
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 40:
            return None
        for height in np.arange(np.percentile(points[:, 2], 98), .10, -.015):
            rim = points[np.abs(points[:, 2] - height) < .01]
            if len(rim) < 30:
                continue
            low, high = np.percentile(rim[:, :2], [3, 97], axis=0)
            size, center = high - low, (high + low) / 2
            if not (.06 <= min(size) <= .65 and .16 <= max(size) <= .75):
                continue
            relative = (rim[:, :2] - center) / size
            edge = np.max(np.abs(relative), axis=1) > .30
            quadrants = {(int(x > 0), int(y > 0)) for x, y in relative[edge]}
            if np.mean(edge) < .70 or len(quadrants) < 4:
                continue
            return {"x": float(center[0]), "y": float(center[1]), "rim_z": float(height),
                    "width": float(min(size)), "length": float(max(size)),
                    "half_x": float(size[0] / 2), "half_y": float(size[1] / 2)}
        return None

    def observe_destination_opening(self, place):
        """62418450 向井悠人
        候補位置と容器種別を照合し、2フレームの開口と誤差を登録する。
        失敗時は同じ観測位置で直近に確認済みの静止容器だけを再使用する。
        """
        import numpy as np

        previous = self.placement_targets.pop(place, None)
        cached = getattr(self, "confirmed_placement_targets", {}).get(place)
        candidate = getattr(self, "placement_candidates", {}).get(place)
        drawer = place in getattr(self, "drawer_places", {})
        try:
            if not self.point_drawer_camera(-.8):
                raise ValueError("Cannot point placement camera")
            base = self.get_robot_pose()
            if base is None:
                raise ValueError("No map localization for destination measurement")
            observations = []
            clouds = []
            stamp = rospy.Time.now()
            for _ in range(2):
                points, stamp = self.read_placement_cloud(stamp)
                hint = candidate or previous
                if drawer:
                    x, y = self.drawer_drops[place]
                    hint = {"x": x, "y": y}
                if candidate:
                    matches = [item for item in self.detect_container_rims(points)
                               if item["kind"] == candidate["kind"]
                               and math.hypot(item["x"] - candidate["x"],
                                              item["y"] - candidate["y"]) <= .12]
                    if len(matches) != 1:
                        raise ValueError("Candidate category/position is missing or ambiguous")
                    hint = matches[0]
                if hint:
                    radius = max(hint.get("width", .3), hint.get("length", .3)) / 2 + .025
                    radius = .30 if drawer else min(radius, .35)
                    keep = ((np.abs(points[:, 0] - hint["x"]) < radius)
                            & (np.abs(points[:, 1] - hint["y"]) < radius))
                else:
                    delta = points[:, :2] - np.array(base[:2])
                    forward = delta.dot(np.array([math.cos(base[2]), math.sin(base[2])]))
                    side = delta.dot(np.array([-math.sin(base[2]), math.cos(base[2])]))
                    keep = (forward > .15) & (forward < .95) & (np.abs(side) < .4)
                visible = keep & (points[:, 2] > .10) & (points[:, 2] < 1.0)
                limits = getattr(self, "drawer_opening_limits", {}).get(place)
                if limits:
                    visible &= points[:, 2] < limits["max_z"]
                opening = self.estimate_opening(points[visible])
                if opening is None:
                    raise ValueError("No supported opening rim")
                if candidate and math.hypot(opening["x"] - hint["x"],
                                            opening["y"] - hint["y"]) > .04:
                    raise ValueError("Opening does not match the identified container")
                if candidate:
                    # 回転した矩形の軸平行外接枠を開口と誤認しないよう、内接する正方形を使う。
                    half = min(hint["width"], hint["length"]) / (2 * math.sqrt(2))
                    opening["half_x"] = min(opening["half_x"], half)
                    opening["half_y"] = min(opening["half_y"], half)
                if limits:
                    back = max(opening["y"] - opening["half_y"], limits["back"])
                    front = min(opening["y"] + opening["half_y"], limits["front"])
                    if front - back <= .02:
                        raise ValueError("Exposed lower drawer opening is too small")
                    opening["y"] = (back + front) / 2
                    opening["half_y"] = (front - back) / 2
                observations.append(opening)
                clouds.append(points[visible])
            keys = ("x", "y", "rim_z", "half_x", "half_y")
            disagreement = max(abs(observations[0][key] - observations[1][key]) for key in keys)
            if disagreement > .025:
                raise ValueError("Opening observations disagree by {:.3f}m".format(disagreement))
            target = {key: sum(item[key] for item in observations) / 2 for key in keys}
            # 誤差が大きいほど使える開口を縮める。半幅は2観測の小さい方を採る。
            for key in ("half_x", "half_y"):
                target[key] = min(item[key] for item in observations)
            target.update(frame="map", source="depth", measured_at=time.monotonic(),
                          observer_pose=list(base), uncertainty=.005 + disagreement / 2,
                          width=2 * min(target["half_x"], target["half_y"]),
                          length=2 * max(target["half_x"], target["half_y"]),
                          hover=.12, z=target["rim_z"] + .18)
            self.placement_targets[place] = target
            self.confirmed_placement_targets = getattr(self, "confirmed_placement_targets", {})
            self.confirmed_placement_targets[place] = dict(target)
            self.placement_opening_clouds = getattr(self, "placement_opening_clouds", {})
            self.placement_opening_clouds[place] = clouds[-1]
            rospy.loginfo("Confirmed destination %s: %s", place, json.dumps(target))
            return True
        except Exception as error:
            rospy.logwarn("Destination %s measurement failed: %s", place, error)
            if (not drawer and cached is not None
                    and self.placement_target_is_trusted(place, cached, require_same_view=True)):
                self.placement_targets[place] = dict(cached)
                rospy.logwarn("Using recent same-view confirmed geometry for %s", place)
                return True
            rospy.logerr("No trusted opening for %s; keep holding the object", place)
            return False

    def release_into_destination(self, place):
        """62418450 向井悠人
        開口内への収まりと保持を再検証し、開放済み・保持不明を分けて記録する。
        """
        if not hasattr(self, "held_object") or self.held_object.get("released", False):
            rospy.logerr("Cannot release without an unreleased held-object measurement")
            return False
        target = self.placement_targets.get(place)
        if not self.placement_target_is_trusted(place, target):
            rospy.logerr("Destination opening is unknown or expired: %s", place)
            return False
        gap = self.finger_gap()
        if gap is None or not 0.015 <= gap <= 0.12:
            rospy.logerr("Holding unconfirmed before release: gap=%s", gap)
            return False
        hand = self.tf_buffer.lookup_transform(
            "map", "hand_palm_link", rospy.Time(0), rospy.Duration(2.0)).transform
        center, _, extent = self.held_object_world(hand)
        error = target["uncertainty"]
        margin = .01 + error
        bottom = center[2] - extent[2]
        if (abs(center[0] - target["x"]) + extent[0] + margin >= target["half_x"]
                or abs(center[1] - target["y"]) + extent[1] + margin >= target["half_y"]
                or not target["rim_z"] + error <= bottom <= target["rim_z"] + .05 - error):
            rospy.logerr("Object outside safe opening at %s: center=%s extent=%s error=%s",
                         place, center, extent, error)
            return False
        record = self.last_placement_result
        record["release_commanded"] = True
        record["holding_state"] = "unknown"
        # 失敗・例外で途中まで開いた可能性も保持不明として扱う。
        if not gripper.command(1):
            rospy.logerr("Gripper release command failed at %s; holding is unknown", place)
            return False
        gap = self.finger_gap()
        if gap is None or not .10 <= gap <= .20:
            rospy.logerr("Gripper opening unconfirmed at %s: gap=%s", place, gap)
            return False
        self.held_object["released"] = True
        self.grasp_holding_possible = False
        self.task1_holding_object = False
        record.update(released=True, holding_state="released")
        # 開放後は点群の変化で収納を確認する。開口キャッシュは次の物体へ流用しない。
        getattr(self, "confirmed_placement_targets", {}).pop(place, None)
        try:
            self.manipulation_scene.remove_attached_object("hand_palm_link", name="b_held_object")
            self.manipulation_scene.remove_world_object("b_held_object")
        except Exception as error:
            rospy.logerr("Released object scene cleanup failed: %s", error)
            return False
        self.current_grasp_geometry = None
        return True

    def close_object_gripper(self):
        """62401738 いしい
        62418450 向井悠人
        C版の力制御で握り、終了・成功を確認して結果を返す。
        """
        import actionlib
        from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortGoal

        client = actionlib.SimpleActionClient(
            "/hsrb/gripper_controller/grasp", GripperApplyEffortAction)
        if not client.wait_for_server(rospy.Duration(3.0)):
            return False
        client.send_goal(GripperApplyEffortGoal(effort=-0.5))
        if not client.wait_for_result(rospy.Duration(5.0)):
            client.cancel_goal()
            client.wait_for_result(rospy.Duration(1.0))
            return False
        return client.get_state() == GoalStatus.SUCCEEDED

    def navigate_task1_view(self, name):
        """62418450 向井悠人
        観測した床上物体とmove_base経路の台車掃引範囲を照合し、衝突経路では進まない。
        """
        import numpy as np
        from nav_msgs.srv import GetPlan
        from wrs_algorithm.util.mathematics import quaternion_from_euler

        if not self.move_placement_pose("look_at_near_floor"):
            return False
        points, _ = self.read_placement_cloud(rospy.Time.now())
        base = self.get_robot_pose()
        if base is None:
            return False
        distance = np.linalg.norm(points[:, :2] - np.asarray(base[:2]), axis=1)
        obstacles = points[(distance > .35) & (distance < 2.5)
                           & (points[:, 2] > .04) & (points[:, 2] < .45)]
        start, goal = PoseStamped(), PoseStamped()
        start.header.frame_id = goal.header.frame_id = "map"
        start.pose.position.x, start.pose.position.y = base[:2]
        start.pose.orientation = quaternion_from_euler(math.degrees(base[2]), 0, 0)
        x, y, yaw = self.coordinates["positions"][name]
        goal.pose.position.x, goal.pose.position.y = x, y
        goal.pose.orientation = quaternion_from_euler(yaw, 0, 0)
        rospy.wait_for_service("/move_base/make_plan", timeout=2.0)
        path = rospy.ServiceProxy("/move_base/make_plan", GetPlan)(start, goal, 0.0).plan.poses
        if not path:
            return False
        previous = np.asarray(base[:2])
        for waypoint in path:
            current = np.array([waypoint.pose.position.x, waypoint.pose.position.y])
            vector = current - previous
            denominator = float(vector.dot(vector))
            ratio = (np.clip((obstacles[:, :2] - previous).dot(vector) / denominator, 0, 1)
                     if denominator > 1e-8 else np.zeros(len(obstacles)))
            closest = previous + ratio[:, None] * vector
            if len(obstacles) and np.min(np.linalg.norm(obstacles[:, :2] - closest, axis=1)) < .30:
                rospy.logerr("Depth-observed object blocks approach to %s", name)
                return False
            previous = current
        return self.navigate_for_placement(name)

    @staticmethod
    def grasp_approach_position(target, preliminary, distance=0.15):
        """指定した方向からのアプローチ位置を計算する。"""
        offsets = {
            "+x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "+y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "+z": (0.0, 0.0, 1.0),
        }

        if preliminary not in offsets:
            raise ValueError(
                "Unknown grasp approach direction: {}".format(preliminary)
            )

        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("Invalid grasp approach distance")

        direction = offsets[preliminary]

        return [
            float(value) + distance * direction[index]
            for index, value in enumerate(target)
        ]

    @staticmethod
    def move_grasp_hand(position, yaw, pitch, roll, linear=False):
        """62418450 向井悠人
        衝突を確認した完全な経路を実行し、経路計画と軌道実行の失敗を区別して記録する。
        """
        from wrs_algorithm.util.mathematics import quaternion_from_euler

        commander = whole_body.WHOLE_BODY_CMD
        frame = commander.get_pose_reference_frame()
        target = PoseStamped()
        target.header.frame_id = "map"
        target.pose.position.x, target.pose.position.y, target.pose.position.z = position
        target.pose.orientation = quaternion_from_euler(yaw, pitch, roll)
        try:
            commander.set_start_state_to_current_state()
            if not linear:
                commander.set_pose_target(target)
                success = bool(commander.go(wait=True))
                if not success:
                    rospy.logerr("Grasp pose planning/execution failed: target=%s", position)
                return success
            commander.set_pose_reference_frame("map")
            trajectory, fraction = commander.compute_cartesian_path(
                [target.pose], .005, 0., avoid_collisions=True)
            if fraction < 1.0 or not math.isfinite(fraction):
                rospy.logerr("Grasp Cartesian planning incomplete: fraction=%s target=%s; "
                             "trajectory was not executed", fraction, position)
                return False
            success = bool(commander.execute(trajectory, wait=True))
            if not success:
                rospy.logerr("Grasp Cartesian execution failed: fraction=%s target=%s",
                             fraction, position)
            return success
        finally:
            commander.stop()
            commander.clear_pose_targets()
            commander.set_pose_reference_frame(frame)

    def put_in_fallback(self, label, reason):
        """62418450 向井悠人
        ラベルを分類できない物体だけをBin Bへ一時退避し、理由を記録する。
        """
        if (getattr(self, "grasp_motion_fault", False)
                or getattr(self, "drawer_motion_fault", False)):
            self.last_placement_result = {
                "label": label, "stage": "fallback_blocked", "released": False,
                "success": False, "fallback": True, "fallback_reason": reason,
                "error": "Unresolved grasp/drawer control failure"}
            rospy.logerr("Temporary placement blocked: %s", json.dumps(self.last_placement_result))
            return False
        rospy.logwarn("Temporary storage of unclassified %s in Bin B: %s", label, reason)
        return self.put_in_place("bin_b_place", "put_in_bin", label=label, fallback_reason=reason)

    @staticmethod
    def point_drawer_camera(tilt=-0.65):
        """62418450 向井悠人
        腕を動かさずに引き出しの観測方向へ頭部を向ける。
        """
        head = whole_body.HEAD_CMD
        head.set_joint_value_target({"head_pan_joint": 0.0, "head_tilt_joint": tilt})
        try:
            return bool(head.go(wait=True))
        finally:
            head.stop()

    def complete_task1_placement(self, label):
        """62418450 向井悠人
        正しい収納またはBin Bへの一時退避を行い、失敗して保持中なら探索を止める。
        """
        self.task1_holding_object = True
        succeeded = self.put_object_by_label(label)
        released = getattr(self, "last_placement_result", {}).get("released", False)
        self.task1_holding_object = not released
        self.task1_motion_fault = not succeeded
        self.last_task1_result = dict(getattr(self, "last_placement_result", {}))
        if not succeeded:
            rospy.logerr("Task 1 placement failed for %s; holding=%s", label,
                         self.task1_holding_object)
        return bool(succeeded)

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
    #   tools_in_drawertop = trofast_2, tools_in_drawerbottom = trofast_3
    #   shapeitems_in_drawerleft = trofast_1
    # 下段を約0.25m、上段をその半分まで開け、下段手前の開口を収納に使う。
    # 上段は単段引き出しと反対側（mapの-x側）へ回り込んで操作する。
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
    DRAWERS = {
        # x: 取っ手の x、z: 取っ手の高さ、pull: 引く距離、tilt: 取っ手を見るときの頭の傾き [rad]
        "bottom": {"x": 0.20, "z": 0.28, "pull": 0.25, "tilt": -0.85, "stand": "drawer_standpoint"},
        "top": {"x": 0.20, "z": 0.54, "pull": 0.125, "tilt": -0.6, "stand": "drawer_standpoint"},
        "left": {"x": 0.53, "z": 0.28, "pull": 0.25, "tilt": -0.85,
                 "stand": "drawer_left_standpoint"},
    }
    DRAWER_TOP_PULL_ALONE = 0.125           # 上段だけ開ける場合も引きすぎない
    DRAWER_KNOB_FACE_Y = -0.355             # 取っ手の板の前面の y（3 つとも同じ）
    DRAWER_FINGER_REACH = 0.063             # 手のひら(hand_palm_link)から指先までの前後距離（実測）
    DRAWER_GRASP_DEPTH = 0.022              # 指先を板の前面からどれだけ奥へ入れるか（板の裏の軸をつかむ）
    DRAWER_APPROACH = 0.15                  # 差し込む前に手を止める、取っ手の手前の距離
    DRAWER_HAND_RPY = (-90, 100, 0)         # 取っ手をつかむ手の向き（シミュレータで到達を確認済み）
    DRAWER_Z_RETRY = (0.0, 0.012, -0.012)   # つかめなかったときに高さを少しずらして試す
    DRAWER_GRASP_EFFORT = -1.0              # grasp アクションで握る強さ（負で閉じる。-0.5 では途中で外れた）
    DRAWER_GRASP_EFFORT_SAFE = -0.5         # -1.0 で閉じなかったときに使う強さ（04:26 に 3 つとも開いた値）
    DRAWER_PULL_TRIES = 3                   # 途中で外れたときに、つかみ直して引く回数
    DRAWER_HOLD_GAP = 0.03                  # 引き終わりの指の間隔がこれ以上なら、最後までつかんでいた
    DRAWER_PULL_SPEED = 0.05                # 台車で引くときの速さ [m/s]
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
    DRAWER_DROP = {
        # 上段だけ開いた場合は上段の内部へ置く。
        "top": (0.2, -0.36, "put_in_drawer_high"),
        # 下段だけ開いた場合の置き場所。両段が開いたら実測した前縁の間を使う。
        "bottom": (0.2, -0.22, "put_in_bin"),
        # 横の引き出し 0.25: 棚の前面（y -0.39）より手前の空いている所、縁の高さ 0.33
        "left": (0.53, -0.26, "put_in_bin"),
    }
    DRAWER_PLACE_NAME = "drawer_top_place"      # 工具（2 段の引き出し）
    DRAWER_LEFT_PLACE_NAME = "drawer_left_place"  # 形のもの（横の引き出し）

    def lookup_xyz(self, child):
        """62401738 いしい
        map 座標での child フレームの位置 (x, y, z) を返す。取れなければ None。
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                "map", child, rospy.Time(0), rospy.Duration(1.0)).transform.translation
            return trans.x, trans.y, trans.z
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    def finger_gap(self):
        """62401738 いしい
        左右の指先の間隔 [m]。閉じても間隔が残っていれば何かをつかんでいる。取れなければ None。
        """
        left = self.lookup_xyz("hand_l_finger_tip_frame")
        right = self.lookup_xyz("hand_r_finger_tip_frame")
        if left is None or right is None:
            return None
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def base_yaw(self):
        """62401738 いしい
        map 座標での台車の向き [rad]。取れなければ None。
        """
        try:
            rot = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rospy.Time(0), rospy.Duration(1.0)).transform.rotation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        return math.atan2(2 * (rot.w * rot.z + rot.x * rot.y),
                          1 - 2 * (rot.y * rot.y + rot.z * rot.z))

    def move_hand_by_base(self, target_y, keep_x, timeout):
        """62401738 いしい
        腕は止めたまま台車だけを動かし、手のひらを map の y = target_y まで真っすぐ運ぶ。
        全身 IK で引くと台車が回ってしまうことがあるので、引き出しは台車でまっすぐ引く。
        台車の向きが -90° でなくても、map の +y 方向へ動くように速度を変換する。
        """
        rate = rospy.Rate(20)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        try:
            while not rospy.is_shutdown() and rospy.Time.now() < deadline:
                hand = self.lookup_xyz("hand_palm_link")
                yaw = self.base_yaw()
                if hand is None or yaw is None:
                    return False
                remain = target_y - hand[1]
                if abs(remain) < 0.005:
                    return True
                speed = self.DRAWER_PULL_SPEED
                vy_map = max(-speed, min(speed, 2.0 * remain))
                vx_map = max(-0.03, min(0.03, 2.0 * (keep_x - hand[0])))
                # map 座標の速度 → 台車座標の速度
                vel_x = math.cos(yaw) * vx_map + math.sin(yaw) * vy_map
                vel_y = -math.sin(yaw) * vx_map + math.cos(yaw) * vy_map
                omni_base.cmd_vel(vel_x, vel_y, 0)
                rate.sleep()
            return False
        finally:
            for _ in range(3):
                omni_base.cmd_vel(0, 0, 0)
                rospy.sleep(0.05)

    def open_drawer_gripper(self):
        """62418450 向井悠人
        指幅で開放を確認し、必要な場合だけ開く指令を送り、終了または中止を確認する。
        """
        import actionlib
        from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        gap = self.finger_gap()
        if gap is not None and 0.10 <= gap <= 0.20:
            rospy.loginfo("Drawer gripper already open: gap=%.3f", gap)
            return True
        client = actionlib.SimpleActionClient(
            "/hsrb/gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        if not client.wait_for_server(rospy.Duration(3.0)):
            rospy.logerr("Drawer gripper opening controller unavailable")
            return False
        point = JointTrajectoryPoint()
        point.positions = [1.0]
        point.velocities = [0.0]
        point.time_from_start = rospy.Duration(1.0)
        trajectory = JointTrajectory()
        trajectory.joint_names = ["hand_motor_joint"]
        trajectory.points = [point]
        client.send_goal(FollowJointTrajectoryGoal(trajectory=trajectory))
        if not client.wait_for_result(rospy.Duration(6.0)):
            client.cancel_goal()
            if not client.wait_for_result(rospy.Duration(2.0)):
                rospy.logerr("Drawer gripper cancellation unconfirmed: state=%s gap=%s",
                             client.get_state(), self.finger_gap())
                return False
        result = client.get_result()
        state = client.get_state()
        gap = self.finger_gap()
        if gap is None or not 0.10 <= gap <= 0.20:
            rospy.logerr("Drawer gripper did not open: state=%s result=%s gap=%s",
                         state, result, gap)
            return False
        # 目標角度の収束待ちが期限を超えても、指令が終了し指が開いていれば取っ手を離せている。
        if state in (GoalStatus.PENDING, GoalStatus.ACTIVE,
                     GoalStatus.PREEMPTING, GoalStatus.RECALLING):
            rospy.logerr("Drawer gripper command is still active: state=%s gap=%s", state, gap)
            return False
        if state != GoalStatus.SUCCEEDED or result is None or result.error_code != 0:
            rospy.logwarn("Drawer gripper command ended with state=%s result=%s; "
                          "release verified by gap=%.3f", state, result, gap)
        else:
            rospy.loginfo("Drawer gripper release verified: gap=%.3f", gap)
        return True

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

    def grasp_drawer_handle(self, level, knob):
        """62401738 いしい / 62418450 向井悠人
        knob（x, 板の前面の y, z）の取っ手をつかむ。つかめたら (手の x, 手の y, 手の z) を返す。
        つかめなければ高さを少しずつずらして試し、だめなら None。
        """
        yaw, pitch, roll = self.DRAWER_HAND_RPY
        gap_min, gap_max = self.DRAWER_FINGER_GAP_RANGE
        pos_x, face_y, knob_z = knob
        grasp_y = face_y - self.DRAWER_GRASP_DEPTH + self.DRAWER_FINGER_REACH
        if not self.drawer_pose("grasp_on_table"):
            self.drawer_motion_fault = True
            return None
        for dz in self.DRAWER_Z_RETRY:
            pos_z = knob_z + dz
            rospy.loginfo("grasp %s drawer handle (hand x=%.3f y=%.3f z=%.3f)",
                          level, pos_x, grasp_y, pos_z)
            if not self.open_drawer_gripper():
                self.drawer_motion_fault = True
                return None
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
                return pos_x, grasp_y, pos_z
            rospy.logwarn("did not grasp the %s drawer handle (gap %s); retry", level, gap)
            if not self.open_drawer_gripper():
                self.drawer_motion_fault = True
                return None
            if not whole_body.move_end_effector_pose(
                    pos_x, grasp_y + self.DRAWER_APPROACH, pos_z, yaw, pitch, roll):
                self.drawer_motion_fault = True
                return None
        return None

    def pull_drawer(self, level, pull=None):
        """62401738 いしい / 62418450 向井悠人
        level（"bottom" / "top" / "left"）の引き出しを pull [m] だけ引き出す。
        取っ手を深度カメラで見つける → つかむ → 台車でまっすぐ下がって引く → 離す → 手を抜く
        → もう一度取っ手を見て、どこまで開いたか確かめる。
        11/9 の実行では、つかみ方しだいで引いている途中に指が外れることがあった
        （成功した回は引き終わりまで指の間隔が 0.06 前後のまま、失敗した回は 0.01 まで閉じていた）。
        そこで、足りなければ取っ手の今の位置からもう一度つかんで、残りを引く（DRAWER_PULL_TRIES 回まで）。
        """
        yaw, pitch, roll = self.DRAWER_HAND_RPY
        if pull is None:
            pull = self.DRAWERS[level]["pull"]
        gap_min = self.DRAWER_FINGER_GAP_RANGE[0]
        self.look_at_drawer(level)
        knob = self.detect_drawer_knob(level, self.DRAWER_KNOB_FACE_Y)
        if knob is None:
            rospy.logwarn("%s drawer knob not seen; use the fixed position", level)
            knob = (self.DRAWERS[level]["x"], self.DRAWER_KNOB_FACE_Y, self.DRAWERS[level]["z"])
        face0 = knob[1]                 # 閉じているときの取っ手の位置
        goal = face0 + pull             # 引き終わりの取っ手の位置
        # 手を引く先は「最初の取っ手の位置 + pull」で固定する（つかみ直しても、これより先へは引かない）。
        # 上段は傾くと取っ手が下がって見え、開いた量を少なく測る。測った量で引き足すと
        # 引き出しが抜けて落ちた（11/9 06:00）ので、引く先は測った量に頼らない。
        hand_goal_y = face0 - self.DRAWER_GRASP_DEPTH + self.DRAWER_FINGER_REACH + pull
        tries = 1 if level == "top" else self.DRAWER_PULL_TRIES   # 上段は傾くのでつかみ直さない
        opened = 0.0
        for attempt in range(tries):
            remain = goal - knob[1]
            rospy.loginfo("pull %s drawer: try %d, %.3f m left", level, attempt + 1, remain)
            hand = self.grasp_drawer_handle(level, knob)
            if hand is None:
                break
            pos_x, grasp_y, pos_z = hand
            remain = hand_goal_y - grasp_y   # 実際に引く量（決めた所より先へは引かない）
            if remain < 0.03:
                self.open_drawer_gripper()
                break
            pull_reached = self.move_hand_by_base(
                hand_goal_y, pos_x, timeout=remain / self.DRAWER_PULL_SPEED * 3 + 5)
            if not pull_reached:
                rospy.logwarn("base pull did not finish; try whole body IK")
                pull_reached = bool(whole_body.move_end_effector_pose(
                    pos_x, hand_goal_y, pos_z, yaw, pitch, roll))
            gap_end = self.finger_gap()
            if not self.open_drawer_gripper():
                self.drawer_motion_fault = True
                return False
            # 引けなかった場合も、現在の手の位置から離れる。目標へさらに引き足さない。
            current_hand = self.lookup_xyz("hand_palm_link")
            if current_hand is None:
                self.drawer_motion_fault = True
                return False
            retreat_y = current_hand[1] + 0.12
            if not self.move_hand_by_base(retreat_y, current_hand[0], timeout=8):
                if not whole_body.move_end_effector_pose(
                        current_hand[0], retreat_y, current_hand[2], yaw, pitch, roll):
                    rospy.logerr("Cannot retract after releasing the %s drawer handle", level)
                    self.drawer_motion_fault = True
                    return False
            rospy.loginfo("%s drawer: finger gap at end of pull %s", level, gap_end)
            # 取っ手がどこまで来たかをカメラで確かめる（上段は傾くので高さの範囲を広げる）
            self.look_at_drawer(level)
            seen = self.detect_drawer_knob(level, knob[1] + remain, z_tol=0.10, y_tol=0.25)
            if seen is not None:
                knob = seen
                opened = seen[1] - face0
            elif pull_reached and gap_end is not None and gap_end >= self.DRAWER_HOLD_GAP:
                # 取っ手は見えなかったが、最後までつかんでいた → 引けたとみなす
                opened = pull
            rospy.loginfo("%s drawer opened %.3f m (goal %.2f)", level, opened, pull)
            if opened >= pull * 0.8:
                break
            if seen is None:
                break  # 取っ手の位置が分からないので、もう一度つかみに行けない
            self.go_to_drawer(level)  # 立ち位置に戻ってつかみ直す
        if opened >= pull * 0.5:
            self.drawer_opened = getattr(self, "drawer_opened", {})
            self.drawer_opened[level] = opened
            self.drawer_handles = getattr(self, "drawer_handles", {})
            self.drawer_handles[level] = (knob[0], face0 + opened, knob[2])
            return True
        rospy.logwarn("%s drawer did not open (%.3f m)", level, opened)
        return False

    @staticmethod
    def move_side_drawer_hand(x, y, z):
        """62418450 向井悠人
        台車を動かさず、腕だけで外側から上段の取っ手へ手を運ぶ。
        """
        from wrs_algorithm.util.mathematics import quaternion_from_euler

        target = PoseStamped()
        target.header.frame_id = "map"
        target.pose.position.x, target.pose.position.y, target.pose.position.z = x, y, z
        target.pose.orientation = quaternion_from_euler(0, 100, 0)
        arm = whole_body.ARM_CMD
        try:
            arm.set_start_state_to_current_state()
            arm.set_pose_target(target, "hand_palm_link")
            return bool(arm.go(wait=True))
        finally:
            arm.stop()
            arm.clear_pose_targets()

    def pull_top_drawer_from_side(self, pull):
        """62418450 向井悠人
        開いた下段の外側へ回り込み、上段を横からつかんでmapの+y方向へ引く。
        側方の移動先は深度で見た取っ手から決め、腕だけで届かなければ中止する。
        """
        # 開いた下段の正面へ戻らず、まず外側の待機点から上段を見る。
        self.coordinates["positions"]["drawer_top_side_entry"] = [
            self.DRAWERS["top"]["x"] - 0.65, 0.55, -55]
        if not self.drawer_pose("all_neutral"):
            self.drawer_motion_fault = True
            return False
        if not self.navigate_for_placement("drawer_top_side_entry"):
            self.drawer_motion_fault = True
            return False
        if not self.look_at_drawer("top"):
            return False
        knob = self.detect_drawer_knob("top", self.DRAWER_KNOB_FACE_Y)
        if knob is None:
            rospy.logwarn("Top drawer side approach requires an observed handle")
            return False
        x, face_y, z = knob
        # 単段引き出し（+x側）とは反対の外側を通る。手先は台車より左へ約8cmにある。
        side_x = x - 0.65
        grasp_y = face_y + 0.025
        self.coordinates["positions"]["drawer_top_side"] = [
            side_x, grasp_y - self.DRAWER_HAND_LEFT, 0]
        if not self.navigate_for_placement("drawer_top_side"):
            self.drawer_motion_fault = True
            return False
        grasp_x = x + self.DRAWER_GRASP_DEPTH - self.DRAWER_FINGER_REACH
        approached = False
        pulled = False
        try:
            if not self.open_drawer_gripper():
                raise RuntimeError("Cannot open the hand before the side approach")
            if not self.move_side_drawer_hand(grasp_x - self.DRAWER_APPROACH, grasp_y, z):
                raise RuntimeError("Cannot reach the top handle from the side")
            approached = True
            if not self.move_side_drawer_hand(grasp_x, grasp_y, z):
                raise RuntimeError("Cannot insert the hand beside the top handle")
            if not self.close_gripper_on_handle():
                raise RuntimeError("Cannot close the hand on the top handle")
            gap = self.finger_gap()
            low, high = self.DRAWER_FINGER_GAP_RANGE
            if gap is None or not low <= gap <= high:
                raise RuntimeError("Top side grasp failed (gap {})".format(gap))
            hand = self.lookup_xyz("hand_palm_link")
            if hand is None:
                raise RuntimeError("Top side grasp has no hand feedback")
            rospy.loginfo("Pull top drawer from side: %.3f m along map +y", pull)
            if not self.move_hand_by_base(
                    hand[1] + pull, hand[0], timeout=pull / self.DRAWER_PULL_SPEED * 3 + 5):
                raise RuntimeError("Top side pull did not finish")
            gap = self.finger_gap()
            pulled = gap is not None and low <= gap <= high
        except Exception as error:
            rospy.logerr("Top drawer side operation failed: %s", error)
            self.drawer_motion_fault = True
        finally:
            # 取っ手を離し、手を外側（-x）へ抜いてから腕をたたむ。
            try:
                if not self.open_drawer_gripper():
                    raise RuntimeError("Cannot release the top drawer handle")
                if approached:
                    hand = self.lookup_xyz("hand_palm_link")
                    if hand is None or not self.move_side_drawer_hand(
                            hand[0] - self.DRAWER_APPROACH, hand[1], hand[2]):
                        raise RuntimeError("Cannot retract the hand sideways")
                if not self.drawer_pose("all_neutral"):
                    raise RuntimeError("Cannot fold the arm after the side approach")
            except Exception as error:
                rospy.logerr("Top drawer side retreat failed: %s", error)
                self.drawer_motion_fault = True
        if getattr(self, "drawer_motion_fault", False):
            return False
        if not self.navigate_for_placement("drawer_top_side_entry"):
            self.drawer_motion_fault = True
            return False
        # 外側の待機点から、実際に動いた取っ手を見て開いた量を確かめる。
        if not self.look_at_drawer("top"):
            self.drawer_motion_fault = True
            return False
        seen = self.detect_drawer_knob("top", face_y + pull, z_tol=0.10, y_tol=0.25)
        opened = max(0.0, seen[1] - face_y) if seen is not None else 0.0
        rospy.loginfo("Top drawer side result: opened %.3f m, holding at end %s", opened, pulled)
        if seen is not None:
            self.drawer_handles = getattr(self, "drawer_handles", {})
            self.drawer_handles["top"] = seen
        if not pulled or opened < pull * 0.5:
            return False
        self.drawer_opened = getattr(self, "drawer_opened", {})
        self.drawer_opened["top"] = opened
        return True

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
        """62401738 いしい / 62418450 向井悠人
        腕をたためたことを確認してlevelの正面へ移動し、失敗を後続処理へ伝える。
        """
        if not self.drawer_pose("all_neutral"):
            self.drawer_motion_fault = True
            return False
        stand = self.DRAWERS[level]["stand"]
        if not self.navigate_for_placement(stand):
            rospy.logwarn("cannot reach %s for the %s drawer", stand, level)
            self.drawer_motion_fault = True
            return False
        return True

    def open_drawers(self):
        """62401738 いしい / 62418450 向井悠人
        右下（0.25）→右上（外側から下段の半分）→左下（0.25）の順に開ける。
        開いた引き出しに物体を落とす位置を登録する（工具: DRAWER_PLACE_NAME、形のもの: DRAWER_LEFT_PLACE_NAME）。
        """
        if getattr(self, "drawers_attempted", False):
            return getattr(self, "drawers_open", False)
        self.drawers_attempted = True
        self.drawers_open = False
        if not rospy.get_param("~use_drawer", self.USE_DRAWER):
            return False
        rospy.loginfo("#### open drawers ####")
        if not self.open_drawer_gripper():
            self.drawer_motion_fault = True
            return False
        for name, pos in self.DRAWER_STANDPOINTS.items():
            self.coordinates["positions"][name] = list(pos)
        opened = {}
        try:
            if not self.go_to_drawer("bottom"):
                raise RuntimeError("Cannot prepare the bottom drawer")
            opened["bottom"] = self.pull_drawer("bottom")
            if getattr(self, "drawer_motion_fault", False):
                raise RuntimeError("Cannot release/retract at the bottom drawer")
            if opened["bottom"]:
                bottom_pull = getattr(self, "drawer_opened", {}).get(
                    "bottom", self.DRAWERS["bottom"]["pull"])
                top_pull = min(self.DRAWERS["top"]["pull"], bottom_pull * 0.5)
                opened["top"] = self.pull_top_drawer_from_side(top_pull)
            else:
                if not self.go_to_drawer("top"):
                    raise RuntimeError("Cannot prepare the top drawer")
                opened["top"] = self.pull_drawer("top", pull=self.DRAWER_TOP_PULL_ALONE)
            if getattr(self, "drawer_motion_fault", False):
                raise RuntimeError("Stop drawer sequence after failed side operation")
            if not self.go_to_drawer("left"):
                raise RuntimeError("Cannot prepare the left drawer")
            opened["left"] = self.pull_drawer("left")
            if getattr(self, "drawer_motion_fault", False):
                raise RuntimeError("Cannot release/retract at the left drawer")
        except Exception as error:  # 引き出しで失敗しても Task 1 は続ける
            rospy.logerr("drawer error: %s", error)
        finally:
            if not self.open_drawer_gripper():
                self.drawer_motion_fault = True
            if not getattr(self, "drawer_motion_fault", False):
                if not self.drawer_pose("all_neutral"):
                    self.drawer_motion_fault = True
        self.last_drawer_result = dict(opened)
        rospy.loginfo("drawer result: %s", json.dumps(opened, sort_keys=True))

        # 工具の置き場所（2 段の引き出し）
        drop, level = None, None
        if opened.get("bottom"):
            handles = getattr(self, "drawer_handles", {})
            bottom = handles.get("bottom")
            top = handles.get("top")
            if bottom is not None and top is not None:
                # 上段の前縁より手前、下段の前縁より奥にある開口の中央を使う。
                gap = bottom[1] - top[1]
                if gap >= 0.06:
                    drop = (bottom[0], (bottom[1] + top[1]) / 2, "put_in_bin")
                    level = "bottom"
                    self.drawer_opening_limits = getattr(self, "drawer_opening_limits", {})
                    self.drawer_opening_limits[self.DRAWER_PLACE_NAME] = {
                        "back": top[1] + .01, "front": bottom[1] - .01,
                        "max_z": bottom[2] + .12}
                    rospy.loginfo("Lower drawer exposed opening: %.3f m", gap)
                else:
                    rospy.logwarn("Lower drawer opening too narrow: %.3f m", gap)
            else:
                drop, level = self.DRAWER_DROP["bottom"], "bottom"
        elif opened.get("top"):
            drop, level = self.DRAWER_DROP["top"], "top"
        if drop is not None:
            self.register_drawer_place(self.DRAWER_PLACE_NAME, drop, level)
        # 形のものの置き場所（横の引き出し）
        if opened.get("left"):
            self.register_drawer_place(
                self.DRAWER_LEFT_PLACE_NAME, self.DRAWER_DROP["left"], "left")
        self.drawers_open = any(opened.values())
        if not self.drawers_open:
            rospy.logwarn("could not open the drawers. use bin_b instead.")
        return self.drawers_open

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
        if knob is None:
            rospy.logwarn("opened %s drawer knob not seen; use the planned drop point", level)
            return True
        old = self.drawer_drops[place]
        self.drawer_drops[place] = (knob[0] + off_x, knob[1] + off_y)
        rospy.loginfo("drop point for %s: (%.3f, %.3f) -> (%.3f, %.3f) from the knob",
                      place, old[0], old[1],
                      self.drawer_drops[place][0], self.drawer_drops[place][1])
        return True

    DRAWER_PALM_TO_TIPS = 0.058   # 腕を前に水平に出した姿勢での、手のひらから指先までの距離（FK）

    def align_hand_over_drawer(self, place):
        """62401738 いしい
        腕の姿勢はそのまま台車だけ動かし、指先を引き出しの落とす位置（drawer_drops[place]）の真上に合わせる。
        合わせきれなくても、その場で離す（引き出しの中には入っている見込み）。
        """
        x, y = self.drawer_drops[place]
        if not self.move_hand_by_base(y + self.DRAWER_PALM_TO_TIPS, x, timeout=8):
            rospy.logwarn("could not align the hand over %s; release here", place)
        hand = self.lookup_xyz("hand_palm_link")
        if hand is not None:
            rospy.loginfo("release over %s: palm (%.3f, %.3f, %.3f), target tips (%.3f, %.3f)",
                          place, hand[0], hand[1], hand[2], x, y)
        return True

    def select_drawer_place(self, label, category):
        """62401738 いしい
        引き出し行きの物体の置き場所を返す。工具は 2 段の引き出し、形のもの（Shape items）は横の引き出し。
        その引き出しが開いていなければ例外にして、Task 1 側の代わりの置き場所（put_in_fallback）に任せる。
        """
        place = (self.DRAWER_PLACE_NAME if category == "drawer_tools"
                 else self.DRAWER_LEFT_PLACE_NAME)
        places = getattr(self, "drawer_places", {})
        if place not in places:
            raise ValueError("Drawer for {} ({}) is not open".format(label, category))
        rospy.loginfo("place %s (%s) at %s", label, category, place)
        return place, places[place]


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
