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
from std_msgs.msg import String
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

    def get_latest_detection(self):
        """サンプルのまま（担当なし）
        最新の認識結果が到着するまで待つ
        """
        res = self.detection_clt(GetObjectDetectionRequest())
        return res.bboxes

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
        把持しやすさのscoreが最も高い物体を返す。
        Task1で得点対象となる物体を誤って除外しない。
        """
        extracted = []
        extract_str = "detected object list\n"
        ignore_str = ""

        # 元のIGNORE_LISTに含まれていても、Task1の得点対象なら除外しない
        task1_scoreable_labels = [
            "small_marker",
            "large_marker",
            "lego_duplo",
            "spatula",
            "nine_hole_peg_test",
            "fork",
            "spoon"
        ]

        for obj in obj_list:
            # BBoxの値が不正な物体は除外
            if obj.w <= 0 or obj.h <= 0 or obj.score <= 0:
                continue

            info_str = "{:<20}({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(
                obj.label, obj.score, obj.x, obj.y, obj.w, obj.h
            )

            if (obj.label in cls.IGNORE_LIST
                    and obj.label not in task1_scoreable_labels):
                ignore_str += "- ignored  : " + info_str
                continue

            score = cls.calc_score_bbox(obj)

            extracted.append({
                "bbox": obj,
                "score": score,
                "label": obj.label
            })

            extract_str += "- extracted: {:07.3f} ".format(score) + info_str
            rospy.loginfo("Detected label: %s", obj.label)

        rospy.loginfo(extract_str + ignore_str)

        # scoreの高い物体を優先する
        extracted.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        if not extracted:
            rospy.logwarn("No graspable object was detected.")
            return None

        selected = extracted[0]
        obj = selected["bbox"]

        rospy.loginfo(
            "selected bbox: {} ({:.2%}, {}, {}, {}, {})".format(
                obj.label, obj.score,
                obj.x, obj.y, obj.w, obj.h
            )
        )

        return selected

    @classmethod
    def calc_score_bbox(cls, bbox):
        """62408993 Masami Sakuma
        detector_msgs/BBoxのスコアを計算する。

        評価項目：
        ・検出信頼度
        ・画像中央に近いか
        ・BBoxの大きさ
        """

        # BBoxの中心座標
        center_x = bbox.x + bbox.w / 2.0
        center_y = bbox.y + bbox.h / 2.0

        # 画像中央からの距離
        # 元のプログラムで使用していた基準座標を維持する
        dx = abs(320.0 - center_x) / 320.0
        dy = abs(360.0 - center_y) / 240.0

        distance = dx + dy

        # 中央に近いほど高得点
        center_score = max(0.0, 1.0 - distance / 2.0)

        # 検出信頼度（0～1に制限）
        confidence_score = max(
            0.0,
            min(1.0, float(bbox.score))
        )

        # BBoxが小さすぎる物体は把持しにくいと考える
        image_area = 640.0 * 480.0
        bbox_area = float(bbox.w * bbox.h)

        size_score = min(
            1.0,
            bbox_area / 10000.0
        )

        # 総合スコア
        score = (
            confidence_score * 0.5
            + center_score * 0.3
            + size_score * 0.2
        )

        return score

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
        把持の一連の動作を行う。

        preliminary：
            +y, -y, +x, -x のいずれか
        """

        if preliminary not in ["+y", "-y", "+x", "-x"]:
            raise RuntimeError(
                "unknown grasp preliminary type [{}]".format(preliminary)
            )

        rospy.loginfo(
            "move hand to grasp (%.2f, %.2f, %.2f)",
            pos_x, pos_y, pos_z
        )

        # 安全退避位置
        grasp_back_safe = {
            "x": pos_x,
            "y": pos_y,
            "z": pos_z + self.GRASP_BACK_SAFE["z"]
        }

        # 把持直前の位置
        grasp_back = {
            "x": pos_x,
            "y": pos_y,
            "z": pos_z + self.GRASP_BACK["z"]
        }

        # 実際の把持位置
        grasp_pos = {
            "x": pos_x,
            "y": pos_y,
            "z": pos_z
        }

        sign = 1 if "+" in preliminary else -1

        # 接近方向に応じて退避位置をずらす
        if "x" in preliminary:
            grasp_back_safe["x"] += (
                sign * self.GRASP_BACK_SAFE["xy"]
            )
            grasp_back["x"] += (
                sign * self.GRASP_BACK["xy"]
            )

        elif "y" in preliminary:
            grasp_back_safe["y"] += (
                sign * self.GRASP_BACK_SAFE["xy"]
            )
            grasp_back["y"] += (
                sign * self.GRASP_BACK["xy"]
            )

        # グリッパーを開く
        gripper.command(1)

        # 安全退避位置から把持位置へ移動
        whole_body.move_end_effector_pose(
            grasp_back_safe["x"],
            grasp_back_safe["y"],
            grasp_back_safe["z"],
            yaw, pitch, roll
        )

        whole_body.move_end_effector_pose(
            grasp_back["x"],
            grasp_back["y"],
            grasp_back["z"],
            yaw, pitch, roll
        )

        whole_body.move_end_effector_pose(
            grasp_pos["x"],
            grasp_pos["y"],
            grasp_pos["z"],
            yaw, pitch, roll
        )

        # 物体を把持
        gripper.command(0)

        # 安全退避
        whole_body.move_end_effector_pose(
            grasp_back_safe["x"],
            grasp_back_safe["y"],
            grasp_back_safe["z"],
            yaw, pitch, roll
        )

    def grasp_from_front_side(self, grasp_pos):
        """62408993 Masami Sakuma
        正面把持を行う。
        """

        grasp_pos.y -= self.HAND_PALM_OFFSET

        rospy.loginfo(
            "grasp_from_front_side (%.2f, %.2f, %.2f)",
            grasp_pos.x, grasp_pos.y, grasp_pos.z
        )

        self.grasp_from_side(
            grasp_pos.x,
            grasp_pos.y,
            grasp_pos.z,
            -90, -100, 0,
            "-y"
        )

    def grasp_from_upper_side(self, grasp_pos):
        """62408993 Masami Sakuma
        上面把持を行う。
        """

        grasp_pos.z += self.HAND_PALM_Z_OFFSET

        rospy.loginfo(
            "grasp_from_upper_side (%.2f, %.2f, %.2f)",
            grasp_pos.x, grasp_pos.y, grasp_pos.z
        )

        self.grasp_from_side(
            grasp_pos.x,
            grasp_pos.y,
            grasp_pos.z,
            -90, -160, 0,
            "-y"
        )

    def exec_graspable_method(self, grasp_pos, label=""):
        """62408993 Masami Sakuma
        Task1専用。
        物体の位置と種類に応じて把持方法を選択する。
        """

        graspable_y = 1.85
        desk_y = 1.5
        desk_z = 0.35

        # 把持できない位置にある物体は対象外
        if graspable_y < grasp_pos.y and grasp_pos.z < desk_z:
            rospy.logwarn(
                "Object is out of graspable area: y=%.2f, z=%.2f",
                grasp_pos.y, grasp_pos.z
            )
            return False

        # ラベル名の表記ゆれに対応
        label = label.lower().replace(" ", "_").replace("-", "_")

        # 上面から把持した方がよい物体
        upper_grasp_labels = [
            "cup",
            "cups",
            "frisbee",
            "bowl"
        ]

        if label in upper_grasp_labels:
            method = self.grasp_from_upper_side

        elif desk_y < grasp_pos.y and grasp_pos.z < desk_z:
            # 机の下側にある物体は正面から把持
            method = self.grasp_from_front_side

        else:
            # それ以外は上面から把持
            method = self.grasp_from_upper_side

        method(grasp_pos)

        return True

    @staticmethod
    def normalize_placement_label(label):
        """62418450 向井悠人
        YOLO/A側のラベルを配置用に統一し、空のラベルを拒否する。
        """
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Placement requires a nonempty object label")
        label = "_".join(label.strip().lower().replace("-", "_").split())
        return label

    def select_place_for_label(self, label):
        """62418450 向井悠人
        ラベルに対応する配置設定を選び、未設定なら既存のBinへ退避する。
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
        }
        destinations = {
            "tray": (("tray_a_place", "put_in_tray"),
                     ("tray_b_place", "put_in_tray")),
            "container_a": (("container_a_place", "put_in_container_a"),),
            "container_b": (("container_b_place", "put_in_container_b"),),
            "bin_a": (("bin_a_place", "put_in_bin"),),
        }
        category = next((name for name, labels in categories.items() if label in labels), None)
        # Drawerは開閉・配置動作の検証が必要なため、現段階ではBinへ退避する。
        candidates = [(place, pose) for place, pose in destinations.get(category, ())
                      if place in self.coordinates["positions"] and pose in self.poses]
        if candidates:
            selected = self.choose_shortest_placement(candidates)
            rospy.loginfo("place %s (%s) at %s", label, category, selected[0])
            return selected

        fallback = "bin_b_place" if category is None else "bin_a_place"
        rospy.logwarn("No verified destination for %s (%s); use %s", label, category, fallback)
        return fallback, "put_in_bin"

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
            rospy.logerr("Placement failed for %s: %s", label, error)
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
            if (place not in self.coordinates["positions"]
                    or any(not self.poses.get(name)
                           for name in ("all_neutral", into_pose))):
                return False
            stages = (("transport", lambda: self.move_placement_pose("all_neutral")),
                      ("navigate", lambda: self.navigate_for_placement(place)),
                      ("place", lambda: self.move_placement_pose(into_pose)),
                      ("release", lambda: gripper.command(1)),
                      ("retreat", lambda: self.move_placement_pose("all_neutral")))
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
        Task1を実行する。

        ・複数の探索地点を巡回する
        ・把持しやすい物体を選択する
        ・物体カテゴリに応じた投入先へ運ぶ
        ・制限時間を考慮して探索する
        """

        rospy.loginfo("#### start Task 1 ####")

        # 探索地点
        hsr_position = [
            ("tall_table", "look_at_tall_table"),
            ("near_long_table_l", "look_at_near_floor"),
            ("long_table_r", "look_at_tall_table")
        ]

        # 置き場所の判断は B（put_object_by_label）に任せる（統合時に一本化）
        placed = []

        # 探索地点ごとの連続未検出回数
        empty_count = [0] * len(hsr_position)

        # 探索回数の上限
        max_attempts_per_place = 12
        attempt_count = [0] * len(hsr_position)

        # 15分の制限時間に対して、少し余裕を持って終了する
        start_time = rospy.Time.now()
        time_limit = 14 * 60

        rospy.loginfo("Start searching objects.")

        # すべての探索地点で物体が見つからなくなるまで巡回
        while not all(count >= 2 for count in empty_count):

            # 制限時間を確認
            elapsed = (rospy.Time.now() - start_time).to_sec()

            if elapsed >= time_limit:
                rospy.logwarn("Task1 time limit is approaching. Stop searching.")
                break

            for i, (plc, pose) in enumerate(hsr_position):

                if empty_count[i] >= 2:
                    continue

                if attempt_count[i] >= max_attempts_per_place:
                    empty_count[i] = 2
                    continue

                # 探索地点へ移動
                self.goto_name(plc)
                self.change_pose(pose)

                # 把持前にグリッパーを開く
                gripper.command(1)

                # 物体検出
                detected_objs = self.get_latest_detection()

                graspable_obj = self.get_most_graspable_obj(
                    detected_objs.bboxes
                )

                attempt_count[i] += 1

                # 物体が見つからなかった場合
                if graspable_obj is None:
                    empty_count[i] += 1

                    rospy.loginfo(
                        "No object at %s (%d/2)",
                        plc, empty_count[i]
                    )

                    continue

                # 物体が見つかったら未検出回数をリセット
                empty_count[i] = 0

                label = graspable_obj["label"]
                grasp_bbox = graspable_obj["bbox"]

                rospy.loginfo("Selected object: %s", label)

                # BBoxから3次元把持座標を取得
                grasp_pos = self.get_grasp_coordinate(grasp_bbox)

                if grasp_pos is None:
                    rospy.logwarn("Failed to obtain grasp coordinate.")
                    continue

                # 把持姿勢へ移動
                self.change_pose("grasp_on_table")

                # 物体を把持
                grasp_success = self.exec_graspable_method(
                    grasp_pos, label
                )

                self.change_pose("all_neutral")

                if not grasp_success:
                    rospy.logwarn(
                        "Could not grasp object: %s", label
                    )
                    continue

                # B担当の配置処理へラベルを渡す（ラベル→置き場所の対応・移動・配置）
                if self.put_object_by_label(label):
                    placed.append(label)
                else:
                    rospy.logwarn("Placement failed for %s", label)

        rospy.loginfo("#### finish Task 1 ####")
        rospy.loginfo("Placed objects (%d): %s", len(placed), ", ".join(placed))

    # ------------------------------------------------------------------
    # Task 2a / 2b（担当 C）
    # ------------------------------------------------------------------
    TASK2A_LANES_X = [1.7, 2.15, 2.6]   # 通過するレーンの x 座標（サンプルの xa, xb, xc）
    TASK2A_STEPS_Y = [2.5, 2.9, 3.3]    # 各ステップで進む y 座標
    TASK2A_START = (2.5, 1.85)          # standby_2a の座標
    TASK2A_YAW = 90                     # 進行方向（+y）を向いたまま移動し、カメラで前方を見る
    TASK2A_AREA = {"x": (1.2, 3.2), "y": (1.9, 3.9), "z_max": 0.3}  # 床の障害物とみなす範囲
    TASK2A_SAFE_DIST = 0.35   # ロボット中心の経路と障害物中心の最小距離 [m]（HSR 半径約0.22 + 物体）
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
    TASK2A_STEP_MAX = 0.5             # 1回に進む最大距離 [m]（進むたびに検出し直す）
    TASK2A_MAX_MOVES = 10             # 移動回数の上限

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
        経路がなければ安全距離を2割縮めてもう一度探し、それでもなければ None。
        """
        for safe_dist in (cls.TASK2A_SAFE_DIST, cls.TASK2A_SAFE_DIST * 0.8):
            path = cls.plan_path_astar(current, goal, obstacles, safe_dist)
            if path is None:
                rospy.logwarn("A*: no path with safe_dist=%.2f", safe_dist)
                continue
            corners = cls.simplify_path(path, obstacles, safe_dist)
            rospy.loginfo("A* path: %s",
                          " -> ".join(["({:.2f}, {:.2f})".format(p[0], p[1]) for p in corners]))
            target = corners[1] if len(corners) > 1 else corners[0]
            dist = math.hypot(target[0] - current[0], target[1] - current[1])
            if dist > cls.TASK2A_STEP_MAX:
                ratio = cls.TASK2A_STEP_MAX / dist
                target = (current[0] + (target[0] - current[0]) * ratio,
                          current[1] + (target[1] - current[1]) * ratio)
            return target
        return None

    def execute_avoid_blocks(self):
        """62401738 いしい
        床の障害物を避けながら go_throw_2a まで進む。
        1回進むごとに検出し直し、見つけた障害物は map 座標で覚えておいて（近づいて見えなくなっても
        避ける）、A* で経路を計画し直す。A* で経路が見つからないときは、3レーン方式
        （select_next_waypoint）で次の段へ進む。
        """
        obstacles = []
        current = self.TASK2A_START
        for _ in range(self.TASK2A_MAX_MOVES):
            self.update_task2a_obstacles(obstacles)
            target = self.plan_next_point(current, self.TASK2A_GOAL, obstacles)
            if target is not None:
                waypoint = [target[0], target[1], self.TASK2A_YAW]
            else:
                steps = [i for i, s_y in enumerate(self.TASK2A_STEPS_Y) if s_y > current[1] + 0.05]
                if not steps:
                    break
                waypoint = self.select_next_waypoint(steps[0], obstacles, current)
            self.goto_pos(waypoint)
            current = (waypoint[0], waypoint[1])
            if math.hypot(current[0] - self.TASK2A_GOAL[0], current[1] - self.TASK2A_GOAL[1]) < 0.1:
                break

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
        self.execute_avoid_blocks()

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

    def run(self):
        """62418450 向井悠人
        全てのタスクを実行する（Task 1 → Task 2a → Task 2b）。
        Task 2b の指示文は棚の前に行くと届くので、2a を先に実行する。
        """
        self.change_pose("all_neutral")
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
