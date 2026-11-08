#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シミュレータの頭部カメラ画像を、授業データセット（hsr_sim_obj_1000）と同じ形式で保存する。
保存したフォルダを sam3_autolabel.py に渡すと、ラベルが付いて学習に使える。

保存形式（授業データセットと同じ）
  <out>/seed<N>/0000_image.png   … 頭部カメラの画像（640x480）
  <out>/seed<N>/0000_pose.json   … 撮ったときのロボットの位置 {"x", "y", "theta"[rad]}
  （ラベル 0000_image.json は付けない。SAM 3 で付ける）

モード
  tour   : ロボットを coordinates.json の地点に順に動かし、首を振って撮る（他の人が動かしていないとき）
  passive: ロボットは動かさない。start_all.sh で走らせている間、一定間隔で撮る
           （本番と同じ視点の画像が集まる。動いていないときは撮らない）

使い方（WebIDE。シミュレータを起動して Play を押し、Seed を合わせてから）
  source /opt/ros/noetic/setup.bash && source /workspace/devel/setup.bash
  python3 /workspace/tools/yolo/capture_images.py --seed 86 --mode tour
  python3 /workspace/tools/yolo/capture_images.py --seed 25 --mode passive --interval 2
  → /workspace/datasets/capture/seed86/ などに保存。続きの番号から書き足す。
"""
import argparse
import glob
import json
import math
import os
import re
import time

IMAGE_TOPIC = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src",
                          "wrs_algorithm21", "config")
# tour で回る地点（coordinates.json の名前）。物体がある所と、Task 2a・2b の場所
TOUR_PLACES = ["tall_table", "long_table_r", "near_long_table_l", "drawer",
               "bin_a", "standby_2a", "shelf", "person_a", "person_b"]
TOUR_TILTS = (-0.3, -0.6, -1.0)   # 首の上下 [rad]（遠く・机・床）
TOUR_PANS = (-0.6, 0.0, 0.6)      # 首の左右 [rad]


def image_msg_to_rgb(msg):
    """62401738 いしい
    sensor_msgs/Image（rgb8 / bgr8 / rgba8 / bgra8）を、高さ×幅×3 の RGB の numpy 配列にする。
    cv_bridge を使わないので、catkin_py3 でも YOLO の環境でも動く。
    """
    import numpy as np  # pylint: disable=import-outside-toplevel
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
    if msg.encoding not in channels:
        raise ValueError("unsupported image encoding: {}".format(msg.encoding))
    n_ch = channels[msg.encoding]
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    pixels = row[:, :msg.width * n_ch].reshape(msg.height, msg.width, n_ch)[:, :, :3]
    if msg.encoding.startswith("bgr"):
        pixels = pixels[:, :, ::-1]
    return np.ascontiguousarray(pixels)


def next_index(folder):
    """62401738 いしい
    フォルダにある NNNN_image.png の次の番号を返す（書き足すときに上書きしない）
    """
    numbers = [int(m.group(1)) for m in
               (re.match(r"(\d+)_image\.png$", os.path.basename(p))
                for p in glob.glob(os.path.join(folder, "*_image.png"))) if m]
    return max(numbers) + 1 if numbers else 0


def save_sample(folder, index, rgb, pose):
    """62401738 いしい
    画像と位置を授業データセットと同じ名前で保存し、画像のパスを返す
    """
    from PIL import Image  # pylint: disable=import-outside-toplevel
    stem = os.path.join(folder, "%04d" % index)
    Image.fromarray(rgb).save(stem + "_image.png")
    with open(stem + "_pose.json", "w") as pose_file:
        json.dump({"x": round(pose[0], 3), "y": round(pose[1], 3),
                   "theta": round(pose[2], 3)}, pose_file)
    return stem + "_image.png"


def tour_views(coordinates):
    """62401738 いしい
    tour で撮る (地点名, 首の上下, 首の左右) の一覧を返す。coordinates.json にない地点は飛ばす。
    """
    positions = coordinates["positions"]
    return [(place, tilt, pan) for place in TOUR_PLACES if place in positions
            for tilt in TOUR_TILTS for pan in TOUR_PANS]


def moved_enough(last, pose, min_move=0.2, min_turn=0.3):
    """62401738 いしい
    passive で、前に撮った位置 last から min_move [m] 以上動いたか min_turn [rad] 以上回ったか
    （止まっている間に同じ画像を何枚も撮らないため）
    """
    if last is None:
        return True
    turn = abs(math.atan2(math.sin(pose[2] - last[2]), math.cos(pose[2] - last[2])))
    return math.hypot(pose[0] - last[0], pose[1] - last[1]) >= min_move or turn >= min_turn


class Capturer(object):
    """62401738 いしい
    ROS からカメラ画像とロボットの位置を受け取って保存する
    """

    def __init__(self, folder):
        """62401738 いしい
        ROS ノードを立ち上げ、tf を受け取る準備をする
        """
        import rospy  # pylint: disable=import-outside-toplevel,import-error
        import tf2_ros  # pylint: disable=import-outside-toplevel,import-error
        rospy.init_node("capture_images", anonymous=True)
        self.rospy = rospy
        self.folder = folder
        self.index = next_index(folder)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def robot_pose(self):
        """62401738 いしい
        tf から (x, y, theta[rad]) を返す。取れなければ None
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                "map", "base_footprint", self.rospy.Time(0),
                self.rospy.Duration(1.0)).transform
        except Exception:  # pylint: disable=broad-except
            return None
        rot = trans.rotation
        theta = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                           1.0 - 2.0 * (rot.y ** 2 + rot.z ** 2))
        return (trans.translation.x, trans.translation.y, theta)

    def capture(self):
        """62401738 いしい
        新しい画像を1枚待って、その時の位置と一緒に保存する。保存したら True
        """
        from sensor_msgs.msg import Image  # pylint: disable=import-outside-toplevel,import-error
        pose = self.robot_pose()
        if pose is None:
            self.rospy.logwarn("no robot pose from tf; skip")
            return False
        msg = self.rospy.wait_for_message(IMAGE_TOPIC, Image, timeout=5.0)
        path = save_sample(self.folder, self.index, image_msg_to_rgb(msg), pose)
        self.rospy.loginfo("saved %s", path)
        self.index += 1
        return True

    def run_tour(self, coordinates, poses):
        """62401738 いしい
        地点を順に回り、各地点で首を振って撮る。腕は all_neutral にしてカメラに映らないようにする
        """
        from wrs_algorithm.util import omni_base, whole_body  # pylint: disable=all
        neutral = poses.get("all_neutral", {})
        whole_body.move_to_joint_positions(neutral)
        current = None
        for place, tilt, pan in tour_views(coordinates):
            if place != current:
                x_pos, y_pos, yaw = coordinates["positions"][place]
                whole_body.move_to_joint_positions(dict(neutral, head_tilt_joint=-0.3))
                omni_base.go_abs(x_pos, y_pos, yaw)
                current = place
            whole_body.move_to_joint_positions({"head_pan_joint": pan, "head_tilt_joint": tilt})
            self.rospy.sleep(1.0)  # 首が止まってから撮る
            self.capture()

    def run_passive(self, interval):
        """62401738 いしい
        interval 秒ごとに、前に撮った所から動いていれば撮る（Ctrl-C で終了）
        """
        last = None
        while not self.rospy.is_shutdown():
            pose = self.robot_pose()
            if pose is not None and moved_enough(last, pose):
                if self.capture():
                    last = pose
            self.rospy.sleep(interval)


def main():
    """62401738 いしい
    引数を読み、tour か passive で撮影する
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="今シミュレータで使っている Seed")
    parser.add_argument("--mode", choices=("tour", "passive"), default="tour")
    parser.add_argument("--interval", type=float, default=2.0, help="passive で撮る間隔 [s]")
    parser.add_argument("--out", default="/workspace/datasets/capture")
    args = parser.parse_args()

    folder = os.path.join(args.out, "seed%d" % args.seed)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    capturer = Capturer(folder)
    start = capturer.index
    try:
        if args.mode == "tour":
            with open(os.path.join(CONFIG_DIR, "coordinates.json")) as coord_file:
                coordinates = json.load(coord_file)
            with open(os.path.join(CONFIG_DIR, "poses.json")) as pose_file:
                poses = json.load(pose_file)
            capturer.run_tour(coordinates, poses)
        else:
            capturer.run_passive(args.interval)
    finally:
        print("saved %d images -> %s" % (capturer.index - start, folder))
        time.sleep(0.1)


if __name__ == "__main__":
    main()
