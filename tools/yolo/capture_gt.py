#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シミュレータの頭部カメラ画像と、Gazebo が知っている「正解の物体名・位置」を一緒に保存する。
ロボットは動かさない（走らせている間、横で撮るだけ）。保存したフォルダを gt_autolabel.py に渡すと、
正解のクラス名が付いた YOLO 形式のラベルになる。

保存形式
  <out>/seed<N>/0000_image.png … 頭部カメラの画像
  <out>/seed<N>/0000_gt.json   … 画像に写っているはずの物体
      {"objects": [{"model": Gazebo のモデル名, "label": 検出器のクラス名, "u", "v": 画素,
                    "dist": カメラからの距離 [m], "depth": その画素の深度 [m]（なければ null）}]}

使い方（シミュレータを起動して Play を押してから。start_all.sh と同時に動かしてよい）
  source /opt/ros/noetic/setup.bash && source /workspace/devel/setup.bash
  /usr/bin/python3 /workspace/tools/yolo/capture_gt.py --seed 86 --interval 2
  /usr/bin/python3 /workspace/tools/yolo/capture_gt.py --seed 86 --once     # 今の1枚だけ
"""
import argparse
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_images import image_msg_to_rgb, next_index  # noqa: E402  pylint: disable=wrong-import-position

RGB_TOPIC = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
INFO_TOPIC = "/hsrb/head_rgbd_sensor/rgb/camera_info"
DEPTH_TOPIC = "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw"
CAMERA_LINK = "hsrb::head_tilt_link"   # Gazebo に出ているリンク（カメラは固定関節なのでこのリンクにまとめられている）
CAMERA_PARENT = "head_tilt_link"       # TF での同じリンク
CAMERA_FRAME = "head_rgbd_sensor_rgb_frame"   # カメラの光学座標（x 右, y 下, z 前）
CLASS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src",
                          "wrs_detector", "config", "wod-c_classes.json")
# Gazebo のモデル名（ycb_NNN_xxx）と検出器のクラス名が違うもの
YCB_TO_LABEL = {
    "medium_clamp": "clamp", "large_clamp": "clamp", "extra_large_clamp": "clamp",
    "cups": "cup", "colored_wood_blocks": "colored_wood_block", "marbles": "marble",
    "padlock": "padlock", "bleach_cleanser": "bleach_cleanser", "9_peg_hole_test": "nine_hole_peg_test",
    "nine_hole_peg_test": "nine_hole_peg_test", "phillips_screwdriver": "unknown",
    "flat_screwdriver": "unknown", "foam_brick": "foam_brick", "t_shirt": "black_t_shirt",
    "black_t_shirt": "black_t_shirt", "rubiks_cube": "rubiks_cube",
}
MAX_DIST = 3.0      # これより遠い物体は小さすぎるので書かない [m]
MOVE_MIN = 0.03     # 前に撮った位置からこれだけカメラが動いたら撮る [m]
TURN_MIN = 0.03     # または、これだけ向きが変わったら撮る（クォータニオンの差）
STATE_PERIOD = 0.02  # Gazebo の状態をためる間隔 [秒]


def label_of(model, names):
    """62401738 いしい
    Gazebo のモデル名（task1_tool_ycb_050_medium_clamp_4 など）から検出器のクラス名を返す。物体でなければ None。
    """
    match = re.search(r"ycb_\d+(?:-[a-z])?_([a-z0-9_]+?)(?:_\d+)?$", model)
    if match is None:
        return None
    name = match.group(1)
    name = YCB_TO_LABEL.get(name, name)
    return name if name in names else None


def quat_rotate_inv(quat, vec):
    """62401738 いしい
    クォータニオン quat（x, y, z, w）の逆回転を vec に掛ける（world → リンク座標）。
    """
    q_x, q_y, q_z, q_w = -quat.x, -quat.y, -quat.z, quat.w
    v_x, v_y, v_z = vec
    t_x = 2.0 * (q_y * v_z - q_z * v_y)
    t_y = 2.0 * (q_z * v_x - q_x * v_z)
    t_z = 2.0 * (q_x * v_y - q_y * v_x)
    return (v_x + q_w * t_x + q_y * t_z - q_z * t_y,
            v_y + q_w * t_y + q_z * t_x - q_x * t_z,
            v_z + q_w * t_z + q_x * t_y - q_y * t_x)


def depth_at(depth_msg, u, v):
    """62401738 いしい
    深度画像の (u, v) の周り 5x5 画素の中央値 [m]。測れていなければ None。
    """
    import numpy as np  # pylint: disable=import-outside-toplevel
    if depth_msg is None:
        return None
    dtype, scale = {"16UC1": (np.uint16, 0.001), "32FC1": (np.float32, 1.0)}.get(
        depth_msg.encoding, (None, None))
    if dtype is None:
        return None
    img = np.frombuffer(depth_msg.data, dtype=dtype).reshape(depth_msg.height, -1)[:, :depth_msg.width]
    patch = img[max(0, v - 2):v + 3, max(0, u - 2):u + 3].astype(float) * scale
    patch = patch[np.isfinite(patch) & (patch > 0.05)]
    return float(np.median(patch)) if patch.size else None


def main():
    """62401738 いしい
    画像・深度・Gazebo の物体位置を同じ時刻で取り、物体の中心を画像に投影して保存する。
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", required=True, help="今の部屋の Seed（フォルダ名に使う）")
    parser.add_argument("--out", default="/workspace/datasets/capture_gt")
    parser.add_argument("--interval", type=float, default=2.0, help="撮る間隔 [秒]")
    parser.add_argument("--once", action="store_true", help="1枚だけ撮って終わる")
    parser.add_argument("--max", type=int, default=2000, help="撮る枚数の上限")
    args = parser.parse_args()

    import rospy  # pylint: disable=import-outside-toplevel
    from gazebo_msgs.msg import LinkStates, ModelStates  # pylint: disable=import-outside-toplevel
    from sensor_msgs.msg import CameraInfo, Image  # pylint: disable=import-outside-toplevel
    from PIL import Image as PilImage  # pylint: disable=import-outside-toplevel
    import tf2_ros  # pylint: disable=import-outside-toplevel

    with open(CLASS_JSON) as json_file:
        names = {c["name"] for c in json.load(json_file)["classes"]}
    folder = os.path.join(args.out, "seed{}".format(args.seed))
    os.makedirs(folder, exist_ok=True)

    rospy.init_node("capture_gt", anonymous=True)
    info = rospy.wait_for_message(INFO_TOPIC, CameraInfo, timeout=10)
    # head_tilt_link から見たカメラの位置・向き（固定）を TF から1回だけ取る
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)
    mount = tf_buffer.lookup_transform(CAMERA_PARENT, CAMERA_FRAME, rospy.Time(0),
                                       rospy.Duration(10)).transform
    f_x, f_y, c_x, c_y = info.K[0], info.K[4], info.K[2], info.K[5]
    # 画像は撮ってから届くまでに（シミュレータ時間で）10 秒以上遅れることがある。Gazebo の状態を時刻付きで
    # ためておき、画像の撮影時刻に一番近いものを使う（最新の状態と組み合わせると、動いている間は位置がずれる）
    states = collections.deque(maxlen=6000)   # (時刻, カメラの姿勢, [(モデル名, 位置)])
    latest = {"cam": None, "rgb": None, "stored": 0.0}
    depths = collections.deque(maxlen=60)

    def on_links(msg):
        if CAMERA_LINK in msg.name:
            latest["cam"] = msg.pose[msg.name.index(CAMERA_LINK)]

    def on_models(msg):
        now = rospy.get_time()
        if latest["cam"] is None or now - latest["stored"] < STATE_PERIOD:
            return
        latest["stored"] = now
        states.append((now, latest["cam"], [
            (name, pose.position) for name, pose in zip(msg.name, msg.pose)
            if name.startswith("task2a_") or label_of(name, names) is not None]))

    def on_rgb(msg):
        latest["rgb"] = msg

    rospy.Subscriber("/gazebo/link_states", LinkStates, on_links, queue_size=1)
    rospy.Subscriber("/gazebo/model_states", ModelStates, on_models, queue_size=1)
    rospy.Subscriber(RGB_TOPIC, Image, on_rgb, queue_size=1, buff_size=2 ** 24)
    rospy.Subscriber(DEPTH_TOPIC, Image, depths.append, queue_size=1, buff_size=2 ** 24)
    started = rospy.get_time()
    last = None
    used_stamp = 0.0
    count = 0
    while not rospy.is_shutdown() and count < args.max:
        rospy.sleep(0.2 if args.once else args.interval)
        rgb = latest["rgb"]
        if rgb is None or not states:
            continue
        stamp = rgb.header.stamp.to_sec()
        if stamp <= used_stamp or stamp < started or stamp < states[0][0]:
            continue   # 同じ画像、または状態をため始める前に撮った画像
        state = min(states, key=lambda item: abs(item[0] - stamp))
        if abs(state[0] - stamp) > 2 * STATE_PERIOD:
            continue
        used_stamp = stamp
        depth = min(depths, key=lambda msg: abs(msg.header.stamp.to_sec() - stamp), default=None)
        if depth is not None and abs(depth.header.stamp.to_sec() - stamp) > 0.05:
            depth = None
        cam = state[1]
        cam_pos = (cam.position.x, cam.position.y, cam.position.z)
        cam_dir = (cam.orientation.x, cam.orientation.y, cam.orientation.z, cam.orientation.w)
        if last is not None and not args.once:
            moved = math.dist(cam_pos, last[0])
            turned = math.dist(cam_dir, last[1])
            if moved < MOVE_MIN and turned < TURN_MIN:
                continue   # 止まっている間は同じ画像になるので撮らない
        last = (cam_pos, cam_dir)

        objects = []
        for model, position in state[2]:
            label = label_of(model, names)
            rel = (position.x - cam_pos[0], position.y - cam_pos[1], position.z - cam_pos[2])
            in_link = quat_rotate_inv(cam.orientation, rel)
            in_link = (in_link[0] - mount.translation.x, in_link[1] - mount.translation.y,
                       in_link[2] - mount.translation.z)
            right, down, fwd = quat_rotate_inv(mount.rotation, in_link)
            if fwd < 0.15 or fwd > MAX_DIST:
                continue
            u_px = int(round(c_x + f_x * right / fwd))
            v_px = int(round(c_y + f_y * down / fwd))
            if not (0 <= u_px < rgb.width and 0 <= v_px < rgb.height):
                continue
            objects.append({"model": model, "label": label, "u": u_px, "v": v_px,
                            "dist": round(fwd, 3), "depth": depth_at(depth, u_px, v_px)})
        index = next_index(folder)
        PilImage.fromarray(image_msg_to_rgb(rgb)).save(
            os.path.join(folder, "{:04d}_image.png".format(index)))
        with open(os.path.join(folder, "{:04d}_gt.json".format(index)), "w") as json_file:
            json.dump({"stamp": stamp, "objects": objects}, json_file, indent=1)
        count += 1
        rospy.loginfo("%04d: %d objects in view", index, len(objects))
        if args.once:
            break


if __name__ == "__main__":
    main()
