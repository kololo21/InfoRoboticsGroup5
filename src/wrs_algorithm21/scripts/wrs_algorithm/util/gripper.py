# -*- coding: utf-8 -*-
"""
HSRのハンドを制御するためのユーティリティモジュール
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import moveit_commander
import rospy


# moveitでの制御対象としてハンドを指定
GRIPPER_CMD = moveit_commander.MoveGroupCommander(str("gripper"))


def command(value):
    """
    ハンドを制御

    Parameters
    ----------
        value (float): ハンドの開き具合 (0：閉じる、1:開く)

    Return
    ------
        正しく動作すればTrue, そうでなければFalse

    """

    GRIPPER_CMD.set_joint_value_target(str("hand_motor_joint"), value)
    success = GRIPPER_CMD.go()
    rospy.sleep(6)
    return success
