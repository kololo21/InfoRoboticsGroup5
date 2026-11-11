# 授業(研究室)と同じ採点・衝突判定のシミュレータを、CPU だけで動かすためのイメージ。
#
# 授業のシミュレータは keio-smilab25/sim_wrs_ws の Dockerfile で作られ、
# 公開イメージの採点ノード(tmc_gazebo_task_evaluators)と起動ファイル(hsrb_wrs_gazebo_launch)を
# 研究室が変更した版に差し替えている。主な違い:
#   - 衝突判定が働く(公開版は --minimal-comms のため接触情報が流れず、衝突を検出しない)
#   - 衝突すると /detect_msg に相手の名前が流れる(wrs_main_node_*.py が購読している)
#   - 引き出し3つ開放のボーナスは2周期連続で3つ開いているときだけ
#   - task2 の得点表(20/10/30/10)、既定 seed 86
#
# ここでは研究室の Dockerfile と同じ差し替えを、公開の CPU イメージ(tmc_wrs_binary)の上で行う。
# 研究室のイメージは tmc_wrs_docker:noetic-nvidia が土台なので、土台だけが違う。
# ビルドは build_lab_sim.sh から行う(ソースは研究室のリポジトリから、決まったコミットで取り出す)。
FROM ghcr.io/hsr-project/tmc_wrs_binary:noetic-latest

SHELL ["/bin/bash", "-c"]

COPY src/tmc_gazebo_task_evaluators /lab_ws/src/tmc_gazebo_task_evaluators
COPY src/hsrb_wrs_gazebo_launch /lab_ws/src/hsrb_wrs_gazebo_launch
COPY SOURCES.txt /lab_ws/SOURCES.txt

# 2つのパッケージだけをビルドし、公開版の同名パッケージの上に入れる(/opt/ros/noetic)。
# 研究室の Dockerfile は /wrs_ws 全体を作り直すが、他のパッケージは変えていないので、2つだけで同じになる。
RUN cd /lab_ws \
    && source /opt/ros/noetic/setup.bash \
    && catkin config --install --install-space /opt/ros/noetic \
    && catkin build --no-status \
    && rm -rf /lab_ws/build /lab_ws/devel /lab_ws/logs \
    && strings /opt/ros/noetic/lib/tmc_gazebo_task_evaluators/undesired_contact_detector | grep -q detect_msg
