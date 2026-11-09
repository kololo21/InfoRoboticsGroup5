#!/usr/bin/env bash
# 自分の PC(GPU なし・Docker あり。WSL2 の Ubuntu や Linux)に、CPU だけで動くシミュレータ環境を作る。
#
#   tools/local_cpu/setup_local.sh [リポジトリURL] [ブランチ]     # 既定は第5班のリポジトリの haruki
#
# やること(何度実行してもよい。済んだ手順は飛ばす):
#   1. ~/sim_local を作り、docker-compose.yml(CPU 用)を置く
#   2. リポジトリを ~/sim_local/workspace に clone する(授業の ~/sim_ws/workspace と同じ位置づけ)
#   3. 公開イメージを取得して起動する(simulator 約6GB + workspace 約7GB。初回は時間がかかる)
#   4. workspace コンテナ内で catkin build する
#   5. YOLO を CPU で動かすためのライブラリ(CPU 版 PyTorch + ultralytics)を /workspace/.yolo_env に入れる
# docker compose の --build / down は使わない。
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
SIM_DIR="${SIM_DIR:-$HOME/sim_local}"
REPO_URL="${1:-https://github.com/kololo21/InfoRoboticsGroup5.git}"
BRANCH="${2:-haruki}"

command -v docker >/dev/null || { echo "ERROR: docker が見つかりません"; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: docker を使えません(docker グループに入っているか、Docker が起動しているか確認)"; exit 1; }

echo "== 1/5 $SIM_DIR を用意"
mkdir -p "$SIM_DIR"
cp "$HERE/docker-compose.yml" "$SIM_DIR/docker-compose.yml"

echo "== 2/5 リポジトリ($BRANCH)を $SIM_DIR/workspace に取得"
if [ -d "$SIM_DIR/workspace/.git" ]; then
  echo "   (すでにあります。git pull はしません。最新にするときは自分で git pull してください)"
else
  git clone -b "$BRANCH" "$REPO_URL" "$SIM_DIR/workspace"
fi
# 実験ループ(tools/experiment)が Seed を切り替えるのに使う。授業の ~/sim_ws/change_seed.sh と同じもの
cp "$SIM_DIR/workspace/src/wrs_controller21/scripts/change_seed.sh" "$SIM_DIR/change_seed.sh"
chmod +x "$SIM_DIR/change_seed.sh"

cd "$SIM_DIR"
dc() { docker compose "$@" 2> >(grep -v obsolete >&2); }

echo "== 3/5 イメージの取得とコンテナの起動"
dc up -d
sleep 8

echo "== 4/5 catkin build(workspace コンテナ内)"
dc exec -T workspace bash -lc 'cd /workspace && source /opt/ros/noetic/setup.bash && { [ -d .catkin_tools ] || catkin init; } && catkin build 2>&1 | tail -3'

echo "== 5/5 YOLO のライブラリ(CPU 版)"
if dc exec -T workspace test -d /workspace/.yolo_env/ultralytics; then
  echo "   (すでに入っています)"
else
  dc exec -T workspace bash -lc '/usr/bin/python3 -m pip install --no-cache-dir --target /workspace/.yolo_env \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu torchvision==0.19.1+cpu ultralytics==8.3.40 "numpy<2" 2>&1 | tail -2'
fi

echo
echo "完了。次はダッシュボードを起動して、ブラウザから実行する:"
echo "  python3 $SIM_DIR/workspace/tools/experiment/dashboard.py --dir $SIM_DIR"
echo "詳しくは tools/local_cpu/README.md"
