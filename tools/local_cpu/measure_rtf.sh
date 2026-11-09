#!/usr/bin/env bash
# シミュレータの速さ(RTF = シミュレーション時間 / 実時間)を測る。1 に近いほど実時間どおり。
#
#   bash measure_rtf.sh [~/sim_local] [秒数]
#
# 実験が動いていないときに使う(動いているシミュレータの時計を読むだけで、再起動はしない)。
# 物理演算の一時停止を解除する(Play を押したのと同じ)。画面(gzclient)は止めない。
set -u
DIR="${1:-$HOME/sim_local}"
SEC="${2:-30}"
cd "$DIR" || exit 1
ros() { docker compose exec -T simulator bash -c "source /opt/ros/noetic/setup.bash; $*" 2> >(grep -v obsolete >&2); }
clk() { ros 'rostopic echo -n1 /clock' | awk '$1=="secs:"{s=$2} $1=="nsecs:"{n=$2} END{printf "%.3f", s+n/1e9}'; }

if pgrep -f run_experiment.sh >/dev/null; then
  echo "ERROR: 実験が動いています。終わってから測ってください"; exit 1
fi
ros 'rosservice call /gazebo/unpause_physics' >/dev/null
sleep 3
s0=$(clk); t0=$(date +%s.%N)
sleep "$SEC"
s1=$(clk); t1=$(date +%s.%N)
awk -v a="$s0" -v b="$s1" -v c="$t0" -v d="$t1" \
  'BEGIN{printf "シミュレーション %.1f秒 / 実時間 %.1f秒  RTF = %.2f\n", b-a, d-c, (b-a)/(d-c)}'
echo "nproc=$(nproc)"
