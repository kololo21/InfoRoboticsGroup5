#!/usr/bin/env bash
# シミュレータが何で重いかを切り分ける: 条件ごとにシミュレータを再起動して RTF を測る(ノードは動かさない)。
#
#   bash ablate_rtf.sh [条件...]        # 既定: base nomon iters50 step6
#   SEC=60 DIR=~/sim_local bash ablate_rtf.sh base iters50
#
# 条件:
#   base    何も変えない(基準)
#   nomon   採点・監視ノード(*_detector, undesired_contact_detector, wrs_score_counter)を止める
#   kill=REGEX 採点・監視ノードのうち名前が REGEX に一致するものだけ止める(例 kill=drawer kill=undesired)
#   iters50 物理のソルバー反復回数 100 -> 50(実行時に /gazebo/set_physics_properties)
#   step6   物理の刻み幅 3ms -> 6ms
#   iters50+step6 の両方
# どれも再起動で元に戻る。物理を変えた条件は挙動が変わるので、スコアの記録には使わない。
set -u
DIR="${DIR:-$HOME/sim_local}"
SEC="${SEC:-60}"
cd "$DIR" || exit 1
if pgrep -f run_experiment.sh >/dev/null; then echo "ERROR: 実験が動いています。終わってから実行してください"; exit 1; fi

dc() { docker compose "$@" 2> >(grep -v obsolete >&2); }
ros() { dc exec -T simulator bash -c "source /opt/ros/noetic/setup.bash; $*"; }
clk() { ros 'rostopic echo -n1 /clock' | awk '$1=="secs:"{s=$2} $1=="nsecs:"{n=$2} END{printf "%.3f", s+n/1e9}'; }

fresh() {   # シミュレータを再起動して、Play、画面(gzclient)を止める
  dc restart simulator >/dev/null
  for _ in $(seq 1 90); do
    ros 'rosservice list 2>/dev/null | grep -q /gazebo/unpause_physics && rostopic list 2>/dev/null | grep -q "^/score$"' >/dev/null 2>&1 && break
    sleep 2
  done
  sleep 5
  ros 'pkill -x gzclient' >/dev/null 2>&1
  ros 'rosservice call /gazebo/unpause_physics' >/dev/null 2>&1
  sleep 5
}

set_physics() {   # $1=反復回数(空なら変えない) $2=刻み幅(空なら変えない)
  dc exec -T simulator bash -c "source /opt/ros/noetic/setup.bash; python3 - '$1' '$2'" <<'PY'
import sys
import rospy
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
iters, step = sys.argv[1], sys.argv[2]
rospy.init_node("ablate_physics", anonymous=True)
rospy.wait_for_service("/gazebo/set_physics_properties")
g = rospy.ServiceProxy("/gazebo/get_physics_properties", GetPhysicsProperties)()
ode = g.ode_config
if iters:
    ode.sor_pgs_iters = int(iters)
ts = float(step) if step else g.time_step
r = rospy.ServiceProxy("/gazebo/set_physics_properties", SetPhysicsProperties)(
    ts, g.max_update_rate, g.gravity, ode)
print("physics: time_step=%.4f iters=%d success=%s" % (ts, ode.sor_pgs_iters, r.success))
PY
}

measure() {
  local s0 t0 s1 t1 cpu
  s0=$(clk); t0=$(date +%s.%N)
  sleep "$SEC"
  cpu=$(dc exec -T simulator top -b -n1 -w 120 | awk '/gzserver/{print $9; exit}')
  s1=$(clk); t1=$(date +%s.%N)
  awk -v a="$s0" -v b="$s1" -v c="$t0" -v d="$t1" -v cpu="$cpu" 'BEGIN{printf "RTF=%.3f  (sim %.1fs / wall %.1fs, gzserver CPU %s%%)\n",(b-a)/(d-c),b-a,d-c,cpu}'
}

CONDS=("$@"); [ ${#CONDS[@]} -eq 0 ] && CONDS=(base nomon iters50 step6)
for c in "${CONDS[@]}"; do
  echo "=== $c"
  fresh
  case "$c" in
    base) ;;
    nomon)
      ros 'for n in $(rosnode list | grep -E "_detector$|undesired_contact|wrs_score_counter" | grep -v frcnn_detector); do rosnode kill $n >/dev/null 2>&1; done; sleep 3; echo "監視ノード停止後: $(rosnode list | grep -cE "_detector$|undesired_contact|wrs_score_counter") 個残り"' ;;
    kill=*)   # kill=正規表現: 採点・監視ノードのうち、名前が一致するものだけ止める(例 kill=drawer)
      pat="${c#kill=}"
      ros "for n in \$(rosnode list | grep -E '_detector\$|undesired_contact|wrs_score_counter' | grep -v frcnn_detector | grep -E '$pat'); do rosnode kill \$n >/dev/null 2>&1; done; sleep 3; echo \"止めたあとの採点・監視ノード: \$(rosnode list | grep -cE '_detector\$|undesired_contact|wrs_score_counter') 個\"" ;;
    iters50) set_physics 50 "" ;;
    step6) set_physics "" 0.006 ;;
    iters50+step6) set_physics 50 0.006 ;;
    *) echo "不明な条件: $c"; continue ;;
  esac
  sleep 5
  measure
done
echo "--- 終わりに元の状態へ戻す(再起動)"
fresh
