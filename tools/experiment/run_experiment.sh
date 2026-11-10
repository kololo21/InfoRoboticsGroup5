#!/usr/bin/env bash
# ワンコマンド実験ループ: Seed切替 -> simulator再起動 -> Play -> 実行 -> スコア回収 -> 集計
# GPUサーバ(ホスト側ターミナル)で実行する。WebIDE内ではない。
#
# 使い方:
#   ./run_experiment.sh                      # Seed 86 と 25 を1回ずつ
#   ./run_experiment.sh -s "86 25" -n 3      # 各Seed 3回(平均・ばらつきを見る)
#   ./run_experiment.sh -l tray-fix          # 実験ラベル(コードの版の識別用)
#   ./run_experiment.sh -g                   # 画面(Gazebo GUI)を残す。動画を撮るとき
#   (既定は画面なし=headless。手元の測定でRTF 0.22 -> 0.36 に向上)
#   ./run_experiment.sh -v c -D yolo         # wrs_main_node_c.py を YOLO で(-v は a|b|c|main)
#   ./run_experiment.sh -G haruki            # GitHub の haruki ブランチの wrs_main_node_c.py を取得して実行
#   ./run_experiment.sh -G haruki:src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/wrs_main_node_b.py
#   ./run_experiment.sh -f ~/my_node.py      # 任意のファイル(例 C:\Users\...\wrs_main_node_c.py)を実行
#   ./run_experiment.sh -K undesired         # 高速な動作確認: 接触判定ノードを止める(RTF約2倍。Task 2a の採点は無効)
#   ./run_experiment.sh -Q -v c              # 短縮確認: 1物体だけ 探す→つかむ→置く(_c.py は変更しない)。-q で引き出しも開ける
#                                            # 結果は node.log の "QUICK RESULT: {...}" と、ダッシュボードに出る
#   (-G / -f は wrs_main_node_local.py に置いて start_all.sh local で動かす。元のファイルは変えない)
#
# 前提: ~/sim_ws で docker compose up -d 済み。docker compose restart 以外は破壊的操作をしない。
set -u

SEEDS="86 25"        # 評価するSeed
REPEAT=1             # Seedごとの繰り返し回数
TIMEOUT=900          # 1回の実行の上限(実時間・秒)
DIR="$HOME/sim_ws"   # docker-compose.yml のある場所
LABEL="run"          # 結果フォルダ名に付ける
RESTORE_SEED=86      # 終了後に戻すSeed(空なら戻さない)
VARIANT=""           # 実行するノード: 空=統合版(wrs_main_node.py) / a / b / c
DETECTOR_OPT=""      # 検出器: 空=start_all.sh の既定 / yolo / frcnn

usage() { sed -n '2,28p' "$0"; exit 1; }
INTERLEAVE=0
HEADLESS=1           # 1: 描画クライアント(gzclient)を止めて高速化。画面や動画が必要なら -g
SRC_FILE=""          # -f: 任意のファイルから実行
SRC_GITHUB=""        # -G: GitHub の ref[:path] から実行
DEFAULT_NODE_PATH="src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/wrs_main_node_c.py"
MON_KILL=""          # -K: 止める採点・監視ノードの名前(正規表現)。高速な動作確認用。点数は一部付かなくなる
QUICK=0              # -Q: 短縮確認(1物体だけ 探す→つかむ→置く)。-q を足すと、先に引き出しも開ける
QUICK_DRAWERS=0
while getopts "s:n:t:d:l:r:v:D:f:G:K:Qqigh" o; do
  case "$o" in
    Q) QUICK=1 ;;
    q) QUICK=1; QUICK_DRAWERS=1 ;;
    K) MON_KILL="$OPTARG" ;;
    f) SRC_FILE="$OPTARG" ;;
    G) SRC_GITHUB="$OPTARG" ;;
    i) INTERLEAVE=1 ;;
    g) HEADLESS=0 ;;
    v) VARIANT="$OPTARG" ;;
    D) DETECTOR_OPT="$OPTARG" ;;
    s) SEEDS="$OPTARG" ;; n) REPEAT="$OPTARG" ;; t) TIMEOUT="$OPTARG" ;;
    d) DIR="$OPTARG" ;;   l) LABEL="$OPTARG" ;;  r) RESTORE_SEED="$OPTARG" ;;
    *) usage ;;
  esac
done

if [ -n "$MON_KILL" ] && ! printf '%s' "$MON_KILL" | grep -qE '^[A-Za-z0-9_|.-]{1,80}$'; then
  echo "ERROR: -K は英数字と _ | . - だけ(例: undesired または undesired|drawer)"; exit 1
fi
[ -n "$SRC_FILE" ] && [ -n "$SRC_GITHUB" ] && { echo "ERROR: -f と -G は同時に使えません"; exit 1; }
[ -n "$SRC_FILE$SRC_GITHUB" ] && VARIANT=local   # -f / -G は wrs_main_node_local.py として実行する
case "$VARIANT" in ""|main) VARIANT="" ;; a|b|c|local) ;; *) echo "ERROR: -v は a|b|c|main"; exit 1 ;; esac
case "$DETECTOR_OPT" in ""|yolo|frcnn) ;; *) echo "ERROR: -D は yolo|frcnn"; exit 1 ;; esac
# 短縮確認(-Q): 別ファイルの wrs_main_node_quick.py が、選んだノード(既定 c)のクラスを読み込んで、
# 1物体だけ「探す→つかむ→置く」を本番と同じ関数で実行する。選んだノードのファイルは変更しない
if [ "$QUICK" = 1 ]; then
  [ -z "$VARIANT" ] && VARIANT=c   # 統合版 wrs_main_node.py は、短縮確認に必要な関数を持たない
  if [ "$VARIANT" = a ] || [ "$VARIANT" = b ]; then
    echo "ERROR: 短縮確認は c 系のノード(-v c、または -f / -G で渡した c 系のファイル)が対象です"; exit 1
  fi
fi
# 実行コマンド(START_CMD を環境変数で渡せば上書きできる。動作テスト用)
if [ -z "${START_CMD:-}" ]; then
  if [ "$QUICK" = 1 ]; then
    START_CMD="env QUICK_BASE_NODE=wrs_main_node_${VARIANT}.py QUICK_DRAWERS=${QUICK_DRAWERS} ${DETECTOR_OPT:+DETECTOR=$DETECTOR_OPT }/workspace/src/wrs_algorithm21/scripts/start_all.sh quick"
  else
    START_CMD="${DETECTOR_OPT:+env DETECTOR=$DETECTOR_OPT }/workspace/src/wrs_algorithm21/scripts/start_all.sh${VARIANT:+ $VARIANT}"
  fi
fi
CODE_FILE="${CODE_FILE:-workspace/src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/wrs_main_node${VARIANT:+_$VARIANT}.py}"

cd "$DIR" || { echo "ERROR: $DIR に移動できません"; exit 1; }

# 同じシミュレータで実験を二重に走らせない(再起動し合って、どちらの結果も壊れるため)
LOCK="$DIR/.run_experiment.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ERROR: すでに実験が動いています(PID $(cat "$LOCK"))。終わるのを待つか、止めてから実行してください"
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$DIR/results/${STAMP}_${LABEL}"
mkdir -p "$OUT"
echo "OUT=$OUT"           # ダッシュボードが結果フォルダを見つけるために使う
SUMMARY="$OUT/summary.csv"
echo "seed,run,score_final,score_max,collisions,not_found,exit_code,wall_sec,logdir" > "$SUMMARY"

dc() { docker compose "$@" 2> >(grep -v obsolete >&2); }
# simulator コンテナ内で ROS のコマンドを実行する。ROS_TO 秒(既定 90)で打ち切る
# (終了時のスナップショットの rostopic が戻らず、実行ループが何分も止まったことがあったため)
ros_sim() { dc exec -T simulator timeout -k 3 "${ROS_TO:-90}" bash -c "source /opt/ros/noetic/setup.bash; $*"; }

# 事前確認: 3コンテナが動いていなければ起動(--build は使わない。コンテナは作り直さない)
for svc in xserver simulator workspace; do
  if [ -z "$(dc ps --status running -q "$svc" 2>/dev/null)" ]; then
    echo "($svc が停止していたので docker compose up -d で起動します)"
    dc up -d >/dev/null 2>&1; sleep 5; break
  fi
done

GITDIR="$DIR/workspace"   # リポジトリのルート(src/ と tools/ がある場所)

# -f / -G: 実行するノードを GitHub や任意のファイルから取得して wrs_main_node_local.py に置く
SOURCE="workspace"
if [ "$VARIANT" = local ]; then
  LOCAL_NODE="$GITDIR/src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/wrs_main_node_local.py"
  if [ -n "$SRC_GITHUB" ]; then
    REF="${SRC_GITHUB%%:*}"; NPATH="$DEFAULT_NODE_PATH"
    case "$SRC_GITHUB" in *:*) NPATH="${SRC_GITHUB#*:}" ;; esac
    echo "GitHub から取得: $REF : $NPATH"
    git -C "$GITDIR" fetch --quiet origin "$REF" || { echo "ERROR: GitHub から $REF を取得できません(権限・ブランチ名を確認)"; exit 1; }
    git -C "$GITDIR" show "FETCH_HEAD:$NPATH" > "$LOCAL_NODE" || { echo "ERROR: $REF に $NPATH がありません"; exit 1; }
    SOURCE="github:$REF:$NPATH@$(git -C "$GITDIR" rev-parse --short FETCH_HEAD)"
  else
    F="$SRC_FILE"
    case "$F" in [A-Za-z]:*) command -v wslpath >/dev/null && F=$(wslpath -u "$F") ;; esac
    [ -f "$F" ] || { echo "ERROR: ファイルがありません: $F"; exit 1; }
    cp "$F" "$LOCAL_NODE" || exit 1
    SOURCE="file:$SRC_FILE"
  fi
  chmod +x "$LOCAL_NODE"
  # 文法エラーのファイルで長い実験を始めないよう、先に構文だけ確認する
  if command -v python3 >/dev/null && ! python3 -c "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())" "$LOCAL_NODE" 2>/tmp/ast_err.$$; then
    echo "ERROR: Python の構文エラーです:"; tail -3 /tmp/ast_err.$$; rm -f /tmp/ast_err.$$; exit 1
  fi
  rm -f /tmp/ast_err.$$
  CODE_FILE="$LOCAL_NODE"
fi

# コードの版を記録(あとで「どの版が何点だったか」を追えるように)
[ -f "$CODE_FILE" ] && cp "$CODE_FILE" "$OUT/code_snapshot.py"
{
  echo "label=$LABEL"
  echo "variant=${VARIANT:-main}"
  echo "source=$SOURCE"
  echo "monitors_stopped=${MON_KILL:-none}"
  echo "quick=$QUICK"
  echo "quick_drawers=$QUICK_DRAWERS"
  echo "node_file=$(basename "$CODE_FILE")"
  echo "code_path=$CODE_FILE"
  [ -f "$OUT/code_snapshot.py" ] && echo "code_sha=$(sha256sum "$OUT/code_snapshot.py" | cut -c1-10)"
  echo "detector=${DETECTOR_OPT:-default}"
  echo "seeds=$SEEDS"
  echo "repeat=$REPEAT"
  echo "headless=$HEADLESS"
  echo "interleave=$INTERLEAVE"
  echo "started=$(date +%Y-%m-%dT%H:%M:%S)"
  if git -C "$GITDIR" rev-parse --git-dir >/dev/null 2>&1; then
    echo "git_branch=$(git -C "$GITDIR" rev-parse --abbrev-ref HEAD)"
    echo "git_commit=$(git -C "$GITDIR" rev-parse --short HEAD)"
    echo "git_dirty=$(git -C "$GITDIR" status --porcelain 2>/dev/null | grep -c .)"
  fi
} > "$OUT/meta.txt"

wait_ready() {  # simulator の Gazebo と /score が立ち上がるまで待つ
  for _ in $(seq 1 90); do
    if ros_sim 'rosservice list 2>/dev/null | grep -q /gazebo/unpause_physics && rostopic list 2>/dev/null | grep -q "^/score$"' >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

wait_moveit() {  # Play のあとに、MoveIt(move_group の execute_trajectory)が立ち上がるまで待つ
  # 待たないと、ノードが先に起動して "Unable to connect to move_group action server
  # 'execute_trajectory' within allotted time (5s)" で落ちる(CPU 環境では move_group の起動が遅い)。
  # execute_trajectory は、コントローラが起動してから(=シミュレーションが進んでから)できるので、Play の前には待てない
  for _ in $(seq 1 90); do
    if ros_sim 'rosnode list 2>/dev/null | grep -q "^/move_group$" && rostopic list 2>/dev/null | grep -q "^/execute_trajectory/status$"' >/dev/null 2>&1; then
      sleep 3   # move_group が計画グループを読み込み終わるまで、少し待つ
      return 0
    fi
    sleep 2
  done
  echo "  (警告: MoveIt の準備を確認できないまま続けます)"
  return 1
}

set_seed() {
  if [ -x ./change_seed.sh ]; then
    ./change_seed.sh "$1" >/dev/null
  else
    echo "  (change_seed.sh が無いのでSeed切替をスキップ)"
  fi
}

one_run() {  # $1=seed $2=run番号
  local seed=$1 n=$2 rd="$OUT/seed${1}_run${2}"
  mkdir -p "$rd"
  echo "== Seed $seed  run $n"
  set_seed "$seed"
  dc restart simulator >/dev/null 2>&1
  if ! wait_ready; then
    echo "  ERROR: simulator が起動しませんでした(TAに相談)"
    echo "$seed,$n,,,,,,,$rd" >> "$SUMMARY"; return
  fi
  # 画面(gzclient)を止める。物理演算とセンサ(gzserver)・/score は止まらない。再起動で元に戻る
  [ "$HEADLESS" = 1 ] && ros_sim 'pkill -x gzclient' >/dev/null 2>&1
  # 高速な動作確認: 重い採点・監視ノードを止める(例 -K undesired でRTFが約2倍。止めたノードの分は点が付かない)
  if [ -n "$MON_KILL" ]; then
    ros_sim "for n in \$(rosnode list | grep -E '_detector\$|undesired_contact|wrs_score_counter' | grep -v frcnn_detector | grep -E '$MON_KILL'); do rosnode kill \$n >/dev/null 2>&1; done" >/dev/null 2>&1
  fi
  ros_sim 'rosservice call /gazebo/unpause_physics' >/dev/null 2>&1   # Playボタン相当
  wait_moveit

  # スコアの時系列を記録
  ROS_TO=$((TIMEOUT + 600)) ros_sim 'rostopic echo -p /score' > "$rd/score_timeline.csv" 2>/dev/null &
  local spid=$!
  local t0; t0=$(date +%s)

  # wrs_main が終わったら(正常終了でも、起動時のエラーで return した場合でも)、残っている検出器などの
  # roslaunch を止めて、この回を終える。止めないと、start_all.sh は上限時間まで待ち続ける
  # (以前、起動直後に終了した実行が、1500秒まるごと待った)
  rm -f "$rd/.node_ended"
  (
    seen=0; waited=0
    while [ "$waited" -lt "$TIMEOUT" ]; do
      sleep 3; waited=$((waited + 3))
      if ros_sim 'rosnode list 2>/dev/null | grep -q "^/wrs_main$"' >/dev/null 2>&1; then
        seen=1
      elif [ "$seen" = 1 ] || [ "$waited" -ge 240 ] \
           || ros_sim 'L=$(ls -t /root/.ros/log/*/rosout.log 2>/dev/null | head -1); grep -q " /wrs_main " "$L"' >/dev/null 2>&1; then
        # 出現後に消えた / 240秒たっても出てこない / 一瞬で終わって見逃したが、ログには残っている
        sleep 3
        if ! ros_sim 'rosnode list 2>/dev/null | grep -q "^/wrs_main$"' >/dev/null 2>&1; then
          echo "seen=$seen" > "$rd/.node_ended"
          dc exec -T workspace bash -c 'pkill -f start_all.sh; pkill -f roslaunch' >/dev/null 2>&1
          break
        fi
      fi
    done
  ) &
  local wpid=$!
  dc exec -T workspace bash -lc "timeout $TIMEOUT $START_CMD" > "$rd/run.log" 2>&1
  local code=$?
  kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
  local wall=$(( $(date +%s) - t0 ))
  if [ -f "$rd/.node_ended" ]; then
    echo "  wrs_main が終了したので、この回を終えました(${wall}秒)。ログの node.log を見てください"
  fi
  sleep 2; kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null

  # ノードの ROS ログ(rosout)を保存する。run.log は出力のバッファのため、実行が終わるまで見えないことがあり、
  # 検出器のエラーなど別ノードのログも入らない。rosout にはリアルタイムで全ノードのログが残る
  ros_sim 'L=$(ls -t /root/.ros/log/*/rosout.log 2>/dev/null | head -1); grep -E " /(wrs_main|frcnn_detector) " "$L" | sed -E "s/\[topics: [^]]*\] ?//"' \
    > "$rd/node.log" 2>/dev/null

  # 終了時の物体の位置と、置き場所ごとの判定器の数を保存する。「物体を離したのに0点」のとき、
  # 実際にどこにあるか(置き場所に入っているか)と、採点側の数え方を後から突き合わせるため
  ROS_TO=20 ros_sim 'timeout 10 rostopic echo -n1 /gazebo/model_states' > "$rd/model_states.txt" 2>/dev/null
  ROS_TO=25 ros_sim 'for t in $(rostopic list | grep -E "_detector/count$"); do (echo "$t $(timeout 4 rostopic echo -n1 $t 2>/dev/null | head -1)") & done; wait' \
    > "$rd/counters.txt" 2>/dev/null

  local final max col nf src
  src="$rd/run.log"; [ -s "$rd/node.log" ] && src="$rd/node.log"
  final=$(awk -F, 'NR>1 && $2!=""{v=$2} END{print (v==""?0:v)}' "$rd/score_timeline.csv")
  max=$(awk -F, 'NR>1 && $2!=""&&$2>m{m=$2} END{print m+0}' "$rd/score_timeline.csv")
  col=$(grep -c "Collision Detected" "$src")
  nf=$(grep -ciE "Cannot find object|aborted" "$src")
  echo "  score=$final (max $max)  collisions=$col  not_found=$nf  exit=$code  ${wall}s"
  echo "$seed,$n,$final,$max,$col,$nf,$code,$wall,$rd" >> "$SUMMARY"
}

if [ "$INTERLEAVE" = 1 ]; then   # 86,25,86,25,... の順(時間経過による偏りを避ける)
  for n in $(seq 1 "$REPEAT"); do
    for seed in $SEEDS; do one_run "$seed" "$n"; done
  done
else                              # 86,86,86,25,25,25 の順
  for seed in $SEEDS; do
    for n in $(seq 1 "$REPEAT"); do one_run "$seed" "$n"; done
  done
fi

[ -n "$RESTORE_SEED" ] && { set_seed "$RESTORE_SEED"; dc restart simulator >/dev/null 2>&1; }

echo; echo "===== 集計 ($OUT) ====="
awk -F, 'NR>1 && $3!=""{n[$1]++; s[$1]+=$3; q[$1]+=$3*$3; c[$1]+=$5; if(!($1 in mn)||$3<mn[$1])mn[$1]=$3; if($3>mx[$1])mx[$1]=$3}
  END{printf "%-6s %3s %8s %8s %6s %6s %10s\n","seed","n","mean","stdev","min","max","collisions";
      for(k in n){m=s[k]/n[k]; sd=sqrt(q[k]/n[k]-m*m); tot+=m;
      printf "%-6s %3d %8.1f %8.1f %6.0f %6.0f %10.1f\n",k,n[k],m,sd,mn[k],mx[k],c[k]/n[k]}
      printf "合計(各Seed平均の和): %.1f\n",tot}' "$SUMMARY"
echo "詳細: $SUMMARY"

# 失敗原因の集計(analyze_failures.sh が同じ場所にあれば実行)
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -x "$HERE/analyze_failures.sh" ] && { echo; "$HERE/analyze_failures.sh" "$OUT"; }
