#!/usr/bin/env bash
# 失敗原因の集計: run_experiment.sh が作った results/<日時_ラベル>/ のログを読んで、
# どこで点を落としているかを Seed 別に一覧にする。
#
# 使い方:
#   ./analyze_failures.sh results/20261109_221849_selftest
#   ./analyze_failures.sh                    # 最新の results/ を使う
#
# 分類ルールは下の PATTERNS(ラベル<TAB>正規表現)。自分のコードのログに合わせて足してよい。
# どれにも当てはまらない ERROR/WARN は「未分類」として上位を出すので、新しい失敗の型が見つかる。
set -u

DIR="${1:-}"
if [ -z "$DIR" ]; then
  DIR=$(ls -dt "$HOME"/sim_ws/results/*/ ./results/*/ 2>/dev/null | head -1)
fi
[ -d "$DIR" ] || { echo "results フォルダが見つかりません: $DIR"; exit 1; }
DIR="${DIR%/}"

PATTERNS=$(cat <<'EOF'
衝突	Collision Detected
対象物が見つからない	Cannot find object|no graspable|not found
タスク中断	aborted|abort
把持/軌道計画の失敗	grasp.*(fail|error)|(plan|planning).*(fail|error)|No (IK|solution)|inverse kinematics
タイムアウト	timed out|timeout|Timeout
ノード異常終了	has died \[pid [0-9]+, exit code [1-9]
コンテナ停止(環境の問題)	is not running|No such container
Python例外	Traceback|Exception
配置の失敗(Task1)	Task 1 placement failed|Placement result:.*"success": false
Task1失敗でTask2に進めず	Task 1 failed; stop before Task 2
EOF
)

OUTCSV="$DIR/failures.csv"
echo "seed,run,category,count" > "$OUTCSV"

# 全実行の一覧(summary.csv の順)
RUNS=$(find "$DIR" -maxdepth 1 -type d -name 'seed*_run*' | sort -V)
[ -n "$RUNS" ] || { echo "seed*_run* が見つかりません: $DIR"; exit 1; }

echo "===== 失敗原因の集計: $DIR ====="
printf "%-12s %7s %7s  %s\n" "run" "score" "秒" "検出された失敗(回数)"

for rd in $RUNS; do
  name=$(basename "$rd")
  seed=${name#seed}; seed=${seed%%_*}
  run=${name##*_run}
  log="$rd/run.log"
  [ -f "$log" ] || continue

  final=$(awk -F, 'NR>1 && $2!=""{v=$2} END{print (v==""?0:v)}' "$rd/score_timeline.csv" 2>/dev/null)
  wall=$(awk -F, -v s="$seed" -v r="$run" '$1==s && $2==r {print $8}' "$DIR/summary.csv")
  line=""
  while IFS=$'\t' read -r label re; do
    [ -z "$label" ] && continue
    c=$(grep -ciE "$re" "$log")
    if [ "$c" -gt 0 ]; then
      line="$line [$label:$c]"
      echo "$seed,$run,$label,$c" >> "$OUTCSV"
    fi
  done <<< "$PATTERNS"
  # スコアが全く入らなかった
  awk -v f="$final" 'BEGIN{exit !(f+0==0)}' && { line="$line [得点0]"; echo "$seed,$run,得点0,1" >> "$OUTCSV"; }
  printf "%-12s %7s %7s %s\n" "S${seed}#${run}" "${final:-?}" "${wall:-?}" "${line:- なし}"
done

echo
echo "----- Seed別: 失敗カテゴリの合計(実行あたり平均) -----"
awk -F, 'NR>1{k=$1; c[k","$3]+=$4; if(!(k","$2 in seen)){seen[k","$2]=1; n[k]++}; cat[$3]=1; sd[k]=1}
  END{for(s in sd){printf "Seed %s (%d回):\n",s,n[s]; for(g in cat){v=c[s","g]; if(v>0) printf "   %-16s 合計%4d  平均%6.1f\n",g,v,v/n[s]}}}' "$OUTCSV"

echo
echo "----- 衝突した相手(多い順 上位8) -----"
cat "$DIR"/seed*_run*/run.log 2>/dev/null \
  | grep -oE "Collision Detected with [A-Za-z0-9_]+" | sed 's/Collision Detected with //' \
  | sort | uniq -c | sort -rn | head -8 | awk '{printf "   %-28s %d回\n",$2,$1}'

echo
echo "----- 配置に失敗した段階(Placement result の stage。多い順) -----"
cat "$DIR"/seed*_run*/run.log 2>/dev/null \
  | grep -E 'Placement result:.*"success": false' | grep -oE '"stage": "[a-z_]+"' \
  | sed 's/"stage": //; s/"//g' | sort | uniq -c | sort -rn | awk '{printf "   %-14s %d回\n",$2,$1}'

echo
echo "----- 失敗した物体(Task 1 placement failed for ○○。多い順 上位8) -----"
cat "$DIR"/seed*_run*/run.log 2>/dev/null \
  | grep -oE 'placement failed for [A-Za-z0-9_]+' | sed 's/placement failed for //' \
  | sort | uniq -c | sort -rn | head -8 | awk '{printf "   %-24s %d回\n",$2,$1}'

echo
echo "----- 未分類の ERROR/WARN(正規化して多い順 上位8) -----"
KNOWN=$(echo "$PATTERNS" | cut -f2 | paste -sd'|')
cat "$DIR"/seed*_run*/run.log 2>/dev/null \
  | grep -E "\[ ?(ERROR|WARN|FATAL)" | grep -viE "$KNOWN" \
  | sed -E 's/\[[0-9. ]+\]//g; s/[0-9]+(\.[0-9]+)?/N/g' \
  | sort | uniq -c | sort -rn | head -8 | cut -c1-140

echo
echo "----- スコアの止まり(シミュレーション時刻。得点が最後に増えた時刻と記録終了時刻) -----"
for rd in $RUNS; do
  [ -f "$rd/score_timeline.csv" ] || continue
  awk -F, -v n="$(basename "$rd")" 'NR>1 && $2!=""{ if($2>p){last=$1; p=$2}; end=$1 }
    END{ if(end>0) printf "   %-12s 最終得点%5d  最後に増えた時刻 %5.0fs / 記録終了 %5.0fs\n", n, p+0, last/1e9, end/1e9 }' "$rd/score_timeline.csv"
done
echo
echo "CSV: $OUTCSV"
