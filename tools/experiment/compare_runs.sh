#!/usr/bin/env bash
# 複数Seedの評価結果をラベル(コードの版)ごとに並べて比較する。
#
# 使い方:
#   ./compare_runs.sh                       # ~/sim_ws/results/ の全結果を比較
#   ./compare_runs.sh results/A results/B   # 指定した結果だけ比較
#
# 見方:
#   mean   : Seedごとの平均点。 合計 = 各Seed平均の和(「Seedの和」で評価される有用性に対応)
#   robust : 各Seedの (平均 - ばらつき) の和。運に左右されない度合い。高いほど安定して得点できる
#   worst  : 各Seedの最低点の和。最悪ケースでも取れる点
#   レベル : Lv1 = Seed86で120点以上 / Lv2 = Seed25とSeed86の合計220点以上 (第1週資料の目安)
# 注意: 繰り返し回数nが少ない(1〜2回)と、平均の差はほぼ誤差。n>=3で比べること。
set -u

if [ $# -gt 0 ]; then DIRS="$*"; else DIRS=$(ls -d "$HOME"/sim_ws/results/*/ ./results/*/ 2>/dev/null); fi
[ -n "$DIRS" ] || { echo "結果フォルダがありません"; exit 1; }

for d in $DIRS; do
  d=${d%/}
  [ -f "$d/summary.csv" ] || continue
  label=$(basename "$d")
  awk -F, -v L="$label" '
    NR>1 && $3!="" { n[$1]++; s[$1]+=$3; q[$1]+=$3*$3; if(!($1 in mn)||$3<mn[$1]) mn[$1]=$3; seeds[$1]=1 }
    END {
      tm=0; rb=0; wo=0; ns=0; minn=1e9; cnt=0; s86=-1
      for (k in seeds) {
        m=s[k]/n[k]; v=q[k]/n[k]-m*m; sd=(v>0)?sqrt(v):0
        tm+=m; rb+=m-sd; wo+=mn[k]; ns++; if(n[k]<minn) minn=n[k]
        if(k=="86") s86=m
        det=det sprintf(" S%s:%.0f±%.0f", k, m, sd)
      }
      lv="-"
      if (s86>=120) lv="Lv1"
      if (ns>=2 && tm>=220) lv="Lv2"
      printf "%.1f\t%s\t%d\t%.1f\t%.1f\t%.1f\t%s\t%s\n", tm, L, minn, tm, rb, wo, lv, det
    }' "$d/summary.csv"
done | sort -rn | cut -f2- | {
  printf "%-34s %2s %8s %8s %8s %4s  %s\n" "結果(ラベル)" "n" "合計" "robust" "worst" "Lv" "Seed別 平均±ばらつき"
  while IFS=$'\t' read -r L n tm rb wo lv det; do
    printf "%-34s %2s %8s %8s %8s %4s %s\n" "$L" "$n" "$tm" "$rb" "$wo" "$lv" "$det"
  done
}
