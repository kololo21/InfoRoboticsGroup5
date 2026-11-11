#!/usr/bin/env bash
# 授業と同じ採点・衝突判定の CPU 用シミュレータイメージ(wrs_sim_lab:cpu)を作る。
#
#   bash tools/local_cpu/lab_sim/build_lab_sim.sh
#
# 研究室のリポジトリは読むだけ(clone して、決まったコミットを取り出すだけ。push などはしない)。
# 取り出す版は、研究室の sim_wrs_ws(product)がサブモジュールとして指しているコミット。
# 作ったあと、~/sim_local/.env に SIM_IMAGE=wrs_sim_lab:cpu を書くと、compose がこのイメージを使う
# (USE=1 を付けて実行すると、この .env も書く)。公開版に戻すときは .env の行を消す。
set -euo pipefail

REF_DIR="${REF_DIR:-$HOME/smilab_ref}"     # 研究室リポジトリの clone 置き場(読むだけ)
TAG="${TAG:-wrs_sim_lab:cpu}"
ORG="https://github.com/keio-smilab25"
# sim_wrs_ws product(a327f14)のサブモジュールの指す版
EVAL_COMMIT="${EVAL_COMMIT:-a4f082c5e1fbaabefa8dd5c7d3bbfe4fe096ac45}"
LAUNCH_COMMIT="${LAUNCH_COMMIT:-e2a2f882e940a1e38d787e48b39cb8100cb8d0ee}"

HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$REF_DIR"

fetch() {  # $1=repo  : clone(なければ)して、push できないようにする
    local r="$1"
    if [ ! -d "$REF_DIR/$r/.git" ]; then
        git clone -q "$ORG/$r" "$REF_DIR/$r"
    else
        git -C "$REF_DIR/$r" fetch -q origin || echo "  (注意: $r を更新できませんでした。手元の clone を使います)"
    fi
    git -C "$REF_DIR/$r" remote set-url --push origin NO_PUSH_read_only
}
fetch tmc_gazebo_task_evaluators
fetch hsrb_wrs_gazebo_launch

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/src/tmc_gazebo_task_evaluators" "$CTX/src/hsrb_wrs_gazebo_launch"
# 作業ツリーには触らず、コミットから直接取り出す
git -C "$REF_DIR/tmc_gazebo_task_evaluators" archive "$EVAL_COMMIT" | tar -x -C "$CTX/src/tmc_gazebo_task_evaluators"
git -C "$REF_DIR/hsrb_wrs_gazebo_launch" archive "$LAUNCH_COMMIT" | tar -x -C "$CTX/src/hsrb_wrs_gazebo_launch"
cp "$HERE/Dockerfile" "$CTX/Dockerfile"
{
    echo "tmc_gazebo_task_evaluators $ORG/tmc_gazebo_task_evaluators $EVAL_COMMIT"
    echo "hsrb_wrs_gazebo_launch $ORG/hsrb_wrs_gazebo_launch $LAUNCH_COMMIT"
} > "$CTX/SOURCES.txt"

echo "イメージ $TAG を作ります(数分かかります)"
docker build -t "$TAG" --label "wrs.sources=$(tr '\n' ';' < "$CTX/SOURCES.txt")" "$CTX"
echo
echo "完了: $TAG"
cat "$CTX/SOURCES.txt"

SIM_DIR="${SIM_DIR:-$HOME/sim_local}"
if [ "${USE:-0}" = 1 ] && [ -d "$SIM_DIR" ]; then
    touch "$SIM_DIR/.env"
    grep -v '^SIM_IMAGE=' "$SIM_DIR/.env" > "$SIM_DIR/.env.tmp" || true
    echo "SIM_IMAGE=$TAG" >> "$SIM_DIR/.env.tmp"
    mv "$SIM_DIR/.env.tmp" "$SIM_DIR/.env"
    echo "$SIM_DIR/.env に SIM_IMAGE=$TAG を書きました。次の docker compose up -d から、このイメージを使います。"
else
    echo "使うには: echo SIM_IMAGE=$TAG >> $SIM_DIR/.env  (そのあと docker compose up -d)"
fi
