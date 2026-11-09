#!/usr/bin/env bash
# クラウドの VM(Ubuntu 22.04 / 24.04、GPU なしでよい)に、Docker を入れて setup_local.sh まで進める。
#
#   bash setup_vm.sh [リポジトリURL] [ブランチ]
#
# 自分の VM(使い捨ての実験用)で実行すること。次のことをする:
#   1. Docker の公式 apt リポジトリを登録して Docker Engine と compose プラグインを入れる(sudo を使う)
#   2. 自分を docker グループに入れる(sudo usermod -aG docker)
#   3. setup_local.sh を実行する(この時点で docker グループが反映されていなければ、`sg docker` 経由で実行)
# 未検証: このスクリプトは手元の WSL では動かしていない(Docker はすでに入っているため)。
#         VM で初めて動かすときは、各手順の表示を見ながら進めること。
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"

if ! command -v docker >/dev/null; then
  echo "== Docker を入れます(sudo)"
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl git
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

if ! id -nG | grep -qw docker; then
  echo "== $USER を docker グループに入れます(sudo)"
  sudo usermod -aG docker "$USER"
fi

echo "== CPU とメモリ(目安: 8 コア・16GB 以上)"
nproc
free -g | sed -n 2p

if docker info >/dev/null 2>&1; then
  bash "$HERE/setup_local.sh" "$@"
else
  # グループへの追加は、ログインし直すまで反映されない。sg で docker グループとして続ける
  sg docker -c "bash '$HERE/setup_local.sh' $*"
fi
