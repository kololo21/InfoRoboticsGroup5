# 実験ループとダッシュボード

GPU サーバでの「Seed を切り替える → シミュレータを再起動 → Play → 実行 → スコアを記録 → 集計」を
1つのコマンド（または1つの Web 画面）にまとめたツール。コードの書き換えはしない。
評価対象のコードは授業の `start_all.sh` で動かすだけなので、どの班のコードでも使える。

| ファイル | 役割 |
|---|---|
| `run_experiment.sh` | 実験ループ本体。Seed 切替 → 再起動 → Play → 実行 → `/score` 記録 → 集計 |
| `analyze_failures.sh` | ログから失敗の原因を分類して数える（実験ループの最後に自動で動く） |
| `compare_runs.sh` | 複数の結果を「版ごと」に並べて比較する |
| `dashboard.py` / `dashboard.html` | 上の3つと docker の起動・停止を、ブラウザの1画面で扱う |

## 前提

- GPU サーバの**ホスト側**ターミナルで使う（WebIDE の中ではない）
- `~/sim_ws` に `docker-compose.yml` があり、`docker compose up -d` でシミュレータが動く（授業の標準構成）
- 実験で動かすコードは `/workspace/src/wrs_algorithm21/scripts/start_all.sh`（授業の標準）
- `python3`（Ubuntu 標準）。pip で入れるものはない

## 使い方（ダッシュボード）

GPU サーバ上でリポジトリを最新にして（`cd ~/sim_ws/workspace && git pull`）、次を実行する。
このフォルダの5ファイルだけを任意の場所にコピーして使ってもよい（5つは同じフォルダに置くこと）。

```bash
python3 ~/sim_ws/workspace/tools/experiment/dashboard.py --lan
```

表示された `http://<サーバ名>.local:8765/?token=...` を、ノート PC のブラウザで開く。

| 区画 | できること |
|---|---|
| 環境 | コンテナの状態・現在の Seed・現在のスコア。起動 / 停止 / シミュレータ再起動 / Seed 切替 |
| 実験を実行 | Seed・回数・版・検出器・ラベルを選んで実行。進捗、ライブのスコア、ログ。中止ボタン |
| 結果 | 版ごとの表。行を押すと、スコアの推移グラフ、失敗の内訳、各回のログ。README の「実行記録」表に貼れる行をコピー |

- `--lan` を付けないと、そのサーバのブラウザからしか開けない
- **token を知らない人は操作できない**（実験室の Wi-Fi は他の班も使うため）。URL を他の班に見せない
- `--dir` で `docker-compose.yml` のある場所を変えられる（既定 `~/sim_ws`）。止めるときは Ctrl-C

## 使い方（コマンドだけ）

```bash
tools/experiment/run_experiment.sh -s "86 25" -n 3 -i -l my-change
tools/experiment/compare_runs.sh          # ~/sim_ws/results/ の全結果を比較
```

| オプション | 意味 |
|---|---|
| `-s "86 25"` | 評価する Seed（既定は 86 と 25） |
| `-n 3` | Seed ごとの繰り返し回数。**変更の良し悪しは 3 回以上の平均で見る**（1〜2 回の差はほぼ誤差） |
| `-i` | 86,25,86,25… の順に実行する（時間経過による偏りを避ける） |
| `-l 名前` | ラベル。結果フォルダの名前になる |
| `-t 秒` | 1 回の上限時間（既定 900 秒） |
| `-g` | 画面（Gazebo の GUI）を残す。動画を撮るときに付ける。付けないと画面を止めて速くなる |
| `-v a\|b\|c\|main` | `start_all.sh` に渡す引数（`start_all.sh c` など）。引数を取らない `start_all.sh` なら使わない |
| `-D yolo\|frcnn` | 環境変数 `DETECTOR` を渡す。`DETECTOR` を使わない `start_all.sh` なら使わない |
| `-r 86` | 終了後に戻す Seed（空にすると戻さない） |
| `-d ディレクトリ` | `docker-compose.yml` のある場所（既定 `~/sim_ws`） |

別のコマンドで動かしたいときは、環境変数 `START_CMD` で上書きできる（WebIDE 内で実行される）。

## 結果の読み方

結果は `~/sim_ws/results/日時_ラベル/` に残る。`summary.csv`（全実行の一覧）、各回の `run.log` と
`score_timeline.csv`（スコアの時系列）、`code_snapshot.py`（そのときのコード）、
`meta.txt`（版・検出器・git のブランチとコミット）、`failures.csv`（失敗の分類）。

| 指標 | 意味 |
|---|---|
| 合計 | 各 Seed の平均点の和（「Seed の和」で評価される点に対応） |
| robust | 各 Seed の（平均 − ばらつき）の和。運に左右されず取れる点。高いほど安定 |
| worst | 各 Seed の最低点の和。最悪でも取れる点 |
| Lv | Lv1 = Seed 86 で 120 点以上 / Lv2 = Seed 25 と 86 の合計 220 点以上 |

1 回だけ高得点が出て、他が 0 点という版は、`合計` が高くても `robust` が低くなる。

## 失敗の分類を自分のログに合わせる

`analyze_failures.sh` の先頭近くにある `PATTERNS`（ラベル、タブ、正規表現）を編集する。
どれにも当てはまらない ERROR / WARN は「未分類」として多い順に表示されるので、
そこに出てきた新しい失敗の型を `PATTERNS` に足していく。

## 安全のために

- 使う docker コマンドは `compose up -d` / `stop` / `restart` / `exec` / `ps` だけ。
  手順書で禁止されている `--build` や `down` は使わない
- 実験のたびに `docker compose restart simulator` を行う。コードやコンテナは作り直さない
- ダッシュボードが実行するコマンドは決まったものだけで、Seed・ラベルなどの入力値は検証してから渡す

## 検証の状況（正直に）

- 手元の PC の公開版シミュレータ（`tmc_wrs_docker`、CPU のみ）で、実験ループ・失敗集計・比較・ダッシュボードの
  通しの動作を確認した。そのとき `start_all.sh` と `change_seed.sh` は無かったので、代わりに `sleep` を実行した
- **授業のサーバでの実測はまだ**。点数が `/score` に記録されることと、`change_seed.sh` での Seed 切替は、
  最初の 1 回を実際に回して確かめること
- 画面を止めると速くなる（手元で RTF 0.22 → 0.36）。検出ノードを動かしたときの効果は未測定
- 失敗の分類のうち「配置の失敗」「Task 1 で失敗して Task 2 に進めない」は、第 5 班のログの形式に合わせてある。
  他の班のログでは `PATTERNS` の調整が要る
