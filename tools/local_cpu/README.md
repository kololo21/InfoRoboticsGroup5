# 自分の PC(GPU なし・CPU だけ)でシミュレータを動かす

授業の GPU サーバと同じ構成(`simulator` / `workspace` / `xserver`、ポート 3000・3001)を、
GPU も NVIDIA ドライバもなしで自分の PC に作る。競技場・物体は授業と同じ公開イメージ
(`ghcr.io/hsr-project/tmc_wrs_binary`)を使う。コードの動作確認、実験ループ、ダッシュボードの練習に向く。

**採点と衝突判定は、公開イメージのままだと授業と違う。** 下の「授業と同じ採点にする」で、研究室版のイメージを作って使うこと。

## 授業と同じ採点にする(研究室版イメージ。推奨)

授業のシミュレータは、研究室のリポジトリ `keio-smilab25/sim_wrs_ws` の Dockerfile で作られ、採点ノード
(`tmc_gazebo_task_evaluators`)と起動ファイル(`hsrb_wrs_gazebo_launch`)を研究室が変更した版に差し替えている。

| | 公開イメージ | 研究室版(授業) |
|---|---|---|
| 衝突判定 | 働かない(`fast_physics` の `--minimal-comms` で接触情報が流れない) | 働く(前後のバンパーの接触を見る) |
| `/detect_msg`(`wrs_main_node_*.py` が購読) | 流れない | 衝突すると相手のモデル名が流れる |
| 引き出し3つ開放のボーナス | 1 回見れば加点 | 2 周期連続で3つ開いているときだけ |
| task2 の得点 | 100 / 40 / 100 / 50 | 20 / 10 / 30 / 10 |
| 既定の seed | 1 | 86 |

```bash
USE=1 bash tools/local_cpu/lab_sim/build_lab_sim.sh   # イメージ wrs_sim_lab:cpu を作り、~/sim_local/.env に設定を書く(1分ほど)
cd ~/sim_local && docker compose up -d                 # 以後このイメージで起動する
```

- 研究室のリポジトリは clone して決まったコミットを取り出すだけ(`~/smilab_ref`。push できないようにしてある)。
  取り出すのは `sim_wrs_ws`(product)がサブモジュールで指している版。研究室のコードはこのリポジトリに入れていない。
- 研究室のイメージとの違いは土台だけ(研究室: `tmc_wrs_docker:noetic-nvidia`、ここ: 公開の CPU イメージ)。
- **速くもなる**。シミュレータ単体・画面なし・何もしていない状態の RTF は、公開イメージ 0.33〜0.35 に対して
  研究室版 0.85〜0.99(2026-11-10、交互に計 5 回測定)。公開版の衝突ノードは世界中の接触情報を購読しているが、
  研究室版はバンパーの接触だけを購読するためと考えられる(推測)。ノードを動かしている最中にどれだけ速くなるかは未測定。
- 衝突判定の確認: ロボットを壁にぶつけると、`/detect_msg` に `wrc_frame` が流れた(2026-11-10)。
- 公開イメージに戻すには、`~/sim_local/.env` の `SIM_IMAGE=` の行を消して `docker compose up -d`。

## WSL が勝手に止まるとき

WSL は、Windows 側から何もつながっていない状態が約 1 分続くと、自動で終了する(コンテナも全部止まる。終了コード 0)。
ターミナルで WSL を開いたままにしておくか、`C:\Users\<ユーザ名>\.wslconfig` の `[wsl2]` に `vmIdleTimeout=-1` を書いて
`wsl --shutdown` する(PC のメモリを使い続けるので、使わないときは `wsl --shutdown` で止める)。

**授業サーバの代わりにはならない。** 手元では遅く(後述)、点数は GPU サーバと同じになるとは限らない。
デモ動画と最終的な点数の確認は、授業サーバで行うこと。

## 必要なもの

- Docker が使える Linux か **WSL2 の Ubuntu**(自分が `docker` グループに入っていること)
- メモリ 8GB 以上(シミュレータだけで約 3GB、YOLO を動かすとさらに増える)、空きディスク 20GB 以上
- GitHub のリポジトリを `git clone` できること(private なので、WSL 内で認証済みであること)

## 1. 初回セットアップ(1回だけ。数十分)

WSL の Ubuntu で、リポジトリの `tools/local_cpu/` にあるスクリプトを実行する。

```bash
bash tools/local_cpu/setup_local.sh            # 既定: 第5班のリポジトリの haruki ブランチ
bash tools/local_cpu/setup_local.sh <URL> <ブランチ>   # 別のリポジトリ・ブランチ
```

`~/sim_local/` に、`docker-compose.yml` と、リポジトリの clone(`~/sim_local/workspace/`)ができる。
その中で、イメージの取得・起動・`catkin build`・YOLO の CPU 用ライブラリの導入までを行う。
`docker compose` の `--build` と `down` は使わない。途中で止めても、もう一度実行すれば続きから進む。

## 2. ブラウザから実行する(ダッシュボード)

```bash
python3 ~/sim_local/workspace/tools/experiment/dashboard.py --dir ~/sim_local
```

表示された `http://localhost:8765/?token=...` を、Windows のブラウザで開く。

1. 「環境」の **起動** を押す(止まっていたコンテナが立ち上がる。WSL を閉じたあとはここから)
2. 「実験を実行」で、Seed・回数・コードの取得元・検出器を選んで **実行**
3. 進捗とログが出る。終わると「結果」の表に行が増える。行を押すとグラフと失敗の内訳が見える
4. Gazebo の画面を見たいときは「画面なし」のチェックを外して実行し、`http://localhost:3000` を開く
   (画面を出すとさらに遅くなる)

### コードの取得元(`_c.py` を変えたとき)

「コード」のプルダウンで、実行するファイルの出どころを選ぶ。元のファイルは変更されない
(`wrs_main_node_local.py` という一時ファイルに置いて実行する)。

| 選ぶもの | 使う場面 | 入力 |
|---|---|---|
| 作業ツリーのファイル | `~/sim_local/workspace` の中のファイルをそのまま。WebIDE(`http://localhost:3001`)で編集したとき | 版(統合 / a / b / c) |
| GitHub のブランチ/コミット | push したコードを実行する | ref(例 `haruki`、コミットID)と、任意でファイルパス(空なら `wrs_main_node_c.py`) |
| 任意のファイル | PC 上のどこにあるファイルでも | パス(`C:\Users\...\wrs_main_node_c.py` も `/home/...` も可) |

- 結果の表の「版 / 検出器」の下に、どこから取った何のコードかが出る(コミットIDやファイルの指紋は `meta.txt`)
- 構文エラーのファイルは、実験を始める前に断られる
- GitHub から取るには、WSL でそのリポジトリを `git fetch` できること(最初の `git clone` が通っていれば通る)

## コマンドだけで実行する

```bash
cd ~/sim_local/workspace/tools/experiment
./run_experiment.sh -d ~/sim_local -s 86 -n 1 -t 600 -G haruki -D yolo -l try1
./run_experiment.sh -d ~/sim_local -s 86 -f /mnt/c/Users/harur/Desktop/wrs_main_node_c.py -D yolo
```

`-G` は GitHub の ref(`haruki` または `haruki:パス`)、`-f` は任意のファイル、`-v c` は作業ツリーの `_c.py`。
オプションの一覧は `tools/experiment/README.md`。

## 注意(手元の環境でわかっていること)

- **遅い**。RTF(シミュレーション時間 / 実時間)は、シミュレータだけで約 0.3、検出器・移動計画まで全部動かして
  約 0.3(12 コア割り当て、画面なし。実測)。つまりシミュレーション 1 分が実時間で約 3 分強。
  Task 1 を通すには、実時間で数十分かかる。`-t`(上限時間)は 1500 秒以上にする
- **WSL の既定の割り当てを確認する。** `C:\Users\<名前>\.wslconfig` で `processors` と `memory` を絞っていると、
  PC の実力の一部しか使えない(8 コア 16 スレッドの PC で 4 コア・8GB だったことがある)。
  ただし、コアを増やしても RTF はほとんど変わらなかった(`gzserver` の重いスレッドが数本で、順番に処理するため)
- **何が重いかの実測**(`ablate_rtf.sh`。シミュレータ単体、画面なし、60 秒、基準 RTF 0.30):
  `undesired_contact_detector` を止めると 0.62、採点ノード全部を止めると 1.23、引き出し系の採点ノードだけ止めると 0.37。
  トレイ・容器・ビン・人の位置の判定ノード、物理の反復回数(100→50)は効果なし。刻み幅 3ms→6ms は 0.53 だが挙動が変わる。
  採点ノードは `/gazebo/get_model_state` などのサービスを呼び続けて、`gzserver` を詰まらせているとみている(未検証)
- **ノードを動かしている最中の RTF は、`-K undesired` でも上がらなかった**(約 0.3 のまま)。上の「2 倍」は、
  ノードを動かさないときの値。タスクの実行中は、MoveIt・検出器・センサの描画など、ほかの負荷が支配的になる
- `-K 正規表現`(`run_experiment.sh`)は、採点・監視ノードを止める高速な動作確認用。止めたノードの分の点は付かない
  (`undesired_contact_detector` は Task 2a の採点に関わる可能性が高い)。結果の記録に `monitors_stopped` が残る
- **同じシミュレータで実験を二重に動かさない。** どの実行も、始めるたびにシミュレータを再起動するので、
  重なるとお互いを壊す(実験ループは、すでに動いていれば断る)。測るときも、実験が終わってから
- **実行したファイルを確かめる。** 結果の表と `meta.txt` に、実行したファイル名・パス・sha が残る
  (統合版 `wrs_main_node.py` と `wrs_main_node_c.py` は別物。ダッシュボードの既定は統合版)
- **手元で、検出から把持・配置まで動くことは確認した**(`_c.py` と統合版の両方)。ただし、点数が 0 のままだった:
  `_c.py` では、引き出しが開かず(`cannot reach the front of the ... drawer handle`)、把持した物体を運ぶ途中で
  落とした(`lost X on the way (finger gap 0.001)`)。落ちたのが環境(CPU・遅さ)のせいか、コードのせいかは
  **切り分けられていない**。授業サーバで同じ症状が出るかを見るのが、いちばん確実
- GPU 決め打ちだった検出器は直した(`yolo_detector.py` / `frcnn_detector.py` は GPU がなければ CPU を使う)。
  `start_all.sh` は、NVIDIA GPU のない環境では、既定の検出器を YOLO にする
- 検出器は **YOLO(`-D yolo`)を使う**。Faster R-CNN(`frcnn_finetuned.launch`)が読む重み
  `id9_model_e50.pth` はリポジトリに入っていない(`.gitignore` で除外)ため、手元では動かない可能性が高い。
  `a`・`b` は既定が Faster R-CNN なので、手元で動かすときは `-D yolo` を付ける
- YOLO は CPU で動く(1枚あたり数百ミリ秒の見込み)。GPU サーバとは速度も、`sleep` のタイミングによる
  挙動の差も出る。点数の比較は、同じ環境どうしでする
- WSL を一定時間使わないと、コンテナが止まることがある。ダッシュボードの「起動」で戻せる
- Gazebo のシミュレータは、起動するたびに、物体の配置は同じ Seed で毎回同じになるが、
  物理演算とカメラ画像には揺らぎがある。変更の良し悪しは、3 回以上の平均で見る

## クラウドの VM で動かす(コア数が多い CPU を借りる)

遅さの原因は GPU ではなく CPU(Gazebo の物理演算)なので、Colab・Kaggle の無料 GPU は向かない
(Docker が使えず、CPU も少ない)。**コア数の多い Ubuntu の VM**(GPU なし)を借りて、同じ手順を実行する。
学生向けのクレジット(Azure for Students の $100 など)で足りるか、料金は各サービスの料金表で確認すること。

1. 8〜16 コア・メモリ 16GB 以上・ディスク 60GB の Ubuntu 22.04 / 24.04 の VM を作る
   (Azure for Students は、コア数の上限(クォータ)が小さいことがある。作る前に確認する)
2. SSH でログインし、リポジトリを置く。private なので、次のどちらか
   - 読み取り専用の短期トークンで `git clone`(**トークンを VM に残さない**。使い終わったら無効にする)
   - 自分の PC から `scp -r` で `~/sim_local/workspace` に送る
3. `bash tools/local_cpu/setup_vm.sh`(Docker の導入と `setup_local.sh`。**VM では未検証**。
   途中の表示を見ながら進めること)
4. 他に何も動いていない状態で、速さを測る: `bash tools/local_cpu/measure_rtf.sh`
   (手元の 4 コアでは 0.27 ほど。借りる価値があるかは、この値で判断する)
5. ダッシュボードは、自分の PC から SSH のポート転送で見る(`--lan` は使わない)

   ```bash
   ssh -L 8765:localhost:8765 ユーザー@VMのアドレス
   python3 ~/sim_local/workspace/tools/experiment/dashboard.py --dir ~/sim_local    # VM 側
   ```

   自分の PC のブラウザで `http://localhost:8765/?token=...` を開く
6. **使い終わったら VM を止める(削除する)。** 起動したままだとクレジットが減り続ける

授業の資料やコードを外部のサービスに置いてよいかは、念のため先生や TA に確認すること。
授業の評価は授業サーバで出した点数なので、VM は開発と練習用。デモ動画と最終確認は授業サーバで行う。

## 止める・消す

- 止める: ダッシュボードの「停止」、または `cd ~/sim_local && docker compose stop`
- 消す: `~/sim_local` を消す(イメージも消すなら `docker image rm` で個別に)。`docker compose down` は使わない
