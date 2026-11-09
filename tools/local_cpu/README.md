# 自分の PC(GPU なし・CPU だけ)でシミュレータを動かす

授業の GPU サーバと同じ構成(`simulator` / `workspace` / `xserver`、ポート 3000・3001)を、
GPU も NVIDIA ドライバもなしで自分の PC に作る。競技場・物体・採点は授業と同じ公開イメージ
(`ghcr.io/hsr-project/tmc_wrs_binary`)を使う。コードの動作確認、実験ループ、ダッシュボードの練習に向く。

**授業サーバの代わりにはならない。** 手元では遅く(後述)、点数は GPU サーバと同じにならない。
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

- **遅い**。4 コアの PC で、シミュレータだけなら実時間の 0.2〜0.36 倍、検出器・移動計画まで全部動かして
  0.27 倍ほど(画面なし。重い処理の最中はもっと落ちる可能性があり、未測定)。`-t`(上限時間)を長めにする
- **同じシミュレータで実験を二重に動かさない。** どの実行も、始めるたびにシミュレータを再起動するので、
  重なるとお互いを壊す(実験ループは、すでに動いていれば断る)。測るときも、実験が終わってから
- **手元で最後まで通した確認(点数が `/score` に入るところまで)は、まだできていない。** 起動して、YOLO の検出器と
  `wrs_main` が初期化されるところまでは確認した。点数まで確かめるには、他に何も動いていない状態で、1 回通す
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
