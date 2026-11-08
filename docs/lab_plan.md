# 実験室でやることの手順（11/8 作成）

- 作業ブランチ: `feature/list-items`（`main` より新しい。キャプチャのツールはこのブランチにだけある）
- 実行するノード: `wrs_main_node_c.py`（統合版 + 今回の変更）。統合版 `wrs_main_node.py` は main のまま

---

## 1. 実装したこと

| 項目 | ファイル | 内容 |
|---|---|---|
| 障害物を持って退かす（Task 2a） | `wrs_main_node_c.py` | 床の障害物を手前から持ち上げ、スタートの後ろ（`TASK2A_DROP_POS`）に置いてから直進。失敗したら A* で回避 |
| 人の間をスライドして渡す（Task 2b） | `wrs_main_node_c.py` | もう1人の前から、腕を出したまま届け先の人の前へスライドして離す |
| 引き出し（下を先に、上を大きく） | `wrs_main_node_c.py` | Task 1 の最初に下段 0.20m → 上段 0.32m 引き出し、重なる位置に落とす。工具系・形状系と、置けなかった物体を引き出しへ |
| 水増し | `tools/yolo/augment_dataset.py` | train を複製し、傾け・ノイズ・明るさ。BBox も回す |
| SAM 3 自動ラベル | `tools/yolo/sam3_autolabel.py` | 画像に56クラスの名前を渡して YOLO ラベルを作る |
| シミュレータの画像の保存 | `tools/yolo/capture_images.py` | 授業データセットと同じ形式（`seedN/NNNN_image.png`・`_pose.json`）で保存 |
| 記録 | `README.md`、`docs/RELEASE_NOTES.md` | 11/2 Seed 86 合計110、リリースノート（タグ `week1`） |

## 2. 実装したが、まだ確認していないこと

| 項目 | 確認済み | 未確認（実験室で確認する） |
|---|---|---|
| 引き出し | 置き場所の振り分け（ROS を差し替えたテスト） | 取っ手をつかめるか、上段が傾くか、点が2つ分入るか |
| 引き出しの座標 | 公開版 WRS ワールドから計算 | 授業のシミュレータでの値（特に取っ手の高さ 0.29 / 0.55 は推測） |
| Task 2a 持ち上げ | 一番手前を選ぶ・近づく位置 | 持ち上がるか、置き場所が邪魔にならないか |
| Task 2b スライド | 速度の向き | 点が入るか |
| `augment_dataset.py` | BBox のずれ1画素未満（Pillow 12） | サーバの Python 3.8 で動くか |
| `sam3_autolabel.py` | SAM 3 以外の部分（ダミーで確認） | SAM 3 本体でのラベルの質 |
| `capture_images.py` | 画像変換・番号・保存形式・地点一覧 | ROS につないで撮れるか |

## 3. まだやっていないこと（未着手）

- TA への確認（シミュレータの真値を使ってよいか）
- YOLO の再学習
- シミュレータ画像の撮影と SAM 3 でのラベル付け
- ベースライン計測（Task ごとの点・物体1個の秒数・失敗理由）
- Seed 25 での実行
- 今回の変更の統合版への取り込み（むかい）
- 提出物（動画・スライド・lint・レポート）

---

## 4. 実験室での手順

### 4.0 毎回の準備

【GPU サーバのターミナル】シミュレータを起動し、Seed を合わせる（86 または 25）

```bash
cd ~/sim_ws && docker compose up -d
```

```bash
cd ~/sim_ws && ./change_seed.sh 86 && docker compose restart simulator
```

ブラウザで `http://<サーバ名>.local:3000`（シミュレータ）を開いて **Play** を押す。`http://<サーバ名>.local:3001` が WebIDE。

【WebIDE】ブランチを最新にする（`devel/` がない・`src/` を変えたときはビルドも）

```bash
cd /workspace && git fetch origin && git checkout feature/list-items && git pull origin feature/list-items
```

```bash
cd /workspace && catkin build
```

【WebIDE】実行（ログをファイルにも残す）

```bash
/workspace/src/wrs_algorithm21/scripts/start_all.sh c 2>&1 | tee /workspace/run_$(date +%m%d_%H%M).log
```

2回目以降は【GPU サーバ】で `cd ~/sim_ws && docker compose restart simulator` → Play → WebIDE で Ctrl-C してから再実行。
RViz（1人だけ）: `rviz -d /workspace/src/wrs_algorithm21/config/wrs_visualize.rviz`

### 4.1 サーバの分担（2台を同時に使う）

| サーバ | やること |
|---|---|
| 1台目 | 4.4 YOLO 再学習（学習中はシミュレーションしない） |
| 2台目 | 4.2 引き出し → 4.3 Task 2a・2b の確認 |

学習が長引くときは、2台目は今の重みのまま確認を進め、新しい重みができたら 4.5 で測り直す。

### 4.2 引き出しの確認と調整（2台目）

1. **座標を測る**: Gazebo 左の World → Models → `trofast_2`（下段）・`trofast_3`（上段）の pose を見る。
   Gazebo の座標（world）から map 座標への変換（公開版ワールドから求めた式。ロボットの初期位置が world(-2.1, 1.2) のとき）:

   ```
   map_x = 1.2 - world_y
   map_y = world_x + 2.1
   ```

   まず `hsrb` の pose が world(-2.1, 1.2) になっているか確認する（違えばこの式は使えない）。

2. **定数を直す**（`wrs_main_node_c.py` の `DRAWER_` で始まる定数。下段の中心を map(cx, cy)、箱の底の高さを z0 とする）

   | 定数 | 今の値 | 決め方 |
   |---|---|---|
   | `DRAWER_HANDLE_X` | 0.2 | cx |
   | `DRAWER_HANDLE_Y` | -0.29 | cy + 0.31（前面 cy + 0.21 より 0.1 手前） |
   | `DRAWER_HANDLE_Z` | 下 0.29 / 上 0.55 | 各段の z0 + 0.19。取っ手の高さを見て合わせる |
   | `DRAWER_PULL` | 下 0.20 / 上 0.32 | 上段が前に傾くまで上を増やす |
   | `DRAWER_DROP` | y -0.33, z 0.70 | y = cy + 0.27（上下段が重なる所）、z は上段の縁より 0.1 上 |
   | `DRAWER_STANDPOINT` | [0.2, 0.25, -90] | 開けた引き出しに台車が当たらない位置 |

3. **引き出しだけ試す**: `wrs_main_node_c.py` の `run()` を一時的に次にして実行（確認後に元へ戻す）

   ```python
   self.change_pose("all_neutral")
   self.open_drawers()
   ```

   見る所: 下段 → 上段の順に開くか、上段が前に傾いて下段に重なるか。ログ:

   ```bash
   grep -E "pull .* drawer|could not open" /workspace/run_*.log
   ```

4. **点が2つ分入るか**: `run()` を元に戻して普通に実行し、引き出しに入れたときの Score の増え方を見る（1個で +20 より多く増えれば二重判定が効いている）。

   ```bash
   grep -E "Placement result|drawer_top_place" /workspace/run_*.log
   ```

5. 開かないときは `USE_DRAWER = False` にすれば、今までどおり bin に入れる。

### 4.3 Task 2a・2b の確認（2台目）

1. Task 2a・2b だけ試すときは `run()` を一時的に次にする（2b の指示文は 2a で棚の前に行かないと届かないので 2a は外せない）

   ```python
   self.change_pose("all_neutral")
   self.execute_task2a()
   self.execute_task2b()
   ```

2. Task 2a: 障害物を持ち上げて `TASK2A_DROP_POS`（[2.5, 1.45]）に置けるか。置いた物体が通り道の邪魔になるなら置き場所を変える。ログ:

   ```bash
   grep -E "task2a|Collision Detected with task2a" /workspace/run_*.log
   ```

   `task2a: no collision` で Score が +30 なら成功。持ち上げに失敗すると自動で A* 回避に切り替わる（`TASK2A_MODE = "avoid"` で最初から回避）。

3. Task 2b: スライド後に離して Score が +30（正しい物体）/ +20（違う物体）入るか。速すぎて判定されないなら `TASK2B_SLIDE_SPEED`（0.1 m/s）を下げる。

### 4.4 YOLO の再学習（1台目、誰もシミュレーションしていないとき）

1. **データを置く**: WebIDE のエクスプローラで `/workspace/datasets/raw` を作り、PC の `hsr_sim_obj_1000` フォルダをドラッグする。
   `a/test` は番号体系の違う YOLO ラベル（クラス番号 77 まである）なので消す。

   ```bash
   mkdir -p /workspace/datasets/raw
   ```

   ```bash
   rm -rf /workspace/datasets/raw/hsr_sim_obj_1000/a/test
   ```

2. **変換**（最後に `images: train ○ / val ○` と出れば成功。val はフォルダ単位で選ばれる）

   ```bash
   cd /workspace && /usr/bin/python3 tools/yolo/convert_to_yolo.py /workspace/datasets/raw --out /workspace/datasets/wrs_yolo
   ```

3. **今の重みを退避**（`train_yolo.sh` は最後に `yolo_wrs.pt` を上書きする）

   ```bash
   cp /workspace/src/wrs_detector/checkpoints/yolo_wrs.pt /workspace/yolo_wrs_old.pt
   ```

4. **水増し**（train だけ。1枚から2枚。もう一度実行すると作り直す）

   ```bash
   cd /workspace && PYTHONPATH=/workspace/.yolo_env /usr/bin/python3 tools/yolo/augment_dataset.py /workspace/datasets/wrs_yolo --copies 2
   ```

5. **学習**

   ```bash
   cd /workspace && tools/yolo/train_yolo.sh yolov8s.pt 100 16
   ```

6. **新旧を同じ val で比べる**（mAP50 が2つ出る。上が旧、下が新）

   ```bash
   cd /workspace && PYTHONPATH=/workspace/.yolo_env /usr/bin/python3 -c "from ultralytics import YOLO; [print(w, YOLO(w).val(data='/workspace/datasets/wrs_yolo/data.yaml', imgsz=640, device=0, plots=False).box.map50) for w in ['/workspace/yolo_wrs_old.pt', '/workspace/runs/wrs/weights/best.pt']]"
   ```

7. **採用 or 戻す**
   - 新が高い → そのまま（`yolo_wrs.pt` は新）。push する
     ```bash
     cd /workspace && git add src/wrs_detector/checkpoints/yolo_wrs.pt && git commit -m "Retrain YOLO with augmented data" && git push origin feature/list-items
     ```
   - 新が低い → 戻す
     ```bash
     cp /workspace/yolo_wrs_old.pt /workspace/src/wrs_detector/checkpoints/yolo_wrs.pt
     ```

8. もう1台のサーバは `git pull` で新しい重みを受け取る（YOLO のライブラリ `/workspace/.yolo_env` は README の手順で各サーバに入れる）。

### 4.5 記録（毎回）

検出器と動作が決まったら、Seed 86 で 2〜3 回走らせ、README の「実行記録」に1行ずつ書く:
日付 / Seed / 変更内容 / Task 1 / Task 2a / Task 2b / 合計 / 失敗した理由。
物体1個にかかった秒数は `Placement result` の `elapsed_seconds` で分かる。

```bash
grep -E "Placement result" /workspace/run_*.log
```

### 4.6 画像を撮る（SAM 3 用。4.4 の後、余裕があれば）

【WebIDE】ROS の環境を読み込む（numpy・PIL が見つからないときは2行目の形で実行）

```bash
source /opt/ros/noetic/setup.bash && source /workspace/devel/setup.bash
```

- tour（ロボットを9地点に動かし首を振る。81枚。他の人がシミュレーションしていないとき）
  ```bash
  cd /workspace && python3 tools/yolo/capture_images.py --seed 86 --mode tour
  ```
  ```bash
  cd /workspace && PYTHONPATH=$PYTHONPATH:/workspace/.yolo_env python3 tools/yolo/capture_images.py --seed 86 --mode tour
  ```
- passive（別のターミナルで `start_all.sh c` を走らせている間、2秒ごと・動いたときだけ撮る。Ctrl-C で終了）
  ```bash
  cd /workspace && python3 tools/yolo/capture_images.py --seed 86 --mode passive --interval 2
  ```

Seed 25 でも同じことをする（4.0 で `./change_seed.sh 25`）。保存先は `/workspace/datasets/capture/seed86/` など。
zip にして WebIDE のエクスプローラで右クリック → Download:

```bash
cd /workspace/datasets && /usr/bin/python3 -m zipfile -c capture.zip capture
```

### 4.7 SAM 3 でラベルを付ける（自分の PC か Colab の GPU）

必要: Python 3.12 以上・PyTorch 2.7 以上・CUDA 12.6 以上の GPU。
先に https://huggingface.co/facebook/sam3 で利用申請し、通ってから進める。

1. Colab（ランタイムを GPU に）で SAM 3 を入れる（SAM 3 の README の手順）
   ```
   !pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
   !git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .
   ```
   ```python
   from huggingface_hub import login; login()
   ```
2. 必要なファイルを、リポジトリと同じ並びで置く（スクリプトが `../../src/...` のクラス表を読むため）
   - `/content/repo/tools/yolo/sam3_autolabel.py`
   - `/content/repo/src/wrs_detector/config/wod-c_classes.json`
   - `capture.zip` をアップロードして展開: `!unzip -q capture.zip -d /content/`
3. ラベル付け（preview に BBox を描いた画像が出る）
   ```
   !python /content/repo/tools/yolo/sam3_autolabel.py /content/capture --out /content/wrs_sam3 --preview
   ```
4. **preview を全部目で見て**、間違ったものは `images/train/X.png` と `labels/train/X.txt` を両方消す。
5. `!cd /content && zip -qr wrs_sam3.zip wrs_sam3` → ダウンロード → WebIDE で `/workspace/datasets` にドラッグ。
6. 【WebIDE】展開して train に足し、4.4 の 4〜7（水増し → 学習 → 比較 → 採用）をやり直す
   ```bash
   cd /workspace/datasets && /usr/bin/python3 -m zipfile -e wrs_sam3.zip . && cp wrs_sam3/images/train/* wrs_yolo/images/train/ && cp wrs_sam3/labels/train/* wrs_yolo/labels/train/
   ```

### 4.8 統合とレベル

1. 確認できた変更を、むかいに統合版 `wrs_main_node.py` へ取り込んでもらう（README のルール）。
2. Seed 25 でも走らせて記録（レベル2: Seed 25 + 86 の合計 220 以上）。
3. 両 Seed を同じコードで動かす（レベル3: わかりやすさに加点）。
4. `feature/list-items` を main にマージ。

### 4.9 提出・発表（第3週）

1. lint（提出する .py ごと。pycodestyle 警告 10 以下、pylint 8.5 以上）
   ```bash
   cd /workspace && pycodestyle --max-line-length=100 src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/wrs_main_node.py
   ```
   ```bash
   cd /workspace && pylint --max-line-length=100 --rcfile /home/developer/workspace/src/wrs_algorithm21/pylintrc src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/wrs_main_node.py
   ```
2. 動画: OBS で得点が映るように撮る（ないと大幅減点）。N 倍速は自分の PC で（例: 3倍速）
   ```bash
   ffmpeg -i input.mp4 -vcodec h264 -vf setpts=PTS/3.0 output.mp4
   ```
3. 発表スライド（5分）: タイトルに班番号（5班）と全員の学籍番号・氏名（ないと0点）。得点の動画かスクショを入れる。
4. 各自の担当説明スライド（2枚以上、学籍番号・氏名）。
5. **第3週 17:30 までに push**（サーバは初期化される）。
6. 発表当日: 発表者以外はマイク・スピーカ off。Google フォームでコメント（未提出は減点）、自分の班以外に2票、送信を押す。
7. レポート（第3週の次の日曜 18:00、K-LMS、zip 50MB 以下）: 表紙 / `IR_5班発表スライド.pdf`（代表のみ）/ `IR_5班担当説明_氏名.pdf` / .py。

### 4.10 帰る前

```bash
cd /workspace && git status && git push origin feature/list-items
```

【GPU サーバ】止めてからシャットダウン（コーディングエージェントを使った人は認証情報も消す）

```bash
cd ~/sim_ws && docker compose stop
```

```bash
rm ~/.claude/.credentials.json
```
