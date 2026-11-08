# 知能ロボティクス実験 第5班

HSR シミュレータ上で Task 1（片付け）・Task 2a（障害物回避）・Task 2b（配達）を行うコード。
ルール・スコア定義・方針の詳細は [docs/rules_and_strategy.md](docs/rules_and_strategy.md) を参照。

## メンバーと担当

| 担当 | 名前 | 作業ファイル | 範囲 | 主な関数 |
|---|---|---|---|---|
| A: Task 1「取る」 | さくま | `wrs_main_node_a.py` | 物体検出結果の選別、掴みやすさの採点、把持、探索地点の追加 | `get_most_graspable_obj`, `calc_score_bbox`, `exec_graspable_method`, `grasp_from_*`, `execute_task1` の探索ループ |
| B: Task 1「置く」＋まとめ | むかい | `wrs_main_node_b.py` | ラベル→置き場所の対応、置く位置・姿勢、引き出し、`run()`、スコア記録、統合版への取り込み | `put_in_place`, `pull_out_trofast`, `push_in_trofast`, `run` |
| C: Task 2a・2b | いしい | `wrs_main_node_c.py` | 障害物回避経路、指示文の解析、配達 | `execute_task2a`, `execute_avoid_blocks`, `select_next_waypoint`, `execute_task2b`, `extract_target_obj_and_person`, `deliver_to_target` |

### ファイルの分け方

`src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/` に、同じ中身のファイルが4つある。

| ファイル | 用途 |
|---|---|
| `wrs_main_node.py` | 統合版（提出用）。3人の担当関数を取り込んだもの |
| `wrs_main_node_a.py` / `_b.py` / `_c.py` | A・B・C それぞれの作業用コピー。自分のファイルだけを編集する |

- 自分のファイルの中では、`run()` を自分のタスクだけに書き換えてよい（例: C は `execute_task2a()` と `execute_task2b()` だけ。2b の指示文は 2a で棚の前に行かないと届かないので、2a は外せない）
- 担当関数ができたら、その関数を統合版 `wrs_main_node.py` に移す。移す作業はむかいがまとめて行う
- 他人の作業ファイルは編集しない。ファイルが分かれているので、git の競合は起きにくい

モデル・数学の改善は、それを使うタスクの担当が持つ（例: 検出しきい値・掴みやすさの式は A、回避の幾何は C）。
物体検出モデルの再学習は後回し。第2週時点で余裕があれば独自機能として検討する。

### 担当関数の書き方（提出要件）

担当の判定は docstring で行われる。自分が書いた関数には必ず入れる。

```python
def select_next_waypoint(self, current_stp, pos_bboxes):
    """61X11111 氏名
    説明...
    """
```

1つの関数を複数人で書かない。必要なら関数を分割する。

## スコアの目安

```
Score(seed) = S1 + S2a + S2b
S1  = Σ 物体ごとに [置き場所に入った 10点 + カテゴリが正しい +10点]
S2a = 障害物に触れずに次の部屋へ 30点
S2b = 正しい物体を届ける 30点（誤った物体なら 20点）
```

| 目標 | 条件 | 期限 |
|---|---|---|
| レベル1 | Seed 86 で 120点以上 | 第1週 18:00（Slack 報告） |
| レベル2 | Seed 25 + Seed 86 の合計 220点以上 | 第2週 18:00（Slack 報告） |
| レベル3 | 両 Seed 共通のコード、linter 対応、スライド | 第3週 16:00 デモ |

## 実行方法

```bash
# 【in GPU server】シミュレータ起動（--build は使用禁止）
cd ~/sim_ws
docker compose up -d

# Seed の変更（86 または 25）→ シミュレータ再起動
./change_seed.sh 86
docker compose restart simulator

# 【in WebIDE】可視化とタスク実行
rviz -d /workspace/src/wrs_algorithm21/config/wrs_visualize.rviz
/workspace/src/wrs_algorithm21/scripts/start_all.sh      # 統合版 wrs_main_node.py
/workspace/src/wrs_algorithm21/scripts/start_all.sh a    # wrs_main_node_a.py（b, c も同様）
```

- シミュレータ画面: `http://<GPUサーバ名>.local:3000`、WebIDE: `http://<GPUサーバ名>.local:3001`
- 2回目以降は `docker compose restart simulator` で物体を元に戻し、Gazebo の Play を押してから実行
- GPU サーバは2台。同時にシミュレーションできるのは2人まで
- ビルド（`devel/` がないとき、`src/` の構成を変えたとき）: `cd /workspace && catkin build`

## 物体検出を YOLOv8 にする（担当: いしい）

今の検出器（torchvision の Faster R-CNN）の代わりに YOLOv8 を使えるようにしてある。
統合版（引数なし）と `c` は YOLO が標準。`a`・`b` は今までどおり Faster R-CNN。
`DETECTOR=yolo` / `DETECTOR=frcnn` で明示的に選べる。YOLO のライブラリか重みがないサーバでは、
自動で Faster R-CNN に戻る。

```bash
/workspace/src/wrs_algorithm21/scripts/start_all.sh                      # 統合版を YOLO で
DETECTOR=frcnn /workspace/src/wrs_algorithm21/scripts/start_all.sh       # 統合版を Faster R-CNN で
DETECTOR=yolo /workspace/src/wrs_algorithm21/scripts/start_all.sh b      # b を YOLO で試す
```

Task 2 の関数（いしい）は検出器を見て、YOLO のときは安定化の検出回数を 3→5 回（3回以上見えたら採用）に増やす。

ultralytics は Python 3.8 以上が必要なので、ROS noetic 本体の `/usr/bin/python3` と
`/workspace/.yolo_env`（git 管理外）のライブラリで検出ノードを動かす。

| ファイル | 内容 |
|---|---|
| `src/wrs_detector/scripts/wrs_detector/yolo_detector.py` | YOLO で検出するクラス（入出力は Faster R-CNN と同じ） |
| `src/wrs_detector/launch/yolo.launch` | YOLO 版の検出ノード（トピック・サービス名は同じ） |
| `tools/yolo/convert_to_yolo.py` | 授業のデータセット（COCO / VOC）を YOLO 形式に変換 |
| `tools/yolo/augment_dataset.py` | train を水増し（複製・ノイズ・傾け。BBox も回転に合わせる） |
| `tools/yolo/capture_images.py` | シミュレータの頭部カメラ画像を授業データセットと同じ形式（`seedN/NNNN_image.png` と `_pose.json`）で保存。`--mode tour`（地点を回って首を振る）/ `--mode passive`（走行中に一定間隔で撮る） |
| `tools/yolo/sam3_autolabel.py` | シミュレータの画像に SAM 3 で自動ラベル付け（理論上の手順。GPU・Python 3.12 が必要で未実行） |
| `tools/yolo/train_yolo.sh` | 学習して `checkpoints/yolo_wrs.pt` に置く |

### 準備（サーバごとに1回）

```bash
# 1. ライブラリ（約3GB、/workspace に入るのでコンテナを作り直しても残る）
/usr/bin/python3 -m pip install --target /workspace/.yolo_env \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.4.1+cu118" "torchvision==0.19.1+cu118" "ultralytics==8.3.40" "numpy<2"

# 2. データセット: 実験手順書の「独自データセット」（慶應アカウントの Google Drive）を
#    ダウンロードし、WebIDE にドラッグして /workspace/datasets/raw に置く
/usr/bin/python3 tools/yolo/convert_to_yolo.py /workspace/datasets/raw --out /workspace/datasets/wrs_yolo

# 3. 学習（GPU をシミュレータと取り合うので、誰もシミュレーションしていない時間に）
tools/yolo/train_yolo.sh yolov8s.pt 100 16
```

学習済みの重み `src/wrs_detector/checkpoints/yolo_wrs.pt`（22.5MB）は git に入れてあるので、
`git pull` すれば手に入る。もう1台のサーバで使うときは、上の 1. のライブラリだけ入れればよい（学習はしなくてよい）。

### 学習結果（2026-11-02、YOLOv8s、640px、100エポック中ベストは18エポック目）

データ: 授業の `hsr_sim_obj_1000`（labelme 形式、1017枚・5791個）。
検証用には seed6・8・15 の150枚を丸ごと使った（同じ部屋の配置の画像を学習に入れないため）。

| 指標 | 値 |
|---|---|
| mAP50 | 0.802 |
| mAP50-95 | 0.730 |
| 適合率 / 再現率 | 0.88 / 0.73 |
| 推論時間 | 約3ms / 枚（RTX 3090） |

- よく当たる: `cup` 0.98、`mug` 0.96、`plate` 0.95、果物・箱・缶はほぼ 0.99
- 弱い: `fork` 0.14、`spoon` 0.61（Container B 行き）、`lego_duplo` 0.73（IGNORE_LIST 対象）
- 学習データにない・少ないクラスは検出できない: `wine_glass`, `pitcher_lid`, `chain`, `rope`, `timer`,
  `magazine`, `black_t_shirt`, `bolt_and_nut`, `credit_card_blank`（0個）、`golf_ball`（1個）
- 今の Faster R-CNN は同じデータで学習されていて検証用の画像も学習に入っている可能性があるため、
  mAP の数字どうしでは比べられない。比較はシミュレータ上で行う（16:00 の統合時）

## Linter（提出前に必ず実行）

基準: pycodestyle の警告 10 以下、pylint のスコア 8.5 以上（1行の最大文字数は 100）。

```bash
cd /workspace
pycodestyle --max-line-length=100 src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/wrs_main_node.py
pylint --max-line-length=100 --rcfile src/wrs_algorithm21/pylintrc \
  src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/wrs_main_node.py
```

## Git の運用

- 各自が編集するのは自分の作業ファイル（`wrs_main_node_a.py` など）だけ。ファイルが分かれているので、同じブランチに push しても競合しにくい
- 作業前に `git pull`、区切りがついたら commit して push する。2台の GPU サーバ間の同期もこれで行う
- 統合版 `wrs_main_node.py` と `run()` の変更はむかいが行う
- 他人の担当関数は書き換えない。変更が必要なら担当者に相談する
- GPU サーバは第3週の後に初期化されるため、**第3週 17:30 までに必ず push** する

## 実行記録

結果のぶれがあるので、変更の良し悪しは 2〜3 回の平均で判断する。記録はむかいがまとめる。

| 日付 | Seed | 変更内容 | Task 1 | Task 2a | Task 2b | 合計 | 失敗した理由 |
|---|---|---|---|---|---|---|---|
| | 86 | サンプルそのまま | | | | | |
| 11/2 | 86 | | | | | 110 | |

## 提出物（第3週の次の日曜 18:00、K-LMS、zip 50MB 以下）

1. 表紙（.pdf、K-LMS のテンプレート）
2. 発表スライド `IR_5班発表スライド.pdf`（代表者のみ）
3. 自分の担当部分の説明スライド `IR_5班担当説明_氏名.pdf`（各自、2枚以上）
4. コード（.py）
