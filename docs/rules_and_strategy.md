# スコア定義・ルール解読・方針

出典: 実験手順書26 / 第1〜3週スライド / WRS2020 Partner Robot Detailed Rules (EN, §3.7, §3.10–3.11, §4.2)
本実験はルールブックを基にしているが、配点の一部が変更されている（手順書 FAQ A-1）。**本実験の値が優先**。

---

## 1. 評価の全体像

| 評価軸 | 中身 | 対策 |
|---|---|---|
| ①有用性 | Seed 86 + Seed 25 のスコア合計（or 独自機能の便利さ） | 本書の §2〜§4 |
| ②独自性 | 他班と似ていないか | 独自の工夫を1つ以上（例: 把持成功率を見た物体選択） |
| ③わかりやすさ | 他班評価 + コードの汎用性・可読性 | Seed 共通コード化、pycodestyle 警告≤10、pylint≥8.5 |

レベル目安（第1週スライド p.18）

| レベル | 条件 |
|---|---|
| 1 | Seed 86 で **120点以上** |
| 2 | Seed 25 + Seed 86 の合計 **220点以上**（Seed ごとに別コードでも可） |
| 3 | 両 Seed で動く共通コード → わかりやすさに加点（スコアが下がっても不利にならない） |

締切: 第1週・第2週とも 18:00 に Seed 86 のスコアを Slack 報告。第3週 16:00 デモ。

---

## 2. スコア定義（本実験版）

```
Score(seed) = S1 + S2a + S2b
```

### Task 1（片付け）: 上限なし・物体ごとに加算

```
S1 = Σ_物体 i [ 10 × (置き場所に入った) + 10 × (カテゴリが正しい) ] × ペナルティ係数_i
```

- 1物体あたり **最大20点**（正しい置き場所）／**10点**（置き場所に入ったがカテゴリ違い）
- 向きボーナス（ルールブックでは +10）→ **本実験では 0点**
- 判定は「物体の中心座標が置き場所の立方体内に入ったか」。判定が厳しい・バグあり（FAQ A-10）
- ルールブック上のペナルティ（シミュレータで適用されるかは**要確認**）
  - 衝突（指・手のひら以外が物体/家具に接触）: その物体の点 ×0.5
  - 落下: 次の物体の点 ×0.5（拾い直せばなし）
  - 何も持たずに置き場所へ行く（False delivery）: その物体 ×0

### Task 2a（障害物回避）: 30点
- 床の物体に**一切接触せず**ダイニング（Goal Area）へ到達 → **30点**（ルールブックでは100点）

### Task 2b（配達）: 最大30点
- 指示された正しい物体を時間内に届ける → **30点**
- 誤った物体を届ける → **20点**
- 指示文は棚の前に移動すると届く → **2a を先に実行する必要あり**（FAQ A-5）

### ルールブックにあるが本実験で扱いが不明なもの（要確認）
完遂ボーナス50点、残り時間ボーナス20点/分、引き出しを3つ開けるボーナス50点、Boss 50点、制限時間（15分＋5分）

### 1 Seed あたりの内訳イメージ

| 部分 | 上限 | 備考 |
|---|---|---|
| S2a + S2b | 60 | 固定配点。ここを落とさないのが前提 |
| S1 | 20 × 運べた物体数 | 唯一スケールする部分 |

→ **Lv1（120）には S1 ≥ 60＝正しいカテゴリで3個**（カテゴリ無視なら6個）が必要。
→ **Lv2（合計220）には 1 Seed あたり S1 ≈ 50〜60**（各 Seed で正しい置き場所に約3個）。

---

## 3. カテゴリと置き場所（ルールブック Table 2 × 検出ラベル56種）

| 置き場所 | 場所 | カテゴリ | 検出ラベル |
|---|---|---|---|
| Tray A / Tray B | Long Table | Food | cracker_box, sugar_box, pudding_box, gelatin_box, potted_meat_can, master_chef_can, tuna_fish_can, chips_can, mustard_bottle, tomato_soup_can, banana, strawberry, apple, lemon, peach, pear, orange, plum |
| Container A（大） | Long Table | Kitchen items | windex_bottle, bleach_cleanser, sponge, pitcher_base, pitcher_lid, plate, bowl, spatula, wine_glass, mug |
| Container B（小） | Long Table | Orientation-based | fork, spoon, large_marker, small_marker（各自のカテゴリの置き場所でも正解扱い） |
| Drawer top / bottom | 階段状の引き出し | Tools | large_marker, small_marker, padlock, bolt_and_nut, clamp |
| Drawer left | 階段状の引き出し | Shape items | credit_card_blank, mini_soccer_ball, softball, baseball, tennis_ball, racquetball, golf_ball, marble, **cup**, foam_brick, dice, rope, chain |
| Bin A | Bin A | Task items | rubiks_cube, colored_wood_block, nine_hole_peg_test, toy_airplane, lego_duplo, magazine, black_t_shirt, timer |
| Bin B | Bin B | Unknown / Boss | （検出器は "unknown" を出さない。扱いは要確認） |

注意
- **Container A = 大きい方、B = 小さい方**。シミュレータ上の名前は逆になっているが、ルールブックが正（FAQ A-7）
- `cup` は台所用品ではなく **Shape items**（YCB の重ねカップ）
- Drawer は開ける動作が必要な可能性大（サンプルに `pull_out_trofast` / `push_in_trofast` あり）

---

## 4. 現状のサンプルコードとのギャップ

| 箇所 | 現状 | 失っている点 |
|---|---|---|
| `execute_task1` の置き場所 | Bin A / Bin B に**交互に**入れる | カテゴリ点 10/物体をほぼ取れない（Task items 以外は不正解） |
| `execute_task1` の探索 | `tall_table` のみ、`DETECT_CNT = 1` → 最大1個 | S1 が最大20点で頭打ち |
| `coordinates.json` | Tray / Container / Drawer の置き位置がない | Food / Kitchen を正しく置けない |
| `extract_target_obj_and_person` | `"apple"` 固定（TODO） | 2b で 20点止まり or 失敗 |
| `deliver_to_target` | 届け先が `person_b` 固定 | 人物の取り違え（減点の有無は要確認） |
| `select_next_waypoint` | x 座標だけで判定（y を無視） | 2a の衝突で 30点を失う可能性 |
| `run()` | Task 2a / 2b がコメントアウト | 60点を失う |

---

## 5. 方針

基本原則（第2週スライド）: **「うまくいけば高得点」ではなく「平均点」を上げる**。失敗しやすい把持は最初から狙わない。

物体1個あたりの期待値で選ぶ:

```
E[点] = P(把持・運搬成功) × (10 + 10 × P(正しい置き場所に置ける)) − 衝突・落下リスク
```

### 優先順位（効果 ÷ 手間 の順）

1. **Task 2a / 2b を有効化**して 60点の土台を作る
   - 2a: `select_next_waypoint` で y 座標も見る。衝突ログ（`Collision Detected with task2a_*`）でチェック
   - 2b: 指示文をパースして物体名と人物（左右など）を取り出す → 届け先を切り替える
2. **Task 1 をカテゴリ別の置き場所に変更**（1物体 10点 → 20点）
   - ラベル → 置き場所の辞書を1つ定義（§3 の表）し、Seed 共通で使う
   - Tray / Container の座標・姿勢をシミュレータで調べて `coordinates.json` / `poses.json` に追加
3. **Task 1 で運ぶ物体を増やす**（唯一スケールする部分）
   - 探索地点を増やす: `tall_table` → `long_table_r` → `near_long_table_l`（床）
   - 同じ地点で検出 → 把持 → 運搬を、対象がなくなるまで繰り返す
4. **確実性を上げる**
   - 置きやすい置き場所の物体を優先する: Bin A（Task items）＞ Tray（Food）＞ Container A（Kitchen）＞ Drawer（Tools / Shape）
   - 引き出しが不安定なら Tools / Shape は「Bin に入れて確実に10点」を代替策にする
   - 小さい・掴みにくい・誤検出しやすいクラスは `IGNORE_LIST` に入れる
   - 衝突しやすい位置（机の奥・下）の物体は避ける
5. **Seed 25 対応 → 共通化（Lv3）**、linter 対応

### 進め方
- 1回走らせるごとに **Seed・各タスクの点・失敗理由**を記録する（Slack 報告とデモ動画の材料にもなる）
- 結果が毎回ぶれる（ランダム要素あり）ので、変更の良し悪しは **2〜3回の平均**で判断する

---

## 6. TA に確認すべき点

1. 第1週スライドの「Seed 1」は Seed 86 のことか（`change_seed.sh` の選択肢は 86 / 25 のみ）
2. 衝突・落下ペナルティ（×0.5）はシミュレータのスコアに反映されるか
3. 制限時間、完遂ボーナス50点、時間ボーナスはあるか
4. Task 2b で人物を間違えた場合の点数
5. Unknown objects（Bin B 行き）はシミュレータに存在するか
6. 引き出しを開ける必要があるか（最初から開いているか）
