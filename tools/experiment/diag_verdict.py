#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
node.log の "[DIAG] {json}" の行（wrs_main_node_c.py の Diagnostics が出す）を集計して、
腕の指令が通らない原因を 3 つの仮説に振り分ける。

  仮説1 CPU 環境で計画が間に合わない
  仮説2 計画の負荷でコントローラが止まる
  仮説3 move_joints が毎回新しい接続を作り、5 秒の待ちでは間に合わない

使い方:
  python3 tools/experiment/diag_verdict.py ~/sim_local/results/<実行のフォルダ>/node.log
  （フォルダを渡すと、その中の node.log を読む。複数渡すと実行ごとに集計する）

判定の基準（しきい値はここに書いたとおり。変えるときは下の定数を直す）:
  仮説1  失敗した MoveIt 呼び出しの 6 割以上が「PLANNING のまま終わった（MONITOR＝実行に入っていない）」
         → 計画の段階で失敗している。そのとき PC の CPU 使用率の中央値が 0.8 以上、
           または load1/コア数が 1 以上なら「CPU 不足と整合」、そうでなければ
           「CPU 不足ではなく、目標が到達不能・IK の失敗の可能性」。
         6 割以上が MONITOR に入っていれば「計画は通り、実行の段階で時間切れ」→ 仮説1 は否定寄り。
  仮説2  腕の指令が繋がらなかった瞬間に、常駐の購読で action の status が 2 秒以上届いていない（一度も届いていない
         を含む）、または controller_manager が応答しない・arm/head のコントローラが running でない
         → サーバー側が止まっている。
  仮説3  繋がらなかった瞬間に、status が 1 秒以内に届いていて約 3Hz 以上ある
         → サーバーは動いているのに、新しく作ったクライアントが 5 秒で繋がらない。
  どれも「そうである可能性が高い」という判定で、証明ではない（1 回の実行の記録なので、複数回で確かめること）。
  CPU 使用率・load は PC 全体の値で、シミュレータのプロセス単体ではない。
"""
import json
import os
import statistics
import sys

PLANNING_RATIO = 0.6
CPU_HIGH = 0.8
LOAD_HIGH = 1.0
ARRIVED_M = 0.03
STALE_S = 2.0
FRESH_S = 1.0
FRESH_HZ = 3.0


def read_records(path):
    """node.log から [DIAG] の json を集める"""
    if os.path.isdir(path):
        path = os.path.join(path, "node.log")
    records = []
    with open(path, encoding="utf-8", errors="replace") as log:
        for line in log:
            if "[DIAG] " not in line:
                continue
            try:
                records.append(json.loads(line.split("[DIAG] ", 1)[1]))
            except ValueError:
                continue
    return records


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=2):
    return "-" if value is None else ("{:.%df}" % digits).format(value)


def load_ratio(rec):
    if rec.get("load1") is None or not rec.get("cores"):
        return None
    return rec["load1"] / rec["cores"]


def hypothesis1(records):
    moves = [r for r in records if r.get("kind") == "moveit" and r.get("label") != "gripper"]
    failed = [r for r in moves if r.get("ok") is not True]
    ok = [r for r in moves if r.get("ok") is True]
    print("== 仮説1 CPU 環境で計画が間に合わない ==")
    print("  MoveIt の呼び出し {} 回、うち失敗 {} 回".format(len(moves), len(failed)))
    if not failed:
        print("  判定: 判定不能（MoveIt の失敗が記録されていない）")
        return "判定不能"
    planning = [r for r in failed if r.get("states") and "MONITOR" not in r["states"]]
    execution = [r for r in failed if "MONITOR" in (r.get("states") or [])]
    unknown = [r for r in failed if not r.get("states")]
    print("  失敗の内訳: 計画のまま終わった {} / 実行に入ってから失敗 {} / move_group の状態が取れず不明 {}".format(
        len(planning), len(execution), len(unknown)))
    print("  所要時間(壁時計)の中央値: 失敗 {} 秒 / 成功 {} 秒".format(
        fmt(med([r["wall_s"] for r in failed])), fmt(med([r["wall_s"] for r in ok]))))
    cpu_failed = med([r.get("cpu_util") for r in failed])
    print("  CPU 使用率の中央値: 失敗時 {} / 成功時 {}   load1/コア数: 失敗時 {} / 成功時 {}".format(
        fmt(cpu_failed), fmt(med([r.get("cpu_util") for r in ok])),
        fmt(med([load_ratio(r) for r in failed])), fmt(med([load_ratio(r) for r in ok]))))
    print("  シミュレータの速さ RTF の中央値: 失敗時 {} / 成功時 {}".format(
        fmt(med([r.get("rtf") for r in failed])), fmt(med([r.get("rtf") for r in ok]))))
    codes = {}
    for r in failed:
        codes[r.get("mg_code")] = codes.get(r.get("mg_code"), 0) + 1
    print("  move_group のエラーコード(-1 計画失敗, -4 制御失敗, -6 時間切れ, None 不明): {}".format(codes))
    arrived = [r for r in execution if r.get("end_dist_m") is not None and r["end_dist_m"] <= ARRIVED_M]
    if execution:
        print("  実行に入って失敗した {} 回のうち、手が目標から {} m 以内に着いていたもの: {} 回".format(
            len(execution), ARRIVED_M, len(arrived)))
    busy = (cpu_failed is not None and cpu_failed >= CPU_HIGH) or \
           ((med([load_ratio(r) for r in failed]) or 0) >= LOAD_HIGH)
    if len(planning) >= PLANNING_RATIO * len(failed):
        if busy:
            verdict = "支持（計画の段階で失敗しており、CPU も逼迫している）"
        else:
            verdict = ("CPU 不足とは言えない（計画の段階で失敗しているが CPU は逼迫していない。"
                       "目標が到達不能・IK の失敗の可能性）")
    elif len(execution) >= PLANNING_RATIO * len(failed):
        verdict = "否定寄り（計画は通り、実行の段階で時間切れ。仮説2・3 を確認）"
    elif len(unknown) >= PLANNING_RATIO * len(failed):
        verdict = "判定不能（move_group の状態が取れなかった。所要時間の中央値 {} 秒を計画時間の上限と見比べる）".format(
            fmt(med([r["wall_s"] for r in failed])))
    else:
        verdict = "判定不能（失敗の種類が混ざっている）"
    print("  判定: " + verdict)
    return verdict


def classify_unavailable(rec):
    """繋がらなかった 1 回を 'stopped' / 'alive' / 'unclear' に分ける"""
    controllers = rec.get("controllers") or {}
    if "error" in controllers and "no response" in controllers["error"]:
        return "stopped"
    bad_state = [name for name, state in controllers.items()
                 if name in ("arm_trajectory_controller", "head_trajectory_controller") and state != "running"]
    age, hz = rec.get("status_age_s"), rec.get("status_hz")
    if bad_state or age is None or age > STALE_S:
        return "stopped"
    if age <= FRESH_S and (hz or 0) >= FRESH_HZ:
        return "alive"
    return "unclear"


def hypotheses23(records):
    events = [r for r in records if r.get("kind") == "controller_unavailable"]
    joints = [r for r in records if r.get("kind") == "joints"]
    print("== 仮説2 計画の負荷でコントローラが止まる / 仮説3 接続が毎回の新規で間に合わない ==")
    connects = [v for r in joints for v in (r.get("connect_wall_s") or {}).values()]
    print("  繋がった move_joints {} 回の接続時間(壁時計) 中央値 {} 秒 / 最大 {} 秒".format(
        len(joints), fmt(med(connects), 3), fmt(max(connects) if connects else None, 3)))
    print("  繋がらなかった回数: {}".format(len(events)))
    if not events:
        print("  判定: 判定不能（'joint controller unavailable' が起きなかった）")
        return "判定不能", "判定不能"
    kinds = {"stopped": 0, "alive": 0, "unclear": 0}
    for rec in events:
        kind = classify_unavailable(rec)
        kinds[kind] += 1
        retry = rec.get("retry") or {}
        controllers = rec.get("controllers") or {}
        last = rec.get("last_moveit") or {}
        tick = rec.get("last_tick") or {}
        print("  t={:>6}s {:<4} 分類={:<7} status の最終受信から {} 秒({} Hz) joint_states {} 秒 "
              "arm/head={}/{}".format(
                  rec.get("t_sim"), rec.get("action"), kind, fmt(rec.get("status_age_s")), rec.get("status_hz"),
                  fmt(rec.get("js_age_s")), controllers.get("arm_trajectory_controller", "?"),
                  controllers.get("head_trajectory_controller", "?")))
        print("           待ち直し {} 秒で{}  直前の MoveIt: {} ok={} {} 秒  直前の tick: RTF {} CPU {}".format(
            fmt(retry.get("wall_s")), "繋がった" if retry.get("connected") else "繋がらず",
            last.get("label"), last.get("ok"), fmt(last.get("wall_s")), tick.get("rtf"), tick.get("cpu_util")))
    total = len(events)
    if kinds["stopped"] >= 0.6 * total:
        verdict2 = "支持（サーバー側の status が止まっている、またはコントローラが running でない）"
    elif kinds["alive"] >= 0.6 * total:
        verdict2 = "否定寄り（サーバーは status を出し続けていた）"
    else:
        verdict2 = "判定不能（分類が混ざっている）"
    if kinds["alive"] >= 0.6 * total:
        verdict3 = "支持（サーバーは動いているのに、新しいクライアントが 5 秒で繋がらなかった）"
    elif kinds["stopped"] >= 0.6 * total:
        verdict3 = "否定寄り（サーバー側が止まっていた。待ち直しで繋がるかも併せて見る）"
    else:
        verdict3 = "判定不能（分類が混ざっている）"
    print("  判定 仮説2: " + verdict2)
    print("  判定 仮説3: " + verdict3)
    return verdict2, verdict3


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    for path in paths:
        records = read_records(path)
        print("#" * 70)
        print("# {}   [DIAG] の行: {}".format(path, len(records)))
        print("#" * 70)
        if not records:
            print("  [DIAG] の行がない（診断が無効か、古いコードで走った実行）")
            continue
        start = [r for r in records if r.get("kind") == "start"]
        if start:
            print("  PC のコア数 {}、起動時の load1 {}、MoveIt の計画時間 {} 秒 × 試行 {}".format(
                start[0].get("cores"), start[0].get("load1"),
                start[0].get("planning_time"), start[0].get("planning_attempts")))
        hypothesis1(records)
        hypotheses23(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
