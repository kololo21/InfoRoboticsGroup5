#!/usr/bin/env python3
"""実験ダッシュボード: シミュレータの起動/停止、実験の実行、スコア表を1つのWeb画面で扱う。

GPUサーバ(ホスト側。WebIDEの中ではない)で起動する。標準ライブラリだけで動く(pip不要)。

    python3 dashboard.py                  # 自分のPCだけから見る: http://localhost:8765/?token=...
    python3 dashboard.py --lan            # 同じ実験室のPC(ノートPC)から見る: http://<サーバ名>.local:8765/?token=...

画面に出るURL(token付き)を開く。tokenを知らない人は操作できない(実験室のWi-Fiは他の班も使うため)。
動かすコマンドは決まったものだけで、入力値は検証してから渡す。docker compose の --build や down は使わない。
"""
import argparse
import csv
import json
import os
import re
import secrets
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
SEEDS_ALLOWED = ["1", "25", "31", "36", "59", "86"]
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
NAME_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[A-Za-z0-9_.-]{1,40}$")
RUN_RE = re.compile(r"^seed[0-9]+_run[0-9]+$")
SEED_FILE = "/opt/ros/noetic/share/hsrb_wrs_gazebo_launch/launch/wrs_practice0_easy_tmc.launch"


def clean(text):
    return ANSI.sub("", text).replace("\r", "\n")


def tail_text(path, nbytes=6000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - nbytes))
            return clean(f.read().decode("utf-8", "replace"))
    except OSError:
        return ""


def read_kv(path):
    out = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k] = v
    except OSError:
        pass
    return out


def last_score(timeline):
    """score_timeline.csv の最後の値。無ければ None"""
    for line in reversed(tail_text(timeline, 3000).splitlines()):
        parts = line.split(",")
        if len(parts) == 2 and parts[0].isdigit():
            try:
                return float(parts[1])
            except ValueError:
                pass
    return None


def stats(values):
    n = len(values)
    mean = sum(values) / n
    var = sum(v * v for v in values) / n - mean * mean
    return mean, (var ** 0.5 if var > 0 else 0.0)


def summarize(rows):
    """summary.csv の行 -> Seed別の平均・ばらつき・合計・robust・worst・レベル"""
    by_seed = {}
    for r in rows:
        if r["score_final"] != "":
            by_seed.setdefault(r["seed"], []).append(float(r["score_final"]))
    per_seed, total, robust, worst = {}, 0.0, 0.0, 0.0
    for seed, vals in by_seed.items():
        mean, sd = stats(vals)
        per_seed[seed] = {"n": len(vals), "mean": mean, "sd": sd, "min": min(vals), "max": max(vals)}
        total += mean
        robust += mean - sd
        worst += min(vals)
    level = "-"
    if per_seed.get("86", {}).get("mean", 0) >= 120:
        level = "Lv1"
    if "86" in per_seed and "25" in per_seed and per_seed["86"]["mean"] + per_seed["25"]["mean"] >= 220:
        level = "Lv2"
    n_min = min((s["n"] for s in per_seed.values()), default=0)
    return {"per_seed": per_seed, "total": total, "robust": robust, "worst": worst,
            "level": level, "n": n_min}


class App:
    def __init__(self, args):
        self.dir = Path(args.dir).expanduser().resolve()
        self.results = self.dir / "results"
        self.script = HERE / "run_experiment.sh"
        self.start_cmd = args.start_cmd
        self.token = args.token or secrets.token_urlsafe(9)
        self.lan = args.lan
        self.lock = threading.Lock()
        self.job = None
        self.cache = {}

    # ---- docker -------------------------------------------------------
    def dc(self, *argv, timeout=20):
        try:
            p = subprocess.run(["docker", "compose", *argv], cwd=self.dir, capture_output=True,
                               text=True, timeout=timeout)
            return p.returncode, "\n".join(l for l in p.stdout.splitlines() if "obsolete" not in l)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 1, str(e)

    def cached(self, key, ttl, fn):
        now = time.time()
        hit = self.cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        val = fn()
        self.cache[key] = (now, val)
        return val

    def containers(self):
        def fn():
            rc, out = self.dc("ps", "-a", "--format", "json")
            states = {}
            if rc == 0 and out.strip():
                try:
                    data = json.loads(out)
                    items = data if isinstance(data, list) else [data]
                except json.JSONDecodeError:
                    items = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
                for it in items:
                    states[it.get("Service", "?")] = it.get("State", "?")
            return states
        return self.cached("containers", 2.5, fn)

    def current_seed(self):
        def fn():
            rc, out = self.dc("exec", "-T", "simulator", "bash", "-c",
                              f"grep -ho 'name=\"seed\" default=\"[0-9]*\"' {SEED_FILE} | head -1", timeout=10)
            m = re.search(r'default="([0-9]+)"', out) if rc == 0 else None
            return m.group(1) if m else None
        return self.cached("seed", 30, fn)

    def idle_score(self):
        def fn():
            rc, out = self.dc("exec", "-T", "simulator", "bash", "-c",
                              "source /opt/ros/noetic/setup.bash; timeout 4 rostopic echo -n1 /score", timeout=10)
            m = re.search(r"data:\s*([-0-9.]+)", out) if rc == 0 else None
            return float(m.group(1)) if m else None
        return self.cached("score", 10, fn)

    # ---- ジョブ(同時に1つだけ) -----------------------------------------
    def start_job(self, kind, argv, params=None, shell_env=None):
        with self.lock:
            if self.job and self.job["proc"].poll() is None:
                return False, f"実行中のジョブがあります({self.job['kind']})。終わるか中止してください"
            if self.job:
                self.job["logf"].close()
            self.results.mkdir(parents=True, exist_ok=True)
            logdir = self.results / "_dashboard"
            logdir.mkdir(exist_ok=True)
            logpath = logdir / "job.log"
            env = dict(os.environ)
            if shell_env:
                env.update(shell_env)
            logf = open(logpath, "w", encoding="utf-8")
            proc = subprocess.Popen(argv, cwd=self.dir, stdout=logf, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            self.job = {"kind": kind, "proc": proc, "log": logpath, "params": params or {},
                        "started": time.time(), "logf": logf}
            self.cache.clear()
            return True, "開始しました"

    def abort_job(self):
        with self.lock:
            job = self.job
            if not job or job["proc"].poll() is not None:
                return False, "実行中のジョブはありません"
            os.killpg(os.getpgid(job["proc"].pid), signal.SIGTERM)
        # ワークスペース側で動いているROSノード(roslaunch)も止める。コンテナは消さない
        self.dc("exec", "-T", "workspace", "pkill", "-f", "roslaunch")
        self.cache.clear()
        return True, "中止しました"

    def job_info(self):
        job = self.job
        if not job:
            return None
        rc = job["proc"].poll()
        text = tail_text(job["log"], 20000)
        info = {"kind": job["kind"], "running": rc is None, "exit": rc, "params": job["params"],
                "elapsed": int(time.time() - job["started"]), "log": text[-3500:]}
        if job["kind"] == "run":
            runs = re.findall(r"== Seed (\d+)\s+run (\d+)", text)
            total = len(job["params"].get("seeds", [])) * job["params"].get("repeat", 1)
            info["progress"] = {"done": max(0, len(runs) - (1 if rc is None else 0)), "total": total,
                                "current": runs[-1] if runs else None}
            m = re.findall(r"^OUT=(.+)$", text, re.M)
            if m:
                info["out"] = Path(m[-1]).name
                if runs and rc is None:
                    tl = Path(m[-1]) / f"seed{runs[-1][0]}_run{runs[-1][1]}" / "score_timeline.csv"
                    info["live_score"] = last_score(tl)
        return info

    # ---- 結果 ----------------------------------------------------------
    def read_rows(self, d):
        try:
            with open(d / "summary.csv", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except OSError:
            return []

    def list_results(self):
        out = []
        if not self.results.is_dir():
            return out
        for d in sorted(self.results.iterdir(), reverse=True):
            if not (NAME_RE.match(d.name) and (d / "summary.csv").exists()):
                continue
            meta = read_kv(d / "meta.txt")
            s = summarize(self.read_rows(d))
            out.append({"name": d.name, "meta": meta, **s})
        return out

    def result_detail(self, name):
        d = self.results / name
        if not NAME_RE.match(name) or not d.is_dir():
            return None
        rows = self.read_rows(d)
        runs = []
        for r in rows:
            run_dir = d / f"seed{r['seed']}_run{r['run']}"
            pts, prev, t0 = [], None, None
            try:
                for line in run_dir.joinpath("score_timeline.csv").read_text().splitlines()[1:]:
                    a, _, b = line.partition(",")
                    if not a.isdigit() or not b:
                        continue
                    t0 = int(a) if t0 is None else t0
                    v = float(b)
                    if v != prev:
                        pts.append([round((int(a) - t0) / 1e9, 1), v])
                        prev = v
                    last = [round((int(a) - t0) / 1e9, 1), v]
                if pts:
                    pts.append(last)
            except (OSError, ValueError):
                pass
            runs.append({**r, "timeline": pts})
        fails = []
        try:
            with open(d / "failures.csv", encoding="utf-8") as f:
                fails = list(csv.DictReader(f))
        except OSError:
            pass
        return {"name": name, "meta": read_kv(d / "meta.txt"), "runs": runs, "failures": fails,
                **summarize(rows)}

    def run_log(self, name, run):
        if not (NAME_RE.match(name) and RUN_RE.match(run)):
            return None
        return tail_text(self.results / name / run / "run.log", 12000)

    # ---- 操作 ----------------------------------------------------------
    def do_action(self, body):
        act = body.get("action")
        if act == "abort":
            return self.abort_job()
        if act == "up":
            return self.start_job("up", ["docker", "compose", "up", "-d"])
        if act == "stop":
            return self.start_job("stop", ["docker", "compose", "stop"])
        if act == "restart_sim":
            return self.start_job("restart_sim", ["docker", "compose", "restart", "simulator"])
        if act == "seed":
            seed = str(body.get("seed", ""))
            if seed not in SEEDS_ALLOWED:
                return False, "Seed が不正です"
            if not (self.dir / "change_seed.sh").exists():
                return False, "change_seed.sh が見つかりません(授業のサーバーにあります)"
            return self.start_job("seed", ["bash", "-c", f"./change_seed.sh {seed} && docker compose restart simulator"])
        if act == "run":
            return self.run_experiment(body)
        return False, "不明な操作です"

    def run_experiment(self, b):
        seeds = [str(s) for s in b.get("seeds", [])]
        if not seeds or any(s not in SEEDS_ALLOWED for s in seeds):
            return False, "Seed を1つ以上選んでください"
        try:
            repeat = int(b.get("repeat", 1))
            timeout = int(b.get("timeout", 900))
        except (TypeError, ValueError):
            return False, "回数・時間が不正です"
        if not (1 <= repeat <= 20 and 60 <= timeout <= 3600):
            return False, "回数は1〜20、上限時間は60〜3600秒にしてください"
        variant, det = b.get("variant", "main"), b.get("detector", "")
        label = str(b.get("label", "run"))
        if variant not in ("main", "a", "b", "c") or det not in ("", "yolo", "frcnn"):
            return False, "版または検出器が不正です"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,30}", label):
            return False, "ラベルは英数字と - _ の30文字までです"
        argv = [str(self.script), "-d", str(self.dir), "-s", " ".join(seeds), "-n", str(repeat),
                "-t", str(timeout), "-l", label, "-v", variant]
        if det:
            argv += ["-D", det]
        if b.get("interleave"):
            argv.append("-i")
        if not b.get("headless", True):
            argv.append("-g")
        env = {"START_CMD": self.start_cmd} if self.start_cmd else None
        return self.start_job("run", argv, {"seeds": seeds, "repeat": repeat, "label": label}, env)


class Handler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, *args):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if not self.app.lan:   # localhost限定のときはDNS rebinding対策としてHostも確認
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if host not in ("localhost", "127.0.0.1", "::1"):
                return False
        return secrets.compare_digest(self.headers.get("X-Token", ""), self.app.token)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self.send(200, (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        if not u.path.startswith("/api/"):
            return self.send(404, {"error": "not found"})
        if not self.authorized():
            return self.send(403, {"error": "token が違います"})
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        a = self.app
        if u.path == "/api/status":
            job = a.job_info()
            running = bool(job and job["running"])
            cont = a.containers()
            sim_up = cont.get("simulator") == "running"
            score = job.get("live_score") if running else (a.idle_score() if sim_up else None)
            return self.send(200, {"containers": cont, "seed": a.current_seed() if sim_up else None,
                                   "score": score, "job": job, "dir": str(a.dir)})
        if u.path == "/api/results":
            return self.send(200, a.list_results())
        if u.path == "/api/result":
            d = a.result_detail(q.get("name", ""))
            return self.send(200 if d else 404, d or {"error": "not found"})
        if u.path == "/api/runlog":
            t = a.run_log(q.get("name", ""), q.get("run", ""))
            return self.send(200 if t is not None else 404, {"log": t})
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/action":
            return self.send(404, {"error": "not found"})
        if not self.authorized():
            return self.send(403, {"error": "token が違います"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(min(n, 10000)) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self.send(400, {"error": "JSON が不正です"})
        ok, msg = self.app.do_action(body)
        return self.send(200 if ok else 409, {"ok": ok, "message": msg})


def main():
    p = argparse.ArgumentParser(description="実験ダッシュボード")
    p.add_argument("--dir", default="~/sim_ws", help="docker-compose.yml のある場所(既定 ~/sim_ws)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--lan", action="store_true", help="同じネットワークの他のPCからも開けるようにする")
    p.add_argument("--token", help="固定のtoken(省略すると起動ごとにランダム)")
    p.add_argument("--start-cmd", help="動作テスト用: start_all.sh の代わりに実行するコマンド")
    args = p.parse_args()
    app = App(args)
    if not (app.dir / "docker-compose.yml").exists():
        print(f"ERROR: {app.dir} に docker-compose.yml がありません(--dir で指定)")
        raise SystemExit(1)
    Handler.app = app
    server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), Handler)
    host = "<サーバ名>.local" if args.lan else "localhost"
    print(f"ダッシュボード: http://{host}:{args.port}/?token={app.token}")
    print("止めるには Ctrl-C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
