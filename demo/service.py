"""SQLite-backed demo orchestration around the unchanged research engine.

The server never sends orders. A RUNNING instance means a historical replay is
active in the UI. Ad-hoc experiments are kept separate from frozen run_001.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import uuid

import numpy as np
import pandas as pd

from quantlab.data import load_data
from quantlab.engine import Costs, backtest, features, training_score
from quantlab.search import optimize
from .performance import ledger_metrics
from .portfolio import backtest_portfolio
from .tuning import run_tuning


TABLES = ("instances", "tasks", "genomes", "backtests", "events")
TEMPLATES = [
    {"id": "sigmoid", "name": "自适应仓位", "description": "价格偏离与动量映射为目标仓位；支持参数搜索", "features": ["optimization", "backtest", "replay"]},
    {"id": "momentum", "name": "趋势动量", "description": "63 日动量决定持仓或现金；固定规则", "features": ["backtest", "replay"]},
    {"id": "half", "name": "均衡配置", "description": "50% 资产仓位，偏离 5 个百分点再平衡", "features": ["backtest", "replay"]},
    {"id": "buy_hold_reinvest", "name": "被动持有", "description": "整股买入持有，分红代理到账后复投", "features": ["backtest", "replay"]},
]


class Cancelled(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value, name: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须为数值")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须为数值") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} 必须在 {low:g} 至 {high:g} 之间")
    return value


def _integer(value, name: str, low: int, high: int) -> int:
    parsed = _number(value, name, low, high)
    if parsed != int(parsed):
        raise ValueError(f"{name} 必须为整数")
    return int(parsed)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()[:10]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class DemoService:
    def __init__(self, db_path: Path, root: Path):
        self.root = Path(root).resolve()
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.root / "data/manifest.json").read_text(encoding="utf-8"))
        self.synthetic_data = manifest.get("synthetic", False) is True
        self.datasets = {item["symbol"]: item for item in manifest["datasets"]}
        additional = self.root / "data/portfolio/manifest.json"
        if additional.is_file():
            for item in json.loads(additional.read_text(encoding="utf-8"))["datasets"]:
                self.datasets.setdefault(item["symbol"], item)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        for table in TABLES:
            self.db.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        self.db.commit()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quant-demo")
        self.cancel_events: dict[str, threading.Event] = {}
        self.frames: dict[str, pd.DataFrame] = {}
        self.replay_cache = {}
        self.closed = False
        self._recover()
        self._seed()

    def _get(self, table: str, item_id: str) -> dict:
        if table not in TABLES:
            raise ValueError("Unknown record type")
        with self.lock:
            row = self.db.execute(f"SELECT payload FROM {table} WHERE id=?", (str(item_id),)).fetchone()
        if row is None:
            raise KeyError(f"{table}: {item_id}")
        return json.loads(row[0])

    def _all(self, table: str, limit: int = 200) -> list[dict]:
        if table not in TABLES:
            raise ValueError("Unknown record type")
        with self.lock:
            rows = self.db.execute(f"SELECT payload FROM {table} ORDER BY created_at DESC,id DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _put(self, table: str, item: dict, *, commit: bool = True) -> dict:
        if table not in TABLES:
            raise ValueError("Unknown record type")
        item = _jsonable(dict(item))
        item.setdefault("created_at", now())
        item["updated_at"] = now()
        payload = json.dumps(item, ensure_ascii=False, allow_nan=False)
        with self.lock:
            self.db.execute(f"INSERT INTO {table}(id,payload,created_at,updated_at) VALUES(?,?,?,?) "
                            "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at",
                            (item["id"], payload, item["created_at"], item["updated_at"]))
            if commit:
                self.db.commit()
        return item

    def _event(self, kind: str, message: str, payload: dict | None = None):
        return self._put("events", {"id": "evt-" + uuid.uuid4().hex[:16], "kind": kind,
                                    "message": message, "payload": payload or {}, "created_at": now()})

    def _recover(self):
        for table in ("tasks", "backtests"):
            for row in self._all(table, 10000):
                if row["status"] in ("pending", "queued", "running"):
                    row.update(status="interrupted", error="本地服务重启，未完成任务已中断；请创建新任务", finished_at=now())
                    self._put(table, row)

    def _seed(self):
        with self.lock:
            if not self._all("genomes"):
                selection = json.loads((self.root / "results/run_001/selection.json").read_text(encoding="utf-8"))
                winner = selection["winner"]
                for row in selection["candidates"]:
                    chosen = row["method"] == winner["method"] and row["seed"] == winner["seed"]
                    self._put("genomes", {"id": f"genome-frozen-{row['method']}-{row['seed']}",
                        "name": f"冻结研究 · {row['method'].upper()} / {row['seed']}",
                        "role": "champion" if chosen else "challenger", "params": row["params"],
                        "method": row["method"], "seed": row["seed"], "training_score": row["training_score"],
                        "validation": row["validation"], "source": "frozen_run_001", "symbol": "SPY",
                        "created_at": selection["selected_utc"]})
            champion = self._champion()
            if not self._all("instances"):
                for symbol, state in (("SPY", "running"), ("GLD", "paused")):
                    self._put("instances", {"id": f"inst-{symbol.lower()}", "name": f"{symbol} 仓位研究",
                        "symbol": symbol, "template": "sigmoid", "status": state,
                        "initial_cash": self.config["initial_cash"], "capital": self.config["initial_cash"],
                        "monthly_budget": 0, "genome_id": champion["id"], "mode": "historical_replay"})
                self._event("initialized", "已载入真实研究参数与两个历史回放实例")

    def _champion(self) -> dict:
        return next((g for g in self._all("genomes", 10000) if g["role"] == "champion"), None) or {}

    def _frame(self, symbol: str) -> pd.DataFrame:
        dataset = self.datasets.get(symbol)
        if dataset is None:
            raise ValueError(f"没有 {symbol} 的本地行情数据")
        with self.lock:
            if symbol not in self.frames:
                path = (self.root / dataset["csv_path"]).resolve()
                if not path.is_relative_to(self.root / "data"):
                    raise ValueError("行情文件必须存放在 data 目录中")
                if hashlib.sha256(path.read_bytes()).hexdigest() != dataset["csv_sha256"]:
                    raise ValueError(f"{symbol} 的行情文件与保存的数据校验值不一致")
                self.frames[symbol] = load_data(path)
            return self.frames[symbol]

    def bootstrap(self) -> dict:
        instances, jobs = self._all("instances"), self._all("tasks")
        tasks = [row for row in jobs if row.get("type") != "joint_tuning"]
        tuning_runs = [{key: value for key, value in row.items() if key not in ("histories", "history", "finalists")}
                       for row in jobs if row.get("type") == "joint_tuning"]
        # Lists contain summaries; full chart/trade payloads are fetched by ID.
        backtests = [{k: v for k, v in row.items() if k not in ("chart", "daily", "bars", "trades", "fills")}
                     for row in self._all("backtests", 50)]
        datasets = [{key: d[key] for key in ("symbol", "rows", "first_session", "last_session", "name") if key in d}
                    for d in self.datasets.values()]
        champion = self._champion()
        return {"instances": instances, "genomes": self._all("genomes"), "tasks": tasks,
            "tuning_runs": tuning_runs, "backtests": backtests,
            "current_genome": champion, "templates": TEMPLATES, "events": self._all("events", 50),
            "overview": {"instance_count": len(instances), "running_instances": sum(i["status"] == "running" for i in instances),
                "current_genome_id": champion.get("id"), "total_initial_cash": sum(i["initial_cash"] for i in instances),
                "completed_optimizations": sum(t["status"] == "completed" for t in jobs),
                "completed_backtests": sum(t["status"] == "completed" for t in backtests),
                "datasets": datasets, "mode": "historical_replay"},
            "protocol": self.config, "mode": "historical_replay", "broker_connected": False,
            "synthetic_data": self.synthetic_data,
            "notice": "本地历史研究演示；运行中表示历史回放，不连接券商、不提交真实订单。交互实验不会改写冻结研究结果。"}

    def _params(self, value: dict) -> dict:
        if not isinstance(value, dict) or set(value) != set(self.config["bounds"]):
            raise ValueError("参数必须完整包含 mean_reversion、momentum 和 band")
        return {name: _number(value[name], name, *bounds) for name, bounds in self.config["bounds"].items()}

    def _instance_fields(self, body: dict, existing: dict | None = None) -> dict:
        if not isinstance(body, dict):
            raise ValueError("请求体必须为对象")
        row = dict(existing or {})
        if "name" in body or not existing:
            name = body.get("name", "新研究实例")
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
                raise ValueError("实例名称须为 1 至 80 个字符")
            row["name"] = name.strip()
        symbol = body.get("symbol", row.get("symbol", "SPY"))
        if symbol not in ("SPY", "GLD"):
            raise ValueError("标的只支持 SPY 与 GLD")
        row["symbol"] = symbol
        template = body.get("template", body.get("strategy", row.get("template", "sigmoid")))
        if not isinstance(template, str) or template not in {t["id"] for t in TEMPLATES}:
            raise ValueError("不支持该策略模板")
        row["template"] = template
        capital = body.get("initial_cash", body.get("capital", row.get("initial_cash", 100000)))
        row["initial_cash"] = row["capital"] = _number(capital, "初始资金", 1000, 10000000)
        status = body.get("status", row.get("status", "paused"))
        if status not in ("running", "paused", "stopped"):
            raise ValueError("实例状态应为 running、paused 或 stopped")
        row["status"] = status
        if "monthly_budget" in body and _number(body["monthly_budget"], "每月注资", 0, 10000000) != 0:
            raise ValueError("当前研究引擎采用固定初始资金，每月注资请设为 0")
        row["monthly_budget"] = 0
        genome_id = body.get("genome_id", row.get("genome_id", self._champion().get("id")))
        if genome_id:
            self._get("genomes", genome_id)
        row["genome_id"] = genome_id
        row["mode"] = "historical_replay"
        return row

    def create_instance(self, body: dict) -> dict:
        with self.lock:
            if len(self._all("instances", 100)) >= 30:
                raise RuntimeError("演示最多保留 30 个实例")
            row = self._instance_fields(body)
            row["id"] = "inst-" + uuid.uuid4().hex[:12]
            row = self._put("instances", row)
            self._event("instance_created", f"已创建实例 {row['name']}", {"instance_id": row["id"]})
            return row

    def update_instance(self, item_id: str, body: dict) -> dict:
        with self.lock:
            row = self._instance_fields(body, self._get("instances", item_id))
            row = self._put("instances", row)
            self.replay_cache.clear()
            self._event("instance_updated", f"实例 {row['name']} 已更新，状态 {row['status']}", {"instance_id": row["id"]})
            return row

    def _worker_capacity(self):
        if self.closed:
            raise RuntimeError("服务已关闭")
        active = sum(r["status"] in ("pending", "queued", "running") for table in ("tasks", "backtests") for r in self._all(table))
        if active >= 2:
            raise RuntimeError("已有两个计算任务运行，请等待完成或取消优化任务")

    def start_optimization(self, body: dict) -> dict:
        if not isinstance(body, dict):
            raise ValueError("请求体必须为对象")
        method = body.get("method", "ga")
        if method not in ("ga", "random"):
            raise ValueError("搜索方法应为 ga 或 random")
        if body.get("symbol", "SPY") != "SPY":
            raise ValueError("优化固定使用 SPY 训练数据；GLD 保留为不调参的迁移检验")
        instance = self._get("instances", body["instance_id"]) if body.get("instance_id") else None
        if instance and instance["template"] != "sigmoid":
            raise ValueError("只有自适应仓位模板支持参数优化")
        population = _integer(body.get("population", 32), "预算批量", 16, 128)
        generations = _integer(body.get("generations", 4), "预算轮数", 1, 32)
        seed = _integer(body.get("seed", 11), "随机种子", 0, 2147483647)
        budget = min(512, population * generations)
        mode = body.get("spawn_mode", body.get("inheritance", "current"))
        if mode == "inherit":
            mode = "current"
        if mode not in ("current", "random", "manual"):
            raise ValueError("参数起点应为 current、random 或 manual")
        full_bounds = self.config["bounds"]
        source_genome_id = body.get("genome_id", instance.get("genome_id") if instance else None)
        inherited = self._get("genomes", source_genome_id) if source_genome_id else self._champion()
        anchor = None if mode == "random" else self._params(body.get("params", {}) if mode == "manual" else inherited["params"])
        bounds = {k: list(pair) for k, pair in full_bounds.items()}
        if anchor:
            for key, (lo, hi) in full_bounds.items():
                radius = (hi - lo) * .25
                bounds[key] = [max(lo, anchor[key] - radius), min(hi, anchor[key] + radius)]
        row = {"id": "opt-" + uuid.uuid4().hex[:12], "status": "queued", "method": method,
            "population": population, "generations": generations, "algorithm_population": 32 if method == "ga" else None,
            "seed": seed, "spawn_mode": mode, "anchor_params": anchor, "effective_bounds": bounds,
            "instance_id": instance.get("id") if instance else None, "source_genome_id": inherited.get("id") if mode == "current" else None,
            "budget": budget, "progress": {"completed": 0, "total": budget}, "completed": 0, "total": budget,
            "best_score": None, "params": anchor, "history": [], "symbol": "SPY", "train": self.config["train"],
            "validation_period": self.config["validation"], "test_used": False,
            "budget_note": "预算批量×轮数决定唯一候选评估次数，上限 512；原 GA 算法实际种群固定为 32。",
            "inheritance_note": "继承/手动模式在起点周围各维完整范围的 ±25% 区域搜索，并裁剪至合法边界；不声称起点被强制纳入首代。随机模式探索完整边界。"}
        with self.lock:
            self._worker_capacity()
            row = self._put("tasks", row)
            token = self.cancel_events[row["id"]] = threading.Event()
            self._event("optimization_started", f"启动 {method.upper()} 参数搜索，预算 {budget}", {"task_id": row["id"]})
            self.executor.submit(self._optimization_worker, row["id"], token)
        return self.get_task(row["id"])

    def start_joint_tuning(self, body: dict) -> dict:
        if set(body) - {"budget_per_run"}:
            raise ValueError("联合调参仅接受每轮搜索预算，标的与训练、验证区间固定")
        budget = _integer(body.get("budget_per_run", 256), "每轮搜索预算", 64, 1024)
        config = self.config
        protocol = {key: config[key] for key in ("train", "training_folds", "validation", "costs",
                                                 "initial_cash", "bounds", "seeds")}
        protocol.update(budget_per_run=budget, symbol="SPY", data_sha256=self.datasets["SPY"]["csv_sha256"],
                        objective="mean of three net Sharpe differences vs half minus 0.25 times their population standard deviation",
                        ranking="highest validation net Sharpe, then lower drawdown, turnover, method and seed")
        total = len(protocol["seeds"]) * 2 * budget + 10
        with self.lock:
            self._worker_capacity()
            if any(row.get("type") == "joint_tuning" and row["status"] in ("queued", "running", "pending")
                   for row in self._all("tasks")):
                raise RuntimeError("已有三参数联合调参任务在运行")
            incumbent = self._champion()
            row = self._put("tasks", {"id": "tune-" + uuid.uuid4().hex[:12], "type": "joint_tuning",
                "symbol": "SPY", "status": "queued", "budget_per_run": budget, "seeds": protocol["seeds"],
                "train": protocol["train"], "validation_period": protocol["validation"], "test_used": False,
                "protocol": protocol, "incumbent_genome_id": incumbent["id"], "incumbent_params": incumbent["params"],
                "progress": {"completed": 0, "total": total, "phase": "queued"}})
            token = self.cancel_events[row["id"]] = threading.Event()
            self._event("tuning_started", "启动三参数联合调参", {"task_id": row["id"]})
            self.executor.submit(self._joint_tuning_worker, row["id"], token)
        return self.get_task(row["id"])

    def _joint_tuning_worker(self, task_id: str, token: threading.Event):
        try:
            with self.lock:
                state = self.get_task(task_id)
                if token.is_set() or state["status"] == "cancelled":
                    raise Cancelled()
                state.update(status="running", started_at=now())
                state["progress"]["phase"] = "train"
                self._put("tasks", state)
            protocol = state["protocol"]
            frame = self._frame(state["symbol"])
            train = frame.loc[:protocol["train"][1]].copy()
            validation = frame.loc[:protocol["validation"][1]].copy()

            last_progress = None

            def on_progress(completed, total, method, seed, best_score, phase):
                nonlocal last_progress
                if token.is_set():
                    raise Cancelled()
                scope = (phase, method, seed)
                if scope == last_progress and completed % 12 and completed != total:
                    return
                with self.lock:
                    row = self.get_task(task_id)
                    if token.is_set() or row["status"] == "cancelled":
                        raise Cancelled()
                    row["progress"] = {"completed": completed, "total": total, "phase": phase,
                                       "method": method, "seed": seed, "best_score": best_score}
                    self._put("tasks", row)
                    last_progress = scope

            result = run_tuning(train, validation, protocol, state["incumbent_params"], on_progress)
            winner = result["winner"]
            with self.lock:
                state = self.get_task(task_id)
                if token.is_set() or state["status"] == "cancelled":
                    raise Cancelled()
                genome_id = "genome-" + uuid.uuid4().hex[:12]
                self._put("genomes", {"id": genome_id, "name": f"联合调参 · {winner['method'].upper()} / {winner['seed']}",
                    "role": "challenger", "params": winner["params"], "method": winner["method"], "seed": winner["seed"],
                    "training_score": winner["training_score"], "validation": winner["validation"],
                    "source": "joint_tuning", "task_id": task_id, "symbol": state["symbol"], "test_used": False}, commit=False)
                total = state["progress"]["total"]
                state.update(result, status="completed", finished_at=now(), genome_id=genome_id,
                             progress={"completed": total, "total": total, "phase": "completed"})
                self._put("tasks", state, commit=False)
                self.db.commit()
                self._event("tuning_completed", "联合调参完成，验证期首选参数已存入参数库",
                            {"task_id": task_id, "genome_id": genome_id})
        except Cancelled:
            with self.lock:
                state = self.get_task(task_id)
                if state["status"] != "cancelled":
                    state.update(status="cancelled", finished_at=now())
                    self._put("tasks", state)
        except Exception as exc:
            with self.lock:
                self.db.rollback()
                state = self.get_task(task_id)
                if state["status"] not in ("cancelled", "completed"):
                    state.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {str(exc)[:250]}")
                    self._put("tasks", state)
        finally:
            with self.lock:
                self.cancel_events.pop(task_id, None)

    def _optimization_worker(self, task_id: str, token: threading.Event):
        try:
            with self.lock:
                row = self._get("tasks", task_id)
                if row["status"] == "cancelled" or token.is_set():
                    return
                row.update(status="running", started_at=now())
                self._put("tasks", row)
            # Physically exclude validation and test rows from optimizer input.
            train = self._frame("SPY").loc[:self.config["train"][1]].copy()
            train_features = features(train)
            costs = Costs(**self.config["costs"])

            def objective(params):
                if token.is_set():
                    raise Cancelled()
                score = float(training_score(train, train_features, params, self.config["training_folds"], costs,
                                             self.config["initial_cash"]))
                if token.is_set():
                    raise Cancelled()
                with self.lock:
                    state = self._get("tasks", task_id)
                    if state["status"] == "cancelled" or token.is_set():
                        raise Cancelled()
                    completed = state["completed"] + 1
                    best = state["best_score"]
                    if best is None or score > best:
                        state["best_score"], state["params"] = score, dict(params)
                    state["completed"] = completed
                    state["progress"] = {"completed": completed, "total": state["total"]}
                    state["history"].append({"evaluation": completed, "best_score": state["best_score"], "score": score})
                    self._put("tasks", state)
                return score

            result = optimize(objective, row["effective_bounds"], method=row["method"], budget=row["budget"], seed=row["seed"])
            if token.is_set():
                raise Cancelled()
            validation = self._frame("SPY").loc[:self.config["validation"][1]].copy()
            metrics = backtest(validation, "sigmoid", result["best_params"], *self.config["validation"], costs,
                               self.config["initial_cash"], features(validation)).metrics
            with self.lock:
                state = self._get("tasks", task_id)
                if token.is_set() or state["status"] == "cancelled":
                    raise Cancelled()
                genome = {"id": "genome-" + uuid.uuid4().hex[:12], "name": f"候选 · {row['method'].upper()} / {row['seed']}",
                    "role": "challenger", "params": result["best_params"], "method": row["method"], "seed": row["seed"],
                    "training_score": result["best_score"], "validation": metrics, "source": "interactive_demo",
                    "task_id": task_id, "symbol": "SPY", "effective_bounds": row["effective_bounds"], "test_used": False}
                self._put("genomes", genome, commit=False)
                state.update(status="completed", params=result["best_params"], best_score=result["best_score"],
                             completed=result["evaluations"], progress={"completed": result["evaluations"], "total": state["total"]},
                             evaluations=result["evaluations"], cache_hits=result["cache_hits"], requests=result["requests"],
                             validation=metrics, genome_id=genome["id"], finished_at=now())
                self._put("tasks", state, commit=False)
                self.db.commit()
                self._event("optimization_completed", "参数搜索完成，候选待审批；未使用最终测试数据", {"task_id": task_id, "genome_id": genome["id"]})
        except Cancelled:
            with self.lock:
                state = self._get("tasks", task_id)
                if state["status"] != "completed":
                    state.update(status="cancelled", finished_at=now())
                    self._put("tasks", state)
        except Exception as exc:
            with self.lock:
                self.db.rollback()
                state = self._get("tasks", task_id)
                if state["status"] != "cancelled":
                    state.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:250]}", finished_at=now())
                    self._put("tasks", state)
                    self._event("optimization_failed", "参数搜索失败", {"task_id": task_id, "error": state["error"]})
        finally:
            with self.lock:
                self.cancel_events.pop(task_id, None)

    def get_task(self, task_id: str) -> dict:
        return self._get("tasks", task_id)

    def cancel_task(self, task_id: str) -> dict:
        with self.lock:
            row = self._get("tasks", task_id)
            if row["status"] not in ("queued", "pending", "running"):
                return row
            token = self.cancel_events.get(task_id)
            if token:
                token.set()
            row.update(status="cancelled", finished_at=now())
            row = self._put("tasks", row)
            self._event("optimization_cancelled", "参数搜索已取消，未完成结果不会晋升", {"task_id": task_id})
            return row

    def promote_genome(self, genome_id: str, instance_id: str | None = None) -> dict:
        with self.lock:
            genome = self._get("genomes", genome_id)
            instance = self._get("instances", instance_id) if instance_id else None
            if instance and instance["template"] != "sigmoid":
                raise ValueError("只有自适应仓位实例可以绑定参数版本")
            previous_id = self._champion().get("id")
            try:
                for row in self._all("genomes", 10000):
                    if row["role"] == "champion" and row["id"] != genome_id:
                        row["role"] = "retired"
                        self._put("genomes", row, commit=False)
                genome.update(role="champion", promoted_at=now())
                genome = self._put("genomes", genome, commit=False)
                if instance:
                    instance["genome_id"] = genome_id
                    self._put("instances", instance, commit=False)
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            self.replay_cache.clear()
            self._event("genome_promoted", "候选已设为当前参数；既有实例仅在显式应用时换版", {"genome_id": genome_id, "previous_genome_id": previous_id, "instance_id": instance_id})
            return {**genome, "applied_instance_id": instance_id}

    def _backtest_request(self, body: dict) -> dict:
        if not isinstance(body, dict):
            raise ValueError("请求体必须为对象")
        allocations = self._portfolio_allocations(body) if "allocations" in body else None
        instance = self._get("instances", body["instance_id"]) if body.get("instance_id") else {}
        if allocations:
            symbol = " + ".join(allocations)
        else:
            symbol = body.get("symbol", instance.get("symbol", "SPY"))
            if not isinstance(symbol, str) or symbol not in self.datasets:
                raise ValueError("请从已保存的行情数据中选择标的")
        strategy = body.get("strategy", body.get("template", instance.get("template", "sigmoid")))
        strategy_ids = {"sigmoid", "half"} if allocations else {t["id"] for t in TEMPLATES}
        if not isinstance(strategy, str) or strategy not in strategy_ids:
            raise ValueError("不支持该策略模板")
        genome_id = body.get("genome_id", instance.get("genome_id", self._champion().get("id")))
        genome = self._get("genomes", genome_id) if genome_id else self._champion()
        params = self._params(body["params"]) if "params" in body else dict(genome["params"])
        start, end = body.get("start", self.config["test"][0]), body.get("end", self.config["test"][1])
        try:
            parsed_start, parsed_end = date.fromisoformat(start), date.fromisoformat(end)
        except (TypeError, ValueError):
            raise ValueError("日期格式应为 YYYY-MM-DD") from None
        if parsed_start > parsed_end or parsed_start < date(2004, 1, 1):
            raise ValueError("开始日期不得早于 2004-01-01 或晚于结束日期")
        if allocations:
            if parsed_end > min(date.fromisoformat(self.datasets[ticker]["last_session"]) for ticker in allocations):
                raise ValueError("结束日期超过某个标的的行情覆盖范围")
            sessions = None
            for ticker in allocations:
                dates = self._frame(ticker).loc[start:end].index
                sessions = dates if sessions is None else sessions.intersection(dates)
            if sessions is None or sessions.empty:
                raise ValueError("所选日期区间内没有共同交易日")
        else:
            if parsed_end > date.fromisoformat(self.datasets[symbol]["last_session"]):
                raise ValueError("结束日期超过该标的的行情覆盖范围")
            if self._frame(symbol).loc[start:end].empty:
                raise ValueError("所选日期区间没有可用交易日")
        amount = _number(body.get("initial_cash", body.get("capital", instance.get("initial_cash", 100000))), "初始资金", 1000, 10000000)
        costs_body = body.get("costs", {})
        if not isinstance(costs_body, dict):
            raise ValueError("costs 必须为对象")
        costs = {"commission_bps": _number(body.get("commission_bps", costs_body.get("commission_bps", self.config["costs"]["commission_bps"])), "佣金 bps", 0, 100),
                 "slippage_bps": _number(body.get("slippage_bps", costs_body.get("slippage_bps", self.config["costs"]["slippage_bps"])), "滑点 bps", 0, 200),
                 "dividend_cash_lag_days": self.config["costs"]["dividend_cash_lag_days"]}
        request = {"symbol": symbol, "strategy": strategy, "genome_id": None if "params" in body else genome_id,
                   "params": params, "start": start, "end": end, "initial_cash": amount, "costs": costs,
                   "instance_id": instance.get("id"), "mode": "historical_replay", "observed_history": True,
                   "notice": "已知历史数据的交互实验，不是新的独立样本外检验，也不提交真实交易。"}
        if allocations:
            request["allocations"] = allocations
            request["data_sha256"] = {ticker: self.datasets[ticker]["csv_sha256"] for ticker in allocations}
        return request

    def _portfolio_allocations(self, body: dict) -> dict[str, float]:
        if body.get("instance_id") or "symbol" in body:
            raise ValueError("组合回测请按标的提交预算比例")
        raw = body["allocations"]
        if not isinstance(raw, dict) or not 2 <= len(raw) <= 20:
            raise ValueError("请选择 2 至 20 个标的，并填写每个标的的预算比例")
        if any(ticker not in self.datasets for ticker in raw):
            raise ValueError("请从已保存的行情数据中选择标的")
        weights = {ticker: _number(raw[ticker], f"{ticker} 预算比例", 0, 1)
                   for ticker in sorted(raw)}
        total = math.fsum(weights.values())
        if any(weight <= 0 for weight in weights.values()) or not math.isclose(total, 1, rel_tol=0, abs_tol=1e-8):
            raise ValueError("每个标的的预算比例必须大于 0，且合计为 100%")
        return {ticker: weight / total for ticker, weight in weights.items()}

    def start_backtest(self, body: dict) -> dict:
        request = self._backtest_request(body)
        with self.lock:
            self._worker_capacity()
            row = self._put("backtests", {"id": "bt-" + uuid.uuid4().hex[:12], "status": "queued", **request})
            self._event("backtest_started", f"启动 {row['symbol']} 历史回测", {"backtest_id": row["id"]})
            self.executor.submit(self._backtest_worker, row["id"])
            return self.get_backtest(row["id"])

    def _compute_backtest(self, request: dict, *, full_replay: bool = False) -> dict:
        if request.get("allocations"):
            return self._compute_portfolio_backtest(request)
        frame = self._frame(request["symbol"])
        feat = features(frame)
        costs = Costs(**request["costs"])
        args = (request["start"], request["end"], costs, request["initial_cash"], feat)
        selected = backtest(frame, request["strategy"], request["params"], *args)
        passive = backtest(frame, "buy_hold_reinvest", {}, *args)
        half = backtest(frame, "half", {}, *args)
        data = selected.daily.copy()
        data["benchmark"] = passive.daily.equity
        data["half"] = half.daily.equity
        eq = data.equity.to_numpy(float)
        data["drawdown"] = eq / np.maximum.accumulate(np.r_[request["initial_cash"], eq])[1:] - 1
        points = np.arange(len(data)) if full_replay else np.unique(np.linspace(0, len(data) - 1, min(400, len(data)), dtype=int))
        sampled = data.iloc[points].reset_index()
        chart = _jsonable(sampled.to_dict(orient="records"))
        trades = _jsonable(selected.trades.to_dict(orient="records"))
        prices = frame.loc[data.index].iloc[points].reset_index()
        bars = _jsonable(prices[["date", "open", "high", "low", "close", "volume"]].to_dict(orient="records"))
        last = data.iloc[-1]
        portfolio = {"date": str(data.index[-1].date()), "equity": float(last.equity), "cash": float(last.cash),
            "shares": int(last.shares), "receivable": float(last.receivable), "price": float(last.close),
            "market_value": float(last.shares * last.close), "exposure": float(last.exposure),
            "asset_pct": float(last.exposure), "cash_pct": float(last.cash / last.equity)}
        metrics = ledger_metrics(selected, half, request["initial_cash"])
        return {"metrics": metrics, "summary": metrics,
                "baseline_metrics": {"buy_hold_reinvest": ledger_metrics(passive, half, request["initial_cash"]),
                                     "half": ledger_metrics(half, half, request["initial_cash"])},
                "chart": chart, "daily": chart, "bars": bars, "trades": trades, "fills": trades,
                "trade_count": len(trades), "chart_total_sessions": len(data), "chart_sampled": len(data) > len(points),
                "portfolio": portfolio, "risk": {"max_drawdown": selected.metrics["max_drawdown"],
                    "cash_nonnegative": bool((data.cash >= -1e-6).all()), "leverage": False, "mode": "historical_replay"}}

    def _compute_portfolio_backtest(self, request: dict) -> dict:
        weights = request["allocations"]
        frames = {ticker: self._frame(ticker) for ticker in weights}
        costs = Costs(**request["costs"])
        runs = {
            strategy: backtest_portfolio(frames, weights, strategy, request["params"],
                                         request["start"], request["end"], costs, request["initial_cash"])
            for strategy in ("sigmoid", "half")
        }
        selected = runs[request["strategy"]]
        data = selected.daily
        last = data.iloc[-1]
        points = np.unique(np.linspace(0, len(data) - 1, min(400, len(data)), dtype=int))
        chart = data[["equity", "cash", "exposure"]].copy()
        for strategy, run in runs.items():
            chart[strategy] = run.daily.equity
        trades = _jsonable(selected.trades.to_dict(orient="records"))
        metrics_by_strategy = {strategy: ledger_metrics(run, runs["half"], request["initial_cash"])
                               for strategy, run in runs.items()}
        metrics = metrics_by_strategy[request["strategy"]]
        return {
            "metrics": metrics,
            "summary": metrics,
            "baseline_metrics": metrics_by_strategy,
            "chart": _jsonable(chart.iloc[points].reset_index().to_dict(orient="records")),
            "chart_total_sessions": len(data),
            "chart_sampled": len(data) > len(points),
            "trades": trades,
            "fills": trades,
            "trade_count": len(trades),
            "portfolio": _jsonable({"date": data.index[-1], "equity": last.equity, "cash": last.cash,
                                    "receivable": last.receivable, "holdings": last.holdings}),
            "risk": {"max_drawdown": selected.metrics["max_drawdown"],
                     "cash_nonnegative": bool((data.cash >= -1e-6).all()), "leverage": False,
                     "mode": "historical_replay"},
        }

    def _backtest_worker(self, backtest_id: str):
        try:
            with self.lock:
                row = self._get("backtests", backtest_id)
                row.update(status="running", started_at=now())
                self._put("backtests", row)
            result = self._compute_backtest(row)
            with self.lock:
                row = self._get("backtests", backtest_id)
                row.update(result, status="completed", finished_at=now())
                self._put("backtests", row)
                self._event("backtest_completed", f"{row['symbol']} 历史回测完成", {"backtest_id": backtest_id})
        except Exception as exc:
            with self.lock:
                row = self._get("backtests", backtest_id)
                row.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:250]}", finished_at=now())
                self._put("backtests", row)

    def get_backtest(self, backtest_id: str) -> dict:
        return self._get("backtests", backtest_id)

    def get_replay(self, instance_id: str) -> dict:
        instance = self._get("instances", instance_id)
        request = self._backtest_request({"instance_id": instance_id})
        cache_key = (instance_id, instance["updated_at"])
        with self.lock:
            cached = self.replay_cache.get(cache_key)
        if cached is None:
            computed = self._compute_backtest(request, full_replay=True)
            with self.lock:
                self.replay_cache[cache_key] = computed
        else:
            computed = cached
        return {**request, **computed, "instance": instance, "status": instance["status"],
                "events": [e for e in self._all("events", 50) if e["payload"].get("instance_id") == instance_id],
                "replay": {"source": "historical", "state": instance["status"], "can_send_orders": False,
                           "start": request["start"], "end": request["end"], "deterministic": True}}

    def close(self, wait: bool = True):
        with self.lock:
            self.closed = True
            for token in self.cancel_events.values():
                token.set()
        self.executor.shutdown(wait=wait, cancel_futures=True)
        if wait:
            with self.lock:
                self.db.close()
