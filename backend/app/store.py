"""SQLite 存储：自定义模块库 + 已训练模型注册表 + 训练任务记录。"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "modelforge.db"
MODULES_DIR = DATA_DIR / "modules"
RUNS_DIR = DATA_DIR / "runs"
GENERATED_DIR = DATA_DIR / "generated"
DATASETS_DIR = DATA_DIR / "datasets"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS modules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    topology TEXT NOT NULL,
    dataset TEXT,
    metrics TEXT DEFAULT '{}',
    run_dir TEXT NOT NULL,
    status TEXT DEFAULT 'trained',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trainings (
    job_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL,
    dataset TEXT,
    hyperparams TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    finished_at TEXT
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    for d in (DATA_DIR, MODULES_DIR, RUNS_DIR, GENERATED_DIR, DATASETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(_SCHEMA)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 自定义模块
# ---------------------------------------------------------------------------

def list_modules() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM modules ORDER BY updated_at DESC").fetchall()
    return [_module_row(r) for r in rows]


def get_module(module_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM modules WHERE id=?", (module_id,)).fetchone()
    return _module_row(row) if row else None


def save_module(module_id: str | None, name: str, description: str, data: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    with get_conn() as conn:
        if module_id and conn.execute("SELECT id FROM modules WHERE id=?", (module_id,)).fetchone():
            conn.execute(
                "UPDATE modules SET name=?, description=?, data=?, updated_at=? WHERE id=?",
                (name, description, json.dumps(data, ensure_ascii=False), now, module_id),
            )
            mid = module_id
        else:
            mid = module_id or new_id("mod")
            conn.execute(
                "INSERT INTO modules (id, name, description, data, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (mid, name, description, json.dumps(data, ensure_ascii=False), now, now),
            )
    # with 退出后统一回读：连接内提前读会拿到未提交的旧值
    return get_module(mid)  # type: ignore[return-value]


def delete_module(module_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM modules WHERE id=?", (module_id,))
        return cur.rowcount > 0


def _module_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "data": json.loads(row["data"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ---------------------------------------------------------------------------
# 已训练模型注册表
# ---------------------------------------------------------------------------

def register_model(model_id: str, name: str, topology: dict[str, Any], dataset: str,
                   metrics: dict[str, Any], run_dir: str) -> dict[str, Any]:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO models (model_id, name, topology, dataset, metrics, run_dir, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (model_id, name, json.dumps(topology, ensure_ascii=False), dataset,
             json.dumps(metrics, ensure_ascii=False), run_dir, "trained", _now()),
        )
    return get_model(model_id)  # type: ignore[return-value]


def list_models() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        out.append({
            "model_id": r["model_id"], "name": r["name"], "dataset": r["dataset"],
            "metrics": json.loads(r["metrics"]), "status": r["status"],
            "created_at": r["created_at"],
        })
    return out


def get_model(model_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        r = conn.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
    if not r:
        return None
    return {
        "model_id": r["model_id"], "name": r["name"], "topology": json.loads(r["topology"]),
        "dataset": r["dataset"], "metrics": json.loads(r["metrics"]), "run_dir": r["run_dir"],
        "status": r["status"], "created_at": r["created_at"],
    }


def delete_model(model_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM models WHERE model_id=?", (model_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# 训练任务
# ---------------------------------------------------------------------------

def record_training(job_id: str, model_id: str, dataset: str, hyperparams: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO trainings (job_id, model_id, status, dataset, hyperparams, created_at) VALUES (?,?,?,?,?,?)",
            (job_id, model_id, "running", dataset, json.dumps(hyperparams, ensure_ascii=False), _now()),
        )


def finish_training(job_id: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE trainings SET status=?, finished_at=? WHERE job_id=?",
                     (status, _now(), job_id))


def get_training(job_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM trainings WHERE job_id=?", (job_id,)).fetchone()
    return _training_row(row) if row else None


def list_trainings() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM trainings ORDER BY created_at DESC").fetchall()
    return [_training_row(r) for r in rows]


def mark_stale_trainings(status: str = "failed") -> int:
    """服务启动时调用：上个进程遗留的 running 行其子进程已不存在，落为终态。"""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE trainings SET status=?, finished_at=? WHERE status='running'",
            (status, _now()),
        )
        return cur.rowcount


def _training_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "model_id": row["model_id"],
        "status": row["status"],
        "dataset": row["dataset"],
        "hyperparams": json.loads(row["hyperparams"] or "{}"),
        "created_at": row["created_at"],
        "finished_at": row["finished_at"],
    }
