"""训练任务管理：子进程执行生成的 train.py，按行采集 JSON 指标。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ..store import (
    DATA_DIR, RUNS_DIR, finish_training, get_model, get_training, list_trainings,
    register_model,
)

# job_id -> job dict（仅存本进程的实时任务；SQLite trainings 表是跨重启的权威历史）
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


class JobNotFound(Exception):
    pass


class JobNotRunning(Exception):
    """任务存在但已结束（无法再停止）。"""

    pass


def start_job(job_id: str, model_id: str, run_dir: Path, dataset: str,
              hyperparams: dict[str, Any], topology: dict[str, Any], model_name: str) -> dict[str, Any]:
    code_dir = run_dir / "code"
    cmd = [
        sys.executable, str(code_dir / "train.py"),
        "--dataset", dataset,
        "--epochs", str(hyperparams.get("epochs", 3)),
        "--batch-size", str(hyperparams.get("batch_size", 32)),
        "--lr", str(hyperparams.get("lr", 1e-3)),
        "--num-samples", str(hyperparams.get("num_samples", 512)),
        "--out", str(run_dir),
        "--data-dir", str(DATA_DIR / "datasets"),
    ]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    job: dict[str, Any] = {
        "job_id": job_id,
        "model_id": model_id,
        "model_name": model_name,
        "dataset": dataset,
        "hyperparams": hyperparams,
        "topology": topology,
        "run_dir": str(run_dir),
        "status": "running",
        "events": [],           # 结构化事件（JSON 行）
        "raw": [],              # 原始输出行（人看）
        "progress": None,       # 最近一条 metric
    }
    with _lock:
        _jobs[job_id] = job

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(code_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except OSError as e:
        # 进程没起来：DB 行不能停在 running
        with _lock:
            job["status"] = "failed"
            job["events"].append({"type": "error", "message": f"训练进程启动失败: {e}"})
        finish_training(job_id, "failed")
        raise
    job["proc"] = proc

    def pump() -> None:
        final = "failed"
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                with _lock:
                    job["raw"].append(line)
                    if len(job["raw"]) > 5000:
                        job["raw"] = job["raw"][-2500:]
                try:
                    evt = json.loads(line)
                    with _lock:
                        job["events"].append(evt)
                        if evt.get("type") == "metric":
                            job["progress"] = evt
                except json.JSONDecodeError:
                    pass
            proc.wait()
            if proc.returncode == 0:
                final = "done"
                try:
                    _finalize(job)
                except Exception as e:  # 注册失败不能记为 done
                    final = "failed"
                    with _lock:
                        job["events"].append({"type": "error", "message": f"训练完成但模型注册失败: {e}"})
            else:
                with _lock:
                    job["events"].append({"type": "error", "message": f"训练进程退出码 {proc.returncode}"})
        except Exception as e:
            with _lock:
                job["events"].append({"type": "error", "message": f"训练日志采集异常终止: {e}"})
        finally:
            # 先落库、后改内存：观察到终态时 DB 必然已是终态，避免重启后遗留 running
            with _lock:
                if job["status"] == "stopped":
                    final = "stopped"  # 手动停止优先（stop_job 已写库，此处幂等）
            finish_training(job_id, final)
            with _lock:
                if job["status"] == "running":
                    job["status"] = final

    threading.Thread(target=pump, daemon=True).start()
    return get_job(job_id)


def _finalize(job: dict[str, Any]) -> None:
    """训练成功后写入模型注册表。"""
    metrics: dict[str, Any] = {}
    metrics_path = Path(job["run_dir"]) / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    register_model(
        job["model_id"], job["model_name"], job["topology"],
        job["dataset"], metrics, job["run_dir"],
    )


def get_job(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            return _public(job)
    row = get_training(job_id)  # 重启后的历史任务：以 DB 为准
    if row is None:
        raise JobNotFound(job_id)
    return _job_from_row(row)


def list_jobs() -> list[dict[str, Any]]:
    """DB 为权威历史（含重启前的任务），内存任务覆盖实时字段（events/progress）。"""
    with _lock:
        live = {j["job_id"]: _public(j) for j in _jobs.values()}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in list_trainings():
        seen.add(row["job_id"])
        out.append(live.get(row["job_id"]) or _job_from_row(row))
    # 未入库的实时任务（record_training 失败等异常情况）保底不丢
    out[:0] = [pub for jid, pub in live.items() if jid not in seen]
    return out


def iter_events(job_id: str, from_index: int = 0):
    """轮询式事件流（供 SSE 使用）。返回 (events, done, status)。"""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            events = job["events"][from_index:]
            done = job["status"] in ("done", "failed", "stopped")
            status = job["status"]
            return events, done, status
    row = get_training(job_id)
    if row is None:
        raise JobNotFound(job_id)
    # 重启后无实时事件：给一条说明后关闭流，而不是报「任务不存在」
    evt = {"type": "status", "message": "服务已重启，该任务的实时日志不可用（结果以任务列表为准）",
           "status": row["status"]}
    return ([evt] if from_index == 0 else []), True, row["status"]


def _job_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """从 DB 记录还原任务视图（无实时事件），与 _public 的字段保持一致。"""
    model = get_model(row["model_id"])
    return {
        "job_id": row["job_id"],
        "model_id": row["model_id"],
        "model_name": (model or {}).get("name") or row["model_id"],
        "dataset": row["dataset"] or "",
        "hyperparams": row["hyperparams"] or {},
        "status": row["status"],
        "progress": None,
        "events": [],
        "run_dir": (model or {}).get("run_dir") or str(RUNS_DIR / row["model_id"]),
    }


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "model_id": job["model_id"],
        "model_name": job["model_name"],
        "dataset": job["dataset"],
        "hyperparams": job["hyperparams"],
        "status": job["status"],
        "progress": job["progress"],
        "events": job["events"][-200:],
        "run_dir": job["run_dir"],
    }


def stop_job(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        proc = job.get("proc") if job is not None else None
        if job is not None:
            job["status"] = "stopped"
            job["events"].append({"type": "error", "message": "任务被手动停止"})
    if job is not None:
        if proc and proc.poll() is None:
            proc.terminate()
        finish_training(job_id, "stopped")
        return
    row = get_training(job_id)
    if row is None:
        raise JobNotFound(job_id)
    if row["status"] != "running":
        raise JobNotRunning(job_id)
    finish_training(job_id, "stopped")  # 遗留 running 行（子进程已不存在）：直接落库停止
