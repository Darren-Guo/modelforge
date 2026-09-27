"""训练任务管理：子进程执行生成的 train.py，按行采集 JSON 指标。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ..store import DATA_DIR, finish_training, register_model

# job_id -> job dict
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


class JobNotFound(Exception):
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

    proc = subprocess.Popen(
        cmd, cwd=str(code_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    job["proc"] = proc

    def pump() -> None:
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
        with _lock:
            if proc.returncode == 0 and job["status"] == "running":
                job["status"] = "done"
                _finalize(job)
            elif job["status"] == "running":
                job["status"] = "failed"
                job["events"].append({"type": "error", "message": f"训练进程退出码 {proc.returncode}"})
        finish_training(job_id, job["status"])

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
        if job is None:
            raise JobNotFound(job_id)
        return _public(job)


def list_jobs() -> list[dict[str, Any]]:
    with _lock:
        return [_public(j) for j in sorted(_jobs.values(), key=lambda j: j["job_id"], reverse=True)]


def iter_events(job_id: str, from_index: int = 0):
    """轮询式事件流（供 SSE 使用）。返回 (events, done)。"""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        events = job["events"][from_index:]
        done = job["status"] in ("done", "failed", "stopped")
        status = job["status"]
    return events, done, status


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
        if job is None:
            raise JobNotFound(job_id)
        proc = job.get("proc")
        job["status"] = "stopped"
        job["events"].append({"type": "error", "message": "任务被手动停止"})
    if proc and proc.poll() is None:
        proc.terminate()
    finish_training(job_id, "stopped")
