"""ModelForge 后端入口：FastAPI + CORS，聚合 /api 路由。"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .store import init_db, mark_stale_trainings

app = FastAPI(title="ModelForge Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本地单机工具
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    # 上个进程遗留的 running 任务其子进程已不存在，落为失败（终态）
    mark_stale_trainings()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
