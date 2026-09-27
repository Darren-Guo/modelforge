"""FastAPI 路由：算子注册表 / 图校验 / 代码生成 / 自定义模块 / 训练 / 推理。"""
from __future__ import annotations

import asyncio
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..codegen.generator import CodegenError, generate_package, node_snippet
from ..ir.schema import Graph, ModuleDef, Port
from ..ir.validate import validate_graph
from ..registry.ops import OPS, AttrDef, OpDef
from ..store import (
    delete_module, delete_model, get_model, get_module, list_models, list_modules,
    list_trainings, new_id, record_training, save_module, GENERATED_DIR, RUNS_DIR,
)
from ..training import infer
from ..training.jobs import JobNotFound, get_job, iter_events, list_jobs, start_job, stop_job

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# 算子注册表
# ---------------------------------------------------------------------------

def _attr_json(a: AttrDef) -> dict[str, Any]:
    return {"name": a.name, "type": a.type, "default": a.default, "label": a.label,
            "min": a.min, "max": a.max, "choices": a.choices, "help": a.help}


def _op_json(o: OpDef) -> dict[str, Any]:
    return {
        "op": o.op, "group": o.group, "label": o.label, "doc": o.doc,
        "attrs": [_attr_json(a) for a in o.attrs],
        "inputs": [{"name": p.name, "dtype": p.dtype, "label": p.label} for p in o.inputs],
        "outputs": [{"name": p.name, "dtype": p.dtype, "label": p.label} for p in o.outputs],
    }


@router.get("/ops")
def get_ops() -> list[dict[str, Any]]:
    return [_op_json(o) for o in OPS]


# ---------------------------------------------------------------------------
# 图校验 / 代码生成
# ---------------------------------------------------------------------------

class GraphBody(BaseModel):
    graph: Graph


class CodegenBody(BaseModel):
    graph: Graph
    save_as: str | None = None  # 可选：生成后保存到 generated/{save_as}


class SnippetBody(BaseModel):
    graph: Graph
    node_id: str


def _modules_map() -> dict[str, ModuleDef]:
    out: dict[str, ModuleDef] = {}
    for m in list_modules():
        try:
            out[m["id"]] = ModuleDef.model_validate(m["data"])
        except Exception:
            continue
    return out


@router.post("/validate")
def api_validate(body: GraphBody) -> dict[str, Any]:
    return validate_graph(body.graph, _modules_map())


@router.post("/codegen")
def api_codegen(body: CodegenBody) -> dict[str, Any]:
    try:
        files = generate_package(body.graph, _modules_map())
    except CodegenError as e:
        raise HTTPException(status_code=400, detail=e.report)
    folder = re.sub(r"[^\w\-]", "_", body.save_as or body.graph.model.name or "model")
    target = GENERATED_DIR / folder
    target.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (target / name).write_text(content, encoding="utf-8")
    return {"files": files, "folder": str(target)}


@router.post("/codegen/snippet")
def api_snippet(body: SnippetBody) -> dict[str, str]:
    try:
        return node_snippet(body.graph, body.node_id, _modules_map())
    except CodegenError as e:
        raise HTTPException(status_code=400, detail=e.report)


@router.get("/codegen/download/{folder}")
def api_download(folder: str) -> StreamingResponse:
    target = GENERATED_DIR / re.sub(r"[^\w\-]", "_", folder)
    if not target.exists():
        raise HTTPException(status_code=404, detail="生成目录不存在")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in target.iterdir():
            if f.is_file():
                z.write(f, f.name)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename={folder}.zip"})


# ---------------------------------------------------------------------------
# 自定义模块
# ---------------------------------------------------------------------------

class ModuleBody(BaseModel):
    id: str | None = None
    name: str
    description: str = ""
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    graph: Graph


@router.get("/modules")
def api_list_modules() -> list[dict[str, Any]]:
    return [
        {"id": m["id"], "name": m["name"], "description": m["description"],
         "inputs": m["data"].get("inputs", []), "outputs": m["data"].get("outputs", []),
         "node_count": len(m["data"].get("graph", {}).get("nodes", [])),
         "created_at": m["created_at"], "updated_at": m["updated_at"]}
        for m in list_modules()
    ]


@router.get("/modules/{module_id}")
def api_get_module(module_id: str) -> dict[str, Any]:
    m = get_module(module_id)
    if not m:
        raise HTTPException(status_code=404, detail="模块不存在")
    return m


@router.post("/modules")
def api_save_module(body: ModuleBody) -> dict[str, Any]:
    mod = ModuleDef(
        id=body.id or new_id("mod"),
        name=body.name,
        description=body.description,
        inputs=[Port.model_validate(p) for p in body.inputs],
        outputs=[Port.model_validate(p) for p in body.outputs],
        graph=body.graph,
        created_at="",
        updated_at="",
    )
    if not mod.name:
        raise HTTPException(status_code=400, detail="模块名称不能为空")
    # 保存前校验内部子图
    report = validate_graph(mod.graph, {})
    if not report["ok"]:
        raise HTTPException(status_code=400, detail={"message": "内部子图校验未通过", "report": report})
    data = mod.model_dump(by_alias=True)
    return save_module(body.id, mod.name, mod.description, data)


@router.delete("/modules/{module_id}")
def api_delete_module(module_id: str) -> dict[str, Any]:
    if not delete_module(module_id):
        raise HTTPException(status_code=404, detail="模块不存在")
    return {"ok": True}


# ---------------------------------------------------------------------------
# 内置数据集
# ---------------------------------------------------------------------------

@router.get("/datasets")
def api_datasets() -> list[dict[str, Any]]:
    return [
        {"id": "random", "name": "随机张量拟合", "task": "regression",
         "doc": "按模型输入输出形状生成随机数据做 MSE 拟合，任意结构都能冒烟训练。"},
        {"id": "mnist", "name": "MNIST 图像分类", "task": "classification",
         "doc": "手写数字识别。要求模型输入 float32 [batch, 1, 28, 28]，输出 [batch, 10]。"},
        {"id": "text_cls", "name": "文本情感分类（合成语料）", "task": "classification",
         "doc": "英文正/负面情感分类。要求模型输入 int64 [batch, seq]，输出 [batch, 2]。"},
    ]


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

class TrainBody(BaseModel):
    graph: Graph
    model_id: str | None = None
    dataset: str = "random"
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    num_samples: int = 512


@router.post("/trainings")
def api_start_training(body: TrainBody) -> dict[str, Any]:
    try:
        files = generate_package(body.graph, _modules_map())
    except CodegenError as e:
        raise HTTPException(status_code=400, detail=e.report)

    model_name = body.graph.model.name or "model"
    slug = re.sub(r"[^\w\-]", "_", model_name)
    default_id = slug + "-" + time.strftime("%Y%m%d-%H%M%S")
    model_id = body.model_id or default_id
    model_id = re.sub(r"[^\w\-.]", "_", model_id)
    if get_model(model_id):
        raise HTTPException(status_code=400, detail=f"模型 ID「{model_id}」已存在，请换一个")

    run_dir = RUNS_DIR / model_id
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (code_dir / name).write_text(content, encoding="utf-8")

    job_id = new_id("job")
    hyper = {"epochs": body.epochs, "batch_size": body.batch_size,
             "lr": body.lr, "num_samples": body.num_samples}
    record_training(job_id, model_id, body.dataset, hyper)
    job = start_job(job_id, model_id, run_dir, body.dataset, hyper,
                    body.graph.model_dump(by_alias=True), model_name)
    return {"job": job, "model_id": model_id}


@router.get("/trainings")
def api_list_trainings() -> list[dict[str, Any]]:
    return list_jobs()


@router.get("/trainings/{job_id}")
def api_get_training(job_id: str) -> dict[str, Any]:
    try:
        return get_job(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="任务不存在")


@router.post("/trainings/{job_id}/stop")
def api_stop_training(job_id: str) -> dict[str, Any]:
    try:
        stop_job(job_id)
        return {"ok": True}
    except JobNotFound:
        raise HTTPException(status_code=404, detail="任务不存在")


@router.get("/trainings/{job_id}/logs")
async def api_training_logs(job_id: str) -> StreamingResponse:
    """SSE：实时推送训练事件（先重放历史，再跟随新事件）。"""
    async def gen():
        idx = 0
        try:
            while True:
                events, done, status = iter_events(job_id, idx)
                for e in events:
                    idx += 1
                    yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
                if done:
                    yield f"data: {json.dumps({'type': 'status', 'message': 'stream closed', 'status': status})}\n\n"
                    break
                await asyncio.sleep(0.5)
        except JobNotFound:
            yield f"data: {json.dumps({'type': 'error', 'message': '任务不存在'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# 已训练模型 / 推理
# ---------------------------------------------------------------------------

@router.get("/models")
def api_list_models() -> list[dict[str, Any]]:
    return list_models()


@router.get("/models/{model_id}")
def api_get_model(model_id: str) -> dict[str, Any]:
    m = get_model(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="模型不存在")
    run_dir = Path(m["run_dir"])
    has_vocab = (run_dir / "vocab.json").exists()
    code_files = [f.name for f in (run_dir / "code").iterdir() if f.is_file()] if (run_dir / "code").exists() else []
    return {
        **{k: v for k, v in m.items() if k != "topology"},
        "topology": m["topology"],
        "has_vocab": has_vocab,
        "code_files": code_files,
    }


@router.get("/models/{model_id}/source")
def api_model_source(model_id: str, file: str = "model.py") -> dict[str, str]:
    m = get_model(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="模型不存在")
    safe = Path(file).name
    path = Path(m["run_dir"]) / "code" / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"文件 {safe} 不存在")
    return {"file": safe, "content": path.read_text(encoding="utf-8")}


@router.get("/models/{model_id}/files")
def api_model_files(model_id: str) -> dict[str, Any]:
    m = get_model(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="模型不存在")
    code_dir = Path(m["run_dir"]) / "code"
    files = {}
    if code_dir.exists():
        for f in code_dir.iterdir():
            if f.is_file() and f.suffix in (".py", ".json", ".md"):
                files[f.name] = f.read_text(encoding="utf-8")
    return {"files": files}


class PredictBody(BaseModel):
    inputs: dict[str, Any]


@router.post("/models/{model_id}/predict")
def api_predict(model_id: str, body: PredictBody) -> dict[str, Any]:
    m = get_model(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="模型不存在")
    try:
        return infer.predict(model_id, m["run_dir"], body.inputs)
    except infer.InferError as e:
        raise HTTPException(status_code=400, detail=str(e))
