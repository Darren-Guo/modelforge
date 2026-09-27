"""API 级端到端验收：真实训练 + 推理 + 自定义模块 + 代码生成。

覆盖：
  A. 小 CNN + MNIST 真实训练（GPU/CPU）→ 模型注册表 → 图片推理
  B. 文本 MLP + text_cls 训练 → 文本推理（词表链路）
  C. 自定义模块保存 → 在新模型中引用 → 生成代码含独立 class
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000/api"
SUFFIX = time.strftime("%H%M%S")


def req(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        raise RuntimeError(f"{method} {path} -> {e.code}: {detail[:500]}") from None


MNIST_ID = f"e2e-mnist-cnn-{SUFFIX}"
TEXT_ID = f"e2e-text-mlp-{SUFFIX}"


def p(name, dtype="float32", shape=None):
    return {"name": name, "dtype": dtype, "shape": shape or []}


def node(nid, name, op, group, attrs, inputs, outputs):
    return {"id": nid, "name": name, "op": op, "group": group, "attrs": attrs,
            "inputs": [p(x) for x in inputs], "outputs": [p(x) for x in outputs], "ui": {}}


def edge(eid, src, sport, tgt, tport):
    return {"id": eid, "from": {"node": src, "port": sport}, "to": {"node": tgt, "port": tport}}


def cnn_graph():
    """Conv2d→ReLU→MaxPool→Flatten→Linear：输入 [batch,1,28,28] 输出 [batch,10]。"""
    return {
        "format": "modelforge/graph", "version": "0.1",
        "model": {"name": "MNIST小CNN", "description": "E2E 验收模型"},
        "inputs": [p("x", "float32", ["batch", 1, 28, 28])],
        "outputs": [p("y", "float32", ["batch", 10])],
        "nodes": [
            node("n0", "Conv_1", "Conv2d", "basic",
                 {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "stride": 1,
                  "padding": 1, "dilation": 1, "bias": True}, ["x"], ["y"]),
            node("n1", "ReLU_1", "ReLU", "basic", {}, ["x"], ["y"]),
            node("n2", "Pool_1", "MaxPool2d", "basic",
                 {"kernel_size": 2, "stride": 2, "padding": 0}, ["x"], ["y"]),
            node("n3", "Flatten_1", "Flatten", "basic", {"start_dim": 1, "end_dim": -1}, ["x"], ["y"]),
            node("n4", "Linear_1", "Linear", "basic",
                 {"in_features": 8 * 14 * 14, "out_features": 10, "bias": True}, ["x"], ["y"]),
        ],
        "edges": [
            edge("e0", "__graph_in__", "x", "n0", "x"),
            edge("e1", "n0", "y", "n1", "x"),
            edge("e2", "n1", "y", "n2", "x"),
            edge("e3", "n2", "y", "n3", "x"),
            edge("e4", "n3", "y", "n4", "x"),
            edge("e5", "n4", "y", "__graph_out__", "y"),
        ],
    }


def text_graph():
    """TokenEmbedding→Flatten→Linear：输入 int64 [batch,32] 输出 [batch,2]。"""
    return {
        "format": "modelforge/graph", "version": "0.1",
        "model": {"name": "TextMLP", "description": "文本分类 E2E"},
        "inputs": [p("ids", "int64", ["batch", 32])],
        "outputs": [p("y", "float32", ["batch", 2])],
        "nodes": [
            {"id": "t0", "name": "Embed_1", "op": "TokenEmbedding", "group": "llm",
             "attrs": {"vocab_size": 200, "d_model": 32, "scale": True},
             "inputs": [p("ids", "int64")], "outputs": [p("y")], "ui": {}},
            node("t1", "Flatten_1", "Flatten", "basic", {"start_dim": 1, "end_dim": -1}, ["x"], ["y"]),
            node("t2", "Linear_1", "Linear", "basic",
                 {"in_features": 1024, "out_features": 2, "bias": True}, ["x"], ["y"]),
        ],
        "edges": [
            edge("f0", "__graph_in__", "ids", "t0", "ids"),
            edge("f1", "t0", "y", "t1", "x"),
            edge("f2", "t1", "y", "t2", "x"),
            edge("f3", "t2", "y", "__graph_out__", "y"),
        ],
    }


def wait_job(job_id: str, timeout: float = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = req("GET", f"/trainings/{job_id}")
        st = job["status"]
        if st in ("done", "failed", "stopped"):
            return job
        time.sleep(2)
    raise TimeoutError(f"训练超时: {job_id}")


def main() -> None:
    # ---------- A. MNIST CNN 真实训练 ----------
    g = cnn_graph()
    rep = req("POST", "/validate", {"graph": g})
    assert rep["ok"], rep["errors"]
    shapes = {n["id"]: n["outputs"][0]["shape"] for n in rep["nodes"]}
    assert shapes["n0"] == ["batch", 8, 28, 28], shapes["n0"]
    assert shapes["n2"] == ["batch", 8, 14, 14], shapes["n2"]
    assert shapes["n3"] == ["batch", 1568], shapes["n3"]
    print("[PASS] A1. CNN 形状推导：Conv→[b,8,28,28]→Pool→[b,8,14,14]→Flatten→[b,1568]")

    tr = req("POST", "/trainings", {
        "graph": g, "model_id": MNIST_ID, "dataset": "mnist",
        "epochs": 2, "batch_size": 64, "lr": 0.001, "num_samples": 512,
    })
    job = wait_job(tr["job"]["job_id"])
    assert job["status"] == "done", f"训练失败: {json.dumps(job['events'][-3:], ensure_ascii=False)}"
    losses = [e for e in job["events"] if e.get("type") == "metric"]
    assert len(losses) == 2 and losses[-1]["val_loss"] <= losses[0]["val_loss"] * 1.5, losses
    print(f"[PASS] A2. MNIST 训练完成：loss {losses[0]['train_loss']:.4f} → {losses[-1]['train_loss']:.4f}，"
          f"val_acc {losses[-1].get('val_acc')}")

    models = req("GET", "/models")
    assert any(m["model_id"] == MNIST_ID for m in models), models
    detail = req("GET", f"/models/{MNIST_ID}")
    assert "model.py" in detail["code_files"]
    print("[PASS] A3. 模型注册表按 ID 找到 e2e-mnist-cnn，含源码快照")

    import numpy as np
    from PIL import Image
    img = (np.random.rand(28, 28) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    pred = req("POST", f"/models/{MNIST_ID}/predict",
               {"inputs": {"x": {"kind": "image", "base64": b64}}})
    probs = pred["outputs"]["y"]["probabilities"]
    assert len(probs) == 10 and abs(sum(probs) - 1) < 0.01, probs
    print(f"[PASS] A4. 图片推理：预测类别 {pred['outputs']['y']['predicted_class']}，概率归一 ✓")

    # ---------- B. 文本分类 ----------
    gt = text_graph()
    tr2 = req("POST", "/trainings", {
        "graph": gt, "model_id": TEXT_ID, "dataset": "text_cls",
        "epochs": 6, "batch_size": 32, "lr": 0.01, "num_samples": 600,
    })
    job2 = wait_job(tr2["job"]["job_id"])
    assert job2["status"] == "done", job2["events"][-3:]
    losses2 = [e for e in job2["events"] if e.get("type") == "metric"]
    assert losses2[-1]["val_acc"] is not None and losses2[-1]["val_acc"] > 0.6, losses2[-1]
    print(f"[PASS] B1. 文本分类训练完成：val_acc {losses2[-1]['val_acc']}（应明显>0.5 机会水平）")

    pred2 = req("POST", f"/models/{TEXT_ID}/predict",
                {"inputs": {"ids": {"kind": "text", "text": "the movie was great and wonderful"}}})
    probs2 = pred2["outputs"]["y"]["probabilities"]
    assert len(probs2) == 2 and pred2["outputs"]["y"]["predicted_class"] == 1, pred2
    print(f"[PASS] B2. 文本推理：正向句子 → 预测类别 1（正面），概率 {probs2}")

    # ---------- C. 自定义模块 ----------
    ffn_inner = {
        "format": "modelforge/graph", "version": "0.1",
        "model": {"name": "", "description": ""},
        "inputs": [p("x", "float32", ["batch", 64])],
        "outputs": [p("y", "float32", ["batch", 64])],
        "nodes": [
            node("m0", "up", "Linear", "basic", {"in_features": 64, "out_features": 128, "bias": True}, ["x"], ["y"]),
            node("m1", "act", "GELU", "basic", {"approximate": "none"}, ["x"], ["y"]),
            node("m2", "down", "Linear", "basic", {"in_features": 128, "out_features": 64, "bias": True}, ["x"], ["y"]),
        ],
        "edges": [
            edge("x0", "__graph_in__", "x", "m0", "x"),
            edge("x1", "m0", "y", "m1", "x"),
            edge("x2", "m1", "y", "m2", "x"),
            edge("x3", "m2", "y", "__graph_out__", "y"),
        ],
    }
    mod = req("POST", "/modules", {
        "name": "FFN", "description": "Linear+GELU+Linear 前馈块",
        "inputs": [p("x", "float32", ["batch", 64])],
        "outputs": [p("y", "float32", ["batch", 64])],
        "graph": ffn_inner,
    })
    mod_id = mod["id"]
    print(f"[PASS] C1. 自定义模块 FFN 保存成功：{mod_id}")

    use_graph = {
        "format": "modelforge/graph", "version": "0.1",
        "model": {"name": "FFN下游模型", "description": ""},
        "inputs": [p("x", "float32", ["batch", 64])],
        "outputs": [p("y", "float32", ["batch", 4])],
        "nodes": [
            node("u0", "FFN_1", f"custom:{mod_id}", "custom", {}, ["x"], ["y"]),
            node("u1", "Linear_out", "Linear", "basic",
                 {"in_features": 64, "out_features": 4, "bias": True}, ["x"], ["y"]),
        ],
        "edges": [
            edge("u0e", "__graph_in__", "x", "u0", "x"),
            edge("u1e", "u0", "y", "u1", "x"),
            edge("u2e", "u1", "y", "__graph_out__", "y"),
        ],
    }
    rep2 = req("POST", "/validate", {"graph": use_graph})
    assert rep2["ok"], rep2["errors"]
    gen = req("POST", "/codegen", {"graph": use_graph})
    assert "class CustomFFN(nn.Module)" in gen["files"]["model.py"]
    assert "self.m_u0 = CustomFFN()" in gen["files"]["model.py"]
    print("[PASS] C2. 自定义模块在新模型中引用，生成代码含独立 class CustomFFN + 组合实例化")

    # 双击节点源码（场景4扩展）
    sn = req("POST", "/codegen/snippet", {"graph": g, "node_id": "n0"})
    assert "nn.Conv2d" in sn["code"] and "Conv" in sn["title"]
    print(f"[PASS] C3. 双击算子节点弹出源码片段：{sn['title']}")

    print("\nALL API E2E TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
