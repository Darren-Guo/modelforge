"""推理服务：加载训练产物（code/model.py + model.pt）执行预测。

输入按模型的 input_specs 自适应准备：
  - int64 且 run 目录有 vocab.json → 文本（分词/截断/填充）
  - float32 形如 [.., 1, 28, 28] → 图片（灰度 28x28）
  - 其他 → 直接把 JSON 数组转张量
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import re
import threading
from pathlib import Path
from typing import Any

import torch

_lock = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}


class InferError(Exception):
    pass


def _load(model_id: str, run_dir: str) -> dict[str, Any]:
    with _lock:
        cached = _cache.get(model_id)
        if cached is not None:
            return cached

    code_path = Path(run_dir) / "code"
    model_py = code_path / "model.py"
    weights = Path(run_dir) / "model.pt"
    if not model_py.exists() or not weights.exists():
        raise InferError(f"模型 {model_id} 的产物不完整（缺少 model.py / model.pt）")

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    spec = importlib.util.spec_from_file_location(f"modelforge_model_{model_id}", model_py)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise InferError(f"加载 model.py 失败: {e}")

    cls_name = ckpt.get("class_name")
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise InferError(f"model.py 中找不到类 {cls_name}")
    model = cls()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    vocab = None
    vocab_path = Path(run_dir) / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    entry = {
        "model": model,
        "input_specs": ckpt.get("input_specs", []),
        "output_specs": ckpt.get("output_specs", []),
        "task": ckpt.get("task", "regression"),
        "dataset": ckpt.get("dataset", ""),
        "vocab": vocab,
    }
    with _lock:
        _cache[model_id] = entry
    return entry


def invalidate(model_id: str) -> None:
    with _lock:
        _cache.pop(model_id, None)


def _prepare_tensor(spec: dict[str, Any], value: dict[str, Any]) -> torch.Tensor:
    dtype = spec.get("dtype", "float32")
    shape = spec.get("shape", [])
    kind = value.get("kind", "tensor")

    if kind == "image":
        try:
            from PIL import Image
        except ImportError:
            raise InferError("图片推理需要 pillow：pip install pillow")
        try:
            raw = base64.b64decode(value["base64"])
            img = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise InferError(f"图片解码失败: {e}")
        target = [d for d in shape[1:] if isinstance(d, int)]
        if target == [1, 28, 28]:
            img = img.convert("L").resize((28, 28))
            t = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, 28, 28) / 255.0
            return t.unsqueeze(0)
        raise InferError(f"暂不支持的图片输入形状 {shape}（目前支持 [batch,1,28,28]），请用张量输入")

    if kind == "text":
        text = value.get("text", "")
        vocab = value.get("vocab")
        if not vocab:
            raise InferError("该输入端口没有词表，无法文本输入")
        seq_len = shape[1] if len(shape) > 1 and isinstance(shape[1], int) else 32
        toks = [vocab.get(w, vocab.get("<unk>", 1)) for w in re.findall(r"\w+", text.lower())][:seq_len]
        toks = toks + [vocab.get("<pad>", 0)] * (seq_len - len(toks))
        return torch.tensor([toks], dtype=torch.int64 if dtype == "int64" else torch.int32)

    # 默认：JSON 数组
    data = value.get("data")
    if data is None:
        raise InferError(f"输入端口缺少数据")
    try:
        t = torch.tensor(data, dtype=torch.float32 if dtype == "float32" else
                         (torch.int64 if dtype == "int64" else torch.int32))
    except Exception as e:
        raise InferError(f"输入数据转换失败: {e}")
    # 自动补 batch 维
    expected_rank = len(shape)
    if expected_rank and t.dim() == expected_rank - 1:
        t = t.unsqueeze(0)
    return t


def predict(model_id: str, run_dir: str, inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _load(model_id, run_dir)
    specs = entry["input_specs"]
    if not specs:
        raise InferError("模型没有输入端口")
    tensors = []
    for spec in specs:
        name = spec["name"]
        value = inputs.get(name)
        if value is None:
            raise InferError(f"缺少输入「{name}」")
        v = dict(value)
        if v.get("kind") == "text" and entry["vocab"]:
            v["vocab"] = entry["vocab"]
        tensors.append(_prepare_tensor(spec, v))

    with torch.no_grad():
        out = entry["model"](*tensors)

    outputs = out if isinstance(out, tuple) else (out,)
    results: dict[str, Any] = {}
    for spec, t in zip(entry["output_specs"], outputs):
        arr = t.detach().cpu()
        item: dict[str, Any] = {"shape": list(arr.shape)}
        if arr.dim() == 2 and entry["task"] == "classification":
            probs = torch.softmax(arr.float(), dim=-1)[0]
            item["probabilities"] = [round(float(p), 5) for p in probs]
            item["predicted_class"] = int(probs.argmax().item())
        else:
            flat = arr.flatten().float()
            item["data"] = [round(float(x), 5) for x in flat[:256].tolist()]
            item["preview_truncated"] = flat.numel() > 256
        results[spec["name"]] = item
    return {"task": entry["task"], "dataset": entry["dataset"], "outputs": results}
