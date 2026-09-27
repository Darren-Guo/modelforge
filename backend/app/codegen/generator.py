"""代码生成管线：拓扑 JSON → 可运行的代码包。

流程：图校验 → 形状推导 → 拓扑排序 → 逐节点渲染 Jinja2 模板 → 组装文件。

生成产物（自包含，可独立运行；文件名由 TargetSpec 描述符决定）：
  model.py    — nn.Module（自定义模块生成为独立 class，组合引用）
  train.py    — 训练入口（内置数据集：mnist / text_cls / random）
  config.json — 拓扑快照
  README.md   — 使用说明

代码生成目标（target）通过 OpDef 模板族（init_tpl/fwd_tpl 可按 target 索引）
与 TargetSpec 描述符扩展；当前内置 "torch" 目标。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from jinja2 import BaseLoader, Environment

from ..ir.schema import Graph, ModuleDef
from ..ir.shapes import ShapeError
from ..ir.validate import GRAPH_IN, GRAPH_OUT, node_port_specs, validate_graph
from ..registry.ops import HELPERS, OP_INDEX, OpDef, normalize_attrs, pylit, resolve_tpl

_templates = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)


# ---------------------------------------------------------------------------
# 代码生成目标描述符：产物文件名与训练入口收敛于此（新增后端时补一条即可）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSpec:
    id: str
    label: str
    files: dict[str, str] = field(default_factory=dict)  # 角色 → 文件名
    entry_file: str = ""   # 训练入口文件（jobs 等执行端按此解析）
    entry_cmd: str = ""    # README 快速开始中的训练命令


TARGETS: dict[str, TargetSpec] = {
    "torch": TargetSpec(
        id="torch",
        label="PyTorch",
        files={"model": "model.py", "train": "train.py",
               "config": "config.json", "readme": "README.md"},
        entry_file="train.py",
        entry_cmd="python train.py --dataset random --epochs 3 --out ./run",
    ),
}


class CodegenError(Exception):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("图校验未通过，无法生成代码")


# ---------------------------------------------------------------------------
# 模板渲染：属性值既要能嵌入 Python 字面量，又要支持 Jinja 条件判断
# ---------------------------------------------------------------------------

class Lit:
    """str() 输出 Python 字面量，比较/布尔运算按原始值。"""

    __slots__ = ("raw", "text")

    def __init__(self, raw: Any, text: str):
        self.raw = raw
        self.text = text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Lit):
            return self.raw == other.raw
        return self.raw == other

    def __bool__(self) -> bool:
        return bool(self.raw)


def _wrap_attrs(attrs: dict[str, Any]) -> dict[str, Lit]:
    return {k: Lit(v, pylit(v)) for k, v in attrs.items()}


def _render(tpl: str, **ctx: Any) -> str:
    return _templates.from_string(tpl).render(**ctx).strip()


# ---------------------------------------------------------------------------
# 命名工具
# ---------------------------------------------------------------------------

_PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
}


def sanitize(name: str) -> str:
    s = re.sub(r"\W+", "_", name or "x")
    if not s or s[0].isdigit():
        s = "_" + s
    if s in _PY_KEYWORDS:
        s += "_"
    return s


def _unique(base: str, taken: set[str]) -> str:
    name = base
    i = 2
    while name in taken:
        name = f"{base}_{i}"
        i += 1
    taken.add(name)
    return name


_HELPER_CLASS_RE = re.compile(r"^class (\w+)", re.M)


def _helper_class_names() -> set[str]:
    """从 HELPERS 源码里提取辅助类名（RMSNorm/RotaryEmbedding/...），用于命名保留。"""
    return {m for src in HELPERS.values() for m in _HELPER_CLASS_RE.findall(src)}


# ---------------------------------------------------------------------------
# 自定义模块收集与命名
# ---------------------------------------------------------------------------

def _collect_custom(graph: Graph, modules: dict[str, ModuleDef], acc: dict[str, ModuleDef]) -> None:
    for n in graph.nodes:
        if n.op.startswith("custom:"):
            mid = n.op.split(":", 1)[1]
            if mid not in acc:
                mod = modules.get(mid)
                if mod is None:
                    raise CodegenError({"ok": False, "errors": [{"where": n.name or n.id, "message": f"自定义模块 {mid} 不存在"}], "warnings": [], "nodes": [], "order": []})
                acc[mid] = mod
                _collect_custom(mod.graph, modules, acc)


def _camel(name: str) -> str:
    """FFN → FFN，res_block → ResBlock（保留缩写大小写）。"""
    parts = re.split(r"[^0-9a-zA-Z]+", name or "")
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _assign_class_names(customs: dict[str, ModuleDef]) -> dict[str, str]:
    taken: set[str] = set()
    names: dict[str, str] = {}
    for mid, mod in customs.items():
        base = "Custom" + (_camel(mod.name or mid) or "Module")
        names[mid] = _unique(base, taken)
    return names


# ---------------------------------------------------------------------------
# 单个 class 的渲染（模型顶层与自定义模块共用）
# ---------------------------------------------------------------------------

def _render_class(
    graph: Graph,
    modules: dict[str, ModuleDef],
    class_names: dict[str, str],
    class_name: str,
    doc: str,
    helpers_used: set[str],
    target: str = "torch",
) -> str:
    report = validate_graph(graph, modules)
    if not report["ok"]:
        raise CodegenError(report)
    order: list[str] = report["order"]
    nodes = {n.id: n for n in graph.nodes}
    edge_by_target = {(e.target.node, e.target.port): e for e in graph.edges}

    # 命名去重：参数 / 子模块变量 / 前向局部变量共用 taken，
    # sanitize 后的碰撞（如 "a b" 与 "a_b"）自动加后缀，避免共用同一变量。
    taken: set[str] = {"_h"}  # 模板内部临时变量保留名
    arg_names: list[str] = []
    for p in graph.inputs:
        arg_names.append(_unique(sanitize(p.name), taken))
    node_var: dict[str, str] = {}
    port_var: dict[tuple[str, str], str] = {}
    for nid in order:
        node_var[nid] = "m_" + _unique(sanitize(nid), taken)
        _, outs = node_port_specs(nodes[nid], modules)
        for p in outs:
            port_var[(nid, p.name)] = "v_" + _unique(f"{sanitize(nid)}_{sanitize(p.name)}", taken)

    def src_var(tgt_node: str, tgt_port: str) -> str:
        e = edge_by_target.get((tgt_node, tgt_port))
        if e is None:
            raise CodegenError({
                "ok": False,
                "errors": [{"where": f"{tgt_node}.{tgt_port}", "message": "端口没有连线"}],
                "warnings": [], "nodes": [], "order": [],
            })
        if e.source.node == GRAPH_IN:
            return arg_names[[p.name for p in graph.inputs].index(e.source.port)]
        return port_var[(e.source.node, e.source.port)]

    init_lines: list[str] = []
    fwd_lines: list[str] = []
    for nid in order:
        n = nodes[nid]
        var = node_var[nid]
        ins, outs = node_port_specs(n, modules)
        in_ctx = {p.name: src_var(nid, p.name) for p in ins}
        out_ctx = {p.name: port_var[(nid, p.name)] for p in outs}

        if n.op.startswith("custom:"):
            mid = n.op.split(":", 1)[1]
            init_lines.append(f"self.{var} = {class_names[mid]}()")
            out_vars = ", ".join(out_ctx[p.name] for p in outs)
            in_vars = ", ".join(in_ctx[p.name] for p in ins)
            fwd_lines.append(f"{out_vars} = self.{var}({in_vars})" if len(outs) > 1
                             else f"{out_ctx[outs[0].name]} = self.{var}({in_vars})")
        else:
            op = OP_INDEX[n.op]
            attrs = normalize_attrs(op, n.attrs)[0]  # 校验已通过，此处补默认值保证模板不渲染空值
            ctx = {"var": var, "in": in_ctx, "out": out_ctx, "attrs": _wrap_attrs(attrs)}
            try:
                init_lines.append(_render(resolve_tpl(op.init_tpl, op.op, "初始化", target), **ctx))
                fwd_lines.append(_render(resolve_tpl(op.fwd_tpl, op.op, "前向", target), **ctx))
            except ShapeError as e:
                raise CodegenError({
                    "ok": False,
                    "errors": [{"where": n.name or n.id, "message": str(e)}],
                    "warnings": [], "nodes": [], "order": [],
                }) from None
            if op.helper:
                helpers_used.add(op.helper)

    # 图级输出：取 __graph_out__ 各端口的上游变量
    ret_vars = [src_var(GRAPH_OUT, p.name) for p in graph.outputs] if graph.outputs else []
    ret_expr = ", ".join(ret_vars)
    if len(ret_vars) == 1:
        ret_line = f"return {ret_expr}"
    elif ret_vars:
        ret_line = f"return ({ret_expr},)"
    else:
        ret_line = "return ()"

    sig = ", ".join(arg_names) if arg_names else ""
    code_lines = [
        f"class {class_name}(nn.Module):",
        f'    """{doc}"""',
        "",
        "    def __init__(self):",
        "        super().__init__()" + ("" if init_lines else "  # 空图"),
    ]
    for block in init_lines:
        for line in block.splitlines() or [""]:
            code_lines.append(("        " + line).rstrip())
    code_lines += [
        "",
        f"    def forward(self, {sig}):",
    ]
    if not fwd_lines and len(ret_vars) == 1 and ret_vars[0] in arg_names:
        code_lines.append(f"        {ret_line}")
    else:
        for block in fwd_lines:
            for line in block.splitlines() or [""]:
                code_lines.append(("        " + line).rstrip())
        code_lines.append(f"        {ret_line}")
    return "\n".join(code_lines)


def _render_custom_classes(
    graph: Graph,
    modules: dict[str, ModuleDef],
    customs: dict[str, ModuleDef],
    class_names: dict[str, str],
    helpers_used: set[str],
    memo: dict[str, str],
    done: list[str],
    target: str = "torch",
) -> None:
    """依赖优先（内层模块先生成）渲染全部自定义模块 class。"""
    for n in graph.nodes:
        if n.op.startswith("custom:"):
            mid = n.op.split(":", 1)[1]
            if mid in done:
                continue
            _render_custom_classes(modules[mid].graph, modules, customs, class_names,
                                   helpers_used, memo, done, target)
            if mid not in done:
                mod = modules[mid]
                memo[mid] = _render_class(
                    mod.graph, modules, class_names, class_names[mid],
                    f"自定义模块「{mod.name}」：{mod.description or '（无描述）'}",
                    helpers_used,
                    target,
                )
                done.append(mid)


# ---------------------------------------------------------------------------
# 文件组装
# ---------------------------------------------------------------------------

def _python_literal(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def generate_package(graph: Graph, modules: dict[str, ModuleDef] | None = None,
                     target: str = "torch") -> dict[str, str]:
    """生成完整代码包，返回 {文件名: 内容}（文件名由 TARGETS[target] 描述符决定）。"""
    if target not in TARGETS:
        raise CodegenError({"ok": False, "errors": [
            {"where": "target", "message": f"未知代码生成目标「{target}」，可用：{'、'.join(TARGETS)}"}],
            "warnings": [], "nodes": [], "order": []})
    spec = TARGETS[target]
    modules = modules or {}
    report = validate_graph(graph, modules)
    if not report["ok"]:
        raise CodegenError(report)

    customs: dict[str, ModuleDef] = {}
    _collect_custom(graph, modules, customs)
    class_names = _assign_class_names(customs)

    helpers_used: set[str] = set()
    memo: dict[str, str] = {}
    done: list[str] = []
    _render_custom_classes(graph, modules, customs, class_names, helpers_used, memo, done, target)

    # 模型类名不得覆盖自定义模块类名 / 辅助类名 / 关键字
    reserved = set(class_names.values()) | _PY_KEYWORDS | _helper_class_names()
    base_name = sanitize(graph.model.name or "GeneratedModel")
    model_name = base_name
    for _ in range(4):
        if model_name not in reserved:
            break
        model_name += "Model"
    if model_name in reserved:
        model_name = _unique(base_name, reserved)

    model_class_code = _render_class(
        graph, modules, class_names, model_name,
        f"模型「{graph.model.name}」：{graph.model.description or '（无描述）'}"
        " — 由 ModelForge 可视化拼接工具生成",
        helpers_used,
        target,
    )

    # model.py：辅助类按 HELPERS 键序注入（名单从 OPS 的 helper 引用推导，无第二份清单）
    blocks = [HELPERS[h] for h in HELPERS if h in helpers_used]
    blocks += [memo[mid] for mid in done]
    blocks.append(model_class_code)
    model_py = _templates.from_string(MODEL_PY_TPL).render(
        model_name=graph.model.name,
        blocks=blocks,
    )

    # train.py
    input_specs = [{"name": p.name, "dtype": p.dtype or "float32", "shape": p.shape} for p in graph.inputs]
    output_specs = [{"name": p.name, "dtype": p.dtype or "float32", "shape": p.shape} for p in graph.outputs]
    train_py = _templates.from_string(TRAIN_PY_TPL).render(
        class_name=model_name,
        input_specs=_python_literal(input_specs),
        output_specs=_python_literal(output_specs),
        model_name=graph.model.name,
    )

    config = {
        "format": graph.format,
        "version": graph.version,
        "generated_by": "ModelForge",
        "model_class": model_name,
        "target": target,
        **graph.model_dump(by_alias=True, exclude={"format", "version"}),
    }
    config_json = json.dumps(config, ensure_ascii=False, indent=2)

    readme = (
        f"# {graph.model.name}\n\n"
        f"{graph.model.description or '（无描述）'}\n\n"
        f"由 ModelForge 可视化拼接工具生成（目标：{spec.label}）。\n\n"
        "## 文件\n\n"
        f"- `{spec.files['model']}` — 模型定义（`" + model_name + "`）\n"
        f"- `{spec.files['train']}` — 训练入口\n"
        f"- `{spec.files['config']}` — 拓扑快照\n\n"
        "## 快速开始\n\n"
        "```bash\n"
        "python -c \"import torch; from " + spec.files["model"].rsplit(".", 1)[0] + " import "
        + model_name + "; "
        "m = " + model_name + "(); print(m)\"\n"
        + spec.entry_cmd + "\n"
        "```\n\n"
        "内置数据集：`random`（随机张量拟合，任意结构可跑，多输出按各输出分别求损失）、"
        "`mnist`（图像分类）、`text_cls`（合成文本分类）。\n"
    )

    return {
        spec.files["model"]: model_py,
        spec.files["train"]: train_py,
        spec.files["config"]: config_json,
        spec.files["readme"]: readme,
    }


def node_snippet(graph: Graph, node_id: str, modules: dict[str, ModuleDef] | None = None,
                 target: str = "torch") -> dict[str, str]:
    """双击节点查看源码：自定义模块 → 整个 class；算子 → 初始化 + 前向片段。"""
    modules = modules or {}
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if node is None:
        raise CodegenError({"ok": False, "errors": [{"where": node_id, "message": "节点不存在"}], "warnings": [], "nodes": [], "order": []})

    if node.op.startswith("custom:"):
        mid = node.op.split(":", 1)[1]
        mod = modules.get(mid)
        if mod is None:
            raise CodegenError({"ok": False, "errors": [
                {"where": node.name or node.id, "message": f"自定义模块 {mid} 不存在"}],
                "warnings": [], "nodes": [], "order": []})
        customs: dict[str, ModuleDef] = {}
        _collect_custom(graph, modules, customs)
        class_names = _assign_class_names(customs)
        helpers_used: set[str] = set()
        memo: dict[str, str] = {}
        done: list[str] = []
        _render_custom_classes(graph, modules, customs, class_names, helpers_used, memo, done, target)
        code = "\n\n".join([memo[d] for d in done])
        return {"kind": "class", "title": f"自定义模块 {mod.name}", "code": code}

    op = OP_INDEX.get(node.op)
    if op is None:
        raise CodegenError({"ok": False, "errors": [
            {"where": node.name or node.id, "message": f"未知算子: {node.op}"}],
            "warnings": [], "nodes": [], "order": []})
    attrs = normalize_attrs(op, node.attrs)[0]  # 缺失属性回填默认值，片段永远是合法代码
    ctx = {
        "var": "block",
        "in": {p.name: f"x_{p.name}" for p in op.inputs},  # type: ignore[attr-defined]
        "out": {p.name: f"y_{p.name}" for p in op.outputs},  # type: ignore[attr-defined]
        "attrs": _wrap_attrs(attrs),
    }
    try:
        init_code = _render(resolve_tpl(op.init_tpl, op.op, "初始化", target), **ctx)
        fwd_code = _render(resolve_tpl(op.fwd_tpl, op.op, "前向", target), **ctx)
    except ShapeError as e:
        raise CodegenError({"ok": False, "errors": [
            {"where": node.name or node.id, "message": str(e)}],
            "warnings": [], "nodes": [], "order": []}) from None
    code = (
        f"# {op.label} — {node.name or node.id}\n"
        f"# {op.doc}\n\n"
        "# __init__ 中：\n"
        f"{init_code}\n\n"
        "# forward 中：\n"
        f"{fwd_code}\n"
    )
    return {"kind": "snippet", "title": f"{op.label}（{node.name or node.id}）", "code": code}


# ---------------------------------------------------------------------------
# 文件级 Jinja2 模板
# ---------------------------------------------------------------------------

MODEL_PY_TPL = '''"""{{ model_name }} — 由 ModelForge 可视化拼接工具生成。

本文件是自包含的 PyTorch 模型定义，可直接运行查看结构：
    python -c "from model import *; import torch; \
m = [c for c in globals().values() if isinstance(c, type) and issubclass(c, nn.Module)][0]; print(m)"
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

{% for block in blocks %}
{{ block }}
{% endfor %}
'''

TRAIN_PY_TPL = '''"""训练脚本 — 由 ModelForge 生成，可独立运行。

用法示例：
    python train.py --dataset random --epochs 5 --out ./run
    python train.py --dataset mnist --epochs 3 --batch-size 64 --out ./run
    python train.py --dataset text_cls --epochs 5 --out ./run

训练过程按行输出 JSON 指标（{"type": "metric", ...}），便于上层采集。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model import {{ class_name }} as GeneratedModel

INPUT_SPECS = {{ input_specs }}
OUTPUT_SPECS = {{ output_specs }}
N_IN = len(INPUT_SPECS)
N_OUT = len(OUTPUT_SPECS)


def log(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def dim_value(d, default=16):
    return d if isinstance(d, int) else default


def check_classification_io(dataset_name: str) -> int:
    """分类数据集要求：单输入单输出，输出 [batch, num_classes]。"""
    if len(INPUT_SPECS) != 1 or len(OUTPUT_SPECS) != 1:
        raise SystemExit(f"[error] 数据集 {dataset_name} 需要单输入单输出的模型（当前 {len(INPUT_SPECS)} 入 {len(OUTPUT_SPECS)} 出）；任意结构请用 --dataset random")
    out = OUTPUT_SPECS[0]["shape"]
    if len(out) != 2 or not isinstance(out[1], int):
        raise SystemExit(f"[error] {dataset_name} 要求输出形状 [batch, num_classes]，当前 {out}")
    return out[1]


def build_random(n: int, seed: int):
    """随机张量拟合：任意结构都能冒烟训练（各输出分别 MSE，多输出取损失之和）。"""
    g = torch.Generator().manual_seed(seed)
    xs = []
    for spec in INPUT_SPECS:
        shape = [n] + [dim_value(d) for d in spec["shape"][1:]]
        if spec["dtype"].startswith("int"):
            xs.append(torch.randint(0, 100, shape))
        else:
            xs.append(torch.randn(shape, generator=g))
    ys = [torch.randn([n] + [dim_value(d) for d in spec["shape"][1:]], generator=g)
          for spec in OUTPUT_SPECS]
    return TensorDataset(*xs, *ys), "regression"


def split_batch(batch):
    """TensorDataset 批次 → (输入张量列表, 输出张量列表)。"""
    tensors = list(batch)
    return tensors[:N_IN], tensors[N_IN:]


def compute_loss(criterion, task, outs, ys):
    if task == "classification":
        return criterion(outs[0], ys[0])
    return sum(criterion(o, y.to(o.dtype)) for o, y in zip(outs, ys))


def build_mnist(n: int, data_dir: str):
    """MNIST 图像分类：输入 [batch, 1, 28, 28]，输出 [batch, 10]。"""
    num_classes = check_classification_io("mnist")
    if num_classes != 10:
        raise SystemExit(f"[error] mnist 要求输出维度 10，当前 {num_classes}")
    spec = INPUT_SPECS[0]
    if spec["dtype"] != "float32" or [dim_value(d) for d in spec["shape"][1:]] != [1, 28, 28]:
        raise SystemExit(f"[error] mnist 要求输入 float32 [batch, 1, 28, 28]，当前 {spec['dtype']} {spec['shape']}；请改用 --dataset random")
    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise SystemExit("[error] mnist 需要 torchvision：pip install torchvision")
    ds = datasets.MNIST(data_dir, train=True, download=True,
                        transform=transforms.ToTensor())
    idx = list(range(len(ds)))
    random.Random(0).shuffle(idx)
    idx = idx[:n]
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx], dtype=torch.long)
    return TensorDataset(xs, ys), "classification"


def build_text_cls(n: int, seq_len: int, seed: int, vocab_out=None):
    """合成英文情感语料分类：输入 int64 [batch, seq_len]，输出 [batch, 2]。"""
    num_classes = check_classification_io("text_cls")
    if num_classes != 2:
        raise SystemExit(f"[error] text_cls 要求输出维度 2，当前 {num_classes}")
    spec = INPUT_SPECS[0]
    if not spec["dtype"].startswith("int"):
        raise SystemExit(f"[error] text_cls 要求输入 int64 [batch, seq]，当前 {spec['dtype']} {spec['shape']}；请改用 --dataset random")

    pos_words = ["good", "great", "excellent", "wonderful", "amazing", "love",
                 "fantastic", "enjoyable", "brilliant", "delightful"]
    neg_words = ["bad", "terrible", "awful", "boring", "poor", "hate",
                 "disappointing", "dreadful", "mediocre", "horrible"]
    templates = [
        "the movie was {w}", "i think it is {w}", "absolutely {w}",
        "this is a {w} story", "what a {w} experience", "the food tasted {w}",
        "the service was really {w}", "i found it quite {w}", "overall {w} quality",
        "everyone said it was {w}",
    ]
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        label = i % 2
        words = pos_words if label == 1 else neg_words
        sent = rng.choice(templates).format(w=rng.choice(words))
        if rng.random() < 0.5:
            sent += " " + rng.choice(templates).format(w=rng.choice(words))
        samples.append((sent, label))
    vocab = {"<pad>": 0, "<unk>": 1}
    for sent, _ in samples:
        for w in sent.split():
            if w not in vocab:
                vocab[w] = len(vocab)
    if vocab_out:
        with open(vocab_out, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
    ids = torch.zeros(len(samples), seq_len, dtype=torch.long)
    labels = torch.zeros(len(samples), dtype=torch.long)
    for i, (sent, label) in enumerate(samples):
        toks = [vocab.get(w, 1) for w in sent.split()][:seq_len]
        ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
        labels[i] = label
    return TensorDataset(ids, labels), "classification"


def main() -> None:
    ap = argparse.ArgumentParser(description="ModelForge 生成的训练脚本")
    ap.add_argument("--dataset", default="random", choices=["random", "mnist", "text_cls"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-samples", type=int, default=512, help="样本数（random/text_cls）")
    ap.add_argument("--seq-len", type=int, default=32, help="文本序列长度")
    ap.add_argument("--out", default="./run", help="产物输出目录")
    ap.add_argument("--data-dir", default=None, help="数据集缓存目录（默认 <out>/data）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else (out_dir / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    log({"type": "status", "message": f"device={device.type} dataset={args.dataset}"})

    try:
        if args.dataset == "mnist":
            dataset, task = build_mnist(args.num_samples, str(data_dir))
        elif args.dataset == "text_cls":
            dataset, task = build_text_cls(args.num_samples, args.seq_len, args.seed,
                                           vocab_out=out_dir / "vocab.json")
        else:
            dataset, task = build_random(args.num_samples, args.seed)
    except SystemExit as e:
        log({"type": "error", "message": str(e)})
        raise

    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = GeneratedModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log({"type": "status", "message": f"params={n_params:,} task={task}"})

    criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, total_correct, total_count = 0.0, 0, 0
        for batch in train_loader:
            xs, ys = split_batch(batch)
            xs = [x.to(device) for x in xs]
            ys = [y.to(device) for y in ys]
            optim.zero_grad()
            outs = model(*xs)
            if not isinstance(outs, tuple):
                outs = (outs,)
            loss = compute_loss(criterion, task, outs, ys)
            loss.backward()
            optim.step()
            total_loss += loss.item() * ys[0].shape[0]
            if task == "classification":
                total_correct += (outs[0].argmax(dim=-1) == ys[0]).sum().item()
            total_count += ys[0].shape[0]
        train_loss = total_loss / max(1, total_count)
        train_acc = total_correct / max(1, total_count) if task == "classification" else None

        model.eval()
        val_loss, val_correct, val_count = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                xs, ys = split_batch(batch)
                xs = [x.to(device) for x in xs]
                ys = [y.to(device) for y in ys]
                outs = model(*xs)
                if not isinstance(outs, tuple):
                    outs = (outs,)
                loss = compute_loss(criterion, task, outs, ys)
                val_loss += loss.item() * ys[0].shape[0]
                if task == "classification":
                    val_correct += (outs[0].argmax(dim=-1) == ys[0]).sum().item()
                val_count += ys[0].shape[0]
        val_loss /= max(1, val_count)
        val_acc = val_correct / max(1, val_count) if task == "classification" else None

        rec = {"type": "metric", "epoch": epoch, "train_loss": round(train_loss, 5),
               "val_loss": round(val_loss, 5), "seconds": round(time.time() - t0, 2)}
        if train_acc is not None:
            rec["train_acc"] = round(train_acc, 4)
            rec["val_acc"] = round(val_acc, 4)
        log(rec)
        history.append(rec)

        if val_loss < best:
            best = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "class_name": "{{ class_name }}",
                "input_specs": INPUT_SPECS,
                "output_specs": OUTPUT_SPECS,
                "task": task,
                "dataset": args.dataset,
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_dir / "model.pt")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_val_loss": best, "task": task,
                   "dataset": args.dataset, "params": n_params}, f, ensure_ascii=False, indent=2)
    log({"type": "done", "best_val_loss": best, "epochs": args.epochs,
         "message": f"训练完成，产物在 {out_dir}"})
    print(f"[done] best val loss = {best:.5f}, saved to {out_dir}")


if __name__ == "__main__":
    main()
'''
