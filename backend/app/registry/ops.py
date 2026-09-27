"""算子注册表：系统的单一事实来源。

每个算子的 UI 元数据（属性表单、端口）、形状推导规则、Jinja2 代码模板
都定义在这里。前端通过 GET /api/ops 动态渲染面板和属性表单，
新增算子只需在本文件加一条定义 + 模板，前端零改动。

模板上下文：
  init 模板: {{ var }}（self.后的属性名）、{{ attrs.xxx }}（已格式化为 Python 字面量）
  fwd  模板: {{ var }}、{{ in.xxx }}（输入张量变量）、{{ out.yyy }}（输出张量变量）、{{ attrs.xxx }}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Union

from ..ir.schema import Dim
from ..ir.shapes import (
    ShapeError,
    assert_dim,
    assert_min_rank,
    assert_rank,
    conv_out_dim,
    flatten_shape,
    fmt_shape,
    merge_shapes,
)

ShapeRule = Callable[[dict[str, Any], list[list[Dim]]], list[list[Dim]]]

# 模板族：str 是 torch 目标的简写；dict 按 target id 索引（如 {"torch": ..., "cpp": ...}）
Template = Union[str, dict]


@dataclass
class AttrDef:
    name: str
    type: str  # int | float | bool | str | enum | int_list
    default: Any
    label: str
    min: float | None = None
    max: float | None = None
    choices: list[str] | None = None
    help: str = ""


@dataclass
class PortDef:
    name: str
    dtype: str = "float32"
    label: str = ""


@dataclass
class OpDef:
    op: str
    group: str  # basic | llm
    label: str
    doc: str
    attrs: list[AttrDef]
    inputs: list[PortDef]
    outputs: list[PortDef]
    shape_rule: ShapeRule
    init_tpl: Template
    fwd_tpl: Template
    helper: str = ""  # 注入生成文件的辅助类代码（如 RMSNorm、RoPE）
    attr_check: Callable[[dict[str, Any]], None] | None = None  # 纯属性交叉约束（如整除、奇偶）


def resolve_tpl(tpl: Template, op: str, kind: str, target: str = "torch") -> str:
    """按 target 解析算子模板；str 视为 torch 目标简写。"""
    table: dict[str, str] = {"torch": tpl} if isinstance(tpl, str) else dict(tpl)
    if target not in table:
        raise ShapeError(f"算子 {op} 缺少 {target} 目标的{kind}模板")
    return table[target]


def normalize_attrs(op: OpDef, attrs: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """补全缺失属性（回填 AttrDef.default）并按 AttrDef 校验取值。

    返回 (完整属性, 错误列表, 警告列表)：
    - 缺失键：报错 + 回填默认值（避免把空值渲染进模板产出非法代码）
    - 类型/min/max/choices 不合法：报错（保留原值，交由校验报告展示）
    - 未知键：warning（不阻断，容忍旧拓扑/前端扩展字段）
    """
    out: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []
    known = {a.name: a for a in op.attrs}
    for a in op.attrs:
        if a.name not in attrs:
            errors.append(f"缺少属性「{a.name}」（{a.label}），已回填默认值 {pylit(a.default)}")
            out[a.name] = list(a.default) if isinstance(a.default, list) else a.default
            continue
        v = attrs[a.name]
        bad = _check_attr_value(a, v)
        if bad:
            errors.append(bad)
            out[a.name] = list(a.default) if isinstance(a.default, list) else a.default
        else:
            out[a.name] = v
    for k in attrs:
        if k not in known:
            warnings.append(f"未知属性「{k}」（算子 {op.op} 未定义），生成代码将忽略")
    return out, errors, warnings


def _check_attr_value(a: AttrDef, v: Any) -> str | None:
    """校验单个属性值，返回错误消息或 None。"""
    t = a.type
    if t == "int":
        if isinstance(v, bool) or not isinstance(v, int):
            if isinstance(v, float) and v.is_integer():
                pass  # 4.0 → 容忍
            else:
                return f"属性「{a.name}」应为整数，实际 {pylit(v)}（{type(v).__name__}）"
    elif t == "float":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return f"属性「{a.name}」应为数字，实际 {pylit(v)}（{type(v).__name__}）"
    elif t == "bool":
        if not isinstance(v, bool):
            return f"属性「{a.name}」应为布尔值，实际 {pylit(v)}（{type(v).__name__}）"
    elif t == "str":
        if not isinstance(v, str):
            return f"属性「{a.name}」应为字符串，实际 {pylit(v)}（{type(v).__name__}）"
    elif t == "enum":
        if not isinstance(v, str) or (a.choices and v not in a.choices):
            return f"属性「{a.name}」应为 {'/'.join(a.choices or [])} 之一，实际 {pylit(v)}"
    elif t == "int_list":
        if not isinstance(v, (list, tuple)):
            return f"属性「{a.name}」应为整数列表，实际 {pylit(v)}（{type(v).__name__}）"
        for item in v:
            if isinstance(item, bool) or not isinstance(item, int):
                return f"属性「{a.name}」应为整数列表，实际含 {pylit(item)}（{type(item).__name__}）"
    if t in ("int", "float") and isinstance(v, (int, float)) and not isinstance(v, bool):
        if a.min is not None and v < a.min:
            return f"属性「{a.name}」不能小于 {a.min}，实际 {v}"
        if a.max is not None and v > a.max:
            return f"属性「{a.name}」不能大于 {a.max}，实际 {v}"
    return None


def pylit(v: Any) -> str:
    """把属性值格式化成 Python 字面量（模板里直接嵌入）。"""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return repr(list(v))
    return str(v)


# ---------------------------------------------------------------------------
# 通用形状规则小工具
# ---------------------------------------------------------------------------

def _identity(_attrs: dict, ins: list[list[Dim]]) -> list[list[Dim]]:
    return [list(ins[0])]


def _linear_last(attrs: dict, ins: list[list[Dim]], in_f: int, out_f: int) -> list[list[Dim]]:
    s = ins[0]
    assert_min_rank(s, 1, "Linear")
    assert_dim(s, -1, in_f, "Linear")
    return [list(s[:-1]) + [out_f]]


def _seq_like(attrs: dict, ins: list[list[Dim]], feature: int) -> list[list[Dim]]:
    s = ins[0]
    assert_rank(s, 3, "序列模块")
    assert_dim(s, -1, feature, "序列模块")
    return [list(s)]


def _softmax_shape(attrs: dict, ins: list[list[Dim]]) -> list[list[Dim]]:
    s = ins[0]
    d = attrs["dim"]
    rank = len(s)
    norm = d + rank if d < 0 else d
    if not (0 <= norm < rank):
        raise ShapeError(f"Softmax: dim={d} 超出输入秩 {rank}（{fmt_shape(s)}）")
    return [list(s)]


def _reshape_shape(attrs: dict, ins: list[list[Dim]]) -> list[list[Dim]]:
    """Reshape：-1 至多一个；能算元素数时校验整除；keep_batch 与 fwd 模板一致。"""
    s = ins[0]
    target = list(attrs["target_shape"])
    for d in target:
        if isinstance(d, bool) or not isinstance(d, int):
            raise ShapeError(f"Reshape: target_shape 只支持整数（-1 表示推断），实际 {target}")
        if d < -1:
            raise ShapeError(f"Reshape: target_shape 含非法维度 {d}（仅 -1 表示自动推断）")
    n_inf = sum(1 for d in target if d == -1)
    if n_inf > 1:
        raise ShapeError("Reshape: target_shape 最多只能有一个 -1（自动推断维）")
    keep = bool(attrs["keep_batch"])
    if keep and len(s) < 1:
        raise ShapeError(f"Reshape: keep_batch=True 需要至少 1 维输入，实际 {fmt_shape(s)}")
    out: list[Dim] = [s[0]] if keep else []
    out += [d if d != -1 else "inferred" for d in target]
    # 元素数校验（输入全为静态维时可算）
    if all(isinstance(d, int) and not isinstance(d, bool) for d in s):
        n_in = 1
        for d in s:
            n_in *= int(d)
        known = 1
        for d in out:
            if d != "inferred":
                known *= int(d)
        if n_inf == 0:
            if n_in != known:
                raise ShapeError(
                    f"Reshape: 元素数不匹配，输入 {fmt_shape(s)}（{n_in}）→ 目标 {fmt_shape(out)}（{known}）")
        elif known == 0 or n_in % known != 0:
            raise ShapeError(
                f"Reshape: -1 无法整除推断，输入 {fmt_shape(s)}（{n_in}）→ 目标 {fmt_shape(out)}")
    return [out]


def _divisible(dim_key: str, heads_key: str, op: str) -> Callable[[dict[str, Any]], None]:
    """attr_check：embed_dim/d_model 必须能被头数整除。"""

    def check(a: dict[str, Any]) -> None:
        dim, heads = a[dim_key], a[heads_key]
        if heads and dim % heads != 0:
            raise ShapeError(f"{op}: {dim_key}={dim} 必须能被 {heads_key}={heads} 整除")

    return check


def _rope_attr_check(a: dict[str, Any]) -> None:
    if a["dim"] % 2 != 0:
        raise ShapeError(f"RoPE: dim 必须为偶数，实际 {a['dim']}")


def _posemb_attr_check(a: dict[str, Any]) -> None:
    if a["kind"] == "sinusoidal" and a["d_model"] % 2 != 0:
        raise ShapeError(f"PositionalEncoding: kind=sinusoidal 要求 d_model 为偶数，实际 {a['d_model']}")


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------

OPS: list[OpDef] = [
    OpDef(
        op="Linear",
        group="basic",
        label="线性层 Linear",
        doc="y = xW + b，最后一维做特征变换。",
        attrs=[
            AttrDef("in_features", "int", 768, "输入特征数", min=1),
            AttrDef("out_features", "int", 768, "输出特征数", min=1),
            AttrDef("bias", "bool", True, "使用偏置"),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: _linear_last(a, ins, a["in_features"], a["out_features"]),
        init_tpl="self.{{ var }} = nn.Linear({{ attrs.in_features }}, {{ attrs.out_features }}, bias={{ attrs.bias }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="Conv1d",
        group="basic",
        label="一维卷积 Conv1d",
        doc="输入 [N, C, L]，常用于文本/序列特征。",
        attrs=[
            AttrDef("in_channels", "int", 64, "输入通道", min=1),
            AttrDef("out_channels", "int", 64, "输出通道", min=1),
            AttrDef("kernel_size", "int", 3, "卷积核", min=1),
            AttrDef("stride", "int", 1, "步长", min=1),
            AttrDef("padding", "int", 0, "填充", min=0),
            AttrDef("dilation", "int", 1, "膨胀", min=1),
            AttrDef("bias", "bool", True, "使用偏置"),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_rank(ins[0], 3, "Conv1d")
            or assert_dim(ins[0], 1, a["in_channels"], "Conv1d")
            or [[ins[0][0], a["out_channels"], conv_out_dim(ins[0][2], a["kernel_size"], a["stride"], a["padding"], a["dilation"], "Conv1d")]]
        ),
        init_tpl="self.{{ var }} = nn.Conv1d({{ attrs.in_channels }}, {{ attrs.out_channels }}, kernel_size={{ attrs.kernel_size }}, stride={{ attrs.stride }}, padding={{ attrs.padding }}, dilation={{ attrs.dilation }}, bias={{ attrs.bias }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="Conv2d",
        group="basic",
        label="二维卷积 Conv2d",
        doc="输入 [N, C, H, W]，图像特征提取。",
        attrs=[
            AttrDef("in_channels", "int", 1, "输入通道", min=1),
            AttrDef("out_channels", "int", 32, "输出通道", min=1),
            AttrDef("kernel_size", "int", 3, "卷积核", min=1),
            AttrDef("stride", "int", 1, "步长", min=1),
            AttrDef("padding", "int", 1, "填充", min=0),
            AttrDef("dilation", "int", 1, "膨胀", min=1),
            AttrDef("bias", "bool", True, "使用偏置"),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_rank(ins[0], 4, "Conv2d")
            or assert_dim(ins[0], 1, a["in_channels"], "Conv2d")
            or [[
                ins[0][0],
                a["out_channels"],
                conv_out_dim(ins[0][2], a["kernel_size"], a["stride"], a["padding"], a["dilation"], "Conv2d"),
                conv_out_dim(ins[0][3], a["kernel_size"], a["stride"], a["padding"], a["dilation"], "Conv2d"),
            ]]
        ),
        init_tpl="self.{{ var }} = nn.Conv2d({{ attrs.in_channels }}, {{ attrs.out_channels }}, kernel_size={{ attrs.kernel_size }}, stride={{ attrs.stride }}, padding={{ attrs.padding }}, dilation={{ attrs.dilation }}, bias={{ attrs.bias }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="Embedding",
        group="basic",
        label="词嵌入 Embedding",
        doc="token id → 向量。输入 int64 的 [..., seq]，输出 [..., seq, embedding_dim]。",
        attrs=[
            AttrDef("num_embeddings", "int", 32000, "词表大小", min=1),
            AttrDef("embedding_dim", "int", 768, "嵌入维度", min=1),
        ],
        inputs=[PortDef("ids", "int64")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: [list(ins[0]) + [a["embedding_dim"]]],
        init_tpl="self.{{ var }} = nn.Embedding({{ attrs.num_embeddings }}, {{ attrs.embedding_dim }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.ids }})",
    ),
    OpDef(
        op="LayerNorm",
        group="basic",
        label="层归一化 LayerNorm",
        doc="对最后一维做归一化。",
        attrs=[AttrDef("normalized_shape", "int", 768, "归一化维度", min=1),
               AttrDef("eps", "float", 1e-5, "数值稳定项")],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_dim(ins[0], -1, a["normalized_shape"], "LayerNorm") or [list(ins[0])]
        ),
        init_tpl="self.{{ var }} = nn.LayerNorm({{ attrs.normalized_shape }}, eps={{ attrs.eps }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="RMSNorm",
        group="basic",
        label="均方根归一化 RMSNorm",
        doc="LLM 常用归一化（LLaMA 风格），生成代码内含 RMSNorm 实现。",
        attrs=[AttrDef("normalized_shape", "int", 768, "归一化维度", min=1),
               AttrDef("eps", "float", 1e-6, "数值稳定项")],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_dim(ins[0], -1, a["normalized_shape"], "RMSNorm") or [list(ins[0])]
        ),
        init_tpl="self.{{ var }} = RMSNorm({{ attrs.normalized_shape }}, eps={{ attrs.eps }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
        helper="rmsnorm",
    ),
    OpDef(
        op="BatchNorm1d",
        group="basic",
        label="批归一化 BatchNorm1d",
        doc="输入 [N, C, L] 或 [N, C]。",
        attrs=[AttrDef("num_features", "int", 64, "通道数", min=1),
               AttrDef("eps", "float", 1e-5, "数值稳定项"),
               AttrDef("momentum", "float", 0.1, "动量")],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_dim(ins[0], 1, a["num_features"], "BatchNorm1d") or [list(ins[0])]
        ),
        init_tpl="self.{{ var }} = nn.BatchNorm1d({{ attrs.num_features }}, eps={{ attrs.eps }}, momentum={{ attrs.momentum }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="BatchNorm2d",
        group="basic",
        label="批归一化 BatchNorm2d",
        doc="输入 [N, C, H, W]。",
        attrs=[AttrDef("num_features", "int", 32, "通道数", min=1),
               AttrDef("eps", "float", 1e-5, "数值稳定项"),
               AttrDef("momentum", "float", 0.1, "动量")],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_dim(ins[0], 1, a["num_features"], "BatchNorm2d") or [list(ins[0])]
        ),
        init_tpl="self.{{ var }} = nn.BatchNorm2d({{ attrs.num_features }}, eps={{ attrs.eps }}, momentum={{ attrs.momentum }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="Dropout",
        group="basic",
        label="Dropout",
        doc="训练时随机置零，推理自动关闭。",
        attrs=[AttrDef("p", "float", 0.1, "丢弃概率", min=0.0, max=1.0)],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=_identity,
        init_tpl="self.{{ var }} = nn.Dropout({{ attrs.p }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="ReLU",
        group="basic",
        label="ReLU",
        doc="max(0, x)。",
        attrs=[],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=_identity,
        init_tpl="self.{{ var }} = nn.ReLU()",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="GELU",
        group="basic",
        label="GELU",
        doc="Transformer 常用激活函数。",
        attrs=[AttrDef("approximate", "enum", "none", "近似方式", choices=["none", "tanh"])],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=_identity,
        init_tpl="self.{{ var }} = nn.GELU(approximate={{ attrs.approximate }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="SiLU",
        group="basic",
        label="SiLU / Swish",
        doc="x * sigmoid(x)。",
        attrs=[],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=_identity,
        init_tpl="self.{{ var }} = nn.SiLU()",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="Tanh",
        group="basic",
        label="Tanh",
        doc="双曲正切激活。",
        attrs=[],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=_identity,
        init_tpl="self.{{ var }} = nn.Tanh()",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="Sigmoid",
        group="basic",
        label="Sigmoid",
        doc="把输出压到 (0, 1)。",
        attrs=[],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=_identity,
        init_tpl="self.{{ var }} = nn.Sigmoid()",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="Softmax",
        group="basic",
        label="Softmax",
        doc="在指定轴上归一化为概率分布。",
        attrs=[AttrDef("dim", "int", -1, "作用轴")],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=_softmax_shape,
        init_tpl="self.{{ var }} = nn.Softmax(dim={{ attrs.dim }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="MaxPool2d",
        group="basic",
        label="最大池化 MaxPool2d",
        doc="输入 [N, C, H, W]。",
        attrs=[AttrDef("kernel_size", "int", 2, "池化窗口", min=1),
               AttrDef("stride", "int", 2, "步长", min=1),
               AttrDef("padding", "int", 0, "填充", min=0)],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_rank(ins[0], 4, "MaxPool2d")
            or [[
                ins[0][0], ins[0][1],
                conv_out_dim(ins[0][2], a["kernel_size"], a["stride"], a["padding"], 1, "MaxPool2d"),
                conv_out_dim(ins[0][3], a["kernel_size"], a["stride"], a["padding"], 1, "MaxPool2d"),
            ]]
        ),
        init_tpl="self.{{ var }} = nn.MaxPool2d(kernel_size={{ attrs.kernel_size }}, stride={{ attrs.stride }}, padding={{ attrs.padding }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="AvgPool2d",
        group="basic",
        label="平均池化 AvgPool2d",
        doc="输入 [N, C, H, W]。",
        attrs=[AttrDef("kernel_size", "int", 2, "池化窗口", min=1),
               AttrDef("stride", "int", 2, "步长", min=1),
               AttrDef("padding", "int", 0, "填充", min=0)],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_rank(ins[0], 4, "AvgPool2d")
            or [[
                ins[0][0], ins[0][1],
                conv_out_dim(ins[0][2], a["kernel_size"], a["stride"], a["padding"], 1, "AvgPool2d"),
                conv_out_dim(ins[0][3], a["kernel_size"], a["stride"], a["padding"], 1, "AvgPool2d"),
            ]]
        ),
        init_tpl="self.{{ var }} = nn.AvgPool2d(kernel_size={{ attrs.kernel_size }}, stride={{ attrs.stride }}, padding={{ attrs.padding }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="Flatten",
        group="basic",
        label="展平 Flatten",
        doc="把 [start_dim, end_dim] 区间合并为一维。",
        attrs=[AttrDef("start_dim", "int", 1, "起始轴"),
               AttrDef("end_dim", "int", -1, "结束轴")],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: [
            flatten_shape(ins[0], a["start_dim"] if a["start_dim"] >= 0 else len(ins[0]) + a["start_dim"],
                          a["end_dim"] if a["end_dim"] >= 0 else len(ins[0]) + a["end_dim"], "Flatten")
        ],
        init_tpl="self.{{ var }} = nn.Flatten(start_dim={{ attrs.start_dim }}, end_dim={{ attrs.end_dim }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="Reshape",
        group="basic",
        label="重塑 Reshape",
        doc="保持 batch 维（可选），把其余维度重塑为 target_shape；-1 表示自动推断。",
        attrs=[AttrDef("target_shape", "int_list", [768], "目标形状"),
               AttrDef("keep_batch", "bool", True, "保留 batch 维")],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=_reshape_shape,
        init_tpl="# Reshape 无参数",
        fwd_tpl=(
            "{% if attrs.keep_batch %}"
            "{{ out.y }} = {{ in.x }}.reshape([{{ in.x }}.shape[0]] + {{ attrs.target_shape }})"
            "{% else %}"
            "{{ out.y }} = {{ in.x }}.reshape({{ attrs.target_shape }})"
            "{% endif %}"
        ),
    ),
    OpDef(
        op="Concat",
        group="basic",
        label="拼接 Concat",
        doc="把两个同秩张量沿指定轴拼接。",
        attrs=[AttrDef("dim", "int", -1, "拼接轴")],
        inputs=[PortDef("x1", "float32"), PortDef("x2", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: [
            _concat_shape(ins, a["dim"])
        ],
        init_tpl="# Concat 无参数",
        fwd_tpl="{{ out.y }} = torch.cat([{{ in.x1 }}, {{ in.x2 }}], dim={{ attrs.dim }})",
    ),
    OpDef(
        op="Add",
        group="basic",
        label="残差相加 Add",
        doc="两个形状相同的张量逐元素相加（残差连接）。",
        attrs=[],
        inputs=[PortDef("x1", "float32"), PortDef("x2", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: [merge_shapes(ins[0], ins[1], "Add")],
        init_tpl="# Add 无参数",
        fwd_tpl="{{ out.y }} = {{ in.x1 }} + {{ in.x2 }}",
    ),
    OpDef(
        op="MultiHeadAttention",
        group="basic",
        label="多头注意力 MHA",
        doc="自注意力（q=k=v=输入）。输入输出形状 [seq, batch, emb]（batch_first=False）或 [batch, seq, emb]（True）。",
        attrs=[
            AttrDef("embed_dim", "int", 768, "嵌入维度", min=1),
            AttrDef("num_heads", "int", 8, "头数", min=1),
            AttrDef("dropout", "float", 0.0, "注意力 dropout", min=0.0, max=1.0),
            AttrDef("bias", "bool", True, "使用偏置"),
            AttrDef("batch_first", "bool", True, "batch 在前"),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: _seq_like(a, ins, a["embed_dim"]),
        init_tpl="self.{{ var }} = nn.MultiheadAttention({{ attrs.embed_dim }}, {{ attrs.num_heads }}, dropout={{ attrs.dropout }}, batch_first={{ attrs.batch_first }}, bias={{ attrs.bias }})",
        fwd_tpl=(
            "{{ out.y }}, _ = self.{{ var }}({{ in.x }}, {{ in.x }}, {{ in.x }}, need_weights=False)"
        ),
        attr_check=_divisible("embed_dim", "num_heads", "MultiHeadAttention"),
    ),
    OpDef(
        op="LSTM",
        group="basic",
        label="LSTM",
        doc="循环网络。输入 [.., seq, input_size]（支持 2 维非批 [seq, input_size]），输出最后一维变为 hidden_size * 方向数。",
        attrs=[
            AttrDef("input_size", "int", 768, "输入特征", min=1),
            AttrDef("hidden_size", "int", 256, "隐藏维度", min=1),
            AttrDef("num_layers", "int", 1, "层数", min=1),
            AttrDef("bidirectional", "bool", False, "双向"),
            AttrDef("batch_first", "bool", True, "batch 在前"),
            AttrDef("dropout", "float", 0.0, "层间 dropout", min=0.0, max=1.0),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: _rnn_shape(a, ins, "LSTM"),
        init_tpl="self.{{ var }} = nn.LSTM({{ attrs.input_size }}, {{ attrs.hidden_size }}, num_layers={{ attrs.num_layers }}, batch_first={{ attrs.batch_first }}, dropout={{ attrs.dropout }} if {{ attrs.num_layers }} > 1 else 0.0, bidirectional={{ attrs.bidirectional }})",
        fwd_tpl="{{ out.y }}, _ = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="GRU",
        group="basic",
        label="GRU",
        doc="轻量循环网络，结构同 LSTM。",
        attrs=[
            AttrDef("input_size", "int", 768, "输入特征", min=1),
            AttrDef("hidden_size", "int", 256, "隐藏维度", min=1),
            AttrDef("num_layers", "int", 1, "层数", min=1),
            AttrDef("bidirectional", "bool", False, "双向"),
            AttrDef("batch_first", "bool", True, "batch 在前"),
            AttrDef("dropout", "float", 0.0, "层间 dropout", min=0.0, max=1.0),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: _rnn_shape(a, ins, "GRU"),
        init_tpl="self.{{ var }} = nn.GRU({{ attrs.input_size }}, {{ attrs.hidden_size }}, num_layers={{ attrs.num_layers }}, batch_first={{ attrs.batch_first }}, dropout={{ attrs.dropout }} if {{ attrs.num_layers }} > 1 else 0.0, bidirectional={{ attrs.bidirectional }})",
        fwd_tpl="{{ out.y }}, _ = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="RoPE",
        group="basic",
        label="旋转位置编码 RoPE",
        doc="对 [.., seq, dim] 的最后一维做旋转位置编码（dim 需为偶数）。",
        attrs=[
            AttrDef("dim", "int", 64, "特征维度（偶数）", min=2),
            AttrDef("base", "float", 10000.0, "频率基数"),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_dim(ins[0], -1, a["dim"], "RoPE") or [list(ins[0])]
        ),
        init_tpl="self.{{ var }} = RotaryEmbedding({{ attrs.dim }}, base={{ attrs.base }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
        helper="rope",
        attr_check=_rope_attr_check,
    ),
    # -----------------------------------------------------------------------
    # 大模型模块
    # -----------------------------------------------------------------------
    OpDef(
        op="TransformerEncoderLayer",
        group="llm",
        label="Transformer 编码层",
        doc="自注意力 + 前馈 + 残差 + 归一化的完整编码层。输入 [.., seq, d_model]。",
        attrs=[
            AttrDef("d_model", "int", 768, "模型维度", min=1),
            AttrDef("nhead", "int", 8, "注意力头数", min=1),
            AttrDef("dim_feedforward", "int", 3072, "前馈维度", min=1),
            AttrDef("dropout", "float", 0.1, "dropout", min=0.0, max=1.0),
            AttrDef("activation", "enum", "gelu", "激活函数", choices=["relu", "gelu"]),
            AttrDef("layer_norm_eps", "float", 1e-5, "LayerNorm eps"),
            AttrDef("batch_first", "bool", True, "batch 在前"),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: _seq_like(a, ins, a["d_model"]),
        init_tpl="self.{{ var }} = nn.TransformerEncoderLayer({{ attrs.d_model }}, {{ attrs.nhead }}, dim_feedforward={{ attrs.dim_feedforward }}, dropout={{ attrs.dropout }}, activation={{ attrs.activation }}, layer_norm_eps={{ attrs.layer_norm_eps }}, batch_first={{ attrs.batch_first }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
        attr_check=_divisible("d_model", "nhead", "TransformerEncoderLayer"),
    ),
    OpDef(
        op="TransformerDecoderLayer",
        group="llm",
        label="Transformer 解码层",
        doc="带交叉注意力的解码层。tgt/memory 均为 [.., seq, d_model]（seq 可不同）。",
        attrs=[
            AttrDef("d_model", "int", 768, "模型维度", min=1),
            AttrDef("nhead", "int", 8, "注意力头数", min=1),
            AttrDef("dim_feedforward", "int", 3072, "前馈维度", min=1),
            AttrDef("dropout", "float", 0.1, "dropout", min=0.0, max=1.0),
            AttrDef("activation", "enum", "gelu", "激活函数", choices=["relu", "gelu"]),
            AttrDef("layer_norm_eps", "float", 1e-5, "LayerNorm eps"),
            AttrDef("batch_first", "bool", True, "batch 在前"),
        ],
        inputs=[PortDef("tgt", "float32"), PortDef("memory", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            _seq_like(a, [ins[0]], a["d_model"])
            if not (assert_rank(ins[1], 3, "TransformerDecoderLayer")
                    or assert_dim(ins[1], -1, a["d_model"], "TransformerDecoderLayer"))
            else [list(ins[0])]
        ),
        init_tpl="self.{{ var }} = nn.TransformerDecoderLayer({{ attrs.d_model }}, {{ attrs.nhead }}, dim_feedforward={{ attrs.dim_feedforward }}, dropout={{ attrs.dropout }}, activation={{ attrs.activation }}, layer_norm_eps={{ attrs.layer_norm_eps }}, batch_first={{ attrs.batch_first }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.tgt }}, {{ in.memory }})",
        attr_check=_divisible("d_model", "nhead", "TransformerDecoderLayer"),
    ),
    OpDef(
        op="MLP",
        group="llm",
        label="前馈网络 MLP/FFN",
        doc="Linear → 激活 → Dropout → Linear，Transformer 内部的 FFN。",
        attrs=[
            AttrDef("dim", "int", 768, "输入/输出维度", min=1),
            AttrDef("hidden_dim", "int", 3072, "隐藏维度", min=1),
            AttrDef("activation", "enum", "gelu", "激活函数", choices=["relu", "gelu", "silu"]),
            AttrDef("dropout", "float", 0.0, "dropout", min=0.0, max=1.0),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_dim(ins[0], -1, a["dim"], "MLP") or [list(ins[0])]
        ),
        init_tpl=(
            "self.{{ var }} = nn.Sequential(\n"
            "    nn.Linear({{ attrs.dim }}, {{ attrs.hidden_dim }}),\n"
            "    {% if attrs.activation == 'relu' %}nn.ReLU(){% elif attrs.activation == 'silu' %}nn.SiLU(){% else %}nn.GELU(){% endif %},\n"
            "    nn.Dropout({{ attrs.dropout }}),\n"
            "    nn.Linear({{ attrs.hidden_dim }}, {{ attrs.dim }}),\n"
            ")"
        ),
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="PositionalEncoding",
        group="llm",
        label="位置编码",
        doc="正弦或可学习位置编码。输入 [.., seq, d_model]（batch_first=True）。",
        attrs=[
            AttrDef("d_model", "int", 768, "模型维度", min=1),
            AttrDef("max_len", "int", 512, "最大长度", min=1),
            AttrDef("kind", "enum", "sinusoidal", "编码类型", choices=["sinusoidal", "learned"]),
            AttrDef("dropout", "float", 0.0, "dropout", min=0.0, max=1.0),
            AttrDef("batch_first", "bool", True, "batch 在前"),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: _seq_like(a, ins, a["d_model"]),
        init_tpl="self.{{ var }} = PositionalEncoding({{ attrs.d_model }}, max_len={{ attrs.max_len }}, kind={{ attrs.kind }}, dropout={{ attrs.dropout }}, batch_first={{ attrs.batch_first }})",
        fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
        helper="posemb",
        attr_check=_posemb_attr_check,
    ),
    OpDef(
        op="TokenEmbedding",
        group="llm",
        label="Token 嵌入",
        doc="LLM 输入嵌入层，可选 √d_model 缩放。输入 int64 的 [.., seq]。",
        attrs=[
            AttrDef("vocab_size", "int", 32000, "词表大小", min=1),
            AttrDef("d_model", "int", 768, "嵌入维度", min=1),
            AttrDef("scale", "bool", True, "按 √d_model 缩放"),
        ],
        inputs=[PortDef("ids", "int64")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: [list(ins[0]) + [a["d_model"]]],
        init_tpl="self.{{ var }} = nn.Embedding({{ attrs.vocab_size }}, {{ attrs.d_model }})",
        fwd_tpl=(
            "{{ out.y }} = self.{{ var }}({{ in.ids }})"
            "{% if attrs.scale %} * ({{ attrs.d_model }} ** 0.5){% endif %}"
        ),
    ),
    OpDef(
        op="LMHead",
        group="llm",
        label="语言模型头 LM Head",
        doc="把隐藏向量映射回词表 logits。输入 [.., d_model]，输出 [.., vocab_size]。",
        attrs=[
            AttrDef("d_model", "int", 768, "输入维度", min=1),
            AttrDef("vocab_size", "int", 32000, "词表大小", min=1),
            AttrDef("bias", "bool", False, "使用偏置"),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("logits", "float32")],
        shape_rule=lambda a, ins: (
            assert_dim(ins[0], -1, a["d_model"], "LMHead")
            or [list(ins[0][:-1]) + [a["vocab_size"]]]
        ),
        init_tpl="self.{{ var }} = nn.Linear({{ attrs.d_model }}, {{ attrs.vocab_size }}, bias={{ attrs.bias }})",
        fwd_tpl="{{ out.logits }} = self.{{ var }}({{ in.x }})",
    ),
    OpDef(
        op="ViTBlock",
        group="llm",
        label="ViT Block",
        doc="Vision Transformer 标准块：Pre-LN 自注意力 + MLP + 双残差。输入 [batch, seq, dim]。",
        attrs=[
            AttrDef("dim", "int", 768, "特征维度", min=1),
            AttrDef("num_heads", "int", 8, "注意力头数", min=1),
            AttrDef("mlp_ratio", "float", 4.0, "MLP 扩展比", min=1.0),
            AttrDef("dropout", "float", 0.0, "dropout", min=0.0, max=1.0),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: _seq_like(a, ins, a["dim"]),
        init_tpl=(
            "self.{{ var }}_norm1 = nn.LayerNorm({{ attrs.dim }})\n"
            "self.{{ var }}_attn = nn.MultiheadAttention({{ attrs.dim }}, {{ attrs.num_heads }}, dropout={{ attrs.dropout }}, batch_first=True)\n"
            "self.{{ var }}_norm2 = nn.LayerNorm({{ attrs.dim }})\n"
            "self.{{ var }}_mlp = nn.Sequential(\n"
            "    nn.Linear({{ attrs.dim }}, int({{ attrs.dim }} * {{ attrs.mlp_ratio }})),\n"
            "    nn.GELU(),\n"
            "    nn.Dropout({{ attrs.dropout }}),\n"
            "    nn.Linear(int({{ attrs.dim }} * {{ attrs.mlp_ratio }}), {{ attrs.dim }}),\n"
            ")"
        ),
        fwd_tpl=(
            "_h = self.{{ var }}_norm1({{ in.x }})\n"
            "_h, _ = self.{{ var }}_attn(_h, _h, _h, need_weights=False)\n"
            "{{ out.y }} = {{ in.x }} + _h + self.{{ var }}_mlp(self.{{ var }}_norm2({{ in.x }} + _h))"
        ),
        attr_check=_divisible("dim", "num_heads", "ViTBlock"),
    ),
    OpDef(
        op="ResNetBlock",
        group="llm",
        label="ResNet Block",
        doc="两个 3x3 卷积 + BN + 残差的经典块。输入 [N, C, H, W]。",
        attrs=[
            AttrDef("in_channels", "int", 64, "输入通道", min=1),
            AttrDef("out_channels", "int", 64, "输出通道", min=1),
            AttrDef("stride", "int", 1, "步长", min=1),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_rank(ins[0], 4, "ResNetBlock")
            or assert_dim(ins[0], 1, a["in_channels"], "ResNetBlock")
            or [[
                ins[0][0], a["out_channels"],
                conv_out_dim(ins[0][2], 3, a["stride"], 1, 1, "ResNetBlock"),
                conv_out_dim(ins[0][3], 3, a["stride"], 1, 1, "ResNetBlock"),
            ]]
        ),
        init_tpl=(
            "self.{{ var }}_conv1 = nn.Conv2d({{ attrs.in_channels }}, {{ attrs.out_channels }}, 3, stride={{ attrs.stride }}, padding=1, bias=False)\n"
            "self.{{ var }}_bn1 = nn.BatchNorm2d({{ attrs.out_channels }})\n"
            "self.{{ var }}_conv2 = nn.Conv2d({{ attrs.out_channels }}, {{ attrs.out_channels }}, 3, padding=1, bias=False)\n"
            "self.{{ var }}_bn2 = nn.BatchNorm2d({{ attrs.out_channels }})\n"
            "self.{{ var }}_skip = nn.Identity() if ({{ attrs.stride }} == 1 and {{ attrs.in_channels }} == {{ attrs.out_channels }}) else nn.Sequential(\n"
            "    nn.Conv2d({{ attrs.in_channels }}, {{ attrs.out_channels }}, 1, stride={{ attrs.stride }}, bias=False),\n"
            "    nn.BatchNorm2d({{ attrs.out_channels }}),\n"
            ")\n"
            "self.{{ var }}_act = nn.ReLU()"
        ),
        fwd_tpl=(
            "_h = self.{{ var }}_act(self.{{ var }}_bn1(self.{{ var }}_conv1({{ in.x }})))\n"
            "_h = self.{{ var }}_bn2(self.{{ var }}_conv2(_h))\n"
            "{{ out.y }} = self.{{ var }}_act(_h + self.{{ var }}_skip({{ in.x }}))"
        ),
    ),
    OpDef(
        op="LoRAAdapter",
        group="llm",
        label="LoRA 适配器",
        doc="低秩适配：y = x + (alpha/rank) * up(down(x))，up 零初始化。",
        attrs=[
            AttrDef("dim", "int", 768, "特征维度", min=1),
            AttrDef("rank", "int", 8, "低秩秩数", min=1),
            AttrDef("alpha", "float", 16.0, "缩放系数 alpha"),
            AttrDef("dropout", "float", 0.0, "dropout", min=0.0, max=1.0),
        ],
        inputs=[PortDef("x", "float32")],
        outputs=[PortDef("y", "float32")],
        shape_rule=lambda a, ins: (
            assert_dim(ins[0], -1, a["dim"], "LoRAAdapter") or [list(ins[0])]
        ),
        init_tpl=(
            "self.{{ var }}_down = nn.Linear({{ attrs.dim }}, {{ attrs.rank }}, bias=False)\n"
            "self.{{ var }}_up = nn.Linear({{ attrs.rank }}, {{ attrs.dim }}, bias=False)\n"
            "self.{{ var }}_drop = nn.Dropout({{ attrs.dropout }})\n"
            "nn.init.zeros_(self.{{ var }}_up.weight)  # LoRA up 零初始化，初始等价恒等\n"
            "self.{{ var }}_scale = {{ attrs.alpha }} / {{ attrs.rank }}"
        ),
        fwd_tpl=(
            "{{ out.y }} = {{ in.x }} + self.{{ var }}_scale * self.{{ var }}_up(self.{{ var }}_down(self.{{ var }}_drop({{ in.x }})))"
        ),
    ),
]


# ---------------------------------------------------------------------------
# 形状规则里用到的多输入辅助
# ---------------------------------------------------------------------------

def _concat_shape(ins: list[list[Dim]], dim: int) -> list[Dim]:
    a, b = ins[0], ins[1]
    if len(a) != len(b):
        raise ShapeError(f"Concat: 形状秩不匹配（{fmt_shape(a)} vs {fmt_shape(b)}）")
    axis = dim if dim >= 0 else len(a) + dim
    if not (0 <= axis < len(a)):
        raise ShapeError(f"Concat: 轴 {dim} 不存在，输入形状 {fmt_shape(a)}")
    out: list[Dim] = []
    for i in range(len(a)):
        if i == axis:
            if isinstance(a[i], int) and isinstance(b[i], int):
                out.append(int(a[i]) + int(b[i]))
            else:
                out.append(f"cat_{a[i]}_{b[i]}")
        else:
            # torch.cat 不广播：非拼接轴必须精确同形
            out.append(merge_shapes([a[i]], [b[i]], "Concat", broadcast=False)[0])
    return out


def _rnn_shape(attrs: dict, ins: list[list[Dim]], op: str) -> list[list[Dim]]:
    """torch 2.x 支持非批输入 [L, H_in]（2 维）与批输入 [.., L/S, H_in]（3 维+）。"""
    s = ins[0]
    assert_min_rank(s, 2, op)
    assert_dim(s, -1, attrs["input_size"], op)
    directions = 2 if attrs["bidirectional"] else 1
    return [list(s[:-1]) + [attrs["hidden_size"] * directions]]


# ---------------------------------------------------------------------------
# 辅助类代码（注入到生成的 model.py）
# ---------------------------------------------------------------------------

HELPERS: dict[str, str] = {
    "rmsnorm": '''
class RMSNorm(nn.Module):
    """均方根归一化（LLaMA 风格）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight
''',
    "rope": '''
class RotaryEmbedding(nn.Module):
    """旋转位置编码：对 [.., seq, dim] 的最后一维两两配对旋转。"""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE 维度必须为偶数")
        inv_freq = base ** (-torch.arange(0, dim // 2, dtype=torch.float32) / (dim // 2))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq, dim/2]
        cos = freqs.cos().view([1] * (x.dim() - 2) + list(freqs.shape))
        sin = freqs.sin().view([1] * (x.dim() - 2) + list(freqs.shape))
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out
''',
    "posemb": '''
class PositionalEncoding(nn.Module):
    """正弦 / 可学习位置编码。"""

    def __init__(self, d_model: int, max_len: int = 512, kind: str = "sinusoidal",
                 dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        if kind == "learned":
            self.pe = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.normal_(self.pe, std=0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            x = x + self.pe[: x.size(1)].unsqueeze(0)
        else:
            x = x + self.pe[: x.size(0)].unsqueeze(1)
        return self.dropout(x)
''',
}

OP_INDEX: dict[str, OpDef] = {o.op: o for o in OPS}
