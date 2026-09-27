"""符号形状代数与形状推导的辅助函数。

维度要么是 int（静态维），要么是 str（符号维，如 "batch"、"seq"）。
规则：int 与 int 必须相等；符号与任何维都兼容（宽松匹配，展示用）。
"""
from __future__ import annotations

from typing import Sequence

from .schema import Dim


class ShapeError(Exception):
    """形状推导失败（携带人类可读的原因）。"""


def dims_compatible(a: Dim, b: Dim) -> bool:
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return True  # 至少一边是符号：宽松通过


def dims_equal(a: Dim, b: Dim) -> bool:
    return a == b


def shapes_compatible(a: Sequence[Dim], b: Sequence[Dim]) -> bool:
    return len(a) == len(b) and all(dims_compatible(x, y) for x, y in zip(a, b))


def fmt_shape(shape: Sequence[Dim]) -> str:
    return "[" + ", ".join(str(d) for d in shape) + "]"


def assert_rank(shape: Sequence[Dim], rank: int, op: str) -> None:
    if len(shape) != rank:
        raise ShapeError(f"{op}: 期望 {rank} 维输入，实际 {fmt_shape(shape)}")


def assert_min_rank(shape: Sequence[Dim], rank: int, op: str) -> None:
    if len(shape) < rank:
        raise ShapeError(f"{op}: 期望至少 {rank} 维输入，实际 {fmt_shape(shape)}")


def assert_dim(shape: Sequence[Dim], axis: int, value: int, op: str) -> None:
    """断言 shape[axis] == value（value 为 int 时）。支持负轴。"""
    n = len(shape)
    norm = axis + n if axis < 0 else axis
    if not (0 <= norm < n):
        raise ShapeError(f"{op}: 轴 {axis} 不存在，实际形状 {fmt_shape(shape)}")
    actual = shape[norm]
    if isinstance(actual, int) and actual != value:
        raise ShapeError(f"{op}: 轴 {axis} 期望 {value}，实际 {actual}")


def merge_dim(a: Dim, b: Dim, op: str, axis: int, broadcast: bool = True) -> Dim:
    """逐元素类算子的维度合并（对齐 torch 广播语义）。

    - int vs int：相等直接取；broadcast 时一方为 1 取另一方；否则报错
    - str vs str：必须同名（不同符号在运行期未必相等，宁可报错也不静默取值）
    - str vs int：int 为 1 广播取符号，否则取 int（宽松匹配，展示用）
    """
    if isinstance(a, int) and isinstance(b, int):
        if a == b:
            return a
        if broadcast and a == 1:
            return b
        if broadcast and b == 1:
            return a
        raise ShapeError(f"{op}: 轴 {axis} 维度不匹配（{a} vs {b}）")
    if isinstance(a, str) and isinstance(b, str):
        if a == b:
            return a
        raise ShapeError(f"{op}: 轴 {axis} 符号维冲突（{a} vs {b}，两个不同符号在运行期未必相等）")
    s, i = (a, b) if isinstance(a, str) else (b, a)
    return s if i == 1 else i


def merge_shapes(a: Sequence[Dim], b: Sequence[Dim], op: str, broadcast: bool = True) -> list[Dim]:
    """逐元素合并两个形状；broadcast=True 时右对齐补 1（torch 广播）。

    broadcast=False 用于 Concat 等要求精确同形的场景（torch.cat 不广播）。
    """
    ra, rb = list(a), list(b)
    if len(ra) != len(rb):
        if not broadcast:
            raise ShapeError(f"{op}: 形状秩不匹配（{fmt_shape(a)} vs {fmt_shape(b)}）")
        if len(ra) < len(rb):
            ra = [1] * (len(rb) - len(ra)) + ra
        else:
            rb = [1] * (len(ra) - len(rb)) + rb
    return [merge_dim(x, y, op, i, broadcast) for i, (x, y) in enumerate(zip(ra, rb))]


def conv_out_dim(size: Dim, kernel: int, stride: int, padding: int, dilation: int, op: str) -> Dim:
    """卷积/池化输出尺寸公式；输入是符号时输出用携带卷积参数的派生符号表示。

    不同 kernel/stride/padding/dilation 得到不同派生符号，下游逐元素合并时
    能检出参数不一致的分支（运行期长度必然不同）。
    """
    if isinstance(size, str):
        return f"{size}_k{kernel}_s{stride}_p{padding}_d{dilation}"
    out = (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if out <= 0:
        raise ShapeError(f"{op}: 输入尺寸 {size} 配合 kernel={kernel} stride={stride} padding={padding} 输出为空")
    return out


def flatten_shape(shape: Sequence[Dim], start_dim: int, end_dim: int, op: str) -> list[Dim]:
    if start_dim < 0 or end_dim >= len(shape) or start_dim > end_dim:
        raise ShapeError(f"{op}: 非法的 flatten 范围 [{start_dim}, {end_dim}]，形状 {fmt_shape(shape)}")
    head = list(shape[:start_dim])
    tail = list(shape[end_dim + 1 :])
    mid = shape[start_dim : end_dim + 1]
    if all(isinstance(d, int) for d in mid):
        merged: Dim = 1
        for d in mid:
            merged = merged * int(d)  # type: ignore[operator]
    else:
        merged = "_x_".join(str(d) for d in mid)
    return head + [merged] + tail
