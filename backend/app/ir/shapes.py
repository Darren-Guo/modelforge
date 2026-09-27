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
    """断言 shape[axis] == value（value 为 int 时）。"""
    if axis >= len(shape):
        raise ShapeError(f"{op}: 轴 {axis} 不存在，实际形状 {fmt_shape(shape)}")
    actual = shape[axis]
    if isinstance(actual, int) and actual != value:
        raise ShapeError(f"{op}: 轴 {axis} 期望 {value}，实际 {actual}")


def merge_dim(a: Dim, b: Dim, op: str, axis: int) -> Dim:
    """逐元素类算子的维度合并：int 相等直接取，符号与 int 取 int。"""
    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            raise ShapeError(f"{op}: 轴 {axis} 维度不匹配（{a} vs {b}）")
        return a
    return a if isinstance(a, int) else b


def merge_shapes(a: Sequence[Dim], b: Sequence[Dim], op: str) -> list[Dim]:
    if len(a) != len(b):
        raise ShapeError(f"{op}: 形状秩不匹配（{fmt_shape(a)} vs {fmt_shape(b)}）")
    return [merge_dim(x, y, op, i) for i, (x, y) in enumerate(zip(a, b))]


def conv_out_dim(size: Dim, kernel: int, stride: int, padding: int, dilation: int, op: str) -> Dim:
    """卷积/池化输出尺寸公式；输入是符号时输出用派生符号表示。"""
    if isinstance(size, str):
        return f"{size}_out"
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
