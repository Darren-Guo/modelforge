"""Graph IR：拓扑 JSON 的 Pydantic schema（与前端 TS 类型一一对应）。

格式参考 ONNX 的图结构（nodes + edges + 带 type/shape 的端口），
但定位更高层：节点可以是基础算子，也可以是 TransformerBlock 级别的大模块。
"""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_serializer

# 维度：整数（静态维）或符号名（如 "batch"、"seq"）
Dim = Union[int, str]

Dtype = Literal["float32", "float16", "int64", "int32", "bool"]


class Port(BaseModel):
    """模块端口：名称 + 数据类型 + 形状，全部可由用户修改。

    dtype 缺省为 None（未声明）：节点端口回落到算子/模块规格的 dtype，
    图级端口回落 float32。显式声明永远优先。
    """

    name: str
    dtype: Dtype | None = None
    shape: list[Dim] = []


class Node(BaseModel):
    """图中的一个模块实例。"""

    id: str
    name: str = ""
    op: str  # 算子 key（如 "Linear"）或自定义模块 "custom:<module_id>"
    group: Literal["basic", "llm", "custom"] = "basic"
    attrs: dict[str, Any] = {}
    inputs: list[Port] = []
    outputs: list[Port] = []
    ui: dict[str, float] = Field(default_factory=dict)  # 画布坐标 x/y


class PortRef(BaseModel):
    node: str
    port: str


class Edge(BaseModel):
    """有向边：from 节点端口 → to 节点端口。

    wire 契约：JSON 字段名固定为 from/to（前端 GraphEdge 只认这两个名字）。
    Python 侧属性名保持 source/target；序列化（model_dump / model_dump_json，
    无论 by_alias）统一输出 from/to，防止新增调用点忘了 by_alias=True 而破坏契约。
    """

    id: str
    source: PortRef = Field(alias="from")
    target: PortRef = Field(alias="to")

    model_config = {"populate_by_name": True}

    @model_serializer(mode="wrap")
    def _ser_wire(self, handler):
        d = handler(self)
        if "source" in d:
            d["from"] = d.pop("source")
        if "target" in d:
            d["to"] = d.pop("target")
        return d


class ModelMeta(BaseModel):
    name: str = "MyModel"
    description: str = ""


class Graph(BaseModel):
    """完整拓扑图（导出 JSON/YAML 的顶层结构）。"""

    format: Literal["modelforge/graph"] = "modelforge/graph"
    version: str = "0.1"
    model: ModelMeta = Field(default_factory=ModelMeta)
    inputs: list[Port] = []
    outputs: list[Port] = []
    nodes: list[Node] = []
    edges: list[Edge] = []


class ModuleDef(BaseModel):
    """自定义模块定义：内部是一张子图，对外暴露可编辑的输入输出端口。"""

    id: str
    name: str
    description: str = ""
    inputs: list[Port] = []
    outputs: list[Port] = []
    graph: Graph = Field(default_factory=Graph)
    created_at: str = ""
    updated_at: str = ""


class PortUpdate(BaseModel):
    """前端回传的端口 shape/dtype 修改（批量应用到图上）。"""

    node_id: str | None = None  # None 表示图级输入/输出
    port_name: str
    port_kind: Literal["input", "output"]
    dtype: Dtype | None = None
    shape: list[Dim] | None = None
