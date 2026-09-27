"""图校验与形状推导引擎。

约定：边可以引用两个虚拟端点——
  源 `"__graph_in__"`  : 图级输入（端口名 = graph.inputs[i].name）
  目标 `"__graph_out__"`: 图级输出（端口名 = graph.outputs[i].name）

职责：
1. 结构校验：DAG 无环、端口引用有效、输入端口有连接
2. 形状推导：从图级输入出发按拓扑序传播 type/shape（支持符号维）
3. 连线兼容性：dtype 一致 + 形状兼容（int 相等 / 符号宽松）
4. 输出拓扑排序（代码生成直接复用）
"""
from __future__ import annotations

from typing import Any

from ..registry.ops import OP_INDEX, OpDef
from .schema import Dim, Graph, ModuleDef, Node, Port
from .shapes import ShapeError, fmt_shape

GRAPH_IN = "__graph_in__"
GRAPH_OUT = "__graph_out__"


def _port_map(ports: list[Port]) -> dict[str, Port]:
    return {p.name: p for p in ports}


def node_port_specs(node: Node, modules: dict[str, ModuleDef]) -> tuple[list[Port], list[Port]]:
    """节点的输入/输出端口规格（自定义模块取模块定义的接口）。"""
    if node.op.startswith("custom:"):
        mod = modules.get(node.op.split(":", 1)[1])
        if mod is None:
            raise ShapeError(f"自定义模块不存在: {node.op}")
        return mod.inputs, mod.outputs
    op = OP_INDEX.get(node.op)
    if op is None:
        raise ShapeError(f"未知算子: {node.op}")
    return (
        [Port(name=p.name, dtype=p.dtype) for p in op.inputs],  # type: ignore[arg-type]
        [Port(name=p.name, dtype=p.dtype) for p in op.outputs],  # type: ignore[arg-type]
    )


def validate_graph(graph: Graph, modules: dict[str, ModuleDef] | None = None) -> dict[str, Any]:
    modules = modules or {}
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def err(where: str, msg: str) -> None:
        errors.append({"where": where, "message": msg})

    def warn(where: str, msg: str) -> None:
        warnings.append({"where": where, "message": msg})

    nodes = {n.id: n for n in graph.nodes}
    if len(nodes) != len(graph.nodes):
        err("graph", "节点 id 重复")

    port_specs: dict[str, tuple[list[Port], list[Port]]] = {}
    for n in graph.nodes:
        try:
            port_specs[n.id] = node_port_specs(n, modules)
        except ShapeError as e:
            err(n.name or n.id, str(e))

    in_names = [p.name for p in graph.inputs]
    if len(set(in_names)) != len(in_names):
        err("graph.inputs", "图级输入端口重名")
    out_names = [p.name for p in graph.outputs]
    if len(set(out_names)) != len(out_names):
        err("graph.outputs", "图级输出端口重名")

    # ---- 边引用与连线兼容性 ---------------------------------------------
    edge_by_target: dict[tuple[str, str], Any] = {}
    for e in graph.edges:
        src_id, tgt_id = e.source.node, e.target.node

        if src_id == GRAPH_IN:
            src_port = _port_map(graph.inputs).get(e.source.port)
            if src_port is None:
                err(f"边 {e.id}", f"模型没有输入「{e.source.port}」")
                continue
            src_dtype, src_shape = src_port.dtype, list(src_port.shape)
        elif src_id in nodes and src_id in port_specs:
            sp = _port_map(port_specs[src_id][1]).get(e.source.port)
            if sp is None:
                err(f"边 {e.id}", f"节点 {nodes[src_id].name or src_id} 没有输出端口 {e.source.port}")
                continue
            src_dtype = _declared_dtype(nodes[src_id], e.source.port, "output", sp.dtype)
            src_shape = _declared_shape(nodes[src_id], e.source.port, "output")
        else:
            err(f"边 {e.id}", f"引用了不存在的源节点 {src_id}")
            continue

        if tgt_id == GRAPH_OUT:
            tp = _port_map(graph.outputs).get(e.target.port)
            if tp is None:
                err(f"边 {e.id}", f"模型没有输出「{e.target.port}」")
                continue
            tgt_dtype, tgt_shape = tp.dtype, list(tp.shape)
        elif tgt_id in nodes and tgt_id in port_specs:
            tp = _port_map(port_specs[tgt_id][0]).get(e.target.port)
            if tp is None:
                err(f"边 {e.id}", f"节点 {nodes[tgt_id].name or tgt_id} 没有输入端口 {e.target.port}")
                continue
            tgt_dtype = _declared_dtype(nodes[tgt_id], e.target.port, "input", tp.dtype)
            tgt_shape = _declared_shape(nodes[tgt_id], e.target.port, "input")
            key = (tgt_id, e.target.port)
            if key in edge_by_target:
                err(f"边 {e.id}", f"输入端口 {nodes[tgt_id].name or tgt_id}.{e.target.port} 被多条边连接")
                continue
            edge_by_target[key] = e
        else:
            err(f"边 {e.id}", f"引用了不存在的目标节点 {tgt_id}")
            continue

        if src_dtype != tgt_dtype:
            err(f"边 {e.id}", f"数据类型不匹配：{src_dtype} → {tgt_dtype}")
        if src_shape is not None and tgt_shape:
            if len(src_shape) != len(tgt_shape):
                err(f"边 {e.id}", f"形状秩不匹配：{fmt_shape(src_shape)} → {fmt_shape(tgt_shape)}")
            else:
                for i, (a, b) in enumerate(zip(src_shape, tgt_shape)):
                    if isinstance(a, int) and isinstance(b, int) and a != b:
                        err(f"边 {e.id}", f"第 {i} 维不匹配：{a} → {b}（{fmt_shape(src_shape)} → {fmt_shape(tgt_shape)}）")
                        break

    # ---- 输入端口必须有连接 ---------------------------------------------
    for n in graph.nodes:
        if n.id not in port_specs:
            continue
        for p in port_specs[n.id][0]:
            if (n.id, p.name) not in edge_by_target:
                err(n.name or n.id, f"输入端口「{p.name}」未连接")

    # ---- 图级输出必须有来源（空图也不豁免：未接线的模型不是合法模型） ----
    out_sources: dict[str, Any] = {}
    for e in graph.edges:
        if e.target.node == GRAPH_OUT:
            if e.target.port in out_sources:
                err(f"边 {e.id}", f"模型输出「{e.target.port}」被多条边连接")
            out_sources[e.target.port] = e
    for p in graph.outputs:
        if p.name not in out_sources:
            err("graph.outputs", f"模型输出「{p.name}」没有连线")

    # ---- 拓扑排序 + 形状推导 --------------------------------------------
    order: list[str] = []
    inferred: dict[tuple[str, str, str], tuple[str, list[Dim]]] = {}
    if not errors:
        try:
            order = topo_sort(graph, edge_by_target)
        except ShapeError as e:
            err("graph", str(e))

    if order:
        env: dict[tuple[str, str], tuple[str, list[Dim]]] = {
            (GRAPH_IN, p.name): (p.dtype, list(p.shape)) for p in graph.inputs
        }
        for nid in order:
            n = nodes[nid]
            ins, outs = port_specs[nid]
            in_shapes: list[list[Dim]] = []
            ok = True
            for p in ins:
                e = edge_by_target.get((nid, p.name))
                if e is None:
                    ok = False
                    break
                src = env.get((e.source.node, e.source.port))
                if src is None:
                    ok = False
                    break
                decl = _declared_shape(n, p.name, "input")
                if decl is not None:
                    _check_declared(decl, src[1], n, p.name, "输入", err)
                in_shapes.append(list(src[1]))
                inferred[(nid, p.name, "input")] = (src[0], list(src[1]))
            if not ok:
                continue

            try:
                if n.op.startswith("custom:"):
                    mod = modules[n.op.split(":", 1)[1]]
                    out_shapes = [list(p.shape) for p in mod.outputs]
                    out_dtypes: list[str] = [p.dtype for p in mod.outputs]
                else:
                    op: OpDef = OP_INDEX[n.op]
                    out_shapes = op.shape_rule(dict(n.attrs), in_shapes)
                    out_dtypes = [p.dtype for p in op.outputs]  # type: ignore[attr-defined]
            except ShapeError as e:
                err(n.name or n.id, str(e))
                continue
            except Exception as e:
                err(n.name or n.id, f"形状推导失败: {e}")
                continue

            for p, shape, dt in zip(outs, out_shapes, out_dtypes):
                decl = _declared_shape(n, p.name, "output")
                if decl is not None:
                    conflict = _check_declared(decl, shape, n, p.name, "输出", err)
                    final = decl if not conflict else shape
                else:
                    final = shape
                env[(n.id, p.name)] = (dt, list(final))
                inferred[(n.id, p.name, "output")] = (dt, list(final))

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "nodes": [
            {
                "id": nid,
                "inputs": [
                    {"name": k[1], "dtype": v[0], "shape": v[1]}
                    for k, v in inferred.items()
                    if k[0] == nid and k[2] == "input"
                ],
                "outputs": [
                    {"name": k[1], "dtype": v[0], "shape": v[1]}
                    for k, v in inferred.items()
                    if k[0] == nid and k[2] == "output"
                ],
            }
            for nid in nodes
        ],
        "order": order,
    }


def topo_sort(graph: Graph, edge_by_target: dict[tuple[str, str], Any]) -> list[str]:
    nodes = {n.id: n for n in graph.nodes}
    deps: dict[str, set[str]] = {nid: set() for nid in nodes}
    for e in graph.edges:
        if e.source.node in nodes and e.target.node in nodes and e.source.node != e.target.node:
            deps[e.target.node].add(e.source.node)
    order: list[str] = []
    ready = sorted(nid for nid, d in deps.items() if not d)
    pending = {nid: set(d) for nid, d in deps.items() if d}
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for other in sorted(list(pending)):
            pending[other].discard(nid)
            if not pending[other]:
                ready.append(other)
                del pending[other]
    if pending:
        raise ShapeError("图中存在环（循环依赖）：" + "、".join(nodes[n].name or n for n in pending))
    return order


# ---------------------------------------------------------------------------
# 内部小工具
# ---------------------------------------------------------------------------

def _declared_dtype(node: Node, port: str, kind: str, default: str) -> str:
    ports = node.inputs if kind == "input" else node.outputs
    p = _port_map(ports).get(port)
    return (p.dtype if p else default) or default


def _declared_shape(node: Node, port: str, kind: str) -> list[Dim] | None:
    """节点端口上用户填写的形状；空列表视为未声明。"""
    ports = node.inputs if kind == "input" else node.outputs
    p = _port_map(ports).get(port)
    if p is None or not p.shape:
        return None
    return list(p.shape)


def _check_declared(decl: list[Dim], actual: list[Dim], node: Node, port: str, kind: str, err) -> bool:
    """声明形状与推导/上游形状冲突检查；返回是否冲突。"""
    conflict = False
    if len(decl) != len(actual):
        err(node.name or node.id,
            f"{kind}端口「{port}」声明形状 {fmt_shape(decl)} 与实际 {fmt_shape(actual)} 秩不一致")
        return True
    for i, (a, b) in enumerate(zip(decl, actual)):
        if isinstance(a, int) and isinstance(b, int) and a != b:
            err(node.name or node.id,
                f"{kind}端口「{port}」声明形状 {fmt_shape(decl)} 与实际 {fmt_shape(actual)} 冲突（第 {i} 维 {a} vs {b}）")
            conflict = True
            break
        if isinstance(a, str) and isinstance(b, str) and a != b:
            err(node.name or node.id,
                f"{kind}端口「{port}」声明形状 {fmt_shape(decl)} 与实际 {fmt_shape(actual)} 冲突（第 {i} 维 {a} vs {b}）")
            conflict = True
            break
    return conflict
