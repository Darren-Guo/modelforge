"""后端冒烟测试：建图 → 校验 → 生成代码 → 实际前向运行。

覆盖验收点：
  1. Embedding→TransformerEncoderLayer→Linear→Softmax 图校验通过、形状推导正确
  2. shape 不匹配的连线（768 vs 3072）被拒绝并给出原因
  3. 生成的 model.py 可直接前向运行，输出 shape 正确
  4. 自定义模块（Linear+GELU+Linear）生成独立 class 并被组合引用
  5. 回归：空图校验失败且 codegen 报干净错误
  6. 回归：自环/循环依赖被拒绝，不产出引用未赋值变量的 forward
  7. 回归：attrs 缺键/非法取值被 validate 拒绝，片段仍渲染合法代码
  8. 回归：模块接口与内部推导交叉核对；含 custom 节点的嵌套模块可生成；改接口后实例同步
  9. 回归：逐元素合并的符号维/广播语义；卷积派生符号携带参数
  10. 回归：Edge 序列化契约固定 from/to
  11. 回归：多输出模型的 train.py 可运行
  12. 回归：sanitize 命名碰撞去重，模型类名不覆盖 helper 类名
  13. 回归：LSTM/GRU 接受 torch 2.x 合法的 2 维非批输入
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.codegen.generator import CodegenError, generate_package, node_snippet  # noqa: E402
from app.ir.schema import Edge, Graph, ModelMeta, ModuleDef, Node, Port, PortRef  # noqa: E402
from app.ir.shapes import ShapeError, merge_shapes  # noqa: E402
from app.ir.validate import validate_graph  # noqa: E402


def p(name, dtype="float32", shape=None):
    return Port(name=name, dtype=dtype, shape=shape or [])


def sample_graph() -> Graph:
    return Graph(
        model=ModelMeta(name="MyTransformer", description="冒烟测试模型"),
        inputs=[p("ids", "int64", ["batch", "seq"])],
        outputs=[p("y", "float32", ["batch", "seq", 10])],
        nodes=[
            Node(id="n0", name="Embedding_1", op="Embedding", group="basic",
                 attrs={"num_embeddings": 1000, "embedding_dim": 64},
                 inputs=[p("ids", "int64")], outputs=[p("y")], ui={"x": 0, "y": 0}),
            Node(id="n1", name="Encoder_1", op="TransformerEncoderLayer", group="llm",
                 attrs={"d_model": 64, "nhead": 4, "dim_feedforward": 128, "dropout": 0.1,
                        "activation": "gelu", "layer_norm_eps": 1e-5, "batch_first": True},
                 inputs=[p("x")], outputs=[p("y")], ui={"x": 200, "y": 0}),
            Node(id="n2", name="Linear_1", op="Linear", group="basic",
                 attrs={"in_features": 64, "out_features": 10, "bias": True},
                 inputs=[p("x")], outputs=[p("y")], ui={"x": 400, "y": 0}),
            Node(id="n3", name="Softmax_1", op="Softmax", group="basic",
                 attrs={"dim": -1},
                 inputs=[p("x")], outputs=[p("y")], ui={"x": 600, "y": 0}),
        ],
        edges=[
            Edge(id="e0", **{"from": PortRef(node="__graph_in__", port="ids"), "to": PortRef(node="n0", port="ids")}),
            Edge(id="e1", **{"from": PortRef(node="n0", port="y"), "to": PortRef(node="n1", port="x")}),
            Edge(id="e2", **{"from": PortRef(node="n1", port="y"), "to": PortRef(node="n2", port="x")}),
            Edge(id="e3", **{"from": PortRef(node="n2", port="y"), "to": PortRef(node="n3", port="x")}),
            Edge(id="e4", **{"from": PortRef(node="n3", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )


def bad_graph() -> Graph:
    """768 → 3072 的非法连线。"""
    g = sample_graph()
    g.nodes[1].attrs["d_model"] = 3072  # 与 Embedding 输出 64 冲突
    return g


def ffn_module() -> ModuleDef:
    """自定义模块 FFN：Linear+GELU+Linear，作用在 [..., 64] 的最后一维。

    接口声明与内部图输入输出声明保持一致（[batch, seq, 64]）：
    校验器会核对上游实际形状、接口声明与内部推导三者一致。
    """
    inner = Graph(
        inputs=[p("x", "float32", ["batch", "seq", 64])],
        outputs=[p("y", "float32", ["batch", "seq", 64])],
        nodes=[
            Node(id="m0", name="Linear_up", op="Linear", group="basic",
                 attrs={"in_features": 64, "out_features": 128, "bias": True},
                 inputs=[p("x")], outputs=[p("y")]),
            Node(id="m1", name="GELU_1", op="GELU", group="basic",
                 attrs={"approximate": "none"},
                 inputs=[p("x")], outputs=[p("y")]),
            Node(id="m2", name="Linear_down", op="Linear", group="basic",
                 attrs={"in_features": 128, "out_features": 64, "bias": True},
                 inputs=[p("x")], outputs=[p("y")]),
        ],
        edges=[
            Edge(id="f0", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="m0", port="x")}),
            Edge(id="f1", **{"from": PortRef(node="m0", port="y"), "to": PortRef(node="m1", port="x")}),
            Edge(id="f2", **{"from": PortRef(node="m1", port="y"), "to": PortRef(node="m2", port="x")}),
            Edge(id="f3", **{"from": PortRef(node="m2", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    return ModuleDef(id="mod_ffn", name="FFN", description="前馈块",
                     inputs=[p("x", "float32", ["batch", "seq", 64])],
                     outputs=[p("y", "float32", ["batch", "seq", 64])],
                     graph=inner)


def custom_graph(mod: ModuleDef) -> Graph:
    """Embedding → 自定义FFN → Linear 的模型。"""
    return Graph(
        model=ModelMeta(name="CustomFFNModel"),
        inputs=[p("ids", "int64", ["batch", "seq"])],
        outputs=[p("y", "float32", ["batch", "seq", 4])],
        nodes=[
            Node(id="c0", name="Embedding_1", op="Embedding", group="basic",
                 attrs={"num_embeddings": 500, "embedding_dim": 64},
                 inputs=[p("ids", "int64")], outputs=[p("y")] ),
            Node(id="c1", name="FFN_1", op="custom:mod_ffn", group="custom",
                 attrs={}, inputs=[p("x")], outputs=[p("y")]),
            Node(id="c2", name="Linear_out", op="Linear", group="basic",
                 attrs={"in_features": 64, "out_features": 4, "bias": True},
                 inputs=[p("x")], outputs=[p("y")] ),
        ],
        edges=[
            Edge(id="g0", **{"from": PortRef(node="__graph_in__", port="ids"), "to": PortRef(node="c0", port="ids")}),
            Edge(id="g1", **{"from": PortRef(node="c0", port="y"), "to": PortRef(node="c1", port="x")}),
            Edge(id="g2", **{"from": PortRef(node="c1", port="y"), "to": PortRef(node="c2", port="x")}),
            Edge(id="g3", **{"from": PortRef(node="c2", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )


def run_forward_smoke(files: dict[str, str], class_name: str, test_code: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for name, content in files.items():
            (tdp / name).write_text(content, encoding="utf-8")
        script = test_code.replace("__CLASS__", class_name)
        r = subprocess.run([sys.executable, "-c", script], cwd=td, capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            raise AssertionError("生成代码前向冒烟失败")


def main() -> None:
    ok = True

    # 1) 主链路：校验
    g = sample_graph()
    rep = validate_graph(g)
    assert rep["ok"], f"校验应通过: {rep['errors']}"
    n1_out = [n for n in rep["nodes"] if n["id"] == "n1"][0]["outputs"][0]
    assert n1_out["shape"] == ["batch", "seq", 64], n1_out
    n2_out = [n for n in rep["nodes"] if n["id"] == "n2"][0]["outputs"][0]
    assert n2_out["shape"] == ["batch", "seq", 10], n2_out
    print("[PASS] 1. Embedding→Encoder→Linear→Softmax 校验通过，形状推导正确")

    # 2) 非法连线被拒绝
    rep2 = validate_graph(bad_graph())
    assert not rep2["ok"], "d_model=3072 应校验失败"
    assert any("3072" in e["message"] or "不匹配" in e["message"] for e in rep2["errors"]), rep2["errors"]
    print(f"[PASS] 2. 非法连线被拒绝：{rep2['errors'][0]['message']}")

    # 3) 生成 + 前向
    files = generate_package(g)
    assert set(files) == {"model.py", "train.py", "config.json", "README.md"}, files.keys()
    test = (
        "import torch\n"
        "from model import __CLASS__\n"
        "m = __CLASS__().eval()\n"
        "ids = torch.randint(0, 1000, (2, 7))\n"
        "y = m(ids)\n"
        "assert tuple(y.shape) == (2, 7, 10), y.shape\n"
        "assert abs(y.sum(-1) - 1).max().item() < 1e-4, 'softmax 输出应为概率'\n"
        "print('forward OK', tuple(y.shape))\n"
    )
    run_forward_smoke(files, "MyTransformer", test)
    print("[PASS] 3. 生成的 model.py 前向运行通过，输出 [2, 7, 10] 且为概率分布")

    # 4) 自定义模块
    mod = ffn_module()
    rep3 = validate_graph(mod.graph)
    assert rep3["ok"], rep3["errors"]
    cg = custom_graph(mod)
    rep4 = validate_graph(cg, {"mod_ffn": mod})
    assert rep4["ok"], rep4["errors"]
    files2 = generate_package(cg, {"mod_ffn": mod})
    assert "class CustomFFN" in files2["model.py"], "应生成自定义模块 class"
    assert "self.m_c1 = CustomFFN()" in files2["model.py"], "模型应组合引用自定义模块"
    test2 = (
        "import torch\n"
        "from model import __CLASS__\n"
        "m = __CLASS__().eval()\n"
        "ids = torch.randint(0, 500, (2, 5))\n"
        "y = m(ids)\n"
        "assert tuple(y.shape) == (2, 5, 4), y.shape\n"
        "print('custom forward OK', tuple(y.shape))\n"
    )
    run_forward_smoke(files2, "CustomFFNModel", test2)
    print("[PASS] 4. 自定义模块生成独立 class，组合引用，前向通过")

    # 5) 回归：空图（有声明输出但无连线）应校验失败，codegen 报干净错误而非 500
    empty = Graph(model=ModelMeta(name="EmptyModel"),
                  inputs=[p("x", "float32", ["batch", 8])],
                  outputs=[p("y", "float32", ["batch", 4])])
    rep5 = validate_graph(empty)
    assert not rep5["ok"], "空图应校验失败"
    assert any("没有连线" in e["message"] for e in rep5["errors"]), rep5["errors"]
    try:
        generate_package(empty)
        raise AssertionError("空图 codegen 应抛出 CodegenError")
    except CodegenError as e:
        assert e.report["errors"], e.report
    print("[PASS] 5. 回归：空图校验失败且 codegen 返回干净错误")

    # 6) 回归：自环被拒绝（曾漏检并生成引用未赋值变量的 forward）
    self_loop = Graph(
        model=ModelMeta(name="SelfLoop"),
        inputs=[p("x", "float32", [2, 4])],
        outputs=[p("y", "float32", [2, 4])],
        nodes=[Node(id="n0", name="Add_1", op="Add", group="basic", attrs={},
                    inputs=[p("x1"), p("x2")], outputs=[p("y")])],
        edges=[
            Edge(id="e0", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="n0", port="x1")}),
            Edge(id="e1", **{"from": PortRef(node="n0", port="y"), "to": PortRef(node="n0", port="x2")}),
            Edge(id="e2", **{"from": PortRef(node="n0", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    rep6 = validate_graph(self_loop)
    assert not rep6["ok"] and any("环" in e["message"] for e in rep6["errors"]), rep6["errors"]
    try:
        generate_package(self_loop)
        raise AssertionError("自环图 codegen 应抛出 CodegenError")
    except CodegenError as e:
        assert e.report["errors"], e.report
    print("[PASS] 6. 回归：自环被拒绝，codegen 报干净错误")

    # 7) 回归：attrs 缺键/非法值被拒绝；缺键节点的源码片段仍是合法 Python（曾生成 dim= 空值）
    bare = Graph(
        model=ModelMeta(name="BareSoftmax"),
        inputs=[p("x", "float32", ["batch", 8])],
        outputs=[p("y", "float32", ["batch", 8])],
        nodes=[Node(id="n0", name="Softmax_1", op="Softmax", group="basic", attrs={},
                    inputs=[p("x")], outputs=[p("y")])],
        edges=[
            Edge(id="e0", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="n0", port="x")}),
            Edge(id="e1", **{"from": PortRef(node="n0", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    rep7 = validate_graph(bare)
    assert not rep7["ok"] and any("dim" in e["message"] for e in rep7["errors"]), rep7["errors"]
    sn = node_snippet(bare, "n0")
    assert "nn.Softmax(dim=-1)" in sn["code"], sn["code"]
    ast.parse(sn["code"].replace("# __init__ 中：", "").replace("# forward 中：", ""))
    bad_gelu = Graph(
        model=ModelMeta(name="BadGELU"),
        inputs=[p("x", "float32", ["batch", 8])],
        outputs=[p("y", "float32", ["batch", 8])],
        nodes=[Node(id="n0", op="GELU", group="basic", attrs={"approximate": "bogus"},
                    inputs=[p("x")], outputs=[p("y")])],
        edges=bare.edges,
    )
    rep7b = validate_graph(bad_gelu)
    assert not rep7b["ok"], rep7b["errors"]
    print("[PASS] 7. 回归：attrs 缺键/非法值被拒绝，片段仍渲染合法代码")

    # 8) 回归：模块接口 ↔ 内部推导 ↔ 上游形状交叉核对；嵌套 custom 模块；改接口后实例同步
    liar_inner = Graph(
        inputs=[p("x", "float32", ["batch", 4])],
        outputs=[p("y", "float32", ["batch", 3])],  # 声明说谎：内部 Linear 4→8
        nodes=[Node(id="m0", op="Linear", group="basic",
                    attrs={"in_features": 4, "out_features": 8, "bias": True},
                    inputs=[p("x")], outputs=[p("y")])],
        edges=[
            Edge(id="f0", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="m0", port="x")}),
            Edge(id="f1", **{"from": PortRef(node="m0", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    liar = ModuleDef(id="mod_lie", name="Liar",
                     inputs=[p("x", "float32", ["batch", 4])],
                     outputs=[p("y", "float32", ["batch", 3])], graph=liar_inner)
    lie_graph = Graph(
        model=ModelMeta(name="LieModel"),
        inputs=[p("x", "float32", ["batch", 4])],
        outputs=[p("y", "float32", ["batch", 3])],
        nodes=[Node(id="c0", op="custom:mod_lie", group="custom", attrs={},
                    inputs=[p("x")], outputs=[p("y")])],
        edges=[
            Edge(id="g0", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="c0", port="x")}),
            Edge(id="g1", **{"from": PortRef(node="c0", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    rep8 = validate_graph(lie_graph, {"mod_lie": liar})
    assert not rep8["ok"], "接口声明 [batch,3] 与内部推导 [batch,8] 冲突应报错"
    assert any("3" in e["message"] or "8" in e["message"] for e in rep8["errors"]), rep8["errors"]

    # 上游喂与接口不符的形状也应报错（曾静默放行）
    bad_feed = Graph(
        model=ModelMeta(name="BadFeed"),
        inputs=[p("x", "float32", ["batch", 5])],
        outputs=[p("y", "float32", ["batch", 3])],
        nodes=lie_graph.nodes,
        edges=lie_graph.edges,
    )
    rep8b = validate_graph(bad_feed, {"mod_lie": liar})
    assert not rep8b["ok"], rep8b["errors"]

    # 嵌套模块（模块内含 custom 节点）可生成并前向；改接口后实例同步
    mod = ffn_module()
    wrap_inner = Graph(
        inputs=[p("x", "float32", ["batch", "seq", 64])],
        outputs=[p("y", "float32", ["batch", "seq", 64])],
        nodes=[Node(id="w0", op="custom:mod_ffn", group="custom", attrs={},
                    inputs=[p("x")], outputs=[p("y")])],
        edges=[
            Edge(id="w0e", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="w0", port="x")}),
            Edge(id="w1e", **{"from": PortRef(node="w0", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    wrap = ModuleDef(id="mod_wrap", name="FFNWrapper",
                     inputs=[p("x", "float32", ["batch", "seq", 64])],
                     outputs=[p("y", "float32", ["batch", "seq", 64])], graph=wrap_inner)
    nested = Graph(
        model=ModelMeta(name="NestedModel"),
        inputs=[p("ids", "int64", ["batch", "seq"])],
        outputs=[p("y", "float32", ["batch", "seq", 4])],
        nodes=[
            Node(id="c0", name="Embedding_1", op="Embedding", group="basic",
                 attrs={"num_embeddings": 500, "embedding_dim": 64},
                 inputs=[p("ids", "int64")], outputs=[p("y")]),
            Node(id="c1", name="Wrap_1", op="custom:mod_wrap", group="custom", attrs={},
                 inputs=[p("x")], outputs=[p("y")]),
            Node(id="c2", name="Linear_out", op="Linear", group="basic",
                 attrs={"in_features": 64, "out_features": 4, "bias": True},
                 inputs=[p("x")], outputs=[p("y")]),
        ],
        edges=[
            Edge(id="g0", **{"from": PortRef(node="__graph_in__", port="ids"), "to": PortRef(node="c0", port="ids")}),
            Edge(id="g1", **{"from": PortRef(node="c0", port="y"), "to": PortRef(node="c1", port="x")}),
            Edge(id="g2", **{"from": PortRef(node="c1", port="y"), "to": PortRef(node="c2", port="x")}),
            Edge(id="g3", **{"from": PortRef(node="c2", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    mods = {"mod_ffn": mod, "mod_wrap": wrap}
    rep8c = validate_graph(nested, mods)
    assert rep8c["ok"], rep8c["errors"]
    files3 = generate_package(nested, mods)
    assert "class CustomFFNWrapper" in files3["model.py"] and "class CustomFFN" in files3["model.py"], \
        "嵌套模块应生成两个独立 class"
    test3 = (
        "import torch\n"
        "from model import __CLASS__\n"
        "m = __CLASS__().eval()\n"
        "ids = torch.randint(0, 500, (2, 5))\n"
        "y = m(ids)\n"
        "assert tuple(y.shape) == (2, 5, 4), y.shape\n"
        "print('nested custom forward OK', tuple(y.shape))\n"
    )
    run_forward_smoke(files3, "NestedModel", test3)

    # 改模块接口后，实例按新接口校验（曾只信接口声明，改端口后不同步）
    mod2 = ffn_module()
    mod2.outputs = [p("y", "float32", ["batch", "seq", 16])]  # 与内部推导 64 冲突
    rep8d = validate_graph(custom_graph(ffn_module()), {"mod_ffn": mod2})
    assert not rep8d["ok"], "接口改动后实例应按新接口校验"
    print("[PASS] 8. 回归：模块接口交叉核对、嵌套 custom 模块生成前向、改接口后实例同步")

    # 9) 回归：合并语义（不同符号报错、广播放行、卷积派生符号带参数）
    try:
        merge_shapes(["batch", 768], ["seq", 768], "Add")
        raise AssertionError("不同符号维合并应报错")
    except ShapeError:
        pass
    assert merge_shapes([2, 4], [1, 4], "Add") == [2, 4], "torch 广播 1-vs-N 应放行"
    add_graph = Graph(
        model=ModelMeta(name="AddBroadcast"),
        inputs=[p("a", "float32", [2, 4]), p("b", "float32", [1, 4])],
        outputs=[p("y", "float32", [2, 4])],
        nodes=[Node(id="n0", op="Add", group="basic", attrs={},
                    inputs=[p("x1"), p("x2")], outputs=[p("y")])],
        edges=[
            Edge(id="e0", **{"from": PortRef(node="__graph_in__", port="a"), "to": PortRef(node="n0", port="x1")}),
            Edge(id="e1", **{"from": PortRef(node="__graph_in__", port="b"), "to": PortRef(node="n0", port="x2")}),
            Edge(id="e2", **{"from": PortRef(node="n0", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    rep9 = validate_graph(add_graph)
    out9 = [n for n in rep9["nodes"] if n["id"] == "n0"][0]["outputs"][0]["shape"]
    assert rep9["ok"] and out9 == [2, 4], (rep9["errors"], out9)
    # 不同 stride 的两个 Conv1d 输出相加：派生符号带参数，必须报错（曾同名 seq_out 被静默放行）
    convs = Graph(
        model=ModelMeta(name="ConvMix"),
        inputs=[p("x", "float32", ["batch", 4, "seq"])],
        outputs=[p("y", "float32", ["batch", 8, "seq"])],
        nodes=[
            Node(id="n1", op="Conv1d", group="basic",
                 attrs={"in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 1,
                        "padding": 0, "dilation": 1, "bias": True},
                 inputs=[p("x")], outputs=[p("y")]),
            Node(id="n3", op="Conv1d", group="basic",
                 attrs={"in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 3,
                        "padding": 0, "dilation": 1, "bias": True},
                 inputs=[p("x")], outputs=[p("y")]),
            Node(id="nA", op="Add", group="basic", attrs={},
                 inputs=[p("x1"), p("x2")], outputs=[p("y")]),
        ],
        edges=[
            Edge(id="e0", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="n1", port="x")}),
            Edge(id="e1", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="n3", port="x")}),
            Edge(id="e2", **{"from": PortRef(node="n1", port="y"), "to": PortRef(node="nA", port="x1")}),
            Edge(id="e3", **{"from": PortRef(node="n3", port="y"), "to": PortRef(node="nA", port="x2")}),
            Edge(id="e4", **{"from": PortRef(node="nA", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    rep9b = validate_graph(convs)
    assert not rep9b["ok"], "不同 stride 卷积输出相加应报错"
    print("[PASS] 9. 回归：符号维冲突报错、广播放行、卷积派生符号携带参数")

    # 10) 回归：Edge 序列化契约（wire 恒为 from/to，前端只认这两个名字）
    j_default = sample_graph().model_dump_json()
    j_alias = sample_graph().model_dump_json(by_alias=True)
    for j in (j_default, j_alias):
        assert '"from"' in j and '"to"' in j, j[-200:]
        assert '"source"' not in j and '"target"' not in j, j[-200:]
    e_parsed = Edge.model_validate({"id": "x", "from": {"node": "a", "port": "p"},
                                    "to": {"node": "b", "port": "q"}})
    assert e_parsed.source.node == "a" and e_parsed.target.port == "q"
    print("[PASS] 10. 回归：Edge 序列化恒为 from/to（by_alias 与否均一致）")

    # 11) 回归：多输出模型的 train.py 可运行（曾 tuple 输出直接喂 criterion 崩溃）
    multi = Graph(
        model=ModelMeta(name="MultiOut"),
        inputs=[p("x", "float32", [2, 4])],
        outputs=[p("o1", "float32", [2, 3]), p("o2", "float32", [2, 5])],
        nodes=[
            Node(id="n0", op="Linear", group="basic",
                 attrs={"in_features": 4, "out_features": 3, "bias": True},
                 inputs=[p("x")], outputs=[p("y")]),
            Node(id="n1", op="Linear", group="basic",
                 attrs={"in_features": 4, "out_features": 5, "bias": True},
                 inputs=[p("x")], outputs=[p("y")]),
        ],
        edges=[
            Edge(id="e0", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="n0", port="x")}),
            Edge(id="e1", **{"from": PortRef(node="n0", port="y"), "to": PortRef(node="__graph_out__", port="o1")}),
            Edge(id="e2", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="n1", port="x")}),
            Edge(id="e3", **{"from": PortRef(node="n1", port="y"), "to": PortRef(node="__graph_out__", port="o2")}),
        ],
    )
    files4 = generate_package(multi)
    ast.parse(files4["train.py"])
    ast.parse(files4["model.py"])
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for name, content in files4.items():
            (tdp / name).write_text(content, encoding="utf-8")
        r = subprocess.run([sys.executable, "train.py", "--dataset", "random",
                            "--epochs", "1", "--num-samples", "16", "--batch-size", "8",
                            "--out", str(tdp / "run")], cwd=td, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise AssertionError("多输出 train.py 应可运行")
    print("[PASS] 11. 回归：多输出模型 train.py 可运行（各输出分别求损失）")

    # 12) 回归：sanitize 命名碰撞去重；模型名不覆盖 helper 类名
    collide = Graph(
        model=ModelMeta(name="RMSNorm"),
        inputs=[p("x", "float32", [2, 4])],
        outputs=[p("y1", "float32", [2, 4]), p("y2", "float32", [2, 4])],
        nodes=[
            Node(id="a b", op="Linear", group="basic",
                 attrs={"in_features": 4, "out_features": 4, "bias": True},
                 inputs=[p("x")], outputs=[p("y")]),
            Node(id="a_b", op="RMSNorm", group="basic",
                 attrs={"normalized_shape": 4, "eps": 1e-6},
                 inputs=[p("x")], outputs=[p("y")]),
        ],
        edges=[
            Edge(id="e0", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="a b", port="x")}),
            Edge(id="e1", **{"from": PortRef(node="a b", port="y"), "to": PortRef(node="__graph_out__", port="y1")}),
            Edge(id="e2", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="a_b", port="x")}),
            Edge(id="e3", **{"from": PortRef(node="a_b", port="y"), "to": PortRef(node="__graph_out__", port="y2")}),
        ],
    )
    files5 = generate_package(collide)
    mvars = [ln.strip().split(" = ")[0] for ln in files5["model.py"].splitlines()
             if ln.strip().startswith("self.m_")]
    assert len(set(mvars)) == 2, f"碰撞节点应生成两个不同子模块变量：{mvars}"
    classes = [ln.split()[1].split("(")[0] for ln in files5["model.py"].splitlines()
               if ln.startswith("class ")]
    assert len(classes) == len(set(classes)), f"类名不应重复：{classes}"
    assert "RMSNormModel" in classes, f"模型名 RMSNorm 应改名避开 helper 类：{classes}"
    print(f"[PASS] 12. 回归：命名碰撞去重（{mvars}），模型类改名 {classes[-1]} 避开 helper")

    # 13) 回归：LSTM 接受 torch 2.x 合法的 2 维非批输入
    lstm = Graph(
        model=ModelMeta(name="LSTM2D"),
        inputs=[p("x", "float32", [10, 32])],
        outputs=[p("y", "float32", [10, 8])],
        nodes=[Node(id="n0", op="LSTM", group="basic",
                    attrs={"input_size": 32, "hidden_size": 8, "num_layers": 1,
                           "bidirectional": False, "batch_first": True, "dropout": 0.0},
                    inputs=[p("x")], outputs=[p("y")])],
        edges=[
            Edge(id="e0", **{"from": PortRef(node="__graph_in__", port="x"), "to": PortRef(node="n0", port="x")}),
            Edge(id="e1", **{"from": PortRef(node="n0", port="y"), "to": PortRef(node="__graph_out__", port="y")}),
        ],
    )
    rep13 = validate_graph(lstm)
    assert rep13["ok"] and rep13["nodes"][0]["outputs"][0]["shape"] == [10, 8], \
        (rep13["errors"], rep13["nodes"])
    print("[PASS] 13. 回归：LSTM 2 维非批输入通过校验，输出 [10, 8]")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
