"""后端冒烟测试：建图 → 校验 → 生成代码 → 实际前向运行。

覆盖验收点：
  1. Embedding→TransformerEncoderLayer→Linear→Softmax 图校验通过、形状推导正确
  2. shape 不匹配的连线（768 vs 3072）被拒绝并给出原因
  3. 生成的 model.py 可直接前向运行，输出 shape 正确
  4. 自定义模块（Linear+GELU+Linear）生成独立 class 并被组合引用
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.codegen.generator import CodegenError, generate_package  # noqa: E402
from app.ir.schema import Edge, Graph, ModelMeta, ModuleDef, Node, Port, PortRef  # noqa: E402
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
    """自定义模块 FFN：Linear+GELU+Linear。"""
    inner = Graph(
        inputs=[p("x", "float32", ["batch", 64])],
        outputs=[p("y", "float32", ["batch", 64])],
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
                     inputs=[p("x", "float32", ["batch", 64])],
                     outputs=[p("y", "float32", ["batch", 64])],
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

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
