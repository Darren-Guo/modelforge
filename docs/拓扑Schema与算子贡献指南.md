# 拓扑 Graph IR 与算子贡献指南

## 1. 拓扑 JSON（Graph IR）

顶层结构（Pydantic 定义见 `backend/app/ir/schema.py`，TS 镜像见 `frontend/src/schema/graph.ts`）：

```
Graph
├── format: "modelforge/graph"
├── version: "0.1"
├── model: {name, description}
├── inputs:  Port[]          # 模型级输入端口
├── outputs: Port[]          # 模型级输出端口
├── nodes:   Node[]
└── edges:   Edge[]
```

### Port
| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 端口名（同一模块内唯一） |
| `dtype` | `float32 \| float16 \| int64 \| int32 \| bool` | 数据类型 |
| `shape` | `(int \| string)[]` | 维度；int 为静态维，字符串为符号维（如 `"batch"`、`"seq"`） |

### Node
| 字段 | 说明 |
|---|---|
| `id` | 节点唯一 id |
| `name` | 显示名（可改） |
| `op` | 算子 key（如 `Linear`）或 `custom:<module_id>` |
| `group` | `basic \| llm \| custom` |
| `attrs` | 算子参数字典（类型由算子注册表定义） |
| `inputs` / `outputs` | 端口列表（shape 可手改，与推导冲突时校验报错） |
| `ui` | 画布坐标 `{x, y}` |

### Edge
`{"id", "from": {node, port}, "to": {node, port}}`，虚拟端点约定：
- 源 `"__graph_in__"`：图级输入（port = `inputs[i].name`）
- 目标 `"__graph_out__"`：图级输出（port = `outputs[i].name`）

### 校验规则（`backend/app/ir/validate.py`）
1. DAG 无环；输入端口恰好一条入边；模型输出必须有来源
2. 连线兼容：dtype 相等；形状秩相等；逐维 int 相等（符号维宽松通过）
3. 形状推导：从图级输入按拓扑序传播；节点上手改的 shape 作为断言与推导比对，冲突报错
4. 返回拓扑排序（代码生成直接复用）

## 2. 新增算子

只需改 `backend/app/registry/ops.py`，前端零改动（面板/属性表单/连线校验全部动态生成）。

```python
OpDef(
    op="SwiGLU",                      # 唯一 key
    group="llm",                      # basic | llm
    label="SwiGLU 门控 FFN",
    doc="y = (xW₁ ⊙ SiLU(xV)) W₂",
    attrs=[                           # 属性表单自动生成
        AttrDef("dim", "int", 768, "输入维度", min=1),
        AttrDef("hidden_dim", "int", 2048, "隐藏维度", min=1),
    ],
    inputs=[PortDef("x", "float32")],
    outputs=[PortDef("y", "float32")],
    shape_rule=lambda a, ins: (        # 抛 ShapeError 表示形状不合法
        assert_dim(ins[0], -1, a["dim"], "SwiGLU") or [list(ins[0])]
    ),
    init_tpl="self.{{ var }} = SwiGLU({{ attrs.dim }}, {{ attrs.hidden_dim }})",
    fwd_tpl="{{ out.y }} = self.{{ var }}({{ in.x }})",
    helper="swiglu",                   # 可选：注入 HELPERS 里的辅助类
)
```

模板上下文：
- `init_tpl`：`{{ var }}`（self.后的属性名）、`{{ attrs.xxx }}`（已格式化为 Python 字面量；支持 `{% if attrs.x == 'y' %}` 条件）
- `fwd_tpl`：额外有 `{{ in.端口名 }}` / `{{ out.端口名 }}`（张量变量名）
- 多行模板允许（如 ViTBlock 的多语句 forward）

改完重启后端即可；`GET /api/ops` 会带上新算子。

## 3. 自定义模块

- 存储：SQLite 表 `modules`（`backend/app/store.py`），`data` 字段为 ModuleDef JSON
- ModuleDef：`{id, name, description, inputs, outputs, graph}`，`graph` 内部子图同样遵循虚拟端点约定
- 代码生成：每个自定义模块生成独立 `class CustomXXX(nn.Module)`，模型内组合实例化；嵌套模块按依赖序生成

## 4. 代码生成管线

`backend/app/codegen/generator.py`：
1. `validate_graph` 通过（失败抛 `CodegenError` → HTTP 400 干净报错）
2. 拓扑排序 → 逐节点渲染 init/fwd Jinja2 模板
3. 组装文件：`model.py`（辅助类 + 自定义模块类 + 模型类）、`train.py`（自包含训练脚本，内置 mnist/text_cls/random 数据集）、`config.json`（拓扑快照）、`README.md`

生成代码自包含可独立运行：

```bash
python train.py --dataset random --epochs 5 --out ./run
python train.py --dataset mnist --epochs 3 --batch-size 64 --out ./run
```

训练按行输出 JSON 指标（`{"type":"metric",...}`），后端 SSE 采集实时展示；产物 `model.pt`（state_dict + 输入输出 spec + 任务类型）、`metrics.json`、`vocab.json`（文本任务）供推理服务加载。

## 5. 训练/推理约定

- 训练任务 = 子进程跑 run 目录内 `code/train.py` 的快照（训练时源码可回看）
- 模型注册表按 `model_id`（用户指定或自动生成）索引
- 推理输入自适应：`int64 + vocab.json` → 文本；`float32 [..,1,28,28]` → 图片；其他 → JSON 张量
