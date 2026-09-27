# ModelForge — 可视化模块化拼接模型

浏览器里的"拖拽搭模型"工作台：左侧算子/模块面板 → 中间画布连线组装 → 生成 ONNX 风格拓扑 JSON → 后端生成可运行 PyTorch 代码 → 一键训练/推理 → 双击任意模块/模型查看源码。

## 四个场景

| 场景 | 能力 | 入口 |
|---|---|---|
| 1. 创建模型 | 拖拽算子（基础算子/大模型模块/自定义模块 3 组，34 个算子）到画布连线；所有参数可改；端口 type/shape 可编辑；形状推导自动回填；非法连线实时拒绝 | 左面板 + 画布 + 右属性面板 |
| 2. 创建模块 | 框选子图 → 保存为自定义模块（输入输出接口全可编辑）→ 在任意模型中拖入复用；双击/右键回改定义 | 工具栏「保存为模块」 |
| 3. 训练推理 | 内置数据集（MNIST/文本分类/随机拟合）真实训练（自动 CUDA）→ SSE 实时日志 → 模型列表按模型 ID 查找 → 自适应输入表单推理（文本/图片/张量） | 工具栏「训练」「已训练模型」 |
| 4. 代码生成 | 拓扑 JSON/YAML 导出导入（参考 ONNX 图结构）→ Jinja2 模板生成自包含 PyTorch 包（model.py/train.py/config.json）→ 一键下载 zip；双击任意节点弹出对应粒度源码 | 工具栏「生成代码」、双击节点/画布空白 |

## 快速开始

```powershell
# 1. 后端（端口 8000）
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2. 前端（端口 5173，另开终端）
cd frontend
npm install
npm run dev
```

打开 http://127.0.0.1:5173 。也可以直接运行 `start.ps1` 一键启动两个服务。

> GPU：机器有 NVIDIA 显卡时 PyTorch 自动使用 CUDA；CPU 也能跑内置任务。

## 使用流程

1. **搭模型**：从左侧拖算子到画布（或点击放置），从输出端口拖到输入端口连线；右侧面板改参数/端口 shape；绿色"校验通过"徽标实时反馈。
2. **存模块**：Shift+点击多选节点 → 「保存为模块」→ 命名并定义对外接口 → 面板「自定义模块」组出现，可复用。
3. **看代码**：「生成代码」查看完整代码包（Monaco + 文件树 + zip 下载）；双击节点看该节点源码；双击画布空白生成整个模型。
4. **训练**：「训练」→ 选数据集（随机拟合/MNIST/文本分类）→ 启动 → 实时日志 → 自动注册到模型列表。
5. **推理**：「已训练模型」→ 选模型 → 按模型输入自动出表单（文本框/图片上传/张量 JSON）→ 运行推理。

## 目录结构

```
├── frontend/           # React 18 + TS + Vite + React Flow + Zustand + AntD + Monaco
│   └── src/
│       ├── components/     # Palette / Canvas / PropPanel / SourceViewer / TrainModal / ModelsDrawer ...
│       ├── nodes/          # React Flow 自定义节点
│       ├── stores/         # graphStore（图状态/序列化/校验调度）、uiStore
│       ├── schema/         # 拓扑 IR 的 TS 类型（与后端 Pydantic 对齐）
│       └── api/            # 后端 API 客户端
├── backend/            # FastAPI + PyTorch
│   ├── app/
│   │   ├── api/routes.py   # 全部 REST + SSE 路由
│   │   ├── ir/             # 拓扑 schema（Pydantic）/ 符号形状代数 / 校验+形状推导
│   │   ├── registry/ops.py # 算子注册表（单一事实来源：UI 元数据+形状规则+代码模板）
│   │   ├── codegen/        # Jinja2 代码生成管线
│   │   ├── training/       # 训练任务管理（子进程+SSE）/ 推理服务
│   │   └── store.py        # SQLite（自定义模块库 + 模型注册表）
│   └── tests/              # smoke.py（5 用例）+ e2e_api.py（9 用例）
├── docs/               # 拓扑 schema 说明、算子贡献指南
├── gui-test-screenshots/  # GUI 测试截图证据
└── start.ps1           # 一键启动脚本
```

## 拓扑 JSON（Graph IR）

参考 ONNX 的图结构（nodes + edges + 带 type/shape 的端口），但定位更高层：节点可以是基础算子，也可以是 TransformerBlock 级模块（ONNX opset 没有这类算子）。维度支持符号（`"batch"`、`"seq"`）。将来需要标准格式时，由生成的 PyTorch 代码 `torch.onnx.export` 即可。

```json
{
  "format": "modelforge/graph", "version": "0.1",
  "model":  {"name": "MyTransformer", "description": ""},
  "inputs":  [{"name": "ids", "dtype": "int64", "shape": ["batch", "seq"]}],
  "outputs": [{"name": "y", "dtype": "float32", "shape": ["batch", "seq", 10]}],
  "nodes": [
    {"id": "n0", "name": "Embedding_1", "op": "Embedding", "group": "basic",
     "attrs": {"num_embeddings": 32000, "embedding_dim": 768},
     "inputs": [{"name": "ids", "dtype": "int64", "shape": []}],
     "outputs": [{"name": "y", "dtype": "float32", "shape": []}],
     "ui": {"x": 100, "y": 200}}
  ],
  "edges": [
    {"id": "e0", "from": {"node": "__graph_in__", "port": "ids"}, "to": {"node": "n0", "port": "ids"}}
  ]
}
```

约定：边可以引用两个虚拟端点 `"__graph_in__"` / `"__graph_out__"`（对应画布上的「模型输入/模型输出」固定节点）。

## 扩展算子

新增算子只需改 `backend/app/registry/ops.py`（一条 OpDef：属性 schema + 端口 + 形状规则 + init/forward Jinja2 模板），前端面板/表单自动出现，前端零改动。指南见 [docs/](docs/)。

## 测试

```powershell
cd backend
python tests/smoke.py     # 5 用例：形状推导/非法连线拒绝/代码生成前向/自定义模块/空图回归
python tests/e2e_api.py   # 9 用例：MNIST 真实训练→图片推理、文本训练→文本推理、自定义模块→codegen
```
