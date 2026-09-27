# AGENTS.md — ModelForge（可视化模块化拼接模型）

浏览器拖拽搭模型工作台：前端画布拼图 → Graph IR JSON → 后端生成可运行 PyTorch 代码 → 一键训练/推理（SSE 实时日志）。本地单机工具，无鉴权（CORS `*`）。

## 常用命令

```powershell
# 后端（127.0.0.1:8000）
cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 前端（127.0.0.1:5173，另开终端）
cd frontend && npm run dev

# 测试与质量
cd backend
python tests/smoke.py      # 13 用例，纯离线（无需起服务）：校验/形状推导/前向运行/回归
python tests/e2e_api.py    # 需要 backend 已在 8000 运行；真实 MNIST/文本训练+推理
cd frontend
npm run build              # tsc -b && vite build（typecheck 就靠这一条，没有独立 typecheck 脚本）
npm run lint               # oxlint
```

根目录 `start.ps1` 一键起双服务。改完后端 Python 需手动重启 uvicorn（无 reload）；前端有 HMR。

## 架构与边界

- **`backend/app/registry/ops.py` 是单一事实来源**：每个算子的属性 schema、端口、形状规则、init/fwd Jinja2 模板都在这一处定义（34 个算子）。前端通过 `GET /api/ops` 动态渲染面板/属性表单——**新增算子只改这个文件，前端零改动**。
- **Graph IR 是双镜像约定**：`backend/app/ir/schema.py`（Pydantic）与 `frontend/src/schema/graph.ts`（TS）必须一一对应，改一边必须同步另一边。字段级说明见 `docs/拓扑Schema与算子贡献指南.md`。
- 虚拟端点 `"__graph_in__"` / `"__graph_out__"`（常量定义在 `backend/app/ir/validate.py` 的 `GRAPH_IN/GRAPH_OUT`，前端 `schema/graph.ts` 同名导出）表示画布上的模型输入/输出固定节点，会出现在拓扑 JSON 的 edges 里。
- 代码生成管线：`ir/validate.py` 校验 → `ir/shapes.py` 符号形状代数（维度支持 `"batch"`/`"seq"` 字符串）→ 拓扑排序 → 逐节点渲染 Jinja2 → `model.py/train.py/config.json`（`codegen/generator.py`）。训练用子进程执行生成的 train.py，按行采集 JSON 指标（`training/jobs.py`）；`_jobs` 仅存本进程内存态，SQLite `backend/data/modelforge.db` 才是跨重启权威历史。
- 前端状态集中在 `stores/graphStore.ts`（图/校验调度/序列化）与 `uiStore.ts`，API 全部走 `api/client.ts`（Vite proxy `/api` → 8000）。
- 全部 REST+SSE 路由都在 `backend/app/api/routes.py`。

## 关键坑（实测教训）

- **React Flow v12 受控拖拽**：必须用 `applyNodeChanges` 应用变更，并**丢弃 `rf.setNodes` 直写的 replace 回声**（`change.type === 'replace'` 时忽略），否则拖拽冻结。直接 `rf.setNodes` 会冻结拖拽不报错；HMR 后旧代码残留会让拖拽失灵，但代码本身没错——先重启 dev server 再排查。
- **Handle 定位**：节点行容器必须 `position: relative`，不能用内联 `top` 定位 Handle，否则把手错位（连不上线且拖不动节点，同一根因）。
- **Vite 必须绑 `127.0.0.1`**（vite.config.ts 已固定）。改成 `localhost` 在本机 Windows 会解析到 `::1` 导致连接问题；后端也应绑 IPv4。
- **Windows 编码**：训练子进程 env 强制 `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1`（`training/jobs.py` 已处理）；Python 3.11 的 f-string 内不允许反斜杠。
- **GUI 自动化测试**：本环境内嵌浏览器 Playwright 点击会卡死，DOM 定位 + 截图 + API 级 E2E 分工验证（截图证据放 `gui-test-screenshots/`）。

## 改动须知

- 改 Graph IR / 形状推导 / 校验 / 算子注册表前，先读 `docs/拓扑Schema与算子贡献指南.md`；后端改动的回归基线以 `docs/审查报告.md`（v2）记录的缺陷为准。
- 拓扑 JSON 参考但高于 ONNX：节点可以是 TransformerBlock 级模块；需要标准格式时由生成的代码走 `torch.onnx.export`。
- 后端模块间依赖方向：`registry → ir`，`codegen/training/api → registry+ir`；`store.py`（SQLite）只管持久化，不掺业务。
