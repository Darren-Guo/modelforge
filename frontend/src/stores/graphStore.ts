/** 全局图状态：React Flow 节点/边 + 图级输入输出 + 校验调度 + 序列化。 */
import { message } from 'antd';
import { create } from 'zustand';
import {
  applyEdgeChanges, applyNodeChanges,
  type Connection, type Edge, type EdgeChange, type Node, type NodeChange,
} from '@xyflow/react';
import * as api from '../api/client';
import type {
  Dim, Dtype, Graph, GraphEdge, GraphNode, ModuleDef, ModuleSummary,
  OpDef, Port, ValidationReport,
} from '../schema/graph';
import { GRAPH_IN, GRAPH_OUT } from '../schema/graph';
import { checkConnection, portOfGraph } from '../utils/shape';
import { computeAutoLayout, positionsOverlap } from '../utils/autoLayout';

export interface ModuleNodeData {
  op: string;
  group: 'basic' | 'llm' | 'custom';
  label: string;
  name: string;
  doc: string;
  attrs: Record<string, unknown>;
  inputs: Port[];
  outputs: Port[];
  error?: boolean;
  [key: string]: unknown;
}

export interface IONodeData {
  kind: 'in' | 'out';
  label: string;
  ports: Port[];
  [key: string]: unknown;
}

export type RFNode = Node<ModuleNodeData | IONodeData>;
export type RFEdge = Edge;

export interface ModuleEditMode {
  moduleId: string;
  name: string;
  description: string;
  inputs: Port[];
  outputs: Port[];
  // 模型画布的备份（含图级端口，退出编辑时一并恢复）
  snapshot: {
    nodes: RFNode[]; edges: RFEdge[];
    graphInputs: Port[]; graphOutputs: Port[];
  } | null;
}

interface BoundaryGroup {
  srcNode: string;
  srcPort: string;
  dtype: Dtype;
  shape: Dim[];
  targets: { node: string; port: string }[];
  suggestedName: string;
}

export interface SubgraphExtract {
  nodes: GraphNode[];
  inputGroups: BoundaryGroup[];
  outputGroups: BoundaryGroup[];
  danglingInputs: string[];
}

interface GraphState {
  // 算子/模块注册表
  ops: OpDef[];
  opIndex: Record<string, OpDef>;
  modules: ModuleSummary[];
  moduleIndex: Record<string, ModuleDef>;

  // 画布
  nodes: RFNode[];
  edges: RFEdge[];
  graphInputs: Port[];
  graphOutputs: Port[];
  modelMeta: { name: string; description: string };

  // 选择与校验
  selectedNodeIds: string[];
  selectedEdgeIds: string[];
  validation: ValidationReport | null;
  validating: boolean;

  // 自定义模块编辑模式
  editing: ModuleEditMode | null;

  // 自增计数：导入/自动布局后 Canvas 据此触发一次 fitView
  fitViewTick: number;

  // ---- actions ----
  init: () => Promise<void>;
  refreshModules: () => Promise<void>;
  addOpNode: (op: string, group: 'basic' | 'llm', position: { x: number; y: number }) => void;
  addCustomNode: (moduleId: string, position: { x: number; y: number }) => void;
  onNodesChange: (changes: unknown[]) => void;
  onEdgesChange: (changes: unknown[]) => void;
  onConnect: (conn: Connection) => boolean;
  setSelection: (nodeIds: string[], edgeIds: string[]) => void;
  updateNodeName: (id: string, name: string) => void;
  updateNodeAttrs: (id: string, attrs: Record<string, unknown>) => void;
  updateNodePort: (id: string, kind: 'inputs' | 'outputs', portName: string, patch: Partial<Port>) => void;
  setGraphPorts: (kind: 'inputs' | 'outputs', ports: Port[]) => void;
  updateGraphPort: (kind: 'inputs' | 'outputs', index: number, patch: Partial<Port>) => void;
  setModelMeta: (meta: Partial<{ name: string; description: string }>) => void;
  deleteElements: (nodeIds: string[], edgeIds: string[]) => void;
  clearCanvas: () => void;
  autoLayout: () => void;
  requestFitView: () => void;

  exportGraph: () => Graph;
  importGraph: (g: Graph) => void;
  extractSubgraph: (nodeIds: string[]) => SubgraphExtract;
  buildModuleGraph: (extract: SubgraphExtract, inputs: Port[], outputs: Port[]) => Graph;

  scheduleValidate: () => void;
  validateNow: () => Promise<ValidationReport | null>;

  // 自定义模块编辑
  startEditModule: (mod: ModuleDef) => void;
  saveEditModule: () => Promise<void>;
  cancelEditModule: () => void;
}

let idCounter = 1;
const nextId = (prefix: string) => `${prefix}${idCounter++}`;

/** 整体载入图后把 idCounter 推进到已用 id 之后，避免新增节点/边撞 id。 */
function advanceIdCounter(ids: string[]) {
  let max = idCounter - 1;
  for (const id of ids) {
    const m = /(\d+)$/.exec(id);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  idCounter = max + 1;
}

/** 用户手动改过 shape/dtype 的端口（nodeId:kind:portName），校验回填跳过以免静默还原手动声明。 */
const manualPorts = new Set<string>();
const manualKey = (id: string, kind: string, port: string) => `${id}:${kind}:${port}`;

let validateTimer: ReturnType<typeof setTimeout> | null = null;
let validateSeq = 0;

// ---- 拖拽流畅度：变更按动画帧合并 -----------------------------------------
// React Flow 拖拽时每个 mousemove 都发一批 position 变更；若每批都 set()，
// 全应用（工具栏/面板/弹窗）会以 100Hz+ 重渲染，拖拽就一卡一卡。
// 这里把变更缓冲到 requestAnimationFrame 每帧只应用一次。
// 同时拖拽期间冻结校验回填（validateNow 的整表节点重建会打断拖拽）。
let nodeBuf: NodeChange<RFNode>[] = [];
let edgeBuf: EdgeChange[] = [];
let rafId: number | null = null;
let draggingNow = false;
let validateWhileDragging = false;

function flushGraphChanges() {
  rafId = null;
  const store = useGraphStore;
  const nChanges = nodeBuf;
  const eChanges = edgeBuf;
  nodeBuf = [];
  edgeBuf = [];
  if (nChanges.length === 0 && eChanges.length === 0) return;

  const wasDragging = draggingNow;
  draggingNow = nChanges.some(
    (c) => c.type === 'position' && (c as { dragging?: boolean }).dragging === true,
  );
  const dragEnded = wasDragging && !draggingNow;

  if (nChanges.length > 0) {
    // 丢弃 replace 回声：RF 在对象身份不一致时发 {type:'replace'}，而 applyNodeChanges
    // 会把 item 复制成新对象——身份又不一致 → RF 再发 replace → 无限回声循环
    // （每轮都是一次全店更新，表现为拖拽"一顿一顿"）。replace 携带的就是我们自己的
    // 对象，直接忽略即可。
    const filtered = nChanges.filter(
      (c) => c.type !== 'replace' && !(c.type === 'remove' && (c.id === GRAPH_IN || c.id === GRAPH_OUT)),
    );
    if (filtered.length > 0) {
      const st = store.getState();
      const nodes = applyNodeChanges(filtered, st.nodes);
      const removedIds = new Set(filtered.filter((c) => c.type === 'remove').map((c) => c.id));
      const edges = removedIds.size > 0
        ? st.edges.filter((e) => !removedIds.has(e.source) && !removedIds.has(e.target))
        : st.edges;
      store.setState({ nodes, edges });
      if (removedIds.size > 0) st.scheduleValidate();
    }
  }
  const eFiltered = eChanges.filter((c) => c.type !== 'replace');
  if (eFiltered.length > 0) {
    const st = store.getState();
    const edges = applyEdgeChanges(eFiltered, st.edges);
    store.setState({ edges });
    if (eFiltered.some((c) => c.type === 'remove')) st.scheduleValidate();
  }
  if (dragEnded && validateWhileDragging) {
    validateWhileDragging = false;
    store.getState().scheduleValidate();
  }
}

function scheduleGraphFlush() {
  if (rafId === null && typeof requestAnimationFrame === 'function') {
    rafId = requestAnimationFrame(flushGraphChanges);
  } else if (rafId === null) {
    flushGraphChanges(); // 无 rAF 环境退化为同步
  }
}

const defaultPorts = (defs: { name: string; dtype: string }[]): Port[] =>
  defs.map((d) => ({ name: d.name, dtype: d.dtype as Dtype, shape: [] }));

/** 整体重排节点坐标（dagre 分层 + 网格对齐，见 utils/autoLayout）。 */
function withAutoLayout(nodes: RFNode[], edges: RFEdge[]): RFNode[] {
  const pos = computeAutoLayout(nodes, edges);
  return nodes.map((n) => (pos.has(n.id) ? { ...n, position: pos.get(n.id)! } : n));
}

/** 画布上的「模型输入/模型输出」固定虚拟节点。 */
function makeIONodes(inputs: Port[], outputs: Port[]): RFNode[] {
  return [
    {
      id: GRAPH_IN, type: 'io', position: { x: 0, y: 120 }, draggable: true, deletable: false,
      data: { kind: 'in', label: '模型输入', ports: inputs.map((p) => ({ ...p })) },
    },
    {
      id: GRAPH_OUT, type: 'io', position: { x: 720, y: 120 }, draggable: true, deletable: false,
      data: { kind: 'out', label: '模型输出', ports: outputs.map((p) => ({ ...p })) },
    },
  ];
}

export const useGraphStore = create<GraphState>((set, get) => ({
  ops: [],
  opIndex: {},
  modules: [],
  moduleIndex: {},

  nodes: makeIONodes(
    [{ name: 'x', dtype: 'float32', shape: ['batch', 'seq', 768] }],
    [{ name: 'y', dtype: 'float32', shape: ['batch', 'seq', 10] }],
  ),
  edges: [],
  graphInputs: [{ name: 'x', dtype: 'float32', shape: ['batch', 'seq', 768] }],
  graphOutputs: [{ name: 'y', dtype: 'float32', shape: ['batch', 'seq', 10] }],
  modelMeta: { name: 'MyModel', description: '' },

  selectedNodeIds: [],
  selectedEdgeIds: [],
  validation: null,
  validating: false,
  editing: null,
  fitViewTick: 0,

  async init() {
    const ops = await api.fetchOps();
    const opIndex: Record<string, OpDef> = {};
    for (const o of ops) opIndex[o.op] = o;
    set({ ops, opIndex });
    await get().refreshModules();
    get().scheduleValidate();
  },

  async refreshModules() {
    const list = await api.listModules();
    // 列表接口只有摘要（无 nodes/edges），moduleIndex 必须存全量定义，
    // 否则「编辑模块定义」遍历 mod.graph.nodes 会崩（见 GET /api/modules 契约）
    const defs = await Promise.all(list.map((m) => api.getModule(m.id)));
    const moduleIndex: Record<string, ModuleDef> = {};
    for (const d of defs) moduleIndex[d.id] = d;
    set({ modules: list, moduleIndex });
  },

  addOpNode(op, group, position) {
    const def = get().opIndex[op];
    if (!def) return;
    const count = get().nodes.filter((n) => (n.data as ModuleNodeData).op === op).length + 1;
    const node: RFNode = {
      id: nextId('n'),
      type: 'module',
      position,
      data: {
        op, group,
        label: def.label,
        name: `${op}_${count}`,
        doc: def.doc,
        attrs: Object.fromEntries(def.attrs.map((a) => [a.name, a.default])),
        inputs: defaultPorts(def.inputs),
        outputs: defaultPorts(def.outputs),
      },
    };
    set({ nodes: [...get().nodes, node] });
    get().scheduleValidate();
  },

  addCustomNode(moduleId, position) {
    const mod = get().moduleIndex[moduleId];
    if (!mod) return;
    const count = get().nodes.filter((n) => (n.data as ModuleNodeData).op === `custom:${moduleId}`).length + 1;
    const node: RFNode = {
      id: nextId('n'),
      type: 'module',
      position,
      data: {
        op: `custom:${moduleId}`,
        group: 'custom',
        label: mod.name,
        name: `${mod.name}_${count}`,
        doc: mod.description || '自定义模块',
        attrs: {},
        inputs: mod.inputs.map((p) => ({ ...p })),
        outputs: mod.outputs.map((p) => ({ ...p })),
      },
    };
    set({ nodes: [...get().nodes, node] });
    get().scheduleValidate();
  },

  onNodesChange(changes) {
    if (changes.length === 0) return;
    nodeBuf.push(...(changes as NodeChange<RFNode>[]));
    scheduleGraphFlush();
  },

  onEdgesChange(changes) {
    if (changes.length === 0) return;
    edgeBuf.push(...(changes as EdgeChange[]));
    scheduleGraphFlush();
  },

  onConnect(conn) {
    const { nodes, edges, exportGraph } = get();
    if (!conn.source || !conn.target || !conn.sourceHandle || !conn.targetHandle) return false;
    if (conn.target === GRAPH_IN || conn.source === GRAPH_OUT) return false;
    if (edges.some((e) => e.target === conn.target && e.targetHandle === conn.targetHandle)) return false;

    const graph = exportGraph();
    const src = portOfGraph(graph, conn.source, conn.sourceHandle);
    const tgt = portOfGraph(graph, conn.target, conn.targetHandle);
    if (!src || !tgt) return false;
    const reason = checkConnection({ ...src.port, name: src.port.name }, { ...tgt.port, name: tgt.port.name });
    if (reason) return false;

    const edge: RFEdge = {
      id: nextId('e'),
      source: conn.source, target: conn.target,
      sourceHandle: conn.sourceHandle, targetHandle: conn.targetHandle,
    };
    set({ nodes, edges: [...edges, edge] });
    get().scheduleValidate();
    return true;
  },

  setSelection(nodeIds, edgeIds) {
    // 幂等：值没变就不触发 set，避免 onSelectionChange → set → 重渲染 的死循环
    const { selectedNodeIds, selectedEdgeIds } = get();
    const same =
      selectedNodeIds.length === nodeIds.length &&
      selectedEdgeIds.length === edgeIds.length &&
      selectedNodeIds.every((id, i) => id === nodeIds[i]) &&
      selectedEdgeIds.every((id, i) => id === edgeIds[i]);
    if (!same) set({ selectedNodeIds: nodeIds, selectedEdgeIds: edgeIds });
  },

  updateNodeName(id, name) {
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, name } as RFNode['data'] } : n),
    });
  },

  updateNodeAttrs(id, attrs) {
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, attrs } as RFNode['data'] } : n),
    });
    get().scheduleValidate();
  },

  updateNodePort(id, kind, portName, patch) {
    // 手动改过 shape/dtype 的端口不再被校验回填覆盖
    if (patch.shape !== undefined || patch.dtype !== undefined) manualPorts.add(manualKey(id, kind, portName));
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id) return n;
        const data = n.data as ModuleNodeData;
        const ports = (data[kind] as Port[]).map((p) => (p.name === portName ? { ...p, ...patch } : p));
        return { ...n, data: { ...data, [kind]: ports } as RFNode['data'] };
      }),
    });
    get().scheduleValidate();
  },

  setGraphPorts(kind, ports) {
    set({ [kind === 'inputs' ? 'graphInputs' : 'graphOutputs']: ports } as Partial<GraphState>);
    get().scheduleValidate();
  },

  updateGraphPort(kind, index, patch) {
    const key = kind === 'inputs' ? 'graphInputs' : 'graphOutputs';
    const ports = [...(get()[key] as Port[])];
    ports[index] = { ...ports[index], ...patch };
    set({ [key]: ports } as Partial<GraphState>);
    get().scheduleValidate();
  },

  setModelMeta(meta) {
    set({ modelMeta: { ...get().modelMeta, ...meta } });
  },

  deleteElements(nodeIds, edgeIds) {
    const keepNodes = get().nodes.filter((n) =>
      !nodeIds.includes(n.id) || n.id === GRAPH_IN || n.id === GRAPH_OUT);
    const removed = nodeIds.filter((id) => id !== GRAPH_IN && id !== GRAPH_OUT);
    const keepEdges = get().edges.filter((e) =>
      !edgeIds.includes(e.id) && !removed.includes(e.source) && !removed.includes(e.target));
    set({ nodes: keepNodes, edges: keepEdges });
    get().scheduleValidate();
  },

  clearCanvas() {
    const inputs: Port[] = [{ name: 'x', dtype: 'float32', shape: ['batch', 'seq', 768] }];
    const outputs: Port[] = [{ name: 'y', dtype: 'float32', shape: ['batch', 'seq', 10] }];
    manualPorts.clear();
    set({
      nodes: makeIONodes(inputs, outputs), edges: [],
      graphInputs: inputs, graphOutputs: outputs,
      selectedNodeIds: [], selectedEdgeIds: [],
      validation: null,
    });
    get().scheduleValidate();
  },

  autoLayout() {
    const { nodes, edges } = get();
    if (nodes.length === 0) return;
    set({ nodes: withAutoLayout(nodes, edges) });
    get().requestFitView();
  },

  requestFitView() {
    set({ fitViewTick: get().fitViewTick + 1 });
  },

  exportGraph() {
    const { nodes, edges, graphInputs, graphOutputs, modelMeta } = get();
    const gnodes: GraphNode[] = nodes
      .filter((n) => n.id !== GRAPH_IN && n.id !== GRAPH_OUT)
      .map((n) => {
        const d = n.data as ModuleNodeData;
        return {
          id: n.id, name: d.name, op: d.op, group: d.group, attrs: d.attrs,
          inputs: d.inputs, outputs: d.outputs,
          ui: { x: Math.round(n.position.x), y: Math.round(n.position.y) },
        };
      });
    const gedges: GraphEdge[] = edges.map((e) => ({
      id: e.id,
      from: { node: e.source, port: e.sourceHandle ?? '' },
      to: { node: e.target, port: e.targetHandle ?? '' },
    }));
    return {
      format: 'modelforge/graph', version: '0.1',
      model: { ...modelMeta },
      inputs: graphInputs, outputs: graphOutputs,
      nodes: gnodes, edges: gedges,
    };
  },

  importGraph(g) {
    const nodes: RFNode[] = makeIONodes(g.inputs, g.outputs);
    for (const n of g.nodes) {
      const def = get().opIndex[n.op];
      const mod = n.op.startsWith('custom:') ? get().moduleIndex[n.op.slice(7)] : undefined;
      nodes.push({
        id: n.id, type: 'module',
        position: { x: n.ui?.x ?? 200, y: n.ui?.y ?? 100 },
        data: {
          op: n.op, group: n.group,
          label: def?.label ?? mod?.name ?? n.op,
          name: n.name, doc: def?.doc ?? mod?.description ?? '',
          attrs: n.attrs, inputs: n.inputs, outputs: n.outputs,
        },
      });
    }
    const edges: RFEdge[] = g.edges.map((e) => ({
      id: e.id, source: e.from.node, target: e.to.node,
      sourceHandle: e.from.port, targetHandle: e.to.port,
    }));
    // 沿用文件里的 id，必须把计数器推到已用 id 之后，否则新增节点会撞 id
    advanceIdCounter([...g.nodes.map((n) => n.id), ...g.edges.map((e) => e.id)]);
    manualPorts.clear();
    set({
      // 文件缺坐标或坐标互相重叠（缩成一团）→ 自动分层重排；排布合理则原样沿用
      nodes: positionsOverlap(nodes) ? withAutoLayout(nodes, edges) : nodes,
      edges,
      graphInputs: g.inputs.map((p) => ({ ...p })),
      graphOutputs: g.outputs.map((p) => ({ ...p })),
      modelMeta: { ...g.model },
      selectedNodeIds: [], selectedEdgeIds: [],
      validation: null,
    });
    get().requestFitView();
    get().scheduleValidate();
  },

  extractSubgraph(nodeIds) {
    const { nodes, edges } = get();
    const selected = new Set(nodeIds.filter((id) => id !== GRAPH_IN && id !== GRAPH_OUT));
    const inMap = new Map<string, BoundaryGroup>();
    const outMap = new Map<string, BoundaryGroup>();
    const danglingInputs: string[] = [];

    const nodeById = new Map(nodes.map((n) => [n.id, n]));
    const nameOf = (id: string) => {
      const n = nodeById.get(id);
      return n ? ((n.data as ModuleNodeData).name || id) : id;
    };

    for (const e of edges) {
      const srcIn = selected.has(e.source);
      const tgtIn = selected.has(e.target);
      if (srcIn && tgtIn) continue;
      if (!srcIn && tgtIn) {
        const key = `${e.source}.${e.sourceHandle}`;
        let g = inMap.get(key);
        const srcNode = nodeById.get(e.source);
        const ports = srcNode
          ? (e.source === GRAPH_IN
            ? (srcNode.data as IONodeData).ports
            : (srcNode.data as ModuleNodeData).outputs)
          : [];
        const port = ports.find((p) => p.name === e.sourceHandle);
        if (!g) {
          g = {
            srcNode: e.source, srcPort: e.sourceHandle ?? '',
            dtype: port?.dtype ?? 'float32', shape: port?.shape ?? [],
            targets: [], suggestedName: `in_${e.sourceHandle ?? 'x'}`,
          };
          inMap.set(key, g);
        }
        g.targets.push({ node: e.target, port: e.targetHandle ?? '' });
      } else if (srcIn && !tgtIn) {
        const key = `${e.source}.${e.sourceHandle}`;
        let g = outMap.get(key);
        const srcNode = nodeById.get(e.source);
        const port = (srcNode?.data as ModuleNodeData)?.outputs.find((p) => p.name === e.sourceHandle);
        if (!g) {
          g = {
            srcNode: e.source, srcPort: e.sourceHandle ?? '',
            dtype: port?.dtype ?? 'float32', shape: port?.shape ?? [],
            targets: [], suggestedName: `out_${nameOf(e.source)}_${e.sourceHandle ?? 'y'}`,
          };
          outMap.set(key, g);
        }
        g.targets.push({ node: e.target, port: e.targetHandle ?? '' });
      }
    }

    // 选中节点未连接的输入端口
    for (const id of selected) {
      const n = nodeById.get(id);
      if (!n) continue;
      const d = n.data as ModuleNodeData;
      for (const p of d.inputs) {
        const connected = edges.some((e) => e.target === id && e.targetHandle === p.name);
        if (!connected) danglingInputs.push(`${d.name}.${p.name}`);
      }
    }

    const gnodes: GraphNode[] = nodes
      .filter((n) => selected.has(n.id))
      .map((n) => {
        const d = n.data as ModuleNodeData;
        return {
          id: n.id, name: d.name, op: d.op, group: d.group, attrs: d.attrs,
          inputs: d.inputs, outputs: d.outputs,
          ui: { x: Math.round(n.position.x), y: Math.round(n.position.y) },
        };
      });

    return {
      nodes: gnodes,
      inputGroups: [...inMap.values()],
      outputGroups: [...outMap.values()],
      danglingInputs,
    };
  },

  buildModuleGraph(extract, inputs, outputs) {
    const { nodes, inputGroups, outputGroups } = extract;
    const edges: GraphEdge[] = [];
    // 内部边
    const selected = new Set(nodes.map((n) => n.id));
    for (const e of get().edges) {
      if (selected.has(e.source) && selected.has(e.target)) {
        edges.push({
          id: e.id,
          from: { node: e.source, port: e.sourceHandle ?? '' },
          to: { node: e.target, port: e.targetHandle ?? '' },
        });
      }
    }
    // 边界输入：graph input → 选区内目标
    inputGroups.forEach((g, i) => {
      for (const t of g.targets) {
        edges.push({
          id: `bi_${i}_${t.node}_${t.port}`,
          from: { node: GRAPH_IN, port: inputs[i]?.name ?? g.suggestedName },
          to: { node: t.node, port: t.port },
        });
      }
    });
    // 边界输出：选区内源 → graph output
    outputGroups.forEach((g, i) => {
      edges.push({
        id: `bo_${i}`,
        from: { node: g.srcNode, port: g.srcPort },
        to: { node: GRAPH_OUT, port: outputs[i]?.name ?? g.suggestedName },
      });
    });
    return {
      format: 'modelforge/graph', version: '0.1',
      model: { name: '', description: '' },
      inputs, outputs, nodes, edges,
    };
  },

  scheduleValidate() {
    if (validateTimer) clearTimeout(validateTimer);
    validateTimer = setTimeout(() => void get().validateNow(), 400);
  },

  async validateNow() {
    const seq = ++validateSeq;
    set({ validating: true });
    try {
      const report = await api.validateGraph(get().exportGraph());
      if (seq !== validateSeq) return null; // 过期响应丢弃，避免覆盖新状态
      if (draggingNow) {
        // 拖拽进行中：整表节点重建会打断拖拽，只记报告，形状回填延到拖拽结束
        validateWhileDragging = true;
        set({ validation: report, validating: false });
        return report;
      }
      // 形状推导回填（以校验结果为准；用户手动改过的端口不回填，冲突由校验报错暴露）
      const byId = new Map(report.nodes.map((n) => [n.id, n]));
      const errorNames = new Set(report.errors.map((e) => e.where));
      const nodes = get().nodes.map((n) => {
        if (n.id === GRAPH_IN || n.id === GRAPH_OUT) return n;
        const d = n.data as ModuleNodeData;
        const inf = byId.get(n.id);
        const merge = (kind: 'inputs' | 'outputs', ports: Port[], inferred: { name: string; shape: Dim[]; dtype: Dtype }[]) =>
          ports.map((p) => {
            if (manualPorts.has(manualKey(n.id, kind, p.name))) return p;
            const ip = inferred.find((x) => x.name === p.name);
            return ip ? { ...p, shape: ip.shape, dtype: ip.dtype } : p;
        });
        return {
          ...n,
          data: {
            ...d,
            error: errorNames.has(d.name) || errorNames.has(n.id),
            inputs: inf ? merge('inputs', d.inputs, inf.inputs) : d.inputs,
            outputs: inf ? merge('outputs', d.outputs, inf.outputs) : d.outputs,
          } as ModuleNodeData,
        };
      });
      set({ validation: report, validating: false, nodes });
      return report;
    } catch (e) {
      if (seq !== validateSeq) return null; // 过期失败同样不覆盖新状态
      set({
        validating: false,
        validation: {
          ok: false,
          errors: [{ where: '网络', message: e instanceof Error ? e.message : String(e) }],
          warnings: [], nodes: [], order: [],
        },
      });
      return null;
    }
  },

  startEditModule(mod) {
    const snapshot = {
      nodes: get().nodes, edges: get().edges,
      graphInputs: get().graphInputs, graphOutputs: get().graphOutputs,
    };
    const nodes: RFNode[] = makeIONodes(mod.inputs, mod.outputs).map((n) => ({
      ...n,
      data: { ...(n.data as IONodeData), label: (n.data as IONodeData).kind === 'in' ? '模块输入' : '模块输出' },
    }));
    for (const n of mod.graph.nodes) {
      const def = get().opIndex[n.op];
      nodes.push({
        id: n.id, type: 'module',
        position: { x: n.ui?.x ?? 200, y: n.ui?.y ?? 100 },
        data: {
          op: n.op, group: n.group,
          label: def?.label ?? n.op, name: n.name, doc: def?.doc ?? '',
          attrs: n.attrs, inputs: n.inputs, outputs: n.outputs,
        },
      });
    }
    const edges: RFEdge[] = mod.graph.edges.map((e) => ({
      id: e.id, source: e.from.node, target: e.to.node,
      sourceHandle: e.from.port, targetHandle: e.to.port,
    }));
    // 模块内部节点沿用其 id，同样要把计数器推到已用 id 之后
    advanceIdCounter([...mod.graph.nodes.map((n) => n.id), ...mod.graph.edges.map((e) => e.id)]);
    set({
      nodes: positionsOverlap(nodes) ? withAutoLayout(nodes, edges) : nodes,
      edges,
      graphInputs: mod.inputs.map((p) => ({ ...p })),
      graphOutputs: mod.outputs.map((p) => ({ ...p })),
      selectedNodeIds: [], selectedEdgeIds: [],
      editing: {
        moduleId: mod.id, name: mod.name, description: mod.description,
        inputs: mod.inputs.map((p) => ({ ...p })),
        outputs: mod.outputs.map((p) => ({ ...p })),
        snapshot,
      },
    });
    get().requestFitView();
    get().scheduleValidate();
  },

  async saveEditModule() {
    const ed = get().editing;
    if (!ed) return;
    const graph = get().exportGraph();
    graph.inputs = get().graphInputs;
    graph.outputs = get().graphOutputs;
    try {
      await api.saveModule({
        id: ed.moduleId, name: ed.name, description: ed.description,
        inputs: get().graphInputs, outputs: get().graphOutputs, graph,
      });
    } catch (e) {
      // 失败时停留在编辑态并给出原因，避免被误认为保存成功
      const detail = e instanceof api.ApiError ? e.detail : null;
      const msg = detail && typeof detail === 'object' && 'report' in (detail as object)
        ? (detail as { report: { errors: { message: string }[] } }).report.errors
            .map((x) => x.message).join('；')
        : e instanceof Error ? e.message : '保存失败';
      message.error({ content: `保存模块定义失败：${msg}`, duration: 6 });
      return;
    }
    await get().refreshModules();
    get().cancelEditModule();
  },

  cancelEditModule() {
    const ed = get().editing;
    if (!ed?.snapshot) return;
    set({
      nodes: ed.snapshot.nodes, edges: ed.snapshot.edges,
      graphInputs: ed.snapshot.graphInputs, graphOutputs: ed.snapshot.graphOutputs,
      selectedNodeIds: [], selectedEdgeIds: [],
      editing: null,
    });
    get().scheduleValidate();
  },
}));

export function nodeData(n: RFNode): ModuleNodeData | IONodeData {
  return n.data;
}
