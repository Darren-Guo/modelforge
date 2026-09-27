/** 中央画布：React Flow 封装（拖拽放置、连线校验、双击查看源码）。
 *
 * 注意：传给 React Flow 的回调必须保持稳定身份（useCallback + getState()），
 * 否则内部 store 同步 → onSelectionChange → set → 重渲染 会形成死循环。
 */
import { useCallback, useEffect, useMemo, useRef } from 'react';
import {
  Background, Controls, MiniMap, ReactFlow, useReactFlow,
  type Connection, type Edge, type IsValidConnection, type Node, type OnConnectEnd,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { message } from 'antd';
import { useGraphStore } from '../stores/graphStore';
import { useUIStore } from '../stores/uiStore';
import { nodeTypes } from '../nodes/ModuleNode';
import { checkConnection, portOfGraph } from '../utils/shape';
import { SNAP_GRID } from '../utils/autoLayout';
import * as api from '../api/client';
import { apiErrorText } from './errors';
import { GRAPH_IN, GRAPH_OUT } from '../schema/graph';

/** 连接被拒绝的原因：与 store.onConnect 的判定一致，另补自环/占用等文案（checkConnection 只看 dtype/shape）。 */
function connectionIssue(conn: Connection | Edge): string | null {
  const store = useGraphStore.getState();
  if (!conn.source || !conn.target || !conn.sourceHandle || !conn.targetHandle) return '端口不存在';
  if (conn.target === GRAPH_IN || conn.source === GRAPH_OUT) return '模型输入只能作为起点、模型输出只能作为终点';
  if (conn.source === conn.target) return '不能连接节点自身';
  if (store.edges.some((e) => e.target === conn.target && e.targetHandle === conn.targetHandle)) {
    return '输入端口已被占用';
  }
  const graph = store.exportGraph();
  const src = portOfGraph(graph, conn.source, conn.sourceHandle);
  const tgt = portOfGraph(graph, conn.target, conn.targetHandle);
  if (!src || !tgt) return '端口不存在';
  return checkConnection(src.port, tgt.port);
}

export default function Canvas() {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const fitViewTick = useGraphStore((s) => s.fitViewTick);
  const { screenToFlowPosition, fitView } = useReactFlow();
  const wrapper = useRef<HTMLDivElement>(null);

  // 导入 / 自动布局 / 打开模块定义后把视野收敛到整图
  useEffect(() => {
    if (fitViewTick > 0) void fitView({ padding: 0.15, duration: 240 });
  }, [fitViewTick, fitView]);

  const styledEdges = useMemo(
    () => edges.map((e) => ({
      ...e,
      animated: true,
      style: { strokeWidth: 2 },
      markerEnd: { type: 'arrowclosed' as const },
    })),
    [edges],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const raw = e.dataTransfer.getData('application/modelforge-op');
    if (!raw) return;
    const { op, group, kind } = JSON.parse(raw) as { op: string; group: 'basic' | 'llm' | 'custom'; kind?: string };
    const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
    const store = useGraphStore.getState();
    if (kind === 'custom' || group === 'custom') {
      store.addCustomNode(op, position);
    } else {
      store.addOpNode(op, group, position);
    }
  }, [screenToFlowPosition]);

  // 连接是否已被 onConnect 处理过（成功或已提示原因），避免 onConnectEnd 重复/误报
  const connectHandled = useRef(false);

  const isValidConnection: IsValidConnection = useCallback((conn) => connectionIssue(conn) === null, []);

  const onConnect = useCallback((conn: Connection) => {
    connectHandled.current = true;
    const ok = useGraphStore.getState().onConnect(conn);
    if (!ok) {
      message.error(`无法连接：${connectionIssue(conn) ?? '未知原因'}`);
    }
  }, []);

  const onConnectStart = useCallback(() => {
    connectHandled.current = false;
  }, []);

  // isValidConnection 会静默拦掉非法连接（onConnect 不触发），在拖拽松手时补一条拒绝原因
  const onConnectEnd = useCallback<OnConnectEnd>((_event, state) => {
    if (connectHandled.current) return;
    const toNode = state.toNode;
    const toHandle = state.toHandle;
    if (!toNode || !toHandle?.id) return; // 松手在空白处：视为放弃，不打扰
    const issue = connectionIssue({
      source: state.fromNode?.id ?? null,
      sourceHandle: state.fromHandle?.id ?? null,
      target: toNode.id,
      targetHandle: toHandle.id,
    });
    if (issue) message.error(`无法连接：${issue}`);
  }, []);

  const onSelectionChange = useCallback(({ nodes: selNodes, edges: selEdges }: { nodes: Node[]; edges: Edge[] }) => {
    useGraphStore.getState().setSelection(
      selNodes.map((n) => n.id), selEdges.map((e) => e.id));
  }, []);

  const onNodesChange = useCallback((ch: unknown[]) => {
    const store = useGraphStore.getState();
    store.onNodesChange(ch);
    // 孤立节点被删除时 React Flow 只发 nodeChanges（不带 edgeChanges），
    // 这里补一次校验调度，避免徽标/错误列表停留在旧状态
    if ((ch as { type: string }[]).some((c) => c.type === 'remove')) store.scheduleValidate();
  }, []);

  const onEdgesChange = useCallback((ch: unknown[]) => {
    useGraphStore.getState().onEdgesChange(ch);
  }, []);

  const onNodeDoubleClick = useCallback(async (_: unknown, node: Node) => {
    if (node.id === GRAPH_IN || node.id === GRAPH_OUT) return;
    try {
      const graph = useGraphStore.getState().exportGraph();
      const res = await api.nodeSnippet(graph, node.id);
      useUIStore.getState().openSource({ title: res.title, code: res.code });
    } catch (e) {
      message.error(e instanceof Error ? e.message : '获取源码失败');
    }
  }, []);

  const onPaneDoubleClick = useCallback(async () => {
    try {
      const graph = useGraphStore.getState().exportGraph();
      const res = await api.codegen(graph);
      useUIStore.getState().openSource({
        title: `「${graph.model.name}」生成的代码包`,
        code: res.files['model.py'] ?? '',
        files: res.files,
        downloadFolder: res.folder,
      });
    } catch (e) {
      message.error(`生成失败：${apiErrorText(e, '未知错误')}`);
    }
  }, []);

  return (
    <div
      className="mf-canvas"
      ref={wrapper}
      onDoubleClick={(e) => {
        // React Flow 没有 onPaneDoubleClick：容器双击且不在节点上 → 生成整个模型代码
        if ((e.target as HTMLElement).closest('.react-flow__node')) return;
        void onPaneDoubleClick();
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={styledEdges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onConnectStart={onConnectStart}
        onConnectEnd={onConnectEnd}
        isValidConnection={isValidConnection}
        onSelectionChange={onSelectionChange}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeDoubleClick={onNodeDoubleClick}
        fitView
        minZoom={0.2}
        snapToGrid
        snapGrid={[SNAP_GRID, SNAP_GRID]}
        deleteKeyCode={['Backspace', 'Delete']}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={SNAP_GRID} size={1} />
        <Controls showInteractive={false} />
        <MiniMap
          nodeStrokeWidth={3}
          pannable zoomable
          style={{ height: 90 }}
        />
      </ReactFlow>
    </div>
  );
}
