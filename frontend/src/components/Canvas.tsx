/** 中央画布：React Flow 封装（拖拽放置、连线校验、双击查看源码）。
 *
 * 注意：传给 React Flow 的回调必须保持稳定身份（useCallback + getState()），
 * 否则内部 store 同步 → onSelectionChange → set → 重渲染 会形成死循环。
 */
import { useCallback, useMemo, useRef } from 'react';
import {
  Background, Controls, MiniMap, ReactFlow, useReactFlow,
  type Connection, type Edge, type Node, type IsValidConnection,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { message } from 'antd';
import { useGraphStore } from '../stores/graphStore';
import { useUIStore } from '../stores/uiStore';
import { nodeTypes } from '../nodes/ModuleNode';
import { checkConnection, portOfGraph } from '../utils/shape';
import * as api from '../api/client';
import { GRAPH_IN, GRAPH_OUT } from '../schema/graph';

export default function Canvas() {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const { screenToFlowPosition } = useReactFlow();
  const wrapper = useRef<HTMLDivElement>(null);

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

  const isValidConnection: IsValidConnection = useCallback((conn) => {
    const store = useGraphStore.getState();
    const graph = store.exportGraph();
    if (!conn.source || !conn.target || !conn.sourceHandle || !conn.targetHandle) return false;
    if (conn.target === GRAPH_IN || conn.source === GRAPH_OUT) return false;
    if (conn.source === conn.target) return false;
    const already = store.edges.some(
      (e) => e.target === conn.target && e.targetHandle === conn.targetHandle);
    if (already) return false;
    const src = portOfGraph(graph, conn.source, conn.sourceHandle);
    const tgt = portOfGraph(graph, conn.target, conn.targetHandle);
    return src && tgt ? checkConnection(src.port, tgt.port) === null : false;
  }, []);

  const onConnect = useCallback((conn: Connection) => {
    const store = useGraphStore.getState();
    const ok = store.onConnect(conn);
    if (!ok) {
      const graph = store.exportGraph();
      const src = conn.source && conn.sourceHandle ? portOfGraph(graph, conn.source, conn.sourceHandle) : undefined;
      const tgt = conn.target && conn.targetHandle ? portOfGraph(graph, conn.target, conn.targetHandle) : undefined;
      const reason = src && tgt ? checkConnection(src.port, tgt.port) : '端口不存在';
      message.error(`无法连接：${reason ?? '输入端口已被占用'}`);
    }
  }, []);

  const onSelectionChange = useCallback(({ nodes: selNodes, edges: selEdges }: { nodes: Node[]; edges: Edge[] }) => {
    useGraphStore.getState().setSelection(
      selNodes.map((n) => n.id), selEdges.map((e) => e.id));
  }, []);

  const onNodesChange = useCallback((ch: unknown[]) => {
    useGraphStore.getState().onNodesChange(ch);
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
      const detail = e instanceof api.ApiError ? e.detail : null;
      const msg = detail && typeof detail === 'object' && 'errors' in (detail as object)
        ? (detail as { errors: { where?: string; message: string }[] }).errors
            .map((x) => `${x.where ? `${x.where}: ` : ''}${x.message}`).join('；')
        : e instanceof Error ? e.message : '生成失败';
      message.error(`生成失败：${msg}`);
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
        isValidConnection={isValidConnection}
        onSelectionChange={onSelectionChange}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeDoubleClick={onNodeDoubleClick}
        fitView
        minZoom={0.2}
        deleteKeyCode={['Backspace', 'Delete']}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={18} size={1} />
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
