/** 自动布局（netron / draw.io 风格）：dagre 按拓扑分层排列（数据流从左到右），
 *  坐标对齐到网格。导入的拓扑缺坐标或互相重叠时整体重排，也可从工具栏手动触发。
 *
 * 节点尺寸用估算值（与 ModuleNode / IONode 的固定宽高一致），无需等 React Flow
 * 测量；dagre 只用相对尺寸定列距与行距，±10px 的误差不影响观感。
 */
import { Graph, layout } from '@dagrejs/dagre';
import type { Edge, Node } from '@xyflow/react';

/** 画布网格（布局对齐与拖拽吸附共用）。 */
export const SNAP_GRID = 20;

const MODULE_W = 236;
const MODULE_BASE_H = 72; // 头部 62 + 底部内边距 10
const IO_W = 200;
const IO_BASE_H = 38; // IO 头部 + 端口容器内边距
const ROW_H = 22; // 单个端口行
const ATTR_H = 18; // 属性摘要行

function nodeSize(n: Node): { width: number; height: number } {
  if (n.type === 'io') {
    const ports = (n.data as { ports?: unknown[] }).ports?.length ?? 0;
    return { width: IO_W, height: IO_BASE_H + Math.max(ports, 1) * ROW_H };
  }
  const d = n.data as { inputs?: unknown[]; outputs?: unknown[]; attrs?: Record<string, unknown> };
  const rows = Math.max(d.inputs?.length ?? 0, d.outputs?.length ?? 0, 1);
  const attrs = d.attrs && Object.keys(d.attrs).length > 0 ? ATTR_H : 0;
  return { width: MODULE_W, height: MODULE_BASE_H + rows * ROW_H + attrs };
}

const snap = (v: number) => Math.round(v / SNAP_GRID) * SNAP_GRID;

/** 计算分层布局：返回 nodeId → 节点左上角坐标（已对齐网格）。 */
export function computeAutoLayout(nodes: Node[], edges: Edge[]): Map<string, { x: number; y: number }> {
  const g = new Graph();
  g.setGraph({ rankdir: 'LR', nodesep: 40, ranksep: 100, marginx: 20, marginy: 20 });
  g.setDefaultEdgeLabel(() => ({}));
  const sizes = new Map<string, { width: number; height: number }>();
  for (const n of nodes) {
    const s = nodeSize(n);
    sizes.set(n.id, s);
    g.setNode(n.id, s);
  }
  const ids = new Set(nodes.map((n) => n.id));
  for (const e of edges) {
    if (e.source !== e.target && ids.has(e.source) && ids.has(e.target)) g.setEdge(e.source, e.target);
  }
  layout(g);

  const out = new Map<string, { x: number; y: number }>();
  for (const n of nodes) {
    // dagre 返回中心点坐标，换算成 React Flow 的左上角坐标
    const p = g.node(n.id) as { x?: number; y?: number } | undefined;
    const s = sizes.get(n.id)!;
    out.set(n.id, {
      x: snap((p?.x ?? 0) - s.width / 2),
      y: snap((p?.y ?? 0) - s.height / 2),
    });
  }
  return out;
}

/** 自带坐标是否互相重叠（含完全相同）：重叠视为「没整理过」，导入时整体重排。 */
export function positionsOverlap(nodes: Node[]): boolean {
  const rects = nodes.map((n) => ({ x: n.position.x, y: n.position.y, ...nodeSize(n) }));
  for (let i = 0; i < rects.length; i++) {
    for (let j = i + 1; j < rects.length; j++) {
      const a = rects[i];
      const b = rects[j];
      if (
        a.x < b.x + b.width && b.x < a.x + a.width &&
        a.y < b.y + b.height && b.y < a.y + a.height
      ) {
        return true;
      }
    }
  }
  return false;
}
