/** 外部来源图的节点/边 id 改写到独立命名空间。
 *
 * 会话内 nextId 只会生成 n1/e1… 形态的 id（graphStore.ts:116-117），
 * 而文件导入、存储中的模块定义会把历史 id 原样带进画布（graphStore.importGraph /
 * startEditModule 都不回拨计数器），之后新增节点就会撞出重复 id，校验报「节点 id 重复」。
 * 在两个入口处统一改名为 x<n>_ 前缀，与 nextId 的命名空间彻底隔离。
 * 节点/边的对外 name 不受影响，生成代码与校验文案仍以 name 为准。
 */
import type { Graph, GraphEdge, GraphNode } from '../schema/graph';

let seq = 0;

export function namespaceGraphIds<T extends Pick<Graph, 'nodes' | 'edges'>>(g: T): T {
  seq += 1;
  const ns = `x${seq}_`;
  const nmap = new Map(g.nodes.map((n) => [n.id, ns + n.id]));
  const emap = new Map(g.edges.map((e) => [e.id, ns + e.id]));
  const nodes: GraphNode[] = g.nodes.map((n) => ({ ...n, id: nmap.get(n.id) ?? n.id }));
  const edges: GraphEdge[] = g.edges.map((e) => ({
    ...e,
    id: emap.get(e.id) ?? e.id,
    // 端点可能是虚拟节点 __graph_in__/__graph_out__，不在 nodes 里，保持原样
    from: { ...e.from, node: nmap.get(e.from.node) ?? e.from.node },
    to: { ...e.to, node: nmap.get(e.to.node) ?? e.to.node },
  }));
  return { ...g, nodes, edges };
}
