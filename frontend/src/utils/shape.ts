/** 形状/端口兼容性工具（前端快速校验，后端 /api/validate 为权威结果）。 */
import type { Dim, Graph, GraphNode, Port } from '../schema/graph';
import { GRAPH_IN, GRAPH_OUT } from '../schema/graph';

export function dimsCompatible(a: Dim, b: Dim): boolean {
  if (typeof a === 'number' && typeof b === 'number') return a === b;
  return true; // 至少一边是符号：宽松通过
}

export function shapesCompatible(a: Dim[], b: Dim[]): boolean {
  if (a.length === 0 || b.length === 0) return true; // 未声明的形状不拦截
  return a.length === b.length && a.every((d, i) => dimsCompatible(d, b[i]));
}

export function fmtShape(shape: Dim[]): string {
  return `[${shape.join(', ')}]`;
}

export function parseShape(text: string): Dim[] {
  return text
    .split(/[,，\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => (/^-?\d+$/.test(s) ? parseInt(s, 10) : s));
}

/** 连线兼容性检查：返回 null 表示可连，否则返回原因。 */
export function checkConnection(
  sourcePort: Port | undefined,
  targetPort: Port | undefined,
): string | null {
  if (!sourcePort || !targetPort) return '端口不存在';
  if (sourcePort.dtype !== targetPort.dtype) {
    return `数据类型不匹配：${sourcePort.dtype} → ${targetPort.dtype}`;
  }
  if (sourcePort.shape.length && targetPort.shape.length) {
    if (sourcePort.shape.length !== targetPort.shape.length) {
      return `形状秩不匹配：${fmtShape(sourcePort.shape)} → ${fmtShape(targetPort.shape)}`;
    }
    for (let i = 0; i < sourcePort.shape.length; i++) {
      const a = sourcePort.shape[i];
      const b = targetPort.shape[i];
      if (typeof a === 'number' && typeof b === 'number' && a !== b) {
        return `第 ${i} 维不匹配：${a} → ${b}（${fmtShape(sourcePort.shape)} → ${fmtShape(targetPort.shape)}）`;
      }
    }
  }
  return null;
}

export function portOfNode(node: GraphNode, portName: string, kind: 'input' | 'output'): Port | undefined {
  return (kind === 'input' ? node.inputs : node.outputs).find((p) => p.name === portName);
}

/** 从 Graph JSON 中取指定端口（含虚拟端点）。 */
export function portOfGraph(
  graph: Graph,
  nodeId: string,
  portName: string,
): { port: Port; kind: 'input' | 'output' } | undefined {
  if (nodeId === GRAPH_IN) {
    const p = graph.inputs.find((x) => x.name === portName);
    return p ? { port: p, kind: 'output' } : undefined;
  }
  if (nodeId === GRAPH_OUT) {
    const p = graph.outputs.find((x) => x.name === portName);
    return p ? { port: p, kind: 'input' } : undefined;
  }
  const node = graph.nodes.find((n) => n.id === nodeId);
  if (!node) return undefined;
  const pi = node.inputs.find((x) => x.name === portName);
  if (pi) return { port: pi, kind: 'input' };
  const po = node.outputs.find((x) => x.name === portName);
  return po ? { port: po, kind: 'output' } : undefined;
}
