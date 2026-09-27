/** React Flow 自定义节点：模块节点（基础/大模型/自定义）+ 模型输入/输出虚拟节点。 */
import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { IONodeData, ModuleNodeData } from '../stores/graphStore';
import type { Port } from '../schema/graph';
import { fmtShape } from '../utils/shape';

const GROUP_COLOR: Record<string, string> = {
  basic: '#1677ff',
  llm: '#722ed1',
  custom: '#fa8c16',
};
const GROUP_NAME: Record<string, string> = {
  basic: '基础', llm: '大模型', custom: '自定义',
};
// 把手配色与 .mf-port.in/.out 同色系（输入蓝 / 输出品红），半圆随默认 translate 伸出边界
const HANDLE_COLOR = { in: '#1677ff', out: '#eb2f96' };
const handleStyle = (c: string) => ({ width: 10, height: 10, background: c, borderColor: '#fff' });

const ROW_H = 22;
const HEAD_H = 62;

export function shapeText(p: Port): string {
  return p.shape.length ? fmtShape(p.shape) : '?';
}

/** 只有 data/选中/拖动态变化才重渲染：React Flow 每帧传 positionAbsoluteX/Y，
 *  默认浅比较拦不住，是拖拽卡顿的主要来源之一。 */
const sameNodeProps = (a: NodeProps, b: NodeProps) =>
  a.id === b.id && a.data === b.data && a.selected === b.selected &&
  (a as unknown as { dragging?: boolean }).dragging === (b as unknown as { dragging?: boolean }).dragging;

export const ModuleNode = memo(function ModuleNode({ data, selected }: NodeProps) {
  const d = data as unknown as ModuleNodeData;
  const color = GROUP_COLOR[d.group] ?? '#1677ff';
  const rows = Math.max(d.inputs.length, d.outputs.length, 1);
  const height = HEAD_H + rows * ROW_H + 10;
  const attrSummary = Object.entries(d.attrs)
    .map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(',')}]` : String(v)}`)
    .join('  ');

  return (
    <div
      className={`mf-node ${selected ? 'selected' : ''} ${d.error ? 'error' : ''}`}
      style={{ borderColor: d.error ? '#ff4d4f' : color, width: 236, minHeight: height }}
    >
      <div className="mf-node-head" style={{ background: color }}>
        <span className="mf-node-group">{GROUP_NAME[d.group] ?? d.group}</span>
        <span className="mf-node-label">{d.label}</span>
      </div>
      <div className="mf-node-name" title={d.doc}>{d.name}</div>
      {attrSummary && (
        <div className="mf-node-attrs" title={attrSummary} style={{ fontSize: 11 }}>{attrSummary}</div>
      )}
      <div className="mf-node-ports">
        {d.inputs.map((p) => (
          <div key={`in-${p.name}`} className="mf-port-row">
            {/* 把手由 RF 默认样式 top:50% 在行内垂直居中（行是定位上下文），勿加内联 top */}
            <Handle type="target" position={Position.Left} id={p.name} style={handleStyle(HANDLE_COLOR.in)} />
            <span className="mf-port in">● {p.name}</span>
            <span className="mf-port-shape">{p.dtype} {shapeText(p)}</span>
          </div>
        ))}
        {d.outputs.map((p) => (
          <div key={`out-${p.name}`} className="mf-port-row out">
            <span className="mf-port-shape">{p.dtype} {shapeText(p)}</span>
            <span className="mf-port out">{p.name} ●</span>
            <Handle type="source" position={Position.Right} id={p.name} style={handleStyle(HANDLE_COLOR.out)} />
          </div>
        ))}
      </div>
    </div>
  );
}, sameNodeProps);

export const IONode = memo(function IONode({ data, selected }: NodeProps) {
  const d = data as unknown as IONodeData;
  const isIn = d.kind === 'in';
  const color = isIn ? '#52c41a' : '#eb2f96';
  return (
    <div
      className={`mf-node io ${selected ? 'selected' : ''}`}
      style={{ borderColor: color, width: 200 }}
    >
      <div className="mf-node-head" style={{ background: color }}>
        <span className="mf-node-label">{d.label}</span>
      </div>
      <div className="mf-node-ports">
        {d.ports.map((p) => (
          <div key={p.name} className={`mf-port-row ${isIn ? 'out' : ''}`}>
            {isIn ? (
              <>
                <span className="mf-port-shape">{p.dtype} {shapeText(p)}</span>
                <span className="mf-port out">{p.name} ●</span>
                <Handle type="source" position={Position.Right} id={p.name} style={handleStyle(HANDLE_COLOR.out)} />
              </>
            ) : (
              <>
                <Handle type="target" position={Position.Left} id={p.name} style={handleStyle(HANDLE_COLOR.in)} />
                <span className="mf-port in">● {p.name}</span>
                <span className="mf-port-shape">{p.dtype} {shapeText(p)}</span>
              </>
            )}
          </div>
        ))}
        {d.ports.length === 0 && <div className="mf-port-row"><span className="mf-port-shape">（未定义端口）</span></div>}
      </div>
    </div>
  );
}, sameNodeProps);

export const nodeTypes = { module: ModuleNode, io: IONode };
