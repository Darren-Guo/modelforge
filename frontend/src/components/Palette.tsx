/** 左侧算子面板：基础算子 / 大模型模块 / 自定义模块 三组，拖拽或点击放置。 */
import { useMemo, useState } from 'react';
import { Collapse, Empty, Input, Tag, Tooltip, message } from 'antd';
import { useGraphStore } from '../stores/graphStore';
import { fmtShape } from '../utils/shape';
import * as api from '../api/client';

export default function Palette() {
  const { ops, modules, addOpNode, addCustomNode } = useGraphStore();
  const [q, setQ] = useState('');

  const groups = useMemo(() => {
    const kw = q.trim().toLowerCase();
    const match = (label: string, op: string, doc: string) =>
      !kw || `${label} ${op} ${doc}`.toLowerCase().includes(kw);
    return {
      basic: ops.filter((o) => o.group === 'basic' && match(o.label, o.op, o.doc)),
      llm: ops.filter((o) => o.group === 'llm' && match(o.label, o.op, o.doc)),
      custom: modules.filter((m) => match(m.name, m.id, m.description)),
    };
  }, [ops, modules, q]);

  const dropAt = (i: number) => ({ x: 260 + (i % 3) * 60, y: 80 + (i % 8) * 60 });

  const card = (key: string, label: string, doc: string, payload: object, onAdd: () => void, extra?: React.ReactNode, onDblClick?: () => void) => (
    <div
      key={key}
      className="mf-palette-item"
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData('application/modelforge-op', JSON.stringify(payload));
        e.dataTransfer.effectAllowed = 'move';
      }}
      onClick={onAdd}
      onDoubleClick={onDblClick}
    >
      <Tooltip title={doc} placement="right">
        <div>
          <div className="mf-palette-label">{label}</div>
          {extra}
        </div>
      </Tooltip>
    </div>
  );

  const openModule = async (id: string) => {
    try {
      const mod = await api.getModule(id);
      useGraphStore.getState().startEditModule(mod);
    } catch (e) {
      message.error(e instanceof Error ? e.message : '打开模块失败');
    }
  };

  return (
    <div className="mf-palette">
      <Input.Search
        placeholder="搜索算子 / 模块"
        allowClear
        onChange={(e) => setQ(e.target.value)}
        size="small"
      />
      <Collapse
        size="small"
        defaultActiveKey={['basic', 'llm', 'custom']}
        className="mf-palette-collapse"
        items={[
          {
            key: 'basic',
            label: <Tag color="blue">基础算子</Tag>,
            children: groups.basic.length ? groups.basic.map((o, i) =>
              card(o.op, o.label, o.doc, { op: o.op, group: 'basic' },
                () => addOpNode(o.op, 'basic', dropAt(i)))) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />,
          },
          {
            key: 'llm',
            label: <Tag color="purple">大模型模块</Tag>,
            children: groups.llm.length ? groups.llm.map((o, i) =>
              card(o.op, o.label, o.doc, { op: o.op, group: 'llm' },
                () => addOpNode(o.op, 'llm', dropAt(i)))) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />,
          },
          {
            key: 'custom',
            label: <Tag color="orange">自定义模块</Tag>,
            children: groups.custom.length ? groups.custom.map((m, i) =>
              card(
                m.id, m.name,
                `${m.description || '（无描述）'}｜${m.node_count} 个内部节点｜双击编辑定义`,
                { op: m.id, group: 'custom', kind: 'custom' },
                () => addCustomNode(m.id, dropAt(i)),
                <div className="mf-palette-shape">
                  入 {m.inputs.map((p) => fmtShape(p.shape)).join(' ')} → 出 {m.outputs.map((p) => fmtShape(p.shape)).join(' ')}
                </div>,
                () => void openModule(m.id),
              )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="框选子图后可保存为模块" />,
          },
        ]}
      />
      <div className="mf-palette-hint">
        拖拽或点击算子放置到画布；双击节点查看源码；双击空白处生成整个模型代码。
      </div>
    </div>
  );
}
