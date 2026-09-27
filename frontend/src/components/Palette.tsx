/** 左侧算子面板：基础算子 / 大模型模块 / 自定义模块 三组，拖拽或点击放置。 */
import { useMemo, useState } from 'react';
import { Button, Collapse, Empty, Input, Popconfirm, Space, Tag, Tooltip, message } from 'antd';
import { useGraphStore, type ModuleNodeData } from '../stores/graphStore';
import { fmtShape } from '../utils/shape';
import * as api from '../api/client';
import { apiErrorText } from './errors';
import { beginModuleEdit } from './moduleEditSession';

export default function Palette() {
  // 性能：拖拽时 nodes 高频变化，这里只订阅低频切片；usage 派生成原语（位置变化不影响）
  const ops = useGraphStore((s) => s.ops);
  const modules = useGraphStore((s) => s.modules);
  const customUsage = useGraphStore((s) => {
    const counts: Record<string, number> = {};
    for (const n of s.nodes) {
      const op = (n.data as ModuleNodeData).op;
      if (op && op.startsWith('custom:')) counts[op.slice(7)] = (counts[op.slice(7)] ?? 0) + 1;
    }
    return JSON.stringify(counts);
  });
  const usage: Record<string, number> = useMemo(
    () => JSON.parse(customUsage || '{}'), [customUsage]);
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

  // 单击卡片 = 放置节点；「编辑定义 / 删除」是卡片上的显式按钮。
  // 不再用「双击编辑」：浏览器双击必然先派发两次 click，会先放两个节点再进编辑器。
  const card = (key: string, label: string, doc: string, payload: object, onAdd: () => void, extra?: React.ReactNode) => (
    <div
      key={key}
      className="mf-palette-item"
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData('application/modelforge-op', JSON.stringify(payload));
        e.dataTransfer.effectAllowed = 'move';
      }}
      onClick={onAdd}
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
      beginModuleEdit(await api.getModule(id));
    } catch (e) {
      message.error(`打开模块定义失败：${apiErrorText(e, '未知错误')}`);
    }
  };

  const removeModule = async (id: string, name: string) => {
    try {
      await api.deleteModule(id);
      await useGraphStore.getState().refreshModules();
      message.success(`已删除模块「${name}」`);
    } catch (e) {
      message.error(`删除模块失败：${apiErrorText(e, '未知错误')}`);
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
                () => useGraphStore.getState().addOpNode(o.op, 'basic', dropAt(i)))) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />,
          },
          {
            key: 'llm',
            label: <Tag color="purple">大模型模块</Tag>,
            children: groups.llm.length ? groups.llm.map((o, i) =>
              card(o.op, o.label, o.doc, { op: o.op, group: 'llm' },
                () => useGraphStore.getState().addOpNode(o.op, 'llm', dropAt(i)))) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />,
          },
          {
            key: 'custom',
            label: <Tag color="orange">自定义模块</Tag>,
            children: groups.custom.length ? groups.custom.map((m, i) =>
              card(
                m.id, m.name,
                `${m.description || '（无描述）'}｜${m.node_count} 个内部节点｜点「编辑定义」可回改`,
                { op: m.id, group: 'custom', kind: 'custom' },
                () => useGraphStore.getState().addCustomNode(m.id, dropAt(i)),
                <>
                  <div className="mf-palette-shape">
                    入 {m.inputs.map((p) => fmtShape(p.shape)).join(' ')} → 出 {m.outputs.map((p) => fmtShape(p.shape)).join(' ')}
                  </div>
                  <Space size={0}>
                    <Button size="small" type="link"
                      onClick={(e) => { e.stopPropagation(); void openModule(m.id); }}>
                      编辑定义
                    </Button>
                    <Popconfirm
                      title={`删除模块「${m.name}」？`}
                      description={(usage[m.id] ?? 0) > 0
                        ? `画布上有 ${usage[m.id] ?? 0} 个节点在使用，删除后它们将无法通过校验`
                        : undefined}
                      okText="删除"
                      cancelText="取消"
                      onConfirm={(e) => { e?.stopPropagation(); void removeModule(m.id, m.name); }}
                    >
                      <Button size="small" type="link" danger onClick={(e) => e.stopPropagation()}>
                        删除
                      </Button>
                    </Popconfirm>
                  </Space>
                </>,
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
