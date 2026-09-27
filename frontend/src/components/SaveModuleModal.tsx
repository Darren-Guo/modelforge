/** 场景2：把框选的子图保存为自定义模块（名称/输入输出端口全可编辑）。 */
import { useEffect, useMemo, useState } from 'react';
import { Alert, Input, Modal, Tag, message } from 'antd';
import { useGraphStore } from '../stores/graphStore';
import { useUIStore } from '../stores/uiStore';
import type { Dtype, Port } from '../schema/graph';
import * as api from '../api/client';
import { apiErrorText } from './errors';

const DTYPES: Dtype[] = ['float32', 'float16', 'int64', 'int32', 'bool'];

export default function SaveModuleModal() {
  // 性能：全店订阅会让弹窗随拖拽高频重渲染——动作走 getState() 快照
  const store = useGraphStore.getState();
  const moduleCount = useGraphStore((s) => s.modules.length);
  const { saveModuleOpen, setSaveModuleOpen } = useUIStore();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [inputs, setInputs] = useState<Port[]>([]);
  const [outputs, setOutputs] = useState<Port[]>([]);
  const [saving, setSaving] = useState(false);

  // 打开时快照计算（打开后画布不再变化）
  const extract = useMemo(
    () => (saveModuleOpen
      ? useGraphStore.getState().extractSubgraph(useGraphStore.getState().selectedNodeIds)
      : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [saveModuleOpen],
  );

  useEffect(() => {
    if (!extract) return;
    setInputs(extract.inputGroups.map((g) => ({ name: g.suggestedName, dtype: g.dtype, shape: [...g.shape] })));
    setOutputs(extract.outputGroups.map((g) => ({ name: g.suggestedName, dtype: g.dtype, shape: [...g.shape] })));
    setName(`MyModule${moduleCount + 1}`);
    setDescription('');
  }, [extract]);

  const patchPort = (list: Port[], setList: (p: Port[]) => void, i: number, patch: Partial<Port>) => {
    const next = [...list];
    next[i] = { ...next[i], ...patch };
    setList(next);
  };

  const onSave = async () => {
    if (!extract) return;
    if (!name.trim()) {
      message.warning('请填写模块名称');
      return;
    }
    setSaving(true);
    try {
      const graph = store.buildModuleGraph(extract, inputs, outputs);
      await api.saveModule({ name: name.trim(), description, inputs, outputs, graph });
      await store.refreshModules();
      message.success(`自定义模块「${name}」已保存，可在左侧面板拖入使用`);
      setSaveModuleOpen(false);
    } catch (e) {
      message.error({ content: `保存失败：${apiErrorText(e, '未知错误')}`, duration: 6 });
    } finally {
      setSaving(false);
    }
  };

  const portEditor = (p: Port, onPatch: (patch: Partial<Port>) => void) => (
    <div className="mf-port-edit-row" key={p.name}>
      <div className="mf-port-edit-line">
        <Input size="small" value={p.name} style={{ width: 120 }}
          placeholder="端口名" onChange={(e) => onPatch({ name: e.target.value })} />
        <select
          className="mf-dtype-select"
          value={p.dtype}
          onChange={(e) => onPatch({ dtype: e.target.value as Dtype })}
        >
          {DTYPES.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
      </div>
      <Input
        size="small"
        value={p.shape.map(String).join(', ')}
        placeholder="shape，如：batch, 768"
        onChange={(e) => {
          const shape = e.target.value.split(/[,，\s]+/).filter(Boolean)
            .map((s) => (/^-?\d+$/.test(s) ? parseInt(s, 10) : s));
          onPatch({ shape });
        }}
      />
    </div>
  );

  return (
    <Modal
      title="保存为自定义模块"
      open={saveModuleOpen}
      onCancel={() => setSaveModuleOpen(false)}
      onOk={() => void onSave()}
      okText="保存"
      confirmLoading={saving}
      width={640}
    >
      {extract && (
        <>
          <div style={{ marginBottom: 12 }}>
            <Tag color="blue">内部节点 {extract.nodes.length} 个</Tag>
            <Tag color="green">输入端口 {extract.inputGroups.length} 个</Tag>
            <Tag color="magenta">输出端口 {extract.outputGroups.length} 个</Tag>
          </div>
          {extract.danglingInputs.length > 0 && (
            <Alert
              type="warning" showIcon style={{ marginBottom: 12 }}
              message={`以下输入端口未连接，保存后模块内部校验会失败：${extract.danglingInputs.join('、')}`}
            />
          )}
          <div className="mf-props-label">模块名称</div>
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="如 FFN" />
          <div className="mf-props-label">描述</div>
          <Input.TextArea rows={2} value={description} onChange={(e) => setDescription(e.target.value)} />

          <div className="mf-props-label" style={{ marginTop: 12 }}>输入端口（对外接口，可改名/dtype/shape）</div>
          {inputs.map((p, i) => portEditor(p, (patch) => patchPort(inputs, setInputs, i, patch)))}
          {inputs.length === 0 && <div className="mf-props-hint">子图没有来自外部的输入</div>}

          <div className="mf-props-label" style={{ marginTop: 12 }}>输出端口</div>
          {outputs.map((p, i) => portEditor(p, (patch) => patchPort(outputs, setOutputs, i, patch)))}
          {outputs.length === 0 && (
            <Alert type="warning" showIcon message="子图没有对外输出，至少选中一个输出被外部使用的节点" />
          )}
          <div className="mf-props-hint" style={{ marginTop: 8 }}>
            保存后出现在左侧面板「自定义模块」组，可在任意模型中拖入复用；点模块卡片上的「编辑定义」可回改。
          </div>
        </>
      )}
    </Modal>
  );
}
