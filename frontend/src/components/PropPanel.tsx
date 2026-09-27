/** 右侧属性面板：模型元信息 / 节点参数与端口 / 连线信息 / 自定义模块编辑条。 */
import { Button, Divider, Empty, Input, InputNumber, Popconfirm, Select, Switch, Tag, message } from 'antd';
import { DeleteOutlined, PlusOutlined, SaveOutlined, StopOutlined } from '@ant-design/icons';
import { useGraphStore, type IONodeData, type ModuleNodeData, type RFEdge, type RFNode } from '../stores/graphStore';
import type { AttrDef, Dim, Dtype, ModuleDef, Port } from '../schema/graph';
import { GRAPH_IN, GRAPH_OUT } from '../schema/graph';
import { fmtShape } from '../utils/shape';
import * as api from '../api/client';
import { apiErrorText } from './errors';
import { beginModuleEdit, endModuleEdit } from './moduleEditSession';

const DTYPES: Dtype[] = ['float32', 'float16', 'int64', 'int32', 'bool'];

function ShapeEditor({ shape, onChange, disabled }: {
  shape: Dim[]; onChange: (s: Dim[]) => void; disabled?: boolean;
}) {
  return (
    <div className="mf-shape-editor">
      {shape.map((d, i) => (
        <span key={i} className="mf-dim">
          <Input
            size="small"
            value={String(d)}
            disabled={disabled}
            style={{ width: 62 }}
            onChange={(e) => {
              const text = e.target.value.trim();
              const v: Dim = /^-?\d+$/.test(text) ? parseInt(text, 10) : text;
              const next = [...shape];
              next[i] = v;
              onChange(next);
            }}
          />
          {!disabled && (
            <a onClick={() => onChange(shape.filter((_, j) => j !== i))}>×</a>
          )}
        </span>
      ))}
      {!disabled && (
        <Button size="small" type="text" icon={<PlusOutlined />} onClick={() => onChange([...shape, 1])} />
      )}
    </div>
  );
}

function PortRow({ port, nameEditable, onPatch, onRemove }: {
  port: Port;
  nameEditable: boolean;
  onPatch: (patch: Partial<Port>) => void;
  onRemove?: () => void;
}) {
  return (
    <div className="mf-port-edit-row">
      <div className="mf-port-edit-line">
        {nameEditable ? (
          <Input size="small" value={port.name} style={{ width: 96 }}
            onChange={(e) => onPatch({ name: e.target.value })} />
        ) : (
          <Tag>{port.name}</Tag>
        )}
        <Select
          size="small"
          value={port.dtype}
          style={{ width: 92 }}
          options={DTYPES.map((d) => ({ value: d, label: d }))}
          onChange={(v) => onPatch({ dtype: v })}
        />
        {onRemove && (
          <Button size="small" type="text" danger icon={<DeleteOutlined />} onClick={onRemove} />
        )}
      </div>
      <ShapeEditor shape={port.shape} onChange={(s) => onPatch({ shape: s })} />
    </div>
  );
}

function AttrField({ def, value, onChange }: {
  def: AttrDef; value: unknown; onChange: (v: unknown) => void;
}) {
  if (def.type === 'bool') {
    return <Switch size="small" checked={Boolean(value)} onChange={onChange} />;
  }
  if (def.type === 'enum') {
    return (
      <Select
        size="small"
        value={String(value)}
        style={{ width: '100%' }}
        options={(def.choices ?? []).map((c) => ({ value: c, label: c }))}
        onChange={onChange}
      />
    );
  }
  if (def.type === 'int' || def.type === 'float') {
    return (
      <InputNumber
        size="small"
        value={value as number}
        min={def.min ?? undefined}
        max={def.max ?? undefined}
        step={def.type === 'int' ? 1 : 0.01}
        style={{ width: '100%' }}
        onChange={(v) => onChange(v ?? 0)}
      />
    );
  }
  if (def.type === 'int_list') {
    return (
      <Input
        size="small"
        value={(value as number[] | undefined)?.join(', ') ?? ''}
        placeholder="如：batch, 768 或 1, 224, 224"
        onChange={(e) => {
          const parts = e.target.value.split(/[,，\s]+/).filter(Boolean);
          onChange(parts.map((p) => (/^-?\d+$/.test(p) ? parseInt(p, 10) : p)));
        }}
      />
    );
  }
  return <Input size="small" value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} />;
}

export default function PropPanel() {
  // 性能：拖拽时 nodes/edges 每帧换新对象——仅订阅面板需要的切片；
  // 选中节点订阅其 data 引用（拖拽只改 position，不触发面板重渲染）。
  // 动作经 getState() 快照调用（引用稳定，store.xxx() 调用点不变）。
  const store = useGraphStore.getState();
  const selectedNodeIds = useGraphStore((s) => s.selectedNodeIds);
  const opIndex = useGraphStore((s) => s.opIndex);
  const moduleIndex = useGraphStore((s) => s.moduleIndex);
  const modelMeta = useGraphStore((s) => s.modelMeta);
  const editing = useGraphStore((s) => s.editing);
  const isEmptyCanvas = useGraphStore((s) => s.nodes.length === 0);
  const selectedNodeData = useGraphStore((s) =>
    s.selectedNodeIds.length === 1 ? s.nodes.find((n) => n.id === s.selectedNodeIds[0])?.data : undefined);
  const selectedEdge = useGraphStore((s) =>
    s.selectedEdgeIds.length === 1 ? s.edges.find((e) => e.id === s.selectedEdgeIds[0]) : undefined);
  const selectedNode = selectedNodeData && selectedNodeIds.length === 1
    ? ({ id: selectedNodeIds[0], data: selectedNodeData } as RFNode)
    : undefined;

  /** 打开自定义模块编辑器：模块索引里只有摘要，必须先取全量定义（否则 startEditModule 遍历 graph.nodes 崩溃）。 */
  const openModuleDef = async (id: string) => {
    try {
      beginModuleEdit(await api.getModule(id));
    } catch (e) {
      message.error(`打开模块定义失败：${apiErrorText(e, '未知错误')}`);
    }
  };

  /** 保存模块定义：失败（如内部子图校验 400）时保持编辑态并报告原因。 */
  const saveEdit = async () => {
    const name = editing?.name ?? '';
    try {
      await store.saveEditModule();
      endModuleEdit();
      message.success(`模块「${name}」已保存`);
    } catch (e) {
      message.error({ content: `保存模块定义失败：${apiErrorText(e, '未知错误')}`, duration: 6 });
    }
  };

  /** 图级端口变更：同步画布 IO 节点显示、迁移/清理相关连线。
   *  端口名即 Handle id（ModuleNode.tsx），只改 graphPorts 会产生悬空边与过期把手。 */
  const applyGraphPorts = (
    kind: 'inputs' | 'outputs',
    ports: Port[],
    touchEdges?: (edges: RFEdge[]) => RFEdge[],
  ) => {
    const s = useGraphStore.getState();
    const ioId = kind === 'inputs' ? GRAPH_IN : GRAPH_OUT;
    const nodes2: RFNode[] = s.nodes.map((n) =>
      n.id === ioId
        ? { ...n, data: { ...(n.data as IONodeData), ports: ports.map((p) => ({ ...p })) } }
        : n);
    useGraphStore.setState({
      graphInputs: kind === 'inputs' ? ports : s.graphInputs,
      graphOutputs: kind === 'outputs' ? ports : s.graphOutputs,
      nodes: nodes2,
      edges: touchEdges ? touchEdges(s.edges) : s.edges,
    });
    s.scheduleValidate();
  };

  const patchGraphPort = (kind: 'inputs' | 'outputs', index: number, patch: Partial<Port>) => {
    const list = kind === 'inputs' ? store.graphInputs : store.graphOutputs;
    const ports = [...list];
    const oldName = ports[index].name;
    ports[index] = { ...ports[index], ...patch };
    const newName = ports[index].name;
    const ioId = kind === 'inputs' ? GRAPH_IN : GRAPH_OUT;
    applyGraphPorts(kind, ports, oldName === newName ? undefined : (eds) => eds.map((e) => {
      if (kind === 'inputs' && e.source === ioId && e.sourceHandle === oldName) return { ...e, sourceHandle: newName };
      if (kind === 'outputs' && e.target === ioId && e.targetHandle === oldName) return { ...e, targetHandle: newName };
      return e;
    }));
  };

  const removeGraphPort = (kind: 'inputs' | 'outputs', index: number) => {
    const s = useGraphStore.getState();
    const list = kind === 'inputs' ? s.graphInputs : s.graphOutputs;
    const removed = list[index];
    const ports = list.filter((_, j) => j !== index);
    const ioId = kind === 'inputs' ? GRAPH_IN : GRAPH_OUT;
    const doomed = s.edges.filter((e) =>
      kind === 'inputs'
        ? e.source === ioId && e.sourceHandle === removed.name
        : e.target === ioId && e.targetHandle === removed.name);
    applyGraphPorts(kind, ports, (eds) => eds.filter((e) => !doomed.includes(e)));
    if (doomed.length > 0) message.info(`已删除端口「${removed.name}」及其 ${doomed.length} 条连线`);
  };

  // ---- 自定义模块编辑条 ----
  if (editing) {
    return (
      <div className="mf-props">
        <div className="mf-props-title">
          <Tag color="orange">正在编辑自定义模块</Tag>
        </div>
        <div className="mf-props-section">
          <div className="mf-props-label">模块名称</div>
          <Input
            size="small"
            value={editing.name}
            onChange={(e) => useGraphStore.setState({ editing: { ...editing, name: e.target.value } })}
          />
          <div className="mf-props-label">描述</div>
          <Input.TextArea
            rows={2}
            value={editing.description}
            onChange={(e) => useGraphStore.setState({ editing: { ...editing, description: e.target.value } })}
          />
        </div>
        <div className="mf-props-section">
          <div className="mf-props-hint">
            画布现在展示模块内部结构：可增删节点与连线，可编辑「模块输入/输出」端口（即对外接口）。
          </div>
          <Button type="primary" block icon={<SaveOutlined />} onClick={() => void saveEdit()}>
            保存模块定义
          </Button>
          <Button block icon={<StopOutlined />} onClick={() => endModuleEdit()} style={{ marginTop: 8 }}>
            取消并返回模型
          </Button>
        </div>
      </div>
    );
  }

  // ---- 节点属性 ----
  if (selectedNode && selectedNode.id !== GRAPH_IN && selectedNode.id !== GRAPH_OUT) {
    const d = selectedNode.data as ModuleNodeData;
    const def = opIndex[d.op];
    const mod: ModuleDef | undefined = d.op.startsWith('custom:') ? moduleIndex[d.op.slice(7)] : undefined;

    return (
      <div className="mf-props">
        <div className="mf-props-title">{d.label}</div>
        <div className="mf-props-section">
          <div className="mf-props-label">节点名称</div>
          <Input size="small" value={d.name} onChange={(e) => store.updateNodeName(selectedNode.id, e.target.value)} />
          <div className="mf-props-hint">{d.doc}</div>
        </div>

        <Divider plain>参数</Divider>
        <div className="mf-props-section">
          {def ? def.attrs.map((a) => (
            <div className="mf-attr-row" key={a.name}>
              <div className="mf-props-label">
                {a.label} <code>{a.name}</code>
              </div>
              <AttrField
                def={a}
                value={d.attrs[a.name]}
                onChange={(v) => store.updateNodeAttrs(selectedNode.id, { ...d.attrs, [a.name]: v })}
              />
            </div>
          )) : (
            <div className="mf-props-hint">
              自定义模块「{mod?.name ?? d.op}」的参数定义在模块内部。
              <Button size="small" block style={{ marginTop: 8 }}
                onClick={() => void openModuleDef(mod?.id ?? d.op.slice(7))}>
                编辑模块定义
              </Button>
            </div>
          )}
        </div>

        <Divider plain>输入端口</Divider>
        <div className="mf-props-section">
          {d.inputs.map((p) => (
            <PortRow key={p.name} port={p} nameEditable={false}
              onPatch={(patch) => store.updateNodePort(selectedNode.id, 'inputs', p.name, patch)} />
          ))}
        </div>

        <Divider plain>输出端口</Divider>
        <div className="mf-props-section">
          {d.outputs.map((p) => (
            <PortRow key={p.name} port={p} nameEditable={false}
              onPatch={(patch) => store.updateNodePort(selectedNode.id, 'outputs', p.name, patch)} />
          ))}
          <div className="mf-props-hint">输出端口 shape 可手动修改（冲突时校验会报错）；输入端口 shape 由上游连线推导自动回填。</div>
        </div>

        <Divider plain>操作</Divider>
        <div className="mf-props-section">
          <Popconfirm
            title={`删除节点「${d.name}」？`}
            description="与它相连的边会一并删除"
            okText="删除"
            cancelText="取消"
            onConfirm={() => store.deleteElements([selectedNode.id], [])}
          >
            <Button block danger icon={<DeleteOutlined />}>删除节点</Button>
          </Popconfirm>
          <div className="mf-props-hint">也可以选中后按 Delete / Backspace。</div>
        </div>
      </div>
    );
  }

  // ---- 图级输入/输出节点 ----
  if (selectedNode && (selectedNode.id === GRAPH_IN || selectedNode.id === GRAPH_OUT)) {
    const kind = selectedNode.id === GRAPH_IN ? 'inputs' : 'outputs';
    const ports = kind === 'inputs' ? store.graphInputs : store.graphOutputs;
    return (
      <div className="mf-props">
        <div className="mf-props-title">{kind === 'inputs' ? '模型输入' : '模型输出'}</div>
        <div className="mf-props-section">
          {ports.map((p, i) => (
            <div key={i}>
              <PortRow
                port={p}
                nameEditable
                onPatch={(patch) => patchGraphPort(kind, i, patch)}
                onRemove={() => removeGraphPort(kind, i)}
              />
              <Divider style={{ margin: '6px 0' }} />
            </div>
          ))}
          <Button block size="small" icon={<PlusOutlined />}
            onClick={() => applyGraphPorts(kind, [...ports, { name: `${kind === 'inputs' ? 'in' : 'out'}${ports.length + 1}`, dtype: 'float32', shape: ['batch', 8] }])}>
            添加端口
          </Button>
          <div className="mf-props-hint">端口的名称、dtype、shape 都可修改，保存为自定义模块时沿用。</div>
          <div className="mf-props-hint">「模型输入 / 模型输出」是图的固定端点，不可删除。</div>
        </div>
      </div>
    );
  }

  // ---- 连线 ----
  if (selectedEdge) {
    return (
      <div className="mf-props">
        <div className="mf-props-title">连线</div>
        <div className="mf-props-section">
          <div className="mf-props-hint">
            {selectedEdge.source}.{selectedEdge.sourceHandle} → {selectedEdge.target}.{selectedEdge.targetHandle}
          </div>
          <Popconfirm title="删除这条连线？" onConfirm={() => store.deleteElements([], [selectedEdge.id])}>
            <Button block danger icon={<DeleteOutlined />}>删除连线</Button>
          </Popconfirm>
        </div>
      </div>
    );
  }

  // ---- 默认：模型元信息 ----
  return (
    <div className="mf-props">
      <div className="mf-props-title">模型信息</div>
      <div className="mf-props-section">
        <div className="mf-props-label">模型名称</div>
        <Input size="small" value={modelMeta.name} onChange={(e) => store.setModelMeta({ name: e.target.value })} />
        <div className="mf-props-label">描述</div>
        <Input.TextArea rows={2} value={modelMeta.description}
          onChange={(e) => store.setModelMeta({ description: e.target.value })} />
      </div>
      <Divider plain>图级端口</Divider>
      <div className="mf-props-section">
        <div className="mf-props-label">输入</div>
        {store.graphInputs.map((p, i) => (
          <div key={i} className="mf-port-edit-line" style={{ marginBottom: 4 }}>
            <Tag color="green">{p.name}</Tag>
            <code>{p.dtype} {fmtShape(p.shape)}</code>
          </div>
        ))}
        <div className="mf-props-label">输出</div>
        {store.graphOutputs.map((p, i) => (
          <div key={i} className="mf-port-edit-line" style={{ marginBottom: 4 }}>
            <Tag color="magenta">{p.name}</Tag>
            <code>{p.dtype} {fmtShape(p.shape)}</code>
          </div>
        ))}
        <div className="mf-props-hint">点击画布上的「模型输入 / 模型输出」节点可编辑端口。</div>
      </div>
      {selectedNodeIds.length > 1 && (
        <>
          <Divider plain>多选</Divider>
          <div className="mf-props-section">
            <div className="mf-props-hint">已选中 {selectedNodeIds.length} 个节点，可在工具栏「保存为模块」。</div>
            <Popconfirm
              title={`删除选中的 ${selectedNodeIds.length} 个节点？`}
              description="与它们相连的边会一并删除"
              okText="删除"
              cancelText="取消"
              onConfirm={() => store.deleteElements(selectedNodeIds, [])}
            >
              <Button block danger icon={<DeleteOutlined />}>删除选中节点</Button>
            </Popconfirm>
          </div>
        </>
      )}
      {isEmptyCanvas && <Empty description="拖动左侧算子到画布开始搭建" style={{ marginTop: 40 }} />}
    </div>
  );
}
