/** 右侧属性面板：模型元信息 / 节点参数与端口 / 连线信息 / 自定义模块编辑条。 */
import { Button, Divider, Empty, Input, InputNumber, Popconfirm, Select, Switch, Tag } from 'antd';
import { DeleteOutlined, PlusOutlined, SaveOutlined, StopOutlined } from '@ant-design/icons';
import { useGraphStore, type ModuleNodeData } from '../stores/graphStore';
import type { AttrDef, Dim, Dtype, ModuleDef, Port } from '../schema/graph';
import { GRAPH_IN, GRAPH_OUT } from '../schema/graph';
import { fmtShape } from '../utils/shape';

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
  const store = useGraphStore();
  const { nodes, edges, selectedNodeIds, selectedEdgeIds, opIndex, moduleIndex, modelMeta, editing } = store;

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
          <Button type="primary" block icon={<SaveOutlined />} onClick={() => void store.saveEditModule()}>
            保存模块定义
          </Button>
          <Button block icon={<StopOutlined />} onClick={() => store.cancelEditModule()} style={{ marginTop: 8 }}>
            取消并返回模型
          </Button>
        </div>
      </div>
    );
  }

  const selectedNode = selectedNodeIds.length === 1
    ? nodes.find((n) => n.id === selectedNodeIds[0])
    : undefined;
  const selectedEdge = selectedEdgeIds.length === 1
    ? edges.find((e) => e.id === selectedEdgeIds[0])
    : undefined;

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
                onClick={() => mod && useGraphStore.getState().startEditModule(mod)}>
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
          <div className="mf-props-hint">端口 shape 由形状推导自动回填；也可手动修改（冲突时校验会报错）。</div>
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
                onPatch={(patch) => store.updateGraphPort(kind, i, patch)}
                onRemove={() => store.setGraphPorts(kind, ports.filter((_, j) => j !== i))}
              />
              <Divider style={{ margin: '6px 0' }} />
            </div>
          ))}
          <Button block size="small" icon={<PlusOutlined />}
            onClick={() => store.setGraphPorts(kind, [...ports, { name: `${kind === 'inputs' ? 'in' : 'out'}${ports.length + 1}`, dtype: 'float32', shape: ['batch', 8] }])}>
            添加端口
          </Button>
          <div className="mf-props-hint">端口的名称、dtype、shape 都可修改，保存为自定义模块时沿用。</div>
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
          </div>
        </>
      )}
      {nodes.length === 0 && <Empty description="拖动左侧算子到画布开始搭建" style={{ marginTop: 40 }} />}
    </div>
  );
}
