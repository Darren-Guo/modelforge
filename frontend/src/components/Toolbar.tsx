/** 顶部工具栏：校验 / 导入导出 / 生成代码 / 保存为模块 / 训练 / 模型列表。 */
import { useRef } from 'react';
import { Badge, Button, Popconfirm, Popover, Space, Tag, Tooltip, message } from 'antd';
import {
  ApiOutlined, ApartmentOutlined, CheckCircleOutlined, CodeOutlined, CloudUploadOutlined,
  DownloadOutlined, ExperimentOutlined, FolderOpenOutlined, ImportOutlined,
  PartitionOutlined, WarningOutlined,
} from '@ant-design/icons';
import { dump as yamlDump, load as yamlLoad } from 'js-yaml';
import { useGraphStore } from '../stores/graphStore';
import { useUIStore } from '../stores/uiStore';
import type { Graph } from '../schema/graph';
import * as api from '../api/client';
import { apiErrorText } from './errors';
import { namespaceGraphIds } from './graphIds';

export default function Toolbar() {
  // 只订阅工具栏展示需要的切片；动作经 getState() 快照调用（引用稳定）
  const validation = useGraphStore((s) => s.validation);
  const modelMeta = useGraphStore((s) => s.modelMeta);
  const editing = useGraphStore((s) => s.editing);
  const selectedNodeIds = useGraphStore((s) => s.selectedNodeIds);
  const store = useGraphStore.getState();
  const ui = useUIStore();
  const fileRef = useRef<HTMLInputElement>(null);
  
  const onExport = (fmt: 'json' | 'yaml') => {
    const graph = store.exportGraph();
    const text = fmt === 'json'
      ? JSON.stringify(graph, null, 2)
      : yamlDump(graph, { lineWidth: 120, noRefs: true });
    const blob = new Blob([text], { type: 'application/octet-stream' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${graph.model.name || 'model'}.${fmt === 'json' ? 'json' : 'yaml'}`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const onImportFile = async (file: File) => {
    const text = await file.text();
    try {
      const g = (text.trim().startsWith('{') ? JSON.parse(text) : yamlLoad(text)) as Graph;
      if (!g.nodes || !g.inputs) throw new Error('不是合法的拓扑文件');
      // 文件里的节点/边 id 改名到独立命名空间，避免与会话内 nextId 撞出重复 id
      store.importGraph(namespaceGraphIds(g));
      useGraphStore.setState({ selectedNodeIds: [], selectedEdgeIds: [] });
      message.success(`已导入「${g.model?.name ?? file.name}」`);
    } catch (e) {
      message.error(`导入失败：${apiErrorText(e, '格式错误')}`);
    }
  };

  const onGenerate = async () => {
    try {
      const res = await api.codegen(store.exportGraph());
      ui.openSource({
        title: `「${modelMeta.name}」生成的代码包`,
        code: res.files['model.py'] ?? '',
        files: res.files,
        downloadFolder: res.folder,
      });
    } catch (e) {
      message.error({ content: `生成失败：${apiErrorText(e, '未知错误')}`, duration: 6 });
    }
  };

  const errorList = validation?.errors ?? [];
  const statusBadge = store.validating
    ? <Badge status="processing" text="校验中…" />
    : validation?.ok
      ? <Tag icon={<CheckCircleOutlined />} color="success">校验通过</Tag>
      : <Popover
          trigger="click"
          content={
            <div style={{ maxWidth: 420, maxHeight: 260, overflow: 'auto' }}>
              {errorList.map((e, i) => (
                <div key={i} style={{ marginBottom: 6 }}>
                  <Tag color="error">{e.where}</Tag>
                  <span style={{ fontSize: 12 }}>{e.message}</span>
                </div>
              ))}
              {!errorList.length && <span>无问题</span>}
            </div>
          }
        >
          <Tag icon={<WarningOutlined />} color="error" style={{ cursor: 'pointer' }}>
            {errorList.length} 个问题
          </Tag>
        </Popover>;

  return (
    <div className="mf-toolbar">
      <Space size="middle">
        <span className="mf-brand">
          <ApiOutlined /> ModelForge
        </span>
        <Tag color="processing">模型：{modelMeta.name || '未命名'}</Tag>
        {editing && <Tag color="orange">编辑模块：{editing.name}</Tag>}
        {statusBadge}
      </Space>
      <Space size="small" wrap>
        <Button size="small" icon={<CheckCircleOutlined />} onClick={() => void store.validateNow()}>
          校验
        </Button>
        <Tooltip title="自动整理布局：按数据流从左到右分层排列（netron / draw.io 风格）">
          <Button size="small" icon={<ApartmentOutlined />} onClick={() => store.autoLayout()}>
            自动布局
          </Button>
        </Tooltip>
        <Button size="small" icon={<ImportOutlined />} onClick={() => fileRef.current?.click()}>
          导入
        </Button>
        <input
          ref={fileRef} type="file" accept=".json,.yaml,.yml" style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void onImportFile(f);
            e.target.value = '';
          }}
        />
        <Tooltip title="导出拓扑 JSON">
          <Button size="small" icon={<DownloadOutlined />} onClick={() => onExport('json')}>JSON</Button>
        </Tooltip>
        <Tooltip title="导出拓扑 YAML（参考 ONNX 图结构）">
          <Button size="small" icon={<DownloadOutlined />} onClick={() => onExport('yaml')}>YAML</Button>
        </Tooltip>
        <Button size="small" type="primary" icon={<CodeOutlined />} onClick={() => void onGenerate()}>
          生成代码
        </Button>
        <Tooltip title="把选中的节点保存为自定义模块（Shift+点击多选）">
          <Button
            size="small" icon={<PartitionOutlined />}
            disabled={selectedNodeIds.filter((id) => !id.startsWith('__')).length === 0}
            onClick={() => ui.setSaveModuleOpen(true)}
          >
            保存为模块
          </Button>
        </Tooltip>
        <Button size="small" icon={<ExperimentOutlined />} onClick={() => ui.setTrainOpen(true)}>
          训练
        </Button>
        <Button size="small" icon={<FolderOpenOutlined />} onClick={() => ui.setModelsOpen(true)}>
          已训练模型
        </Button>
        <Popconfirm
          title="清空画布？此操作不可撤销"
          onConfirm={() => {
            store.clearCanvas();
            useGraphStore.setState({ selectedNodeIds: [], selectedEdgeIds: [] });
          }}
        >
          <Button size="small" danger icon={<CloudUploadOutlined />}>清空</Button>
        </Popconfirm>
      </Space>
    </div>
  );
}
