/** 已训练模型列表 + 详情 + 推理面板（输入表单按模型输入 spec 自适应）。 */
import { useCallback, useEffect, useState } from 'react';
import {
  Alert, Button, Card, Col, Drawer, Empty, Image, Progress, Row,
  Space, Table, Tag, Upload, message,
} from 'antd';
import {
  CodeOutlined, EyeOutlined, PlayCircleOutlined, ReloadOutlined, UploadOutlined,
} from '@ant-design/icons';
import { useUIStore } from '../stores/uiStore';
import type { ModelDetail, ModelSummary, PredictResult } from '../schema/graph';
import type { Port } from '../schema/graph';
import { fmtShape } from '../utils/shape';
import * as api from '../api/client';
import { apiErrorText } from './errors';

interface InputValue {
  kind: 'text' | 'image' | 'tensor';
  text?: string;
  base64?: string;
  dataText?: string;
}

function InputField({ port, hasVocab, value, onChange }: {
  port: Port; hasVocab: boolean; value: InputValue; onChange: (v: InputValue) => void;
}) {
  const isImage = port.dtype === 'float32'
    && port.shape.length >= 3
    && JSON.stringify(port.shape.slice(1)) === JSON.stringify([1, 28, 28]);
  const isText = port.dtype.startsWith('int') && hasVocab;

  return (
    <Card size="small" title={`${port.name}（${port.dtype} ${fmtShape(port.shape)}）`} style={{ marginBottom: 10 }}>
      {isText && (
        <>
          <InputFieldTextArea
            placeholder="输入文本，例如：the movie was great"
            value={value.text ?? ''}
            onChange={(text) => onChange({ kind: 'text', text })}
          />
          <div style={{ fontSize: 12, color: '#888' }}>训练时的词表会自动用于分词。</div>
        </>
      )}
      {isImage && (
        <Space align="start">
          <Upload
            accept="image/*"
            showUploadList={false}
            beforeUpload={(file) => {
              const reader = new FileReader();
              reader.onload = () => onChange({ kind: 'image', base64: String(reader.result).split(',')[1] });
              reader.readAsDataURL(file);
              return false;
            }}
          >
            <Button icon={<UploadOutlined />}>上传图片（28x28 灰度）</Button>
          </Upload>
          {value.base64 && <Image width={84} src={`data:image/png;base64,${value.base64}`} />}
        </Space>
      )}
      {!isText && !isImage && (
        <>
          <InputFieldTextArea
            placeholder="JSON 数组，如 [[0.1, 0.2, ...]] 或 [0.1, 0.2, ...]（batch 维可省略）"
            value={value.dataText ?? ''}
            onChange={(dataText) => onChange({ kind: 'tensor', dataText })}
          />
          <div style={{ fontSize: 12, color: '#888' }}>张量输入：按 JSON 数组填写，形状 {fmtShape(port.shape)}。</div>
        </>
      )}
    </Card>
  );
}

function InputFieldTextArea(props: { placeholder: string; value: string; onChange: (v: string) => void }) {
  return (
    <textarea
      className="mf-infer-textarea"
      placeholder={props.placeholder}
      value={props.value}
      onChange={(e) => props.onChange(e.target.value)}
    />
  );
}

export default function ModelsDrawer() {
  const { modelsOpen, setModelsOpen, openSource } = useUIStore();
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [values, setValues] = useState<Record<string, InputValue>>({});
  const [result, setResult] = useState<PredictResult | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setModels(await api.listModels());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (modelsOpen) void refresh();
    else { setDetail(null); setResult(null); }
  }, [modelsOpen, refresh]);

  const openDetail = async (modelId: string) => {
    try {
      const d = await api.getModel(modelId);
      setDetail(d);
      setResult(null);
      setValues(Object.fromEntries(
        (d.topology?.inputs ?? []).map((p) => [p.name, { kind: 'tensor' as const, text: '', dataText: '' }]),
      ));
    } catch (e) {
      message.error(`打开模型详情失败：${apiErrorText(e, '未知错误')}`);
    }
  };

  const showSource = async () => {
    if (!detail) return;
    try {
      const res = await api.modelFiles(detail.model_id);
      openSource({
        title: `「${detail.model_id}」训练时的源码快照`,
        code: res.files['model.py'] ?? '',
        files: res.files,
      });
    } catch (e) {
      message.error(`获取训练源码失败：${apiErrorText(e, '未知错误')}`);
    }
  };

  const doPredict = async () => {
    if (!detail) return;
    setPredicting(true);
    try {
      const inputs: Record<string, unknown> = {};
      for (const p of detail.topology?.inputs ?? []) {
        const v = values[p.name];
        if (!v) continue;
        if (v.kind === 'text') inputs[p.name] = { kind: 'text', text: v.text ?? '' };
        else if (v.kind === 'image') inputs[p.name] = { kind: 'image', base64: v.base64 ?? '' };
        else {
          try {
            inputs[p.name] = { kind: 'tensor', data: JSON.parse(v.dataText || '0') };
          } catch {
            message.error(`输入「${p.name}」不是合法 JSON 数组`);
            setPredicting(false);
            return;
          }
        }
      }
      setResult(await api.predict(detail.model_id, inputs));
    } catch (e) {
      message.error(e instanceof Error ? e.message : '推理失败');
    } finally {
      setPredicting(false);
    }
  };

  const columns = [
    { title: '模型 ID', dataIndex: 'model_id', render: (v: string) => <code>{v}</code> },
    { title: '名称', dataIndex: 'name' },
    { title: '数据集', dataIndex: 'dataset', render: (v: string) => <Tag>{v}</Tag> },
    {
      title: '指标',
      dataIndex: 'metrics',
      render: (m: Record<string, number>) => (
        <Space size={4}>
          {m.best_val_loss !== undefined && <Tag color="blue">val_loss {m.best_val_loss.toFixed(4)}</Tag>}
          {m.params !== undefined && <Tag>{Number(m.params).toLocaleString()} 参数</Tag>}
        </Space>
      ),
    },
    { title: '创建时间', dataIndex: 'created_at' },
    {
      title: '操作',
      render: (_: unknown, row: ModelSummary) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />} onClick={() => void openDetail(row.model_id)}>详情/推理</Button>
        </Space>
      ),
    },
  ];

  return (
    <Drawer
      title="已训练模型"
      open={modelsOpen}
      onClose={() => setModelsOpen(false)}
      width={detail ? '80%' : 720}
    >
      <Space style={{ marginBottom: 12 }}>
        <Button size="small" icon={<ReloadOutlined />} onClick={() => void refresh()}>刷新</Button>
        {detail && (
          <Button size="small" icon={<CodeOutlined />} onClick={() => void showSource()}>
            查看训练时源码
          </Button>
        )}
      </Space>

      {!detail && (
        <Table
          size="small"
          rowKey="model_id"
          loading={loading}
          columns={columns}
          dataSource={models}
          pagination={{ pageSize: 8 }}
          locale={{ emptyText: <Empty description="还没有训练好的模型：完成一次训练后出现在这里" /> }}
        />
      )}

      {detail && (
        <Row gutter={16}>
          <Col span={11}>
            <Card size="small" title="模型信息" style={{ marginBottom: 12 }}>
              <p><b>模型 ID：</b><code>{detail.model_id}</code></p>
              <p><b>名称：</b>{detail.name}</p>
              <p><b>数据集：</b>{detail.dataset}</p>
              <p><b>创建时间：</b>{detail.created_at}</p>
              <p>
                <b>指标：</b>
                {Object.entries(detail.metrics ?? {}).filter(([k]) => k !== 'history').map(([k, v]) => (
                  <Tag key={k}>{k}: {typeof v === 'number' ? Number(v).toFixed(4) : String(v)}</Tag>
                ))}
              </p>
              <Button size="small" onClick={() => setDetail(null)}>← 返回列表</Button>
            </Card>
            <Card size="small" title="推理">
              {(detail.topology?.inputs ?? []).map((p) => (
                <InputField
                  key={p.name} port={p} hasVocab={detail.has_vocab}
                  value={values[p.name] ?? { kind: 'tensor' }}
                  onChange={(v) => setValues((s) => ({ ...s, [p.name]: v }))}
                />
              ))}
              <Button type="primary" block icon={<PlayCircleOutlined />}
                loading={predicting} onClick={() => void doPredict()}>
                运行推理
              </Button>
            </Card>
          </Col>
          <Col span={13}>
            <Card size="small" title="推理结果">
              {!result && <Empty description="填写输入后点击「运行推理」" />}
              {result && Object.entries(result.outputs).map(([name, out]) => (
                <Card key={name} size="small" type="inner" title={`输出 ${name}  ${fmtShape(out.shape)}`} style={{ marginBottom: 10 }}>
                  {out.probabilities ? (
                    <>
                      <Alert type="success" showIcon message={`预测类别：${out.predicted_class}`} style={{ marginBottom: 8 }} />
                      {out.probabilities.map((p, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                          <span style={{ width: 56 }}>类别 {i}</span>
                          <Progress percent={Math.round(p * 100)} size="small" style={{ flex: 1 }} />
                          <span style={{ width: 56, textAlign: 'right' }}>{(p * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </>
                  ) : (
                    <div className="mf-infer-data">
                      {out.data?.slice(0, 64).map((x, i) => <Tag key={i}>{x}</Tag>)}
                      {out.preview_truncated && <div style={{ fontSize: 12, color: '#888' }}>仅显示前 64 个元素</div>}
                    </div>
                  )}
                </Card>
              ))}
            </Card>
          </Col>
        </Row>
      )}
    </Drawer>
  );
}
