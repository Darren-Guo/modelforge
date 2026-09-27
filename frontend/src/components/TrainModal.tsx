/** 场景3：训练面板（选数据集/超参 → 启动训练 → SSE 实时日志）。 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, Button, Form, Input, InputNumber, Modal, Select, Space, Spin, Tag, message } from 'antd';
import { PauseCircleOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { useGraphStore } from '../stores/graphStore';
import { useUIStore } from '../stores/uiStore';
import type { DatasetInfo, TrainingJob } from '../schema/graph';
import * as api from '../api/client';
import { apiErrorText } from './errors';

export default function TrainModal() {
  // 性能：全店订阅会让弹窗随拖拽高频重渲染——动作走 getState() 快照
  const store = useGraphStore.getState();
  const modelMetaName = useGraphStore((s) => s.modelMeta.name);
  const { trainOpen, setTrainOpen, setModelsOpen } = useUIStore();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [dataset, setDataset] = useState('random');
  const [modelId, setModelId] = useState('');
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(32);
  const [lr, setLr] = useState(0.001);
  const [numSamples, setNumSamples] = useState(512);
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [starting, setStarting] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const esRef = useRef<EventSource | null>(null);
  const streamDone = useRef(false); // 收到终止事件/主动关闭后，不再把流关闭当成断连
  const logEnd = useRef<HTMLDivElement>(null);

  /** 订阅任务日志流（SSE 从 idx=0 重放历史，天然支持中途重新接入）。 */
  const attach = useCallback((j: TrainingJob) => {
    esRef.current?.close();
    streamDone.current = false;
    setJob(j);
    setLogs([]);
    const es = new EventSource(`/api/trainings/${j.job_id}/logs`);
    esRef.current = es;
    es.onmessage = (ev) => {
      try {
        const evt = JSON.parse(ev.data);
        if (evt.type === 'metric') {
          const acc = evt.val_acc !== undefined ? ` acc=${evt.val_acc}` : '';
          setLogs((l) => [...l, `epoch ${evt.epoch}  train_loss=${evt.train_loss}  val_loss=${evt.val_loss}${acc}  (${evt.seconds}s)`]);
        } else if (evt.type === 'status') {
          setLogs((l) => [...l, `[status] ${evt.message ?? JSON.stringify(evt)}`]);
          if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'stopped') {
            // 终止事件：同步任务状态（失败/停止时后端不会再发 done），关闭流
            streamDone.current = true;
            es.close();
            setJob((cur) => (cur ? { ...cur, status: evt.status } : cur));
          }
        } else if (evt.type === 'error') {
          setLogs((l) => [...l, `[error] ${evt.message}`]);
          if (evt.message === '任务不存在') {
            streamDone.current = true;
            es.close();
          }
        } else if (evt.type === 'done') {
          setLogs((l) => [...l, `[done] ${evt.message}`]);
          streamDone.current = true;
          es.close();
          setJob((cur) => (cur ? { ...cur, status: 'done' } : cur));
        }
      } catch { /* 非 JSON 行忽略 */ }
    };
    es.onerror = () => {
      if (streamDone.current) {
        es.close();
        return;
      }
      if (es.readyState === EventSource.CLOSED) {
        setLogs((l) => [...l, '[连接断开]']);
        streamDone.current = true;
        es.close();
      }
      // CONNECTING：浏览器在自动重连（重连后从头重放），不打扰用户
    };
  }, []);

  useEffect(() => {
    if (trainOpen) {
      void api.fetchDatasets().then(setDatasets).catch(() => setDatasets([]));
      setModelId(`${modelMetaName || 'model'}-${new Date().toISOString().slice(5, 16).replace(/[-T:]/g, '')}`);
      // 关闭弹窗只是解绑流；重开时把仍在跑的任务接回来继续看日志
      void api.listTrainings()
        .then((jobs) => {
          const running = jobs.find((j) => j.status === 'running');
          if (running) {
            attach(running);
            setLogs((l) => [...l, `[已重新连接训练任务 ${running.model_id}]`]);
          }
        })
        .catch(() => { /* 后端不可达时忽略 */ });
    } else {
      esRef.current?.close();
      esRef.current = null;
      streamDone.current = true;
      setJob(null);
      setLogs([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainOpen]);

  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const start = async () => {
    setStarting(true);
    setLogs([]);
    try {
      const res = await api.startTraining({
        graph: store.exportGraph(),
        model_id: modelId.trim() || null,
        dataset, epochs, batch_size: batchSize, lr, num_samples: numSamples,
      });
      attach(res.job);
      message.success(`训练已启动，模型 ID：${res.model_id}`);
    } catch (e) {
      message.error({ content: `训练启动失败：${apiErrorText(e, '启动失败')}`, duration: 6 });
    } finally {
      setStarting(false);
    }
  };

  const stop = async () => {
    if (!job) return;
    try {
      await api.stopTraining(job.job_id);
      setJob((j) => (j ? { ...j, status: 'stopped' } : j));
      setLogs((l) => [...l, '[已手动停止]']);
    } catch (e) {
      message.error(`停止训练失败：${apiErrorText(e, '未知错误')}`);
    }
  };

  const ds = datasets.find((d) => d.id === dataset);

  return (
    <Modal
      title="训练模型"
      open={trainOpen}
      onCancel={() => setTrainOpen(false)}
      width={720}
      footer={[
        <Button key="close" onClick={() => setTrainOpen(false)}>关闭</Button>,
        job && job.status === 'running' ? (
          <Button key="stop" danger icon={<PauseCircleOutlined />} onClick={() => void stop()}>停止</Button>
        ) : (
          <Button key="start" type="primary" icon={<PlayCircleOutlined />}
            loading={starting} onClick={() => void start()}>
            开始训练
          </Button>
        ),
        <Button key="models" onClick={() => { setTrainOpen(false); setModelsOpen(true); }}>
          已训练模型列表
        </Button>,
      ]}
    >
      <Form layout="vertical" size="small">
        <Form.Item label="模型 ID（训练完成后凭它在模型列表中查找）">
          <Input value={modelId} onChange={(e) => setModelId(e.target.value)} placeholder="自动生成，可修改" />
        </Form.Item>
        <Form.Item label="数据集">
          <Select value={dataset} onChange={setDataset}
            options={datasets.map((d) => ({ value: d.id, label: `${d.name}（${d.task === 'regression' ? '回归' : '分类'}）` }))} />
          {ds && <Alert type="info" showIcon style={{ marginTop: 6 }} message={ds.doc} />}
        </Form.Item>
        <Space size="large" wrap>
          <Form.Item label="轮数 epochs"><InputNumber min={1} max={100} value={epochs} onChange={(v) => setEpochs(v ?? 3)} /></Form.Item>
          <Form.Item label="批大小"><InputNumber min={1} max={1024} value={batchSize} onChange={(v) => setBatchSize(v ?? 32)} /></Form.Item>
          <Form.Item label="学习率"><InputNumber min={0.00001} max={1} step={0.0001} value={lr} onChange={(v) => setLr(v ?? 0.001)} /></Form.Item>
          <Form.Item label="样本数"><InputNumber min={32} max={20000} step={64} value={numSamples} onChange={(v) => setNumSamples(v ?? 512)} /></Form.Item>
        </Space>
      </Form>

      {job && (
        <div className="mf-train-log">
          <Space style={{ marginBottom: 6 }}>
            <Tag color={job.status === 'running' ? 'processing' : job.status === 'done' ? 'success' : job.status === 'stopped' ? 'warning' : 'error'}>
              {job.status}
            </Tag>
            <span style={{ fontSize: 12, color: '#888' }}>模型 ID: {job.model_id}</span>
            {job.status === 'running' && <Spin size="small" />}
          </Space>
          <div className="mf-train-log-body">
            {logs.map((l, i) => <div key={i}>{l}</div>)}
            <div ref={logEnd} />
          </div>
        </div>
      )}
    </Modal>
  );
}
