/** 后端 API 客户端。 */
import type {
  DatasetInfo, Graph, ModelDetail, ModelSummary, ModuleDef, ModuleSummary,
  OpDef, PredictResult, TrainingJob, ValidationReport,
} from '../schema/graph';

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    let detail: unknown = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? body;
    } catch { /* ignore */ }
    // FastAPI 422 的 detail 是 [{loc, msg}] 数组，拼成可读列表而不是整段 JSON
    const msg = Array.isArray(detail)
      ? detail.map((d) => {
          const item = d as { loc?: unknown[]; msg?: string } | null;
          return item?.msg ? `${(item.loc ?? []).join('.')}: ${item.msg}` : JSON.stringify(d);
        }).join('\n')
      : typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2);
    throw new ApiError(msg, detail);
  }
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  detail: unknown;
  constructor(message: string, detail: unknown) {
    super(message);
    this.detail = detail;
  }
}

// ---- 算子注册表 ----
export const fetchOps = () => request<OpDef[]>('/ops');

// ---- 图 ----
export const validateGraph = (graph: Graph) =>
  request<ValidationReport>('/validate', { method: 'POST', body: JSON.stringify({ graph }) });

export const codegen = (graph: Graph, saveAs?: string) =>
  request<{ files: Record<string, string>; folder: string }>(
    '/codegen', { method: 'POST', body: JSON.stringify({ graph, save_as: saveAs ?? null }) });

export const nodeSnippet = (graph: Graph, nodeId: string) =>
  request<{ kind: string; title: string; code: string }>(
    '/codegen/snippet', { method: 'POST', body: JSON.stringify({ graph, node_id: nodeId }) });

export const downloadUrl = (folder: string) =>
  `${BASE}/codegen/download/${encodeURIComponent(folder.split(/[\\/]/).pop() ?? folder)}`;

// ---- 自定义模块 ----
export const listModules = () => request<ModuleSummary[]>('/modules');
export const getModule = (id: string) => request<ModuleDef>(`/modules/${id}`);
export const deleteModule = (id: string) => request<{ ok: boolean }>(`/modules/${id}`, { method: 'DELETE' });
export const saveModule = (body: {
  id?: string | null; name: string; description: string;
  inputs: unknown[]; outputs: unknown[]; graph: Graph;
}) => request<ModuleDef>('/modules', { method: 'POST', body: JSON.stringify(body) });

// ---- 数据集 ----
export const fetchDatasets = () => request<DatasetInfo[]>('/datasets');

// ---- 训练 ----
export const startTraining = (body: {
  graph: Graph; model_id?: string | null; dataset: string;
  epochs: number; batch_size: number; lr: number; num_samples: number;
}) => request<{ job: TrainingJob; model_id: string }>('/trainings', {
  method: 'POST', body: JSON.stringify(body),
});

export const listTrainings = () => request<TrainingJob[]>('/trainings');
export const getTraining = (id: string) => request<TrainingJob>(`/trainings/${id}`);
export const stopTraining = (id: string) => request<{ ok: boolean }>(`/trainings/${id}/stop`, { method: 'POST' });

// ---- 模型 ----
export const listModels = () => request<ModelSummary[]>('/models');
export const getModel = (id: string) => request<ModelDetail>(`/models/${id}`);
export const modelFiles = (id: string) => request<{ files: Record<string, string> }>(`/models/${id}/files`);
export const modelSource = (id: string, file: string) =>
  request<{ file: string; content: string }>(`/models/${id}/source?file=${encodeURIComponent(file)}`);
export const predict = (id: string, inputs: Record<string, unknown>) =>
  request<PredictResult>(`/models/${id}/predict`, { method: 'POST', body: JSON.stringify({ inputs }) });
