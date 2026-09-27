/** 拓扑 Graph IR 的 TS 类型（与后端 Pydantic schema 一一对应）。 */

export type Dim = number | string;
export type Dtype = 'float32' | 'float16' | 'int64' | 'int32' | 'bool';

export interface Port {
  name: string;
  dtype: Dtype;
  shape: Dim[];
}

export interface GraphNode {
  id: string;
  name: string;
  op: string; // 算子 key 或 custom:<module_id>
  group: 'basic' | 'llm' | 'custom';
  attrs: Record<string, unknown>;
  inputs: Port[];
  outputs: Port[];
  ui: { x: number; y: number };
}

export interface PortRef {
  node: string;
  port: string;
}

export interface GraphEdge {
  id: string;
  from: PortRef;
  to: PortRef;
}

export interface ModelMeta {
  name: string;
  description: string;
}

export interface Graph {
  format: 'modelforge/graph';
  version: string;
  model: ModelMeta;
  inputs: Port[];
  outputs: Port[];
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// ---- 算子注册表 -------------------------------------------------------

export interface AttrDef {
  name: string;
  type: 'int' | 'float' | 'bool' | 'str' | 'enum' | 'int_list';
  default: unknown;
  label: string;
  min: number | null;
  max: number | null;
  choices: string[] | null;
  help: string;
}

export interface PortDef {
  name: string;
  dtype: Dtype;
  label: string;
}

export interface OpDef {
  op: string;
  group: 'basic' | 'llm';
  label: string;
  doc: string;
  attrs: AttrDef[];
  inputs: PortDef[];
  outputs: PortDef[];
}

// ---- 自定义模块 -------------------------------------------------------

export interface ModuleSummary {
  id: string;
  name: string;
  description: string;
  inputs: Port[];
  outputs: Port[];
  node_count: number;
  created_at: string;
  updated_at: string;
}

export interface ModuleDef {
  id: string;
  name: string;
  description: string;
  inputs: Port[];
  outputs: Port[];
  graph: Graph;
  created_at: string;
  updated_at: string;
}

// ---- 校验报告 ---------------------------------------------------------

export interface ValidationError {
  where: string;
  message: string;
}

export interface InferredPort {
  name: string;
  dtype: Dtype;
  shape: Dim[];
}

export interface ValidationReport {
  ok: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  nodes: { id: string; inputs: InferredPort[]; outputs: InferredPort[] }[];
  order: string[];
}

// ---- 训练/模型 --------------------------------------------------------

export interface DatasetInfo {
  id: string;
  name: string;
  task: 'regression' | 'classification';
  doc: string;
}

export interface TrainMetrics {
  type: 'metric';
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc?: number;
  val_acc?: number;
  seconds?: number;
}

export interface TrainingJob {
  job_id: string;
  model_id: string;
  model_name: string;
  dataset: string;
  hyperparams: Record<string, number>;
  status: 'running' | 'done' | 'failed' | 'stopped';
  progress: TrainMetrics | null;
  events: Record<string, unknown>[];
  run_dir: string;
}

export interface ModelSummary {
  model_id: string;
  name: string;
  dataset: string;
  metrics: Record<string, unknown>;
  status: string;
  created_at: string;
}

export interface ModelDetail extends ModelSummary {
  topology: Graph;
  run_dir: string;
  has_vocab: boolean;
  code_files: string[];
}

export interface PredictResult {
  task: string;
  dataset: string;
  outputs: Record<string, {
    shape: number[];
    data?: number[];
    preview_truncated?: boolean;
    probabilities?: number[];
    predicted_class?: number;
  }>;
}

export const GRAPH_IN = '__graph_in__';
export const GRAPH_OUT = '__graph_out__';
