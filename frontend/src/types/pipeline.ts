// ── Block registry ────────────────────────────────────────────────────────────

export type BrowseMode = "open_file" | "save_file" | "directory";

export interface BlockParamSpec {
  key: string;
  type: string;
  default: unknown;
  required: boolean;
  description: string;
  example: unknown;
  browse_mode?: BrowseMode | null;
}

export interface BlockSpec {
  key: string;
  name: string;
  aliases?: string[];
  version: string;
  category: string;
  description: string;
  n_inputs: number;
  input_labels: string[];
  output_labels: string[];
  param_schema?: BlockParamSpec[];
  params: Record<string, unknown>;
  param_types?: Record<string, string>;
  param_descriptions: Record<string, string>;
  required_params?: string[];
  param_examples?: Record<string, unknown>;
  is_custom?: boolean;
  custom_filename?: string | null;
}

// ── Pipeline definition ───────────────────────────────────────────────────────

export interface NodePosition {
  x: number;
  y: number;
}

export interface PipelineNode {
  id: string;
  block: string;
  params: Record<string, unknown>;
  notes?: string | null;
  group_ids?: string[];
  position?: NodePosition;
  width?: number;
  height?: number;
}

export interface PipelineEdge {
  id: string;
  source: string;
  target: string;
  source_output?: number | null;
  sourceHandle?: string | null;
  target_input?: number | null;
  targetHandle?: string | null;
}

export interface CommentItem {
  id: string;
  title: string;
  description: string;
  color?: string | null;
  position: NodePosition;
  width: number;
  height: number;
  managed?: boolean;
  group_id?: string | null;
}

export interface PipelineGroup {
  id: string;
  name: string;
  description: string;
  comment_id?: string | null;
}

export interface Pipeline {
  name: string;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  comments?: CommentItem[];
  groups?: PipelineGroup[];
}

// ── API responses ─────────────────────────────────────────────────────────────

export interface PipelineSummary {
  id: string;
  name: string;
  path: string;
  updated_at: number;
}

export interface PipelineEnvelope {
  id: string;
  pipeline: Pipeline;
}

export interface NodeResult {
  node_id: string;
  checkpoint_id: string;
  history_hash: string;
  status: "executed" | "reused" | "error";
  message?: string;
}

export interface ExecuteResponse {
  pipeline_id: string;
  topological_order: string[];
  executed_nodes: string[];
  reused_nodes: string[];
  node_results: Record<string, NodeResult>;
}

export interface CancelExecutionResponse {
  pipeline_id: string;
  status: "cancelled" | "not_running";
}

export interface StalenessResponse {
  pipeline_id: string;
  stale: Record<string, boolean>;
  history_hashes: Record<string, string>;
}

export interface CheckpointPreview {
  checkpoint_id: string;
  rows: Record<string, unknown>[];
  columns: string[];
  dtypes: Record<string, string>;
  total_rows: number;
}

// ── Node runtime state ────────────────────────────────────────────────────────

export type NodeStatus = "idle" | "stale" | "running" | "complete" | "error";

export interface NodeState {
  status: NodeStatus;
  checkpointId?: string;
  historyHash?: string;
  errorMessage?: string;
  mode?: "executed" | "reused";
  progressCurrent?: number;
  progressTotal?: number;
  progressPercent?: number;
  progressLabel?: string;
  progressDone?: boolean;
}

// ── WebSocket messages ────────────────────────────────────────────────────────

export interface WsRunStatusMsg {
  type: "run_status";
  status: "started" | "complete" | "cancelled" | "error";
  topological_order?: string[];
  message?: string;
}

export interface WsNodeStatusMsg {
  type: "node_status";
  node_id: string;
  status: "running" | "complete" | "error";
  mode?: "executed" | "reused";
  checkpoint_id?: string;
  message?: string;
}

export interface WsNodeProgressMsg {
  type: "node_progress";
  node_id: string;
  current: number;
  total?: number;
  percent?: number;
  label?: string;
  done?: boolean;
}

export interface WsRunResultMsg {
  type: "run_result";
  pipeline_id: string;
  topological_order: string[];
  executed_nodes: string[];
  reused_nodes: string[];
  node_results: Record<string, NodeResult>;
}

export type WsMessage =
  | WsRunStatusMsg
  | WsNodeStatusMsg
  | WsNodeProgressMsg
  | WsRunResultMsg;
