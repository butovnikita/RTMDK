export interface MemoryNode {
  id: string;
  text: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface QueryResult {
  results: MemoryNode[];
  route?: string;
  metrics?: PipelineMetrics;
}

export interface PipelineMetrics {
  embed_ms: number;
  route_ms: number;
  retrieve_ms: number;
  rerank_ms: number;
  calibrate_ms: number;
  explain_ms: number;
}

export interface RTMDKConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
}

export interface AddNodeRequest {
  text: string;
  embedding?: number[];
  metadata?: Record<string, unknown>;
}

export interface QueryRequest {
  query: string;
  top_k?: number;
  session_id?: string;
  use_pipeline?: boolean;
}
