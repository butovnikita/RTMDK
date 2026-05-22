import {
  RTMDKConfig,
  QueryResult,
  AddNodeRequest,
  QueryRequest,
  MemoryNode,
} from "./types";

export class RTMDKClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;

  constructor(config: RTMDKConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
  }

  private async request<T>(
    path: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...((options.headers as Record<string, string>) || {}),
    };
    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`RTMDK HTTP ${response.status}: ${text}`);
      }
      return (await response.json()) as T;
    } catch (err) {
      clearTimeout(timer);
      throw err;
    }
  }

  async health(): Promise<{ status: string }> {
    return this.request<{ status: string }>("/health");
  }

  async addNode(req: AddNodeRequest): Promise<{ id: string }> {
    return this.request<{ id: string }>("/v1/memory/add", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  async query(req: QueryRequest): Promise<QueryResult> {
    const endpoint = req.use_pipeline
      ? "/v1/memory/query_pipeline"
      : "/v1/memory/query";
    return this.request<QueryResult>(endpoint, {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  async retrieveNodes(
    query: string,
    topK: number = 5,
    sessionId?: string
  ): Promise<MemoryNode[]> {
    const result = await this.query({
      query,
      top_k: topK,
      session_id: sessionId,
    });
    return result.results;
  }

  async deleteNode(id: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(`/v1/memory/node/${id}`, {
      method: "DELETE",
    });
  }

  async stats(): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>("/v1/memory/stats");
  }
}
