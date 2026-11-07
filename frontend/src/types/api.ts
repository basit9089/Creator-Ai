/**
 * Shared API types.
 *
 * These mirror the backend Pydantic schemas in
 * `backend/app/schemas/ai.py`. Keeping them in one place means a
 * backend contract change is reflected here once.
 */

/** Health status of a service or provider. */
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy'

/** Response from `GET /` and `GET /health`. */
export interface HealthResponse {
  status: string
  service: string
}

/** Response from `GET /api/health/ai`. */
export interface AIHealthResponse {
  status: HealthStatus
  provider: string
  model: string
  reachable: boolean
  model_available: boolean
  detail: string
  latency_ms: number | null
}

/** Request body for `POST /api/ai/generate`. */
export interface GenerateRequest {
  prompt: string
  system_prompt?: string
  temperature?: number
  /**
   * Reasoning mode for hybrid models such as Qwen3.
   *
   * Omit it to inherit the backend's configured default, which is off.
   * Set it explicitly when a task genuinely needs deeper reasoning.
   */
  think?: boolean
  model?: string
}

/** Successful response from `POST /api/ai/generate`. */
export interface GenerateResponse {
  success: boolean
  model: string
  /** The generated text. */
  response: string
  duration_ms: number | null
  prompt_tokens: number | null
  completion_tokens: number | null
  /** Whether reasoning mode was used, after applying defaults. */
  thinking: boolean | null
}

/** A single chunk from `POST /api/ai/generate/stream`. */
export interface StreamChunk {
  chunk: string
  done: boolean
}

/** Field-level validation problem. */
export interface FieldError {
  field: string
  message: string
}

/** Error payload returned for every failure. */
export interface ErrorDetail {
  code: string
  message: string
  details?: FieldError[]
}

/** Uniform error envelope. */
export interface ErrorResponse {
  success: false
  error: ErrorDetail
}

/**
 * Availability of a registered agent.
 *
 * `not_implemented` means the name resolves but the logic does not exist
 * yet. Distinguishing this from a real agent matters: executing a
 * placeholder fails with a retryable error, not a missing-agent error.
 */
export type AgentStatus = 'available' | 'not_implemented' | 'disabled'

/** One registered agent, from `GET /api/agents`. */
export interface AgentInfo {
  name: string
  version: string
  description: string
  status: AgentStatus
  implemented: boolean
  capabilities: string[]
}

/** Response from `GET /api/agents`. */
export interface AgentListResponse {
  success: true
  total: number
  agents: AgentInfo[]
}

/**
 * Error thrown by the API client for any non-2xx response.
 *
 * Carries the HTTP status and the parsed error envelope so callers can
 * distinguish, for example, a validation problem from an unavailable
 * provider without inspecting message text.
 */
export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly details?: FieldError[]

  constructor(status: number, error: ErrorDetail) {
    super(error.message)
    this.name = 'ApiError'
    this.status = status
    this.code = error.code
    this.details = error.details
  }
}