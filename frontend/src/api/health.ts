/**
 * Health and AI-provider status endpoints.
 */

import { request } from './client'
import type { AIHealthResponse, HealthResponse } from '../types/api'

/** Confirm the API process is alive. */
export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/health', { signal })
}

/**
 * Check the AI provider.
 *
 * Rejects with `ApiError` (status 503) when the provider is unreachable,
 * so callers can distinguish "API down" from "provider down".
 */
export function fetchAIHealth(signal?: AbortSignal): Promise<AIHealthResponse> {
  return request<AIHealthResponse>('/api/health/ai', { signal })
}