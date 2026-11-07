/**
 * Base HTTP client for the Creator AI backend.
 *
 * Responsibilities kept here so that individual endpoint modules stay
 * declarative:
 *   - base URL resolution
 *   - JSON encoding and decoding
 *   - timeout handling
 *   - normalisation of error responses into `ApiError`
 */

import { ApiError, type ErrorDetail, type ErrorResponse } from '../types/api'

/**
 * Backend base URL.
 *
 * In development Vite serves the frontend on port 5173 and proxies or
 * calls the API on 8000. `VITE_API_BASE_URL` overrides this when the
 * backend runs elsewhere.
 */
const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'

/** Requests that exceed this are aborted. Generations can be slow. */
const DEFAULT_TIMEOUT_MS = 300_000

function resolveBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL
  return (configured || DEFAULT_BASE_URL).replace(/\/+$/, '')
}

export const apiConfig = {
  baseUrl: resolveBaseUrl(),
  timeoutMs: Number(import.meta.env.VITE_API_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS,
} as const

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'
  body?: unknown
  signal?: AbortSignal
  timeoutMs?: number
}

/**
 * Perform a JSON request against the backend.
 *
 * @throws {ApiError} for any non-2xx response.
 * @throws {Error} when the request fails at the network level.
 */
export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, timeoutMs } = options

  // Combine the caller's signal with a timeout so a hung request cannot
  // leave the UI waiting indefinitely.
  const timeoutController = new AbortController()
  const timeoutId = setTimeout(
    () => timeoutController.abort(),
    timeoutMs ?? apiConfig.timeoutMs,
  )

  if (signal) {
    if (signal.aborted) timeoutController.abort()
    else signal.addEventListener('abort', () => timeoutController.abort(), { once: true })
  }

  try {
    const response = await fetch(`${apiConfig.baseUrl}${path}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...(body ? { Accept: 'application/json' } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: timeoutController.signal,
    })

    if (!response.ok) {
      throw new ApiError(response.status, await parseError(response))
    }

    return (await response.json()) as T
  } finally {
    clearTimeout(timeoutId)
  }
}

/**
 * Convert a failed response into an `ErrorDetail`.
 *
 * Falls back to a generic message when the body is not the expected
 * envelope, for example when a proxy or gateway returns HTML.
 */
async function parseError(response: Response): Promise<ErrorDetail> {
  try {
    const payload = (await response.json()) as Partial<ErrorResponse>
    if (payload?.error?.message) {
      return payload.error
    }
  } catch {
    // Body was empty or not JSON; fall through to the default.
  }

  return {
    code: 'unexpected_error',
    message: `Request failed with status ${response.status}.`,
  }
}