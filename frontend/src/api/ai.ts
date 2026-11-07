/**
 * AI generation endpoints.
 */

import { apiConfig, request } from './client'
import type { GenerateRequest, GenerateResponse, StreamChunk } from '../types/api'

/** Generate a completion and wait for the full response. */
export function generateCompletion(
  payload: GenerateRequest,
  signal?: AbortSignal,
): Promise<GenerateResponse> {
  return request<GenerateResponse>('/api/ai/generate', {
    method: 'POST',
    body: payload,
    signal,
  })
}

/**
 * Stream a completion, invoking `onChunk` as text arrives.
 *
 * The backend responds with newline-delimited JSON. This uses `fetch`
 * with a reader rather than the `EventSource` API, because EventSource
 * only supports GET requests and this endpoint requires POST.
 *
 * @throws {ApiError} if the request fails before streaming begins.
 */
export async function streamCompletion(
  payload: GenerateRequest,
  onChunk: (text: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${apiConfig.baseUrl}/api/ai/generate/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })

  if (!response.ok || !response.body) {
    // The stream endpoint reports failures mid-body, so an error here
    // means the request never started streaming.
    throw new Error(`Streaming request failed with status ${response.status}.`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Process every complete line, keeping any partial remainder buffered.
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue

      try {
        const chunk = JSON.parse(trimmed) as StreamChunk
        if (chunk.chunk) onChunk(chunk.chunk)
      } catch {
        // Ignore malformed lines rather than aborting the stream.
      }
    }
  }
}