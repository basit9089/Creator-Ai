/**
 * Hook for polling AI provider health.
 *
 * Kept deliberately small: no state-management library is used yet,
 * because React's built-in hooks cover this case on their own.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchAIHealth } from '../api/health'
import type { AIHealthResponse } from '../types/api'

export interface UseAIHealthResult {
  health: AIHealthResponse | null
  isLoading: boolean
  error: string | null
  refresh: () => void
}

/**
 * Fetch AI provider health on mount and on an interval.
 *
 * A 503 response means the provider is unreachable, but the body still
 * describes the state, so the error is treated as data rather than a
 * hard failure.
 */
export function useAIHealth(intervalMs = 30_000): UseAIHealthResult {
  const [health, setHealth] = useState<AIHealthResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)
  const controllerRef = useRef<AbortController | null>(null)

  const refresh = useCallback(() => setNonce((n) => n + 1), [])

  useEffect(() => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    let cancelled = false

    async function load() {
      setIsLoading(true)
      try {
        const result = await fetchAIHealth(controller.signal)
        if (cancelled) return
        setHealth(result)
        setError(null)
      } catch (err) {
        if (cancelled || controller.signal.aborted) return
        setHealth(null)
        setError(err instanceof Error ? err.message : 'Failed to check AI health.')
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    void load()

    const timer = setInterval(load, intervalMs)
    return () => {
      cancelled = true
      controller.abort()
      clearInterval(timer)
    }
  }, [intervalMs, nonce])

  return { health, isLoading, error, refresh }
}