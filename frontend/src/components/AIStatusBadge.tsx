/**
 * Displays the current AI provider status.
 *
 * Intentionally presentational: it takes props and has no knowledge of
 * the API. Data fetching lives in `useAIHealth`.
 */

import type { AIHealthResponse } from '../types/api'

interface AIStatusBadgeProps {
  health: AIHealthResponse | null
  isLoading?: boolean
  error?: string | null
}

const STATUS_LABELS: Record<string, string> = {
  healthy: 'Healthy',
  degraded: 'Degraded',
  unhealthy: 'Unavailable',
}

/** Render a coloured badge summarising provider availability. */
export function AIStatusBadge({ health, isLoading, error }: AIStatusBadgeProps) {
  if (isLoading && !health) {
    return <span className="status-badge status-badge--loading">Checking AI…</span>
  }

  if (error || !health) {
    return (
      <span className="status-badge status-badge--error" title={error ?? undefined}>
        {STATUS_LABELS.unhealthy}
      </span>
    )
  }

  const label = STATUS_LABELS[health.status] ?? health.status

  return (
    <span
      className={`status-badge status-badge--${health.status}`}
      title={`${health.provider} · ${health.model}${health.detail ? ` — ${health.detail}` : ''}`}
    >
      {label}
      <span className="status-badge__model">{health.model}</span>
    </span>
  )
}