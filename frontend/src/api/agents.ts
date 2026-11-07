/**
 * Agent discovery.
 *
 * Read-only for now. The backend registers the planned agents as
 * placeholders, so this reports which names resolve and which are not
 * implemented yet. Executing an agent arrives in a later phase, with the
 * agents themselves.
 */

import { request } from './client'
import type { AgentListResponse } from '../types/api'

/** List every registered agent with its version and availability. */
export function fetchAgents(signal?: AbortSignal): Promise<AgentListResponse> {
  return request<AgentListResponse>('/api/agents', { signal })
}
