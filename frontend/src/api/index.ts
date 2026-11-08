/**
 * Public surface of the API layer.
 *
 * Feature code should import from here rather than reaching into
 * individual modules.
 */

export { apiConfig, request } from './client'
export type { RequestOptions } from './client'

export { generateCompletion, streamCompletion } from './ai'
export { fetchAgents } from './agents'
export { fetchAIHealth, fetchHealth } from './health'
export { produceVideo } from './video'
export type {
  AssetStatus,
  MediaAsset,
  ProductionStatus,
  Scene,
  Timeline,
  TimelineScene,
  VideoProductionRequest,
  VideoProductionResult,
  VisualType,
} from './video'