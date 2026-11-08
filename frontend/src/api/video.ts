/**
 * Video production (`POST /api/video/produce`).
 *
 * Typed client only — no UI in this phase. The types mirror the backend
 * Pydantic schemas in `backend/app/agents/video_producer/schemas.py`
 * so a contract change fails `tsc` instead of surfacing at runtime.
 */

import { request } from './client'

/** Lifecycle of one production run, as reported by the backend. */
export type ProductionStatus =
  | 'draft'
  | 'planning'
  | 'asset_pending'
  | 'ready_to_render'
  | 'rendering'
  | 'completed'
  | 'failed'

/** What a scene needs visually (a description, not a generated asset). */
export type VisualType = 'background' | 'image' | 'video' | 'text' | 'image_sequence' | 'none'

/** Honest status of an asset reference. */
export type AssetStatus = 'planned' | 'available' | 'missing' | 'processing' | 'completed' | 'failed'

/** Metadata for a local media file, existing or not. */
export interface MediaAsset {
  asset_id: string
  filename: string
  path: string
  media_type: string
  mime_type: string
  width: number | null
  height: number | null
  duration_seconds: number | null
  size_bytes: number
  source: string
  attribution: string
  checksum: string
  status: AssetStatus
}

/** One planned scene, including the claims and sources it rests on. */
export interface Scene {
  scene_id: string
  order: number
  heading: string
  narration: string
  duration_seconds: number
  visual_type: VisualType
  visual_prompt: string
  asset_requirements: Record<string, unknown>[]
  asset_references: MediaAsset[]
  transition: string
  background: string
  claim_references: Record<string, unknown>[]
  source_references: Record<string, unknown>[]
}

/** A scene placed on the timeline with deterministic times. */
export interface TimelineScene {
  scene_id: string
  start_time: number
  end_time: number
  duration: number
  visual: Record<string, unknown>
  audio: Record<string, unknown>
  overlays: Record<string, unknown>[]
}

/** The full schedule for one render. */
export interface Timeline {
  width: number
  height: number
  fps: number
  duration_seconds: number
  scenes: TimelineScene[]
}

/** Request body for `POST /api/video/produce`. */
export interface VideoProductionRequest {
  /** The ScriptResult produced by the scriptwriter. */
  script: unknown
  project_id?: string
  output_format?: string
  resolution?: string
  fps?: number
  aspect_ratio?: string
  include_subtitles?: boolean
  output_filename?: string | null
  asset_directory?: string | null
}

/** Response from `POST /api/video/produce`. */
export interface VideoProductionResult {
  project_id: string
  status: ProductionStatus
  scenes: Scene[]
  timeline: Timeline | null
  output_path: string
  output_filename: string
  duration_seconds: number
  width: number
  height: number
  fps: number
  file_size: number
  codec: string
  audio_status: string
  asset_status: string
  warnings: string[]
  errors: string[]
  generated_at: string
}

/**
 * Plan scenes and render the video.
 *
 * @throws {ApiError} for validation failures (422) and render failures (500).
 */
export function produceVideo(
  body: VideoProductionRequest,
  signal?: AbortSignal,
): Promise<VideoProductionResult> {
  return request<VideoProductionResult>('/api/video/produce', {
    method: 'POST',
    body,
    signal,
  })
}
