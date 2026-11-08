/**
 * Captions + thumbnail + SEO metadata (`POST /api/captions-thumbnail`).
 *
 * Typed client only — no UI in this phase. The types mirror the backend
 * Pydantic schemas in `backend/app/agents/captions_thumbnail/schemas.py`
 * so a contract change fails `tsc` instead of surfacing at runtime.
 *
 * HTTP contract: `completed` and `partial` both arrive as HTTP 200 —
 * check `success`/`status` in the payload. A `partial` response carries
 * the real paths of what *was* produced plus per-artifact `error_code`s
 * for what failed. Total failure throws (4xx/5xx error envelope).
 */

import { request } from './client'

/** Honest status of one artifact within the run. */
export type ArtifactStatus = 'completed' | 'failed'

/** Overall outcome: everything verified, something failed, or nothing. */
export type CaptionsThumbnailStatus = 'completed' | 'partial' | 'failed'

/** Caption files that exist on disk, or an honest failure. */
export interface CaptionsInfo {
  status: ArtifactStatus
  srt_path: string
  vtt_path: string
  formats: string[]
  caption_count: number
  duration_seconds: number
  /** `video_timeline` | `video_scene_durations` | `script_wpm`. */
  timing_source: string
  /** Bytes per written file, keyed by format. */
  size_bytes: Record<string, number>
  error_code: string
}

/** A real PNG and its measurable properties, or an honest failure. */
export interface ThumbnailInfo {
  status: ArtifactStatus
  asset_id: string
  path: string
  filename: string
  mime_type: string
  width: number
  height: number
  size_bytes: number
  /** Which renderer produced it (`local` — no AI image generation). */
  renderer: string
  error_code: string
}

/** YouTube title/description/tags/SEO built from the script only. */
export interface MetadataInfo {
  status: ArtifactStatus
  title: string
  alternate_titles: string[]
  description: string
  tags: string[]
  primary_keyword: string
  secondary_keywords: string[]
  search_phrases: string[]
  /** Local heuristic in [0, 100] — not an external ranking system. */
  title_score: number | null
  description_keywords: string[]
  content_category: string
  language: string
  audience: string
  /** Where the metadata JSON was persisted ("" if not written). */
  path: string
  error_code: string
}

/** Traceability: a claim, its sources, and the sections using it. */
export interface ClaimRef {
  claim_id: string
  source_ids: string[]
  sections: string[]
}

/** A source exactly as the script carries it. */
export interface SourceRef {
  id: string
  url: string
  domain: string
}

/** Links metadata back through script → claim → source. */
export interface GroundingInfo {
  claim_refs: ClaimRef[]
  source_refs: SourceRef[]
}

/** Input for `POST /api/captions-thumbnail`. */
export interface CaptionsThumbnailRequest {
  /** ScriptResult produced by the scriptwriter. */
  script: unknown
  /** Optional VideoProductionResult — its scene timing drives captions. */
  video?: unknown
  project_id?: string
  target_audience?: string
  language?: string
  caption_formats?: string[]
  thumbnail_resolution?: string
  /** Existing image under the media root (relative path). */
  thumbnail_image?: string
  watermark?: string
}

/** Everything produced, honestly reported. */
export interface CaptionsThumbnailResult {
  project_id: string
  status: CaptionsThumbnailStatus
  success: boolean
  captions: CaptionsInfo
  thumbnail: ThumbnailInfo
  metadata: MetadataInfo
  grounding: GroundingInfo
  warnings: string[]
  errors: string[]
  generated_at: string
}

/**
 * Generate captions, a thumbnail and SEO metadata for a script.
 *
 * Resolves for both `completed` and `partial` (HTTP 200); always check
 * `result.success` before treating every artifact as present. Rejects
 * with the standard error envelope on validation errors (422), AI
 * provider failures (502/503/504) or total failure (500).
 */
export function produceCaptionsThumbnail(
  body: CaptionsThumbnailRequest,
): Promise<CaptionsThumbnailResult> {
  return request<CaptionsThumbnailResult>('/api/captions-thumbnail', {
    method: 'POST',
    body,
  })
}
