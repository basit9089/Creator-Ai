/// <reference types="vite/client" />

/**
 * Typed access to the frontend environment variables declared in
 * `.env.example`.
 *
 * Only `VITE_`-prefixed variables are available; Vite inlines them into
 * the bundle at build time, so they must never hold secrets.
 */
interface ImportMetaEnv {
  /** Base URL of the backend API. */
  readonly VITE_API_BASE_URL?: string
  /** Request timeout in milliseconds. */
  readonly VITE_API_TIMEOUT_MS?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}