import { AIStatusBadge } from './components'
import { useAIHealth } from './hooks'
import './App.css'

/**
 * Temporary root view.
 *
 * Phase 2 establishes the backend foundation and a frontend skeleton.
 * The dashboard is intentionally not built yet; this view confirms the
 * frontend is talking to the backend by showing live provider health.
 */
function App() {
  const { health, isLoading, error } = useAIHealth()

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>Creator AI</h1>
        <AIStatusBadge health={health} isLoading={isLoading} error={error} />
      </header>

      <section className="app-placeholder">
        <h2>Foundation ready</h2>
        <p>
          The API foundation is in place: typed configuration, a provider-agnostic AI
          interface, structured error handling and a test suite.
        </p>
        <p className="app-placeholder__note">
          The dashboard and agent features arrive in a later phase. The badge above is
          live, reading <code>GET /api/health/ai</code>.
        </p>
      </section>
    </main>
  )
}

export default App