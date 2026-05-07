import { useEffect, useState } from 'react'

function Dashboard({ apiBase }) {
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${apiBase}/health`)
      .then(r => r.json())
      .then(data => setHealth(data))
      .catch(err => setError(err.message))
  }, [apiBase])

  return (
    <div className="dashboard">
      <h2>System Health</h2>
      {error && <div className="error">Error: {error}</div>}
      {health ? (
        <div className="cards">
          <div className="card">
            <h3>Status</h3>
            <p className={health.status === 'healthy' ? 'ok' : 'warn'}>
              {health.status}
            </p>
          </div>
          <div className="card">
            <h3>Nodes</h3>
            <p>{health.nodes ?? '—'}</p>
          </div>
          <div className="card">
            <h3>Version</h3>
            <p>{health.version ?? '—'}</p>
          </div>
          <div className="card">
            <h3>Uptime</h3>
            <p>{health.uptime ? `${(health.uptime / 60).toFixed(1)} min` : '—'}</p>
          </div>
        </div>
      ) : (
        <p>Loading…</p>
      )}
    </div>
  )
}

export default Dashboard
