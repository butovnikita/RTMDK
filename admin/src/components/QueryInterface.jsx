import { useState } from 'react'

function QueryInterface({ apiBase }) {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const resp = await fetch(`${apiBase}/v1/memory/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: topK, threshold: 0.0 })
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="query-interface">
      <h2>Query Memory</h2>
      <form onSubmit={handleSubmit} className="query-form">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Enter query..."
          required
        />
        <input
          type="number"
          value={topK}
          onChange={e => setTopK(Number(e.target.value))}
          min={1}
          max={50}
          style={{ width: '60px' }}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Querying…' : 'Query'}
        </button>
      </form>
      {error && <div className="error">{error}</div>}
      {results && (
        <div className="results">
          <h3>Results ({results.total} total, {results.latency_ms}ms)</h3>
          <ul>
            {results.results.map((r, i) => (
              <li key={i}>
                <strong>{r.node_id}</strong> — score: {r.score.toFixed(4)}<br />
                <span className="result-content">
                  {typeof r.content === 'object'
                    ? r.content?.text || JSON.stringify(r.content)
                    : String(r.content)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default QueryInterface
