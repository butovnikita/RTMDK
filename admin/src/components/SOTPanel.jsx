import { useEffect, useState } from 'react'

function SOTPanel({ apiBase }) {
  const [status, setStatus] = useState(null)
  const [vocab, setVocab] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${apiBase}/v1/sot/status`)
      .then(r => r.ok ? r.json() : null)
      .then(data => setStatus(data))
      .catch(err => setError(err.message))

    fetch(`${apiBase}/v1/sot/vocab?limit=50`)
      .then(r => r.ok ? r.json() : null)
      .then(data => setVocab(data))
      .catch(() => {})
  }, [apiBase])

  return (
    <div className="sot-panel">
      <h2>Self-Organizing Tokenizer (SOT)</h2>
      {error && <div className="error">{error}</div>}
      {status ? (
        <div className="cards">
          <div className="card">
            <h3>Enabled</h3>
            <p>{status.enabled ? 'Yes' : 'No'}</p>
          </div>
          <div className="card">
            <h3>Vocab Size</h3>
            <p>{status.vocab_size}</p>
          </div>
          <div className="card">
            <h3>Max Vocab</h3>
            <p>{status.max_vocab}</p>
          </div>
          <div className="card">
            <h3>Tokenization Mode</h3>
            <p>{status.tokenization_mode ?? '—'}</p>
          </div>
          <div className="card">
            <h3>Merges</h3>
            <p>{status.merges_count}</p>
          </div>
        </div>
      ) : (
        <p>Loading SOT status…</p>
      )}

      {vocab && (
        <>
          <h3>Vocabulary ({vocab.total} tokens)</h3>
          <table className="nodes-table">
            <thead>
              <tr><th>Token ID</th><th>Word</th></tr>
            </thead>
            <tbody>
              {vocab.items.map((item, i) => (
                <tr key={i}>
                  <td>{item.token_id}</td>
                  <td>{item.word}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}

export default SOTPanel
