import { useEffect, useState } from 'react'

function MemoryNodes({ apiBase }) {
  const [nodes, setNodes] = useState([])
  const [error, setError] = useState(null)
  const [page, setPage] = useState(0)
  const [pageSize] = useState(20)

  useEffect(() => {
    fetch(`${apiBase}/v1/memory/nodes?limit=${pageSize}&offset=${page * pageSize}`)
      .then(r => {
        if (!r.ok) throw new Error(r.statusText)
        return r.json()
      })
      .then(data => setNodes(data.nodes || []))
      .catch(err => setError(err.message))
  }, [apiBase, page, pageSize])

  return (
    <div className="memory-nodes">
      <h2>Memory Nodes</h2>
      {error && <div className="error">{error}</div>}
      <div className="pagination">
        <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}>
          Prev
        </button>
        <span>Page {page + 1}</span>
        <button onClick={() => setPage(p => p + 1)} disabled={nodes.length < pageSize}>
          Next
        </button>
      </div>
      <table className="nodes-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Content</th>
            <th>Salience</th>
            <th>Phase</th>
            <th>Amplitude</th>
          </tr>
        </thead>
        <tbody>
          {nodes.map(n => (
            <tr key={n.id}>
              <td>{n.id}</td>
              <td>
                {typeof n.content === 'object'
                  ? n.content?.text || JSON.stringify(n.content)
                  : String(n.content)}
              </td>
              <td>{n.salience?.toFixed?.(3) ?? '—'}</td>
              <td>{n.phase?.toFixed?.(3) ?? '—'}</td>
              <td>{n.amplitude?.toFixed?.(3) ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default MemoryNodes
