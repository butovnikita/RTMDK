import { useState } from 'react'
import Dashboard from './components/Dashboard'
import MemoryNodes from './components/MemoryNodes'
import QueryInterface from './components/QueryInterface'
import SOTPanel from './components/SOTPanel'
import './App.css'

const API_BASE = 'http://localhost:8080'

function App() {
  const [tab, setTab] = useState('dashboard')

  const tabs = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'nodes', label: 'Memory Nodes' },
    { id: 'query', label: 'Query' },
    { id: 'sot', label: 'SOT' },
  ]

  return (
    <div className="app">
      <header className="app-header">
        <h1>RTMDK Admin Panel</h1>
        <nav className="app-nav">
          {tabs.map(t => (
            <button
              key={t.id}
              className={tab === t.id ? 'active' : ''}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="app-main">
        {tab === 'dashboard' && <Dashboard apiBase={API_BASE} />}
        {tab === 'nodes' && <MemoryNodes apiBase={API_BASE} />}
        {tab === 'query' && <QueryInterface apiBase={API_BASE} />}
        {tab === 'sot' && <SOTPanel apiBase={API_BASE} />}
      </main>
    </div>
  )
}

export default App
