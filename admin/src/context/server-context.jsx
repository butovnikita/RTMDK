import { createContext, useContext, useState, useEffect, useCallback, useRef } from "react"

const ServerContext = createContext(null)
const POLL_INTERVAL = 2000
const MAX_STARTUP_WAIT = 60000

export function ServerProvider({ children }) {
  const [status, setStatus] = useState({ running: false, pid: null, port: 8081, uptime: 0 })
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const startupPollRef = useRef(null)

  const fetchStatus = useCallback(async () => {
    try {
      const resp = await fetch("/api/server/status")
      if (resp.ok) {
        const proxyStatus = await resp.json()
        // If proxy says not running, double-check with actual health endpoint
        // in case the server was started externally.
        if (!proxyStatus.running) {
          try {
            const health = await fetch("/api/rtmdk/health", { cache: "no-store" })
            if (health.ok) {
              const h = await health.json()
              setStatus({
                running: true,
                pid: proxyStatus.pid,
                port: proxyStatus.port,
                uptime: h.uptime_ms || 0,
                external: true,
              })
              return
            }
          } catch { /* fall through to proxy status */ }
        }
        setStatus(proxyStatus)
      }
    } catch {
      setStatus(s => ({ ...s, running: false }))
    }
  }, [])

  const fetchConfig = useCallback(async () => {
    try {
      const resp = await fetch("/api/config")
      if (resp.ok) setConfig(await resp.json())
    } catch {
      setConfig(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    fetchConfig()
    const id = setInterval(fetchStatus, 5000)
    return () => clearInterval(id)
  }, [fetchStatus, fetchConfig])

  const startServer = async (env = {}) => {
    const resp = await fetch("/api/server/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ env }),
    })
    await fetchStatus()
    return resp
  }

  const pollUntilReady = useCallback(async () => {
    return new Promise((resolve) => {
      const start = Date.now()
      const check = async () => {
        try {
          const resp = await fetch("/api/server/status")
          if (resp.ok) {
            const s = await resp.json()
            setStatus(s)
            if (s.running) {
              // Also ping RTMDK health endpoint
              try {
                const health = await fetch(`/api/rtmdk/health`, { cache: "no-store" })
                if (health.ok) {
                  resolve({ ok: true, status: s })
                  return
                }
              } catch { /* ignore health check errors */ }
            }
          }
        } catch { /* ignore status fetch errors */ }
        if (Date.now() - start > MAX_STARTUP_WAIT) {
          resolve({ ok: false, error: "Timeout waiting for RTMDK to become ready" })
          return
        }
        startupPollRef.current = setTimeout(check, POLL_INTERVAL)
      }
      check()
    })
  }, [])

  const stopServer = async () => {
    if (startupPollRef.current) clearTimeout(startupPollRef.current)
    const resp = await fetch("/api/server/stop", { method: "POST" })
    await fetchStatus()
    return resp
  }

  const saveConfig = async (cfg) => {
    const payload = { ...cfg }
    // Client-side guard: preserve preset if context already has one
    if (!payload.preset && config?.preset) {
      payload.preset = config.preset
    }
    await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
    await fetchConfig()
  }

  return (
    <ServerContext.Provider value={{ status, config, loading, fetchStatus, startServer, stopServer, saveConfig, fetchConfig, pollUntilReady }}>
      {children}
    </ServerContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useServer() {
  const ctx = useContext(ServerContext)
  if (!ctx) throw new Error("useServer must be used within ServerProvider")
  return ctx
}
