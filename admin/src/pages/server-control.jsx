import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/components/ui/toast"
import { useServer } from "@/context/server-context"
import { Play, Square, RotateCcw, Server, Clock, Hash, Terminal, Loader2 } from "lucide-react"

export default function ServerPage() {
  const { status, startServer, stopServer, fetchStatus, pollUntilReady } = useServer()
  const { toast } = useToast()
  const [logs, setLogs] = useState([])
  const [actionLoading, setActionLoading] = useState(false)
  const logEndRef = useRef(null)
  const esRef = useRef(null)

  useEffect(() => {
    let mounted = true
    const connect = () => {
      if (!mounted) return
      if (esRef.current) {
        esRef.current.close()
        esRef.current = null
      }
      const es = new EventSource("/api/server/logs")
      esRef.current = es
      es.onmessage = (e) => {
        if (!mounted) return
        try {
          const data = JSON.parse(e.data)
          setLogs(prev => [...prev.slice(-200), data.line])
        } catch {
          // skip malformed SSE lines
        }
      }
      es.onerror = () => {
        es.close()
        if (mounted) {
          setTimeout(connect, 3000)
        }
      }
    }
    connect()
    return () => {
      mounted = false
      if (esRef.current) {
        esRef.current.close()
        esRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  const [waitingForReady, setWaitingForReady] = useState(false)

  const handleStart = async () => {
    setActionLoading(true)
    const resp = await startServer()
    if (!resp.ok) {
      let errText = "Unknown error"
      try {
        const err = await resp.json()
        errText = err.error
      } catch { /* ignore JSON parse errors */ }
      toast({ title: "Failed", description: errText, variant: "destructive" })
    } else {
      setWaitingForReady(true)
      toast({ title: "Server starting...", description: "Waiting for RTMDK to be ready", variant: "default" })
      const ready = await pollUntilReady()
      setWaitingForReady(false)
      if (ready.ok) {
        toast({ title: "Server ready", variant: "success" })
      } else {
        toast({ title: "Server start timed out", description: ready.error, variant: "destructive" })
      }
    }
    setActionLoading(false)
  }

  const handleStop = async () => {
    setActionLoading(true)
    await stopServer()
    toast({ title: "Server stopped", variant: "success" })
    setActionLoading(false)
  }

  const [portBusy, setPortBusy] = useState(false)

  const checkPort = async () => {
    try {
      const resp = await fetch("/api/server/free-port", { method: "POST" })
      const data = await resp.json()
      if (data.ok) {
        toast({ title: "Port freed", description: data.message, variant: "success" })
        setPortBusy(false)
      } else {
        toast({ title: "Failed", description: data.error, variant: "destructive" })
      }
    } catch (err) {
      toast({ title: "Failed", description: err.message, variant: "destructive" })
    }
  }

  const handleRestart = async () => {
    setActionLoading(true)
    await stopServer()
    setLogs([])
    await new Promise(r => setTimeout(r, 2000))
    const resp = await startServer()
    if (!resp.ok) {
      let errText = "Unknown error"
      try {
        const err = await resp.json()
        errText = err.error
      } catch { /* ignore JSON parse errors */ }
      toast({ title: "Restart failed", description: errText, variant: "destructive" })
      if (errText.includes("already in use")) setPortBusy(true)
    } else {
      setWaitingForReady(true)
      toast({ title: "Server restarting...", description: "Waiting for RTMDK to be ready", variant: "default" })
      const ready = await pollUntilReady()
      setWaitingForReady(false)
      if (ready.ok) {
        toast({ title: "Server restarted", variant: "success" })
        setPortBusy(false)
      } else {
        toast({ title: "Restart timed out", description: ready.error, variant: "destructive" })
      }
    }
    setActionLoading(false)
  }

  const formatUptime = (ms) => {
    if (!ms) return "—"
    const s = Math.floor(ms / 1000)
    const m = Math.floor(s / 60)
    const h = Math.floor(m / 60)
    return `${h}h ${m % 60}m ${s % 60}s`
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold tracking-tight">Server Control</h2>

      {/* Status cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Status</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <Badge variant={status.running ? "success" : "secondary"}>
              {status.running ? "Running" : "Stopped"}
            </Badge>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">PID</CardTitle>
            <Hash className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{status.pid ?? "—"}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Port</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{status.port}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Uptime</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatUptime(status.uptime)}</div>
          </CardContent>
        </Card>
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-2">
        {!status.running ? (
          <Button onClick={handleStart} disabled={actionLoading}>
            {actionLoading || waitingForReady ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            {waitingForReady ? "Waiting for health…" : "Start Server"}
          </Button>
        ) : (
          <>
            <Button variant="destructive" onClick={handleStop} disabled={actionLoading || waitingForReady}>
              {actionLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Square className="mr-2 h-4 w-4" />}
              Stop Server
            </Button>
            <Button variant="outline" onClick={handleRestart} disabled={actionLoading || waitingForReady}>
              {waitingForReady ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RotateCcw className="mr-2 h-4 w-4" />}
              {waitingForReady ? "Waiting for health…" : "Restart"}
            </Button>
          </>
        )}
        <Button variant="ghost" onClick={fetchStatus}>
          Refresh Status
        </Button>
        {portBusy && !status.running && (
          <Button variant="destructive" onClick={checkPort} disabled={actionLoading}>
            Free Port
          </Button>
        )}
      </div>

      {/* Logs */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Terminal className="h-5 w-5" /> Server Logs
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] overflow-auto rounded-md bg-black p-3 font-mono text-xs text-green-400">
            {logs.length === 0 ? (
              <span className="text-muted-foreground">No logs yet...</span>
            ) : (
              logs.map((line, i) => (
                <div key={i} className="break-all py-0.5">
                  {line}
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
