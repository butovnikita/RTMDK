import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useApi } from "@/hooks/use-api"
import { Activity, Database, Zap, Server, RefreshCw } from "lucide-react"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts"

const mockHistory = [
  { time: "10:00", latency: 12 },
  { time: "10:05", latency: 18 },
  { time: "10:10", latency: 9 },
  { time: "10:15", latency: 22 },
  { time: "10:20", latency: 15 },
  { time: "10:25", latency: 11 },
  { time: "10:30", latency: 14 },
]

export default function DashboardPage() {
  const { apiBase, authFetch } = useApi()
  const [health, setHealth] = useState(null)
  const [deepHealth, setDeepHealth] = useState(null)
  const [loading, setLoading] = useState(true)

  const fetchData = async () => {
    setLoading(true)
    try {
      const [h, d] = await Promise.all([
        authFetch(`${apiBase}/health`).then(r => r.ok ? r.json() : null),
        authFetch(`${apiBase}/health/deep`).then(r => r.ok ? r.json() : null),
      ])
      setHealth(h)
      setDeepHealth(d)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, 30000)
    return () => clearInterval(id)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase])

  const statusBadge = (status) => {
    if (status === "ok" || status === "healthy") return <Badge variant="success">{status}</Badge>
    if (status === "degraded") return <Badge variant="warning">{status}</Badge>
    return <Badge variant="destructive">{status || "unknown"}</Badge>
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">Dashboard</h2>
        <button onClick={fetchData} className="inline-flex h-9 items-center gap-2 rounded-md border border-input bg-background px-3 text-sm font-medium hover:bg-accent">
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {/* Health cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-20" /> : statusBadge(health?.status)}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Memory Nodes</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-16" /> : (
              <div className="text-2xl font-bold">{health?.memory_nodes ?? 0}</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Query Cache</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-16" /> : (
              <div className="text-2xl font-bold">{Math.round((health?.query_cache?.hit_rate ?? 0) * 100)}%</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">LM Studio</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-16" /> : (
              <Badge variant={health?.lm_studio ? "success" : "secondary"}>
                {health?.lm_studio ? "Connected" : "Offline"}
              </Badge>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Query Latency</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] min-h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="latency" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Deep Health Checks</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-2">
                <Skeleton className="h-5 w-full" />
                <Skeleton className="h-5 w-full" />
                <Skeleton className="h-5 w-full" />
              </div>
            ) : deepHealth ? (
              <div className="space-y-2">
                {Object.entries(deepHealth.checks || {}).map(([key, val]) => (
                  <div key={key} className="flex items-center justify-between rounded-md border p-2">
                    <span className="text-sm font-medium capitalize">{key.replace(/_/g, " ")}</span>
                    {typeof val === "object" && val.status ? statusBadge(val.status) : (
                      <span className="text-sm text-muted-foreground">{JSON.stringify(val).slice(0, 40)}</span>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No deep health data</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
