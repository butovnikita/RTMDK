import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
// Badge removed — unused
import { useToast } from "@/components/ui/toast"
import { useApi } from "@/hooks/use-api"
import { BarChart3, TrendingUp, Clock, Database } from "lucide-react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from "recharts"

const COLORS = ["hsl(var(--chart-1))", "hsl(var(--chart-2))", "hsl(var(--chart-3))", "hsl(var(--chart-4))", "hsl(var(--chart-5))"]

export default function AnalyticsPage() {
  const { apiBase, authFetch } = useApi()
  const [overview, setOverview] = useState(null)
  const [pipeline, setPipeline] = useState(null)
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [o, p] = await Promise.all([
          authFetch(`${apiBase}/v1/analytics/overview`).then(r => r.ok ? r.json() : null),
          authFetch(`${apiBase}/v1/analytics/pipeline`).then(r => r.ok ? r.json() : null),
        ])
        setOverview(o)
        setPipeline(p)
      } catch (err) {
        toast({ title: "Failed to load analytics", description: err.message, variant: "destructive" })
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [apiBase, authFetch, toast])

  const mockQueries = [
    { name: "00:00", count: 12 },
    { name: "04:00", count: 8 },
    { name: "08:00", count: 45 },
    { name: "12:00", count: 62 },
    { name: "16:00", count: 38 },
    { name: "20:00", count: 25 },
  ]

  const mockCache = [
    { name: "Hit", value: 68 },
    { name: "Miss", value: 32 },
  ]

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold tracking-tight">Analytics</h2>

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Queries</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-16" /> : (
              <div className="text-2xl font-bold">{overview?.total_queries ?? 0}</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-16" /> : (
              <div className="text-2xl font-bold">{overview?.avg_latency_ms ?? 0}ms</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Cache Hit Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-16" /> : (
              <div className="text-2xl font-bold">{Math.round((overview?.cache_hit_rate ?? 0) * 100)}%</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Memory Nodes</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-16" /> : (
              <div className="text-2xl font-bold">{overview?.memory_nodes ?? 0}</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Query Volume</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={mockQueries}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="hsl(var(--chart-1))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Cache Hit / Miss</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={mockCache} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={5} dataKey="value">
                    {mockCache.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-4">
              {mockCache.map((entry, index) => (
                <div key={entry.name} className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full" style={{ background: COLORS[index] }} />
                  <span className="text-sm">{entry.name} ({entry.value}%)</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Pipeline Metrics */}
      {pipeline && (
        <Card>
          <CardHeader>
            <CardTitle>Pipeline Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(pipeline).map(([key, val]) => (
                <div key={key} className="rounded-md border p-3">
                  <div className="text-xs text-muted-foreground uppercase">{key}</div>
                  <div className="mt-1 text-lg font-semibold">
                    {typeof val === "number" ? val.toFixed(2) : String(val)}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
