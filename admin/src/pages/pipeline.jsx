import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useToast } from "@/components/ui/toast"
import { useApi } from "@/hooks/use-api"
import { Workflow, Activity, ShieldAlert, GitBranch } from "lucide-react"

export default function PipelinePage() {
  const { apiBase, authFetch } = useApi()
  const [dag, setDag] = useState(null)
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [d, h] = await Promise.all([
          authFetch(`${apiBase}/v1/memory/pipeline/dag`).then(r => r.ok ? r.json() : null),
          authFetch(`${apiBase}/v1/memory/pipeline/health`).then(r => r.ok ? r.json() : null),
        ])
        setDag(d)
        setHealth(h)
      } catch (err) {
        toast({ title: "Failed to load pipeline", description: err.message, variant: "destructive" })
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [apiBase, authFetch, toast])

  const breakerColor = (state) => {
    if (state === "open") return "destructive"
    if (state === "half_open") return "warning"
    return "success"
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">Pipeline</h2>
        <Badge variant={health?.overall === "healthy" ? "success" : health?.overall === "degraded" ? "warning" : "destructive"}>
          {health?.overall || "unknown"}
        </Badge>
      </div>

      {/* DAG Visualization */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitBranch className="h-5 w-5" /> Pipeline DAG
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </div>
          ) : dag?.nodes ? (
            <div className="flex flex-col gap-3">
              {dag.nodes.map((node, i) => (
                <div key={node.id} className="flex items-center gap-4">
                  <div className="flex-1 rounded-lg border bg-card p-3 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Workflow className="h-4 w-4 text-muted-foreground" />
                        <span className="font-medium">{node.label}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        {node.enabled ? (
                          <Badge variant="success" className="text-[10px]">enabled</Badge>
                        ) : (
                          <Badge variant="secondary" className="text-[10px]">disabled</Badge>
                        )}
                        {node.has_breaker && node.breaker_state && (
                          <Badge variant={breakerColor(node.breaker_state)} className="text-[10px]">
                            <ShieldAlert className="mr-1 h-3 w-3" />
                            {node.breaker_state}
                          </Badge>
                        )}
                        {node.has_fallback && (
                          <Badge variant="outline" className="text-[10px]">fallback</Badge>
                        )}
                      </div>
                    </div>
                  </div>
                  {i < dag.nodes.length - 1 && (
                    <div className="hidden h-8 w-px bg-border sm:block" />
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Pipeline data not available</p>
          )}
        </CardContent>
      </Card>

      {/* Stage Health */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" /> Stage Health
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-5 w-full" />
              <Skeleton className="h-5 w-full" />
            </div>
          ) : health?.stages ? (
            <div className="space-y-2">
              {health.stages.map(stage => (
                <div key={stage.name} className="flex items-center justify-between rounded-md border p-3">
                  <span className="font-medium">{stage.name}</span>
                  <div className="flex items-center gap-2">
                    <Badge variant={stage.enabled ? "success" : "secondary"}>
                      {stage.enabled ? "enabled" : "disabled"}
                    </Badge>
                    {stage.breaker_state && (
                      <Badge variant={breakerColor(stage.breaker_state)}>{stage.breaker_state}</Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Health data not available</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
