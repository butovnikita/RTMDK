import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/components/ui/toast"
import { useApi } from "@/hooks/use-api"
import { Search, Loader2, Copy, Check } from "lucide-react"

export default function QueryPage() {
  const { apiBase, authFetch } = useApi()
  const [query, setQuery] = useState("")
  const [topK, setTopK] = useState(5)
  const [threshold, setThreshold] = useState(0)
  const [simpleResults, setSimpleResults] = useState(null)
  const [pipelineResults, setPipelineResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(null)
  const { toast } = useToast()

  const handleQuery = async (endpoint) => {
    if (!query.trim()) return
    setLoading(true)
    try {
      const isPipeline = endpoint.includes("pipeline")
      const body = isPipeline
        ? JSON.stringify({ query, top_k: topK, session_id: "web-ui" })
        : JSON.stringify({ query, top_k: topK, threshold })
      const resp = await authFetch(`${apiBase}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      if (isPipeline) {
        setPipelineResults(data)
      } else {
        setSimpleResults(data)
      }
    } catch (err) {
      toast({ title: "Query failed", description: err.message, variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }

  const copyContent = (text, idx) => {
    navigator.clipboard.writeText(text)
    setCopied(idx)
    setTimeout(() => setCopied(null), 1500)
  }

  const renderResults = (data) => {
    const list = data.results || []
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>Total: {data.total ?? list.length}</span>
          {data.latency_ms && <span>• {data.latency_ms}ms</span>}
        </div>
        {list.map((r, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">score: {typeof r.score === "number" ? r.score.toFixed(4) : r.score}</Badge>
                    <span className="text-xs text-muted-foreground">{r.id || r.node_id}</span>
                  </div>
                  <p className="text-sm">
                    {typeof r.content === "object"
                      ? r.content?.content ?? JSON.stringify(r.content)
                      : String(r.content)}
                  </p>
                </div>
                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => copyContent(
                    typeof r.content === "object" ? r.content?.content ?? JSON.stringify(r.content) : String(r.content), i
                  )}
                >
                  {copied === i ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold tracking-tight">Query Memory</h2>

      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col gap-3 sm:flex-row">
            <div className="flex-1">
              <Input
                placeholder="Enter your query..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleQuery("/v1/memory/query")}
              />
            </div>
            <div className="flex gap-2">
              <Input
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={e => setTopK(Number(e.target.value))}
                className="w-20"
              />
              <Input
                type="number"
                min={0}
                max={1}
                step={0.1}
                value={threshold}
                onChange={e => setThreshold(Number(e.target.value))}
                className="w-24"
                placeholder="threshold"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="simple">
        <TabsList>
          <TabsTrigger value="simple">Simple Query</TabsTrigger>
          <TabsTrigger value="pipeline">Pipeline Query</TabsTrigger>
        </TabsList>
        <TabsContent value="simple" className="space-y-4">
          <Button onClick={() => handleQuery("/v1/memory/query")} disabled={loading || !query.trim()}>
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
            Query
          </Button>
          {simpleResults && renderResults(simpleResults)}
        </TabsContent>
        <TabsContent value="pipeline" className="space-y-4">
          <Button onClick={() => handleQuery("/v1/memory/query_pipeline")} disabled={loading || !query.trim()}>
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
            Pipeline Query
          </Button>
          {pipelineResults && renderResults(pipelineResults)}
        </TabsContent>
      </Tabs>
    </div>
  )
}
