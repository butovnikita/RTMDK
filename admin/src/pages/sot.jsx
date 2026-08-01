import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useToast } from "@/components/ui/toast"
import { useApi } from "@/hooks/use-api"
import { Cpu, Search, Play, BookOpen } from "lucide-react"

export default function SOTPage() {
  const { apiBase, authFetch } = useApi()
  const [status, setStatus] = useState(null)
  const [vocab, setVocab] = useState(null)
  const [vocabSearch, setVocabSearch] = useState("")
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  const fetchData = async () => {
    setLoading(true)
    try {
      const [s, v] = await Promise.all([
        authFetch(`${apiBase}/v1/sot/status`).then(r => r.ok ? r.json() : null),
        authFetch(`${apiBase}/v1/sot/vocab?limit=50`).then(r => r.ok ? r.json() : null),
      ])
      setStatus(s)
      setVocab(v)
    } catch (err) {
      toast({ title: "Failed to load SOT", description: err.message, variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { fetchData() }, [apiBase, authFetch, toast])

  const handleSearch = async () => {
    try {
      const resp = await authFetch(`${apiBase}/v1/sot/vocab?limit=50&search=${encodeURIComponent(vocabSearch)}`)
      if (resp.ok) setVocab(await resp.json())
    } catch { /* ignore network errors */ }
  }

  const handleBootstrap = async () => {
    try {
      const resp = await authFetch(`${apiBase}/v1/sot/bootstrap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: ["hello world", "machine learning", "neural network"] }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      toast({ title: "SOT bootstrapped", variant: "success" })
      fetchData()
    } catch (err) {
      toast({ title: "Bootstrap failed", description: err.message, variant: "destructive" })
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">SOT</h2>
        <Button onClick={handleBootstrap} disabled={!status?.enabled}>
          <Play className="mr-2 h-4 w-4" /> Bootstrap
        </Button>
      </div>

      {/* Status cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Enabled</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-12" /> : (
              <Badge variant={status?.enabled ? "success" : "secondary"}>
                {status?.enabled ? "Yes" : "No"}
              </Badge>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Vocab Size</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-16" /> : (
              <div className="text-2xl font-bold">{status?.vocab_size ?? 0}</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Max Vocab</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-16" /> : (
              <div className="text-2xl font-bold">{status?.max_vocab ?? 0}</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Mode</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-6 w-20" /> : (
              <Badge variant="outline">{status?.tokenization_mode ?? "—"}</Badge>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Vocabulary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" /> Vocabulary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex items-center gap-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search tokens..."
              value={vocabSearch}
              onChange={e => setVocabSearch(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSearch()}
              className="max-w-sm"
            />
            <Button size="sm" variant="outline" onClick={handleSearch}>Search</Button>
          </div>

          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-5 w-full" />
              <Skeleton className="h-5 w-full" />
            </div>
          ) : vocab?.items ? (
            <div className="grid gap-1 sm:grid-cols-2 lg:grid-cols-3">
              {vocab.items.map((item, i) => (
                <div key={i} className="flex items-center justify-between rounded-md border px-3 py-2">
                  <span className="font-medium">{item.word}</span>
                  <Badge variant="secondary" className="font-mono text-xs">{item.token_id}</Badge>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No vocabulary data</p>
          )}

          {vocab && (
            <div className="mt-4 text-sm text-muted-foreground">
              Total: {vocab.total} • Limit: {vocab.limit} • Offset: {vocab.offset}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
