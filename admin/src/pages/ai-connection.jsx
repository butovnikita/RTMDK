import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/components/ui/toast"
import { useServer } from "@/context/server-context"
import { BrainCircuit, Zap, Check, Loader2, ExternalLink } from "lucide-react"

const PROVIDERS = [
  { id: "lm_studio", name: "LM Studio", defaultUrl: "http://localhost:12345/v1", needsKey: false },
  { id: "openai", name: "OpenAI", defaultUrl: "https://api.openai.com/v1", needsKey: true },
  { id: "openrouter", name: "OpenRouter", defaultUrl: "https://openrouter.ai/api/v1", needsKey: true },
]

export default function AIPage() {
  const { config, saveConfig } = useServer()
  const { toast } = useToast()
  const env = config?.env || {}

  const [provider, setProvider] = useState("openai")
  const [url, setUrl] = useState("http://localhost:12345/v1")
  const [apiKey, setApiKey] = useState("")
  const [openrouterApiKey, setOpenrouterApiKey] = useState("")
  const [chatModel, setChatModel] = useState("")
  const [embedModel, setEmbedModel] = useState("nomic-ai/nomic-embed-text-v1.5-GGUF")
  const [useSOT, setUseSOT] = useState(true)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [models, setModels] = useState([])
  const [saving, setSaving] = useState(false)
  const [embedTesting, setEmbedTesting] = useState(false)
  const [embedTestResult, setEmbedTestResult] = useState(null)
  const [embedModels, setEmbedModels] = useState([])
  const [customEmbedModel, setCustomEmbedModel] = useState(false)

  useEffect(() => {
    const e = config?.env || {}
    const p = e.RTMDK_AI_PROVIDER || (e.RTMDK_ENABLE_LM_STUDIO === "true" ? "lm_studio" : "openai")
    setProvider(p)
    // Read URL from provider-specific key, fallback to legacy LM_STUDIO_URL
    const providerDefaults = {
      lm_studio: "http://localhost:12345/v1",
      openai: "https://api.openai.com/v1",
      openrouter: "https://openrouter.ai/api/v1",
    }
    const providerUrlKeys = {
      lm_studio: "LM_STUDIO_URL",
      openai: "OPENAI_BASE_URL",
      openrouter: "OPENROUTER_BASE_URL",
    }
    const defaultUrl = providerDefaults[p] || ""
    const savedUrl = e[providerUrlKeys[p]] || e.LM_STUDIO_URL || ""
    // If saved URL looks like it belongs to a different provider, use the default
    const looksLikeOpenRouter = savedUrl.includes("openrouter")
    const looksLikeOpenAI = savedUrl.includes("openai.com")
    const looksLikeLMStudio = savedUrl.includes("localhost") || savedUrl.includes("127.0.0.1")
    let urlToUse = savedUrl || defaultUrl
    if (p === "openai" && looksLikeOpenRouter) urlToUse = defaultUrl
    if (p === "openrouter" && looksLikeOpenAI) urlToUse = defaultUrl
    if (p === "lm_studio" && (looksLikeOpenRouter || looksLikeOpenAI)) urlToUse = defaultUrl
    setUrl(urlToUse)
    setApiKey(e.OPENAI_API_KEY || "")
    setOpenrouterApiKey(e.OPENROUTER_API_KEY || "")
    setChatModel(e.RTMDK_CHAT_MODEL || "")
    setEmbedModel(e.RTMDK_EMBED_MODEL || "nomic-ai/nomic-embed-text-v1.5-GGUF")
    setUseSOT(!e.RTMDK_EMBED_MODEL)
  }, [config])

  const getProviderKey = () => {
    if (provider === "openrouter") return openrouterApiKey
    return apiKey
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    setModels([])
    setEmbedModels([])
    try {
      const [chatResp, embedResp] = await Promise.all([
        fetch("/api/ai/test", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider, url, apiKey: getProviderKey() }),
        }),
        fetch("/api/ai/embed-models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider, url, apiKey: getProviderKey() }),
        }),
      ])
      const chatData = await chatResp.json()
      const embedData = await embedResp.json()
      setTestResult(chatData)
      if (chatData.ok) {
        setModels(chatData.models || [])
        if (!chatModel && chatData.models?.length) setChatModel(chatData.models[0].id)
      }
      if (embedData.ok) {
        setEmbedModels(embedData.models || [])
        if (!embedModel && embedData.models?.length) {
          setEmbedModel(embedData.models[0].id)
          setCustomEmbedModel(false)
        }
      }
      if (chatData.ok || embedData.ok) {
        toast({ title: "Connection successful", variant: "success" })
      } else {
        toast({ title: "Connection failed", description: chatData.error || embedData.error, variant: "destructive" })
      }
    } catch (err) {
      toast({ title: "Connection failed", description: err.message, variant: "destructive" })
    } finally {
      setTesting(false)
    }
  }

  const handleTestEmbedder = async () => {
    setEmbedTesting(true)
    setEmbedTestResult(null)
    try {
      const resp = await fetch("/api/rtmdk/v1/embeddings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: "test sentence for embedding", model: embedModel || undefined }),
      })
      const data = await resp.json()
      if (resp.ok && data.data?.[0]?.embedding) {
        const dim = data.data[0].embedding.length
        setEmbedTestResult({ ok: true, dim })
        toast({ title: `Embedder OK — ${dim} dims`, variant: "success" })
      } else {
        const err = data.error || data.detail || "Unknown error"
        setEmbedTestResult({ ok: false, error: err })
        toast({ title: "Embedder test failed", description: String(err).slice(0, 100), variant: "destructive" })
      }
    } catch (err) {
      setEmbedTestResult({ ok: false, error: err.message })
      toast({ title: "Embedder test failed", description: err.message, variant: "destructive" })
    } finally {
      setEmbedTesting(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    const newEnv = {
      ...env,
      RTMDK_AI_PROVIDER: provider,
      RTMDK_ENABLE_LM_STUDIO: provider === "lm_studio" ? "true" : "false",
      RTMDK_CHAT_MODEL: chatModel || "",
      RTMDK_EMBED_MODEL: useSOT ? "" : embedModel,
    }
    // Save URL to provider-specific key
    if (provider === "lm_studio") {
      newEnv.LM_STUDIO_URL = url
    } else if (provider === "openai") {
      newEnv.OPENAI_BASE_URL = url
    } else if (provider === "openrouter") {
      newEnv.OPENROUTER_BASE_URL = url
    }
    // Clear inactive provider keys to avoid leaking stale credentials
    if (provider === "openai") {
      newEnv.OPENAI_API_KEY = apiKey
      delete newEnv.OPENROUTER_API_KEY
    } else if (provider === "openrouter") {
      newEnv.OPENROUTER_API_KEY = openrouterApiKey
      delete newEnv.OPENAI_API_KEY
    } else {
      delete newEnv.OPENAI_API_KEY
      delete newEnv.OPENROUTER_API_KEY
    }
    await saveConfig({ ...config, env: newEnv })
    toast({ title: "Settings saved", variant: "success" })
    setSaving(false)
  }

  const savedUrl = (() => {
    if (provider === "openai") return env.OPENAI_BASE_URL || env.LM_STUDIO_URL || ""
    if (provider === "openrouter") return env.OPENROUTER_BASE_URL || env.LM_STUDIO_URL || ""
    return env.LM_STUDIO_URL || ""
  })()
  const hasChanges = (
    url !== savedUrl ||
    chatModel !== (env.RTMDK_CHAT_MODEL || "") ||
    embedModel !== (env.RTMDK_EMBED_MODEL || "") ||
    useSOT !== !env.RTMDK_EMBED_MODEL ||
    provider !== (env.RTMDK_AI_PROVIDER || (env.RTMDK_ENABLE_LM_STUDIO === "true" ? "lm_studio" : "openai")) ||
    (provider === "openai" && apiKey !== (env.OPENAI_API_KEY || "")) ||
    (provider === "openrouter" && openrouterApiKey !== (env.OPENROUTER_API_KEY || ""))
  )

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold tracking-tight">AI Connection</h2>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Provider config */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BrainCircuit className="h-5 w-5" /> Provider
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-2">
              {PROVIDERS.map(p => (
                <div
                  key={p.id}
                  onClick={() => {
                    setProvider(p.id)
                    setUrl(p.defaultUrl)
                    setChatModel("")
                    setModels([])
                  }}
                  className={cn(
                    "cursor-pointer rounded-lg border p-4 transition-colors hover:bg-accent",
                    provider === p.id && "border-primary bg-primary/5"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{p.name}</span>
                    {provider === p.id && <Check className="h-4 w-4 text-primary" />}
                  </div>
                </div>
              ))}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Endpoint URL</label>
              <Input value={url} onChange={e => setUrl(e.target.value)} />
            </div>

            {provider === "openai" && (
              <div className="space-y-2">
                <label className="text-sm font-medium">OpenAI API Key</label>
                <Input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="sk-..." />
              </div>
            )}
            {provider === "openrouter" && (
              <div className="space-y-2">
                <label className="text-sm font-medium">OpenRouter API Key</label>
                <Input type="password" value={openrouterApiKey} onChange={e => setOpenrouterApiKey(e.target.value)} placeholder="sk-or-..." />
              </div>
            )}

            <div className="flex gap-2">
              <Button onClick={handleTest} disabled={testing} variant="outline">
                {testing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
                Test Connection
              </Button>
              <Button onClick={handleSave} disabled={saving} className="relative">
                {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : "Save"}
                {hasChanges && !saving && (
                  <span className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-amber-500 ring-2 ring-background" />
                )}
              </Button>
            </div>
            {hasChanges && (
              <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-400">
                Restart the RTMDK server to apply these changes.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Test result */}
        <Card>
          <CardHeader>
            <CardTitle>Connection Status</CardTitle>
          </CardHeader>
          <CardContent>
            {!testResult ? (
              <div className="flex h-40 items-center justify-center text-muted-foreground">
                Click "Test Connection" to verify your AI provider
              </div>
            ) : testResult.ok ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-emerald-500" />
                  <span className="font-medium text-emerald-700 dark:text-emerald-400">Connected successfully</span>
                </div>
                <div>
                  <div className="mb-1 text-sm font-medium">Available Models</div>
                  <div className="flex flex-wrap gap-2">
                    {testResult.models?.slice(0, 10).map(m => (
                      <Badge key={m.id} variant="secondary" className="font-mono text-xs">{m.name}</Badge>
                    )) || <span className="text-sm text-muted-foreground">No models listed</span>}
                  </div>
                </div>
                {testResult.models?.length > 10 && (
                  <div className="text-xs text-muted-foreground">...and {testResult.models.length - 10} more</div>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-destructive" />
                  <span className="font-medium text-destructive">Connection failed</span>
                </div>
                <div className="rounded-md border border-destructive/50 bg-destructive/5 p-3 text-sm text-destructive">
                  {testResult.error}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Model selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BrainCircuit className="h-5 w-5" /> Chat Model
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {models.length > 0 ? (
            <div className="space-y-2">
              <label className="text-sm font-medium">Selected Model</label>
              <select
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                value={chatModel}
                onChange={e => setChatModel(e.target.value)}
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.name}{m.context_length ? ` (${m.context_length} ctx)` : ""}</option>
                ))}
              </select>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">Test connection to load available models.</div>
          )}
        </CardContent>
      </Card>

      {/* Embedder */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" /> Embedder
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-2">
            <input type="checkbox" checked={useSOT} onChange={e => setUseSOT(e.target.checked)} className="h-4 w-4" />
            <label className="text-sm font-medium">Use SOT Built-in Embedder (fallback)</label>
          </div>
          {!useSOT && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Embedding Model</label>
              {embedModels.length > 0 && !customEmbedModel ? (
                <select
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                  value={embedModel}
                  onChange={e => setEmbedModel(e.target.value)}
                >
                  {embedModels.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
              ) : (
                <Input value={embedModel} onChange={e => setEmbedModel(e.target.value)} placeholder="Custom model ID..." />
              )}
              {embedModels.length > 0 && (
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <input type="checkbox" checked={customEmbedModel} onChange={e => setCustomEmbedModel(e.target.checked)} />
                  Use custom model ID
                </label>
              )}
              {embedModels.length === 0 && (
                <p className="text-xs text-muted-foreground">No embedding models discovered. Type a model ID manually or click Test Connection to reload.</p>
              )}
              <Button onClick={handleTestEmbedder} disabled={embedTesting} variant="outline" size="sm">
                {embedTesting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
                Test Embedder
              </Button>
              {embedTestResult && (
                <div className="text-sm">
                  {embedTestResult.ok ? (
                    <span className="text-emerald-600 dark:text-emerald-400">✓ {embedTestResult.dim} dimensions</span>
                  ) : (
                    <span className="text-destructive">✗ {embedTestResult.error}</span>
                  )}
                </div>
              )}
            </div>
          )}
          {useSOT && (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-400">
              SOT embedder will be used. Dimension is auto-detected from tokenizer.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick links */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Links</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <a href="https://lmstudio.ai" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 rounded-md border px-3 py-2 text-sm hover:bg-accent">
              LM Studio <ExternalLink className="h-3 w-3" />
            </a>
            <a href="https://platform.openai.com" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 rounded-md border px-3 py-2 text-sm hover:bg-accent">
              OpenAI Platform <ExternalLink className="h-3 w-3" />
            </a>
            <a href="https://openrouter.ai" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 rounded-md border px-3 py-2 text-sm hover:bg-accent">
              OpenRouter <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function cn(...classes) {
  return classes.filter(Boolean).join(" ")
}
