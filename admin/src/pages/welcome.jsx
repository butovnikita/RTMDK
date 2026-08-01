import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/components/ui/toast"
import { useServer } from "@/context/server-context"
import { Check, ChevronRight, ChevronLeft, Loader2, Zap, Server, BrainCircuit, Settings } from "lucide-react"

const STEPS = [
  { id: "preset", label: "Preset", icon: Server },
  { id: "ai", label: "AI Provider", icon: BrainCircuit },
  { id: "embedder", label: "Embedder", icon: Zap },
  { id: "advanced", label: "Advanced", icon: Settings },
  { id: "review", label: "Review", icon: Check },
]

const PRESETS = [
  { id: "local", name: "Local (LM Studio)", description: "Single user, LM Studio on localhost, built-in embedder" },
  { id: "production", name: "Production Server", description: "Multi-user, external API, full security" },
  { id: "agent", name: "AI Agent", description: "Optimized for autonomous agents, fast queries" },
]

const AI_PROVIDERS = [
  { id: "lm_studio", name: "LM Studio", defaultUrl: "http://localhost:12345/v1", needsKey: false },
  { id: "openai", name: "OpenAI", defaultUrl: "https://api.openai.com/v1", needsKey: true },
  { id: "openrouter", name: "OpenRouter", defaultUrl: "https://openrouter.ai/api/v1", needsKey: true },
]

export default function WelcomePage() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const { startServer, saveConfig, fetchConfig, config, pollUntilReady } = useServer()
  const [step, setStep] = useState(0)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [starting, setStarting] = useState(false)
  const [waitingForReady, setWaitingForReady] = useState(false)
  const [models, setModels] = useState([])

  const [form, setForm] = useState({
    preset: "local",
    provider: "lm_studio",
    url: "http://localhost:12345/v1",
    apiKey: "",
    chatModel: "",
    embedModel: "nomic-ai/nomic-embed-text-v1.5-GGUF",
    useSOT: false,
    latentDim: 768,
    quantization: "none",
    tieredStorage: false,
    autoSave: 60,
    enableAuth: false,
    apiKeyAuth: "rtmdk-local",
    openaiKey: "",
    openrouterKey: "",
  })

  useEffect(() => {
    if (config?.preset) navigate("/")
  }, [config, navigate])

  const update = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    setModels([])
    const apiKey = form.provider === "openrouter" ? form.openrouterKey
                 : form.provider === "openai" ? form.openaiKey
                 : ""
    try {
      const resp = await fetch("/api/ai/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: form.provider, url: form.url, apiKey }),
      })
      const data = await resp.json()
      setTestResult(data)
      if (data.ok) {
        setModels(data.models || [])
        if (!form.chatModel && data.models?.length) update("chatModel", data.models[0].id)
        toast({ title: "Connection successful", variant: "success" })
      } else {
        toast({ title: "Connection failed", description: data.error, variant: "destructive" })
      }
    } catch (err) {
      toast({ title: "Connection failed", description: err.message, variant: "destructive" })
    } finally {
      setTesting(false)
    }
  }

  const handleStart = async () => {
    setStarting(true)
    const env = {
      RTMDK_PRESET: "production",
      LM_STUDIO_URL: form.url,
      RTMDK_EMBED_MODEL: form.useSOT ? "" : form.embedModel,
      RTMDK_ENABLE_LM_STUDIO: form.provider === "lm_studio" ? "true" : "false",
      RTMDK_ENABLE_API_AUTH: form.enableAuth ? "true" : "false",
      RTMDK_API_KEY: form.apiKeyAuth,
      RTMDK_AUTO_SAVE_INTERVAL: String(form.autoSave),
      RTMDK_CHAT_MODEL: form.chatModel || "",
    }
    if (form.provider === "openai" && form.openaiKey) {
      env.OPENAI_API_KEY = form.openaiKey
    } else if (form.provider === "openrouter" && form.openrouterKey) {
      env.OPENROUTER_API_KEY = form.openrouterKey
    }

    const resp = await startServer(env)
    if (resp.ok) {
      setWaitingForReady(true)
      toast({ title: "Server starting...", description: "Waiting for RTMDK to be ready", variant: "default" })
      const ready = await pollUntilReady()
      setWaitingForReady(false)
      if (ready.ok) {
        await saveConfig({ preset: form.preset, env })
        toast({ title: "Server ready", description: "RTMDK is now running", variant: "success" })
        await fetchConfig()
        navigate("/")
      } else {
        toast({ title: "Server start timed out", description: ready.error, variant: "destructive" })
      }
    } else {
      let errText = "Unknown error"
      try {
        const err = await resp.json()
        errText = err.error
      } catch { /* ignore JSON parse errors */ }
      toast({ title: "Failed to start", description: errText, variant: "destructive" })
    }
    setStarting(false)
  }

  const StepIcon = STEPS[step].icon

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-2xl space-y-6">
        <div className="text-center">
          <h1 className="text-3xl font-bold tracking-tight">Welcome to RTMDK</h1>
          <p className="mt-2 text-muted-foreground">Let's set up your memory server</p>
        </div>

        {/* Stepper */}
        <div className="flex items-center justify-between">
          {STEPS.map((s, i) => (
            <div key={s.id} className="flex flex-1 items-center">
              <div className={cn(
                "flex h-8 w-8 items-center justify-center rounded-full border-2 text-xs font-bold",
                i <= step ? "border-primary bg-primary text-primary-foreground" : "border-muted text-muted-foreground"
              )}>
                {i < step ? <Check className="h-4 w-4" /> : i + 1}
              </div>
              {i < STEPS.length - 1 && (
                <div className={cn("h-0.5 flex-1 mx-2", i < step ? "bg-primary" : "bg-muted")} />
              )}
            </div>
          ))}
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <StepIcon className="h-5 w-5" /> {STEPS[step].label}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Step 1: Preset */}
            {step === 0 && (
              <div className="grid gap-3">
                {PRESETS.map(p => (
                  <div
                    key={p.id}
                    onClick={() => update("preset", p.id)}
                    className={cn(
                      "cursor-pointer rounded-lg border p-4 transition-colors hover:bg-accent",
                      form.preset === p.id && "border-primary bg-primary/5"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{p.name}</span>
                      {form.preset === p.id && <Check className="h-4 w-4 text-primary" />}
                    </div>
                    <p className="text-sm text-muted-foreground">{p.description}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Step 2: AI Provider */}
            {step === 1 && (
              <div className="space-y-4">
                <div className="grid gap-3">
                  {AI_PROVIDERS.map(p => (
                    <div
                      key={p.id}
                      onClick={() => {
                        update("provider", p.id)
                        update("url", p.defaultUrl)
                        update("chatModel", "")
                        setModels([])
                      }}
                      className={cn(
                        "cursor-pointer rounded-lg border p-4 transition-colors hover:bg-accent",
                        form.provider === p.id && "border-primary bg-primary/5"
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{p.name}</span>
                        {form.provider === p.id && <Check className="h-4 w-4 text-primary" />}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Endpoint URL</label>
                  <Input value={form.url} onChange={e => update("url", e.target.value)} />
                </div>
                {form.provider === "openai" && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">OpenAI API Key</label>
                    <Input type="password" value={form.openaiKey} onChange={e => update("openaiKey", e.target.value)} placeholder="sk-..." />
                  </div>
                )}
                {form.provider === "openrouter" && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">OpenRouter API Key</label>
                    <Input type="password" value={form.openrouterKey} onChange={e => update("openrouterKey", e.target.value)} placeholder="sk-or-..." />
                  </div>
                )}
                <Button onClick={handleTest} disabled={testing} variant="outline">
                  {testing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
                  Test Connection
                </Button>
                {testResult && (
                  <div className={cn("rounded-md border p-3 text-sm", testResult.ok ? "border-emerald-500/50 bg-emerald-500/5" : "border-destructive/50 bg-destructive/5")}>
                    {testResult.ok ? (
                      <div>
                        <div className="font-medium text-emerald-700 dark:text-emerald-400">Connected!</div>
                        <div className="text-muted-foreground">Models found: {testResult.models?.length || 0}</div>
                      </div>
                    ) : (
                      <div className="text-destructive">{testResult.error}</div>
                    )}
                  </div>
                )}
                {models.length > 0 && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Chat Model</label>
                    <select
                      className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                      value={form.chatModel}
                      onChange={e => update("chatModel", e.target.value)}
                    >
                      {models.map(m => (
                        <option key={m.id} value={m.id}>{m.name}{m.context_length ? ` (${m.context_length} ctx)` : ""}</option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            )}

            {/* Step 3: Embedder */}
            {step === 2 && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={form.useSOT}
                    onChange={e => update("useSOT", e.target.checked)}
                    className="h-4 w-4"
                  />
                  <label className="text-sm font-medium">Use SOT Built-in Embedder (fallback)</label>
                </div>
                {!form.useSOT && (
                  <>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Embedding Model</label>
                      <Input value={form.embedModel} onChange={e => update("embedModel", e.target.value)} />
                      <p className="text-xs text-muted-foreground">Model ID from your provider (e.g. nomic-ai/nomic-embed-text-v1.5-GGUF)</p>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Latent Dimension</label>
                      <Input type="number" value={form.latentDim} onChange={e => update("latentDim", Number(e.target.value))} />
                    </div>
                  </>
                )}
                {form.useSOT && (
                  <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-400">
                    SOT embedder will be used as fallback. Dimension is auto-detected from tokenizer.
                  </div>
                )}
              </div>
            )}

            {/* Step 4: Advanced */}
            {step === 3 && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Quantization</label>
                  <div className="flex flex-wrap gap-2">
                    {["none", "fp16", "int8", "int8_global", "int8_per_dim"].map(q => (
                      <Badge
                        key={q}
                        variant={form.quantization === q ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => update("quantization", q)}
                      >
                        {q}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <input type="checkbox" checked={form.tieredStorage} onChange={e => update("tieredStorage", e.target.checked)} className="h-4 w-4" />
                  <label className="text-sm">Enable Tiered Storage v2</label>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Auto-save interval (seconds)</label>
                  <Input type="number" value={form.autoSave} onChange={e => update("autoSave", Number(e.target.value))} />
                </div>
                <div className="flex items-center gap-2">
                  <input type="checkbox" checked={form.enableAuth} onChange={e => update("enableAuth", e.target.checked)} className="h-4 w-4" />
                  <label className="text-sm">Enable API Authentication</label>
                </div>
                {form.enableAuth && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Admin API Key</label>
                    <Input value={form.apiKeyAuth} onChange={e => update("apiKeyAuth", e.target.value)} />
                  </div>
                )}
              </div>
            )}

            {/* Step 5: Review */}
            {step === 4 && (
              <div className="space-y-3">
                <div className="rounded-md border p-3">
                  <div className="text-xs uppercase text-muted-foreground">Preset</div>
                  <div className="font-medium">{PRESETS.find(p => p.id === form.preset)?.name}</div>
                </div>
                <div className="rounded-md border p-3">
                  <div className="text-xs uppercase text-muted-foreground">AI Provider</div>
                  <div className="font-medium">{AI_PROVIDERS.find(p => p.id === form.provider)?.name} — {form.url}</div>
                  {form.chatModel && (
                    <div className="text-sm text-muted-foreground">Model: {form.chatModel}</div>
                  )}
                </div>
                <div className="rounded-md border p-3">
                  <div className="text-xs uppercase text-muted-foreground">Embedder</div>
                  <div className="font-medium">
                    {form.useSOT ? "SOT Built-in (fallback)" : `${form.embedModel} (${form.latentDim}d)`}
                  </div>
                </div>
                <div className="rounded-md border p-3">
                  <div className="text-xs uppercase text-muted-foreground">Advanced</div>
                  <div className="font-medium">quantization={form.quantization}, tiered={form.tieredStorage ? "on" : "off"}, auth={form.enableAuth ? "on" : "off"}</div>
                </div>
                <Button onClick={handleStart} disabled={starting || waitingForReady} className="w-full">
                  {(starting || waitingForReady) ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
                  {waitingForReady ? "Waiting for health…" : "Start RTMDK Server"}
                </Button>
              </div>
            )}

            {/* Navigation */}
            <div className="flex justify-between pt-2">
              <Button variant="outline" onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}>
                <ChevronLeft className="mr-2 h-4 w-4" /> Back
              </Button>
              {step < STEPS.length - 1 && (
                <Button onClick={() => setStep(s => s + 1)}>
                  Next <ChevronRight className="ml-2 h-4 w-4" />
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function cn(...classes) {
  return classes.filter(Boolean).join(" ")
}
