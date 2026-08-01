import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/components/ui/toast"
import { useApi } from "@/hooks/use-api"
import { useServer } from "@/context/server-context"
import { Key, Webhook, RefreshCw, Trash2, Plus } from "lucide-react"

const CONFIG_WHITELIST = [
  { key: "decay_rate", label: "Decay Rate", type: "number", step: 0.001 },
  { key: "top_k", label: "Top K", type: "number", step: 1 },
  { key: "min_response", label: "Min Response", type: "number", step: 0.1 },
  { key: "bandwidth", label: "Bandwidth", type: "number", step: 0.1 },
  { key: "phase_coupling", label: "Phase Coupling", type: "number", step: 0.1 },
  { key: "tension_threshold", label: "Tension Threshold", type: "number", step: 0.1 },
  { key: "adaptive_threshold", label: "Adaptive Threshold", type: "number", step: 0.1 },
  { key: "chat_model", label: "Chat Model", type: "text" },
  { key: "embed_model", label: "Embed Model", type: "text" },
]

export default function SettingsPage() {
  const { apiBase, authFetch } = useApi()
  const { config: adminConfig, saveConfig } = useServer()
  const [runtimeConfig, setRuntimeConfig] = useState({})
  const [apiKeys, setApiKeys] = useState([])
  const [webhooks, setWebhooks] = useState([])
  const [newKeyTenant, setNewKeyTenant] = useState("")
  const [loading, setLoading] = useState(false)
  const { toast } = useToast()

  const fetchData = async () => {
    try {
      const [keys, hooks] = await Promise.all([
        authFetch(`${apiBase}/v1/admin/api-keys`).then(r => r.ok ? r.json() : null),
        authFetch(`${apiBase}/v1/webhooks`).then(r => r.ok ? r.json() : null),
      ])
      setApiKeys(keys?.keys || [])
      setWebhooks(hooks?.subscriptions || [])
    } catch { /* ignore network errors */ }
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { fetchData() }, [apiBase])

  const handleConfigUpdate = async () => {
    setLoading(true)
    try {
      const resp = await authFetch(`${apiBase}/v1/admin/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(runtimeConfig),
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      // Also persist to admin-config.json so values survive restarts
      const newEnv = { ...adminConfig?.env, ...runtimeConfig }
      await saveConfig({ ...adminConfig, env: newEnv })
      toast({ title: "Config updated", description: `Fields: ${(data.fields || []).join(", ") || "none"}`, variant: "success" })
      setRuntimeConfig({})
    } catch (err) {
      toast({ title: "Update failed", description: err.message, variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }

  const handleCreateKey = async () => {
    if (!newKeyTenant.trim()) return
    try {
      const resp = await authFetch(`${apiBase}/v1/admin/api-keys`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_id: newKeyTenant }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      toast({ title: "API Key created", description: `Key: ${data.api_key?.slice(0, 8)}...`, variant: "success" })
      setNewKeyTenant("")
      fetchData()
    } catch (err) {
      toast({ title: "Failed", description: err.message, variant: "destructive" })
    }
  }

  const handleDeleteKey = async (hash) => {
    try {
      const resp = await authFetch(`${apiBase}/v1/admin/api-keys/${hash}`, { method: "DELETE" })
      if (!resp.ok) throw new Error(await resp.text())
      toast({ title: "Key deleted", variant: "success" })
      fetchData()
    } catch (err) {
      toast({ title: "Failed", description: err.message, variant: "destructive" })
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold tracking-tight">Settings</h2>

      {/* Hot Reload Config */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <RefreshCw className="h-5 w-5" /> Runtime Config
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {CONFIG_WHITELIST.map(field => (
              <div key={field.key} className="space-y-1">
                <label className="text-sm font-medium">{field.label}</label>
                <Input
                  type={field.type}
                  step={field.step}
                  placeholder={`Enter ${field.label}...`}
                  value={runtimeConfig[field.key] ?? ""}
                  onChange={e => setRuntimeConfig(c => ({ ...c, [field.key]: field.type === "number" ? parseFloat(e.target.value) : e.target.value }))}
                />
              </div>
            ))}
          </div>
          <div className="mt-4">
            <Button onClick={handleConfigUpdate} disabled={Object.keys(runtimeConfig).length === 0 || loading}>
              <RefreshCw className="mr-2 h-4 w-4" /> Update Config
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* API Keys */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" /> API Keys
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Tenant ID"
              value={newKeyTenant}
              onChange={e => setNewKeyTenant(e.target.value)}
              className="max-w-xs"
            />
            <Button onClick={handleCreateKey} disabled={!newKeyTenant.trim()}>
              <Plus className="mr-2 h-4 w-4" /> Create Key
            </Button>
          </div>
          <div className="space-y-2">
            {apiKeys.length === 0 ? (
              <p className="text-sm text-muted-foreground">No API keys</p>
            ) : apiKeys.map(k => (
              <div key={k.key_hash} className="flex items-center justify-between rounded-md border p-3">
                <div>
                  <div className="font-medium">{k.tenant_id}</div>
                  <div className="font-mono text-xs text-muted-foreground">{k.key_hash}</div>
                </div>
                <div className="flex items-center gap-2">
                  {k.revoked && <Badge variant="destructive">revoked</Badge>}
                  <Button size="icon" variant="ghost" onClick={() => handleDeleteKey(k.key_hash)}>
                    <Trash2 className="h-4 w-4 text-destructive" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Webhooks */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Webhook className="h-5 w-5" /> Webhooks
          </CardTitle>
        </CardHeader>
        <CardContent>
          {webhooks.length === 0 ? (
            <p className="text-sm text-muted-foreground">No active webhook subscriptions</p>
          ) : webhooks.map(hook => (
            <div key={hook.id} className="flex items-center justify-between rounded-md border p-3">
              <div>
                <div className="font-medium">{hook.url}</div>
                <div className="text-xs text-muted-foreground">{hook.events?.join(", ")}</div>
              </div>
              <Badge variant={hook.active ? "success" : "secondary"}>{hook.active ? "active" : "inactive"}</Badge>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}
