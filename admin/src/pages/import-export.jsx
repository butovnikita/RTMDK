import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/components/ui/toast"
import { useApi } from "@/hooks/use-api"
import { ArrowLeftRight, Download, Upload, FileJson, Loader2 } from "lucide-react"

export default function ImportExportPage() {
  const { apiBase, authFetch } = useApi()
  const [importJson, setImportJson] = useState("")
  const [clearExisting, setClearExisting] = useState(false)
  const [batchDocs, setBatchDocs] = useState("")
  const [loading, setLoading] = useState(false)
  const { toast } = useToast()

  const handleExport = async () => {
    try {
      const resp = await authFetch(`${apiBase}/v1/memory/export`)
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `rtmdk-export-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
      toast({ title: "Export downloaded", variant: "success" })
    } catch (err) {
      toast({ title: "Export failed", description: err.message, variant: "destructive" })
    }
  }

  const handleImport = async () => {
    setLoading(true)
    try {
      const nodes = JSON.parse(importJson)
      const resp = await authFetch(`${apiBase}/v1/memory/import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodes, clear_existing: clearExisting }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      toast({ title: `Imported ${data.imported} nodes`, variant: "success" })
      setImportJson("")
    } catch (err) {
      toast({ title: "Import failed", description: err.message, variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }

  const handleBatchIngest = async () => {
    setLoading(true)
    try {
      const docs = batchDocs.split("\n").filter(Boolean)
      if (docs.length === 0) return
      const resp = await authFetch(`${apiBase}/v1/memory/batch_ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents: docs }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      toast({ title: `Ingested ${data.ingested} documents`, variant: "success" })
      setBatchDocs("")
    } catch (err) {
      toast({ title: "Batch ingest failed", description: err.message, variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold tracking-tight">Import / Export</h2>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Export */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Download className="h-5 w-5" /> Export
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Download all memory nodes as a JSON file.
            </p>
            <Button onClick={handleExport}>
              <FileJson className="mr-2 h-4 w-4" /> Download JSON
            </Button>
          </CardContent>
        </Card>

        {/* Import */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" /> Import
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder={`Paste JSON array of nodes...\nExample:\n[\n  {\n    "id": "n1",\n    "content": { "content": "hello" },\n    "latent_pos": [0.1, 0.2, ...]\n  }\n]`}
              value={importJson}
              onChange={e => setImportJson(e.target.value)}
              rows={8}
            />
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={clearExisting}
                onChange={e => setClearExisting(e.target.checked)}
                className="h-4 w-4 rounded border-gray-300"
              />
              <label className="text-sm">Clear existing nodes before import</label>
            </div>
            <Button onClick={handleImport} disabled={!importJson.trim() || loading}>
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Upload className="mr-2 h-4 w-4" />}
              Import
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Batch Ingest */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ArrowLeftRight className="h-5 w-5" /> Batch Ingest
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Enter one document per line to ingest multiple nodes at once.
          </p>
          <Textarea
            placeholder="Document 1&#10;Document 2&#10;Document 3"
            value={batchDocs}
            onChange={e => setBatchDocs(e.target.value)}
            rows={6}
          />
          <Button onClick={handleBatchIngest} disabled={!batchDocs.trim() || loading}>
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ArrowLeftRight className="mr-2 h-4 w-4" />}
            Ingest Batch
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
