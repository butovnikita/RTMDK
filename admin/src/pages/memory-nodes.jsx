import { useEffect, useState } from "react"
import {
  Card, CardContent,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/components/ui/toast"
import { useApi } from "@/hooks/use-api"
import { Plus, Search, Trash2, Edit2, Loader2, ChevronLeft, ChevronRight } from "lucide-react"

export default function MemoryNodesPage() {
  const { apiBase, authFetch } = useApi()
  const [nodes, setNodes] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [pageSize] = useState(20)
  const [search, setSearch] = useState("")
  const [loading, setLoading] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [editNode, setEditNode] = useState(null)
  const [deleteId, setDeleteId] = useState(null)
  const [form, setForm] = useState({ content: "", metadata: "{}", nodeId: "" })
  const { toast } = useToast()

  const fetchNodes = async () => {
    setLoading(true)
    try {
      const resp = await authFetch(`${apiBase}/v1/memory/nodes?limit=${pageSize}&offset=${page * pageSize}`)
      if (!resp.ok) throw new Error(resp.statusText)
      const data = await resp.json()
      setNodes(data.nodes || [])
      setTotal(data.total || 0)
    } catch (err) {
      toast({ title: "Failed to load nodes", description: err.message, variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { fetchNodes() }, [apiBase, page, pageSize])

  const handleCreate = async () => {
    try {
      let metadata = {}
      try { metadata = JSON.parse(form.metadata) } catch { /* ignore invalid JSON */ }
      const resp = await authFetch(`${apiBase}/v1/memory/nodes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: form.content, metadata, node_id: form.nodeId || undefined }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      toast({ title: "Node created", description: data.id, variant: "success" })
      setCreateOpen(false)
      setForm({ content: "", metadata: "{}", nodeId: "" })
      fetchNodes()
    } catch (err) {
      toast({ title: "Create failed", description: err.message, variant: "destructive" })
    }
  }

  const handleUpdate = async () => {
    try {
      const resp = await authFetch(`${apiBase}/v1/memory/nodes/${editNode.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: editNode.content }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      toast({ title: "Node updated", variant: "success" })
      setEditNode(null)
      fetchNodes()
    } catch (err) {
      toast({ title: "Update failed", description: err.message, variant: "destructive" })
    }
  }

  const handleDelete = async () => {
    try {
      const resp = await authFetch(`${apiBase}/v1/memory/nodes/${deleteId}`, { method: "DELETE" })
      if (!resp.ok) throw new Error(await resp.text())
      toast({ title: "Node deleted", variant: "success" })
      setDeleteId(null)
      fetchNodes()
    } catch (err) {
      toast({ title: "Delete failed", description: err.message, variant: "destructive" })
    }
  }

  const filtered = nodes.filter(n => {
    const text = (typeof n.content === "object" ? n.content?.content ?? JSON.stringify(n.content) : String(n.content)) || ""
    return text.toLowerCase().includes(search.toLowerCase())
  })

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">Memory Nodes</h2>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="mr-2 h-4 w-4" /> Create Node
        </Button>
      </div>

      <div className="flex items-center gap-2">
        <Search className="h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search nodes..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="max-w-sm"
        />
      </div>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Content</TableHead>
                <TableHead>Salience</TableHead>
                <TableHead className="w-[120px]">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={4} className="text-center">
                    <Loader2 className="mx-auto h-5 w-5 animate-spin" />
                  </TableCell>
                </TableRow>
              ) : filtered.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} className="text-center text-muted-foreground">No nodes found</TableCell>
                </TableRow>
              ) : filtered.map(n => (
                <TableRow key={n.id}>
                  <TableCell className="font-mono text-xs">{n.id}</TableCell>
                  <TableCell>
                    {typeof n.content === "object"
                      ? n.content?.content || JSON.stringify(n.content)
                      : String(n.content)}
                  </TableCell>
                  <TableCell>{n.salience?.toFixed?.(3) ?? "—"}</TableCell>
                  <TableCell>
                    <div className="flex gap-1">
                      <Button size="icon" variant="ghost" onClick={() => setEditNode(n)}>
                        <Edit2 className="h-4 w-4" />
                      </Button>
                      <Button size="icon" variant="ghost" onClick={() => setDeleteId(n.id)}>
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Total: {total}</span>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm">Page {page + 1}</span>
          <Button size="sm" variant="outline" onClick={() => setPage(p => p + 1)} disabled={nodes.length < pageSize}>
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Create Dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent onClose={() => setCreateOpen(false)}>
          <DialogHeader>
            <DialogTitle>Create Memory Node</DialogTitle>
            <DialogDescription>Add a new node to the memory field</DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <Input placeholder="Node ID (optional)" value={form.nodeId} onChange={e => setForm(f => ({ ...f, nodeId: e.target.value }))} />
            <Textarea placeholder="Content" value={form.content} onChange={e => setForm(f => ({ ...f, content: e.target.value }))} />
            <Textarea placeholder='Metadata JSON, e.g. {"tag":"greeting"}' value={form.metadata} onChange={e => setForm(f => ({ ...f, metadata: e.target.value }))} />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      {editNode && (
        <Dialog open onOpenChange={() => setEditNode(null)}>
          <DialogContent onClose={() => setEditNode(null)}>
            <DialogHeader>
              <DialogTitle>Edit Node</DialogTitle>
            </DialogHeader>
            <div className="space-y-3">
              <Textarea
                value={typeof editNode.content === "object" ? editNode.content?.content ?? JSON.stringify(editNode.content) : String(editNode.content)}
                onChange={e => setEditNode(n => ({ ...n, content: e.target.value }))}
              />
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setEditNode(null)}>Cancel</Button>
              <Button onClick={handleUpdate}>Save</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      {/* Delete Dialog */}
      {deleteId && (
        <Dialog open onOpenChange={() => setDeleteId(null)}>
          <DialogContent onClose={() => setDeleteId(null)}>
            <DialogHeader>
              <DialogTitle>Delete Node?</DialogTitle>
              <DialogDescription>This action cannot be undone.</DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteId(null)}>Cancel</Button>
              <Button variant="destructive" onClick={handleDelete}>Delete</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  )
}
