import { useState } from "react"
import { NavLink, useLocation } from "react-router-dom"
import { cn } from "@/lib/utils"
import { useServer } from "@/context/server-context"
import {
  LayoutDashboard, Search, Database, Workflow, BarChart3,
  Cpu, ArrowLeftRight, Settings, Menu, X, Sun, Moon,
  Server, BrainCircuit, Activity,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"

const navItems = [
  { path: "/", label: "Dashboard", icon: LayoutDashboard },
  { path: "/query", label: "Query", icon: Search },
  { path: "/nodes", label: "Memory Nodes", icon: Database },
  { path: "/pipeline", label: "Pipeline", icon: Workflow },
  { path: "/analytics", label: "Analytics", icon: BarChart3 },
  { path: "/sot", label: "SOT", icon: Cpu },
  { path: "/import-export", label: "Import / Export", icon: ArrowLeftRight },
  { path: "/settings", label: "Settings", icon: Settings },
]

const systemItems = [
  { path: "/server", label: "Server Control", icon: Server },
  { path: "/ai", label: "AI Connection", icon: BrainCircuit },
]

export function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("rtmdk-theme")
    if (saved) return saved === "dark"
    return document.documentElement.classList.contains("dark")
  })
  const location = useLocation()
  const { status } = useServer()

  const toggleTheme = () => {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle("dark", next)
    localStorage.setItem("rtmdk-theme", next ? "dark" : "light")
  }

  return (
    <div className="flex h-screen w-full bg-background text-foreground">
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 w-64 border-r bg-card transition-transform lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex h-14 items-center border-b px-4">
          <div className="flex items-center gap-2">
            <Activity className={cn("h-5 w-5", status.running ? "text-emerald-500" : "text-muted-foreground")} />
            <h1 className="text-lg font-semibold tracking-tight">RTMDK</h1>
          </div>
          <button className="ml-auto lg:hidden" onClick={() => setSidebarOpen(false)}>
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex flex-col gap-6 p-3">
          <nav className="flex flex-col gap-1">
            {navItems.map(item => {
              const Icon = item.icon
              const active = location.pathname === item.path
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  onClick={() => setSidebarOpen(false)}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    active
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </NavLink>
              )
            })}
          </nav>

          <div>
            <div className="mb-2 px-3 text-xs font-semibold uppercase text-muted-foreground">System</div>
            <nav className="flex flex-col gap-1">
              {systemItems.map(item => {
                const Icon = item.icon
                const active = location.pathname === item.path
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    onClick={() => setSidebarOpen(false)}
                    className={cn(
                      "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                      active
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                    {item.path === "/server" && (
                      <Badge variant={status.running ? "success" : "secondary"} className="ml-auto text-[10px]">
                        {status.running ? "ON" : "OFF"}
                      </Badge>
                    )}
                  </NavLink>
                )
              })}
            </nav>
          </div>
        </div>
      </aside>

      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-14 items-center gap-4 border-b bg-card px-4 lg:px-6">
          <button className="lg:hidden" onClick={() => setSidebarOpen(true)}>
            <Menu className="h-5 w-5" />
          </button>
          <div className="ml-auto flex items-center gap-3">
            {status.running && (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
                </span>
                PID {status.pid}
              </div>
            )}
            <button
              onClick={toggleTheme}
              className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-input bg-background text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </header>
        <main className="flex-1 overflow-auto p-4 lg:p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
