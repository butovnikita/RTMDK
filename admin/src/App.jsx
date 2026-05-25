import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { ToastProvider } from "@/components/ui/toast"
import { ServerProvider, useServer } from "@/context/server-context"
import { Layout } from "@/components/layout"
import WelcomePage from "@/pages/welcome"
import DashboardPage from "@/pages/dashboard"
import QueryPage from "@/pages/query"
import MemoryNodesPage from "@/pages/memory-nodes"
import PipelinePage from "@/pages/pipeline"
import AnalyticsPage from "@/pages/analytics"
import SOTPage from "@/pages/sot"
import ImportExportPage from "@/pages/import-export"
import SettingsPage from "@/pages/settings"
import ServerPage from "@/pages/server-control"
import AIPage from "@/pages/ai-connection"
import React from "react"

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }
  static getDerivedStateFromError() {
    return { hasError: true }
  }
  componentDidCatch(error, info) {
    console.error("RTMDK UI Error:", error, info)
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-screen items-center justify-center bg-background p-4">
          <div className="max-w-md space-y-4 text-center">
            <h1 className="text-2xl font-bold">Something went wrong</h1>
            <p className="text-muted-foreground">The UI encountered an unexpected error. Try reloading the page.</p>
            <button
              onClick={() => window.location.reload()}
              className="inline-flex items-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
              Reload Page
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

function AppRoutes() {
  const { config, loading } = useServer()

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    )
  }

  const needsSetup = !config || !config.preset

  return (
    <Routes>
      <Route path="/welcome" element={<WelcomePage />} />
      <Route path="/*" element={
        needsSetup ? <Navigate to="/welcome" replace /> : (
          <Layout>
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/query" element={<QueryPage />} />
              <Route path="/nodes" element={<MemoryNodesPage />} />
              <Route path="/memory-nodes" element={<Navigate to="/nodes" replace />} />
              <Route path="/pipeline" element={<PipelinePage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/sot" element={<SOTPage />} />
              <Route path="/import-export" element={<ImportExportPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/server" element={<ServerPage />} />
              <Route path="/server-control" element={<Navigate to="/server" replace />} />
              <Route path="/ai" element={<AIPage />} />
              <Route path="/ai-connection" element={<Navigate to="/ai" replace />} />
            </Routes>
          </Layout>
        )
      } />
    </Routes>
  )
}

function App() {
  return (
    <ToastProvider>
      <ServerProvider>
        <BrowserRouter>
          <ErrorBoundary>
            <AppRoutes />
          </ErrorBoundary>
        </BrowserRouter>
      </ServerProvider>
    </ToastProvider>
  )
}

export default App
