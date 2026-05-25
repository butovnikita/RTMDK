const path = require("path")
require("dotenv").config({ path: path.join(__dirname, "..", ".env") })

const express = require("express")
const cors = require("cors")
const { spawn, exec } = require("child_process")
const fs = require("fs")
const os = require("os")
const http = require("http")
const net = require("net")

const app = express()
const server = http.createServer(app)
app.use(cors())
app.use(express.json())

const RTMDK_PORT = process.env.RTMDK_PORT || 8080
const ADMIN_PORT = process.env.ADMIN_PORT || 3000
const CONFIG_DIR = path.join(os.homedir(), ".rtmdk")
const CONFIG_FILE = path.join(CONFIG_DIR, "admin-config.json")
const LOG_HISTORY_MAX = 500

// Ensure config dir exists
if (!fs.existsSync(CONFIG_DIR)) fs.mkdirSync(CONFIG_DIR, { recursive: true })

let pythonProcess = null
let logHistory = []
let logClients = []

function readConfig() {
  try {
    return JSON.parse(fs.readFileSync(CONFIG_FILE, "utf-8"))
  } catch {
    return null
  }
}

function writeConfig(cfg) {
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(cfg, null, 2))
}

function broadcastLog(line) {
  logHistory.push(line)
  if (logHistory.length > LOG_HISTORY_MAX) logHistory = logHistory.slice(-LOG_HISTORY_MAX)
  const msg = `data: ${JSON.stringify({ line, time: Date.now() })}\n\n`
  logClients = logClients.filter(res => {
    try {
      res.write(msg)
      return true
    } catch {
      return false
    }
  })
}

function getProjectRoot() {
  return path.resolve(__dirname, "..")
}

function getPythonCmd() {
  return process.platform === "win32" ? "python" : "python3"
}

function checkPortFree(port) {
  return new Promise((resolve) => {
    const tester = net.createServer()
      .once("error", () => resolve(false))
      .once("listening", () => {
        // Wait for the underlying socket to fully release before resolving
        tester.close(() => resolve(true))
      })
      .listen(port)
  })
}

function isProcessRunning(pid) {
  return new Promise((resolve) => {
    if (process.platform !== "win32") {
      exec(`kill -0 ${pid}`, (err) => resolve(!err))
      return
    }
    exec(`tasklist /FI "PID eq ${pid}" /FO CSV /NH`, (err, stdout) => {
      if (err || !stdout) {
        resolve(false)
        return
      }
      resolve(stdout.includes(String(pid)))
    })
  })
}

async function killProcessOnPort(port) {
  if (process.platform === "win32") {
    return new Promise((resolve) => {
      exec(`netstat -ano | findstr :${port}`, async (err, stdout) => {
        if (err || !stdout) {
          resolve(false)
          return
        }
        const lines = stdout.trim().split("\n")
        const pidsToKill = new Set()
        for (const line of lines) {
          if (!line.includes("LISTENING")) continue
          const parts = line.trim().split(/\s+/)
          const pid = parts[parts.length - 1]
          if (pid && !isNaN(Number(pid))) {
            pidsToKill.add(pid)
          }
        }
        if (pidsToKill.size === 0) {
          resolve(false)
          return
        }
        // Use /T to kill the entire process tree (handles job-object orphans)
        for (const pid of pidsToKill) {
          exec(`taskkill /PID ${pid} /F /T`, () => {})
        }
        // Verify by checking if the port is actually free
        let attempts = 0
        const maxAttempts = 10
        const interval = 500
        const check = async () => {
          attempts++
          const portFree = await checkPortFree(port)
          if (portFree) {
            resolve(true)
            return
          }
          if (attempts < maxAttempts) {
            // Retry kill for stubborn processes
            for (const pid of pidsToKill) {
              exec(`taskkill /PID ${pid} /F /T`, () => {})
            }
            setTimeout(check, interval)
          } else {
            resolve(false)
          }
        }
        setTimeout(check, interval)
      })
    })
  } else {
    return new Promise((resolve) => {
      exec(`lsof -ti:${port}`, (err, stdout) => {
        if (err || !stdout) {
          resolve(false)
          return
        }
        const pids = stdout.trim().split("\n").filter(Boolean)
        for (const pid of pids) {
          exec(`kill -9 ${pid}`, () => {})
        }
        setTimeout(() => resolve(true), 1000)
      })
    })
  }
}

function checkPython() {
  return new Promise((resolve) => {
    exec(`${getPythonCmd()} --version`, (err) => {
      resolve(!err)
    })
  })
}

// ── API Routes ───────────────────────────────────────────────────────────

app.get("/api/server/status", async (req, res) => {
  // Check if an external server is running on the configured port
  let externalRunning = false
  try {
    const check = await new Promise((resolve) => {
      const req = http.get(`http://127.0.0.1:${RTMDK_PORT}/health`, (res) => {
        resolve(res.statusCode === 200)
      })
      req.on("error", () => resolve(false))
      req.setTimeout(2000, () => { req.destroy(); resolve(false) })
    })
    externalRunning = check
  } catch {}

  res.json({
    running: (pythonProcess !== null && !pythonProcess.killed) || externalRunning,
    pid: pythonProcess?.pid || null,
    port: RTMDK_PORT,
    uptime: pythonProcess ? Date.now() - pythonProcess._startTime : 0,
  })
})

app.post("/api/server/free-port", async (req, res) => {
  const portFree = await checkPortFree(RTMDK_PORT)
  if (portFree) {
    return res.json({ ok: true, message: "Port is already free" })
  }
  const killed = await killProcessOnPort(RTMDK_PORT)
  const nowFree = await checkPortFree(RTMDK_PORT)
  if (nowFree) {
    res.json({ ok: true, message: "Orphan process killed, port freed" })
  } else {
    res.status(400).json({ ok: false, error: "Could not free the port" })
  }
})

app.post("/api/server/start", async (req, res) => {
  if (pythonProcess && !pythonProcess.killed) {
    return res.status(400).json({ error: "Server already running" })
  }

  // Pre-flight checks
  const hasPython = await checkPython()
  if (!hasPython) {
    return res.status(400).json({ error: `Python not found. Make sure '${getPythonCmd()}' is in PATH.` })
  }

  let portFree = await checkPortFree(RTMDK_PORT)
  if (!portFree) {
    broadcastLog(`[launcher] Port ${RTMDK_PORT} busy — attempting to free it...`)
    const killed = await killProcessOnPort(RTMDK_PORT)
    if (killed) {
      portFree = await checkPortFree(RTMDK_PORT)
    }
    if (!portFree) {
      return res.status(400).json({ error: `Port ${RTMDK_PORT} is already in use. Stop the other process first.` })
    }
  }

  const cfg = readConfig() || {}
  const bodyEnv = req.body.env || {}
  const env = { ...process.env, ...cfg.env, ...bodyEnv, PORT: String(RTMDK_PORT), PYTHONIOENCODING: "utf-8" }

  logHistory = []
  broadcastLog("[launcher] Starting RTMDK server...")
  // Log key env vars for debugging configuration issues
  broadcastLog(`[launcher] Provider: ${env.RTMDK_AI_PROVIDER || 'not set'}, LM Studio: ${env.LM_STUDIO_URL || 'not set'}`)

  // Small delay to ensure Windows fully releases the test socket
  await new Promise(r => setTimeout(r, 500))

  // Double-check port is still free after the delay
  const stillFree = await checkPortFree(RTMDK_PORT)
  if (!stillFree) {
    broadcastLog(`[launcher] Port ${RTMDK_PORT} still busy after delay — aborting`)
    return res.status(400).json({ error: `Port ${RTMDK_PORT} is still in use after cleanup. Please wait a few seconds and try again.` })
  }
  broadcastLog(`[launcher] Port ${RTMDK_PORT} confirmed free`)

  const root = getProjectRoot()
  pythonProcess = spawn(getPythonCmd(), ["-c", "from rtmdk.server.app import main; main()"], {
    cwd: root,
    env,
    shell: false,
  })

  pythonProcess._startTime = Date.now()

  pythonProcess.stdout.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean)
    lines.forEach(l => broadcastLog(`[stdout] ${l}`))
  })

  pythonProcess.stderr.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean)
    lines.forEach(l => broadcastLog(`[stderr] ${l}`))
  })

  pythonProcess.on("exit", (code) => {
    broadcastLog(`[launcher] Server exited with code ${code}`)
    pythonProcess = null
  })

  pythonProcess.on("error", (err) => {
    broadcastLog(`[launcher] Error: ${err.message}`)
    pythonProcess = null
  })

  res.json({ status: "started", pid: pythonProcess.pid })
})

app.post("/api/server/stop", (req, res) => {
  if (!pythonProcess || pythonProcess.killed) {
    return res.status(400).json({ error: "Server not running" })
  }
  broadcastLog("[launcher] Stopping RTMDK server...")
  if (process.platform === "win32") {
    exec(`taskkill /pid ${pythonProcess.pid} /T /F`, () => {})
  } else {
    pythonProcess.kill("SIGTERM")
  }
  pythonProcess = null
  res.json({ status: "stopped" })
})

app.get("/api/server/logs", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream")
  res.setHeader("Cache-Control", "no-cache")
  res.setHeader("Connection", "keep-alive")
  res.flushHeaders()

  // Send history
  try {
    logHistory.forEach(line => {
      res.write(`data: ${JSON.stringify({ line, time: Date.now() })}\n\n`)
    })
  } catch {
    try { res.end() } catch {}
    return
  }

  logClients.push(res)
  req.on("close", () => {
    logClients = logClients.filter(c => c !== res)
  })
})

app.get("/api/config", (req, res) => {
  const cfg = readConfig()
  res.json(cfg || {})
})

app.post("/api/config", (req, res) => {
  const existing = readConfig() || {}
  const incoming = req.body || {}
  // Preserve preset if it exists and the incoming payload dropped it
  if (existing.preset && !incoming.preset) {
    incoming.preset = existing.preset
  }
  writeConfig(incoming)
  res.json({ status: "saved" })
})

app.get("/api/wizard/presets", (req, res) => {
  res.json([
    {
      id: "local",
      name: "Local (LM Studio)",
      description: "Single user, LM Studio on localhost, built-in embedder",
      defaults: {
        env: {
          RTMDK_PRESET: "production",
          LM_STUDIO_URL: "http://localhost:12345/v1",
          RTMDK_EMBED_MODEL: "nomic-ai/nomic-embed-text-v1.5-GGUF",
          RTMDK_ENABLE_LM_STUDIO: "true",
          RTMDK_ENABLE_API_AUTH: "false",
        },
      },
    },
    {
      id: "production",
      name: "Production Server",
      description: "Multi-user, external API, full security",
      defaults: {
        env: {
          RTMDK_PRESET: "production",
          RTMDK_ENABLE_API_AUTH: "true",
          RTMDK_ENABLE_LM_STUDIO: "false",
        },
      },
    },
    {
      id: "agent",
      name: "AI Agent",
      description: "Optimized for autonomous agents, fast queries",
      defaults: {
        env: {
          RTMDK_PRESET: "production",
          RTMDK_ENABLE_LM_STUDIO: "true",
          RTMDK_AUTO_SAVE_INTERVAL: "30",
        },
      },
    },
  ])
})

app.post("/api/wizard/setup", (req, res) => {
  const { presetId, env } = req.body
  const cfg = readConfig() || {}
  cfg.preset = presetId
  cfg.env = { ...cfg.env, ...env }
  writeConfig(cfg)
  res.json({ status: "saved" })
})

// ── AI Test & Models ─────────────────────────────────────────────────────

function isEmbeddingModel(m) {
  const text = `${m.id} ${m.name || ""} ${m.description || ""}`.toLowerCase()
  return text.includes("embed") || text.includes("text-embedding") || text.includes("e5-")
}

function isChatModel(m) {
  return !isEmbeddingModel(m)
}

async function fetchModels(provider, url, apiKey) {
  if (provider === "lm_studio") {
    const target = url || "http://localhost:12345/v1"
    const response = await fetch(`${target}/models`, { method: "GET" })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const data = await response.json()
    return (data.data || []).filter(isChatModel).map(m => ({ id: m.id, name: m.id, context_length: m.context_length || null }))
  } else if (provider === "openai") {
    const response = await fetch("https://api.openai.com/v1/models", {
      headers: { Authorization: `Bearer ${apiKey}` },
    })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const data = await response.json()
    const chatModels = (data.data || []).filter(m => isChatModel(m) && (m.id.includes("gpt") || m.id.includes("o1") || m.id.includes("o3")))
    return chatModels.map(m => ({ id: m.id, name: m.id, context_length: m.context_length || null }))
  } else if (provider === "openrouter") {
    const response = await fetch("https://openrouter.ai/api/v1/models", {
      headers: { Authorization: `Bearer ${apiKey}` },
    })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const data = await response.json()
    const models = (data.data || [])
    // Filter for chat-capable models only
    const chatModels = models.filter(isChatModel).slice(0, 200)
    return chatModels.map(m => ({ id: m.id, name: m.name || m.id, context_length: m.context_length || null }))
  }
  throw new Error("Unknown provider")
}

async function fetchEmbedModels(provider, url, apiKey) {
  if (provider === "lm_studio") {
    const target = url || "http://localhost:12345/v1"
    const response = await fetch(`${target}/models`, { method: "GET" })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const data = await response.json()
    return (data.data || []).filter(isEmbeddingModel).map(m => ({ id: m.id, name: m.name || m.id }))
  } else if (provider === "openai") {
    // Use static list of known OpenAI embedding models to avoid fetching all models
    const knownEmbedModels = [
      { id: "text-embedding-3-small", name: "text-embedding-3-small (1536d)" },
      { id: "text-embedding-3-large", name: "text-embedding-3-large (3072d)" },
      { id: "text-embedding-ada-002", name: "text-embedding-ada-002 (1536d)" },
    ]
    // Optionally validate by fetching all models, but static list is faster and reliable
    return knownEmbedModels
  } else if (provider === "openrouter") {
    const response = await fetch("https://openrouter.ai/api/v1/models", {
      headers: { Authorization: `Bearer ${apiKey}` },
    })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const data = await response.json()
    const embedModels = (data.data || []).filter(isEmbeddingModel).slice(0, 100)
    return embedModels.map(m => ({ id: m.id, name: m.name || m.id }))
  }
  throw new Error("Unknown provider")
}

app.post("/api/ai/test", async (req, res) => {
  const { provider, url, apiKey } = req.body
  try {
    const models = await fetchModels(provider, url, apiKey)
    res.json({ ok: true, models })
  } catch (err) {
    res.json({ ok: false, error: err.message })
  }
})

app.post("/api/ai/models", async (req, res) => {
  const { provider, url, apiKey } = req.body
  try {
    const models = await fetchModels(provider, url, apiKey)
    res.json({ ok: true, models })
  } catch (err) {
    res.json({ ok: false, error: err.message })
  }
})

app.post("/api/ai/embed-models", async (req, res) => {
  const { provider, url, apiKey } = req.body
  try {
    const models = await fetchEmbedModels(provider, url, apiKey)
    res.json({ ok: true, models })
  } catch (err) {
    res.json({ ok: false, error: err.message })
  }
})

// ── Proxy to RTMDK ───────────────────────────────────────────────────────

app.use("/api/rtmdk", (req, res) => {
  const options = {
    hostname: "127.0.0.1",
    port: RTMDK_PORT,
    path: req.url,
    method: req.method,
    headers: {
      host: `127.0.0.1:${RTMDK_PORT}`,
      "content-type": req.headers["content-type"] || "",
      authorization: req.headers.authorization || "",
      "x-api-key": req.headers["x-api-key"] || "",
    },
  }

  const proxy = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers)
    proxyRes.pipe(res)
  })

  proxy.on("error", (err) => {
    if (!res.headersSent) {
      res.status(502).json({ error: "RTMDK server unavailable", detail: err.message })
    }
  })

  req.on("error", () => {
    proxy.destroy()
  })

  // Forward body. express.json() consumes the raw stream for JSON requests,
  // so we must forward req.body manually. For other requests, pipe the stream.
  if (req.body !== undefined) {
    proxy.write(JSON.stringify(req.body))
    proxy.end()
  } else {
    req.pipe(proxy)
  }
})

// ── Serve React ──────────────────────────────────────────────────────────

const distPath = path.join(__dirname, "dist")
if (fs.existsSync(distPath)) {
  app.use(express.static(distPath))
  app.use((req, res) => {
    res.sendFile(path.join(distPath, "index.html"))
  })
} else {
  app.get("/", (req, res) => {
    res.send("RTMDK Admin — run `npm run build` to serve the UI, or use `npm run dev` for development.")
  })
}

// ── Process resilience ───────────────────────────────────────────────────

process.on("uncaughtException", (err) => {
  console.error("[launcher] Uncaught exception:", err.message)
})

process.on("unhandledRejection", (reason) => {
  console.error("[launcher] Unhandled rejection:", reason)
})

// ── Start ────────────────────────────────────────────────────────────────

server.listen(ADMIN_PORT, () => {
  console.log(`RTMDK Admin running on http://localhost:${ADMIN_PORT}`)
  console.log(`RTMDK API proxy: /api/rtmdk/* → http://127.0.0.1:${RTMDK_PORT}`)
})
