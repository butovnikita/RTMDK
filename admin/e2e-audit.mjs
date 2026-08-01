/**
 * Full E2E UI Audit for RTMDK Admin
 * Usage: node e2e-audit.mjs
 */
import { chromium } from "playwright"

const BASE_URL = "http://localhost:3000"
const TIMEOUT = 15000

const bugs = []
function bug(page, severity, msg, detail = "") {
  const entry = { page, severity, msg, detail }
  bugs.push(entry)
  console.error(`[${severity.toUpperCase()}] ${page}: ${msg}${detail ? " | " + detail : ""}`)
}

async function audit() {
  const browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } })
  const page = await context.newPage()

  page.on("console", (msg) => {
    const type = msg.type()
    const text = msg.text()
    if (type === "error") {
      if (text.includes("Failed to load resource") && text.includes("502")) return
      if (text.includes("the server responded with a status of 502")) return
      bug("console", "high", text, `type=${type}`)
    } else if (type === "warning" && !text.includes("chart should be greater than 0")) {
      bug("console", "low", text, `type=${type}`)
    }
  })

  page.on("pageerror", (err) => {
    bug("page", "high", err.message, err.stack?.slice(0, 200) || "")
  })

  page.on("response", (resp) => {
    const status = resp.status()
    const url = resp.url()
    if (status >= 400 && url.includes("localhost")) {
      if (url.includes("/health/deep") && status === 404) return
      if (url.includes("/api/rtmdk/") && status === 502) return
      bug("network", "medium", `HTTP ${status}`, url)
    }
  })

  // ── Welcome ─────────────────────────────────────────────────────
  console.log("\n=== PAGE: Welcome ===")
  await page.goto(`${BASE_URL}/welcome`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(1000)

  const needsSetup = await page.locator("text=Welcome to RTMDK").isVisible().catch(() => false)
  if (needsSetup) {
    console.log("Welcome wizard visible — filling setup...")
    await page.click("text=Local (LM Studio)")
    await page.fill('input[name="url"]', "http://localhost:12345/v1")
    await page.click("text=Test Connection")
    await page.waitForTimeout(3000)
    const testError = await page.locator("text=Connection failed").isVisible().catch(() => false)
    if (testError) bug("welcome", "medium", "Test Connection failed", "LM Studio not reachable")
    await page.click("text=Continue")
    await page.click("text=Finish Setup")
    await page.waitForURL(`${BASE_URL}/`, { timeout: TIMEOUT })
  } else {
    console.log("Already configured — skipping welcome wizard")
  }

  // ── Dashboard ───────────────────────────────────────────────────
  console.log("\n=== PAGE: Dashboard ===")
  await page.goto(`${BASE_URL}/`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(2000)

  const dashError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (dashError) bug("dashboard", "high", "Error boundary triggered")

  const memNodesVisible = await page.locator("text=Memory Nodes").first().isVisible().catch(() => false)
  if (!memNodesVisible) bug("dashboard", "medium", "Memory Nodes card not visible")

  // ── Query (Simple) ──────────────────────────────────────────────
  console.log("\n=== PAGE: Query (Simple) ===")
  await page.goto(`${BASE_URL}/query`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(1000)

  const queryError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (queryError) bug("query", "high", "Error boundary triggered")

  await page.type('input[placeholder="Enter your query..."]', "architecture", { delay: 20 })
  await page.click("button:has(.lucide-search)")
  await page.waitForTimeout(4000)

  const hasResults = await page.locator("text=/Total: \\d+/").first().isVisible().catch(() => false)
  const hasErrorToast = await page.locator("text=Query failed").isVisible().catch(() => false)
  if (!hasResults && !hasErrorToast) {
    bug("query-simple", "high", "Simple query did not show results or error toast")
  }
  if (hasErrorToast) {
    const errText = await page.locator("[data-state='open']").textContent().catch(() => "")
    bug("query-simple", "high", "Simple query failed with toast", errText.slice(0, 100))
  }

  // ── Query (Pipeline) ────────────────────────────────────────────
  console.log("\n=== PAGE: Query (Pipeline tab) ===")
  await page.click("text=Pipeline Query")
  await page.waitForTimeout(500)
  await page.click("button:has(.lucide-search)")
  await page.waitForTimeout(4000)

  const pipeResults = await page.locator("text=/Total: \\d+/").first().isVisible().catch(() => false)
  const pipeErrorToast = await page.locator("text=Query failed").isVisible().catch(() => false)
  if (!pipeResults && !pipeErrorToast) {
    bug("query-pipeline", "high", "Pipeline query did not show results or error toast")
  }
  if (pipeErrorToast) {
    const errText = await page.locator("[data-state='open']").textContent().catch(() => "")
    bug("query-pipeline", "high", "Pipeline query failed with toast", errText.slice(0, 100))
  }

  // ── Memory Nodes ────────────────────────────────────────────────
  console.log("\n=== PAGE: Memory Nodes ===")
  await page.goto(`${BASE_URL}/nodes`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(2000)

  const nodesError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (nodesError) bug("nodes", "high", "Error boundary triggered")

  const nodeRows = await page.locator("table tbody tr").count().catch(() => 0)
  if (nodeRows === 0) bug("nodes", "high", "No node rows visible")
  console.log(`  Node rows visible: ${nodeRows}`)

  await page.fill('input[placeholder*="Search"]', "project")
  await page.waitForTimeout(1000)
  const filteredRows = await page.locator("table tbody tr").count().catch(() => 0)
  console.log(`  Filtered rows: ${filteredRows}`)

  // ── Analytics ───────────────────────────────────────────────────
  console.log("\n=== PAGE: Analytics ===")
  await page.goto(`${BASE_URL}/analytics`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(1500)

  const analyticsError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (analyticsError) bug("analytics", "high", "Error boundary triggered")

  // ── Import/Export ───────────────────────────────────────────────
  console.log("\n=== PAGE: Import/Export ===")
  await page.goto(`${BASE_URL}/import-export`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(1500)

  const ieError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (ieError) bug("import-export", "high", "Error boundary triggered")

  // ── Settings ────────────────────────────────────────────────────
  console.log("\n=== PAGE: Settings ===")
  await page.goto(`${BASE_URL}/settings`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(1500)

  const settingsError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (settingsError) bug("settings", "high", "Error boundary triggered")

  const saveBtn = await page.locator("text=Save Settings").first()
  if (await saveBtn.isVisible().catch(() => false)) {
    await saveBtn.click()
    await page.waitForTimeout(2000)
  }

  // ── Server Control (use load, not networkidle because of SSE) ───
  console.log("\n=== PAGE: Server Control ===")
  await page.goto(`${BASE_URL}/server`, { waitUntil: "load", timeout: TIMEOUT })
  await page.waitForTimeout(1500)

  const serverError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (serverError) bug("server", "high", "Error boundary triggered")

  const statusText = await page.locator("text=/Running|Stopped|OFF/i").first().isVisible().catch(() => false)
  if (!statusText) bug("server-status", "medium", "Server status not visible")

  // ── AI Connection ───────────────────────────────────────────────
  console.log("\n=== PAGE: AI Connection ===")
  await page.goto(`${BASE_URL}/ai`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(1500)

  const aiError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (aiError) bug("ai", "high", "Error boundary triggered")

  // ── Pipeline page ───────────────────────────────────────────────
  console.log("\n=== PAGE: Pipeline ===")
  await page.goto(`${BASE_URL}/pipeline`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(1500)

  const pipePageError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (pipePageError) bug("pipeline-page", "high", "Error boundary triggered")

  // ── SOT ─────────────────────────────────────────────────────────
  console.log("\n=== PAGE: SOT ===")
  await page.goto(`${BASE_URL}/sot`, { waitUntil: "networkidle", timeout: TIMEOUT })
  await page.waitForTimeout(1500)

  const sotError = await page.locator("text=Something went wrong").isVisible().catch(() => false)
  if (sotError) bug("sot", "high", "Error boundary triggered")

  // ── Summary ─────────────────────────────────────────────────────
  await browser.close()

  console.log("\n" + "=".repeat(60))
  console.log("AUDIT COMPLETE")
  console.log("=".repeat(60))
  if (bugs.length === 0) {
    console.log("✅ No issues found!")
  } else {
    const high = bugs.filter((b) => b.severity === "high").length
    const med = bugs.filter((b) => b.severity === "medium").length
    const low = bugs.filter((b) => b.severity === "low").length
    console.log(`⚠️  Found ${bugs.length} issues (${high} high, ${med} medium, ${low} low)`)
    console.log("\nDetailed list:")
    bugs.forEach((b, i) => {
      console.log(`  ${i + 1}. [${b.severity.toUpperCase()}] ${b.page}: ${b.msg}`)
    })
  }
  console.log("=".repeat(60))

  process.exit(bugs.filter((b) => b.severity === "high").length > 0 ? 1 : 0)
}

audit().catch((err) => {
  console.error("Audit crashed:", err)
  process.exit(1)
})
