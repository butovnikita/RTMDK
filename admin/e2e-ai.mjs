import { chromium } from "playwright"

const BASE_URL = "http://localhost:3000"
const bugs = []
function bug(page, severity, msg) {
  bugs.push({ page, severity, msg })
  console.error(`[${severity.toUpperCase()}] ${page}: ${msg}`)
}

async function audit() {
  const browser = await chromium.launch({ headless: true })
  const page = await browser.newPage()
  page.on("console", msg => { if (msg.type() === "error") console.log("CONSOLE:", msg.text()) })
  page.on("pageerror", err => bug("page", "high", err.message))

  // ── AI Connection ───────────────────────────────────────────────
  console.log("\n=== PAGE: AI Connection ===")
  await page.goto(`${BASE_URL}/ai`, { waitUntil: "load" })
  await page.waitForTimeout(1500)

  // Check provider cards exist
  for (const name of ["LM Studio", "OpenAI", "OpenRouter"]) {
    const visible = await page.locator(`text=${name}`).first().isVisible().catch(() => false)
    if (!visible) bug("ai", "high", `Provider card "${name}" not visible`)
  }

  // Test Connection button
  const testBtn = await page.locator("text=Test Connection").first().isVisible().catch(() => false)
  if (!testBtn) bug("ai", "high", "Test Connection button not visible")

  // Click LM Studio and verify URL updates
  await page.click("text=LM Studio")
  await page.waitForTimeout(300)
  const urlVal = await page.locator('input').first().inputValue().catch(() => "")
  if (!urlVal.includes("12345")) bug("ai", "medium", `LM Studio URL not set correctly: ${urlVal}`)

  // Click OpenRouter and verify URL updates
  await page.click("text=OpenRouter")
  await page.waitForTimeout(300)
  const urlVal2 = await page.locator('input').first().inputValue().catch(() => "")
  if (!urlVal2.includes("openrouter")) bug("ai", "medium", `OpenRouter URL not set correctly: ${urlVal2}`)

  // Check embedder section
  const embedderVisible = await page.locator("text=Embedder").first().isVisible().catch(() => false)
  if (!embedderVisible) bug("ai", "high", "Embedder section not visible")

  // Uncheck SOT and test embedder
  await page.click('input[type="checkbox"]')
  await page.waitForTimeout(300)
  const testEmbedBtn = await page.locator("text=Test Embedder").first().isVisible().catch(() => false)
  if (!testEmbedBtn) bug("ai", "high", "Test Embedder button not visible after unchecking SOT")

  if (testEmbedBtn) {
    await page.click("text=Test Embedder")
    await page.waitForTimeout(5000)
    const okText = await page.locator("text=/dimensions/i").first().isVisible().catch(() => false)
    if (!okText) {
      const errText = await page.locator("text=/✗/i").first().isVisible().catch(() => false)
      if (errText) bug("ai-embedder", "medium", "Embedder test returned error")
      else bug("ai-embedder", "medium", "Embedder test result not visible")
    } else {
      console.log("  ✓ Embedder test passed")
    }
  }

  // ── Settings ────────────────────────────────────────────────────
  console.log("\n=== PAGE: Settings ===")
  await page.goto(`${BASE_URL}/settings`, { waitUntil: "load" })
  await page.waitForTimeout(1500)

  const chatModelField = await page.locator("text=Chat Model").first().isVisible().catch(() => false)
  if (!chatModelField) bug("settings", "high", "Chat Model field not in runtime config")

  const embedModelField = await page.locator("text=Embed Model").first().isVisible().catch(() => false)
  if (!embedModelField) bug("settings", "high", "Embed Model field not in runtime config")

  // ── Summary ─────────────────────────────────────────────────────
  await browser.close()

  console.log("\n" + "=".repeat(60))
  console.log("AI / EMBEDDER AUDIT COMPLETE")
  console.log("=".repeat(60))
  if (bugs.length === 0) {
    console.log("✅ No issues found!")
  } else {
    const high = bugs.filter(b => b.severity === "high").length
    console.log(`⚠️  Found ${bugs.length} issues (${high} high)`)
    bugs.forEach((b, i) => console.log(`  ${i+1}. [${b.severity.toUpperCase()}] ${b.page}: ${b.msg}`))
  }
  console.log("=".repeat(60))
  process.exit(bugs.filter(b => b.severity === "high").length > 0 ? 1 : 0)
}

audit().catch(err => { console.error("Audit crashed:", err); process.exit(1) })
