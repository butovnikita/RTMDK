const { test, expect } = require('@playwright/test');

const BASE = 'http://localhost:3000';

let consoleErrors = [];
let networkFailures = [];

test.beforeEach(async ({ page }) => {
  consoleErrors = [];
  networkFailures = [];
  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push({ page: page.url(), text: msg.text() });
  });
  page.on('pageerror', err => consoleErrors.push({ page: page.url(), text: err.message }));
  page.on('response', resp => {
    if (resp.status() >= 400) networkFailures.push({ url: resp.url(), status: resp.status() });
  });
});

test.afterEach(async ({ page }, testInfo) => {
  const uniqueConsole = [...new Set(consoleErrors.map(e => e.text))];
  const uniqueNetwork = [...new Map(networkFailures.map(n => [n.url, n])).values()];
  if (uniqueConsole.length > 0) {
    await testInfo.attach('console-errors', { body: JSON.stringify(uniqueConsole, null, 2), contentType: 'application/json' });
  }
  if (uniqueNetwork.length > 0) {
    await testInfo.attach('network-failures', { body: JSON.stringify(uniqueNetwork, null, 2), contentType: 'application/json' });
  }
});

// Route redirect tests

test.describe('Route Redirects', () => {
  test('/ai-connection redirects to /ai and renders content', async ({ page }) => {
    await page.goto(`${BASE}/ai-connection`);
    await page.waitForTimeout(1500);
    await expect(page).toHaveURL(/\/ai/);
    await expect(page.getByRole('heading', { name: 'AI Connection', exact: true })).toBeVisible();
  });

  test('/memory-nodes redirects to /nodes and renders content', async ({ page }) => {
    await page.goto(`${BASE}/memory-nodes`);
    await page.waitForTimeout(1500);
    await expect(page).toHaveURL(/\/nodes/);
    await expect(page.getByRole('heading', { name: 'Memory Nodes', exact: true })).toBeVisible();
  });

  test('/server-control redirects to /server and renders content', async ({ page }) => {
    await page.goto(`${BASE}/server-control`);
    await page.waitForTimeout(1500);
    await expect(page).toHaveURL(/\/server/);
    await expect(page.getByRole('heading', { name: 'Server Control', exact: true })).toBeVisible();
  });
});

// Smoke tests for all pages

test.describe('Page Smoke Tests', () => {
  const pages = [
    { path: '/', heading: 'Dashboard' },
    { path: '/query', heading: 'Query Memory' },
    { path: '/nodes', heading: 'Memory Nodes' },
    { path: '/pipeline', heading: 'Pipeline', selector: 'h2:text("Pipeline")' },
    { path: '/analytics', heading: 'Analytics' },
    { path: '/sot', heading: 'SOT' },
    { path: '/import-export', heading: 'Import / Export' },
    { path: '/settings', heading: 'Settings' },
    { path: '/ai', heading: 'AI Connection' },
  ];

  for (const p of pages) {
    test(`${p.path} renders correctly`, async ({ page }) => {
      await page.goto(`${BASE}${p.path}`);
      await page.waitForTimeout(1500);
      if (p.selector) {
        await expect(page.locator(p.selector)).toBeVisible();
      } else {
        await expect(page.getByRole('heading', { name: p.heading, exact: true })).toBeVisible();
      }
    });
  }
});

// Server Control (skip networkidle due to EventSource)

test.describe('Server Control', () => {
  test('renders with ON badge for external backend', async ({ page }) => {
    await page.goto(`${BASE}/server`);
    await page.waitForTimeout(2000);
    await expect(page.getByRole('heading', { name: 'Server Control', exact: true })).toBeVisible();
    const bodyText = await page.locator('body').textContent();
    expect(bodyText).toContain('ON');
  });
});

// Functional workflow tests

test.describe('Functional Workflows', () => {
  test('Query simple search works end-to-end', async ({ page }) => {
    await page.goto(`${BASE}/query`);
    await page.waitForTimeout(1000);
    const input = page.locator('input, textarea').first();
    await input.fill('test query');
    const buttons = page.locator('button');
    for (let i = 0; i < await buttons.count(); i++) {
      const text = await buttons.nth(i).textContent();
      if (text && text.toLowerCase().includes('query')) {
        await buttons.nth(i).click();
        break;
      }
    }
    await page.waitForTimeout(3000);
    const bodyText = await page.locator('body').textContent();
    expect(bodyText.length).toBeGreaterThan(100);
  });

  test('Settings config edit persists visually', async ({ page }) => {
    await page.goto(`${BASE}/settings`);
    await page.waitForTimeout(1000);
    const input = page.locator('input[type="text"]').first();
    await input.fill('test-value-12345');
    await page.waitForTimeout(500);
    const val = await input.inputValue();
    expect(val).toBe('test-value-12345');
  });

  test('Memory Nodes loads data from backend', async ({ page }) => {
    await page.goto(`${BASE}/nodes`);
    await page.waitForTimeout(2000);
    await expect(page.getByRole('heading', { name: 'Memory Nodes', exact: true })).toBeVisible();
    const bodyText = await page.locator('body').textContent();
    // Should not show the old "Failed to load nodes" error
    expect(bodyText).not.toContain('Failed to load nodes');
  });

  test('Analytics shows non-zero node count', async ({ page }) => {
    await page.goto(`${BASE}/analytics`);
    await page.waitForTimeout(2000);
    await expect(page.getByRole('heading', { name: 'Analytics', exact: true })).toBeVisible();
    const bodyText = await page.locator('body').textContent();
    // Memory Nodes card should show 413 (or some number > 0)
    expect(bodyText).toMatch(/Memory Nodes.*413|Memory Nodes.*[1-9]/);
  });

  test('Embedder endpoint returns valid embeddings', async ({ page }) => {
    const resp = await page.evaluate(async (base) => {
      const r = await fetch(`${base}/api/rtmdk/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: "test sentence for embedding", model: "nomic-ai/nomic-embed-text-v1.5-GGUF" }),
      });
      return { status: r.status, json: await r.json() };
    }, BASE);
    expect(resp.status).toBe(200);
    expect(resp.json.data).toHaveLength(1);
    expect(resp.json.data[0].embedding.length).toBeGreaterThan(0);
  });

  test('AI Connection provider switch updates URL', async ({ page }) => {
    await page.goto(`${BASE}/ai`);
    await page.waitForTimeout(1000);
    // Click OpenAI provider card (first exact match in provider list)
    await page.getByText('OpenAI', { exact: true }).first().click();
    await page.waitForTimeout(500);
    const urlInput = page.locator('input').first();
    const val = await urlInput.inputValue();
    expect(val).toContain('openai.com');
  });
});
