# E2E Playwright Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Playwright E2E testing infrastructure to the project with semantic search test coverage, so any developer who clones the repo can run E2E tests.

**Architecture:** Playwright runs against the full stack -- FastAPI backend serving the React SPA at `localhost:8000`. The `webServer` config in `playwright.config.ts` auto-starts the backend. Tests require pre-indexed ChromaDB data (the sample `fashion_small.csv`). The test suite covers the semantic search feature on the Plot page: search bar visibility, text search, result interaction, and clear/reset flows.

**Tech Stack:** Playwright Test, TypeScript, FastAPI, ChromaDB

**Prerequisites:** The developer must have indexed the sample data before running E2E tests. The README documents this requirement with exact commands.

---

## Task 1: Install Playwright and create configuration

**Files:**
- Modify: `frontend/package.json` (via npm install)
- Create: `frontend/playwright.config.ts`
- Create: `frontend/.gitignore` addition for Playwright artifacts

**Step 1: Install Playwright**

Run from `frontend/`:

```bash
npm install -D @playwright/test
npx playwright install chromium
```

This adds `@playwright/test` to devDependencies and installs the Chromium browser binary.

**Step 2: Create Playwright configuration**

Create `frontend/playwright.config.ts`:

```typescript
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  timeout: 60_000,
  use: {
    baseURL: 'http://localhost:8000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'cd .. && RUNNING_MODE=SERVER uv run python -m embedding_cluster',
    url: 'http://localhost:8000/api/health',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
})
```

Key decisions:
- `testDir: './e2e'` -- keeps E2E tests separate from future unit tests
- `baseURL: 'http://localhost:8000'` -- the FastAPI server serves the built SPA
- `webServer` -- auto-starts the backend; `reuseExistingServer: true` locally so devs can keep the server running
- `timeout: 60_000` -- generous timeout since computing clusters involves ML model loading
- Only Chromium for now -- lightweight, add Firefox/WebKit later if needed

**Step 3: Add npm scripts**

Add to `frontend/package.json` scripts:

```json
"test:e2e": "playwright test",
"test:e2e:ui": "playwright test --ui"
```

**Step 4: Add Playwright artifacts to .gitignore**

Append to `frontend/.gitignore`:

```
# Playwright
/test-results/
/playwright-report/
/blob-report/
/playwright/.cache/
/e2e/.auth/
```

**Step 5: Verify Playwright installs and config loads**

Run from `frontend/`:

```bash
npx playwright test --list
```

Expected: `no tests found` (no test files yet), but no config errors.

---

## Task 2: Create test fixtures and helpers

**Files:**
- Create: `frontend/e2e/fixtures.ts`

**Step 1: Create shared fixtures**

Create `frontend/e2e/fixtures.ts`:

```typescript
import { test as base, expect } from '@playwright/test'

/**
 * Extended test fixtures for embedding-clusters E2E tests.
 *
 * Provides:
 * - plotPage: Navigates to the Plot page and waits for collections to load
 */
export const test = base.extend<{
  plotPage: void
}>({
  plotPage: async ({ page }, use) => {
    await page.goto('/plot')
    // Wait for the collection dropdown to be populated
    await expect(
      page.getByRole('combobox').first()
    ).not.toHaveValue('', { timeout: 10_000 })
    await use()
  },
})

export { expect }
```

This fixture navigates to `/plot` and waits for the collections API to respond (the dropdown gets populated). All search tests need this baseline.

---

## Task 3: Create semantic search E2E tests

**Files:**
- Create: `frontend/e2e/search.spec.ts`

**Step 1: Create the test file**

Create `frontend/e2e/search.spec.ts`:

```typescript
import { test, expect } from './fixtures'

// These tests require pre-indexed data in ChromaDB.
// Run the indexing command from the README first:
//   RUNNING_MODE=INDEX LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
//     ID_FIELD=id TEXT_EMBEDDING_FIELDS='["productDisplayName"]' \
//     CHROMADB_COLLECTION_PREFIX=fashion_ uv run python -m embedding_cluster

const COLLECTION_NAME = 'fashion_productDisplayName'

test.describe('Semantic Search', () => {
  test.beforeEach(async ({ page, plotPage: _ }) => {
    // Select the collection from the dropdown
    await page.getByRole('combobox').first().selectOption(COLLECTION_NAME)

    // Wait for collection details to load (Compute button appears)
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    // Click Compute and wait for the plot to render
    await page.getByRole('button', { name: 'Compute Plot' }).click()

    // Wait for computing to finish -- "Computing Clusters..." disappears
    await expect(
      page.getByText('Computing Clusters...')
    ).toBeHidden({ timeout: 120_000 })

    // Verify plot rendered -- the canvas should be present
    await expect(page.locator('canvas')).toBeVisible({ timeout: 10_000 })
  })

  test('search bar appears after computing plot', async ({ page }) => {
    // The "Semantic Search" heading should be visible in the sidebar
    await expect(
      page.getByRole('heading', { name: 'Semantic Search' })
    ).toBeVisible()

    // Text radio should be selected by default
    const textRadio = page.getByRole('radio', { name: 'Text' })
    await expect(textRadio).toBeChecked()

    // Search input should be present
    await expect(
      page.getByPlaceholder('Search by text...')
    ).toBeVisible()

    // Search button should be visible but disabled (no query)
    const searchButton = page.getByRole('button', { name: 'Search' })
    await expect(searchButton).toBeVisible()
    await expect(searchButton).toBeDisabled()
  })

  test('text search returns results', async ({ page }) => {
    // Type a search query
    await page.getByPlaceholder('Search by text...').fill('blue shirt')

    // Search button should be enabled now
    const searchButton = page.getByRole('button', { name: 'Search' })
    await expect(searchButton).toBeEnabled()

    // Click search
    await searchButton.click()

    // Wait for results to appear -- "Results" heading with count
    await expect(
      page.getByRole('heading', { name: /Results \(\d+\)/ })
    ).toBeVisible({ timeout: 30_000 })

    // "Highlight All" button should be visible
    await expect(
      page.getByRole('button', { name: 'Highlight All' })
    ).toBeVisible()

    // Result items should be present (at least one)
    const resultItems = page.locator('button').filter({
      has: page.locator('.text-xs.text-gray-400'),
    })
    await expect(resultItems.first()).toBeVisible()
  })

  test('search via Enter key', async ({ page }) => {
    const searchInput = page.getByPlaceholder('Search by text...')
    await searchInput.fill('casual shoes')
    await searchInput.press('Enter')

    // Results should appear
    await expect(
      page.getByRole('heading', { name: /Results \(\d+\)/ })
    ).toBeVisible({ timeout: 30_000 })
  })

  test('clear search removes results', async ({ page }) => {
    // Perform a search first
    await page.getByPlaceholder('Search by text...').fill('jacket')
    await page.getByRole('button', { name: 'Search' }).click()

    // Wait for results
    await expect(
      page.getByRole('heading', { name: /Results \(\d+\)/ })
    ).toBeVisible({ timeout: 30_000 })

    // Click Clear
    await page.getByRole('button', { name: 'Clear' }).click()

    // Results should disappear
    await expect(
      page.getByRole('heading', { name: /Results \(\d+\)/ })
    ).toBeHidden()

    // Search input should be empty
    await expect(
      page.getByPlaceholder('Search by text...')
    ).toHaveValue('')
  })

  test('clicking a result highlights it', async ({ page }) => {
    // Search
    await page.getByPlaceholder('Search by text...').fill('men')
    await page.getByRole('button', { name: 'Search' }).click()

    await expect(
      page.getByRole('heading', { name: /Results \(\d+\)/ })
    ).toBeVisible({ timeout: 30_000 })

    // Click the first result -- it should get the active style (border-l-2)
    const firstResult = page.locator('button').filter({
      has: page.locator('.text-xs.text-gray-400'),
    }).first()
    await firstResult.click()

    // The clicked result should have the active indicator (blue left border)
    await expect(firstResult).toHaveClass(/border-blue-500/)
  })

  test('highlight all button activates all results', async ({ page }) => {
    // Search
    await page.getByPlaceholder('Search by text...').fill('shirt')
    await page.getByRole('button', { name: 'Search' }).click()

    await expect(
      page.getByRole('heading', { name: /Results \(\d+\)/ })
    ).toBeVisible({ timeout: 30_000 })

    // Click a single result first to narrow highlight
    const firstResult = page.locator('button').filter({
      has: page.locator('.text-xs.text-gray-400'),
    }).first()
    await firstResult.click()

    // Now click "Highlight All"
    await page.getByRole('button', { name: 'Highlight All' }).click()

    // All result buttons should have the active class
    const allResults = page.locator('button').filter({
      has: page.locator('.text-xs.text-gray-400'),
    })
    const count = await allResults.count()
    expect(count).toBeGreaterThan(1)
  })

  test('switch to image URL mode', async ({ page }) => {
    // Click image radio
    await page.getByText('Image URL').click()

    // Placeholder should change
    await expect(
      page.getByPlaceholder('Paste image URL...')
    ).toBeVisible()

    // Text placeholder should be gone
    await expect(
      page.getByPlaceholder('Search by text...')
    ).toBeHidden()
  })

  test('adjusting results slider changes value', async ({ page }) => {
    // The slider label should show the default value
    await expect(
      page.getByText('Results: 10')
    ).toBeVisible()

    // Adjust the slider
    const slider = page.locator('input[type="range"]').last()
    await slider.fill('25')

    // Label should update
    await expect(
      page.getByText('Results: 25')
    ).toBeVisible()
  })
})
```

**Step 2: Run the tests**

First, ensure data is indexed (one-time setup):

```bash
# From project root
RUNNING_MODE=INDEX \
  LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
  ID_FIELD=id \
  TEXT_EMBEDDING_FIELDS='["productDisplayName"]' \
  CHROMADB_COLLECTION_PREFIX=fashion_ \
  uv run python -m embedding_cluster
```

Then build the frontend and run tests:

```bash
cd frontend
npm run build
npx playwright test
```

Expected: All tests pass.

---

## Task 4: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

**Step 1: Add E2E testing section to README.md**

Add to the Development section, after the existing Commands subsection:

```markdown
### E2E Testing

End-to-end tests use [Playwright](https://playwright.dev/) and run
against the full stack (FastAPI backend + React frontend).

#### First-Time Setup

1. Install Playwright browsers:

    ```bash
    cd frontend
    npm install
    npx playwright install chromium
    ```

2. Index sample data for tests (one-time, from project root):

    ```bash
    RUNNING_MODE=INDEX \
      LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
      ID_FIELD=id \
      TEXT_EMBEDDING_FIELDS='["productDisplayName"]' \
      CHROMADB_COLLECTION_PREFIX=fashion_ \
      uv run python -m embedding_cluster
    ```

3. Build the frontend:

    ```bash
    cd frontend
    npm run build
    ```

#### Running E2E Tests

```bash
cd frontend

# Run all E2E tests (headless, auto-starts backend)
npm run test:e2e

# Run with interactive UI for debugging
npm run test:e2e:ui

# Run a specific test file
npx playwright test e2e/search.spec.ts

# Show HTML report after a run
npx playwright show-report
```

The Playwright config auto-starts the FastAPI server. If you
already have the server running (`RUNNING_MODE=SERVER`), it reuses
the existing server instead.
```

**Step 2: Add E2E commands to AGENTS.md**

Add to the Testing section in `AGENTS.md`:

```markdown
## E2E Testing

```bash
# Install Playwright browsers (first-time)
cd frontend && npx playwright install chromium

# Index sample data for E2E tests (first-time, from project root)
RUNNING_MODE=INDEX LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
  ID_FIELD=id TEXT_EMBEDDING_FIELDS='["productDisplayName"]' \
  CHROMADB_COLLECTION_PREFIX=fashion_ uv run python -m embedding_cluster

# Build frontend (required before E2E)
cd frontend && npm run build

# Run E2E tests
cd frontend && npm run test:e2e

# Run E2E tests with UI
cd frontend && npm run test:e2e:ui

# Run single test file
cd frontend && npx playwright test e2e/search.spec.ts
```

E2E tests require pre-indexed ChromaDB data. The `webServer` config in
`playwright.config.ts` auto-starts the FastAPI backend. Tests run against
`http://localhost:8000`.
```

---

## Task 5: Run tests and verify

**Step 1: Build frontend**

```bash
cd frontend && npm run build
```

**Step 2: Run E2E tests**

```bash
cd frontend && npx playwright test
```

Expected: All 8 tests pass.

**Step 3: Fix any failures**

If tests fail, debug with:

```bash
cd frontend && npx playwright test --ui
```

---
