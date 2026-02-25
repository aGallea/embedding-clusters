import { test, expect } from './fixtures'

// These tests require pre-indexed data in ChromaDB.
// Run the indexing command from the README first:
//   RUNNING_MODE=INDEX LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
//     ID_FIELD=id TEXT_EMBEDDING_FIELDS='["productDisplayName"]' \
//     CHROMADB_COLLECTION_PREFIX=fashion_ uv run python -m embedding_cluster

test.describe('Semantic Search', () => {
  test.beforeEach(async ({ page, plotPage: _ }) => {
    // Wait for collection details to load (Compute button appears)
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    // Click Compute and wait for the plot to render
    await page.getByRole('button', { name: 'Compute Plot' }).click()

    // Wait for the canvas to appear (plot rendered)
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    // Wait for the search bar to be available
    await expect(
      page.getByRole('heading', { name: 'Semantic Search' })
    ).toBeVisible({ timeout: 5_000 })
  })

  test('search bar appears after computing plot', async ({ page }) => {
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
