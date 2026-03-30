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

  test('search bar appears in sidebar after compute', async ({ page }) => {
    const sidebar = page.getByTestId('plot-sidebar')
    await expect(sidebar).toBeVisible({ timeout: 5_000 })
    await expect(
      sidebar.getByRole('heading', { name: 'Semantic Search' })
    ).toBeVisible()
  })

  test('text search returns results', async ({ page }) => {
    // Type a search query
    await page.getByPlaceholder('Search by text...').fill('blue shirt')

    // Search button should be enabled now
    const searchButton = page.getByRole('button', { name: 'Search' })
    await expect(searchButton).toBeEnabled()

    // Click search
    await searchButton.click()

    // Wait for results heading
    await expect(
      page.getByRole('heading', { name: /Results \(\d+\)/ })
    ).toBeVisible({ timeout: 30_000 })

    // "Highlight All" button should be visible
    await expect(
      page.getByRole('button', { name: 'Highlight All' })
    ).toBeVisible()

    // Result items should be present (at least one)
    const resultItems = page.getByTestId('search-result-item')
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
    const firstResult = page.getByTestId('search-result-item').first()
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
    const firstResult = page.getByTestId('search-result-item').first()
    await firstResult.click()

    // Now click "Highlight All"
    await page.getByRole('button', { name: 'Highlight All' }).click()

    // All result buttons should have the active class
    const allResults = page.getByTestId('search-result-item')
    const count = await allResults.count()
    expect(count).toBeGreaterThan(1)
    for (let i = 0; i < count; i += 1) {
      await expect(allResults.nth(i)).toHaveClass(/border-blue-500/)
    }
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

    // Adjust the search results slider (scoped near its label)
    const resultsLabel = page.getByText('Results: 10')
    const slider = resultsLabel.locator('..').locator('input[type="range"]')
    await slider.fill('25')

    // Label should update
    await expect(
      page.getByText('Results: 25')
    ).toBeVisible()
  })

  test('hovering points does not trigger visualization error', async ({ page }) => {
    const canvas = page.locator('canvas')
    await expect(canvas).toBeVisible({ timeout: 10_000 })

    const box = await canvas.boundingBox()
    expect(box).not.toBeNull()

    if (!box) {
      throw new Error('Canvas bounding box is null')
    }

    for (let i = 0; i < 12; i += 1) {
      const x = Math.floor(box.x + box.width * (0.25 + (i % 4) * 0.15))
      const y = Math.floor(box.y + box.height * (0.25 + Math.floor(i / 4) * 0.2))
      await page.mouse.move(x, y)
      await page.mouse.move(Math.floor(box.x + 4), Math.floor(box.y + 4))
    }

    await expect(
      page.getByRole('heading', { name: 'Visualization Error' })
    ).toHaveCount(0)
  })

})

test.describe('Plot Sidebar', () => {
  test('plot sidebar does not duplicate visible labels', async ({ page }) => {
    // Mock the API calls so we don't need real ChromaDB data
    await page.route('**/api/collections', async route => {
      await route.fulfill({ json: [{ name: 'fashionimageUrl', count: 100 }] })
    })
    await page.route('**/api/collections/fashionimageUrl', async route => {
      await route.fulfill({
        json: { name: 'fashionimageUrl', count: 100, metadata_fields: ['productDisplayName', 'imageUrl'] }
      })
    })

    // Navigate with collection param
    await page.goto('/plot?collection=fashionimageUrl')

    // Wait for the collection dropdown to be populated with options
    await expect(
      page.locator('select option:not([value=""])').first()
    ).toBeAttached({ timeout: 10_000 })

    const sidebar = page.getByTestId('plot-sidebar')

    await expect(sidebar.getByRole('button', { name: 'Collection', exact: true })).toBeVisible({ timeout: 5_000 })

    const displayFieldsBtn = sidebar.getByRole('button', { name: 'Display Fields', exact: true })
    const imageFieldBtn = sidebar.getByRole('button', { name: 'Image Field', exact: true })

    await displayFieldsBtn.click()
    await imageFieldBtn.click()

    await expect(sidebar.locator('label').filter({ hasText: /^Collection$/ })).toHaveCount(0)
    await expect(sidebar.locator('label').filter({ hasText: /^Display Fields$/ })).toHaveCount(0)
    await expect(sidebar.locator('label').filter({ hasText: /^Image Field$/ })).toHaveCount(0)

    await expect(sidebar.getByRole('combobox', { name: 'Collection' })).toBeVisible()
    await expect(sidebar.getByRole('combobox', { name: 'Image Field' })).toBeVisible()
  })
})
