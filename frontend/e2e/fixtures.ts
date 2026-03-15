import { test as base, expect } from '@playwright/test'

/**
 * Extended test fixtures for embedding-clusters E2E tests.
 *
 * Provides:
 * - plotPage: Navigates to the Plot page with a collection pre-selected
 *   via the URL search param, and waits for collection details to load
 */
export const test = base.extend<{
  plotPage: void
}>({
  plotPage: async ({ page }, use) => {
    // Navigate with collection param so SearchBar renders after compute
    await page.goto('/plot?collection=fashionimageUrl')
    // Wait for the collection dropdown to be populated with options
    await expect(
      page.locator('select option:not([value=""])').first()
    ).toBeAttached({ timeout: 10_000 })
    await use()
  },
})

export { expect }
