import { test, expect } from './fixtures'

type PlotStoreSnapshot = {
  selectedPointIds?: Set<string>
}

type PlotStoreWindow = Window & {
  __plotStore?: {
    getState: () => PlotStoreSnapshot
  }
}

test.describe('Cluster detail drawer', () => {
  test('opens drawer when clicking cluster legend item', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()

    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    const firstCluster = page.getByRole('button', { name: /Group 1\s+\d+ points/i })
    await expect(firstCluster).toBeVisible({ timeout: 10_000 })
    await firstCluster.click()

    await expect(
      page.getByRole('button', { name: 'Close drawer' })
    ).toBeVisible({ timeout: 10_000 })
  })

  test('opens even when plot data lacks job id', async ({ page, plotPage: _ }) => {
    await page.route(/.*\/api\/plot\/data\/.*/, async (route) => {
      const response = await route.fetch()
      const body = await response.json()
      if (typeof body === 'object' && body !== null && 'job_id' in body) {
        delete body.job_id
      }

      await route.fulfill({
        status: response.status(),
        headers: {
          ...response.headers(),
          'content-type': 'application/json',
        },
        body: JSON.stringify(body),
      })
    })

    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()

    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    const firstCluster = page.getByRole('button', { name: /Group 1\s+\d+ points/i })
    await expect(firstCluster).toBeVisible({ timeout: 10_000 })
    await firstCluster.click()

    await expect(
      page.getByRole('button', { name: 'Close drawer' })
    ).toBeVisible({ timeout: 10_000 })
  })

  test('reopens after closing without recompute', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()

    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    const firstCluster = page.getByRole('button', { name: /Group 1\s+\d+ points/i })
    await expect(firstCluster).toBeVisible({ timeout: 10_000 })
    await firstCluster.click()

    const closeButton = page.getByRole('button', { name: 'Close drawer' })
    await expect(closeButton).toBeVisible({ timeout: 10_000 })
    await closeButton.click()

    await expect(closeButton).toBeHidden({ timeout: 10_000 })

    await firstCluster.click()
    await expect(closeButton).toBeVisible({ timeout: 10_000 })
  })

  test('does not require multiple computes to open drawer', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    const firstCluster = page.getByRole('button', { name: /Group 1\s+\d+ points/i })
    await expect(firstCluster).toBeVisible({ timeout: 10_000 })
    await firstCluster.click()

    await expect(
      page.getByRole('button', { name: 'Close drawer' })
    ).toBeVisible({ timeout: 10_000 })
  })

  test('stale plot does not auto-render after visualize', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByRole('link', { name: 'Home' }).click()
    await expect(page.getByRole('link', { name: 'Home' })).toHaveAttribute('aria-current', 'page')

    await page.getByRole('button', { name: 'Visualize' }).first().click()
    await page.waitForURL('**/plot?collection=*')

    await expect(page.getByRole('heading', { name: 'Clusters' })).toBeHidden({ timeout: 10_000 })
  })

  test('opens and reopens drawer at 1472x838 viewport', async ({ page, plotPage: _ }) => {
    await page.setViewportSize({ width: 1472, height: 838 })

    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    const firstCluster = page.getByTestId('cluster-legend-name-0')
    await expect(firstCluster).toBeVisible({ timeout: 10_000 })

    await firstCluster.click()
    const closeButton = page.getByRole('button', { name: 'Close drawer' })
    await expect(closeButton).toBeVisible({ timeout: 10_000 })

    await closeButton.click()
    await expect(closeButton).toBeHidden({ timeout: 10_000 })

    await firstCluster.click()
    await expect(closeButton).toBeVisible({ timeout: 10_000 })
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })
  })

  test('group selection shows visible fixed-width drawer at large viewport', async ({ page, plotPage: _ }) => {
    await page.setViewportSize({ width: 1596, height: 958 })

    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-3').click()

    const drawer = page.getByTestId('cluster-detail-drawer')
    await expect(drawer).toBeVisible({ timeout: 10_000 })

    const drawerWidth = await drawer.evaluate((el) => el.getBoundingClientRect().width)
    expect(drawerWidth).toBeGreaterThanOrEqual(380)

    const drawerRect = await drawer.evaluate((el) => {
      const rect = el.getBoundingClientRect()
      return {
        x: rect.x,
        right: rect.right,
        y: rect.y,
        bottom: rect.bottom,
        viewportW: window.innerWidth,
        viewportH: window.innerHeight,
      }
    })
    expect(drawerRect.x).toBeGreaterThanOrEqual(0)
    expect(drawerRect.right).toBeLessThanOrEqual(drawerRect.viewportW)
    expect(drawerRect.y).toBeGreaterThanOrEqual(0)
    expect(drawerRect.bottom).toBeLessThanOrEqual(drawerRect.viewportH)

    await expect(page.getByRole('button', { name: 'Close drawer' })).toBeVisible({ timeout: 10_000 })
  })

  test('drawer stays in viewport on narrow layouts', async ({ page, plotPage: _ }) => {
    await page.setViewportSize({ width: 980, height: 900 })

    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    const firstCluster = page.getByTestId('cluster-legend-name-0')
    await expect(firstCluster).toBeVisible({ timeout: 10_000 })
    await firstCluster.click()

    const drawer = page.getByTestId('cluster-detail-drawer')
    await expect(drawer).toBeVisible({ timeout: 10_000 })

    const drawerRect = await drawer.evaluate((el) => {
      const rect = el.getBoundingClientRect()
      return {
        x: rect.x,
        right: rect.right,
        y: rect.y,
        bottom: rect.bottom,
        width: rect.width,
        viewportW: window.innerWidth,
        viewportH: window.innerHeight,
      }
    })

    expect(drawerRect.x).toBeGreaterThanOrEqual(0)
    expect(drawerRect.right).toBeLessThanOrEqual(drawerRect.viewportW)
    expect(drawerRect.y).toBeGreaterThanOrEqual(0)
    expect(drawerRect.bottom).toBeLessThanOrEqual(drawerRect.viewportH)
    expect(drawerRect.width).toBeGreaterThan(260)
  })



  test('drawer item click toggles multi-select state', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    const firstCluster = page.getByTestId('cluster-legend-name-0')
    await expect(firstCluster).toBeVisible({ timeout: 10_000 })
    await firstCluster.click()

    const drawer = page.getByTestId('cluster-detail-drawer')
    await expect(drawer).toBeVisible({ timeout: 10_000 })

    const rows = drawer.getByRole('button').filter({ hasText: /dist: /i })
    await expect(rows.first()).toBeVisible({ timeout: 10_000 })

    await rows.nth(0).click()
    await rows.nth(1).click()

    let size = await page.evaluate(() => {
      const plotWindow = window as PlotStoreWindow
      return plotWindow.__plotStore?.getState().selectedPointIds?.size
    })
    expect(size).toBe(2)

    await rows.nth(0).click()

    size = await page.evaluate(() => {
      const plotWindow = window as PlotStoreWindow
      return plotWindow.__plotStore?.getState().selectedPointIds?.size
    })
    expect(size).toBe(1)
  })

  test('drawer supports clear selected and select page', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    const firstCluster = page.getByTestId('cluster-legend-name-0')
    await expect(firstCluster).toBeVisible({ timeout: 10_000 })
    await firstCluster.click()

    const drawer = page.getByTestId('cluster-detail-drawer')
    await expect(drawer).toBeVisible({ timeout: 10_000 })

    const rows = drawer.getByRole('button').filter({ hasText: /dist: /i })
    await expect(rows.first()).toBeVisible({ timeout: 10_000 })

    await rows.nth(0).click()
    await rows.nth(1).click()

    let size = await page.evaluate(() => {
      const plotWindow = window as PlotStoreWindow
      return plotWindow.__plotStore?.getState().selectedPointIds?.size
    })
    expect(size).toBe(2)

    await page.getByRole('button', { name: 'Clear selected' }).click()

    size = await page.evaluate(() => {
      const plotWindow = window as PlotStoreWindow
      return plotWindow.__plotStore?.getState().selectedPointIds?.size
    })
    expect(size).toBe(0)

    await page.getByRole('button', { name: 'Select page' }).click()

    size = await page.evaluate(() => {
      const plotWindow = window as PlotStoreWindow
      return plotWindow.__plotStore?.getState().selectedPointIds?.size
    })
    expect(size).toBeGreaterThan(1)
  })

  test('selected items remain emphasized in particle mode', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Rendering' }).click()
    await page.getByRole('radio', { name: 'Particles' }).check()
    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    const drawer = page.getByTestId('cluster-detail-drawer')
    await expect(drawer).toBeVisible({ timeout: 10_000 })

    const rows = drawer.getByRole('button').filter({ hasText: /dist: /i })
    await rows.nth(0).click()
    await rows.nth(1).click()

    const selectedState = await page.evaluate(() => {
      const plotWindow = window as PlotStoreWindow
      return plotWindow.__plotStore?.getState().selectedPointIds?.size
    })
    expect(selectedState).toBe(2)

    await expect(page.getByRole('heading', { name: 'Visualization Error' })).toHaveCount(0)
  })

  test('selected items persist in spheres and sprites modes', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Rendering' }).click()
    await page.getByRole('radio', { name: 'Spheres' }).check()
    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    const drawer = page.getByTestId('cluster-detail-drawer')
    await expect(drawer).toBeVisible({ timeout: 10_000 })

    const rows = drawer.getByRole('button').filter({ hasText: /dist: /i })
    await rows.nth(0).click()
    await rows.nth(1).click()

    let selectedState = await page.evaluate(() => {
      const plotWindow = window as PlotStoreWindow
      return plotWindow.__plotStore?.getState().selectedPointIds?.size
    })
    expect(selectedState).toBe(2)
    await expect(page.getByRole('heading', { name: 'Visualization Error' })).toHaveCount(0)

    await page.getByRole('button', { name: 'Rendering' }).click()
    await page.getByRole('button', { name: 'Rendering' }).click()
    await page.getByRole('radio', { name: 'Sprites' }).check()

    selectedState = await page.evaluate(() => {
      const plotWindow = window as PlotStoreWindow
      return plotWindow.__plotStore?.getState().selectedPointIds?.size
    })
    expect(selectedState).toBe(2)
    await expect(page.getByRole('heading', { name: 'Visualization Error' })).toHaveCount(0)
  })

  test('distance panel shows pairwise distances for selected products', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    const drawer = page.getByTestId('cluster-detail-drawer')
    await expect(drawer).toBeVisible({ timeout: 10_000 })

    const rows = drawer.getByRole('button').filter({ hasText: /dist: /i })
    await rows.nth(0).click()
    await rows.nth(1).click()

    await expect(page.getByRole('heading', { name: 'Selected distances' })).toBeVisible({ timeout: 10_000 })
    await expect(page.getByTestId('selected-distance-row').first()).toBeVisible({ timeout: 10_000 })
    await expect(page.getByTestId('selected-distance-row').first()).toContainText(/vs/i)
  })
})
