import { test, expect } from './fixtures'
import type { Page } from '@playwright/test'
import { CLUSTER_COLORS } from '../src/stores/plotStore'

type PlotStoreSnapshot = {
  selectedPointIds?: Set<string>
}

type PlotStoreWindow = Window & {
  __plotStore?: {
    getState: () => PlotStoreSnapshot
  }
}

async function expandFirstDrillableSubCluster(page: Page) {
  const toggles = page.locator('[data-testid^="drawer-subcluster-toggle-"]')
  const toggleCount = await toggles.count()

  for (let index = 0; index < toggleCount; index += 1) {
    await toggles.nth(index).click()
    await expect(page.getByTestId(`drawer-subcluster-panel-${index}`)).toBeVisible({
      timeout: 10_000,
    })

    return { index }
  }

  throw new Error('No expandable sub-cluster found in drawer preview')
}

async function selectFirstDrillableSubCluster(page: Page) {
  await expect(page.getByTestId('drawer-subcluster-list')).toBeVisible({ timeout: 10_000 })

  const rows = page.locator('[data-testid^="drawer-subcluster-row-"]')
  const rowCount = await rows.count()

  for (let index = 0; index < rowCount; index += 1) {
    const rowText = (await rows.nth(index).textContent()) ?? ''
    const countMatch = rowText.match(/(\d+)\s*pts/)
    const pointCount = countMatch ? Number(countMatch[1]) : 0

    if (pointCount >= 4) {
      await rows.nth(index).click()
      return { index }
    }
  }

  throw new Error('No selectable drillable sub-cluster found')
}

test.describe('Cluster detail drawer', () => {
  test('drilled legend eye icon hides and shows a sub-group', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    await expect(page.getByTestId('drill-breadcrumb')).toBeVisible({ timeout: 15_000 })

    const drilledToggle = page.getByRole('button', { name: 'Hide Sub 0' })
    await expect(drilledToggle).toBeVisible({ timeout: 10_000 })
    await drilledToggle.click()

    await expect(page.getByRole('button', { name: 'Show Sub 0' })).toBeVisible({
      timeout: 10_000,
    })
    await expect(page.getByTestId('subcluster-legend-name-0')).toContainText(/Sub 0/i)

    await page.getByRole('button', { name: 'Show Sub 0' }).click()

    await expect(page.getByRole('button', { name: 'Hide Sub 0' })).toBeVisible({
      timeout: 10_000,
    })
  })

  test('drilled legend supports isolate and show all for sub-groups', async ({
    page,
    plotPage: _,
  }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    await expect(page.getByTestId('drill-breadcrumb')).toBeVisible({ timeout: 15_000 })

    const drilledToggle = page.getByRole('button', { name: 'Hide Sub 0' })
    await expect(drilledToggle).toBeVisible({ timeout: 10_000 })
    await drilledToggle.click({ modifiers: ['ControlOrMeta'] })

    await expect(page.getByRole('button', { name: 'Hide Sub 1' })).toBeHidden()
    await expect(page.getByRole('button', { name: 'Show Sub 1' })).toBeVisible({
      timeout: 10_000,
    })

    await page.getByRole('button', { name: 'Show All' }).click()

    await expect(page.getByRole('button', { name: 'Hide Sub 0' })).toBeVisible({
      timeout: 10_000,
    })
    await expect(page.getByRole('button', { name: 'Hide Sub 1' })).toBeVisible({
      timeout: 10_000,
    })
  })

  test('drilled sub-group visibility resets after navigating back', async ({
    page,
    plotPage: _,
  }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    await expect(page.getByTestId('drill-breadcrumb')).toBeVisible({ timeout: 15_000 })

    await page.getByRole('button', { name: 'Hide Sub 0' }).click()
    await expect(page.getByRole('button', { name: 'Show Sub 0' })).toBeVisible({
      timeout: 10_000,
    })

    await page.getByTestId('breadcrumb-back').click()
    await expect(page.getByTestId('drill-breadcrumb')).toBeHidden({ timeout: 5_000 })

    await computeBtn.click()
    await expect(page.getByTestId('drill-breadcrumb')).toBeVisible({ timeout: 15_000 })
    await expect(page.getByRole('button', { name: 'Hide Sub 0' })).toBeVisible({
      timeout: 10_000,
    })
  })

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

  test('drawer compute triggers drill-down into sub-clusters', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    const breadcrumb = page.getByTestId('drill-breadcrumb')
    await expect(breadcrumb).toBeVisible({ timeout: 15_000 })

    const root = page.getByTestId('breadcrumb-root')
    await expect(root).toBeVisible()

    const subLegend = page.getByTestId('subcluster-legend-name-0')
    await expect(subLegend).toBeVisible()
  })

  test('sub-cluster chevron expands preview without drilling', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    const breadcrumb = page.getByTestId('drill-breadcrumb')
    await expect(breadcrumb).toBeVisible({ timeout: 15_000 })

    const toggle = page.getByTestId('drawer-subcluster-toggle-0')
    await expect(toggle).toBeVisible({ timeout: 10_000 })
    await toggle.click()

    await expect(page.getByTestId('drawer-subcluster-panel-0')).toBeVisible({ timeout: 10_000 })
    await expect(page.getByTestId('drawer-subcluster-preview-0')).toBeVisible({ timeout: 10_000 })
    await expect(page.getByTestId('drawer-subcluster-preview-item-0').first()).toBeVisible({
      timeout: 10_000,
    })
    await expect(page.getByTestId('drawer-subcluster-preview-meta-0').first()).toBeVisible({
      timeout: 10_000,
    })

    await expect(page.getByTestId('breadcrumb-root')).toBeVisible()
  })

  test('expanded sub-cluster panels do not show fixed preview-limit copy', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    await expect(page.getByTestId('drill-breadcrumb')).toBeVisible({ timeout: 15_000 })

    await expandFirstDrillableSubCluster(page)

    await expect(page.getByText('Showing up to 10 products')).toHaveCount(0)
  })

  test('compute is disabled until a drilled sub-cluster row is selected', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    const breadcrumb = page.getByTestId('drill-breadcrumb')
    await expect(breadcrumb).toBeVisible({ timeout: 15_000 })
    await expect(page.getByTestId('drawer-subcluster-list')).toBeVisible({ timeout: 10_000 })

    await expect(computeBtn).toBeDisabled()

    const { index } = await selectFirstDrillableSubCluster(page)
    await expect(page.getByTestId(`drawer-subcluster-row-${index}`)).toHaveAttribute(
      'data-selected',
      'true',
    )
    await expect(computeBtn).toBeEnabled()
  })

  test('selected sub-cluster row drives recursive compute', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    const breadcrumb = page.getByTestId('drill-breadcrumb')
    await expect(breadcrumb).toBeVisible({ timeout: 15_000 })
    await expect(page.getByTestId('drawer-subcluster-list')).toBeVisible({ timeout: 10_000 })

    await selectFirstDrillableSubCluster(page)
    await expect(computeBtn).toBeEnabled()
    await computeBtn.click()

    await expect(page.getByTestId('breadcrumb-level-1')).toBeVisible({ timeout: 15_000 })
  })

  test('lower product list is hidden after sub-clusters are created', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    await expect(page.getByTestId('drill-breadcrumb')).toBeVisible({ timeout: 15_000 })
    await expect(page.getByTestId('drawer-subcluster-list')).toBeVisible({ timeout: 10_000 })
    await expect(page.getByTestId('cluster-detail-main-list')).toHaveCount(0)
    await expect(page.getByText(/Page 1 \/ 2/i)).toHaveCount(0)
  })

  test('sub-cluster list scrolls when expanded preview is tall', async ({ page, plotPage: _ }) => {
    await page.setViewportSize({ width: 980, height: 700 })

    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    await expect(page.getByTestId('drill-breadcrumb')).toBeVisible({ timeout: 15_000 })

    await expandFirstDrillableSubCluster(page)

    const subClusterList = page.getByTestId('drawer-subcluster-list')
    await expect(subClusterList).toBeVisible({ timeout: 10_000 })

    const scrollMetrics = await subClusterList.evaluate((element) => {
      const listElement = element as HTMLDivElement
      listElement.scrollTop = 200

      return {
        scrollHeight: listElement.scrollHeight,
        clientHeight: listElement.clientHeight,
        scrollTop: listElement.scrollTop,
      }
    })

    expect(scrollMetrics.scrollHeight).toBeGreaterThan(scrollMetrics.clientHeight)
    expect(scrollMetrics.scrollTop).toBeGreaterThan(0)
  })

  test('drilled-mode item cards toggle product selection and plot highlight', async ({
    page,
    plotPage: _,
  }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    await expect(page.getByTestId('drill-breadcrumb')).toBeVisible({ timeout: 15_000 })

    const { index } = await expandFirstDrillableSubCluster(page)
    const firstItem = page.getByTestId(`drawer-subcluster-preview-item-${index}`).first()
    await expect(firstItem).toBeVisible({ timeout: 10_000 })

    await firstItem.click()
    await expect(firstItem).toHaveAttribute('data-selected', 'true')

    const selectedCount = await page.evaluate(() => {
      const plotWindow = window as PlotStoreWindow
      return plotWindow.__plotStore?.getState().selectedPointIds?.size ?? 0
    })
    expect(selectedCount).toBeGreaterThan(0)
  })

  test('breadcrumb back button returns to top level', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    const breadcrumb = page.getByTestId('drill-breadcrumb')
    await expect(breadcrumb).toBeVisible({ timeout: 15_000 })

    const backBtn = page.getByTestId('breadcrumb-back')
    await backBtn.click()

    await expect(breadcrumb).toBeHidden({ timeout: 5_000 })

    await expect(page.getByTestId('cluster-legend-name-0')).toBeVisible()
  })

  test('breadcrumb "All Clusters" resets drill', async ({ page, plotPage: _ }) => {
    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    const breadcrumb = page.getByTestId('drill-breadcrumb')
    await expect(breadcrumb).toBeVisible({ timeout: 15_000 })

    const root = page.getByTestId('breadcrumb-root')
    await root.click()

    await expect(breadcrumb).toBeHidden({ timeout: 5_000 })
  })

  test('sub-cluster legend uses palette colors by index', async ({ page, plotPage: _ }) => {
    const makeSubClusterResponse = () => ({
      parent_cluster_index: 0,
      total_points: 6,
      points: [
        { id: 'a', x: 0, y: 0, z: 0, sub_cluster: 0, metadata: {} },
        { id: 'b', x: 1, y: 0, z: 0, sub_cluster: 0, metadata: {} },
        { id: 'c', x: 0, y: 1, z: 0, sub_cluster: 1, metadata: {} },
        { id: 'd', x: 1, y: 1, z: 0, sub_cluster: 1, metadata: {} },
        { id: 'e', x: 0, y: 0, z: 1, sub_cluster: 2, metadata: {} },
        { id: 'f', x: 1, y: 0, z: 1, sub_cluster: 2, metadata: {} },
      ],
      sub_clusters: [
        { index: 0, count: 2, color: 'hsl(120, 70%, 50%)' },
        { index: 1, count: 2, color: 'hsl(240, 70%, 50%)' },
        { index: 2, count: 2, color: 'hsl(0, 70%, 50%)' },
      ],
    })

    const expectedRgb = (hex: string) => {
      const clean = hex.replace('#', '')
      const r = parseInt(clean.slice(0, 2), 16)
      const g = parseInt(clean.slice(2, 4), 16)
      const b = parseInt(clean.slice(4, 6), 16)
      return `rgb(${r}, ${g}, ${b})`
    }

    await page.route(/.*\/api\/plot\/.*\/cluster\/\d+\/sub-cluster$/, async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(makeSubClusterResponse()),
      })
    })

    await expect(
      page.getByRole('button', { name: 'Compute Plot' })
    ).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Compute Plot' }).click()
    await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 })

    await page.getByTestId('cluster-legend-name-0').click()
    await expect(page.getByTestId('cluster-detail-drawer')).toBeVisible({ timeout: 10_000 })

    const computeBtn = page.getByTestId('sub-cluster-compute')
    await expect(computeBtn).toBeVisible({ timeout: 5_000 })
    await computeBtn.click()

    await expect(page.getByTestId('drill-breadcrumb')).toBeVisible({ timeout: 15_000 })

    const response = makeSubClusterResponse()
    for (const sub of response.sub_clusters) {
      const swatch = page.getByTestId(`subcluster-legend-swatch-${sub.index}`)
      await expect(swatch).toBeVisible({ timeout: 10_000 })
      const color = await swatch.evaluate((el) => getComputedStyle(el).backgroundColor)
      expect(color).toBe(expectedRgb(CLUSTER_COLORS[sub.index]))
    }
  })
})
