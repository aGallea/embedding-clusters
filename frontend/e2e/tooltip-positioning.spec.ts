import { test, expect } from '@playwright/test'
import { computeTooltipPlacement } from '../src/components/plot/tooltipPositioning'

test.describe('Tooltip placement algorithm', () => {
  test('clamps and flips within viewport bounds', async () => {
    const placement = {
      centeredAbove: computeTooltipPlacement({
        x: 400,
        y: 300,
        width: 256,
        height: 180,
        viewportWidth: 1200,
        viewportHeight: 900,
        offset: 12,
        margin: 8,
      }),
      flipBelow: computeTooltipPlacement({
        x: 400,
        y: 40,
        width: 256,
        height: 180,
        viewportWidth: 1200,
        viewportHeight: 900,
        offset: 12,
        margin: 8,
      }),
      clampLeft: computeTooltipPlacement({
        x: 30,
        y: 400,
        width: 256,
        height: 180,
        viewportWidth: 1200,
        viewportHeight: 900,
        offset: 12,
        margin: 8,
      }),
      clampRight: computeTooltipPlacement({
        x: 1180,
        y: 400,
        width: 256,
        height: 180,
        viewportWidth: 1200,
        viewportHeight: 900,
        offset: 12,
        margin: 8,
      }),
    }

    expect(placement.centeredAbove.x).toBe(272)
    expect(placement.centeredAbove.y).toBe(108)

    expect(placement.flipBelow.y).toBe(52)

    expect(placement.clampLeft.x).toBe(8)
    expect(placement.clampRight.x).toBe(936)
  })
})
