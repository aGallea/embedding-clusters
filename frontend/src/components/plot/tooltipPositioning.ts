export interface TooltipPlacementInput {
  x: number
  y: number
  width: number
  height: number
  viewportWidth: number
  viewportHeight: number
  offset: number
  margin: number
}

export interface TooltipPlacementOutput {
  x: number
  y: number
}

export function computeTooltipPlacement(
  input: TooltipPlacementInput
): TooltipPlacementOutput {
  const {
    x,
    y,
    width,
    height,
    viewportWidth,
    viewportHeight,
    offset,
    margin,
  } = input

  const minX = margin
  const maxX = Math.max(margin, viewportWidth - width - margin)
  const centeredX = x - width / 2
  const clampedX = Math.min(Math.max(centeredX, minX), maxX)

  const preferredTopY = y - height - offset
  const bottomY = y + offset
  const canPlaceAbove = preferredTopY >= margin
  const canPlaceBelow = bottomY + height <= viewportHeight - margin

  let nextY = preferredTopY
  if (!canPlaceAbove && canPlaceBelow) {
    nextY = bottomY
  } else if (!canPlaceAbove && !canPlaceBelow) {
    nextY = Math.min(
      Math.max(preferredTopY, margin),
      Math.max(margin, viewportHeight - height - margin)
    )
  }

  return {
    x: clampedX,
    y: nextY,
  }
}
