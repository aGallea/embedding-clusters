import { Html } from '@react-three/drei'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Camera, Object3D } from 'three'
import { Vector3 } from 'three'
import { usePlotStore } from '../../stores/plotStore'
import { computeTooltipPlacement } from './tooltipPositioning'

const TOOLTIP_WIDTH = 256
const TOOLTIP_OFFSET = 12
const VIEWPORT_MARGIN = 8

export default function TooltipCard() {
  const hoveredPointId = usePlotStore((state) => state.hoveredPointId)
  const plotData = usePlotStore((state) => state.plotData)
  const cardRef = useRef<HTMLDivElement | null>(null)
  const projectedPosition = useRef(new Vector3())
  const [cardSize, setCardSize] = useState({ width: TOOLTIP_WIDTH, height: 220 })

  const point = useMemo(() => {
    if (!hoveredPointId || !plotData) return null
    return plotData.points.find((p) => p.id === hoveredPointId)
  }, [hoveredPointId, plotData])

  useEffect(() => {
    const element = cardRef.current
    if (!element) return

    const updateSize = () => {
      const rect = element.getBoundingClientRect()
      if (rect.width > 0 && rect.height > 0) {
        setCardSize({ width: rect.width, height: rect.height })
      }
    }

    updateSize()

    const observer = new ResizeObserver(updateSize)
    observer.observe(element)

    return () => {
      observer.disconnect()
    }
  }, [point?.id])

  const calculatePosition = useCallback((el: Object3D, camera: Camera, size: { width: number; height: number }) => {
    el.getWorldPosition(projectedPosition.current)
    projectedPosition.current.project(camera)

    const anchorX = (projectedPosition.current.x * 0.5 + 0.5) * size.width
    const anchorY = (-projectedPosition.current.y * 0.5 + 0.5) * size.height

    const placement = computeTooltipPlacement({
      x: anchorX,
      y: anchorY,
      width: cardSize.width,
      height: cardSize.height,
      viewportWidth: size.width,
      viewportHeight: size.height,
      offset: TOOLTIP_OFFSET,
      margin: VIEWPORT_MARGIN,
    })

    return [placement.x, placement.y]
  }, [cardSize.height, cardSize.width])

  if (!point) return null

  // Find image URL in metadata if any
  const imageUrl = Object.entries(point.metadata).find(([key, value]) => {
    if (typeof value !== 'string') return false
    const lower = value.toLowerCase()
    return (
      (key.toLowerCase().includes('image') || key.toLowerCase().includes('url')) &&
      (lower.startsWith('http') || lower.startsWith('/'))
    )
  })?.[1] as string | undefined

  return (
    <Html
      position={[point.x, point.y, point.z]}
      style={{ pointerEvents: 'none' }}
      zIndexRange={[100, 0]}
      calculatePosition={calculatePosition}
    >
      <div
        ref={cardRef}
        data-testid="plot-hover-tooltip"
        className="bg-white p-3 rounded-lg shadow-lg border border-gray-200 w-64 text-sm pointer-events-none"
      >
        {imageUrl && (
          <img
            src={imageUrl}
            alt="Thumbnail"
            className="w-full h-32 object-cover rounded mb-2 bg-gray-100"
          />
        )}

        <div className="font-bold text-gray-900 mb-1">ID: {point.id}</div>

        <div className="text-xs text-gray-500 mb-2">
          Cluster: {plotData?.clusters.find(c => c.index === point.cluster)?.name || point.cluster}
        </div>

        <div className="space-y-1 max-h-40 overflow-hidden">
          {Object.entries(point.metadata).map(([key, value]) => (
            <div key={key} className="grid grid-cols-3 gap-1 text-xs">
              <span className="font-semibold text-gray-600 truncate">{key}:</span>
              <span className="col-span-2 text-gray-800 truncate">
                {String(value)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Html>
  )
}
