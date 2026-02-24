import { Html } from '@react-three/drei'
import { useMemo } from 'react'
import { usePlotStore } from '../../stores/plotStore'

export default function TooltipCard() {
  const hoveredPointId = usePlotStore((state) => state.hoveredPointId)
  const plotData = usePlotStore((state) => state.plotData)

  const point = useMemo(() => {
    if (!hoveredPointId || !plotData) return null
    return plotData.points.find((p) => p.id === hoveredPointId)
  }, [hoveredPointId, plotData])

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
    <Html position={[point.x, point.y, point.z]} style={{ pointerEvents: 'none' }} zIndexRange={[100, 0]}>
      <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200 w-64 text-sm pointer-events-none transform -translate-x-1/2 -translate-y-[120%]">
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
