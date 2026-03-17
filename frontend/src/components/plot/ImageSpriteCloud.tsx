import { useMemo, Suspense } from 'react'
import { useTexture } from '@react-three/drei'
import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'
import type { PlotPoint } from '../../types'

const MAX_SPRITES = 500

function getImageUrl(metadata: Record<string, unknown>): string | null {
  for (const [, value] of Object.entries(metadata)) {
    if (typeof value === 'string' && (value.startsWith('http') || value.startsWith('/'))) {
      return value
    }
  }
  return null
}

interface PointSpriteProps {
  point: PlotPoint
  color: string
  imageUrl: string | null
  size: number
  opacity: number
  onHover: (id: string | null) => void
}

function PointSprite({ point, color, imageUrl, size, opacity, onHover }: PointSpriteProps) {
  if (imageUrl) {
    return <TextureSprite point={point} color={color} imageUrl={imageUrl} size={size} opacity={opacity} onHover={onHover} />
  }

  const scale = size * 0.2
  return (
    <sprite
      position={[point.x, point.y, point.z]}
      scale={[scale, scale, scale]}
      onPointerOver={(e) => {
        e.stopPropagation()
        onHover(point.id)
      }}
      onPointerOut={() => onHover(null)}
    >
      <spriteMaterial attach="material" color={color} transparent opacity={opacity} />
    </sprite>
  )
}

function TextureSprite({ point, imageUrl, size, opacity, onHover }: PointSpriteProps & { imageUrl: string }) {
  const texture = useTexture(imageUrl)
  const scale = size * 0.6

  return (
    <sprite
      position={[point.x, point.y, point.z]}
      scale={[scale, scale, scale]}
      onPointerOver={(e) => {
        e.stopPropagation()
        onHover(point.id)
      }}
      onPointerOut={() => onHover(null)}
    >
      <spriteMaterial
        attach="material"
        map={texture}
        color={'white'}
        transparent={true}
        opacity={opacity}
      />
    </sprite>
  )
}

function FallbackSprite({ point, color, size }: { point: PlotPoint; color: string; size: number }) {
  const scale = size * 0.3
  return (
    <sprite position={[point.x, point.y, point.z]} scale={[scale, scale, scale]}>
      <spriteMaterial attach="material" color={color} />
    </sprite>
  )
}

export default function ImageSpriteCloud() {
  const plotData = usePlotStore((state) => state.plotData)
  const visibleClusters = usePlotStore((state) => state.visibleClusters)
  const pointSize = usePlotStore((state) => state.pointSize)
  const highlightedIds = usePlotStore((state) => state.highlightedIds)
  const selectedPointIds = usePlotStore((state) => state.selectedPointIds)
  const setHoveredPointId = usePlotStore((state) => state.setHoveredPointId)
  const subClusterColorMap = usePlotStore((state) => state.subClusterColorMap)
  const drillPath = usePlotStore((state) => state.drillPath)

  const visiblePoints = useMemo(() => {
    if (!plotData) return []
    return plotData.points.filter((p) => visibleClusters.has(p.cluster))
  }, [plotData, visibleClusters])

  const spritesToRender = useMemo(() => {
    return visiblePoints.slice(0, MAX_SPRITES).map((point) => {
      let color: string
      let opacity: number

      if (subClusterColorMap && drillPath.length > 0) {
        const subIdx = subClusterColorMap.get(point.id)
        if (subIdx !== undefined) {
          color = CLUSTER_COLORS[subIdx % CLUSTER_COLORS.length]
          const isHighlighted = highlightedIds.size === 0 || highlightedIds.has(point.id)
          opacity = isHighlighted ? 1.0 : 0.15
        } else {
          color = CLUSTER_COLORS[point.cluster % CLUSTER_COLORS.length]
          opacity = 0.15
        }
      } else {
        color = CLUSTER_COLORS[point.cluster % CLUSTER_COLORS.length]
        const isHighlighted = highlightedIds.size === 0 || highlightedIds.has(point.id)
        opacity = isHighlighted ? 1.0 : 0.15
      }

      const imageUrl = getImageUrl(point.metadata)
      return { point, color, imageUrl, opacity }
    })
  }, [visiblePoints, highlightedIds, subClusterColorMap, drillPath])

  const selectedSprites = useMemo(() => {
    return visiblePoints
      .filter((point) => selectedPointIds.has(point.id))
      .map((point) => ({
        point,
        color: '#ffffff',
        imageUrl: getImageUrl(point.metadata),
        opacity: 1,
      }))
  }, [visiblePoints, selectedPointIds])

  if (!plotData) return null

  return (
    <group>
      {spritesToRender.map(({ point, color, imageUrl, opacity }) => (
        <Suspense key={point.id} fallback={<FallbackSprite point={point} color={color} size={pointSize} />}>
          <PointSprite
            point={point}
            color={color}
            imageUrl={imageUrl}
            size={pointSize}
            opacity={opacity}
            onHover={setHoveredPointId}
          />
        </Suspense>
      ))}

      {selectedSprites.map(({ point, color, imageUrl, opacity }) => (
        <Suspense key={`selected-${point.id}`} fallback={<FallbackSprite point={point} color={color} size={pointSize * 1.4} />}>
          <PointSprite
            point={point}
            color={color}
            imageUrl={imageUrl}
            size={pointSize * 1.4}
            opacity={opacity}
            onHover={setHoveredPointId}
          />
        </Suspense>
      ))}
    </group>
  )
}
