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
  onHover: (id: string | null) => void
}

function PointSprite({ point, color, imageUrl, size, onHover }: PointSpriteProps) {
  if (imageUrl) {
    return <TextureSprite point={point} color={color} imageUrl={imageUrl} size={size} onHover={onHover} />
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
      <spriteMaterial attach="material" color={color} />
    </sprite>
  )
}

function TextureSprite({ point, imageUrl, size, onHover }: PointSpriteProps & { imageUrl: string }) {
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
  const setHoveredPointId = usePlotStore((state) => state.setHoveredPointId)

  const visiblePoints = useMemo(() => {
    if (!plotData) return []
    return plotData.points.filter((p) => visibleClusters.has(p.cluster))
  }, [plotData, visibleClusters])

  const spritesToRender = useMemo(() => {
    return visiblePoints.slice(0, MAX_SPRITES).map((point) => {
      const color = CLUSTER_COLORS[point.cluster % CLUSTER_COLORS.length]
      const imageUrl = getImageUrl(point.metadata)
      return { point, color, imageUrl }
    })
  }, [visiblePoints])

  if (!plotData) return null

  return (
    <group>
      {spritesToRender.map(({ point, color, imageUrl }) => (
        <Suspense key={point.id} fallback={<FallbackSprite point={point} color={color} size={pointSize} />}>
          <PointSprite
            point={point}
            color={color}
            imageUrl={imageUrl}
            size={pointSize}
            onHover={setHoveredPointId}
          />
        </Suspense>
      ))}
    </group>
  )
}
