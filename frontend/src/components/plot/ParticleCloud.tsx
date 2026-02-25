import { useRef, useMemo, useCallback } from 'react'
import * as THREE from 'three'
import type { ThreeEvent } from '@react-three/fiber'
import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'

export default function ParticleCloud() {
  const pointsRef = useRef<THREE.Points>(null!)
  const plotData = usePlotStore((state) => state.plotData)
  const visibleClusters = usePlotStore((state) => state.visibleClusters)
  const pointSize = usePlotStore((state) => state.pointSize)
  const highlightedIds = usePlotStore((state) => state.highlightedIds)
  const setHoveredPointId = usePlotStore((state) => state.setHoveredPointId)

  // Memoize positions, colors, and the mapping back to original point IDs
  const { positions, colors, filteredPointIds } = useMemo(() => {
    if (!plotData) {
      return {
        positions: new Float32Array(0),
        colors: new Float32Array(0),
        filteredPointIds: [],
      }
    }

    const filteredPoints = plotData.points.filter((p) => visibleClusters.has(p.cluster))
    const count = filteredPoints.length

    const pos = new Float32Array(count * 3)
    const cols = new Float32Array(count * 3)
    const ids = new Array(count)

    const colorObjects = CLUSTER_COLORS.map(hex => new THREE.Color(hex))

    const hasHighlights = highlightedIds.size > 0

    for (let i = 0; i < count; i++) {
      const p = filteredPoints[i]

      // Position
      pos[i * 3] = p.x
      pos[i * 3 + 1] = p.y
      pos[i * 3 + 2] = p.z

      // Color
      const color = colorObjects[p.cluster % colorObjects.length]
      const dimFactor = hasHighlights && !highlightedIds.has(p.id) ? 0.15 : 1.0
      cols[i * 3] = color.r * dimFactor
      cols[i * 3 + 1] = color.g * dimFactor
      cols[i * 3 + 2] = color.b * dimFactor

      // ID mapping
      ids[i] = p.id
    }

    return { positions: pos, colors: cols, filteredPointIds: ids }
  }, [plotData, visibleClusters, highlightedIds])

  const handlePointerMove = useCallback((e: ThreeEvent<PointerEvent>) => {
    // Stop event propagation so we don't trigger other things
    e.stopPropagation()

    // index is the index in the buffer geometry (filtered points)
    if (e.index !== undefined && e.index < filteredPointIds.length) {
      const pointId = filteredPointIds[e.index]
      setHoveredPointId(pointId)
    }
  }, [filteredPointIds, setHoveredPointId])

  const handlePointerLeave = useCallback(() => {
    setHoveredPointId(null)
  }, [setHoveredPointId])

  if (!plotData) return null

  return (
    <points
      ref={pointsRef}
      onPointerMove={handlePointerMove}
      onPointerLeave={handlePointerLeave}
    >
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={pointSize * 0.1}
        vertexColors
        sizeAttenuation={true}
        transparent={false}
        opacity={1}
      />
    </points>
  )
}
