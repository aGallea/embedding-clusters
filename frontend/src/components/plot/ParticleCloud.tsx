import { useRef, useMemo, useCallback } from 'react'
import * as THREE from 'three'
import type { ThreeEvent } from '@react-three/fiber'
import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'

export default function ParticleCloud() {
  const pointsRef = useRef<THREE.Points>(null!)
  const plotData = usePlotStore((state) => state.plotData)
  const visibleClusters = usePlotStore((state) => state.visibleClusters)
  const visibleSubClusters = usePlotStore((state) => state.visibleSubClusters)
  const pointSize = usePlotStore((state) => state.pointSize)
  const highlightedIds = usePlotStore((state) => state.highlightedIds)
  const selectedPointIds = usePlotStore((state) => state.selectedPointIds)
  const setHoveredPointId = usePlotStore((state) => state.setHoveredPointId)
  const subClusterColorMap = usePlotStore((state) => state.subClusterColorMap)
  const drillPath = usePlotStore((state) => state.drillPath)

  // Memoize positions, colors, and the mapping back to original point IDs
  const { positions, colors, filteredPointIds, selectedPositions, selectedColors } = useMemo(() => {
    if (!plotData) {
      return {
        positions: new Float32Array(0),
        colors: new Float32Array(0),
        filteredPointIds: [],
        selectedPositions: new Float32Array(0),
        selectedColors: new Float32Array(0),
      }
    }

    const filteredPoints = plotData.points.filter((point) => {
      if (!visibleClusters.has(point.cluster)) {
        return false
      }

      if (subClusterColorMap && drillPath.length > 0) {
        const subClusterIndex = subClusterColorMap.get(point.id)
        if (subClusterIndex === undefined) {
          return false
        }

        return visibleSubClusters.has(subClusterIndex)
      }

      return true
    })
    const count = filteredPoints.length
    const selectedPoints = filteredPoints.filter((p) => selectedPointIds.has(p.id))

    const pos = new Float32Array(count * 3)
    const cols = new Float32Array(count * 3)
    const ids = new Array(count)
    const selectedPos = new Float32Array(selectedPoints.length * 3)
    const selectedCols = new Float32Array(selectedPoints.length * 3)

    const colorObjects = CLUSTER_COLORS.map(hex => new THREE.Color(hex))

    const hasHighlights = highlightedIds.size > 0

    for (let i = 0; i < count; i++) {
      const p = filteredPoints[i]

      // Position
      pos[i * 3] = p.x
      pos[i * 3 + 1] = p.y
      pos[i * 3 + 2] = p.z

      // Color
      let dimFactor: number
      let color: THREE.Color
      if (subClusterColorMap && drillPath.length > 0) {
        const subIdx = subClusterColorMap.get(p.id)
        if (subIdx !== undefined) {
          color = colorObjects[subIdx % colorObjects.length]
          dimFactor = hasHighlights && !highlightedIds.has(p.id) ? 0.15 : 1.0
        } else {
          color = colorObjects[p.cluster % colorObjects.length]
          dimFactor = 0.15
        }
      } else {
        color = colorObjects[p.cluster % colorObjects.length]
        dimFactor = hasHighlights && !highlightedIds.has(p.id) ? 0.15 : 1.0
      }
      cols[i * 3] = color.r * dimFactor
      cols[i * 3 + 1] = color.g * dimFactor
      cols[i * 3 + 2] = color.b * dimFactor

      // ID mapping
      ids[i] = p.id
    }

    for (let i = 0; i < selectedPoints.length; i++) {
      const p = selectedPoints[i]
      selectedPos[i * 3] = p.x
      selectedPos[i * 3 + 1] = p.y
      selectedPos[i * 3 + 2] = p.z

      selectedCols[i * 3] = 1
      selectedCols[i * 3 + 1] = 1
      selectedCols[i * 3 + 2] = 1
    }

    return {
      positions: pos,
      colors: cols,
      filteredPointIds: ids,
      selectedPositions: selectedPos,
      selectedColors: selectedCols,
    }
  }, [plotData, visibleClusters, visibleSubClusters, highlightedIds, selectedPointIds, subClusterColorMap, drillPath])

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
    <group>
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

      {selectedPositions.length > 0 && (
        <points>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={selectedPositions.length / 3}
              array={selectedPositions}
              itemSize={3}
              args={[selectedPositions, 3]}
            />
            <bufferAttribute
              attach="attributes-color"
              count={selectedColors.length / 3}
              array={selectedColors}
              itemSize={3}
              args={[selectedColors, 3]}
            />
          </bufferGeometry>
          <pointsMaterial
            size={pointSize * 0.2}
            vertexColors
            sizeAttenuation={true}
            transparent={true}
            opacity={1}
            depthWrite={false}
          />
        </points>
      )}
    </group>
  )
}
