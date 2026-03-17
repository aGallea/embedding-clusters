import { useRef, useMemo, useEffect, useCallback } from 'react'
import * as THREE from 'three'
import type { ThreeEvent } from '@react-three/fiber'
import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'

export default function InstancedSpheres() {
  const meshRef = useRef<THREE.InstancedMesh>(null!)
  const plotData = usePlotStore((state) => state.plotData)
  const visibleClusters = usePlotStore((state) => state.visibleClusters)
  const visibleSubClusters = usePlotStore((state) => state.visibleSubClusters)
  const pointSize = usePlotStore((state) => state.pointSize)
  const highlightedIds = usePlotStore((state) => state.highlightedIds)
  const selectedPointIds = usePlotStore((state) => state.selectedPointIds)
  const setHoveredPointId = usePlotStore((state) => state.setHoveredPointId)
  const subClusterColorMap = usePlotStore((state) => state.subClusterColorMap)
  const drillPath = usePlotStore((state) => state.drillPath)

  const { filteredPoints, filteredPointIds } = useMemo(() => {
    if (!plotData) return { filteredPoints: [], filteredPointIds: [] }

    const points = plotData.points.filter((point) => {
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
    const ids = points.map(p => p.id)
    return { filteredPoints: points, filteredPointIds: ids }
  }, [plotData, visibleClusters, visibleSubClusters, subClusterColorMap, drillPath])

  useEffect(() => {
    if (!meshRef.current || filteredPoints.length === 0) return
    const mesh = meshRef.current
    const matrix = new THREE.Matrix4()
    const position = new THREE.Vector3()
    const quaternion = new THREE.Quaternion()
    const scale = new THREE.Vector3()
    // Pre-create color objects for efficiency
    const colorObjects = CLUSTER_COLORS.map(hex => new THREE.Color(hex))
    const hasHighlights = highlightedIds.size > 0


    for (let i = 0; i < filteredPoints.length; i++) {
      const p = filteredPoints[i]
      const isSelected = selectedPointIds.has(p.id)
      position.set(p.x, p.y, p.z)
      scale.setScalar(isSelected ? 1.8 : 1)
      matrix.compose(position, quaternion, scale)
      mesh.setMatrixAt(i, matrix)

      let dimFactor: number
      let baseColor: THREE.Color
      if (subClusterColorMap && drillPath.length > 0) {
        const subIdx = subClusterColorMap.get(p.id)
        if (subIdx !== undefined) {
          baseColor = colorObjects[subIdx % colorObjects.length]
          dimFactor = hasHighlights && !highlightedIds.has(p.id) ? 0.15 : 1.0
        } else {
          baseColor = colorObjects[p.cluster % colorObjects.length]
          dimFactor = 0.15
        }
      } else {
        baseColor = colorObjects[p.cluster % colorObjects.length]
        dimFactor = hasHighlights && !highlightedIds.has(p.id) ? 0.15 : 1.0
      }
      const c = baseColor.clone().multiplyScalar(dimFactor)
      if (isSelected) {
        c.lerp(new THREE.Color('#ffffff'), 0.45)
      }
      mesh.setColorAt(i, c)
    }

    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [filteredPoints, highlightedIds, selectedPointIds, subClusterColorMap, drillPath])

  const handlePointerMove = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    if (e.instanceId !== undefined && e.instanceId < filteredPointIds.length) {
      setHoveredPointId(filteredPointIds[e.instanceId])
    }
  }, [filteredPointIds, setHoveredPointId])

  const handlePointerLeave = useCallback(() => {
    setHoveredPointId(null)
  }, [setHoveredPointId])

  if (!plotData || filteredPoints.length === 0) return null

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, filteredPoints.length]}
      onPointerMove={handlePointerMove}
      onPointerLeave={handlePointerLeave}
    >
      <sphereGeometry args={[pointSize * 0.06, 16, 16]} />
      <meshStandardMaterial vertexColors />
    </instancedMesh>
  )
}
