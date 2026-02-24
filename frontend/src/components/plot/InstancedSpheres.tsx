import { useRef, useMemo, useEffect, useCallback } from 'react'
import * as THREE from 'three'
import type { ThreeEvent } from '@react-three/fiber'
import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'

export default function InstancedSpheres() {
  const meshRef = useRef<THREE.InstancedMesh>(null!)
  const plotData = usePlotStore((state) => state.plotData)
  const visibleClusters = usePlotStore((state) => state.visibleClusters)
  const pointSize = usePlotStore((state) => state.pointSize)
  const setHoveredPointId = usePlotStore((state) => state.setHoveredPointId)

  const { filteredPoints, filteredPointIds } = useMemo(() => {
    if (!plotData) return { filteredPoints: [], filteredPointIds: [] }

    const points = plotData.points.filter((p) => visibleClusters.has(p.cluster))
    const ids = points.map(p => p.id)
    return { filteredPoints: points, filteredPointIds: ids }
  }, [plotData, visibleClusters])

  useEffect(() => {
    if (!meshRef.current || filteredPoints.length === 0) return
    const mesh = meshRef.current
    const matrix = new THREE.Matrix4()
    // Pre-create color objects for efficiency
    const colorObjects = CLUSTER_COLORS.map(hex => new THREE.Color(hex))

    for (let i = 0; i < filteredPoints.length; i++) {
      const p = filteredPoints[i]
      matrix.setPosition(p.x, p.y, p.z)
      mesh.setMatrixAt(i, matrix)

      const c = colorObjects[p.cluster % colorObjects.length]
      mesh.setColorAt(i, c)
    }

    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [filteredPoints])

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
