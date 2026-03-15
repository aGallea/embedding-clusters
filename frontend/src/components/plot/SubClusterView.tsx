import { useState, useCallback, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'
import { subCluster } from '../../api/plot'
import type { SubClusterResponse } from '../../types'
import * as THREE from 'three'

interface SubClusterViewProps {
  jobId: string
  clusterIndex: number
}

function SubClusterPoints({ data }: { data: SubClusterResponse }) {
  const geometry = useMemo(() => {
    const positions = new Float32Array(data.points.length * 3)
    const colors = new Float32Array(data.points.length * 3)

    for (let i = 0; i < data.points.length; i++) {
      const p = data.points[i]
      positions[i * 3] = p.x
      positions[i * 3 + 1] = p.y
      positions[i * 3 + 2] = p.z

      const hex = CLUSTER_COLORS[p.sub_cluster % CLUSTER_COLORS.length]
      const color = new THREE.Color(hex)
      colors[i * 3] = color.r
      colors[i * 3 + 1] = color.g
      colors[i * 3 + 2] = color.b
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    return geo
  }, [data])

  return (
    <points geometry={geometry}>
      <pointsMaterial size={4} vertexColors sizeAttenuation={false} />
    </points>
  )
}

export default function SubClusterView({ jobId, clusterIndex }: SubClusterViewProps) {
  const [numSubClusters, setNumSubClusters] = useState(3)
  const subClusterData = usePlotStore((s) => s.subClusterData)
  const isLoadingSubCluster = usePlotStore((s) => s.isLoadingSubCluster)
  const setSubClusterData = usePlotStore((s) => s.setSubClusterData)
  const setIsLoadingSubCluster = usePlotStore((s) => s.setIsLoadingSubCluster)

  const handleCompute = useCallback(() => {
    setIsLoadingSubCluster(true)
    setSubClusterData(null)
    subCluster(jobId, clusterIndex, { num_sub_clusters: numSubClusters })
      .then((data: SubClusterResponse) => setSubClusterData(data))
      .catch(() => setSubClusterData(null))
      .finally(() => setIsLoadingSubCluster(false))
  }, [jobId, clusterIndex, numSubClusters, setSubClusterData, setIsLoadingSubCluster])

  return (
    <div className="p-3">
      <div className="flex items-center space-x-2 mb-2">
        <label className="text-xs text-gray-600">Sub-clusters:</label>
        <input
          type="number"
          min={2}
          max={20}
          value={numSubClusters}
          onChange={(e) => setNumSubClusters(Math.max(2, Number(e.target.value)))}
          className="w-14 text-xs border border-gray-200 rounded px-1.5 py-1 focus:outline-none focus:ring-1 focus:ring-blue-400"
        />
        <button
          onClick={handleCompute}
          disabled={isLoadingSubCluster}
          className="text-xs px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {isLoadingSubCluster ? 'Computing...' : 'Compute'}
        </button>
      </div>

      {subClusterData && (
        <>
          <div className="h-48 bg-gray-900 rounded-lg overflow-hidden">
            <Canvas camera={{ position: [0, 0, 15], fov: 60 }}>
              <color attach="background" args={['#111827']} />
              <ambientLight intensity={0.6} />
              <OrbitControls />
              <SubClusterPoints data={subClusterData} />
            </Canvas>
          </div>
          <div className="flex flex-wrap gap-1 mt-2">
            {subClusterData.sub_clusters.map((sc) => (
              <span
                key={sc.index}
                className="inline-flex items-center space-x-1 text-[10px] text-gray-600 px-1.5 py-0.5 bg-gray-100 rounded"
              >
                <span
                  className="w-2 h-2 rounded-full inline-block"
                  style={{ backgroundColor: sc.color }}
                />
                <span>{sc.count} pts</span>
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
