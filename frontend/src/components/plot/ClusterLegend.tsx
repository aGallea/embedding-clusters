import { useCallback } from 'react'
import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'
import { subCluster, subClusterByPointIds } from '../../api/plot'

export default function ClusterLegend() {
  const plotData = usePlotStore((state) => state.plotData)
  const visibleClusters = usePlotStore((state) => state.visibleClusters)
  const toggleCluster = usePlotStore((state) => state.toggleCluster)
  const isolateCluster = usePlotStore((state) => state.isolateCluster)
  const resetVisibleClusters = usePlotStore((state) => state.resetVisibleClusters)
  const setSelectedCluster = usePlotStore((state) => state.setSelectedCluster)
  const selectedCluster = usePlotStore((state) => state.selectedCluster)
  const annotations = usePlotStore((state) => state.annotations)
  const plotJobId = usePlotStore((state) => state.plotJobId)
  const drillPath = usePlotStore((state) => state.drillPath)
  const drillIntoCluster = usePlotStore((state) => state.drillIntoCluster)
  const drillIntoSubCluster = usePlotStore((state) => state.drillIntoSubCluster)
  const setIsLoadingDrill = usePlotStore((state) => state.setIsLoadingDrill)
  const isLoadingDrill = usePlotStore((state) => state.isLoadingDrill)

  const handleDrillCluster = useCallback(
    async (clusterIndex: number) => {
      if (!plotJobId || isLoadingDrill) return
      setIsLoadingDrill(true)
      try {
        const data = await subCluster(plotJobId, clusterIndex, {
          num_sub_clusters: 4,
        })
        drillIntoCluster(clusterIndex, data)
      } catch {
        setIsLoadingDrill(false)
      }
    },
    [plotJobId, isLoadingDrill, setIsLoadingDrill, drillIntoCluster],
  )

  const handleClickCluster = useCallback(
    (clusterIndex: number, isSelected: boolean) => {
      setSelectedCluster(isSelected ? null : clusterIndex)
    },
    [setSelectedCluster],
  )

  const handleDrillSubCluster = useCallback(
    async (subClusterIndex: number) => {
      if (!plotJobId || isLoadingDrill) return
      const currentLevel = drillPath[drillPath.length - 1]
      if (!currentLevel) return

      const pointIds = currentLevel.subClusterData.points
        .filter((p) => p.sub_cluster === subClusterIndex)
        .map((p) => p.id)

      if (pointIds.length < 4) return // Not enough points to sub-cluster

      setIsLoadingDrill(true)
      try {
        const data = await subClusterByPointIds(plotJobId, {
          num_sub_clusters: 4,
          point_ids: pointIds,
        })
        drillIntoSubCluster(subClusterIndex, data)
      } catch {
        setIsLoadingDrill(false)
      }
    },
    [plotJobId, isLoadingDrill, drillPath, setIsLoadingDrill, drillIntoSubCluster],
  )

  if (!plotData) return null

  const handleShowAll = () => resetVisibleClusters(plotData.clusters.length)

  // When drilled in, show sub-cluster entries
  const isDrilled = drillPath.length > 0
  const currentLevel = isDrilled ? drillPath[drillPath.length - 1] : null
  const subClusters = currentLevel?.subClusterData.sub_clusters

  return (
    <div
      className="bg-white p-4 border-t border-gray-200 overflow-x-auto relative z-50 pointer-events-auto"
      onPointerDown={(event) => event.stopPropagation()}
      data-testid="cluster-legend"
    >
      <div className="flex items-center space-x-4 mb-2">
        <h3 className="text-sm font-bold text-gray-700">
          {isDrilled ? 'Sub-Clusters' : 'Clusters'}
        </h3>
        {!isDrilled && (
          <button
            onClick={handleShowAll}
            className="text-xs text-blue-600 hover:text-blue-800"
          >
            Show All
          </button>
        )}
        {isLoadingDrill && (
          <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600" />
        )}
      </div>
      <div className="flex space-x-4 min-w-max pb-2">
        {isDrilled && subClusters
          ? subClusters.map((sc) => {
              const color = CLUSTER_COLORS[sc.index % CLUSTER_COLORS.length]
              const canDrill = sc.count >= 4
              return (
                <div
                  key={sc.index}
                  className="flex items-center space-x-1 px-2 py-1 rounded border border-gray-300 bg-gray-50 transition-colors"
                >
                  <span
                    data-testid={`subcluster-legend-name-${sc.index}`}
                    className="flex items-center space-x-2 px-1 py-0.5"
                  >
                    <span
                      className="w-3 h-3 rounded-full shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    <div className="text-xs text-left">
                      <div className="font-medium text-gray-900">
                        Sub {sc.index}
                      </div>
                      <div className="text-gray-500 text-[10px]">
                        {sc.count} points
                      </div>
                    </div>
                  </span>
                  {canDrill && (
                    <button
                      data-testid={`subcluster-drill-${sc.index}`}
                      onClick={() => handleDrillSubCluster(sc.index)}
                      className="p-0.5 rounded hover:bg-blue-100 transition-colors shrink-0 text-gray-400 hover:text-blue-600"
                      title="Drill into sub-cluster"
                      aria-label={`Drill into Sub ${sc.index}`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="6 9 12 15 18 9" />
                      </svg>
                    </button>
                  )}
                </div>
              )
            })
          : plotData.clusters.map((cluster) => {
              const isVisible = visibleClusters.has(cluster.index)
              const color = CLUSTER_COLORS[cluster.index % CLUSTER_COLORS.length]
              const isSelected = selectedCluster === cluster.index
              const annotation = annotations?.clusters?.[String(cluster.index)]
              const hasAnnotation = Boolean(
                annotation?.notes || (annotation?.tags && annotation.tags.length > 0)
              )

              return (
                <div
                  key={cluster.index}
                  className={`flex items-center space-x-1 px-2 py-1 rounded border ${
                    isSelected
                      ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-300'
                      : isVisible
                        ? 'border-gray-300 bg-gray-50'
                        : 'border-gray-200 bg-gray-100 opacity-60'
                  } transition-colors`}
                >
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      if (e.metaKey || e.ctrlKey) {
                        isolateCluster(cluster.index)
                      } else {
                        toggleCluster(cluster.index)
                      }
                    }}
                    className="p-0.5 rounded hover:bg-gray-200 transition-colors shrink-0"
                    title={isVisible ? 'Hide cluster (Ctrl/Cmd+click to show only this)' : 'Show cluster (Ctrl/Cmd+click to show only this)'}
                    aria-label={isVisible ? `Hide ${cluster.name}` : `Show ${cluster.name}`}
                  >
                    {isVisible ? (
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                    ) : (
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-400">
                        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                        <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                        <line x1="1" y1="1" x2="23" y2="23" />
                      </svg>
                    )}
                  </button>

                   <button
                     data-testid={`cluster-legend-name-${cluster.index}`}
                     onClick={() => handleClickCluster(cluster.index, isSelected)}
                    className="flex items-center space-x-2 hover:bg-gray-100 rounded px-1 py-0.5 transition-colors"
                    title="Click for details"
                  >
                    <span
                      className="w-3 h-3 rounded-full shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    <div className="text-xs text-left">
                      <div className={`font-medium ${isVisible ? 'text-gray-900' : 'text-gray-500 line-through'}`}>
                        {annotation?.name || cluster.name}
                      </div>
                      <div className="text-gray-500 text-[10px]">{cluster.count} points</div>
                    </div>
                    {hasAnnotation && (
                      <span
                        className="w-1.5 h-1.5 rounded-full bg-blue-500 shrink-0"
                        title="Has annotations"
                      />
                    )}
                  </button>
                  <button
                    data-testid={`cluster-drill-${cluster.index}`}
                    onClick={() => handleDrillCluster(cluster.index)}
                    className="p-0.5 rounded hover:bg-blue-100 transition-colors shrink-0 text-gray-400 hover:text-blue-600"
                    title="Drill into sub-clusters"
                    aria-label={`Drill into ${annotation?.name || cluster.name}`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="6 9 12 15 18 9" />
                    </svg>
                  </button>
                </div>
              )
            })}
      </div>
    </div>
  )
}
