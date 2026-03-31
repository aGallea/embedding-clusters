import { useCallback, useState } from 'react'
import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'
import { loadAiSettings, nameAiClusters } from '../../api/ai'
import { updateAnnotation, getAnnotations } from '../../api/plot'

export default function ClusterLegend() {
  const plotData = usePlotStore((state) => state.plotData)
  const visibleClusters = usePlotStore((state) => state.visibleClusters)
  const visibleSubClusters = usePlotStore((state) => state.visibleSubClusters)
  const toggleCluster = usePlotStore((state) => state.toggleCluster)
  const toggleSubCluster = usePlotStore((state) => state.toggleSubCluster)
  const isolateCluster = usePlotStore((state) => state.isolateCluster)
  const isolateSubCluster = usePlotStore((state) => state.isolateSubCluster)
  const resetVisibleClusters = usePlotStore((state) => state.resetVisibleClusters)
  const resetVisibleSubClusters = usePlotStore((state) => state.resetVisibleSubClusters)
  const setSelectedCluster = usePlotStore((state) => state.setSelectedCluster)
  const selectedCluster = usePlotStore((state) => state.selectedCluster)
  const annotations = usePlotStore((state) => state.annotations)
  const setAnnotations = usePlotStore((state) => state.setAnnotations)
  const drillPath = usePlotStore((state) => state.drillPath)
  const isLoadingDrill = usePlotStore((state) => state.isLoadingDrill)
  const isNamingClusters = usePlotStore((state) => state.isNamingClusters)
  const setIsNamingClusters = usePlotStore((state) => state.setIsNamingClusters)
  const plotJobId = usePlotStore((state) => state.plotJobId)

  const [namingError, setNamingError] = useState<string | null>(null)

  const handleClickCluster = useCallback(
    (clusterIndex: number, isSelected: boolean) => {
      setSelectedCluster(isSelected ? null : clusterIndex)
    },
    [setSelectedCluster],
  )

  const handleNameWithAi = useCallback(async () => {
    if (!plotData || !plotJobId || isNamingClusters) return

    const settings = loadAiSettings()
    if (!settings.apiKey && settings.provider !== 'ollama') {
      setNamingError('Configure AI settings first (Settings page)')
      return
    }

    setIsNamingClusters(true)
    setNamingError(null)

    try {
      const clusterIndices = plotData.clusters.map((c) => c.index)
      const response = await nameAiClusters({
        job_id: plotJobId,
        cluster_indices: clusterIndices,
        api_key: settings.apiKey,
        model: settings.provider ? `${settings.provider}/${settings.model}` : settings.model,
        base_url: settings.baseUrl || undefined,
        temperature: settings.temperature,
      })

      for (const [indexStr, name] of Object.entries(response.names)) {
        await updateAnnotation(plotJobId, Number(indexStr), { name })
      }

      const updated = await getAnnotations(plotJobId)
      setAnnotations(updated)
    } catch (err) {
      setNamingError(err instanceof Error ? err.message : 'AI naming failed')
    } finally {
      setIsNamingClusters(false)
    }
  }, [plotData, plotJobId, isNamingClusters, setIsNamingClusters, setAnnotations])

  if (!plotData) return null

  const handleShowAll = () => resetVisibleClusters(plotData.clusters.length)
  const handleShowAllSubClusters = () => {
    if (!subClusters) {
      return
    }
    resetVisibleSubClusters(subClusters.map((subCluster) => subCluster.index))
  }

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
        {isDrilled ? (
          <button
            onClick={handleShowAllSubClusters}
            className="text-xs text-blue-600 hover:text-blue-800"
          >
            Show All
          </button>
        ) : (
          <>
            <button
              onClick={handleShowAll}
              className="text-xs text-blue-600 hover:text-blue-800"
            >
              Show All
            </button>
            <button
              onClick={handleNameWithAi}
              disabled={isNamingClusters || !plotJobId}
              className="text-xs text-purple-600 hover:text-purple-800 disabled:opacity-50 flex items-center space-x-1"
              title="Name clusters using AI (configure in Settings)"
              data-testid="name-with-ai-button"
            >
              {isNamingClusters ? (
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-purple-600" />
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2l2.09 6.26L20 10l-5.91 1.74L12 18l-2.09-6.26L4 10l5.91-1.74z" />
                </svg>
              )}
              <span>Name with AI</span>
            </button>
            {namingError && (
              <span className="text-xs text-red-500">{namingError}</span>
            )}
          </>
        )}
        {isLoadingDrill && (
          <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600" />
        )}
      </div>
      <div className="flex space-x-4 min-w-max pb-2">
        {isDrilled && subClusters
          ? subClusters.map((sc) => {
              const color = CLUSTER_COLORS[sc.index % CLUSTER_COLORS.length]
              const isVisible = visibleSubClusters.has(sc.index)
              return (
                <div
                  key={sc.index}
                  className={`flex items-center space-x-1 px-2 py-1 rounded border ${
                    isVisible
                      ? 'border-gray-300 bg-gray-50'
                      : 'border-gray-200 bg-gray-100 opacity-60'
                  } transition-colors`}
                >
                  <button
                    onClick={(event) => {
                      event.stopPropagation()
                      if (event.metaKey || event.ctrlKey) {
                        isolateSubCluster(sc.index)
                      } else {
                        toggleSubCluster(sc.index)
                      }
                    }}
                    className="p-0.5 rounded hover:bg-gray-200 transition-colors shrink-0"
                    title={isVisible ? 'Hide sub-group (Ctrl/Cmd+click to show only this)' : 'Show sub-group (Ctrl/Cmd+click to show only this)'}
                    aria-label={isVisible ? `Hide Sub ${sc.index}` : `Show Sub ${sc.index}`}
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
                  <span
                    data-testid={`subcluster-legend-name-${sc.index}`}
                    className="flex items-center space-x-2 px-1 py-0.5"
                  >
                    <span
                      data-testid={`subcluster-legend-swatch-${sc.index}`}
                      className="w-3 h-3 rounded-full shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    <div className="text-xs text-left">
                      <div className={`font-medium ${isVisible ? 'text-gray-900' : 'text-gray-500 line-through'}`}>
                        {sc.name || `Sub ${sc.index}`}
                      </div>
                      <div className="text-gray-500 text-[10px]">
                        {sc.count} points
                      </div>
                    </div>
                  </span>
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
                </div>
              )
            })}
      </div>
    </div>
  )
}
