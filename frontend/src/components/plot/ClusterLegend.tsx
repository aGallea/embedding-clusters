import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'

export default function ClusterLegend() {
  const plotData = usePlotStore((state) => state.plotData)
  const visibleClusters = usePlotStore((state) => state.visibleClusters)
  const toggleCluster = usePlotStore((state) => state.toggleCluster)
  const resetVisibleClusters = usePlotStore((state) => state.resetVisibleClusters)
  const setSelectedCluster = usePlotStore((state) => state.setSelectedCluster)
  const selectedCluster = usePlotStore((state) => state.selectedCluster)
  const annotations = usePlotStore((state) => state.annotations)

  if (!plotData) return null

  const handleShowAll = () => resetVisibleClusters(plotData.clusters.length)

  return (
    <div
      className="bg-white p-4 border-t border-gray-200 overflow-x-auto relative z-50 pointer-events-auto"
      onPointerDown={(event) => event.stopPropagation()}
      data-testid="cluster-legend"
    >
      <div className="flex items-center space-x-4 mb-2">
        <h3 className="text-sm font-bold text-gray-700">Clusters</h3>
        <button
          onClick={handleShowAll}
          className="text-xs text-blue-600 hover:text-blue-800"
        >
          Show All
        </button>
      </div>
      <div className="flex space-x-4 min-w-max pb-2">
        {plotData.clusters.map((cluster) => {
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
                  toggleCluster(cluster.index)
                }}
                className="p-0.5 rounded hover:bg-gray-200 transition-colors shrink-0"
                title={isVisible ? 'Hide cluster' : 'Show cluster'}
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
                 onClick={() => setSelectedCluster(
                   isSelected ? null : cluster.index
                 )}
                className="flex items-center space-x-2 hover:bg-gray-100 rounded px-1 py-0.5 transition-colors"
                title="View cluster details"
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
