import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'

export default function ClusterLegend() {
  const plotData = usePlotStore((state) => state.plotData)
  const visibleClusters = usePlotStore((state) => state.visibleClusters)
  const toggleCluster = usePlotStore((state) => state.toggleCluster)
  const resetVisibleClusters = usePlotStore((state) => state.resetVisibleClusters)

  if (!plotData) return null

  const handleShowAll = () => resetVisibleClusters(plotData.clusters.length)

  return (
    <div className="bg-white p-4 border-t border-gray-200 overflow-x-auto">
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

          return (
            <button
              key={cluster.index}
              onClick={() => toggleCluster(cluster.index)}
              className={`flex items-center space-x-2 px-2 py-1 rounded border ${
                isVisible ? 'border-gray-300 bg-gray-50' : 'border-gray-200 bg-gray-100 opacity-60'
              } hover:bg-gray-100 transition-colors`}
            >
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <div className="text-xs text-left">
                <div className={`font-medium ${isVisible ? 'text-gray-900' : 'text-gray-500 line-through'}`}>
                  {cluster.name}
                </div>
                <div className="text-gray-500 text-[10px]">{cluster.count} points</div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
