import PlotControls from '../components/plot/PlotControls'
import ScatterPlot from '../components/plot/ScatterPlot'
import ClusterLegend from '../components/plot/ClusterLegend'
import { usePlotData } from '../hooks/usePlotData'
import { usePlotStore } from '../stores/plotStore'

export default function PlotPage() {
  const { compute, isComputing, error } = usePlotData()
  const plotData = usePlotStore((state) => state.plotData)

  return (
    <div className="flex flex-col h-screen max-h-[calc(100vh-64px)] overflow-hidden bg-gray-50">
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-80 border-r border-gray-200 bg-white overflow-y-auto shrink-0 z-10 shadow-sm">
          <PlotControls onCompute={compute} isComputing={isComputing} />
        </div>

        {/* Main Area */}
        <div className="flex-1 flex flex-col relative bg-gray-100">
          {error && (
            <div className="absolute top-4 right-4 z-50 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded shadow-lg max-w-md">
              <p className="font-bold">Error</p>
              <p className="text-sm">{(error as Error).message}</p>
            </div>
          )}

          {/* Computing Overlay */}
          {isComputing && (
            <div className="absolute inset-0 z-40 bg-white/50 flex items-center justify-center backdrop-blur-sm">
              <div className="bg-white p-6 rounded-lg shadow-xl text-center border border-gray-200">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <h3 className="text-lg font-medium text-gray-900">Computing Clusters...</h3>
                <p className="text-gray-500 text-sm mt-1">This may take a few moments</p>
              </div>
            </div>
          )}

          {/* Plot Content */}
          <div className="flex-1 p-4 overflow-hidden relative">
            {plotData ? (
              <ScatterPlot />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-400 border-2 border-dashed border-gray-300 rounded-lg">
                <div className="text-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                  <h3 className="text-lg font-medium text-gray-900">No Data Visualized</h3>
                  <p className="mt-1">Select a collection and click Compute to generate a plot.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Legend Footer */}
      {plotData && (
        <div className="border-t border-gray-200 bg-white z-20 shrink-0">
          <ClusterLegend />
        </div>
      )}
    </div>
  )
}
