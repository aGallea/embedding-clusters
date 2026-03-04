import { useRef, useState, useCallback, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import PlotControls from '../components/plot/PlotControls'
import ScatterPlot from '../components/plot/ScatterPlot'
import ClusterLegend from '../components/plot/ClusterLegend'
import SearchBar from '../components/plot/SearchBar'
import SearchResults from '../components/plot/SearchResults'
import ClusterDetailDrawer from '../components/plot/ClusterDetailDrawer'

import { usePlotData } from '../hooks/usePlotData'
import { usePlotStore } from '../stores/plotStore'

export default function PlotPage() {
  const { compute, isComputing, error, jobId } = usePlotData()
  const plotData = usePlotStore((state) => state.plotData)
  const selectedCluster = usePlotStore((state) => state.selectedCluster)
  const imageField = usePlotStore((state) => state.imageField)
  const plotContainerRef = useRef<HTMLDivElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [searchParams] = useSearchParams()
  const collectionName = searchParams.get('collection') ?? ''

  const toggleFullscreen = useCallback(() => {
    if (!plotContainerRef.current) return

    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      plotContainerRef.current.requestFullscreen()
      setIsFullscreen(true)
    }
  }, [])

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement))
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  return (
    <div className="flex flex-col h-screen max-h-[calc(100vh-64px)] overflow-hidden bg-gray-50">
      <div className="flex flex-1 overflow-hidden">
        <div
          data-testid="plot-sidebar"
          className="w-80 border-r border-gray-200 bg-white overflow-y-auto shrink-0 z-10 shadow-sm"
        >
          {plotData && collectionName && (
            <>
              <div className="px-3 py-2 border-b border-gray-200">
                <SearchBar collectionName={collectionName} />
              </div>
              <SearchResults />
            </>
          )}
          <PlotControls onCompute={compute} isComputing={isComputing} />
        </div>

        <div className="flex-1 flex flex-col relative bg-gray-100">
          {error && (
            <div className="absolute top-4 right-4 z-50 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded shadow-lg max-w-md">
              <p className="font-bold">Error</p>
              <p className="text-sm">{(error as Error).message}</p>
            </div>
          )}

          {isComputing && (
            <div className="absolute inset-0 z-40 bg-white/50 flex items-center justify-center backdrop-blur-sm">
              <div className="bg-white p-6 rounded-lg shadow-xl text-center border border-gray-200">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <h3 className="text-lg font-medium text-gray-900">Computing Clusters...</h3>
                <p className="text-gray-500 text-sm mt-1">This may take a few moments</p>
              </div>
            </div>
          )}

          <div ref={plotContainerRef} className="flex-1 p-4 overflow-hidden relative">
            {plotData ? (
              <>
                <ScatterPlot />
                <button
                  onClick={toggleFullscreen}
                  className="absolute top-6 right-14 z-10 p-2 bg-gray-800/50 hover:bg-gray-700/80 text-white/80 hover:text-white rounded-md backdrop-blur-sm transition-all shadow-lg border border-white/10"
                  title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
                  aria-label={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
                >
                  {isFullscreen ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="4 14 10 14 10 20" />
                      <polyline points="20 10 14 10 14 4" />
                      <line x1="14" y1="10" x2="21" y2="3" />
                      <line x1="3" y1="21" x2="10" y2="14" />
                    </svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="15 3 21 3 21 9" />
                      <polyline points="9 21 3 21 3 15" />
                      <line x1="21" y1="3" x2="14" y2="10" />
                      <line x1="3" y1="21" x2="10" y2="14" />
                    </svg>
                  )}
                </button>

                {selectedCluster !== null && jobId && (
                  <ClusterDetailDrawer
                    jobId={jobId}
                    imageField={imageField ?? undefined}
                  />
                )}
              </>
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

      {plotData && (
        <div className="border-t border-gray-200 bg-white z-20 shrink-0">
          <ClusterLegend />
        </div>
      )}
    </div>
  )
}
