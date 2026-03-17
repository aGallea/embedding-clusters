import { useEffect } from 'react'
import { usePlotStore } from '../../stores/plotStore'

export default function DrillBreadcrumb() {
  const drillPath = usePlotStore((state) => state.drillPath)
  const navigateToLevel = usePlotStore((state) => state.navigateToLevel)
  const navigateBack = usePlotStore((state) => state.navigateBack)
  const resetDrill = usePlotStore((state) => state.resetDrill)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && drillPath.length > 0) {
        e.preventDefault()
        navigateBack()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [drillPath.length, navigateBack])

  if (drillPath.length === 0) return null

  return (
    <div
      className="absolute top-4 left-4 z-20 flex items-center space-x-1 bg-gray-800/80 backdrop-blur-sm rounded-lg px-3 py-2 text-sm text-white/90 shadow-lg border border-white/10"
      data-testid="drill-breadcrumb"
    >
      <button
        onClick={resetDrill}
        className="hover:text-white transition-colors font-medium"
        data-testid="breadcrumb-root"
      >
        All Clusters
      </button>

      {drillPath.map((level, index) => (
        <span key={index} className="flex items-center space-x-1">
          <span className="text-white/40">&gt;</span>
          {index === drillPath.length - 1 ? (
            <span className="text-white font-medium" data-testid={`breadcrumb-level-${index}`}>
              {level.label}
            </span>
          ) : (
            <button
              onClick={() => navigateToLevel(index)}
              className="hover:text-white transition-colors"
              data-testid={`breadcrumb-level-${index}`}
            >
              {level.label}
            </button>
          )}
        </span>
      ))}

      <button
        onClick={navigateBack}
        className="ml-2 p-1 hover:bg-white/10 rounded transition-colors"
        title="Go back (Escape)"
        aria-label="Go back one level"
        data-testid="breadcrumb-back"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>
    </div>
  )
}
