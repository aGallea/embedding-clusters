import { usePlotStore } from '../../stores/plotStore'

export default function SearchResults() {
  const searchResults = usePlotStore((state) => state.searchResults)
  const highlightedIds = usePlotStore((state) => state.highlightedIds)
  const setHighlightedIds = usePlotStore((state) => state.setHighlightedIds)
  const setHoveredPointId = usePlotStore((state) => state.setHoveredPointId)

  if (!searchResults || searchResults.length === 0) return null

  const handleClickResult = (id: string) => {
    setHighlightedIds(new Set([id]))
  }

  const handleShowAll = () => {
    setHighlightedIds(new Set(searchResults.map((r) => r.id)))
  }

  const getImageUrl = (metadata: Record<string, unknown>): string | null => {
    for (const value of Object.values(metadata)) {
      if (typeof value === 'string' && (value.startsWith('http') || value.startsWith('/'))) {
        return value
      }
    }
    return null
  }

  return (
    <div className="border-t border-gray-200 bg-white overflow-y-auto max-h-64">
      <div className="flex items-center justify-between p-3 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-700">
          Results ({searchResults.length})
        </h3>
        <button
          onClick={handleShowAll}
          className="text-xs text-blue-600 hover:text-blue-700"
        >
          Highlight All
        </button>
      </div>
      <div className="divide-y divide-gray-100">
        {searchResults.map((result) => {
          const imageUrl = getImageUrl(result.metadata)
          const isActive = highlightedIds.has(result.id)
          return (
            <button
              key={result.id}
              onClick={() => handleClickResult(result.id)}
              onMouseEnter={() => setHoveredPointId(result.id)}
              onMouseLeave={() => setHoveredPointId(null)}
              className={`w-full text-left p-3 hover:bg-blue-50 transition-colors flex items-center gap-3 ${
                isActive ? 'bg-blue-50 border-l-2 border-blue-500' : ''
              }`}
            >
              {imageUrl && (
                <img
                  src={imageUrl}
                  alt=""
                  className="w-10 h-10 object-cover rounded flex-shrink-0"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-gray-900 truncate">
                  {result.id}
                </p>
                <div className="text-xs text-gray-500 truncate">
                  {Object.entries(result.metadata)
                    .filter(([, v]) => typeof v === 'string' && !String(v).startsWith('http'))
                    .slice(0, 2)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(' | ')}
                </div>
              </div>
              <span className="text-xs text-gray-400 flex-shrink-0">
                {result.distance.toFixed(3)}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
