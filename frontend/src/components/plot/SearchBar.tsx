import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { searchCollection } from '../../api/plot'
import { usePlotStore } from '../../stores/plotStore'
import type { SearchRequest } from '../../types'

interface SearchBarProps {
  collectionName: string
}

export default function SearchBar({ collectionName }: SearchBarProps) {
  const [queryType, setQueryType] = useState<'text' | 'image'>('text')
  const [queryValue, setQueryValue] = useState('')
  const [nResults, setNResults] = useState(10)
  const { setSearchResults, setIsSearching, clearSearch, setQueryPoint } = usePlotStore()

  const mutation = useMutation({
    mutationFn: (request: SearchRequest) => searchCollection(request),
    onMutate: () => setIsSearching(true),
    onSuccess: (data) => {
      setSearchResults(data.results)
      setIsSearching(false)

      // Compute query point position as weighted centroid of results
      const plotData = usePlotStore.getState().plotData
      if (plotData && data.results.length > 0) {
        const EPSILON = 1e-6
        let totalWeight = 0
        let wx = 0, wy = 0, wz = 0
        for (const result of data.results) {
          const point = plotData.points.find((p) => p.id === result.id)
          if (point) {
            const weight = 1 / (result.distance + EPSILON)
            wx += point.x * weight
            wy += point.y * weight
            wz += point.z * weight
            totalWeight += weight
          }
        }
        if (totalWeight > 0) {
          setQueryPoint({
            x: wx / totalWeight,
            y: wy / totalWeight,
            z: wz / totalWeight,
          })
        }
      }
    },
    onError: () => setIsSearching(false),
  })

  const handleSearch = () => {
    if (!queryValue.trim()) return
    const request: SearchRequest = {
      collection_name: collectionName,
      n_results: nResults,
      ...(queryType === 'text'
        ? { query_text: queryValue }
        : { query_image_url: queryValue, model_type: 'image' }),
    }
    mutation.mutate(request)
  }

  const handleClear = () => {
    setQueryValue('')
    clearSearch()
  }

  return (
    <div className="space-y-2 p-3">
      <h3 className="text-xs font-semibold text-gray-700">Semantic Search</h3>

      <div className="flex space-x-2">
        {(['text', 'image'] as const).map((type) => (
          <label key={type} className="flex items-center text-xs cursor-pointer">
            <input
              type="radio"
              name="queryType"
              value={type}
              checked={queryType === type}
              onChange={() => setQueryType(type)}
              className="mr-1"
            />
            {type === 'text' ? 'Text' : 'Image URL'}
          </label>
        ))}
      </div>

      <input
        type="text"
        value={queryValue}
        onChange={(e) => setQueryValue(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        placeholder={queryType === 'text' ? 'Search by text...' : 'Paste image URL...'}
        className="w-full border border-gray-300 rounded-md px-2 py-1.5 text-sm"
      />

      <div className="space-y-1">
        <label className="block text-xs text-gray-500">
          Results: {nResults}
        </label>
        <input
          type="range"
          min="5"
          max="50"
          value={nResults}
          onChange={(e) => setNResults(Number(e.target.value))}
          className="w-full"
        />
      </div>

      <div className="flex space-x-2">
        <button
          onClick={handleSearch}
          disabled={mutation.isPending || !queryValue.trim()}
          className={`flex-1 py-1.5 px-3 rounded text-white text-sm font-medium ${
            mutation.isPending || !queryValue.trim()
              ? 'bg-green-300 cursor-not-allowed'
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          {mutation.isPending ? 'Searching...' : 'Search'}
        </button>
        <button
          onClick={handleClear}
          className="py-1.5 px-3 rounded border border-gray-300 text-sm text-gray-600 hover:bg-gray-50"
        >
          Clear
        </button>
      </div>

      {mutation.isError && (
        <p className="text-xs text-red-500">
          Search failed: {(mutation.error as Error).message}
        </p>
      )}
    </div>
  )
}
