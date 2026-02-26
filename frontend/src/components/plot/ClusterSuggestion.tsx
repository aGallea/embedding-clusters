import { useState } from 'react'
import { suggestClusters } from '../../api/plot'
import type { SuggestClustersResponse } from '../../types'

interface ClusterSuggestionProps {
  collectionName: string
  onApply: (k: number) => void
}

function SuggestionChart({ data }: { data: SuggestClustersResponse }) {
  const { k_values, inertias, silhouette_scores, suggested_k } = data
  const width = 280
  const height = 140
  const padding = { top: 10, right: 35, bottom: 25, left: 40 }
  const chartW = width - padding.left - padding.right
  const chartH = height - padding.top - padding.bottom

  const maxInertia = Math.max(...inertias)
  const minInertia = Math.min(...inertias)
  const inertiaRange = maxInertia - minInertia || 1

  const maxSil = Math.max(...silhouette_scores)
  const minSil = Math.min(...silhouette_scores)
  const silRange = maxSil - minSil || 1

  const xScale = (i: number) => padding.left + (i / (k_values.length - 1 || 1)) * chartW
  const yInertia = (v: number) => padding.top + (1 - (v - minInertia) / inertiaRange) * chartH
  const ySil = (v: number) => padding.top + (1 - (v - minSil) / silRange) * chartH

  const inertiaPath = inertias
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i)},${yInertia(v)}`)
    .join(' ')

  const barWidth = Math.max(4, chartW / k_values.length - 2)

  return (
    <svg width={width} height={height} className="text-xs">
      {/* Silhouette bars */}
      {silhouette_scores.map((s, i) => (
        <rect
          key={k_values[i]}
          x={xScale(i) - barWidth / 2}
          y={ySil(s)}
          width={barWidth}
          height={chartH + padding.top - ySil(s)}
          fill={k_values[i] === suggested_k ? '#22c55e' : '#93c5fd'}
          opacity={0.6}
        />
      ))}

      {/* Inertia line */}
      <path d={inertiaPath} fill="none" stroke="#ef4444" strokeWidth={2} />

      {/* Inertia dots */}
      {inertias.map((v, i) => (
        <circle key={`i-${k_values[i]}`} cx={xScale(i)} cy={yInertia(v)} r={2.5} fill="#ef4444" />
      ))}

      {/* X axis labels */}
      {k_values.map((k, i) =>
        k_values.length <= 15 || i % Math.ceil(k_values.length / 10) === 0 ? (
          <text
            key={`x-${k}`}
            x={xScale(i)}
            y={height - 3}
            textAnchor="middle"
            className="fill-gray-500"
            fontSize={9}
          >
            {k}
          </text>
        ) : null,
      )}

      {/* Y axis labels */}
      <text x={2} y={padding.top + 4} fontSize={8} className="fill-red-500">
        Inertia
      </text>
      <text x={width - 2} y={padding.top + 4} fontSize={8} textAnchor="end" className="fill-blue-500">
        Silhouette
      </text>
    </svg>
  )
}

export default function ClusterSuggestion({ collectionName, onApply }: ClusterSuggestionProps) {
  const [data, setData] = useState<SuggestClustersResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSuggest = async () => {
    setIsLoading(true)
    setError(null)
    setData(null)
    try {
      const result = await suggestClusters({ collection_name: collectionName })
      setData(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to suggest clusters')
    } finally {
      setIsLoading(false)
    }
  }

  if (!collectionName) return null

  return (
    <div className="space-y-2">
      <button
        onClick={handleSuggest}
        disabled={isLoading}
        className={`text-xs px-3 py-1 rounded border ${
          isLoading
            ? 'bg-gray-100 text-gray-400 cursor-not-allowed border-gray-200'
            : 'bg-white text-blue-600 hover:bg-blue-50 border-blue-300'
        }`}
      >
        {isLoading ? 'Analyzing...' : 'Suggest'}
      </button>

      {error && <p className="text-xs text-red-500">{error}</p>}

      {data && (
        <div className="bg-gray-50 rounded border border-gray-200 p-2 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">
              Recommended: <strong className="text-green-600 text-sm">{data.suggested_k}</strong> clusters
            </span>
            <button
              onClick={() => onApply(data.suggested_k)}
              className="text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Apply
            </button>
          </div>
          <SuggestionChart data={data} />
          <div className="flex justify-between text-[10px] text-gray-400">
            <span className="flex items-center gap-1">
              <span className="inline-block w-2 h-0.5 bg-red-500" /> Inertia (elbow)
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-2 h-2 bg-blue-300 opacity-60" /> Silhouette
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
