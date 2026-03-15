import type { ClusterItemResponse } from '../../types'

interface SelectedPointsDistancePanelProps {
  selectedItems: ClusterItemResponse[]
}

interface DistanceRow {
  key: string
  leftId: string
  rightId: string
  distance: number
}

function buildDistanceRows(selectedItems: ClusterItemResponse[]): DistanceRow[] {
  const rows: DistanceRow[] = []

  for (let i = 0; i < selectedItems.length; i += 1) {
    for (let j = i + 1; j < selectedItems.length; j += 1) {
      const left = selectedItems[i]
      const right = selectedItems[j]
      rows.push({
        key: `${left.id}-${right.id}`,
        leftId: left.id,
        rightId: right.id,
        distance: Math.abs(left.distance_to_centroid - right.distance_to_centroid),
      })
    }
  }

  return rows
}

export default function SelectedPointsDistancePanel({ selectedItems }: SelectedPointsDistancePanelProps) {
  if (selectedItems.length < 2) {
    return null
  }

  const rows = buildDistanceRows(selectedItems)

  return (
    <div className="px-4 py-3 border-t border-gray-200 space-y-2 shrink-0">
      <h3 className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">
        Selected distances
      </h3>

      <div className="space-y-2 max-h-28 overflow-y-auto">
        {rows.map((row) => (
          <div
            key={row.key}
            data-testid="selected-distance-row"
            className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5 text-[10px] text-gray-700"
          >
            <div className="font-medium text-gray-900">
              {row.leftId} vs {row.rightId}
            </div>
            <div>
              centroid distance delta: {row.distance.toFixed(4)}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
