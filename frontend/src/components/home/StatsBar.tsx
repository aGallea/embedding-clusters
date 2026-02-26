interface StatsBarProps {
  collectionCount: number
  totalItems: number
}

export default function StatsBar({
  collectionCount,
  totalItems,
}: StatsBarProps) {
  return (
    <div className="grid grid-cols-2 gap-4 mb-8">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <p className="text-sm font-medium text-gray-500">Collections</p>
        <p className="text-3xl font-bold text-gray-900 mt-1">
          {collectionCount.toLocaleString()}
        </p>
      </div>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <p className="text-sm font-medium text-gray-500">Total Items</p>
        <p className="text-3xl font-bold text-gray-900 mt-1">
          {totalItems.toLocaleString()}
        </p>
      </div>
    </div>
  )
}
