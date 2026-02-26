import type { CollectionInfo } from '../../types'
import CollectionCard from './CollectionCard'

interface CollectionGridProps {
  collections: CollectionInfo[]
}

export default function CollectionGrid({
  collections,
}: CollectionGridProps) {
  return (
    <div>
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Collections
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {collections.map((collection) => (
          <CollectionCard key={collection.name} collection={collection} />
        ))}
      </div>
    </div>
  )
}
