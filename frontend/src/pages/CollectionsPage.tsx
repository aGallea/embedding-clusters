import { useQuery } from '@tanstack/react-query'
import { fetchCollections } from '../api/collections'
import CollectionList from '../components/collections/CollectionList'

export default function CollectionsPage() {
  const { data: collections, isLoading, isError } = useQuery({
    queryKey: ['collections'],
    queryFn: fetchCollections,
  })

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6 border-b pb-4 border-gray-200">
        <h1 className="text-3xl font-bold mb-2 text-gray-900">Collections</h1>
        <p className="text-gray-500 text-lg">Manage your ChromaDB collections</p>
      </div>

      {isLoading && (
        <div className="flex justify-center items-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
          <span className="ml-3 text-gray-500 text-lg">Loading collections...</span>
        </div>
      )}

      {isError && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">Error!</strong>
          <span className="block sm:inline"> Failed to fetch collections. Please make sure the backend is running.</span>
        </div>
      )}

      {!isLoading && !isError && collections && collections.length === 0 && (
        <div className="text-center py-20 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
          <p className="text-gray-500 text-xl">No collections found</p>
          <p className="text-gray-400 mt-2">Index some data to get started</p>
        </div>
      )}

      {!isLoading && !isError && collections && collections.length > 0 && (
        <CollectionList collections={collections} />
      )}
    </div>
  )
}
