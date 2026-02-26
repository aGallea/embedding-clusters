import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { fetchCollections } from '../api/collections'
import StatsBar from '../components/home/StatsBar'
import CollectionGrid from '../components/home/CollectionGrid'
import EmptyState from '../components/home/EmptyState'

export default function HomePage() {
  const { data: collections, isLoading, isError } = useQuery({
    queryKey: ['collections'],
    queryFn: fetchCollections,
  })

  const totalItems = collections?.reduce((sum, c) => sum + c.count, 0) ?? 0

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6 border-b pb-4 border-gray-200">
        <h1 className="text-3xl font-bold mb-2 text-gray-900">Dashboard</h1>
        <p className="text-gray-500 text-lg">
          Manage your embedding collections
        </p>
      </div>

      {isLoading && (
        <div className="flex justify-center items-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
          <span className="ml-3 text-gray-500 text-lg">
            Loading collections...
          </span>
        </div>
      )}

      {isError && (
        <div
          className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative"
          role="alert"
        >
          <strong className="font-bold">Error!</strong>
          <span className="block sm:inline">
            {' '}
            Failed to fetch collections. Please make sure the backend is
            running.
          </span>
        </div>
      )}

      {!isLoading && !isError && collections && (
        <>
          <StatsBar
            collectionCount={collections.length}
            totalItems={totalItems}
          />

          <Link
            to="/index"
            className="mb-8 flex items-center justify-center rounded-lg border-2 border-dashed border-blue-300 bg-blue-50 p-6 text-blue-700 hover:bg-blue-100 hover:border-blue-400 transition-colors group"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6 mr-2 group-hover:scale-110 transition-transform"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            <span className="text-lg font-medium">Index New Data</span>
          </Link>

          {collections.length === 0 ? (
            <EmptyState />
          ) : (
            <CollectionGrid collections={collections} />
          )}
        </>
      )}
    </div>
  )
}
