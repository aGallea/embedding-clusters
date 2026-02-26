import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchCollection, deleteCollection } from '../../api/collections'
import type { CollectionInfo } from '../../types'

interface CollectionCardProps {
  collection: CollectionInfo
}

export default function CollectionCard({ collection }: CollectionCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: detail, isLoading: isDetailLoading } = useQuery({
    queryKey: ['collection', collection.name],
    queryFn: () => fetchCollection(collection.name),
    enabled: isExpanded,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteCollection,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collections'] })
    },
  })

  const handleDelete = () => {
    if (
      window.confirm(
        `Are you sure you want to delete collection "${collection.name}"?`,
      )
    ) {
      deleteMutation.mutate(collection.name)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200 hover:shadow-lg transition-shadow">
      <div className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 truncate">
          {collection.name}
        </h3>
        <p className="text-2xl font-bold text-gray-700 mt-2">
          {collection.count.toLocaleString()}
          <span className="text-sm font-normal text-gray-500 ml-1">items</span>
        </p>
        {(collection.model_type || collection.model_name) && (
          <div className="mt-2 flex flex-wrap gap-2">
            {collection.model_type && (
              <span className="inline-flex items-center rounded-full bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-600/20">
                {collection.model_type}
              </span>
            )}
            {collection.model_name && (
              <span className="inline-flex items-center rounded-full bg-gray-50 px-2 py-1 text-xs font-medium text-gray-600 ring-1 ring-inset ring-gray-500/10 truncate max-w-full">
                {collection.model_name}
              </span>
            )}
          </div>
        )}
      </div>

      {isExpanded && (
        <div className="border-t border-gray-100 px-6 py-4 bg-gray-50">
          {isDetailLoading ? (
            <div className="flex items-center text-sm text-gray-500">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
              Loading details...
            </div>
          ) : detail ? (
            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">
                Metadata Fields
              </p>
              {detail.metadata_fields.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {detail.metadata_fields.map((field) => (
                    <span
                      key={field}
                      className="inline-flex items-center rounded bg-white px-2 py-0.5 text-xs text-gray-700 border border-gray-200"
                    >
                      {field}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-400">No metadata fields</p>
              )}
            </div>
          ) : null}
        </div>
      )}

      <div className="border-t border-gray-100 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <button
            onClick={() => navigate(`/plot?collection=${collection.name}`)}
            className="text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors"
          >
            Visualize
          </button>
          <span className="text-gray-300">|</span>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors"
          >
            {isExpanded ? 'Collapse' : 'Inspect'}
          </button>
        </div>
        <button
          onClick={handleDelete}
          className="text-sm font-medium text-red-600 hover:text-red-800 transition-colors"
          disabled={deleteMutation.isPending}
        >
          {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
        </button>
      </div>
    </div>
  )
}
