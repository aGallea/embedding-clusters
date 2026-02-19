import { useNavigate } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import type { CollectionInfo } from '../../types'
import { deleteCollection } from '../../api/collections'

interface CollectionListProps {
  collections: CollectionInfo[]
}

export default function CollectionList({ collections }: CollectionListProps) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const deleteMutation = useMutation({
    mutationFn: deleteCollection,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collections'] })
    },
  })

  const handleDelete = (name: string) => {
    if (window.confirm(`Are you sure you want to delete collection "${name}"?`)) {
      deleteMutation.mutate(name)
    }
  }

  return (
    <div className="overflow-x-auto shadow-md sm:rounded-lg bg-white border border-gray-200">
      <table className="min-w-full text-sm text-left text-gray-500">
        <thead className="text-xs text-gray-700 uppercase bg-gray-50 border-b border-gray-200">
          <tr>
            <th scope="col" className="px-6 py-3 font-semibold">
              Name
            </th>
            <th scope="col" className="px-6 py-3 font-semibold">
              Item Count
            </th>
            <th scope="col" className="px-6 py-3 font-semibold">
              Actions
            </th>
          </tr>
        </thead>
        <tbody>
          {collections.map((collection, index) => (
            <tr
              key={collection.name}
              className={`border-b border-gray-100 hover:bg-gray-50 transition-colors ${
                index % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'
              }`}
            >
              <td className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap">
                {collection.name}
              </td>
              <td className="px-6 py-4 font-mono text-gray-600">
                {collection.count.toLocaleString()}
              </td>
              <td className="px-6 py-4 space-x-2">
                <button
                  onClick={() => navigate(`/plot?collection=${collection.name}`)}
                  className="font-medium text-blue-600 dark:text-blue-500 hover:underline hover:text-blue-800 transition-colors"
                >
                  Visualize
                </button>
                <span className="text-gray-300">|</span>
                <button
                  onClick={() => handleDelete(collection.name)}
                  className="font-medium text-red-600 dark:text-red-500 hover:underline hover:text-red-800 transition-colors"
                  disabled={deleteMutation.isPending}
                >
                  {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
