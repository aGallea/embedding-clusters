import { Link } from 'react-router-dom'

export default function EmptyState() {
  return (
    <div className="text-center py-20 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        className="h-16 w-16 mx-auto mb-4 text-gray-400 opacity-50"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1}
          d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
        />
      </svg>
      <p className="text-gray-500 text-xl">No collections yet</p>
      <p className="text-gray-400 mt-2">
        Get started by indexing your first CSV
      </p>
      <Link
        to="/index"
        className="mt-6 inline-flex items-center rounded-md border border-transparent bg-blue-600 px-6 py-3 text-base font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
      >
        Index New Data
      </Link>
    </div>
  )
}
