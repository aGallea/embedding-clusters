interface CsvPreviewProps {
  columns: string[];
  rows: Record<string, string>[];
  totalRows: number;
}

export default function CsvPreview({ columns, rows, totalRows }: CsvPreviewProps) {
  if (!rows || rows.length === 0) {
    return (
      <div className="text-center p-8 text-gray-500 bg-gray-50 rounded-lg border border-gray-200">
        No data available to preview.
      </div>
    );
  }

  return (
    <div className="mt-8">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium text-gray-900">Data Preview</h3>
        <span className="text-sm text-gray-500">
          Showing {rows.length} of {totalRows} total rows
        </span>
      </div>

      <div className="overflow-x-auto shadow-md sm:rounded-lg bg-white border border-gray-200">
        <table className="min-w-full text-sm text-left text-gray-500">
          <thead className="text-xs text-gray-700 uppercase bg-gray-50 border-b border-gray-200">
            <tr>
              {columns.map((column) => (
                <th key={column} className="px-6 py-3 font-semibold whitespace-nowrap">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr
                key={index}
                className={`border-b border-gray-100 hover:bg-gray-50 transition-colors ${
                  index % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'
                }`}
              >
                {columns.map((column) => (
                  <td key={`${index}-${column}`} className="px-6 py-4 whitespace-nowrap">
                    {row[column] || ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
