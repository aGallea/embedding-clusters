import { useState, useRef, useCallback } from 'react';

interface CsvPreviewProps {
  columns: string[];
  rows: Record<string, string>[];
  totalRows: number;
  previewLimit: number;
  onLimitChange: (limit: number) => void;
}

const IMAGE_URL_PATTERN =
  /^https?:\/\/.+\.(jpg|jpeg|png|gif|webp|svg|bmp|avif)(\?.*)?$/i;

function isImageUrl(value: string): boolean {
  return IMAGE_URL_PATTERN.test(value.trim());
}

interface ImagePreviewState {
  url: string;
  x: number;
  y: number;
}

function ImageCell({ value }: { value: string }) {
  const [preview, setPreview] = useState<ImagePreviewState | null>(null);
  const cellRef = useRef<HTMLTableCellElement>(null);
  const hideTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showPreview = useCallback(
    (e: React.MouseEvent) => {
      if (hideTimeout.current) {
        clearTimeout(hideTimeout.current);
        hideTimeout.current = null;
      }
      setPreview({ url: value.trim(), x: e.clientX, y: e.clientY });
    },
    [value],
  );

  const movePreview = useCallback(
    (e: React.MouseEvent) => {
      if (preview) {
        setPreview({ url: value.trim(), x: e.clientX, y: e.clientY });
      }
    },
    [preview, value],
  );

  const hidePreview = useCallback(() => {
    hideTimeout.current = setTimeout(() => setPreview(null), 100);
  }, []);

  return (
    <td
      ref={cellRef}
      className="px-6 py-4 whitespace-nowrap relative"
      onMouseEnter={showPreview}
      onMouseMove={movePreview}
      onMouseLeave={hidePreview}
    >
      <span className="underline decoration-dotted decoration-blue-400 text-blue-600 cursor-pointer">
        {value || ''}
      </span>
      {preview && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            left: preview.x + 16,
            top: preview.y - 80,
          }}
        >
          <div className="bg-white rounded-lg shadow-xl border border-gray-200 p-1.5 max-w-[220px]">
            <img
              src={preview.url}
              alt="Preview"
              className="rounded max-w-[200px] max-h-[200px] object-contain"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
          </div>
        </div>
      )}
    </td>
  );
}

const PAGE_SIZE = 25;

export default function CsvPreview({
  columns,
  rows,
  totalRows,
  previewLimit,
  onLimitChange,
}: CsvPreviewProps) {
  const [currentPage, setCurrentPage] = useState(1);

  // Reset to page 1 when rows change (limit changed, new CSV, etc.)
  const rowCountRef = useRef(rows.length);
  if (rows.length !== rowCountRef.current) {
    rowCountRef.current = rows.length;
    if (currentPage !== 1) setCurrentPage(1);
  }

  if (!rows || rows.length === 0) {
    return (
      <div className="text-center p-8 text-gray-500 bg-gray-50 rounded-lg border border-gray-200">
        No data available to preview.
      </div>
    );
  }

  // Detect which columns contain image URLs by sampling first few rows
  const imageColumns = new Set(
    columns.filter((col) =>
      rows.some((row) => {
        const val = row[col];
        return val && isImageUrl(val);
      }),
    ),
  );

  // Pagination
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const startIndex = (currentPage - 1) * PAGE_SIZE;
  const paginatedRows = rows.length > PAGE_SIZE
    ? rows.slice(startIndex, startIndex + PAGE_SIZE)
    : rows;

  // Generate visible page numbers (show up to 5 around current)
  const pageNumbers: number[] = [];
  const maxVisible = 5;
  let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
  const endPage = Math.min(totalPages, startPage + maxVisible - 1);
  startPage = Math.max(1, endPage - maxVisible + 1);
  for (let i = startPage; i <= endPage; i++) {
    pageNumbers.push(i);
  }

  return (
    <div className="mt-8">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium text-gray-900">Data Preview</h3>
        <div className="flex items-center gap-3">
          <label htmlFor="preview-limit" className="text-sm text-gray-500">
            Rows:
          </label>
          <select
            id="preview-limit"
            value={previewLimit}
            onChange={(e) => onLimitChange(Number(e.target.value))}
            className="rounded-md border border-gray-300 bg-white py-1 px-2 text-sm text-gray-700 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          >
            {[10, 25, 50, 100].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          <span className="text-sm text-gray-500">
            Showing {rows.length} of {totalRows} total rows
          </span>
        </div>
      </div>

      <div className="overflow-x-auto shadow-md sm:rounded-lg bg-white border border-gray-200">
        <table className="min-w-full text-sm text-left text-gray-500">
          <thead className="text-xs text-gray-700 uppercase bg-gray-50 border-b border-gray-200">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="px-6 py-3 font-semibold whitespace-nowrap"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedRows.map((row, index) => {
              const globalIndex = startIndex + index;
              return (
                <tr
                  key={globalIndex}
                  className={`border-b border-gray-100 hover:bg-gray-50 transition-colors ${
                    globalIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'
                  }`}
                >
                  {columns.map((column) =>
                    imageColumns.has(column) && row[column] ? (
                      <ImageCell
                        key={`${globalIndex}-${column}`}
                        value={row[column]}
                      />
                    ) : (
                      <td
                        key={`${globalIndex}-${column}`}
                        className="px-6 py-4 whitespace-nowrap"
                      >
                        {row[column] || ''}
                      </td>
                    ),
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4 px-1">
          <span className="text-sm text-gray-500">
            Page {currentPage} of {totalPages}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setCurrentPage(1)}
              disabled={currentPage === 1}
              className="px-2 py-1 text-sm rounded border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              First
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-2 py-1 text-sm rounded border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Prev
            </button>
            {pageNumbers.map((page) => (
              <button
                key={page}
                onClick={() => setCurrentPage(page)}
                className={`px-2.5 py-1 text-sm rounded border ${
                  page === currentPage
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
              >
                {page}
              </button>
            ))}
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-2 py-1 text-sm rounded border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Next
            </button>
            <button
              onClick={() => setCurrentPage(totalPages)}
              disabled={currentPage === totalPages}
              className="px-2 py-1 text-sm rounded border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Last
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
