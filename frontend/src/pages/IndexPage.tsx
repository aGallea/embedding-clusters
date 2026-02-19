import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { previewCsv } from '../api/csv';
import { startIndex } from '../api/indexing';
import CsvUpload from '../components/csv/CsvUpload';
import CsvPreview from '../components/csv/CsvPreview';
import IndexForm from '../components/index/IndexForm';
import IndexProgress from '../components/index/IndexProgress';
import type { CsvUploadResponse, IndexRequest } from '../types';

type Step = 'upload' | 'preview' | 'form' | 'indexing';

export default function IndexPage() {
  const [step, setStep] = useState<Step>('upload');
  const [uploadedFilename, setUploadedFilename] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [totalRows, setTotalRows] = useState<number>(0);
  const [indexJobId, setIndexJobId] = useState<string | null>(null);

  // Query for CSV Preview
  const { data: previewData, isLoading: isPreviewLoading } = useQuery({
    queryKey: ['previewCsv', uploadedFilename],
    queryFn: () => previewCsv(uploadedFilename!),
    enabled: !!uploadedFilename && step === 'preview',
  });

  // Mutation for Start Indexing
  const startIndexMutation = useMutation({
    mutationFn: startIndex,
    onSuccess: (data) => {
      setIndexJobId(data.job_id);
      setStep('indexing');
    },
  });

  const handleUploadSuccess = (response: CsvUploadResponse) => {
    setUploadedFilename(response.filename);
    setTotalRows(response.rows);
    setColumns(response.columns);
    setStep('preview');
  };

  const handleIndexSubmit = (request: IndexRequest) => {
    startIndexMutation.mutate(request);
  };

  const handleIndexingDone = () => {
    // Reset state to start over
    setStep('upload');
    setUploadedFilename(null);
    setColumns([]);
    setTotalRows(0);
    setIndexJobId(null);
    startIndexMutation.reset();
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6 border-b pb-4 border-gray-200">
        <h1 className="text-3xl font-bold mb-2 text-gray-900">Index</h1>
        <p className="text-gray-500 text-lg">Upload CSV and index embeddings</p>
      </div>

      {step === 'upload' && (
        <div className="mt-8">
          <CsvUpload onUpload={handleUploadSuccess} />
        </div>
      )}

      {step === 'preview' && (
        <div className="space-y-6">
          {isPreviewLoading ? (
             <div className="text-center py-12">
               <svg className="animate-spin h-8 w-8 text-blue-500 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                 <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                 <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
               </svg>
               <p className="mt-2 text-gray-500">Loading preview...</p>
             </div>
          ) : (
            <>
              <div className="flex justify-end">
                <button
                  onClick={() => setStep('form')}
                  className="inline-flex justify-center rounded-md border border-transparent bg-blue-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                >
                  Configure Indexing
                </button>
              </div>
              <CsvPreview
                columns={columns}
                rows={previewData?.rows || []}
                totalRows={totalRows}
              />
            </>
          )}
        </div>
      )}

      {step === 'form' && uploadedFilename && (
        <div className="max-w-4xl mx-auto">
           <div className="mb-6">
             <button
               onClick={() => setStep('preview')}
               className="text-sm text-gray-500 hover:text-gray-700 flex items-center"
             >
               ← Back to Preview
             </button>
           </div>
           <IndexForm
             columns={columns}
             csvFilename={uploadedFilename}
             onSubmit={handleIndexSubmit}
             isSubmitting={startIndexMutation.isPending}
           />
           {startIndexMutation.isError && (
             <div className="mt-4 p-4 text-sm text-red-700 bg-red-100 rounded-lg border border-red-200">
               Error starting index: {startIndexMutation.error instanceof Error ? startIndexMutation.error.message : 'Unknown error'}
             </div>
           )}
        </div>
      )}

      {step === 'indexing' && indexJobId && (
        <div className="max-w-4xl mx-auto">
          <IndexProgress
            jobId={indexJobId}
            onDone={handleIndexingDone}
          />
        </div>
      )}
    </div>
  );
}
