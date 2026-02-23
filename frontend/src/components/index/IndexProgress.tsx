import { useEffect, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useIndexWebSocket } from '../../hooks/useIndexWebSocket';
import { cancelIndex } from '../../api/indexing';

interface IndexProgressProps {
  jobId: string;
  onDone: () => void;
}

export default function IndexProgress({ jobId, onDone }: IndexProgressProps) {
  const { progress, logs, status, isConnected } = useIndexWebSocket(jobId);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const cancelMutation = useMutation({
    mutationFn: cancelIndex,
  });

  // Auto-scroll logs
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const percentage = progress.total_rows && progress.total_rows > 0
    ? Math.round((progress.rows_indexed / progress.total_rows) * 100)
    : 0;

  const isFinished = status === 'completed' || status === 'failed' || status === 'error' || status === 'cancelled';
  const displayStatus = status === 'pending' && isConnected ? 'running' : status;

  return (
    <div className="space-y-6 bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6">

      {/* Header / Status Badge */}
      <div className="flex justify-between items-center border-b border-gray-200 pb-4">
        <div>
          <h3 className="text-lg font-medium leading-6 text-gray-900">Indexing Progress</h3>
          <p className="text-sm text-gray-500">Job ID: {jobId}</p>
        </div>
        <div className="flex items-center space-x-2">
           {!isConnected && !isFinished && (
             <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
               Connecting...
             </span>
           )}
            <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium
              ${displayStatus === 'completed' ? 'bg-green-100 text-green-800' :
                displayStatus === 'failed' || displayStatus === 'error' ? 'bg-red-100 text-red-800' :
                displayStatus === 'cancelled' ? 'bg-gray-100 text-gray-800' :
                'bg-blue-100 text-blue-800'
              }
            `}>
              {displayStatus.charAt(0).toUpperCase() + displayStatus.slice(1)}
            </span>
        </div>
      </div>

      {/* Progress Bar & Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-3 space-y-2">
          <div className="flex justify-between text-sm font-medium text-gray-900">
            <span>{progress.rows_indexed} / {progress.total_rows ?? '?'} rows</span>
            <span>{percentage}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className={`h-2.5 rounded-full transition-all duration-500 ${
                status === 'failed' || status === 'error' ? 'bg-red-600' : 'bg-blue-600'
              }`}
              style={{ width: `${percentage}%` }}
            ></div>
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg text-center">
            <span className="block text-xs text-gray-500 uppercase">Elapsed Time</span>
            <span className="block text-2xl font-bold text-gray-900">{formatTime(progress.elapsed_seconds)}</span>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg text-center">
            <span className="block text-xs text-gray-500 uppercase">Errors</span>
            <span className={`block text-2xl font-bold ${progress.errors > 0 ? 'text-red-600' : 'text-gray-900'}`}>
              {progress.errors}
            </span>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg text-center">
            <span className="block text-xs text-gray-500 uppercase">Status</span>
            <span className="block text-lg font-medium text-gray-900 truncate">
              {displayStatus}
            </span>
        </div>
      </div>

      {/* Logs Panel */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Live Logs</h4>
        <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-xs text-gray-300">
          {logs.length === 0 && !progress.error ? (
            <div className="text-gray-600 italic">Waiting for logs...</div>
          ) : (
            <>
              {progress.error && (
                <div className="mb-2 text-red-400">
                  <span className="opacity-50">[error]</span> {progress.error}
                </div>
              )}
              {logs.map((log, index) => (
                <div key={index} className={`mb-1 ${
                  log.level === 'error' ? 'text-red-400' :
                  log.level === 'warning' ? 'text-yellow-400' :
                  log.level === 'success' ? 'text-green-400' :
                  'text-gray-300'
                }`}>
                  <span className="opacity-50">[{log.level}]</span> {log.message}
                </div>
              ))}
            </>
          )}
          <div ref={logsEndRef} />
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end pt-4 border-t border-gray-200">
        {isFinished ? (
           <button
             onClick={onDone}
             className="inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:text-sm"
           >
             Back to Upload
           </button>
        ) : (
          <button
            onClick={() => cancelMutation.mutate(jobId)}
            disabled={cancelMutation.isPending}
            className="inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-red-600 text-base font-medium text-white hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 sm:text-sm disabled:opacity-50"
          >
            {cancelMutation.isPending ? 'Cancelling...' : 'Cancel Indexing'}
          </button>
        )}
      </div>
    </div>
  );
}
