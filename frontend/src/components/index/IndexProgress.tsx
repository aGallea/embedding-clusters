import { useEffect, useMemo, useRef, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useIndexWebSocket, type LogMessage } from '../../hooks/useIndexWebSocket';
import { cancelIndex } from '../../api/indexing';

type VerbosityLevel = 'low' | 'medium' | 'high';

const VERBOSITY_LEVELS: VerbosityLevel[] = ['low', 'medium', 'high'];

const VERBOSITY_INCLUDES: Record<VerbosityLevel, VerbosityLevel[]> = {
  low: ['low'],
  medium: ['low', 'medium'],
  high: ['low', 'medium', 'high'],
};

interface IndexProgressProps {
  jobId: string;
  onDone: () => void;
}

export default function IndexProgress({ jobId, onDone }: IndexProgressProps) {
  const { progress, logs, status, isConnected, isStuckWarning, isStuckError } = useIndexWebSocket(jobId);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const [verbosity, setVerbosity] = useState<VerbosityLevel>('medium');

  const cancelMutation = useMutation({
    mutationFn: cancelIndex,
  });

  const filteredLogs = useMemo(() => {
    const allowed = VERBOSITY_INCLUDES[verbosity];
    return logs.filter((log: LogMessage) => allowed.includes(log.verbosity as VerbosityLevel));
  }, [logs, verbosity]);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [filteredLogs]);

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

      {/* Stuck error modal */}
      {isStuckError && !isFinished && (
        <div className="rounded-md bg-red-50 p-4 border border-red-200">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Backend Not Responding</h3>
              <p className="mt-1 text-sm text-red-700">
                No messages received for 30+ seconds. The backend may have crashed or become unresponsive.
                Consider cancelling and checking server logs.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stuck warning banner */}
      {isStuckWarning && !isStuckError && !isFinished && (
        <div className="rounded-md bg-yellow-50 p-4 border border-yellow-200">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">Slow Response</h3>
              <p className="mt-1 text-sm text-yellow-700">
                No messages received for 15+ seconds. The backend may be processing a large batch.
              </p>
            </div>
          </div>
        </div>
      )}

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
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-medium text-gray-700">Live Logs</h4>
          <div className="flex items-center space-x-2">
            <label htmlFor="verbosity-select" className="text-xs text-gray-500">Verbosity:</label>
            <select
              id="verbosity-select"
              value={verbosity}
              onChange={e => setVerbosity(e.target.value as VerbosityLevel)}
              className="text-xs border border-gray-300 rounded px-2 py-1 bg-white text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {VERBOSITY_LEVELS.map(level => (
                <option key={level} value={level}>
                  {level.charAt(0).toUpperCase() + level.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-xs text-gray-300">
          {filteredLogs.length === 0 && !progress.error ? (
            <div className="text-gray-600 italic">Waiting for logs...</div>
          ) : (
            <>
              {progress.error && (
                <div className="mb-2 text-red-400">
                  <span className="opacity-50">[error]</span> {progress.error}
                </div>
              )}
              {filteredLogs.map((log, index) => (
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
