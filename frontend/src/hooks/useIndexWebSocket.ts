import { useState, useEffect, useRef } from 'react';
import { createIndexWebSocket } from '../api/indexing';

export interface IndexProgressData {
  rows_indexed: number;
  total_rows: number | null;
  errors: number;
  elapsed_seconds: number;
  error: string | null;
}

export interface LogMessage {
  level: string;
  message: string;
}

export interface UseIndexWebSocketResult {
  progress: IndexProgressData;
  logs: LogMessage[];
  status: string;
  isConnected: boolean;
}

interface WebSocketMessage {
  type?: string;
  status?: string;
  level?: string;
  message?: string;
  rows_indexed?: number;
  total_rows?: number | null;
  errors?: number;
  elapsed_seconds?: number;
  collection_names?: string[];
  total_indexed?: number;
  error?: string;
  progress?: WebSocketMessage;
  [key: string]: unknown;
}

export function useIndexWebSocket(jobId: string | null): UseIndexWebSocketResult {
  const [progress, setProgress] = useState<IndexProgressData>({
    rows_indexed: 0,
    total_rows: null,
    errors: 0,
    elapsed_seconds: 0,
    error: null,
  });
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [status, setStatus] = useState<string>('pending');
  const [isConnected, setIsConnected] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    // Reset state when jobId changes
    setProgress({
      rows_indexed: 0,
      total_rows: null,
      errors: 0,
      elapsed_seconds: 0,
      error: null,
    });
    setLogs([]);
    setStatus('pending');
    setIsConnected(false);

    const ws = createIndexWebSocket(jobId);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketMessage;

        // Handle explicit status updates
        if (data.status) {
            setStatus(data.status);
        }

        if (typeof data.error === 'string') {
          setProgress(prev => ({
            ...prev,
            error: data.error
          }));
        }

        if (data.progress && typeof data.progress === 'object') {
          const progress = data.progress as WebSocketMessage;
          setProgress(prev => ({
            ...prev,
            rows_indexed: typeof progress.rows_indexed === 'number'
              ? progress.rows_indexed
              : prev.rows_indexed,
            total_rows: typeof progress.total_rows === 'number'
              ? progress.total_rows
              : prev.total_rows,
            errors: typeof progress.errors === 'number' ? progress.errors : prev.errors,
            elapsed_seconds: typeof progress.elapsed_seconds === 'number'
              ? progress.elapsed_seconds
              : prev.elapsed_seconds,
            error: typeof progress.error === 'string' ? progress.error : prev.error,
          }));
        }

        if (data.type === 'progress' || data.rows_indexed !== undefined) {
          setProgress(prev => ({
            ...prev,
            rows_indexed: typeof data.rows_indexed === 'number' ? data.rows_indexed : prev.rows_indexed,
            total_rows: typeof data.total_rows === 'number' ? data.total_rows : prev.total_rows,
            errors: typeof data.errors === 'number' ? data.errors : prev.errors,
            elapsed_seconds: typeof data.elapsed_seconds === 'number' ? data.elapsed_seconds : prev.elapsed_seconds,
            error: typeof data.error === 'string' ? data.error : prev.error,
          }));
          if (data.status && data.status !== status) {
            setStatus(data.status);
          }
        } else if (data.type === 'log') {
          setLogs(prev => [...prev, {
            level: data.level || 'info',
            message: data.message || ''
          }]);
        } else if (data.type === 'completed') {
          setStatus('completed');
          setLogs(prev => [...prev, {
            level: 'success',
            message: `Indexing completed. Total indexed: ${data.total_indexed}. Collections: ${Array.isArray(data.collection_names) ? data.collection_names.join(', ') : ''}`
          }]);
        } else if (data.type === 'error') {
          setStatus('error');
          setLogs(prev => [...prev, {
            level: 'error',
            message: data.message || 'Unknown error occurred'
          }]);
          setProgress(prev => ({
            ...prev,
            error: data.message || prev.error || 'Unknown error occurred'
          }));
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatus('error');
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      // Don't overwrite 'completed' or 'failed' status on close
      setStatus(prev => (prev === 'completed' || prev === 'failed' || prev === 'error') ? prev : 'disconnected');
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [jobId]);

  return { progress, logs, status, isConnected };
}
