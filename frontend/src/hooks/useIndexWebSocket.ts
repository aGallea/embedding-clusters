import { useState, useEffect, useRef, useCallback } from 'react';
import { createIndexWebSocket } from '../api/indexing';

const STUCK_WARNING_MS = 15_000;
const STUCK_ERROR_MS = 30_000;

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
  verbosity: string;
}

export interface UseIndexWebSocketResult {
  progress: IndexProgressData;
  logs: LogMessage[];
  status: string;
  isConnected: boolean;
  isStuckWarning: boolean;
  isStuckError: boolean;
}

interface WebSocketMessage {
  type?: string;
  status?: string;
  level?: string;
  message?: string;
  verbosity?: string;
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
  const [isStuckWarning, setIsStuckWarning] = useState<boolean>(false);
  const [isStuckError, setIsStuckError] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastMessageRef = useRef<number>(Date.now());
  const stuckIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Track the last server-reported elapsed_seconds to anchor the client timer
  const serverElapsedRef = useRef<number>(0);
  const serverElapsedAtRef = useRef<number>(Date.now());

  const resetStuckTimer = useCallback(() => {
    lastMessageRef.current = Date.now();
    setIsStuckWarning(false);
    setIsStuckError(false);
  }, []);

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
    setIsStuckWarning(false);
    setIsStuckError(false);
    serverElapsedRef.current = 0;
    serverElapsedAtRef.current = Date.now();
    lastMessageRef.current = Date.now();

    const ws = createIndexWebSocket(jobId);
    wsRef.current = ws;

    // Client-side elapsed timer — ticks every second for smooth display
    timerRef.current = setInterval(() => {
      const now = Date.now();
      const delta = (now - serverElapsedAtRef.current) / 1000;
      setProgress(prev => ({
        ...prev,
        elapsed_seconds: serverElapsedRef.current + delta,
      }));
    }, 1000);

    // Stuck detection interval — checks every 5s
    stuckIntervalRef.current = setInterval(() => {
      const silence = Date.now() - lastMessageRef.current;
      if (silence >= STUCK_ERROR_MS) {
        setIsStuckError(true);
        setIsStuckWarning(true);
      } else if (silence >= STUCK_WARNING_MS) {
        setIsStuckWarning(true);
        setIsStuckError(false);
      } else {
        setIsStuckWarning(false);
        setIsStuckError(false);
      }
    }, 5000);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      resetStuckTimer();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketMessage;
        resetStuckTimer();

        // Sync server elapsed time for client timer anchor
        if (typeof data.elapsed_seconds === 'number') {
          serverElapsedRef.current = data.elapsed_seconds;
          serverElapsedAtRef.current = Date.now();
        }

        // Handle explicit status updates
        if (data.status) {
          setStatus(data.status);
        }

        if (typeof data.error === 'string') {
          setProgress(prev => ({
            ...prev,
            error: data.error as string,
          }));
        }

        if (data.progress && typeof data.progress === 'object') {
          const progressMsg = data.progress as WebSocketMessage;
          if (typeof progressMsg.elapsed_seconds === 'number') {
            serverElapsedRef.current = progressMsg.elapsed_seconds;
            serverElapsedAtRef.current = Date.now();
          }
          setProgress(prev => ({
            ...prev,
            rows_indexed: typeof progressMsg.rows_indexed === 'number'
              ? progressMsg.rows_indexed
              : prev.rows_indexed,
            total_rows: typeof progressMsg.total_rows === 'number'
              ? progressMsg.total_rows
              : prev.total_rows,
            errors: typeof progressMsg.errors === 'number' ? progressMsg.errors : prev.errors,
            elapsed_seconds: typeof progressMsg.elapsed_seconds === 'number'
              ? progressMsg.elapsed_seconds
              : prev.elapsed_seconds,
            error: typeof progressMsg.error === 'string' ? progressMsg.error : prev.error,
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
            message: data.message || '',
            verbosity: data.verbosity || 'low',
          }]);
        } else if (data.type === 'heartbeat') {
          // Heartbeat keeps stuck detection happy — elapsed already synced above
        } else if (data.type === 'completed') {
          setStatus('completed');
          setLogs(prev => [...prev, {
            level: 'success',
            message: `Indexing completed. Total indexed: ${data.total_indexed}. Collections: ${Array.isArray(data.collection_names) ? data.collection_names.join(', ') : ''}`,
            verbosity: 'low',
          }]);
        } else if (data.type === 'error') {
          setStatus('error');
          setLogs(prev => [...prev, {
            level: 'error',
            message: data.message || 'Unknown error occurred',
            verbosity: 'low',
          }]);
          setProgress(prev => ({
            ...prev,
            error: data.message || prev.error || 'Unknown error occurred',
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
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      if (stuckIntervalRef.current) {
        clearInterval(stuckIntervalRef.current);
        stuckIntervalRef.current = null;
      }
    };
  }, [jobId, resetStuckTimer]);

  return { progress, logs, status, isConnected, isStuckWarning, isStuckError };
}
