import { useMutation, useQuery } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { startPlotCompute, getPlotData } from '../api/plot'
import { usePlotStore } from '../stores/plotStore'
import type { PlotRequest } from '../types'

export function usePlotData() {
  const [jobId, setJobId] = useState<string | null>(null)
  const setPlotData = usePlotStore((state) => state.setPlotData)
  const resetVisibleClusters = usePlotStore((state) => state.resetVisibleClusters)

  // 1. Mutation to start compute job
  const mutation = useMutation({
    mutationFn: (params: PlotRequest) => startPlotCompute(params),
    onSuccess: (data) => {
      setJobId(data.job_id)
      setPlotData(null) // Clear previous plot
    },
  })

  // 2. Poll for results
  const { data, error, isError } = useQuery({
    queryKey: ['plotData', jobId],
    queryFn: () => getPlotData(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data
      if (data?.ready) return false
      return 2000
    },
  })

  // 3. Update store when data is ready
  useEffect(() => {
    if (data?.ready) {
      setPlotData(data)
      resetVisibleClusters(data.clusters.length)
      // Stop polling by clearing job ID? No, query stays enabled but refetchInterval stops.
      // But if we want to "stop" the computing state effectively...
    }
  }, [data, setPlotData, resetVisibleClusters])

  const isComputing = mutation.isPending || (!!jobId && !data?.ready && !isError)

  return {
    compute: mutation.mutate,
    isComputing,
    error: mutation.error || error,
    jobId,
  }
}
