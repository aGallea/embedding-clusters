import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'
import { startPlotCompute, getPlotData } from '../api/plot'
import { usePlotStore } from '../stores/plotStore'
import type { PlotRequest } from '../types'

export function usePlotData() {
  const setPlotData = usePlotStore((state) => state.setPlotData)
  const resetVisibleClusters = usePlotStore((state) => state.resetVisibleClusters)
  const setPlotJobId = usePlotStore((state) => state.setPlotJobId)
  const plotJobId = usePlotStore((state) => state.plotJobId)
  const plotData = usePlotStore((state) => state.plotData)
  const setPlotCollectionName = usePlotStore((state) => state.setPlotCollectionName)

  // 1. Mutation to start compute job
  const mutation = useMutation({
    mutationFn: (params: PlotRequest) => startPlotCompute(params),
    onSuccess: (data, params) => {
      setPlotJobId(data.job_id)
      setPlotData(null) // Clear previous plot
      resetVisibleClusters(0)
      setPlotCollectionName(params.chromadb_collection_name)
    },
  })

  // 2. Poll for results
  const { data, error, isError } = useQuery({
    queryKey: ['plotData', plotJobId],
    queryFn: () => getPlotData(plotJobId!),
    enabled: !!plotJobId,
    refetchInterval: (query) => {
      const data = query.state.data
      if (data?.ready) return false
      return 2000
    },
  })

  // 3. Update store when data is ready
  useEffect(() => {
    if (plotJobId && data?.ready) {
      setPlotData(data)
      if (data.job_id) {
        setPlotJobId(data.job_id)
      }
      resetVisibleClusters(data.clusters.length)
    }
  }, [data, plotJobId, resetVisibleClusters, setPlotData, setPlotJobId])

  const isComputing = mutation.isPending || (!!plotJobId && !data?.ready && !isError)

  useEffect(() => {
    if (!plotJobId && plotData) {
      console.warn('[Plot] clearing stale plot data: missing plotJobId')
      setPlotData(null)
      resetVisibleClusters(0)
      setPlotCollectionName(null)
    }
  }, [plotData, plotJobId, resetVisibleClusters, setPlotCollectionName, setPlotData])

  return {
    compute: mutation.mutate,
    isComputing,
    error: mutation.error || error,
    jobId: plotJobId,
  }
}
