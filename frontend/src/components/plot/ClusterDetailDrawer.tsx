import { useState, useEffect, useCallback, useRef } from 'react'
import { usePlotStore, CLUSTER_COLORS } from '../../stores/plotStore'
import {
  getClusterDetail,
  updateAnnotation,
  getAnnotations,
  suggestK,
  subCluster,
  subClusterByPointIds,
} from '../../api/plot'
import { loadAiSettings, nameAiSubClusters } from '../../api/ai'
import type { ClusterDetailResponse, SubClusterResponse, SuggestKResponse } from '../../types'
import SelectedPointsDistancePanel from './SelectedPointsDistancePanel'

interface ClusterDetailDrawerProps {
  jobId: string
  imageField?: string
}

export default function ClusterDetailDrawer({ jobId, imageField }: ClusterDetailDrawerProps) {
  const selectedCluster = usePlotStore((s) => s.selectedCluster)
  const clusterDetail = usePlotStore((s) => s.clusterDetail)
  const annotations = usePlotStore((s) => s.annotations)
  const isLoadingClusterDetail = usePlotStore((s) => s.isLoadingClusterDetail)
  const plotData = usePlotStore((s) => s.plotData)
  const selectedPointIds = usePlotStore((s) => s.selectedPointIds)
  const setClusterDetail = usePlotStore((s) => s.setClusterDetail)
  const setAnnotations = usePlotStore((s) => s.setAnnotations)
  const setIsLoadingClusterDetail = usePlotStore((s) => s.setIsLoadingClusterDetail)
  const setHighlightedIds = usePlotStore((s) => s.setHighlightedIds)
  const clearSelectedPointIds = usePlotStore((s) => s.clearSelectedPointIds)
  const setSelectedPointIds = usePlotStore((s) => s.setSelectedPointIds)
  const clearClusterDrillDown = usePlotStore((s) => s.clearClusterDrillDown)
  const drillPath = usePlotStore((s) => s.drillPath)
  const drillIntoCluster = usePlotStore((s) => s.drillIntoCluster)
  const drillIntoSubCluster = usePlotStore((s) => s.drillIntoSubCluster)
  const setIsLoadingDrill = usePlotStore((s) => s.setIsLoadingDrill)
  const isLoadingDrill = usePlotStore((s) => s.isLoadingDrill)
  const isNamingSubClusters = usePlotStore((s) => s.isNamingSubClusters)
  const setIsNamingSubClusters = usePlotStore((s) => s.setIsNamingSubClusters)
  const updateSubClusterNames = usePlotStore((s) => s.updateSubClusterNames)

  const [page, setPage] = useState(1)
  const [isEditingName, setIsEditingName] = useState(false)
  const [editName, setEditName] = useState('')
  const [notes, setNotes] = useState('')
  const [tagsInput, setTagsInput] = useState('')
  const [subClusterK, setSubClusterK] = useState(4)
  const [suggestedK, setSuggestedK] = useState<SuggestKResponse | null>(null)
  const [isLoadingSuggestK, setIsLoadingSuggestK] = useState(false)
  const [expandedSubClusterIndex, setExpandedSubClusterIndex] = useState<number | null>(null)
  const [selectedSubClusterIndex, setSelectedSubClusterIndex] = useState<number | null>(null)
  const notesTimeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined)
  const tagsTimeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  const clusterIndex = selectedCluster
  const cluster = plotData?.clusters.find((c) => c.index === clusterIndex)
  const color = clusterIndex != null ? CLUSTER_COLORS[clusterIndex % CLUSTER_COLORS.length] : '#999'
  const annotation = clusterIndex != null ? annotations?.clusters[String(clusterIndex)] : undefined

  const isDrilled = drillPath.length > 0
  const currentLevel = isDrilled ? drillPath[drillPath.length - 1] : null
  const subClusters = currentLevel?.subClusterData.sub_clusters
  const hasComputedSubClusters = Boolean(subClusters && subClusters.length > 0)

  // Load cluster detail when selected cluster changes
  useEffect(() => {
    if (clusterIndex == null) return
    setPage(1)
    setSuggestedK(null)
    setIsLoadingSuggestK(false)
    setIsLoadingClusterDetail(true)
    setClusterDetail(null)

    getClusterDetail(jobId, clusterIndex, 1)
      .then((data: ClusterDetailResponse) => setClusterDetail(data))
      .catch(() => setClusterDetail(null))
      .finally(() => setIsLoadingClusterDetail(false))
  }, [jobId, clusterIndex, setClusterDetail, setIsLoadingClusterDetail])

  // Load annotations
  useEffect(() => {
    getAnnotations(jobId)
      .then(setAnnotations)
      .catch(() => setAnnotations(null))
  }, [jobId, setAnnotations])

  // Sync local notes/tags state with annotation
  useEffect(() => {
    setNotes(annotation?.notes ?? '')
    setTagsInput(annotation?.tags?.join(', ') ?? '')
  }, [annotation])

  // Reset sub-cluster k and suggested k when cluster changes
  useEffect(() => {
    setSubClusterK(4)
    setSuggestedK(null)
    setExpandedSubClusterIndex(null)
    setSelectedSubClusterIndex(null)
  }, [clusterIndex])

  useEffect(() => {
    setSelectedSubClusterIndex(null)
    setExpandedSubClusterIndex(null)
  }, [currentLevel?.label])

  const autoNameSubClusters = useCallback(async (
    data: SubClusterResponse,
    parentName: string | undefined,
  ) => {
    const settings = loadAiSettings()
    if ((!settings.apiKey && settings.provider !== 'ollama') || !jobId) return

    setIsNamingSubClusters(true)
    try {
      const pointIds = data.points.map((p) => p.id)
      const labels = data.points.map((p) => p.sub_cluster)
      const response = await nameAiSubClusters({
        job_id: jobId,
        point_ids: pointIds,
        sub_cluster_labels: labels,
        api_key: settings.apiKey,
        model: settings.provider ? `${settings.provider}/${settings.model}` : settings.model,
        base_url: settings.baseUrl || undefined,
        temperature: settings.temperature,
        parent_cluster_name: parentName,
      })
      updateSubClusterNames(response.names)
    } catch {
      // AI naming is best-effort; failures are silent
    } finally {
      setIsNamingSubClusters(false)
    }
  }, [jobId, setIsNamingSubClusters, updateSubClusterNames])

  const handlePageChange = useCallback((newPage: number) => {
    if (clusterIndex == null) return
    setPage(newPage)
    setIsLoadingClusterDetail(true)
    getClusterDetail(jobId, clusterIndex, newPage)
      .then((data: ClusterDetailResponse) => setClusterDetail(data))
      .catch(() => {/* keep previous data */})
      .finally(() => setIsLoadingClusterDetail(false))
  }, [jobId, clusterIndex, setClusterDetail, setIsLoadingClusterDetail])

  const handleSaveName = useCallback(() => {
    if (clusterIndex == null) return
    setIsEditingName(false)
    if (editName.trim()) {
      updateAnnotation(jobId, clusterIndex, { name: editName.trim() })
        .then(setAnnotations)
        .catch(() => {/* silent */})
    }
  }, [jobId, clusterIndex, editName, setAnnotations])

  const handleNotesChange = useCallback((value: string) => {
    setNotes(value)
    if (notesTimeoutRef.current) clearTimeout(notesTimeoutRef.current)
    notesTimeoutRef.current = setTimeout(() => {
      if (clusterIndex != null) {
        updateAnnotation(jobId, clusterIndex, { notes: value })
          .then(setAnnotations)
          .catch(() => {/* silent */})
      }
    }, 800)
  }, [jobId, clusterIndex, setAnnotations])

  const handleTagsChange = useCallback((value: string) => {
    setTagsInput(value)
    if (tagsTimeoutRef.current) clearTimeout(tagsTimeoutRef.current)
    tagsTimeoutRef.current = setTimeout(() => {
      if (clusterIndex != null) {
        const tags = value.split(',').map((t) => t.trim()).filter(Boolean)
        updateAnnotation(jobId, clusterIndex, { tags })
          .then(setAnnotations)
          .catch(() => {/* silent */})
      }
    }, 800)
  }, [jobId, clusterIndex, setAnnotations])

  const handleItemClick = useCallback((id: string) => {
    const nextSelected = new Set(selectedPointIds)
    if (nextSelected.has(id)) {
      nextSelected.delete(id)
    } else {
      nextSelected.add(id)
    }
    setSelectedPointIds(nextSelected)
    setHighlightedIds(nextSelected)
  }, [selectedPointIds, setHighlightedIds, setSelectedPointIds])

  const handleClearSelected = useCallback(() => {
    clearSelectedPointIds()
    setHighlightedIds(new Set())
  }, [clearSelectedPointIds, setHighlightedIds])

  const handleSelectPage = useCallback(() => {
    if (!clusterDetail) return
    const pageIds = new Set(clusterDetail.items.map((item) => item.id))
    setSelectedPointIds(pageIds)
    setHighlightedIds(pageIds)
  }, [clusterDetail, setHighlightedIds, setSelectedPointIds])

  const handleSuggestK = useCallback(async () => {
    if (clusterIndex == null || !jobId) return
    setIsLoadingSuggestK(true)
    setSuggestedK(null)
    try {
      if (isDrilled && currentLevel) {
        // When drilled, use point_ids for suggest-k on sub-cluster
        const result = await suggestK(jobId, {
          point_ids: currentLevel.pointIds,
          max_k: 10,
        })
        setSuggestedK(result)
        setSubClusterK(result.suggested_k)
      } else {
        const result = await suggestK(jobId, {
          cluster_index: clusterIndex,
          max_k: 10,
        })
        setSuggestedK(result)
        setSubClusterK(result.suggested_k)
      }
    } catch {
      setSuggestedK(null)
    } finally {
      setIsLoadingSuggestK(false)
    }
  }, [jobId, clusterIndex, isDrilled, currentLevel])

  const handleComputeSubClusters = useCallback(async () => {
    if (clusterIndex == null || !jobId || isLoadingDrill) return
    if (hasComputedSubClusters && selectedSubClusterIndex == null) return

    setIsLoadingDrill(true)
    try {
      setExpandedSubClusterIndex(null)
      if (hasComputedSubClusters && currentLevel) {
        const selectedSubCluster = selectedSubClusterIndex
        if (selectedSubCluster == null) {
          setIsLoadingDrill(false)
          return
        }
        const selectedPointIds = currentLevel.subClusterData.points
          .filter((point) => point.sub_cluster === selectedSubCluster)
          .map((point) => point.id)

        if (selectedPointIds.length < 4) {
          setIsLoadingDrill(false)
          return
        }

        const data = await subClusterByPointIds(jobId, {
          num_sub_clusters: subClusterK,
          point_ids: selectedPointIds,
        })
        setSelectedSubClusterIndex(null)
        drillIntoSubCluster(selectedSubCluster, data)
        const parentLabel = currentLevel.label
        autoNameSubClusters(data, parentLabel)
      } else {
        const data = await subCluster(jobId, clusterIndex, {
          num_sub_clusters: subClusterK,
        })
        setSelectedSubClusterIndex(null)
        drillIntoCluster(clusterIndex, data)
        const parentName = annotation?.name ?? cluster?.name
        autoNameSubClusters(data, parentName)
      }
    } catch {
      setIsLoadingDrill(false)
    }
  }, [
    jobId,
    clusterIndex,
    subClusterK,
    isLoadingDrill,
    currentLevel,
    hasComputedSubClusters,
    selectedSubClusterIndex,
    setIsLoadingDrill,
    drillIntoCluster,
    drillIntoSubCluster,
    autoNameSubClusters,
    annotation,
    cluster,
  ])

  const handleToggleSubClusterExpanded = useCallback((subClusterIndex: number) => {
    setExpandedSubClusterIndex((currentIndex) =>
      currentIndex === subClusterIndex ? null : subClusterIndex,
    )
  }, [])

  const handleSelectSubCluster = useCallback((subClusterIndex: number) => {
    const subCluster = currentLevel?.subClusterData.sub_clusters.find(
      (entry) => entry.index === subClusterIndex,
    )

    if (!subCluster || subCluster.count < 4) {
      return
    }

    setSelectedSubClusterIndex((currentIndex) =>
      currentIndex === subClusterIndex ? null : subClusterIndex,
    )
  }, [currentLevel])

  if (clusterIndex == null) return null

  const totalPages = clusterDetail ? Math.ceil(clusterDetail.total_items / clusterDetail.page_size) : 0
  const displayName = annotation?.name ?? cluster?.name ?? `Cluster ${clusterIndex}`
  const selectedItems = clusterDetail?.items.filter((item) => selectedPointIds.has(item.id)) ?? []

  return (
    <div
      className="w-[min(24rem,calc(100vw-20rem))] min-w-[280px] max-w-[24rem] bg-white border-l border-gray-200 flex flex-col h-full shadow-lg overflow-hidden relative z-50 pointer-events-auto"
      data-testid="cluster-detail-drawer"
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-2 min-w-0">
          <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: color }} />
          {isEditingName ? (
            <input
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onBlur={handleSaveName}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSaveName()
                if (e.key === 'Escape') setIsEditingName(false)
              }}
              className="text-sm font-bold text-gray-900 border-b border-blue-500 outline-none bg-transparent w-full"
              autoFocus
            />
          ) : (
            <button
              onClick={() => { setEditName(displayName); setIsEditingName(true) }}
              className="text-sm font-bold text-gray-900 truncate hover:text-blue-600 transition-colors text-left"
              title="Click to rename"
            >
              {displayName}
            </button>
          )}
          {clusterDetail && (
            <span className="text-xs text-gray-500 shrink-0">
              ({clusterDetail.total_items} items)
            </span>
          )}
        </div>
        <button
          onClick={clearClusterDrillDown}
          className="p-1 hover:bg-gray-100 rounded transition-colors shrink-0"
          aria-label="Close drawer"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Action buttons */}
      {!hasComputedSubClusters && (
        <div className="px-4 py-2 border-b border-gray-200 flex items-center space-x-2 shrink-0">
          <button
            onClick={handleClearSelected}
            className="text-xs px-3 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-50 transition-colors"
          >
            Clear selected
          </button>
          <button
            onClick={handleSelectPage}
            className="text-xs px-3 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-50 transition-colors"
          >
            Select page
          </button>
        </div>
      )}

      {/* Sub-clustering section */}
      <div className="px-4 py-3 border-b border-gray-200 shrink-0 space-y-2" data-testid="sub-cluster-section">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-gray-600">
            Sub-clusters: {subClusterK}
          </label>
          {isLoadingDrill && (
            <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600" />
          )}
        </div>
        <input
          type="range"
          min="2"
          max="20"
          value={subClusterK}
          onChange={(e) => setSubClusterK(Number(e.target.value))}
          className="w-full"
          data-testid="sub-cluster-k-slider"
        />
        <div className="flex items-center space-x-2">
          <button
            onClick={handleSuggestK}
            disabled={isLoadingSuggestK}
            className="text-xs px-3 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-50 disabled:opacity-50 transition-colors"
            data-testid="sub-cluster-suggest-k"
          >
            {isLoadingSuggestK ? 'Analyzing...' : 'Suggest k'}
          </button>
          {suggestedK && (
            <span className="text-xs text-green-700 font-medium">
              k={suggestedK.suggested_k}
            </span>
          )}
          <button
            onClick={handleComputeSubClusters}
            disabled={isLoadingDrill || (hasComputedSubClusters && selectedSubClusterIndex == null)}
            className="text-xs px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition-colors ml-auto"
            data-testid="sub-cluster-compute"
          >
            {isLoadingDrill ? 'Computing...' : 'Compute Sub-clusters'}
          </button>
        </div>
      </div>

      {/* Sub-cluster list when drilled */}
      {isDrilled && subClusters && (
        <div
          className="px-4 py-2 border-b border-gray-200 flex-1 min-h-0 overflow-y-auto"
          data-testid="drawer-subcluster-list"
        >
          <div className="flex items-center space-x-2 mb-1.5">
            <span className="text-xs font-medium text-gray-500">Current sub-clusters</span>
            {isNamingSubClusters && (
              <div className="flex items-center space-x-1">
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-purple-600" />
                <span className="text-[10px] text-purple-600">Naming...</span>
              </div>
            )}
          </div>
          <div className="space-y-1">
            {subClusters.map((sc) => {
              const scColor = CLUSTER_COLORS[sc.index % CLUSTER_COLORS.length]
              const canDrill = sc.count >= 4
              const isExpanded = expandedSubClusterIndex === sc.index
              const isSelected = selectedSubClusterIndex === sc.index
              const previewItems = currentLevel?.subClusterData.points
                .filter((point) => point.sub_cluster === sc.index)
                ?? []
              return (
                <div
                  key={sc.index}
                  className={`rounded border overflow-hidden transition-colors ${
                    isSelected
                      ? 'bg-blue-50 border-blue-300 ring-1 ring-blue-200'
                      : 'bg-gray-50 border-gray-200'
                  }`}
                  data-testid={`drawer-subcluster-row-${sc.index}`}
                  data-selected={isSelected ? 'true' : 'false'}
                >
                  <div
                    className="flex items-center justify-between py-1 px-2 cursor-pointer"
                    onClick={() => handleSelectSubCluster(sc.index)}
                  >
                    <div className="flex items-center space-x-2 min-w-0">
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: scColor }}
                      />
                      <span className="text-xs text-gray-900 font-medium">
                        {sc.name || `Sub ${sc.index}`}
                      </span>
                      <span className="text-[10px] text-gray-500 shrink-0">
                        {sc.count} pts
                      </span>
                      {canDrill && (
                        <span
                          className={`text-[10px] rounded px-1.5 py-0.5 ${
                            isSelected
                              ? 'bg-blue-100 text-blue-700'
                              : 'bg-gray-200 text-gray-600'
                          }`}
                          data-testid={`drawer-subcluster-compute-source-${sc.index}`}
                        >
                          {isSelected ? 'Selected' : 'Select'}
                        </span>
                      )}
                    </div>
                    <button
                      data-testid={`drawer-subcluster-toggle-${sc.index}`}
                      onClick={(event) => {
                        event.stopPropagation()
                        handleToggleSubClusterExpanded(sc.index)
                      }}
                      className="p-0.5 rounded hover:bg-gray-100 transition-colors text-gray-400 hover:text-gray-700"
                      title={isExpanded ? 'Collapse sub-cluster preview' : 'Expand sub-cluster preview'}
                      aria-label={isExpanded ? `Collapse Sub ${sc.index}` : `Expand Sub ${sc.index}`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={isExpanded ? 'rotate-180 transition-transform' : 'transition-transform'}>
                        <polyline points="6 9 12 15 18 9" />
                      </svg>
                    </button>
                  </div>
                  {isExpanded && (
                    <div
                      className="border-t border-gray-200 px-2 py-2 bg-white"
                      data-testid={`drawer-subcluster-panel-${sc.index}`}
                    >
                      <div
                        className="space-y-1 max-h-80 overflow-y-auto pr-1"
                        data-testid={`drawer-subcluster-preview-${sc.index}`}
                      >
                        {previewItems.length > 0 ? (
                          previewItems.map((item) => {
                            const previewMetadataEntries = Object.entries(item.metadata)
                              .filter(([key]) => key !== imageField)
                              .slice(0, 3)
                            const isItemSelected = selectedPointIds.has(item.id)

                            return (
                              <button
                                key={item.id}
                                type="button"
                                onClick={(event) => {
                                  event.stopPropagation()
                                  handleItemClick(item.id)
                                }}
                                className={`w-full rounded border px-3 text-left flex items-start space-x-3 transition-colors ${
                                  isItemSelected
                                    ? 'py-3 bg-blue-50 border-blue-300 ring-1 ring-blue-200'
                                    : 'py-2 border-gray-100 bg-gray-50 hover:bg-blue-50'
                                }`}
                                data-testid={`drawer-subcluster-preview-item-${sc.index}`}
                                data-selected={isItemSelected ? 'true' : 'false'}
                              >
                                {imageField && imageField in item.metadata && (
                                  <img
                                    src={String(item.metadata[imageField])}
                                    alt=""
                                    className="w-10 h-10 rounded object-cover shrink-0"
                                    loading="lazy"
                                    data-testid={`drawer-subcluster-preview-image-${sc.index}`}
                                  />
                                )}
                                <div className="min-w-0 flex-1">
                                  <div className="text-xs font-medium text-gray-900 truncate">
                                    {item.id}
                                  </div>
                                  {typeof item.metadata.distance_to_centroid === 'number' && (
                                    <div className="text-[10px] text-gray-500 truncate">
                                      dist: {item.metadata.distance_to_centroid.toFixed(4)}
                                    </div>
                                  )}
                                  {previewMetadataEntries.map(([key, value]) => (
                                    <div
                                      key={key}
                                      className="text-[10px] text-gray-400 truncate"
                                      data-testid={`drawer-subcluster-preview-meta-${sc.index}`}
                                    >
                                      {key}: {String(value)}
                                    </div>
                                  ))}
                                </div>
                              </button>
                            )
                          })
                        ) : (
                          <div className="text-[10px] text-gray-500">
                            Preview not available for this sub-cluster yet.
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Items list */}
      {!hasComputedSubClusters && (
        <div className="flex-1 overflow-y-auto" data-testid="cluster-detail-main-list">
        {isLoadingClusterDetail && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600" />
          </div>
        )}

        {clusterDetail && !isLoadingClusterDetail && (
          <div className="divide-y divide-gray-100">
            {clusterDetail.items.map((item) => {
              const isSelected = selectedPointIds.has(item.id)
              return (
              <button
                key={item.id}
                onClick={() => handleItemClick(item.id)}
                className={`w-full px-4 text-left transition-colors flex items-start space-x-3 ${
                  isSelected
                    ? 'py-3 bg-blue-50 border-l-2 border-blue-500'
                    : 'py-2 hover:bg-blue-50'
                }`}
              >
                {imageField && imageField in item.metadata && (
                  <img
                    src={String(item.metadata[imageField])}
                    alt=""
                    className={`rounded object-cover shrink-0 ${isSelected ? 'w-12 h-12' : 'w-10 h-10'}`}
                    loading="lazy"
                  />
                )}
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium text-gray-900 truncate">
                    {item.id}
                  </div>
                  <div className="text-[10px] text-gray-500">
                    dist: {item.distance_to_centroid.toFixed(4)}
                  </div>
                  {Object.entries(item.metadata)
                    .filter(([k]) => k !== imageField)
                    .slice(0, isSelected ? 6 : 2)
                    .map(([key, value]) => (
                      <div key={key} className="text-[10px] text-gray-400 truncate">
                        {key}: {String(value)}
                      </div>
                    ))}
                </div>
              </button>
            )})}
          </div>
        )}
        </div>
      )}

      {/* Pagination */}
      {!hasComputedSubClusters && totalPages > 1 && (
        <div className="px-4 py-2 border-t border-gray-200 flex items-center justify-between text-xs shrink-0">
          <button
            onClick={() => handlePageChange(page - 1)}
            disabled={page <= 1}
            className="px-2 py-1 rounded border border-gray-300 disabled:opacity-30 hover:bg-gray-50"
          >
            Prev
          </button>
          <span className="text-gray-500">
            Page {page} / {totalPages}
          </span>
          <button
            onClick={() => handlePageChange(page + 1)}
            disabled={page >= totalPages}
            className="px-2 py-1 rounded border border-gray-300 disabled:opacity-30 hover:bg-gray-50"
          >
            Next
          </button>
        </div>
      )}

      {!hasComputedSubClusters && <SelectedPointsDistancePanel selectedItems={selectedItems} />}

      {/* Annotation section */}
      <div className="px-4 py-3 border-t border-gray-200 space-y-2 shrink-0">
        <div>
          <label className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">Notes</label>
          <textarea
            value={notes}
            onChange={(e) => handleNotesChange(e.target.value)}
            placeholder="Add notes about this cluster..."
            className="w-full mt-1 text-xs border border-gray-200 rounded px-2 py-1.5 resize-none focus:outline-none focus:ring-1 focus:ring-blue-400"
            rows={2}
          />
        </div>
        <div>
          <label className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">Tags</label>
          <input
            type="text"
            value={tagsInput}
            onChange={(e) => handleTagsChange(e.target.value)}
            placeholder="tag1, tag2, ..."
            className="w-full mt-1 text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-400"
          />
        </div>
      </div>
    </div>
  )
}
