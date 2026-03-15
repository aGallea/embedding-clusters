import { create } from 'zustand'
import type { AnnotationsResponse, ClusterDetailResponse, PlotResponse, ReductionAlgorithm, SearchResult, SubClusterResponse } from '../types'

interface PlotState {
  plotData: PlotResponse | null
  visibleClusters: Set<number>
  hoveredPointId: string | null
  renderMode: 'particles' | 'sprites' | 'spheres'
  pointSize: number
  searchResults: SearchResult[] | null
  highlightedIds: Set<string>
  isSearching: boolean
  queryPoint: { x: number; y: number; z: number } | null
  reductionAlgorithm: ReductionAlgorithm
  tsnePerplexity: number
  tsneLearningRate: string
  umapNNeighbors: number
  umapMinDist: number
  umapMetric: string
  selectedCluster: number | null
  clusterDetail: ClusterDetailResponse | null
  subClusterData: SubClusterResponse | null
  annotations: AnnotationsResponse | null
  isLoadingClusterDetail: boolean
  isLoadingSubCluster: boolean
  imageField: string | null
  plotJobId: string | null
  plotCollectionName: string | null
  // actions
  setPlotData: (data: PlotResponse | null) => void
  toggleCluster: (index: number) => void
  setHoveredPointId: (id: string | null) => void
  setRenderMode: (mode: 'particles' | 'sprites' | 'spheres') => void
  setPointSize: (size: number) => void
  resetVisibleClusters: (clusterCount: number) => void
  setSearchResults: (results: SearchResult[] | null) => void
  setHighlightedIds: (ids: Set<string>) => void
  setIsSearching: (searching: boolean) => void
  clearSearch: () => void
  setQueryPoint: (point: { x: number; y: number; z: number } | null) => void
  setReductionAlgorithm: (algorithm: ReductionAlgorithm) => void
  setTsnePerplexity: (perplexity: number) => void
  setTsneLearningRate: (rate: string) => void
  setUmapNNeighbors: (n: number) => void
  setUmapMinDist: (dist: number) => void
  setUmapMetric: (metric: string) => void
  setSelectedCluster: (index: number | null) => void
  setClusterDetail: (detail: ClusterDetailResponse | null) => void
  setSubClusterData: (data: SubClusterResponse | null) => void
  setAnnotations: (annotations: AnnotationsResponse | null) => void
  setIsLoadingClusterDetail: (loading: boolean) => void
  setIsLoadingSubCluster: (loading: boolean) => void
  clearClusterDrillDown: () => void
  setImageField: (field: string | null) => void
  setPlotJobId: (jobId: string | null) => void
  resetPlotJobId: () => void
  setPlotCollectionName: (name: string | null) => void
  resetPlotCollectionName: () => void
}

export const CLUSTER_COLORS = [
  '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
  '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
  '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
  '#e5c494', '#b3b3b3', '#8dd3c7', '#fb8072', '#80b1d3',
]

export const usePlotStore = create<PlotState>((set) => ({
  plotData: null,
  visibleClusters: new Set(),
  hoveredPointId: null,
  renderMode: 'particles',
  pointSize: 5,
  searchResults: null,
  highlightedIds: new Set(),
  isSearching: false,
  queryPoint: null,
  reductionAlgorithm: 'tsne',
  tsnePerplexity: 30,
  tsneLearningRate: 'auto',
  umapNNeighbors: 15,
  umapMinDist: 0.1,
  umapMetric: 'cosine',
  selectedCluster: null,
  clusterDetail: null,
  subClusterData: null,
  annotations: null,
  isLoadingClusterDetail: false,
  isLoadingSubCluster: false,
  imageField: null,
  plotJobId: null,
  plotCollectionName: null,

  setPlotData: (data) => set({ plotData: data }),

  toggleCluster: (index) =>
    set((state) => {
      const newVisible = new Set(state.visibleClusters)
      if (newVisible.has(index)) {
        newVisible.delete(index)
      } else {
        newVisible.add(index)
      }
      return { visibleClusters: newVisible }
    }),

  setHoveredPointId: (id) => set({ hoveredPointId: id }),

  setRenderMode: (mode) => set({ renderMode: mode }),

  setPointSize: (size) => set({ pointSize: size }),

  resetVisibleClusters: (clusterCount) =>
    set({
      visibleClusters: new Set(Array.from({ length: clusterCount }, (_, i) => i)),
    }),

  setSearchResults: (results) => set({
    searchResults: results,
    highlightedIds: new Set(results?.map(r => r.id) ?? []),
  }),
  setHighlightedIds: (ids) => set({ highlightedIds: ids }),
  setIsSearching: (searching) => set({ isSearching: searching }),
  setQueryPoint: (point) => set({ queryPoint: point }),
  clearSearch: () => set({
    searchResults: null,
    highlightedIds: new Set(),
    isSearching: false,
    queryPoint: null,
  }),
  setReductionAlgorithm: (algorithm) => set({ reductionAlgorithm: algorithm }),
  setTsnePerplexity: (perplexity) => set({ tsnePerplexity: perplexity }),
  setTsneLearningRate: (rate) => set({ tsneLearningRate: rate }),
  setUmapNNeighbors: (n) => set({ umapNNeighbors: n }),
  setUmapMinDist: (dist) => set({ umapMinDist: dist }),
  setUmapMetric: (metric) => set({ umapMetric: metric }),
  setSelectedCluster: (index) => set({ selectedCluster: index }),
  setClusterDetail: (detail) => set({ clusterDetail: detail }),
  setSubClusterData: (data) => set({ subClusterData: data }),
  setAnnotations: (annotations) => set({ annotations }),
  setIsLoadingClusterDetail: (loading) => set({ isLoadingClusterDetail: loading }),
  setIsLoadingSubCluster: (loading) => set({ isLoadingSubCluster: loading }),
  setImageField: (field) => set({ imageField: field }),
  setPlotJobId: (jobId) => set({ plotJobId: jobId }),
  resetPlotJobId: () => set({ plotJobId: null }),
  setPlotCollectionName: (name) => set({ plotCollectionName: name }),
  resetPlotCollectionName: () => set({ plotCollectionName: null }),
  clearClusterDrillDown: () => set({
    selectedCluster: null,
    clusterDetail: null,
    subClusterData: null,
    isLoadingClusterDetail: false,
    isLoadingSubCluster: false,
  }),
}))

if (typeof window !== 'undefined') {
  (window as Window & { __plotStore?: typeof usePlotStore }).__plotStore = usePlotStore
}
