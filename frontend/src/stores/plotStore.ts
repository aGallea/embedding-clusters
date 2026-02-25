import { create } from 'zustand'
import type { PlotResponse, SearchResult } from '../types'

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
}))
