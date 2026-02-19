import { create } from 'zustand'
import type { PlotResponse } from '../types'

interface PlotState {
  plotData: PlotResponse | null
  visibleClusters: Set<number>
  hoveredPointId: string | null
  renderMode: 'particles' | 'sprites' | 'spheres'
  // actions
  setPlotData: (data: PlotResponse | null) => void
  toggleCluster: (index: number) => void
  setHoveredPointId: (id: string | null) => void
  setRenderMode: (mode: 'particles' | 'sprites' | 'spheres') => void
  resetVisibleClusters: (clusterCount: number) => void
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

  resetVisibleClusters: (clusterCount) =>
    set({
      visibleClusters: new Set(Array.from({ length: clusterCount }, (_, i) => i)),
    }),
}))
