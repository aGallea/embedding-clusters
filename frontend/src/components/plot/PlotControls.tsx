import { useQuery } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { fetchCollections, fetchCollection } from '../../api/collections'
import { usePlotStore } from '../../stores/plotStore'
import type { PlotRequest, ReductionAlgorithm } from '../../types'
import ClusterSuggestion from './ClusterSuggestion'

function CollapsibleSection({ title, defaultOpen = false, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  return (
    <div>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full text-xs font-medium text-gray-600 hover:text-gray-900 py-1"
      >
        {title}
        <svg className={`w-3.5 h-3.5 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && <div className="mt-1 space-y-1.5">{children}</div>}
    </div>
  )
}

interface PlotControlsProps {
  onCompute: (params: PlotRequest) => void
  isComputing: boolean
}

export default function PlotControls({ onCompute, isComputing }: PlotControlsProps) {
  const [searchParams, setSearchParams] = useSearchParams()
  const [selectedCollection, setSelectedCollection] = useState(searchParams.get('collection') || '')
  const [numClusters, setNumClusters] = useState(10)
  const [textDisplayFields, setTextDisplayFields] = useState<string[]>([])
  const [imageField, setImageField] = useState('')
  const [gptEnabled, setGptEnabled] = useState(false)
  const [gptModel, setGptModel] = useState('gpt-3.5-turbo')
  const [gptTemperature, setGptTemperature] = useState(0.51)

  const {
    renderMode, setRenderMode, pointSize, setPointSize,
    reductionAlgorithm, setReductionAlgorithm,
    tsnePerplexity, setTsnePerplexity,
    tsneLearningRate, setTsneLearningRate,
    umapNNeighbors, setUmapNNeighbors,
    umapMinDist, setUmapMinDist,
    umapMetric, setUmapMetric,
    setImageField: setStoreImageField,
    resetPlotJobId,
    clearClusterDrillDown,
    setPlotData,
    resetVisibleClusters,
  } = usePlotStore()

  // 1. Fetch collection list
  const { data: collections } = useQuery({
    queryKey: ['collections'],
    queryFn: fetchCollections,
  })

  // 2. Fetch details when collection selected
  const { data: collectionDetails } = useQuery({
    queryKey: ['collection', selectedCollection],
    queryFn: () => fetchCollection(selectedCollection),
    enabled: !!selectedCollection,
  })

  // Auto-select first image field if available
  useEffect(() => {
    if (collectionDetails?.metadata_fields) {
      // Reset fields when collection changes
      setTextDisplayFields([])
      setImageField('')

      const imgField = collectionDetails.metadata_fields.find(f =>
        f.toLowerCase().includes('image') || f.toLowerCase().includes('img') || f.toLowerCase().includes('url')
      )
      if (imgField) setImageField(imgField)
    }
  }, [collectionDetails])

  // Sync local imageField to store for drawer access
  useEffect(() => {
    setStoreImageField(imageField || null)
  }, [imageField, setStoreImageField])

  const handleCompute = () => {
    if (!selectedCollection) return

    resetPlotJobId()
    clearClusterDrillDown()
    setPlotData(null)
    resetVisibleClusters(0)

    const request: PlotRequest = {
      chromadb_collection_name: selectedCollection,
      num_clusters: numClusters,
      text_display_fields: textDisplayFields,
      image_field: imageField || undefined,
      gpt_generate_cluster_name: gptEnabled,
      gpt_default_model: gptEnabled ? gptModel : undefined,
      gpt_default_temperature: gptEnabled ? gptTemperature : undefined,
      reduction_algorithm: reductionAlgorithm,
      ...(reductionAlgorithm === 'tsne' && {
        tsne_perplexity: tsnePerplexity,
        tsne_learning_rate: tsneLearningRate,
      }),
      ...(reductionAlgorithm === 'umap' && {
        umap_n_neighbors: umapNNeighbors,
        umap_min_dist: umapMinDist,
        umap_metric: umapMetric,
      }),
    }
    onCompute(request)
  }

  const toggleField = (field: string) => {
    setTextDisplayFields(prev =>
      prev.includes(field) ? prev.filter(f => f !== field) : [...prev, field]
    )
  }

  return (
    <div className="space-y-2 p-3">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-1">Configuration</h2>

      {/* Collection Selection */}
      <CollapsibleSection title="Collection" defaultOpen={true}>
        <div className="space-y-1.5">
          <select
            aria-label="Collection"
            className="w-full border border-gray-300 rounded-md px-2 py-1.5 text-sm"
            value={selectedCollection}
            onChange={(e) => {
              setSelectedCollection(e.target.value)
              setSearchParams(e.target.value ? { collection: e.target.value } : {})
            }}>
            <option value="">Select a collection...</option>
            {collections?.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.count})
              </option>
            ))}
          </select>
        </div>
      </CollapsibleSection>

      {collectionDetails && (
        <>
          {/* Number of Clusters */}
          <CollapsibleSection title="Clusters" defaultOpen={true}>
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-gray-600">
                Clusters: {numClusters}
              </label>
              <input
                type="range"
                min="2"
                max="50"
                value={numClusters}
                onChange={(e) => setNumClusters(Number(e.target.value))}
                className="w-full"
              />
              <ClusterSuggestion
                collectionName={selectedCollection}
                onApply={(k) => setNumClusters(k)}
              />
            </div>
          </CollapsibleSection>

          {/* Reduction Algorithm */}
          <CollapsibleSection title="Reduction Algorithm" defaultOpen={false}>
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-gray-600">
                Reduction Algorithm
              </label>
              <div className="flex space-x-2">
                {(['tsne', 'umap', 'pca'] as const).map((algo) => (
                  <label key={algo} className="flex items-center text-xs cursor-pointer">
                    <input
                      type="radio"
                      name="reductionAlgorithm"
                      value={algo}
                      checked={reductionAlgorithm === algo}
                      onChange={() => setReductionAlgorithm(algo as ReductionAlgorithm)}
                      className="mr-1"
                    />
                    {algo.toUpperCase()}
                  </label>
                ))}
              </div>

              {/* t-SNE parameters */}
              {reductionAlgorithm === 'tsne' && (
                <div className="pl-4 space-y-2 border-l-2 border-blue-200 mt-1">
                  <div>
                    <label className="block text-xs text-gray-500">
                      Perplexity: {tsnePerplexity}
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="50"
                      value={tsnePerplexity}
                      onChange={(e) => setTsnePerplexity(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500">Learning Rate</label>
                    <select
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      value={tsneLearningRate}
                      onChange={(e) => setTsneLearningRate(e.target.value)}
                    >
                      <option value="auto">auto</option>
                      <option value="50">50</option>
                      <option value="100">100</option>
                      <option value="200">200</option>
                      <option value="500">500</option>
                      <option value="1000">1000</option>
                    </select>
                  </div>
                </div>
              )}

              {/* UMAP parameters */}
              {reductionAlgorithm === 'umap' && (
                <div className="pl-4 space-y-2 border-l-2 border-green-200 mt-1">
                  <div>
                    <label className="block text-xs text-gray-500">
                      Neighbors: {umapNNeighbors}
                    </label>
                    <input
                      type="range"
                      min="2"
                      max="100"
                      value={umapNNeighbors}
                      onChange={(e) => setUmapNNeighbors(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500">
                      Min Distance: {umapMinDist}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={umapMinDist}
                      onChange={(e) => setUmapMinDist(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500">Metric</label>
                    <select
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      value={umapMetric}
                      onChange={(e) => setUmapMetric(e.target.value)}
                    >
                      <option value="cosine">cosine</option>
                      <option value="euclidean">euclidean</option>
                      <option value="manhattan">manhattan</option>
                      <option value="correlation">correlation</option>
                    </select>
                  </div>
                </div>
              )}

              {/* PCA has no extra parameters */}
              {reductionAlgorithm === 'pca' && (
                <p className="text-xs text-gray-400 mt-1">
                  PCA has no additional parameters.
                </p>
              )}
            </div>
          </CollapsibleSection>

          {/* Text Display Fields */}
          <CollapsibleSection title="Display Fields" defaultOpen={false}>
            <div className="space-y-1.5">
              <div className="max-h-32 overflow-y-auto border rounded p-2 space-y-1">
                {collectionDetails.metadata_fields.map((field) => (
                  <div key={field} className="flex items-center">
                    <input
                      type="checkbox"
                      id={`field-${field}`}
                      checked={textDisplayFields.includes(field)}
                      onChange={() => toggleField(field)}
                      className="mr-2"
                    />
                    <label htmlFor={`field-${field}`} className="text-sm truncate cursor-pointer">
                      {field}
                    </label>
                  </div>
                ))}
              </div>
            </div>
          </CollapsibleSection>

          {/* Image Field */}
          <CollapsibleSection title="Image Field" defaultOpen={false}>
            <div className="space-y-1.5">
              <select
                aria-label="Image Field"
                className="w-full border border-gray-300 rounded-md px-2 py-1.5 text-sm"
                value={imageField}
                onChange={(e) => setImageField(e.target.value)}
              >
                <option value="">None</option>
                {collectionDetails.metadata_fields.map((field) => (
                  <option key={field} value={field}>
                    {field}
                  </option>
                ))}
              </select>
            </div>
          </CollapsibleSection>

          {/* GPT Settings */}
          <CollapsibleSection title="GPT Cluster Naming" defaultOpen={false}>
            <div className="space-y-1.5">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="gpt-enabled"
                  checked={gptEnabled}
                  onChange={(e) => setGptEnabled(e.target.checked)}
                  className="mr-2"
                />
                <label htmlFor="gpt-enabled" className="text-xs font-medium text-gray-600 cursor-pointer">
                  GPT Cluster Naming
                </label>
              </div>

              {gptEnabled && (
                <div className="pl-6 space-y-2">
                  <div>
                    <label className="block text-xs text-gray-500">Model</label>
                    <input
                      type="text"
                      value={gptModel}
                      onChange={(e) => setGptModel(e.target.value)}
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500">Temperature: {gptTemperature}</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={gptTemperature}
                      onChange={(e) => setGptTemperature(Number(e.target.value))}
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                    />
                  </div>
                </div>
              )}
            </div>
          </CollapsibleSection>

          {/* Rendering (Render Mode + Point Size) */}
          <CollapsibleSection title="Rendering" defaultOpen={false}>
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-gray-600">Render Mode</label>
              <div className="flex space-x-2">
                {(['particles', 'sprites', 'spheres'] as const).map((mode) => (
                  <label key={mode} className="flex items-center text-xs cursor-pointer">
                    <input
                      type="radio"
                      name="renderMode"
                      value={mode}
                      checked={renderMode === mode}
                      onChange={() => setRenderMode(mode)}
                      className="mr-1"
                    />
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </label>
                ))}
              </div>
            </div>

            <div className="space-y-1.5 mt-2">
              <label className="block text-xs font-medium text-gray-600">
                Point Size: {pointSize}
              </label>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={pointSize}
                onChange={(e) => setPointSize(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </CollapsibleSection>
        </>
      )}

      {/* Compute Button */}
      <button
        onClick={handleCompute}
        disabled={isComputing}
        className={`w-full py-2 px-4 rounded text-white font-bold mt-2 ${
          isComputing
            ? 'bg-blue-300 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700'
        }`}
      >
        {isComputing ? 'Computing...' : 'Compute Plot'}
      </button>
    </div>
  )
}
