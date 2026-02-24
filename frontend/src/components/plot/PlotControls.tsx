import { useQuery } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { fetchCollections, fetchCollection } from '../../api/collections'
import { usePlotStore } from '../../stores/plotStore'
import type { PlotRequest } from '../../types'

interface PlotControlsProps {
  onCompute: (params: PlotRequest) => void
  isComputing: boolean
}

export default function PlotControls({ onCompute, isComputing }: PlotControlsProps) {
  const [searchParams] = useSearchParams()
  const [selectedCollection, setSelectedCollection] = useState(searchParams.get('collection') || '')
  const [numClusters, setNumClusters] = useState(10)
  const [textDisplayFields, setTextDisplayFields] = useState<string[]>([])
  const [imageField, setImageField] = useState('')
  const [gptEnabled, setGptEnabled] = useState(false)
  const [gptModel, setGptModel] = useState('gpt-3.5-turbo')
  const [gptTemperature, setGptTemperature] = useState(0.51)

  const { renderMode, setRenderMode, pointSize, setPointSize } = usePlotStore()

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

  const handleCompute = () => {
    if (!selectedCollection) return

    const request: PlotRequest = {
      chromadb_collection_name: selectedCollection,
      num_clusters: numClusters,
      text_display_fields: textDisplayFields,
      image_field: imageField || undefined,
      gpt_generate_cluster_name: gptEnabled,
      gpt_default_model: gptEnabled ? gptModel : undefined,
      gpt_default_temperature: gptEnabled ? gptTemperature : undefined,
    }
    onCompute(request)
  }

  const toggleField = (field: string) => {
    setTextDisplayFields(prev =>
      prev.includes(field) ? prev.filter(f => f !== field) : [...prev, field]
    )
  }

  return (
    <div className="space-y-6 p-4">
      <h2 className="text-xl font-bold mb-4">Configuration</h2>

      {/* Collection Selection */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">Collection</label>
        <select
          className="w-full border border-gray-300 rounded-md p-2"
          value={selectedCollection}
          onChange={(e) => setSelectedCollection(e.target.value)}
        >
          <option value="">Select a collection...</option>
          {collections?.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name} ({c.count})
            </option>
          ))}
        </select>
      </div>

      {collectionDetails && (
        <>
          {/* Number of Clusters */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
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
          </div>

          {/* Text Display Fields */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">Display Fields</label>
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

          {/* Image Field */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">Image Field</label>
            <select
              className="w-full border border-gray-300 rounded-md p-2"
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

          {/* GPT Settings */}
          <div className="space-y-2 border-t pt-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="gpt-enabled"
                checked={gptEnabled}
                onChange={(e) => setGptEnabled(e.target.checked)}
                className="mr-2"
              />
              <label htmlFor="gpt-enabled" className="text-sm font-medium text-gray-700 cursor-pointer">
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

          {/* Render Mode */}
          <div className="space-y-2 border-t pt-4">
            <label className="block text-sm font-medium text-gray-700">Render Mode</label>
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

          {/* Point Size */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
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

          {/* Compute Button */}
          <button
            onClick={handleCompute}
            disabled={isComputing}
            className={`w-full py-2 px-4 rounded text-white font-bold mt-4 ${
              isComputing
                ? 'bg-blue-300 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isComputing ? 'Computing...' : 'Compute Plot'}
          </button>
        </>
      )}
    </div>
  )
}
