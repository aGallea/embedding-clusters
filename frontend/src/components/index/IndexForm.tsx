import { useState, type FormEvent } from 'react';
import type { IndexRequest } from '../../types';

interface IndexFormProps {
  columns: string[];
  csvFilename: string;
  totalRows: number;
  onSubmit: (request: IndexRequest) => void;
  isSubmitting: boolean;
}

export default function IndexForm({
  columns,
  csvFilename,
  totalRows,
  onSubmit,
  isSubmitting,
}: IndexFormProps) {
  const [formData, setFormData] = useState<IndexRequest>({
    csv_filename: csvFilename,
    id_field: undefined,
    image_embedding_fields: [],
    text_embedding_fields: [],
    image_model_name: 'openai/clip-vit-base-patch32',
    text_model_name: 'BAAI/bge-small-en-v1.5',
    chromadb_collection_prefix: '',
    number_of_async_tasks: 1,
    index_bulk_size: 100,
    index_start_line: undefined,
    index_end_line: undefined,
    process_unit_device: 'cpu',
    embedding_fields_prefix: 'embedding_',
    total_rows: undefined,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;

    setFormData(prev => ({
      ...prev,
      [name]: type === 'number'
        ? (value === '' ? undefined : parseInt(value, 10))
        : (value === '' ? undefined : value)
    }));
  };

  const handleCheckboxChange = (field: 'image_embedding_fields' | 'text_embedding_fields', value: string) => {
    setFormData(prev => {
      const currentList = prev[field] || [];
      const newList = currentList.includes(value)
        ? currentList.filter(item => item !== value)
        : [...currentList, value];

      return {
        ...prev,
        [field]: newList
      };
    });
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit({
      ...formData,
      total_rows: totalRows,
    });
  };

  const renderSectionHeader = (title: string) => (
    <h3 className="text-lg font-medium leading-6 text-gray-900 border-b border-gray-200 pb-2 mb-4 mt-6 first:mt-0">
      {title}
    </h3>
  );

  const renderLabel = (label: string, htmlFor: string) => (
    <label htmlFor={htmlFor} className="block text-sm font-medium text-gray-700 mb-1">
      {label}
    </label>
  );

  const renderInput = (
    name: keyof IndexRequest,
    label: string,
    type: string = 'text',
    placeholder?: string,
    required: boolean = false
  ) => (
    <div>
      {renderLabel(label, name)}
      <input
        type={type}
        name={name}
        id={name}
        value={formData[name] as string | number || ''}
        onChange={handleChange}
        placeholder={placeholder}
        required={required}
        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm border p-2"
      />
    </div>
  );

  return (
    <form onSubmit={handleSubmit} className="space-y-6 bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6">

      {/* Data Source Section */}
      <div>
        {renderSectionHeader("Data Source")}
        <div className="grid grid-cols-1 gap-y-6 gap-x-4 sm:grid-cols-2">
          <div>
            {renderLabel("ID Field", "id_field")}
            <select
              name="id_field"
              id="id_field"
              value={formData.id_field || ''}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm border p-2"
            >
              <option value="">-- Auto-generate --</option>
              {columns.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
            <p className="mt-1 text-xs text-gray-500">Unique identifier for each row.</p>
          </div>

          <div>
            {renderLabel("ChromaDB Collection Prefix", "chromadb_collection_prefix")}
            <input
              type="text"
              name="chromadb_collection_prefix"
              id="chromadb_collection_prefix"
              value={formData.chromadb_collection_prefix || ''}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm border p-2"
            />
          </div>
        </div>
      </div>

      {/* Embedding Fields Section */}
      <div>
        {renderSectionHeader("Embedding Fields")}
        <div className="grid grid-cols-1 gap-y-6 gap-x-4 sm:grid-cols-2">
          <div>
            <span className="block text-sm font-medium text-gray-700 mb-2">Image Embedding Fields</span>
            <div className="max-h-48 overflow-y-auto border border-gray-300 rounded-md p-2 bg-gray-50">
              {columns.map(col => (
                <div key={`img-${col}`} className="flex items-center h-6 mb-1">
                  <input
                    id={`img-${col}`}
                    name={`img-${col}`}
                    type="checkbox"
                    checked={(formData.image_embedding_fields || []).includes(col)}
                    onChange={() => handleCheckboxChange('image_embedding_fields', col)}
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <label htmlFor={`img-${col}`} className="ml-2 text-sm text-gray-900 truncate" title={col}>
                    {col}
                  </label>
                </div>
              ))}
            </div>
            <p className="mt-1 text-xs text-gray-500">Columns containing image URLs or paths.</p>
          </div>

          <div>
            <span className="block text-sm font-medium text-gray-700 mb-2">Text Embedding Fields</span>
            <div className="max-h-48 overflow-y-auto border border-gray-300 rounded-md p-2 bg-gray-50">
              {columns.map(col => (
                <div key={`txt-${col}`} className="flex items-center h-6 mb-1">
                  <input
                    id={`txt-${col}`}
                    name={`txt-${col}`}
                    type="checkbox"
                    checked={(formData.text_embedding_fields || []).includes(col)}
                    onChange={() => handleCheckboxChange('text_embedding_fields', col)}
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <label htmlFor={`txt-${col}`} className="ml-2 text-sm text-gray-900 truncate" title={col}>
                    {col}
                  </label>
                </div>
              ))}
            </div>
            <p className="mt-1 text-xs text-gray-500">Columns containing text to embed.</p>
          </div>
        </div>
      </div>

      {/* Model Configuration Section */}
      <div>
        {renderSectionHeader("Model Configuration")}
        <div className="grid grid-cols-1 gap-y-6 gap-x-4 sm:grid-cols-2">
          {renderInput("image_model_name", "Image Model Name", "text")}
          {renderInput("text_model_name", "Text Model Name", "text")}
          {renderInput("embedding_fields_prefix", "Embedding Fields Prefix", "text")}

          <div>
            {renderLabel("Processing Device", "process_unit_device")}
            <select
              name="process_unit_device"
              id="process_unit_device"
              value={formData.process_unit_device || 'cpu'}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm border p-2"
            >
              <option value="cpu">CPU</option>
              <option value="mps">MPS (Apple Silicon)</option>
              <option value="cuda">CUDA (Nvidia GPU)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Processing Options Section */}
      <div>
        {renderSectionHeader("Processing Options")}
        <div className="grid grid-cols-1 gap-y-6 gap-x-4 sm:grid-cols-2">
          {renderInput("number_of_async_tasks", "Concurrent Async Tasks", "number")}
          {renderInput("index_bulk_size", "Bulk Size", "number")}
          {renderInput("index_start_line", "Start Line (Optional)", "number", "Start from beginning")}
          {renderInput("index_end_line", "End Line (Optional)", "number", "Until end")}
        </div>
      </div>

      {/* Form Actions */}
      <div className="pt-5 border-t border-gray-200">
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={isSubmitting}
            className={`
              ml-3 inline-flex justify-center rounded-md border border-transparent
              py-2 px-4 text-sm font-medium text-white shadow-sm
              ${isSubmitting ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'}
            `}
          >
            {isSubmitting ? 'Starting Index...' : 'Start Indexing'}
          </button>
        </div>
      </div>
    </form>
  );
}
