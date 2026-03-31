import { useState, useEffect } from 'react';
import type { AiProvider, StoredAiSettings } from '../api/ai';
import {
  AI_PROVIDERS,
  loadAiSettings,
  saveAiSettings,
  testAiConnection,
  fetchOllamaModels,
  DEFAULT_AI_SETTINGS,
} from '../api/ai';
import type { OllamaModel } from '../types';

export default function SettingsPage() {
  const [settings, setSettings] = useState<StoredAiSettings>(DEFAULT_AI_SETTINGS);
  const [isSaved, setIsSaved] = useState(false);
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [testMessage, setTestMessage] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [ollamaModelsLoading, setOllamaModelsLoading] = useState(false);
  const [ollamaModelsError, setOllamaModelsError] = useState('');

  useEffect(() => {
    const loaded = loadAiSettings();
    setSettings(loaded);
    if (loaded.provider === 'ollama') {
      void loadOllamaModels(loaded.baseUrl || 'http://localhost:11434');
    }
  }, []);

  const loadOllamaModels = async (baseUrl: string) => {
    setOllamaModelsLoading(true);
    setOllamaModelsError('');
    try {
      const response = await fetchOllamaModels(baseUrl);
      setOllamaModels(response.models);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to fetch models';
      setOllamaModelsError(msg);
      setOllamaModels([]);
    } finally {
      setOllamaModelsLoading(false);
    }
  };

  const handleChange = (field: keyof StoredAiSettings, value: string | number) => {
    setSettings((prev) => ({ ...prev, [field]: value }));
    setIsSaved(false);
    setTestStatus('idle');
  };

  const handleProviderChange = (provider: AiProvider) => {
    const providerConfig = AI_PROVIDERS.find((p) => p.value === provider);
    const newBaseUrl = providerConfig?.defaultBaseUrl ?? '';
    setSettings((prev) => ({
      ...prev,
      provider,
      baseUrl: newBaseUrl,
      model: provider === prev.provider ? prev.model : '',
    }));
    setIsSaved(false);
    setTestStatus('idle');

    if (provider === 'ollama') {
      void loadOllamaModels(newBaseUrl || 'http://localhost:11434');
    } else {
      setOllamaModels([]);
      setOllamaModelsError('');
    }
  };

  const handleSave = () => {
    saveAiSettings(settings);
    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 3000);
  };

  const handleTestConnection = async () => {
    setTestStatus('testing');
    setTestMessage('');
    try {
      const result = await testAiConnection({
        api_key: settings.apiKey,
        model: settings.model,
        base_url: settings.baseUrl || undefined,
      });

      if (result.success) {
        setTestStatus('success');
        setTestMessage('Connection successful!');
      } else {
        setTestStatus('error');
        setTestMessage(result.error || 'Connection failed.');
      }
    } catch (err: unknown) {
      setTestStatus('error');
      const msg = err instanceof Error ? err.message : 'Unknown error occurred.';
      setTestMessage(`Failed to test connection: ${msg}`);
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6 border-b pb-4 border-gray-200">
        <h1 className="text-3xl font-bold mb-2 text-gray-900">AI Settings</h1>
        <p className="text-gray-500 text-lg">
          Configure AI provider for cluster naming
        </p>
      </div>

      <div className="bg-white rounded-lg shadow border border-gray-200 p-6 max-w-2xl">
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Provider
            </label>
            <select
              value={settings.provider}
              onChange={(e) => handleProviderChange(e.target.value as AiProvider)}
              className="w-full border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border bg-white"
            >
              {AI_PROVIDERS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model
            </label>
            {settings.provider === 'ollama' && ollamaModels.length > 0 ? (
              <select
                value={settings.model}
                onChange={(e) => handleChange('model', e.target.value)}
                className="w-full border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border bg-white"
              >
                <option value="">Select a model...</option>
                {ollamaModels.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}{m.parameter_size ? ` (${m.parameter_size})` : ''}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={settings.model}
                onChange={(e) => handleChange('model', e.target.value)}
                className="w-full border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
                placeholder="e.g. gpt-4o-mini"
              />
            )}
            {settings.provider === 'ollama' && ollamaModelsLoading && (
              <p className="mt-1 text-sm text-gray-500">Loading models...</p>
            )}
            {settings.provider === 'ollama' && ollamaModelsError && (
              <p className="mt-1 text-sm text-red-500">{ollamaModelsError}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              API Key
            </label>
            <div className="relative">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={settings.apiKey}
                onChange={(e) => handleChange('apiKey', e.target.value)}
                className="w-full border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border pr-10"
                placeholder="sk-..."
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute inset-y-0 right-0 px-3 flex items-center text-sm text-gray-500 hover:text-gray-700 focus:outline-none"
              >
                {showApiKey ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Base URL <span className="text-gray-400 font-normal">(Optional)</span>
            </label>
            <input
              type="text"
              value={settings.baseUrl}
              onChange={(e) => handleChange('baseUrl', e.target.value)}
              className="w-full border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
              placeholder="e.g. https://api.openai.com/v1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Temperature
            </label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={settings.temperature}
              onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
              className="w-full border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
            />
          </div>

          <div className="pt-4 flex items-center justify-between border-t border-gray-200">
            <div className="flex items-center space-x-4">
              <button
                type="button"
                onClick={() => { void handleTestConnection(); }}
                disabled={testStatus === 'testing' || (!settings.apiKey && settings.provider !== 'ollama')}
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {testStatus === 'testing' ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-gray-700" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Testing...
                  </>
                ) : (
                  'Test Connection'
                )}
              </button>

              {testStatus === 'success' && (
                <span className="flex items-center text-sm text-green-600 font-medium">
                  <svg className="mr-1.5 h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  {testMessage}
                </span>
              )}

              {testStatus === 'error' && (
                <span className="text-sm text-red-600 font-medium max-w-xs break-words">
                  {testMessage}
                </span>
              )}
            </div>

            <div className="flex items-center space-x-3">
              {isSaved && (
                <span className="text-sm text-green-600 transition-opacity duration-300">
                  Settings saved!
                </span>
              )}
              <button
                type="button"
                onClick={handleSave}
                className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
