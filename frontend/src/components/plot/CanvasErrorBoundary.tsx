import { Component, type ReactNode, type ErrorInfo } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class CanvasErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Canvas error:', error, errorInfo)
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-gray-900 text-white p-6 text-center">
          <div className="bg-red-900/50 border border-red-700 rounded-lg p-6 max-w-md backdrop-blur-sm">
            <h3 className="text-xl font-bold mb-2 text-red-200">Visualization Error</h3>
            <p className="text-gray-300 mb-4 text-sm font-mono break-words">
              {this.state.error?.message || 'Unknown WebGL error occurred'}
            </p>
            <button
              onClick={this.handleRetry}
              className="px-4 py-2 bg-red-700 hover:bg-red-600 text-white rounded transition-colors text-sm font-medium"
            >
              Retry Visualization
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
