import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query'
import { BrowserRouter, Link, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { useEffect, useRef } from 'react'
import HomePage from './pages/HomePage'
import IndexPage from './pages/IndexPage'
import PlotPage from './pages/PlotPage'
import SettingsPage from './pages/SettingsPage'
import { usePlotStore } from './stores/plotStore'

const queryClient = new QueryClient()


function NavBar() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive
        ? 'bg-gray-900 text-white'
        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`

  return (
    <nav className="bg-gray-800">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-white font-bold text-lg">
              Embedding Clusters
            </Link>
            <div className="ml-10 flex items-baseline space-x-4">
              <NavLink to="/" end className={linkClass}>
                Home
              </NavLink>
              <NavLink to="/index" className={linkClass}>
                Index
              </NavLink>
              <NavLink to="/plot" className={linkClass}>
                Plot
              </NavLink>
              <NavLink to="/settings" className={linkClass}>
                Settings
              </NavLink>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}

function PlotStateResetter() {
  const location = useLocation()
  const setPlotData = usePlotStore((state) => state.setPlotData)
  const resetPlotJobId = usePlotStore((state) => state.resetPlotJobId)
  const clearClusterDrillDown = usePlotStore((state) => state.clearClusterDrillDown)
  const resetPlotCollectionName = usePlotStore((state) => state.resetPlotCollectionName)
  const queryClient = useQueryClient()
  const previousLocationRef = useRef(location)

  useEffect(() => {
    const previousLocation = previousLocationRef.current
    const leftPlot = previousLocation.pathname === '/plot' && location.pathname !== '/plot'
    const enteredPlotWithNewQuery =
      location.pathname === '/plot' && previousLocation.search !== location.search
    const enteredPlot =
      location.pathname === '/plot' && previousLocation.pathname !== '/plot'

    if (leftPlot || enteredPlotWithNewQuery || enteredPlot) {
      setPlotData(null)
      resetPlotJobId()
      clearClusterDrillDown()
      resetPlotCollectionName()
      queryClient.removeQueries({ queryKey: ['plotData'] })
    }

    previousLocationRef.current = location
  }, [clearClusterDrillDown, location, resetPlotCollectionName, resetPlotJobId, setPlotData, queryClient])

  return null
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-100">
          <NavBar />
          <PlotStateResetter />
          <main>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/index" element={<IndexPage />} />
              <Route path="/plot" element={<PlotPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
