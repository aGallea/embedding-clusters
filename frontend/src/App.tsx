import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Link, NavLink, Route, Routes } from 'react-router-dom'
import HomePage from './pages/HomePage'
import IndexPage from './pages/IndexPage'
import PlotPage from './pages/PlotPage'

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
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-100">
          <NavBar />
          <main>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/index" element={<IndexPage />} />
              <Route path="/plot" element={<PlotPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
