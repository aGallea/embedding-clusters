import { useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { usePlotStore } from '../../stores/plotStore'
import ParticleCloud from './ParticleCloud'
import ImageSpriteCloud from './ImageSpriteCloud'
import InstancedSpheres from './InstancedSpheres'
import TooltipCard from './TooltipCard'
import CanvasErrorBoundary from './CanvasErrorBoundary'
import QueryMarker from './QueryMarker'

export default function ScatterPlot() {
  const renderMode = usePlotStore((state) => state.renderMode)
  const controlsRef = useRef<OrbitControlsImpl>(null!)

  const handleResetCamera = () => {
    controlsRef.current?.reset()
  }

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden shadow-inner relative group">
      <CanvasErrorBoundary>
        <Canvas camera={{ position: [0, 0, 50], fov: 60 }}>
          <color attach="background" args={['#111827']} />

          <ambientLight intensity={0.6} />
          <directionalLight position={[10, 10, 5]} intensity={1} />

          <OrbitControls ref={controlsRef} makeDefault />

          {renderMode === 'particles' && <ParticleCloud />}
          {renderMode === 'sprites' && <ImageSpriteCloud />}
          {renderMode === 'spheres' && <InstancedSpheres />}

          <QueryMarker />

          <TooltipCard />
        </Canvas>
      </CanvasErrorBoundary>

      <button
        onClick={handleResetCamera}
        className="absolute top-4 right-4 z-10 p-2 bg-gray-800/50 hover:bg-gray-700/80 text-white/80 hover:text-white rounded-md backdrop-blur-sm transition-all shadow-lg border border-white/10"
        title="Reset Camera"
        aria-label="Reset Camera"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
          <path d="M3 3v5h5" />
        </svg>
      </button>
    </div>
  )
}
