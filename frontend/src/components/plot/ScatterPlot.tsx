import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { usePlotStore } from '../../stores/plotStore'
import ParticleCloud from './ParticleCloud'
import TooltipCard from './TooltipCard'

export default function ScatterPlot() {
  const renderMode = usePlotStore((state) => state.renderMode)

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden shadow-inner relative">
      <Canvas camera={{ position: [0, 0, 50], fov: 60 }}>
        <color attach="background" args={['#111827']} />

        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={1} />

        <OrbitControls makeDefault />

        {renderMode === 'particles' && <ParticleCloud />}
        {/* Placeholders for future modes */}
        {renderMode === 'sprites' && null}
        {renderMode === 'spheres' && null}

        <TooltipCard />
      </Canvas>

      {/* Fallback for empty modes */}
      {renderMode !== 'particles' && (
        <div className="absolute top-4 left-4 text-white bg-black/50 px-2 py-1 rounded text-xs pointer-events-none">
          Mode "{renderMode}" not yet implemented
        </div>
      )}
    </div>
  )
}
