import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { usePlotStore } from '../../stores/plotStore'
import ParticleCloud from './ParticleCloud'
import ImageSpriteCloud from './ImageSpriteCloud'
import InstancedSpheres from './InstancedSpheres'
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
        {renderMode === 'sprites' && <ImageSpriteCloud />}
        {renderMode === 'spheres' && <InstancedSpheres />}

        <TooltipCard />
      </Canvas>

    </div>
  )
}
